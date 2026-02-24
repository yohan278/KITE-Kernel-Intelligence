import uvicorn
import time
import uuid
from typing import Dict, Any, List, Optional, Literal

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents import BaseAgent


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="agent")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = Field(default=False)
    # Additional fields for compatibility
    top_p: Optional[float] = Field(default=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"


class Usage(BaseModel):
    """OpenAI-compatible usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    """OpenAI-compatible model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "agent-server"


class ModelsResponse(BaseModel):
    """OpenAI-compatible models list response."""
    object: str = "list"
    data: List[ModelInfo]


class RunRequest(BaseModel):
    """Request model for running the orchestrator (legacy endpoint)."""
    input: str
    kwargs: Dict[str, Any] = {}


class AgentServer:
    app = FastAPI(title="OpenAI-Compatible Agent Server")
    router = APIRouter()
    
    def __init__(self, orchestrater: BaseAgent, model_name: str = "agent") -> None:
        """Initialize the agent server.
        
        Args:
            orchestrater: The orchestrator instance to use.
            model_name: The model name to expose in OpenAI-compatible endpoints.
        """
        self.orchestrater = orchestrater
        self.model_name = model_name

        self._setup_routes()
        
    def _setup_routes(self) -> None:
        """Setup API routes."""
        # OpenAI-compatible endpoints
        self.router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        
        # Legacy endpoints
        self.router.add_api_route("/run", self.run_orchestrator, methods=["POST"])
        self.router.add_api_route("/health", self.health_check, methods=["GET"])
        
        self.app.include_router(self.router)
    
    async def health_check(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}
    
    async def list_models(self) -> ModelsResponse:
        """List available models (OpenAI-compatible)."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=self.model_name,
                    created=int(time.time())
                )
            ]
        )
    
    def _messages_to_input(self, messages: List[ChatMessage]) -> str:
        """Convert OpenAI messages format to orchestrator input format."""
        # Combine messages into a single string
        # System messages are prepended, user/assistant messages are formatted
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        
        return "\n\n".join(parts)
    
    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion requests (OpenAI-compatible)."""
        if request.stream:
            raise HTTPException(
                status_code=400, 
                detail="Streaming is not yet supported"
            )
        
        try:
            # Convert messages to orchestrator input format
            input_text = self._messages_to_input(request.messages)
            
            # Prepare kwargs for orchestrator
            orchestrator_kwargs = {}
            if request.temperature is not None:
                orchestrator_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                orchestrator_kwargs["max_tokens"] = request.max_tokens
            
            # Run the orchestrator
            result = self.orchestrater.run(input=input_text, **orchestrator_kwargs)
            
            # Convert result to string if needed
            if not isinstance(result, str):
                result = str(result)
            
            # Create OpenAI-compatible response
            response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            
            # Estimate token usage (rough approximation)
            prompt_tokens = len(input_text.split()) * 1.3  # Rough estimate
            completion_tokens = len(result.split()) * 1.3
            
            return ChatCompletionResponse(
                id=response_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=result
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(prompt_tokens + completion_tokens)
                )
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def run_orchestrator(self, request: RunRequest) -> Dict[str, Any]:
        """Execute the orchestrator with the given input (legacy endpoint)."""
        try:
            result = self.orchestrater.run(input=request.input, **request.kwargs)
            return {"result": result, "status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the FastAPI server."""
        uvicorn.run(self.app, host=host, port=port)


def main() -> None:
    """CLI entry point for serve-agent command.

    This provides a simple way to start the agent server from command line.
    For production use, instantiate AgentServer directly with your agent.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Start the OpenAI-compatible agent server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--agent", default="react", choices=["react", "terminus", "openhands", "orchestrator"],
                        help="Agent type to use")
    args = parser.parse_args()

    # Import and instantiate the requested agent
    if args.agent == "react":
        from agents import React
        agent = React()
    elif args.agent == "terminus":
        from agents import Terminus
        agent = Terminus()
    elif args.agent == "openhands":
        from agents import OpenHands
        agent = OpenHands()
    elif args.agent == "orchestrator":
        from agents import Orchestrator
        agent = Orchestrator()
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    server = AgentServer(agent, model_name=args.agent)
    server.serve(host=args.host, port=args.port)
