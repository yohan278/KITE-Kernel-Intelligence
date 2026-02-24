"""Anthropic MCP server with cost tracking."""

from __future__ import annotations

import time
from typing import Any, Optional

from anthropic import Anthropic, AnthropicError

from .base import BaseMCPServer, MCPToolResult


class AnthropicMCPServer(BaseMCPServer):
    """MCP server for Anthropic Claude models with cost tracking.

    Example:
        server = AnthropicMCPServer(
            model_name="claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        result = server.execute("Write a haiku about AI")
        print(result.content)
        print(f"Cost: ${result.cost_usd:.4f}")
    """

    # Pricing per 1M tokens
    # Source: https://www.anthropic.com/pricing
    PRICING = {
        # Claude 4.5 models
        "claude-opus-4-5-20251101": {"input": 20.00, "output": 100.00},
        "claude-sonnet-4-5-20250929": {"input": 4.00, "output": 20.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        # Claude 4 models
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        # Claude 3 models
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        telemetry_collector: Optional[Any] = None,
        **anthropic_params: Any,
    ):
        """Initialize Anthropic MCP server.

        Args:
            model_name: Claude model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            telemetry_collector: Energy monitor collector
            **anthropic_params: Additional params (temperature, max_tokens, etc.)
        """
        super().__init__(
            name=f"anthropic:{model_name}",
            telemetry_collector=telemetry_collector,
        )

        self.model_name = model_name
        self.anthropic_params = anthropic_params

        # Initialize Anthropic client
        self._client = Anthropic(api_key=api_key)

        # Get pricing for this model
        self.pricing = self.PRICING.get(model_name)
        if not self.pricing:
            # Fallback to Sonnet pricing if model not in table
            print(f"Warning: No pricing info for {model_name}, using Sonnet 4.5 rates")
            self.pricing = self.PRICING["claude-sonnet-4-5-20250929"]

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        """Execute Anthropic API call with cost tracking.

        Args:
            prompt: Input prompt
            **params: Override anthropic_params for this request

        Returns:
            MCPToolResult with response, usage, and calculated cost
        """
        # Merge default params with request params
        payload = {**self.anthropic_params, **params}
        payload["model"] = self.model_name
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["max_tokens"] = payload.get("max_tokens", 4096)  # Required param
        payload["stream"] = True

        # Call Anthropic API
        start = time.perf_counter()
        try:
            stream = self._client.messages.create(**payload)
        except AnthropicError as exc:
            raise RuntimeError(f"Anthropic error for {self.model_name}: {exc}") from exc

        # Consume stream and collect response
        content_chunks: list[str] = []
        ttft_ms: Optional[float] = None
        input_tokens = 0
        output_tokens = 0

        with stream as event_stream:
            for event in event_stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        content_chunks.append(event.delta.text)

                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens

        content = "".join(content_chunks)

        # Calculate cost based on token usage
        cost_usd = self._calculate_cost(input_tokens, output_tokens)

        return MCPToolResult(
            content=content,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            cost_usd=cost_usd,
            ttft_seconds=(ttft_ms / 1000.0) if ttft_ms else None,
            metadata={
                "model": self.model_name,
                "backend": "anthropic",
                "pricing_input_per_1m": self.pricing["input"],
                "pricing_output_per_1m": self.pricing["output"],
            },
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost in USD.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost

    def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Simple API call to test connectivity
            response = self._client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return response is not None
        except Exception:
            return False
