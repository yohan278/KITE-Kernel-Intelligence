"""Policy model for inference - loads trained orchestrator checkpoint."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..prompt_registry import build_system_prompt

# Optional imports - only needed when loading actual model
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None  # type: ignore


@dataclass
class Action:
    """Orchestrator action: thought + tool selection + tool prompt."""

    thought: str
    """Reasoning about what to do next"""

    tool_name: str
    """Selected tool (e.g., 'calculator', 'local_llm', 'cloud_llm')"""

    tool_prompt: str
    """Prompt/query to send to the tool"""

    is_final_answer: bool = False
    """Whether this action provides the final answer"""


@dataclass
class InferencePolicyOutput:
    """Output from policy model during inference."""

    action: Action
    """Predicted action"""

    raw_output: str
    """Raw model output text"""

    confidence: float = 1.0
    """Confidence score (optional)"""


class InferencePolicy:
    """Policy model wrapper for inference.

    Loads a trained GRPO checkpoint and generates actions during orchestration.

    Example:
        # Load from checkpoint
        policy = InferencePolicy.from_checkpoint("checkpoints/qwen_1.7b/epoch_10")

        # Generate action
        state = {"history": [...], "question": "What is 2+2?"}
        output = policy.predict_action(state, available_tools=["calculator"])
        print(output.action.tool_name)  # "calculator"
        print(output.action.tool_prompt)  # "2+2"
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize policy.

        Args:
            model: Loaded model (optional, for mock mode set to None)
            tokenizer: Loaded tokenizer (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Set device
        if HAS_TRANSFORMERS and model is not None and torch is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> InferencePolicy:
        """Load policy from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            InferencePolicy instance

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ImportError: If transformers library not installed
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Train a model first using: python src/cli/train.py"
            )

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )

        # Load model and tokenizer
        print(f"Loading checkpoint from {checkpoint_path}...")

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

        # Set to eval mode
        model.eval()

        print(f"✓ Loaded model from {checkpoint_path}")

        return cls(
            model=model,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def predict_action(
        self,
        state: Dict[str, Any],
        available_tools: List[str],
    ) -> InferencePolicyOutput:
        """Predict next action given current state.

        Args:
            state: Current episode state with history
            available_tools: List of available tool names

        Returns:
            InferencePolicyOutput with predicted action

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Load checkpoint with InferencePolicy.from_checkpoint() first."
            )

        # Build prompt from state
        prompt = self._build_prompt(state, available_tools)

        # Generate output
        output_text = self._generate(prompt)

        # Parse output into action
        action = self._parse_output(output_text, available_tools)

        return InferencePolicyOutput(
            action=action,
            raw_output=output_text,
            confidence=1.0,
        )

    def _build_prompt(self, state: Dict[str, Any], available_tools: List[str]) -> str:
        """Build prompt from state and available tools.

        Uses the canonical system prompt from prompt_registry so inference
        matches training exactly.

        Args:
            state: Current episode state
            available_tools: List of available tool names

        Returns:
            Formatted prompt string
        """
        system = build_system_prompt(available_tools)

        # Task
        question = state.get("question", "")
        conversation = [f"Task: {question}"]

        # History (if any) — same format as training
        history = state.get("history", [])
        for turn in history:
            thought = turn.get("thought", "")
            tool = turn.get("tool", "")
            tool_input = turn.get("tool_input", turn.get("prompt", ""))
            observation = turn.get("observation", "")
            conversation.append(f"\nTHOUGHT: {thought}")
            conversation.append(f"TOOL: {tool}")
            conversation.append(f"INPUT: {tool_input}")
            conversation.append(f"OBSERVATION: {observation}")

        conversation.append("\nWhat is your next step?")

        return system + "\n\n" + "\n".join(conversation)

    def _generate(self, prompt: str) -> str:
        """Generate output using the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        return generated

    def _parse_output(
        self, output_text: str, available_tools: List[str]
    ) -> Action:
        """Parse model output into Action.

        Uses the canonical THOUGHT/TOOL/INPUT/FINAL_ANSWER format that
        matches training.

        Args:
            output_text: Raw model output
            available_tools: List of available tools

        Returns:
            Parsed Action
        """
        # Strip <think>...</think> blocks (some models output thinking)
        output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
        output_text = re.sub(r".*</think>", "", output_text, flags=re.DOTALL)
        output_text = output_text.strip()

        # Check for FINAL_ANSWER (use .* so empty answers are recognised)
        final_match = re.search(
            r"FINAL_ANSWER:\s*(.*)",
            output_text, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            thought_match = re.search(
                r"THOUGHT:\s*(.+?)(?=\n(?:FINAL_ANSWER)|$)",
                output_text, re.DOTALL | re.IGNORECASE
            )
            return Action(
                thought=thought_match.group(1).strip() if thought_match else "",
                tool_name="final_answer",
                tool_prompt=final_match.group(1).strip(),
                is_final_answer=True,
            )

        # Extract THOUGHT, TOOL, INPUT
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=\n(?:TOOL|FINAL_ANSWER)|$)",
            output_text, re.DOTALL | re.IGNORECASE
        )
        tool_match = re.search(r"TOOL:\s*(\S+)", output_text, re.IGNORECASE)
        input_match = re.search(
            r"INPUT:\s*(.+?)(?=\n(?:THOUGHT|TOOL)|$)",
            output_text, re.DOTALL | re.IGNORECASE
        )

        thought = thought_match.group(1).strip() if thought_match else "Proceeding with action"
        tool_name = tool_match.group(1).strip() if tool_match else (available_tools[0] if available_tools else "unknown")
        tool_prompt = input_match.group(1).strip() if input_match else ""

        return Action(
            thought=thought,
            tool_name=tool_name,
            tool_prompt=tool_prompt,
            is_final_answer=False,
        )


