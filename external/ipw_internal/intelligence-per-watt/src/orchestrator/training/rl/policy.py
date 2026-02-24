"""Policy model wrapper for orchestrator.

Wraps a language model (e.g., Qwen3-1.7B) to predict actions in the orchestrator
environment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from orchestrator.data.episode_builder import Action
from orchestrator.training.rl.environment import EpisodeState


@dataclass
class PolicyOutput:
    """Output from policy model."""

    thought: str
    """Reasoning about what to do"""

    tool_name: str
    """Selected tool"""

    tool_prompt: str
    """Prompt for the tool"""

    is_final_answer: bool = False
    """Whether this provides the final answer"""

    raw_text: str = ""
    """Raw model output"""

    confidence: float = 1.0
    """Confidence score (if available)"""


class PolicyModel:
    """Wrapper around language model for orchestrator policy.

    The policy model takes the current state and predicts the next action.

    Input format (prompt):
        Task: {initial_prompt}

        Available tools: calculator, ollama:llama3.2:1b, openai:gpt-4o

        History:
        Turn 1:
          Thought: ...
          Tool: ...
          Observation: ...

        What should you do next?

    Output format (from model):
        Thought: [reasoning about next step]
        Tool: [tool_name]
        Prompt: [prompt for tool]
        Final: [yes/no]

    Example:
        policy = PolicyModel.from_pretrained("Qwen/Qwen3-1.7B")

        # Predict action
        state = EpisodeState(initial_prompt="What is 2+2?")
        action = policy.predict_action(state, available_tools=["calculator"])
        print(f"Tool: {action.tool_name}")
        print(f"Thought: {action.thought}")
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Initialize policy model.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-1.7B",
        gradient_checkpointing: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> PolicyModel:
        """Load policy model from pretrained checkpoint.

        Args:
            model_name: HuggingFace model name or path
            gradient_checkpointing: Enable gradient checkpointing to reduce memory
            load_in_8bit: Load model in 8-bit quantization (for ref model)
            device: Target device (e.g., "cuda:0"). If None, uses auto device mapping.
            **kwargs: Additional arguments for PolicyModel

        Returns:
            PolicyModel instance

        Raises:
            ImportError: If transformers library is not installed
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build model loading kwargs
        model_kwargs = {"torch_dtype": torch.bfloat16}

        if load_in_8bit:
            # 8-bit quantization for memory efficiency (reference model)
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                print("Warning: bitsandbytes not available, loading in bf16 instead")

        # Handle device placement
        # device=None with no special handling = let Accelerate manage (for multi-GPU)
        # device="cuda:0" or similar = load to specific device
        # device="auto" = use device_map="auto" (single GPU auto-placement)
        if device is not None and device != "accelerate":
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": device}
        # else: no device_map - Accelerate will handle placement

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print(f"  Gradient checkpointing enabled for {model_name}")

        return cls(model=model, tokenizer=tokenizer, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs: Any) -> PolicyModel:
        """Load policy from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            **kwargs: Additional arguments

        Returns:
            PolicyModel instance
        """
        return cls.from_pretrained(checkpoint_path, **kwargs)

    def predict_action(
        self,
        state: EpisodeState,
        available_tools: List[str],
    ) -> Action:
        """Predict next action given current state.

        Args:
            state: Current episode state
            available_tools: List of available tool names

        Returns:
            Action to take
        """
        # Build prompt
        prompt = self._build_prompt(state, available_tools)

        # Generate response
        if self.model is not None:
            output_text = self._generate(prompt)
        else:
            raise RuntimeError(
                "Cannot generate actions without a loaded model. "
                "Load a model with PolicyModel.from_pretrained() first."
            )

        # Parse output
        policy_output = self._parse_output(output_text, available_tools)

        # Convert to Action
        return self._output_to_action(policy_output)

    def _build_prompt(
        self,
        state: EpisodeState,
        available_tools: List[str],
        max_reasoning_chars: int = 1000,
    ) -> str:
        """Build prompt for policy model.

        Args:
            state: Current state
            available_tools: Available tools
            max_reasoning_chars: Max characters for example reasoning

        Returns:
            Prompt string
        """
        prompt_parts = []

        # Task
        prompt_parts.append(f"Task: {state.initial_prompt}")
        prompt_parts.append("")

        # Include example reasoning if available (from GeneralThought dataset)
        if state.example_reasoning:
            reasoning_text = state.example_reasoning
            if len(reasoning_text) > max_reasoning_chars:
                reasoning_text = reasoning_text[:max_reasoning_chars] + "..."
            prompt_parts.append("Example reasoning for similar problems:")
            prompt_parts.append(reasoning_text)
            prompt_parts.append("")

        # Available tools
        tools_str = ", ".join(available_tools)
        prompt_parts.append(f"Available tools: {tools_str}")
        prompt_parts.append("")

        # History
        if state.history:
            prompt_parts.append("History:")
            for i, (action, observation) in enumerate(state.history, 1):
                prompt_parts.append(f"Turn {i}:")
                prompt_parts.append(f"  Thought: {action.thought}")
                prompt_parts.append(f"  Tool: {action.tool_name}")
                prompt_parts.append(f"  Observation: {observation.content[:100]}...")
                prompt_parts.append("")

        # Instruction
        prompt_parts.append("What should you do next?")
        prompt_parts.append("Format your response as:")
        prompt_parts.append("Thought: [your reasoning]")
        prompt_parts.append("Tool: [tool_name]")
        prompt_parts.append("Prompt: [prompt for tool]")
        prompt_parts.append("Final: [yes/no]")
        prompt_parts.append("")

        return "\n".join(prompt_parts)

    def _generate(self, prompt: str) -> str:
        """Generate response from model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
        )

        output_text = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True,
        )

        return output_text

    def _output_to_action(self, policy_output: PolicyOutput) -> Action:
        """Convert PolicyOutput to Action.

        Args:
            policy_output: Parsed policy output

        Returns:
            Action object
        """
        return Action(
            thought=policy_output.thought,
            tool_name=policy_output.tool_name,
            tool_prompt=policy_output.tool_prompt,
            is_final_answer=policy_output.is_final_answer,
        )

    def _parse_output(
        self,
        output_text: str,
        available_tools: List[str],
    ) -> PolicyOutput:
        """Parse model output into structured format.

        Args:
            output_text: Raw model output
            available_tools: Available tools

        Returns:
            PolicyOutput
        """
        # Extract fields using regex
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", output_text, re.IGNORECASE)
        tool_match = re.search(r"Tool:\s*(.+?)(?:\n|$)", output_text, re.IGNORECASE)
        prompt_match = re.search(r"Prompt:\s*(.+?)(?:\n|$)", output_text, re.IGNORECASE)
        final_match = re.search(r"Final:\s*(.+?)(?:\n|$)", output_text, re.IGNORECASE)

        # Extract values
        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        tool_name = tool_match.group(1).strip() if tool_match else available_tools[0]
        tool_prompt = prompt_match.group(1).strip() if prompt_match else ""
        final_text = final_match.group(1).strip().lower() if final_match else "yes"
        is_final = final_text in ["yes", "true", "1"]

        # Validate tool name
        if tool_name not in available_tools:
            # Fuzzy match or fallback
            tool_name = available_tools[0] if available_tools else "unknown"

        return PolicyOutput(
            thought=thought,
            tool_name=tool_name,
            tool_prompt=tool_prompt,
            is_final_answer=is_final,
            raw_text=output_text,
        )

    def save(self, save_path: str):
        """Save policy model to disk.

        Args:
            save_path: Directory to save model
        """
        if self.model is not None:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        else:
            print("Warning: No model to save (mock policy)")

    def __repr__(self) -> str:
        model_name = "None" if self.model is None else type(self.model).__name__
        return f"PolicyModel(model={model_name}, max_tokens={self.max_tokens})"
