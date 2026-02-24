"""GRPO (Group Relative Policy Optimization) trainer for orchestrator.

Based on ToolOrchestra's training approach using GRPO instead of PPO.
GRPO is simpler than PPO because it doesn't require a separate critic model.

Algorithm:
1. For each problem, generate multiple candidate solutions (group sampling)
2. Score all candidates using reward function
3. Compute relative rewards within each group
4. Update policy to increase probability of better solutions

Reference: ToolOrchestra uses verl library with GRPO
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional imports - only needed for actual training with model
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

from orchestrator.data import (
    ToolScaleDataset,
    TelemetryCache,
    EpisodeBuilder,
    GeneralThoughtDataset,
    MixedDataset,
    UnifiedSample,
    create_mixed_dataset,
)
from orchestrator.training.rl.reward import MultiObjectiveReward, RewardWeights, Normalizers
from orchestrator.training.rl.environment import OrchestratorEnvironment, OrchestratorEnvironmentReal
from orchestrator.training.rl.policy import PolicyModel


@dataclass
class GRPOConfig:
    """Configuration for GRPO training (matching ToolOrchestra)."""

    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    max_prompt_length: int = 24000
    max_response_length: int = 8768

    # Training
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # GRPO specific
    num_samples_per_prompt: int = 8  # Group size (rollout_agents in ToolOrchestra)
    temperature: float = 1.0
    kl_coef: float = 0.0001  # KL divergence coefficient
    clip_ratio: float = 0.2  # Policy clip ratio

    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 100

    # Data - Single dataset mode
    dataset_name: str = "nvidia/ToolScale"
    dataset_split: str = "train"
    dataset_limit: Optional[int] = None

    # Data - Multi-dataset mode
    use_mixed_dataset: bool = False
    """Enable mixed dataset training (ToolScale + GeneralThought)"""
    toolscale_weight: float = 0.5
    """Weight for ToolScale samples in mixed mode"""
    generalthought_weight: float = 0.5
    """Weight for GeneralThought samples in mixed mode"""
    min_verifier_score: float = 0.5
    """Minimum verifier score for GeneralThought quality filtering"""

    # Caching
    telemetry_cache_path: str = "data/telemetry_cache.db"

    # Environment
    available_tools: List[str] = None
    max_turns: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_last_n: int = 3

    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "orchestrator"

    # Online data collection
    use_online_collection: bool = False
    """Use real MCP execution instead of cached telemetry (slower but accurate for IPJ)"""

    # Memory optimization
    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to reduce activation memory (trades compute for memory)"""
    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps (effective batch = batch_size * accumulation_steps)"""
    use_8bit_ref: bool = True
    """Load reference model in 8-bit quantization to save memory"""
    use_8bit_optimizer: bool = False
    """Use 8-bit AdamW optimizer (requires bitsandbytes)"""

    def __post_init__(self):
        if self.available_tools is None:
            # All available tools matching ToolOrchestra capabilities
            self.available_tools = [
                # Free utility tools
                "calculator",
                "think",
                "code_interpreter",
                "web_search",
                # Local models via Ollama (fast, free)
                "ollama:qwen2.5:0.5b",
                "ollama:qwen2.5:1.5b",
                "ollama:llama3.2:1b",
                "ollama:llama3.2:3b",
                # Cloud models via OpenAI
                "openai:gpt-5-mini-2025-08-07",
                "openai:gpt-4o",
                "openai:o1-mini",
                "openai:o1",
                # Cloud models via Anthropic
                "anthropic:claude-3-5-haiku-20241022",
                "anthropic:claude-sonnet-4-20250514",
                "anthropic:claude-opus-4-20250514",
                # Large models via vLLM
                "vllm:qwen3-8b",
                "vllm:qwen3-32b",
                "vllm:llama-8b",
                "vllm:llama-70b",
                # Specialist models via vLLM
                "vllm:qwen-math-7b",
                "vllm:glm-4.7",
                "vllm:qwen-coder-7b",
                "vllm:qwen3-coder-plus",
            ]


class GRPOTrainer:
    """GRPO trainer for orchestrator (simplified verl-style implementation).

    Example:
        config = GRPOConfig(
            model_name="Qwen/Qwen3-1.7B",
            num_epochs=3,
            batch_size=16,
        )

        trainer = GRPOTrainer(config)
        trainer.train()
    """

    def __init__(self, config: GRPOConfig):
        """Initialize GRPO trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Check if torch is available
        if HAS_TORCH and torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        # Initialize components
        print("Initializing GRPO trainer...")
        self._init_model()
        self._init_data()
        self._init_optimizer()
        self._init_logging()

        self.global_step = 0

    def _init_model(self):
        """Initialize policy model with memory optimizations."""
        print(f"Loading model: {self.config.model_name}")

        # Get device string for explicit device placement
        device_str = str(self.device) if self.device else None

        # Load policy model with gradient checkpointing
        self.policy = PolicyModel.from_pretrained(
            self.config.model_name,
            max_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            gradient_checkpointing=self.config.gradient_checkpointing,
            device=device_str,
        )

        if self.policy.model is not None:
            self.policy.model.train()

        # Keep a copy of initial policy for KL divergence
        # Use 8-bit quantization if enabled (saves ~50% memory for ref model)
        print(f"Loading reference model (8-bit={self.config.use_8bit_ref})...")
        self.ref_policy = PolicyModel.from_pretrained(
            self.config.model_name,
            max_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            load_in_8bit=self.config.use_8bit_ref,
            device=device_str,
        )

        if self.ref_policy.model is not None:
            self.ref_policy.model.eval()
            # Freeze reference model parameters
            for param in self.ref_policy.model.parameters():
                param.requires_grad = False

    def _init_data(self):
        """Initialize dataset and environment."""
        if self.config.use_mixed_dataset:
            # Mixed dataset mode: combine ToolScale + GeneralThought
            print("Loading mixed dataset (ToolScale + GeneralThought)...")

            # Load ToolScale
            print(f"  Loading ToolScale ({self.config.dataset_name})...")
            toolscale = ToolScaleDataset(
                split=self.config.dataset_split,
                limit=self.config.dataset_limit,
                dataset_path=self.config.dataset_name,
            )

            # Load GeneralThought with quality filtering
            print(f"  Loading GeneralThought (min_verifier_score={self.config.min_verifier_score})...")
            generalthought = GeneralThoughtDataset(
                split="train",
                limit=self.config.dataset_limit,
                min_verifier_score=self.config.min_verifier_score,
            )

            # Create mixed dataset with configured weights
            self.dataset = create_mixed_dataset(
                toolscale_dataset=toolscale,
                generalthought_dataset=generalthought,
                toolscale_weight=self.config.toolscale_weight,
                generalthought_weight=self.config.generalthought_weight,
            )
            print(f"Mixed dataset: {len(self.dataset)} total samples")
        else:
            # Single dataset mode: ToolScale only
            print(f"Loading dataset: {self.config.dataset_name}")
            self.dataset = ToolScaleDataset(
                split=self.config.dataset_split,
                limit=self.config.dataset_limit,
                dataset_path=self.config.dataset_name,
            )

        # Load telemetry cache
        cache_path = Path(self.config.telemetry_cache_path)
        if not cache_path.exists():
            print(f"Warning: Telemetry cache not found at {cache_path}")
            print("Creating empty cache...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache = TelemetryCache(cache_path)

        # Create environment (cached for fast training)
        self.env = OrchestratorEnvironment(
            telemetry_cache=self.cache,
            available_tools=self.config.available_tools,
            max_turns=self.config.max_turns,
        )

        # Initialize MCP tools and online environment if using online collection
        self.mcp_tools = None
        self.online_env = None
        if self.config.use_online_collection:
            print("Initializing MCP tools for online data collection...")
            self.mcp_tools = self._init_mcp_tools()
            self.online_env = OrchestratorEnvironmentReal(
                mcp_tools=self.mcp_tools,
                max_turns=self.config.max_turns,
            )
            print(f"  Initialized {len(self.mcp_tools)} tools for online execution")

        # Create reward function with separate coefficients for each metric
        weights = RewardWeights(
            alpha=0.4,           # Accuracy (most important)
            beta_cost=0.15,      # API cost
            beta_energy=0.15,    # Energy consumption
            gamma_latency=0.15,  # Response time
            gamma_power=0.15,    # Peak power usage
        )
        normalizers = Normalizers()
        self.reward_fn = MultiObjectiveReward(weights, normalizers)

    def _init_mcp_tools(self) -> Dict[str, Any]:
        """Initialize MCP server instances for online data collection.

        Returns:
            Dictionary mapping tool names to MCP server instances
        """
        mcp_tools = {}

        for tool_name in self.config.available_tools:
            try:
                if tool_name == "calculator":
                    from agents.mcp.tool_server import CalculatorServer
                    mcp_tools[tool_name] = CalculatorServer()

                elif tool_name == "think":
                    from agents.mcp.tool_server import ThinkServer
                    mcp_tools[tool_name] = ThinkServer()

                elif tool_name == "code_interpreter":
                    from agents.mcp.tool_server import CodeInterpreterServer
                    mcp_tools[tool_name] = CodeInterpreterServer()

                elif tool_name == "web_search":
                    from agents.mcp.tool_server import WebSearchServer
                    mcp_tools[tool_name] = WebSearchServer()

                elif tool_name.startswith("ollama:"):
                    from agents.mcp.ollama_server import OllamaMCPServer
                    model_name = tool_name.split(":", 1)[1]
                    mcp_tools[tool_name] = OllamaMCPServer(model_name=model_name)

                elif tool_name.startswith("openai:"):
                    from agents.mcp.openai_server import OpenAIMCPServer
                    model_name = tool_name.split(":", 1)[1]
                    mcp_tools[tool_name] = OpenAIMCPServer(model_name=model_name)

                elif tool_name.startswith("anthropic:"):
                    from agents.mcp.anthropic_server import AnthropicMCPServer
                    model_name = tool_name.split(":", 1)[1]
                    mcp_tools[tool_name] = AnthropicMCPServer(model_name=model_name)

                elif tool_name.startswith("vllm:"):
                    from agents.mcp.vllm_server import VLLMMCPServer
                    model_name = tool_name.split(":", 1)[1]
                    mcp_tools[tool_name] = VLLMMCPServer(model_name=model_name)

                else:
                    print(f"Warning: Unknown tool type for '{tool_name}', skipping")

            except ImportError as e:
                print(f"Warning: Could not import MCP server for '{tool_name}': {e}")
            except Exception as e:
                print(f"Warning: Could not initialize '{tool_name}': {e}")

        return mcp_tools

    def _init_optimizer(self):
        """Initialize optimizer with optional 8-bit quantization."""
        if self.policy.model is None:
            self.optimizer = None
            return

        if self.config.use_8bit_optimizer:
            # Use bitsandbytes 8-bit AdamW to save ~50% optimizer memory
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.policy.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                )
                print("  Using 8-bit AdamW optimizer (bitsandbytes)")
            except ImportError:
                print("Warning: bitsandbytes not available, falling back to standard AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.policy.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                weight_decay=self.config.weight_decay,
            )

    def _init_logging(self):
        """Initialize logging."""
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []

        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    config=asdict(self.config),
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, logging locally only")
                self.wandb = None
        else:
            self.wandb = None

    def train(self):
        """Run GRPO training loop."""
        print("=" * 70)
        print("Starting GRPO Training")
        print("=" * 70)
        print(f"Model: {self.config.model_name}")
        print(f"Dataset: {len(self.dataset)} samples")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Group size: {self.config.num_samples_per_prompt}")
        print("=" * 70)
        print()

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 70)

            epoch_metrics = self._train_epoch(epoch)

            # Log epoch metrics
            self._log_metrics(epoch_metrics, epoch)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        print()
        print("=" * 70)
        print("Training complete!")
        print("=" * 70)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of epoch metrics
        """
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        # Iterate over batches using dataset's iter_batches method
        batch_size = self.config.batch_size
        for batch in self.dataset.iter_batches(batch_size):
            # Run GRPO step
            loss, reward = self._grpo_step(batch)

            total_loss += loss
            total_reward += reward
            num_batches += 1

            self.global_step += 1

            # Log progress
            if self.global_step % self.config.log_every_n_steps == 0:
                print(
                    f"Step {self.global_step}: "
                    f"loss={loss:.4f}, reward={reward:.4f}"
                )

        # Compute average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_reward = total_reward / num_batches if num_batches > 0 else 0.0

        return {
            "epoch": epoch,
            "loss": avg_loss,
            "reward": avg_reward,
            "num_batches": num_batches,
        }

    def _grpo_step(self, batch: List) -> tuple[float, float]:
        """Perform one GRPO training step.

        GRPO Algorithm:
        1. For each problem, sample N solutions (group)
        2. Compute rewards for all solutions
        3. Compute relative advantages within group
        4. Update policy to increase probability of better solutions

        Args:
            batch: Batch of samples

        Returns:
            (loss, average_reward)
        """
        if self.policy.model is None or not HAS_TORCH:
            raise RuntimeError(
                "Cannot train without PyTorch and model. "
                "Install with: pip install torch transformers"
            )

        self.policy.model.train()

        # Store all data for batch update
        all_prompts = []
        all_responses = []
        all_advantages = []
        all_rewards = []

        # 1. Generate multiple solutions per problem (group sampling)
        for sample in batch:
            group_prompts = []
            group_responses = []
            group_log_probs = []
            group_rewards = []

            # Get sample fields - works with both ToolScaleSample and UnifiedSample
            question = sample.question
            answer = sample.answer
            task_id = sample.task_id

            # Build initial prompt - pass sample directly to get reasoning context if available
            # For UnifiedSample, this passes reasoning; for ToolScaleSample, reset() handles it
            if isinstance(sample, UnifiedSample):
                state = self.env.reset(sample)
            else:
                state = self.env.reset(question)
            prompt = self.policy._build_prompt(state, self.config.available_tools)

            # Generate N candidate solutions for this problem
            for sample_idx in range(self.config.num_samples_per_prompt):
                # Generate response with log probabilities
                response, log_probs = self._generate_with_log_probs(prompt)

                # Parse response into action
                policy_output = self.policy._parse_output(
                    response, self.config.available_tools
                )

                # Execute action in environment
                # Use online environment for real execution if enabled, otherwise use cached
                action = self.policy._output_to_action(policy_output)
                env = self.online_env if self.config.use_online_collection and self.online_env else self.env
                if isinstance(sample, UnifiedSample):
                    state_copy = env.reset(sample)
                else:
                    state_copy = env.reset(question)

                try:
                    state_copy, observation = env.step(state_copy, action)

                    # If online, also update telemetry cache for future use
                    if self.config.use_online_collection and hasattr(observation, 'telemetry'):
                        self.cache.save_profile(observation.telemetry)
                except Exception as e:
                    # Fallback to cached environment on error
                    if self.config.use_online_collection:
                        print(f"Warning: Online execution failed, using cached: {e}")
                    if isinstance(sample, UnifiedSample):
                        state_copy = self.env.reset(sample)
                    else:
                        state_copy = self.env.reset(question)
                    state_copy, observation = self.env.step(state_copy, action)

                # Build episode and compute reward
                episode = state_copy.to_episode(
                    task_id=task_id,
                    ground_truth=answer,
                    correct=(state_copy.final_answer == answer),
                )
                reward = self.reward_fn.compute(episode)

                # Store for this group
                group_prompts.append(prompt)
                group_responses.append(response)
                group_log_probs.append(log_probs)
                group_rewards.append(reward)

            # 2. Compute group-relative advantages
            mean_reward = sum(group_rewards) / len(group_rewards)
            std_reward = (
                sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)
            ) ** 0.5
            # Normalize advantages
            if std_reward > 1e-8:
                group_advantages = [
                    (r - mean_reward) / std_reward for r in group_rewards
                ]
            else:
                group_advantages = [0.0] * len(group_rewards)

            # Add to batch data
            all_prompts.extend(group_prompts)
            all_responses.extend(group_responses)
            all_advantages.extend(group_advantages)
            all_rewards.extend(group_rewards)

        # 3. Compute policy gradient loss
        total_loss = 0.0
        total_kl = 0.0

        for prompt, response, advantage in zip(
            all_prompts, all_responses, all_advantages
        ):
            # Get log probs from current policy
            current_log_probs = self._compute_log_probs(prompt, response)

            # Get log probs from reference policy (for KL penalty)
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs_ref(prompt, response)

            # Compute log probability ratio
            log_ratio = current_log_probs - ref_log_probs
            ratio = torch.exp(log_ratio)
            # Clamp ratio to prevent numerical instability
            # Values outside [0.01, 100] indicate severe policy divergence
            ratio = torch.clamp(ratio, min=0.01, max=100.0)

            # Clipped policy gradient loss (PPO-style clipping)
            clip_ratio = self.config.clip_ratio
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

            # Policy loss: -min(ratio * advantage, clipped_ratio * advantage)
            # This is the standard PPO clipped objective
            policy_loss = -torch.min(
                ratio * advantage, clipped_ratio * advantage
            )

            # KL penalty
            kl_divergence = (ratio - 1) - log_ratio
            kl_penalty = self.config.kl_coef * kl_divergence

            # Total loss for this sample
            loss = policy_loss + kl_penalty

            total_loss += loss
            total_kl += kl_divergence.item()

        # 4. Backward pass and optimization
        avg_loss = total_loss / len(all_prompts)

        # Skip update if loss is NaN/inf or too high (numerical stability)
        loss_val = avg_loss.item()
        if torch.isnan(avg_loss) or torch.isinf(avg_loss) or abs(loss_val) > 100.0:
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            return 0.0, float(avg_reward)  # Skip unstable update

        self.optimizer.zero_grad()
        avg_loss.backward()

        # Check for NaN gradients and skip update if found
        has_nan_grad = False
        for param in self.policy.model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if has_nan_grad:
            self.optimizer.zero_grad()  # Clear bad gradients
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            return float(loss_val), float(avg_reward)  # Skip update with NaN grads

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), self.config.max_grad_norm
        )

        self.optimizer.step()

        # Clear GPU cache after step to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute metrics
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        avg_kl = total_kl / len(all_prompts) if all_prompts else 0.0

        return float(loss_val), float(avg_reward)

    def _generate_with_log_probs(self, prompt: str) -> tuple[str, torch.Tensor]:
        """Generate response and compute log probabilities.

        Args:
            prompt: Input prompt

        Returns:
            (generated_text, log_probs)
        """
        # Clear GPU cache before generation to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Tokenize prompt with explicit max_length to prevent overflow
        max_prompt_tokens = min(self.config.max_prompt_length, 16000)  # Safe limit
        inputs = self.policy.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(self.device)

        input_len = inputs.input_ids.shape[1]

        # Limit generation to stay within model context window
        # Qwen3 has 32K context, leave room for prompt
        max_new = min(self.config.max_response_length, 32000 - input_len - 100)
        max_new = max(min(max_new, 2048), 128)  # Cap at 2048 tokens, at least 128

        # Generate with sampling
        try:
            with torch.no_grad():
                outputs = self.policy.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=self.config.temperature,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.policy.tokenizer.pad_token_id,
                    eos_token_id=self.policy.tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            print(f"Generation error: {e}")
            print(f"  Input length: {input_len} tokens")
            print(f"  Max new tokens: {max_new}")
            raise

        # Extract generated tokens (excluding prompt)
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]) :]

        # Handle empty generation
        if len(generated_ids) == 0:
            print("Warning: Empty generation, returning dummy values")
            return "", torch.tensor(0.0, device=self.device, requires_grad=True)

        # Decode generated text
        generated_text = self.policy.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Compute log probabilities for generated tokens
        # Note: outputs.scores contains logits for each generation step
        log_probs = []
        vocab_size = self.policy.tokenizer.vocab_size
        for i, (token_id, logits) in enumerate(
            zip(generated_ids, outputs.scores)
        ):
            # Validate token_id is within vocabulary bounds
            token_id_val = token_id.item()
            if token_id_val < 0 or token_id_val >= vocab_size:
                print(f"Warning: Invalid token_id {token_id_val} at position {i}, vocab_size={vocab_size}")
                continue

            # Apply softmax to get probabilities
            probs = F.softmax(logits[0], dim=-1)
            # Get log prob of selected token (with bounds check)
            if token_id_val < probs.shape[0]:
                log_prob = torch.log(probs[token_id_val] + 1e-10)  # Add epsilon for numerical stability
                log_probs.append(log_prob)

        # Sum log probabilities
        if log_probs:
            total_log_prob = torch.stack(log_probs).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)

        return generated_text, total_log_prob

    def _compute_log_probs(self, prompt: str, response: str) -> torch.Tensor:
        """Compute log probabilities of response given prompt using current policy.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Log probability tensor
        """
        # Combine prompt and response
        full_text = prompt + response

        # Tokenize
        inputs = self.policy.tokenizer(
            full_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        prompt_inputs = self.policy.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Get model outputs
        with torch.enable_grad():
            outputs = self.policy.model(**inputs)
            logits = outputs.logits

        # Get log probabilities for response tokens only
        response_start = len(prompt_inputs.input_ids[0])
        response_end = len(inputs.input_ids[0])

        log_probs = []
        for i in range(response_start, response_end - 1):
            # Get logits for position i
            position_logits = logits[0, i, :]
            # Get token at position i+1
            target_token = inputs.input_ids[0, i + 1]
            # Compute log probability
            log_prob = F.log_softmax(position_logits, dim=-1)[target_token]
            log_probs.append(log_prob)

        # Sum log probabilities
        total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)

        return total_log_prob

    def _compute_log_probs_ref(self, prompt: str, response: str) -> torch.Tensor:
        """Compute log probabilities using reference policy (for KL penalty).

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Log probability tensor
        """
        # Combine prompt and response
        full_text = prompt + response

        # Tokenize
        inputs = self.ref_policy.tokenizer(
            full_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        prompt_inputs = self.ref_policy.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Get model outputs (no gradients needed for reference policy)
        with torch.no_grad():
            outputs = self.ref_policy.model(**inputs)
            logits = outputs.logits

        # Get log probabilities for response tokens only
        response_start = len(prompt_inputs.input_ids[0])
        response_end = len(inputs.input_ids[0])

        log_probs = []
        for i in range(response_start, response_end - 1):
            position_logits = logits[0, i, :]
            target_token = inputs.input_ids[0, i + 1]
            log_prob = F.log_softmax(position_logits, dim=-1)[target_token]
            log_probs.append(log_prob)

        # Sum log probabilities
        total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)

        return total_log_prob

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}"

        if self.policy.model is not None:
            self.policy.save(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

            # Save training state
            state_path = checkpoint_path / "training_state.json"
            state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "config": asdict(self.config),
            }
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return

        # Get all epoch checkpoints
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
            key=lambda x: int(x.name.split("_")[1]),
            reverse=True,
        )

        # Remove old checkpoints
        for checkpoint in checkpoints[self.config.keep_last_n :]:
            import shutil
            shutil.rmtree(checkpoint)
            print(f"Removed old checkpoint: {checkpoint.name}")

    def _log_metrics(self, metrics: Dict[str, Any], epoch: int):
        """Log metrics to file and wandb.

        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        # Add to history
        self.metrics_history.append(metrics)

        # Save to file
        metrics_file = self.log_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Log to wandb
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)

        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
