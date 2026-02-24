#!/usr/bin/env python3
"""CLI for running evaluation benchmarks.

Usage:
    python evals/run_eval.py --benchmark hle --limit 200 --seed 42 --model checkpoints/model
    python evals/run_eval.py --benchmark gaia --limit 200 --seed 42 --model checkpoints/model
    python evals/run_eval.py --benchmark simpleqa --limit 200 --seed 42 --model checkpoints/model
    python evals/run_eval.py --benchmark deepresearch --limit 50 --seed 42 --model checkpoints/model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def str2bool(v):
    """Parse boolean from string for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src/ to path so "agents.mcp.*" imports work for orchestrator tools
sys.path.insert(1, str(Path(__file__).parent.parent.parent))

from evals.benchmarks.hle import HLERunner
from evals.benchmarks.gaia import GAIARunner
from evals.benchmarks.simpleqa import SimpleQARunner
from evals.benchmarks.deepresearch import DeepResearchRunner


def verify_api_keys(verbose: bool = True) -> bool:
    """Verify that required API keys are set and working.
    
    Checks: OpenAI, OpenRouter, Anthropic, Gemini, Tavily
    Returns True if all keys are valid, False otherwise.
    
    Legend:
      ✅ Working - key is set and verified working
      ⚠️  Set but error - key is set but API call failed
      ❌ NOT SET - key is missing
    """
    keys = {
        "OPENAI_API_KEY": None,
        "OPENROUTER_API_KEY": None,
        "ANTHROPIC_API_KEY": None,
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        "TAVILY_API_KEY": None,
    }
    
    # Check which keys are set
    for key in keys:
        if keys[key] is None:
            keys[key] = os.environ.get(key)
    
    if verbose:
        print("\n" + "=" * 50)
        print("API Key Verification")
        print("=" * 50)
    
    all_valid = True
    results = {}
    
    # 1. OpenAI
    key = keys["OPENAI_API_KEY"]
    if not key:
        results["OpenAI"] = ("❌ NOT SET", False)
        all_valid = False
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            client.models.list()  # Quick API call
            results["OpenAI"] = ("✅ Working", True)
        except Exception as e:
            results["OpenAI"] = (f"⚠️  Set but error: {str(e)[:35]}", False)
            all_valid = False
    
    # 2. OpenRouter
    key = keys["OPENROUTER_API_KEY"]
    if not key:
        results["OpenRouter"] = ("❌ NOT SET", False)
        all_valid = False
    else:
        try:
            import httpx
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10
            )
            if resp.status_code == 200:
                results["OpenRouter"] = ("✅ Working", True)
            else:
                results["OpenRouter"] = (f"⚠️  Set but error: HTTP {resp.status_code}", False)
                all_valid = False
        except Exception as e:
            results["OpenRouter"] = (f"⚠️  Set but error: {str(e)[:35]}", False)
            all_valid = False
    
    # 3. Anthropic (make actual API call with cheapest model)
    key = keys["ANTHROPIC_API_KEY"]
    if not key:
        results["Anthropic"] = ("❌ NOT SET", False)
        all_valid = False
    else:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=key)
            # Minimal API call to verify key works (costs ~$0.00001)
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            results["Anthropic"] = ("✅ Working", True)
        except Exception as e:
            results["Anthropic"] = (f"⚠️  Set but error: {str(e)[:35]}", False)
            all_valid = False
    
    # 4. Gemini
    key = keys["GEMINI_API_KEY"]
    if not key:
        results["Gemini"] = ("❌ NOT SET (GEMINI_API_KEY or GOOGLE_API_KEY)", False)
        all_valid = False
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            genai.list_models()  # Quick API call
            results["Gemini"] = ("✅ Working", True)
        except Exception as e:
            results["Gemini"] = (f"⚠️  Set but error: {str(e)[:35]}", False)
            all_valid = False
    
    # 5. Tavily
    key = keys["TAVILY_API_KEY"]
    if not key:
        results["Tavily"] = ("❌ NOT SET", False)
        all_valid = False
    else:
        try:
            import httpx
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": "test", "max_results": 1},
                timeout=10
            )
            if resp.status_code == 200:
                results["Tavily"] = ("✅ Working", True)
            else:
                results["Tavily"] = (f"⚠️  Set but error: HTTP {resp.status_code}", False)
                all_valid = False
        except Exception as e:
            results["Tavily"] = (f"⚠️  Set but error: {str(e)[:35]}", False)
            all_valid = False
    
    # Print results
    if verbose:
        for name, (status, _) in results.items():
            print(f"  {name:12} {status}")
        print("=" * 50)
        
        # Count issues
        missing = sum(1 for s, _ in results.values() if "NOT SET" in s)
        errors = sum(1 for s, _ in results.values() if "Set but error" in s)
        
        if all_valid:
            print("✅ All API keys verified!")
        elif missing > 0 and errors > 0:
            print(f"❌ {missing} key(s) missing, ⚠️  {errors} key(s) not working")
        elif missing > 0:
            print(f"❌ {missing} API key(s) missing")
        else:
            print(f"⚠️  {errors} API key(s) set but not working")
        print()
    
    return all_valid


def verify_tools(verbose: bool = True) -> bool:
    """Verify that all orchestrator tools are functional by running each with a minimal test input.

    Uses the same initialization path as the actual evaluation.

    Legend:
      ✅ Working       - tool loaded and executed successfully
      ⚠️  Error        - tool loaded but execution failed
      ❌ Failed to load - tool could not be imported/initialized
    """
    from evals.orchestrator_eval import OrchestratorEvaluator
    from prompt_registry import AVAILABLE_TOOLS

    # Minimal inputs: free/local tools get deterministic inputs; LLM tools get a
    # one-word prompt to keep token cost as low as possible.
    DUMMY_INPUTS = {
        "calculator": "1 + 1",
        "think": "sanity check",
        "code_interpreter": "print(1 + 1)",
        "web_search": "test",
    }
    LLM_INPUT = "Say 'ok'."

    if verbose:
        print("\n" + "=" * 50)
        print("Tool Verification")
        print("=" * 50)

    # Initialise tools through the same code path used during evaluation.
    evaluator = OrchestratorEvaluator(
        model_fn=lambda s, m: "FINAL_ANSWER: test",
        max_turns=1,
        verbose=False,
    )

    all_valid = True
    results = {}

    for tool_name in AVAILABLE_TOOLS:
        if tool_name not in evaluator.tools:
            reason = evaluator.tool_errors.get(tool_name, "unknown error")
            results[tool_name] = (f"❌ Failed to load: {reason}", False)
            all_valid = False
            continue

        dummy = DUMMY_INPUTS.get(tool_name, LLM_INPUT)
        try:
            result = evaluator.tools[tool_name].execute(dummy)
            if result.content:
                results[tool_name] = ("✅ Working", True)
            else:
                results[tool_name] = ("⚠️  Empty response", False)
                all_valid = False
        except Exception as e:
            results[tool_name] = (f"⚠️  Error: {str(e)[:50]}", False)
            all_valid = False

    if verbose:
        for name, (status, _) in results.items():
            print(f"  {name:50} {status}")
        print("=" * 50)

        failing = sum(1 for _, ok in results.values() if not ok)
        if all_valid:
            print("✅ All tools verified!")
        else:
            print(f"❌ {failing} tool(s) failed")
        print()

    return all_valid


def create_descriptive_output_dir(
    base_dir: str,
    model_name: str,
    benchmark: str,
    limit: Optional[int] = None,
    seed: int = 42,
    split: Optional[str] = None,
    domain: Optional[str] = None,
    orchestrator: bool = True,
) -> Path:
    """Create a descriptive output directory name.
    
    Format: {benchmark}-{model}-n{limit}-s{seed}-{split}-{domain}-{orch_flag}
    """
    import re
    
    # Sanitize model name (remove path separators, special chars)
    model_safe = re.sub(r'[^\w\-.]', '_', model_name.replace('/', '_').replace('\\', '_'))
    # Truncate if too long
    if len(model_safe) > 50:
        model_safe = model_safe[:50]
    
    # Build directory name parts
    parts = [benchmark, model_safe]
    
    if limit:
        parts.append(f"n{limit}")
    
    parts.append(f"s{seed}")
    
    if split and split != "test":
        parts.append(f"split{split}")
    
    if domain:
        # For benchmarks with default domains, only include if not default
        default_domains = {
            "gaia": "all",
            "simpleqa": "all",
            "deepresearch": "all",
        }
        if domain != default_domains.get(benchmark, None):
            parts.append(f"d{domain}")
    
    # Add orchestrator flag
    parts.append("orch" if orchestrator else "no-orch")
    
    dir_name = "-".join(parts)
    output_path = Path(base_dir) / dir_name
    return output_path


def create_model_fn(model_name: str, temperature: float = 0.7, max_tokens: int = 8192, enable_thinking: bool = False):
    """Create model inference function based on model name.

    Supports:
    - openrouter:<model> - Use OpenRouter API
    - openai:<model> - Use OpenAI API
    - anthropic:<model> - Use Anthropic API
    - vllm:<model>@<port> - Use vLLM server
    - Local path - Load with transformers
    """
    if model_name.startswith("openrouter:"):
        from agents.mcp.openrouter_server import OpenRouterMCPServer
        model = model_name.split(":", 1)[1]
        server = OpenRouterMCPServer(model_name=model, temperature=temperature)

        def model_fn(system_prompt, messages):
            # Format messages for OpenRouter
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            # Use server's execute method with formatted prompt
            full_prompt = (system_prompt + "\n\n") if system_prompt else ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                full_prompt += f"{role}: {msg['content']}\n"
            full_prompt += "Assistant:"
            result = server.execute(full_prompt)
            return result.content

        return model_fn

    elif model_name.startswith("openai:"):
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model = model_name.split(":", 1)[1]

        def model_fn(system_prompt, messages):
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        return model_fn

    elif model_name.startswith("anthropic:"):
        import os
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        model = model_name.split(":", 1)[1]

        def model_fn(system_prompt, messages):
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            kwargs = dict(
                model=model,
                messages=formatted_messages,
                max_tokens=4096,
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            response = client.messages.create(**kwargs)
            return response.content[0].text

        return model_fn

    elif model_name.startswith("vllm:"):
        from agents.mcp.vllm_server import VLLMMCPServer
        model_spec = model_name.split(":", 1)[1]
        if "@" in model_spec:
            model, port = model_spec.rsplit("@", 1)
            vllm_url = f"http://localhost:{port}"
        else:
            model = model_spec
            vllm_url = "http://localhost:8000"

        server = VLLMMCPServer(model_name=model, vllm_url=vllm_url)

        def model_fn(system_prompt, messages):
            full_prompt = (system_prompt + "\n\n") if system_prompt else ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                full_prompt += f"{role}: {msg['content']}\n"
            full_prompt += "Assistant:"
            result = server.execute(full_prompt)
            return result.content

        return model_fn

    else:
        # Assume it's a HuggingFace model path - load with transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading model: {model_name}")
            
            # Check if this is a local path
            model_path = Path(model_name)
            is_local = model_path.exists()
            
            # Check if this is a LoRA adapter (has adapter_config.json)
            is_lora = is_local and (model_path / "adapter_config.json").exists()
            
            if is_lora:
                print("Detected LoRA adapter, loading base model first...")
                from peft import PeftModel
                
                # Read adapter config to get base model name
                with open(model_path / "adapter_config.json") as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-4B")
                print(f"Base model: {base_model_name}")
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                
                # Load LoRA adapter (use absolute path)
                model = PeftModel.from_pretrained(base_model, str(model_path.absolute()))
                print("LoRA adapter loaded successfully")
                
                # Load tokenizer from base model
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            elif is_local:
                # Local full model
                print(f"Loading local model from: {model_path.absolute()}")
                tokenizer = AutoTokenizer.from_pretrained(str(model_path.absolute()), trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path.absolute()),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # HuggingFace Hub model
                print(f"Loading from HuggingFace Hub: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            def model_fn(system_prompt, messages):
                # Format for chat
                chat_messages = []
                if system_prompt:
                    chat_messages.append({"role": "system", "content": system_prompt})
                for msg in messages:
                    chat_messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

                # Apply chat template
                template_kwargs = dict(
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if enable_thinking is not None:
                    template_kwargs["enable_thinking"] = enable_thinking
                text = tokenizer.apply_chat_template(
                    chat_messages,
                    **template_kwargs,
                )

                # Generate (use same parameters as training: max_tokens=8192, temperature=0.7)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=0.9 if temperature > 0 else None,  # Match training
                )
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return response

            return model_fn

        except Exception as e:
            raise ValueError(f"Could not load model {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["gaia", "hle", "simpleqa", "deepresearch"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Benchmark domain (simpleqa: topic, gaia: level)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., openrouter:qwen/qwen3-8b, Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for generation",
    )

    # Evaluation settings
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output with full orchestrator traces",
    )

    # Periodic save
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save intermediate results every N minutes (default: 10)",
    )

    # Orchestrator mode - matches training setup exactly
    parser.add_argument(
        "--orchestrator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable orchestrator mode with full tool execution (matches training)",
    )

    # Thinking mode (for Qwen3 and similar models)
    parser.add_argument(
        "--enable-thinking",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable thinking mode (<think> blocks) for models that support it (e.g. Qwen3). "
             "Off by default — the Think tool provides explicit reasoning when needed. "
             "Usage: --enable-thinking, --enable-thinking true, --enable-thinking false",
    )

    args = parser.parse_args()

    # Load environment
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"Loading environment from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.replace("export ", "").strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

    verbose = args.verbose  # False=default summary, True=full traces

    # Verify API keys and tools before running evaluation
    if args.orchestrator:
        if not verify_api_keys(verbose=True):
            print("❌ Some API keys are missing or invalid. Cannot run evaluation.")
            print("   Please set all required API keys and try again.")
            sys.exit(1)

        if not verify_tools(verbose=True):
            print("❌ Some tools failed to load or execute. Cannot run evaluation.")
            print("   Please check the errors above and fix the failing tools.")
            sys.exit(1)

    # Create descriptive output directory
    descriptive_output_dir = create_descriptive_output_dir(
        base_dir=args.output_dir,
        model_name=args.model,
        benchmark=args.benchmark,
        limit=args.limit,
        seed=args.seed,
        split=args.split,
        domain=args.domain,
        orchestrator=args.orchestrator,
    )
    
    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Domain: {args.domain or '(benchmark default)'}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Limit: {args.limit or 'None (all samples)'}")
    print(f"Orchestrator Mode: {args.orchestrator}")
    print(f"Thinking Mode: {args.enable_thinking}")
    print(f"Verbose: {verbose}")
    print(f"Output: {descriptive_output_dir}")
    print("=" * 60)

    # Create model function with training-matched parameters
    print(f"\nInitializing model: {args.model}")
    print(f"  Temperature: {args.temperature} (training used 0.7)")
    print(f"  Max tokens: {args.max_tokens} (training used 8192)")
    print(f"  Thinking: {args.enable_thinking}")
    model_fn = create_model_fn(args.model, args.temperature, args.max_tokens, enable_thinking=args.enable_thinking)

    # Wrap with orchestrator if enabled (matches training setup exactly)
    if args.orchestrator:
        print("Enabling orchestrator mode with full tool execution...")
        print("  Using EXACT training prompts with all examples")
        print("  Using EXACT training tools (15+ tools including all LLMs)")
        from evals.orchestrator_eval import create_orchestrator_model_fn
        from prompt_registry import AVAILABLE_TOOLS
        model_fn = create_orchestrator_model_fn(
            model_fn=model_fn,
            max_turns=args.max_turns,
            verbose=verbose,
        )
        tool_names = list(AVAILABLE_TOOLS)
        print(f"Orchestrator mode enabled with {len(tool_names)} tools: {', '.join(tool_names[:5])}...")

    # Run benchmark
    if args.benchmark == "hle":
        runner = HLERunner(
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            verbose=verbose,
            output_dir=str(descriptive_output_dir),
            save_interval_minutes=args.save_interval,
        )

        metrics = runner.run(model_fn=model_fn, orchestrator=args.orchestrator)

        print("\n" + "=" * 60)
        print("Results Summary - Humanity's Last Exam")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics.accuracy:.2%}")
        print(f"Multiple Choice Accuracy: {metrics.mc_accuracy:.2%}")
        print(f"Short Answer Accuracy: {metrics.short_answer_accuracy:.2%}")
        print(f"Average Latency: {metrics.avg_latency:.2f}s")
        print(f"Total Questions: {metrics.total_questions}")
        print(f"Correct Answers: {metrics.correct_answers}")
        if metrics.subject_metrics:
            print("\nPer-Subject Accuracy:")
            for subject, data in sorted(metrics.subject_metrics.items()):
                print(f"  {subject}: {data['accuracy']:.2%} ({data['total']} questions)")
        print("=" * 60)

    elif args.benchmark == "gaia":
        runner = GAIARunner(
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            domain=args.domain or "all",
            verbose=verbose,
            output_dir=str(descriptive_output_dir),
            save_interval_minutes=args.save_interval,
        )

        metrics = runner.run(model_fn=model_fn, orchestrator=args.orchestrator)

        print("\n" + "=" * 60)
        print("Results Summary - GAIA")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics.accuracy:.2%}")
        print(f"Level 1 Accuracy: {metrics.level1_accuracy:.2%}")
        print(f"Level 2 Accuracy: {metrics.level2_accuracy:.2%}")
        print(f"Level 3 Accuracy: {metrics.level3_accuracy:.2%}")
        print(f"Average Latency: {metrics.avg_latency:.2f}s")
        print(f"Total Tasks: {metrics.total_tasks}")
        print(f"Correct Tasks: {metrics.correct_tasks}")
        print("=" * 60)

    elif args.benchmark == "simpleqa":
        runner = SimpleQARunner(
            split="eval",
            limit=args.limit,
            seed=args.seed,
            domain=args.domain or "all",
            verbose=verbose,
            output_dir=str(descriptive_output_dir),
            save_interval_minutes=args.save_interval,
        )

        metrics = runner.run(model_fn=model_fn, orchestrator=args.orchestrator)

        print("\n" + "=" * 60)
        print("Results Summary - SimpleQA Verified")
        print("=" * 60)
        print(f"F1 Score: {metrics.f1:.2%}")
        print(f"Accuracy: {metrics.accuracy:.2%}")
        print(f"Correct: {metrics.correct_rate:.2%} ({metrics.correct_count})")
        print(f"Incorrect: {metrics.incorrect_rate:.2%} ({metrics.incorrect_count})")
        print(f"Not Attempted: {metrics.not_attempted_rate:.2%} ({metrics.not_attempted_count})")
        print(f"Average Latency: {metrics.avg_latency:.2f}s")
        print(f"Total Questions: {metrics.total_questions}")
        if metrics.topic_metrics:
            print("\nPer-Topic F1:")
            for topic, data in sorted(metrics.topic_metrics.items()):
                print(f"  {topic}: {data['f1']:.2%} ({data['total']} questions)")
        print("=" * 60)

    elif args.benchmark == "deepresearch":
        runner = DeepResearchRunner(
            limit=args.limit,
            seed=args.seed,
            domain=args.domain or "all",
            verbose=verbose,
            output_dir=str(descriptive_output_dir),
            save_interval_minutes=args.save_interval,
        )

        metrics = runner.run(model_fn=model_fn, orchestrator=args.orchestrator)

        print("\n" + "=" * 60)
        print("Results Summary - DeepResearch-Bench (RACE)")
        print("=" * 60)
        print(f"Overall Score: {metrics.overall_score:.4f}")
        print(f"Comprehensiveness: {metrics.comprehensiveness_score:.4f}")
        print(f"Insight: {metrics.insight_score:.4f}")
        print(f"Instruction Following: {metrics.instruction_following_score:.4f}")
        print(f"Readability: {metrics.readability_score:.4f}")
        print(f"Average Latency: {metrics.avg_latency:.2f}s")
        print(f"Total Tasks: {metrics.total_tasks}")
        print(f"(Scores are 0-1 ratios: target / (target + reference))")
        if metrics.category_metrics:
            print("\nPer-Category Scores:")
            for cat, data in sorted(metrics.category_metrics.items()):
                print(f"  {cat}: {data['overall_score']:.4f} ({data['total']} tasks)")
        print("=" * 60)

    else:
        print(f"Benchmark '{args.benchmark}' not yet implemented")
        print("Available: hle, gaia, simpleqa, deepresearch")
        sys.exit(1)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
