#!/usr/bin/env python3
"""Direct evaluation via vLLM without loading models locally.

Evaluates trained orchestrator model on benchmarks by:
1. Sending questions to vLLM server
2. Parsing tool-use format responses
3. Executing tools (calculator, think) as needed
4. Grading final answers

Usage:
    python evals/scripts/run_direct_eval.py \
        --model-url http://localhost:8002/v1 \
        --model-name "path/to/checkpoint" \
        --limit 200 --seed 42 \
        --output results/eval.json
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))


@dataclass
class EvalSample:
    task_id: str
    question: str
    answer: str
    benchmark: str
    metadata: Dict[str, Any]


class VLLMClient:
    """Client for vLLM OpenAI-compatible API."""
    
    SYSTEM_PROMPT = """You are an intelligent orchestrator that solves tasks by using tools step by step. Available tools include calculator, code interpreter, web search, and LLM models. Be efficient and prefer low-cost tools when appropriate."""

    def __init__(self, base_url: str, model: str, max_turns: int = 5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_turns = max_turns
        
    def run(self, prompt: str) -> Any:
        """Run multi-turn orchestration."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        all_content = []
        total_tokens = 0
        final_answer = ""
        
        for turn in range(self.max_turns):
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            all_content.append(content)
            total_tokens += data.get("usage", {}).get("total_tokens", 0)
            
            if "FINAL_ANSWER:" in content:
                final_match = re.search(r"FINAL_ANSWER:\s*(.+)", content, re.DOTALL)
                final_answer = final_match.group(1).strip() if final_match else content
                break
            
            tool_match = re.search(r"TOOL:\s*(\w+)", content)
            input_match = re.search(r"INPUT:\s*(.+?)(?:\n|$)", content)
            
            if tool_match and input_match:
                tool_name = tool_match.group(1).lower()
                tool_input = input_match.group(1).strip()
                
                tool_result = self._execute_tool(tool_name, tool_input)
                
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "tool", "content": tool_result})
            else:
                final_answer = content
                break
        else:
            final_answer = all_content[-1] if all_content else ""
        
        class Response:
            def __init__(self, content: str, usage: Dict):
                self.content = content
                self.usage = type('Usage', (), usage)()
        
        return Response(
            content=final_answer,
            usage={"total_tokens": total_tokens, "prompt_tokens": 0, "completion_tokens": 0}
        )
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return result."""
        if tool_name == "calculator":
            try:
                allowed_chars = set("0123456789+-*/.() ^%")
                if all(c in allowed_chars or c.isspace() for c in tool_input):
                    expr = tool_input.replace("^", "**")
                    result = eval(expr, {"__builtins__": {}}, {})
                    return str(result)
                else:
                    return f"Error: Invalid expression"
            except Exception as e:
                return f"Error: {e}"
        elif tool_name == "think":
            return "Thought recorded."
        elif tool_name in ["code_interpreter", "web_search"]:
            return "Tool not available in evaluation mode."
        else:
            return f"Unknown tool: {tool_name}"


def load_hle_samples(limit: int, seed: int) -> List[EvalSample]:
    """Load HLE samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/hle", split="test")
        
        indices = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:limit]
        
        samples = []
        for idx in indices:
            item = ds[idx]
            samples.append(EvalSample(
                task_id=f"hle_{idx}",
                question=item.get("question", item.get("problem", "")),
                answer=item.get("answer", ""),
                benchmark="hle",
                metadata={"category": item.get("category", "unknown")}
            ))
        return samples
    except Exception as e:
        logging.warning(f"Could not load HLE: {e}")
        return []


def load_gaia_samples(limit: int, seed: int) -> List[EvalSample]:
    """Load GAIA samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
        
        indices = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:limit]
        
        samples = []
        for idx in indices:
            item = ds[idx]
            samples.append(EvalSample(
                task_id=item.get("task_id", f"gaia_{idx}"),
                question=item.get("Question", ""),
                answer=item.get("Final answer", ""),
                benchmark="gaia",
                metadata={"level": item.get("Level", 1)}
            ))
        return samples
    except Exception as e:
        logging.warning(f"Could not load GAIA: {e}")
        return []


def grade_answer(predicted: str, gold: str, benchmark: str) -> bool:
    """Grade predicted answer against gold answer."""
    pred = predicted.strip().lower()
    gold_clean = gold.strip().lower()
    
    if not gold_clean:
        return False
    
    return pred == gold_clean or gold_clean in pred or pred in gold_clean


def run_evaluation(
    client: Any,
    samples: List[EvalSample],
    benchmark_name: str,
) -> Dict[str, Any]:
    """Run evaluation on samples."""
    results = []
    correct = 0
    total_latency = 0.0
    
    for idx, sample in enumerate(samples):
        logging.info(f"[{benchmark_name}] Processing {idx+1}/{len(samples)}: {sample.task_id}")
        
        start_time = time.time()
        try:
            response = client.run(sample.question)
            content = response.content if hasattr(response, 'content') else str(response)
            error = None
        except Exception as e:
            logging.error(f"Error on {sample.task_id}: {e}")
            content = ""
            error = str(e)
        
        latency = time.time() - start_time
        total_latency += latency
        
        is_correct = grade_answer(content, sample.answer, sample.benchmark) if not error else False
        if is_correct:
            correct += 1
        
        results.append({
            "task_id": sample.task_id,
            "question": sample.question[:200] + "..." if len(sample.question) > 200 else sample.question,
            "gold_answer": sample.answer,
            "response": content[:500] + "..." if len(content) > 500 else content,
            "is_correct": is_correct,
            "latency_seconds": latency,
            "error": error,
            **sample.metadata,
        })
    
    accuracy = correct / len(samples) if samples else 0.0
    
    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "num_correct": correct,
        "num_samples": len(samples),
        "avg_latency_seconds": total_latency / len(samples) if samples else 0.0,
        "total_latency_seconds": total_latency,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run direct vLLM evaluation")
    parser.add_argument("--model-url", default="http://localhost:8001/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", nargs="+", default=["hle", "gaia"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    
    print("=" * 70)
    print(f"Direct vLLM Evaluation")
    print("=" * 70)
    print(f"Model URL: {args.model_url}")
    print(f"Model Name: {args.model_name}")
    print(f"Limit per benchmark: {args.limit}")
    print(f"Seed: {args.seed}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Max turns: {args.max_turns}")
    print("=" * 70)
    
    client = VLLMClient(args.model_url, args.model_name, args.max_turns)
    
    all_results = {
        "model_url": args.model_url,
        "model_name": args.model_name,
        "limit": args.limit,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "benchmarks": {},
    }
    
    loaders = {
        "hle": load_hle_samples,
        "gaia": load_gaia_samples,
    }
    
    for benchmark in args.benchmarks:
        if benchmark not in loaders:
            logging.warning(f"Unknown benchmark: {benchmark}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running {benchmark.upper()} benchmark")
        print(f"{'='*60}")
        
        samples = loaders[benchmark](args.limit, args.seed)
        if not samples:
            logging.warning(f"No samples loaded for {benchmark}")
            continue
        
        print(f"Loaded {len(samples)} samples")
        
        benchmark_results = run_evaluation(client, samples, benchmark)
        all_results["benchmarks"][benchmark] = benchmark_results
        
        print(f"\n{benchmark.upper()} Results:")
        print(f"  Accuracy: {benchmark_results['accuracy']:.2%}")
        print(f"  Correct: {benchmark_results['num_correct']}/{benchmark_results['num_samples']}")
        print(f"  Avg Latency: {benchmark_results['avg_latency_seconds']:.2f}s")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Benchmark':<15} {'Accuracy':<15} {'Correct':<15} {'Avg Latency':<15}")
    print("-" * 60)
    for bench, res in all_results["benchmarks"].items():
        print(f"{bench.upper():<15} {res['accuracy']:.2%}          {res['num_correct']}/{res['num_samples']:<10} {res['avg_latency_seconds']:.2f}s")
    print("=" * 70)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
