"""Evaluation framework for SFT training with ReAct orchestrator.

Evaluates trained model on actual tasks using the orchestrator executor,
providing meaningful signals for early stopping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from orchestrator.inference.policy import InferencePolicy
    from orchestrator.inference.executor import OrchestratorExecutor, ExecutorResult
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False
    InferencePolicy = None
    OrchestratorExecutor = None
    ExecutorResult = None


@dataclass
class EvalTask:
    """Single evaluation task."""
    
    task_id: str
    prompt: str
    ground_truth: str
    category: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalTask":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", data.get("sample_id", "")),
            prompt=data.get("prompt", data.get("query", "")),
            ground_truth=data.get("ground_truth", data.get("answer", "")),
            category=data.get("category"),
        )


@dataclass
class EvalResult:
    """Results from evaluation."""
    
    task_id: str
    success: bool
    predicted: str
    ground_truth: str
    num_turns: int
    latency_seconds: float
    cost_usd: float
    error: Optional[str] = None


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    
    num_tasks: int
    success_rate: float
    avg_turns: float
    avg_latency: float
    avg_cost: float
    results: List[EvalResult] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"Success: {self.success_rate:.1%} "
            f"({int(self.success_rate * self.num_tasks)}/{self.num_tasks}), "
            f"Avg turns: {self.avg_turns:.1f}, "
            f"Avg latency: {self.avg_latency:.1f}s"
        )


class SFTEvaluator:
    """Evaluator for SFT training using ReAct orchestrator.
    
    Runs the trained orchestrator on evaluation tasks and computes
    success rate and other metrics for early stopping.
    
    Example:
        evaluator = SFTEvaluator(
            eval_tasks_path="data/eval_tasks.jsonl",
            mcp_tools=mcp_tools,
            max_turns=10,
        )
        
        # During training
        metrics = evaluator.evaluate(checkpoint_path="checkpoints/epoch_3")
        print(f"Success rate: {metrics.success_rate:.1%}")
    """
    
    def __init__(
        self,
        eval_tasks_path: str,
        mcp_tools: Optional[Dict[str, Any]] = None,
        max_turns: int = 10,
        max_eval_tasks: Optional[int] = None,
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        """Initialize evaluator.
        
        Args:
            eval_tasks_path: Path to eval tasks (JSONL file or trajectory dataset)
            mcp_tools: MCP tools dictionary (if None, loads default)
            max_turns: Maximum turns per task
            max_eval_tasks: Maximum number of tasks to evaluate (None = all)
            ollama_base_url: Ollama base URL for tool execution
            temperature: Sampling temperature
        """
        if not HAS_INFERENCE:
            raise ImportError("Inference modules required for evaluation")
        
        self.max_turns = max_turns
        self.mcp_tools = mcp_tools or {}
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_eval_tasks = max_eval_tasks
        
        # Load eval tasks
        self.eval_tasks = self._load_eval_tasks(eval_tasks_path)
        print(f"Loaded {len(self.eval_tasks)} evaluation tasks")
    
    def _load_eval_tasks(self, path: str) -> List[EvalTask]:
        """Load evaluation tasks from file.
        
        Supports:
        - JSONL files with task records
        - Arrow datasets (trajectory format)
        """
        path = Path(path)
        tasks = []
        
        if path.suffix == ".jsonl":
            # Load from JSONL
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    tasks.append(EvalTask.from_dict(data))
        
        elif path.is_dir():
            # Try loading as Arrow dataset
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(str(path))
                
                for item in dataset:
                    # Extract user query from conversations
                    prompt = None
                    for conv in item.get("conversations", []):
                        if conv.get("role") == "user":
                            prompt = conv.get("content")
                            break
                    
                    if prompt:
                        tasks.append(EvalTask(
                            task_id=item.get("sample_id", ""),
                            prompt=prompt,
                            ground_truth=item.get("ground_truth", ""),
                            category=item.get("category"),
                        ))
            except ImportError:
                raise ImportError("datasets library required for Arrow format")
        
        else:
            raise ValueError(f"Unsupported eval tasks format: {path}")
        
        # Apply limit if specified
        if self.max_eval_tasks and len(tasks) > self.max_eval_tasks:
            tasks = tasks[:self.max_eval_tasks]
        
        return tasks
    
    def evaluate(
        self,
        checkpoint_path: str,
        verbose: bool = False,
    ) -> EvalMetrics:
        """Evaluate orchestrator on eval tasks.
        
        Args:
            checkpoint_path: Path to model checkpoint
            verbose: Print per-task results
            
        Returns:
            EvalMetrics with aggregated results
        """
        print(f"\n{'='*70}")
        print(f"Running evaluation on {len(self.eval_tasks)} tasks")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*70}")
        
        # Load policy from checkpoint
        policy = InferencePolicy.from_checkpoint(
            checkpoint_path,
            temperature=self.temperature,
        )
        
        # Create executor
        executor = OrchestratorExecutor(
            policy=policy,
            mcp_tools=self.mcp_tools,
            max_turns=self.max_turns,
        )
        
        # Run eval on each task
        results = []
        for i, task in enumerate(self.eval_tasks):
            if verbose or (i + 1) % 10 == 0:
                print(f"  Evaluating task {i+1}/{len(self.eval_tasks)}...")
            
            try:
                # Execute task
                exec_result = executor.execute(task.prompt)
                
                # Check correctness (simple string match for now)
                success = self._check_answer(
                    exec_result.final_answer,
                    task.ground_truth
                )
                
                results.append(EvalResult(
                    task_id=task.task_id,
                    success=success,
                    predicted=exec_result.final_answer,
                    ground_truth=task.ground_truth,
                    num_turns=len(exec_result.turns),
                    latency_seconds=exec_result.total_latency_ms / 1000.0,
                    cost_usd=exec_result.total_cost_usd,
                    error=exec_result.error,
                ))
                
                if verbose:
                    status = "✓" if success else "✗"
                    print(f"    {status} {task.task_id}: {exec_result.final_answer[:50]}")
            
            except Exception as e:
                results.append(EvalResult(
                    task_id=task.task_id,
                    success=False,
                    predicted="",
                    ground_truth=task.ground_truth,
                    num_turns=0,
                    latency_seconds=0.0,
                    cost_usd=0.0,
                    error=str(e),
                ))
                if verbose:
                    print(f"    ✗ {task.task_id}: Error - {e}")
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        print(f"\n{metrics}")
        print(f"{'='*70}\n")
        
        return metrics
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.
        
        Uses simple normalization and string matching.
        Can be extended with fuzzy matching or custom graders.
        """
        # Normalize
        pred = predicted.lower().strip()
        gt = ground_truth.lower().strip()
        
        # Exact match
        if pred == gt:
            return True
        
        # Contains match
        if gt in pred:
            return True
        
        # Numeric match (for math problems)
        try:
            pred_num = float(pred.replace(",", ""))
            gt_num = float(gt.replace(",", ""))
            return abs(pred_num - gt_num) < 1e-6
        except (ValueError, AttributeError):
            pass
        
        return False
    
    def _compute_metrics(self, results: List[EvalResult]) -> EvalMetrics:
        """Compute aggregated metrics from results."""
        num_tasks = len(results)
        num_success = sum(1 for r in results if r.success)
        
        # Filter out failed executions for aggregates
        completed = [r for r in results if r.error is None]
        
        return EvalMetrics(
            num_tasks=num_tasks,
            success_rate=num_success / num_tasks if num_tasks > 0 else 0.0,
            avg_turns=sum(r.num_turns for r in completed) / len(completed) if completed else 0.0,
            avg_latency=sum(r.latency_seconds for r in completed) / len(completed) if completed else 0.0,
            avg_cost=sum(r.cost_usd for r in completed) / len(completed) if completed else 0.0,
            results=results,
        )
    
    def save_results(self, metrics: EvalMetrics, output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "num_tasks": metrics.num_tasks,
            "success_rate": metrics.success_rate,
            "avg_turns": metrics.avg_turns,
            "avg_latency": metrics.avg_latency,
            "avg_cost": metrics.avg_cost,
            "results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "predicted": r.predicted,
                    "ground_truth": r.ground_truth,
                    "num_turns": r.num_turns,
                    "latency_seconds": r.latency_seconds,
                    "cost_usd": r.cost_usd,
                    "error": r.error,
                }
                for r in metrics.results
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved evaluation results to {output_path}")
