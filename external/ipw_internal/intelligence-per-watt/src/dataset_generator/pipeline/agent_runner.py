"""Agent runner — executes workload queries and captures token/tool metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_generator.datasets.base import DatasetSample


@dataclass
class AgentRunResult:
    """Result of running a single query through the agent pipeline."""

    query: str
    response: str
    workload_type: str
    prefill_tokens: int
    decode_tokens: int
    num_steps: int
    tool_calls: List[str]
    total_latency_s: float
    energy_j: Optional[float] = None
    step_details: List[Dict[str, Any]] = field(default_factory=list)


class AgentRunner:
    """Runs dataset queries through an LLM agent and collects metrics.

    Tries OpenHands agent first; falls back to direct vLLM API calls.
    """

    def __init__(
        self,
        model_url: str = "http://localhost:8001/v1",
        dataset_name: str = "wildchat",
        limit: Optional[int] = None,
    ):
        self.model_url = model_url
        self.dataset_name = dataset_name
        self.limit = limit

    def run(self) -> List[AgentRunResult]:
        """Load the dataset and run each sample, returning results."""
        from dataset_generator.datasets.registry import load_dataset

        samples = load_dataset(self.dataset_name, limit=self.limit)

        results: List[AgentRunResult] = []
        for sample in samples:
            result = self._run_single(sample)
            results.append(result)
        return results

    def _run_single(self, sample: DatasetSample) -> AgentRunResult:
        """Run a single sample — try agent, fall back to direct API."""
        try:
            return self._run_with_agent(sample)
        except (ImportError, Exception):
            return self._run_direct_api(sample)

    def _run_direct_api(self, sample: DatasetSample) -> AgentRunResult:
        """Call vLLM chat/completions endpoint directly via httpx."""
        import httpx

        url = f"{self.model_url}/chat/completions"
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": sample.query}],
        }

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            elapsed = time.perf_counter() - start

            content = ""
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            usage = data.get("usage", {})
            prefill = usage.get("prompt_tokens", 0)
            decode = usage.get("completion_tokens", 0)

            # Extract tool calls from the response
            tool_calls: List[str] = []
            if choices:
                tc_list = choices[0].get("message", {}).get("tool_calls", [])
                for tc in tc_list:
                    fn = tc.get("function", {})
                    tool_calls.append(fn.get("name", "unknown"))

            return AgentRunResult(
                query=sample.query,
                response=content,
                workload_type=sample.workload_type,
                prefill_tokens=prefill,
                decode_tokens=decode,
                num_steps=1,
                tool_calls=tool_calls,
                total_latency_s=elapsed,
                step_details=[{"type": "direct_api", "latency_s": elapsed}],
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            return AgentRunResult(
                query=sample.query,
                response="",
                workload_type=sample.workload_type,
                prefill_tokens=0,
                decode_tokens=0,
                num_steps=0,
                tool_calls=[],
                total_latency_s=elapsed,
                step_details=[{"type": "error", "error": str(exc)}],
            )

    def _run_with_agent(self, sample: DatasetSample) -> AgentRunResult:
        """Try running through OpenHands agent."""
        from agents.agents.openhands import OpenHandsAgent  # type: ignore[import-untyped]

        start = time.perf_counter()
        agent = OpenHandsAgent(model_url=self.model_url)
        response = agent.run(sample.query)
        elapsed = time.perf_counter() - start

        return AgentRunResult(
            query=sample.query,
            response=str(response),
            workload_type=sample.workload_type,
            prefill_tokens=getattr(response, "prompt_tokens", 0),
            decode_tokens=getattr(response, "completion_tokens", 0),
            num_steps=getattr(response, "num_steps", 1),
            tool_calls=getattr(response, "tool_calls", []),
            total_latency_s=elapsed,
            step_details=[{"type": "openhands_agent", "latency_s": elapsed}],
        )

    def save_results(self, results: List[AgentRunResult], output_dir: Path) -> Path:
        """Save results to a JSON-lines file in output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.dataset_name}_runs.jsonl"

        with open(output_path, "w") as f:
            for r in results:
                row = {
                    "query": r.query,
                    "response": r.response,
                    "workload_type": r.workload_type,
                    "prefill_tokens": r.prefill_tokens,
                    "decode_tokens": r.decode_tokens,
                    "num_steps": r.num_steps,
                    "tool_calls": r.tool_calls,
                    "total_latency_s": r.total_latency_s,
                    "energy_j": r.energy_j,
                    "step_details": r.step_details,
                }
                f.write(json.dumps(row) + "\n")

        return output_path

    @staticmethod
    def load_results(path: Path) -> List[AgentRunResult]:
        """Load AgentRunResults from a JSONL file."""
        results: List[AgentRunResult] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                results.append(
                    AgentRunResult(
                        query=data["query"],
                        response=data["response"],
                        workload_type=data.get("workload_type", ""),
                        prefill_tokens=data.get("prefill_tokens", 0),
                        decode_tokens=data.get("decode_tokens", 0),
                        num_steps=data.get("num_steps", 0),
                        tool_calls=data.get("tool_calls", []),
                        total_latency_s=data.get("total_latency_s", 0.0),
                        energy_j=data.get("energy_j"),
                        step_details=data.get("step_details", []),
                    )
                )
        return results
