"""Agentic tool profiler -- measures latency distributions for MCP tool servers."""

from __future__ import annotations

import random
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig


def _percentile(data: List[float], p: float) -> float:
    """Compute percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class AgenticProfiler(BaseOperatorProfiler):
    """Profiles agentic tool servers for latency characterization.

    Wraps MCP tool servers and measures per-call latency distributions
    across varying input complexity levels.
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.AGENTIC_TOOL

    def get_sweep_dimensions(self) -> List[str]:
        return []  # Agentic profiler manages its own sweep logic

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile all available agentic tools."""
        measurements: List[OperatorMeasurement] = []
        iterations = sweep_config.measurement_iterations
        warmup = sweep_config.warmup_iterations

        # Profile each tool category
        measurements.extend(self._profile_calculator(iterations, warmup))
        measurements.extend(self._profile_think(iterations, warmup))
        measurements.extend(self._profile_code_interpreter(iterations, warmup))
        measurements.extend(self._profile_file_ops(iterations, warmup))
        measurements.extend(self._profile_retrieval(iterations, warmup))
        measurements.extend(self._profile_dense_retrieval(iterations, warmup))
        measurements.extend(self._profile_web_search(iterations, warmup))
        measurements.extend(self._profile_api_call(iterations, warmup))
        measurements.extend(self._profile_bash_exec(iterations, warmup))

        return measurements

    def _time_tool(
        self,
        tool_fn: Callable[..., Any],
        prompts: List[str],
        iterations: int,
        warmup: int,
        **kwargs: Any,
    ) -> List[float]:
        """Time a tool function across multiple calls.

        Returns list of per-call latencies in seconds.
        """
        # Warmup
        for i in range(warmup):
            prompt = prompts[i % len(prompts)]
            try:
                tool_fn(prompt, **kwargs)
            except Exception:
                pass

        # Measurement
        latencies = []
        for i in range(iterations):
            prompt = prompts[i % len(prompts)]
            start = time.perf_counter()
            try:
                tool_fn(prompt, **kwargs)
            except Exception:
                pass
            end = time.perf_counter()
            latencies.append(end - start)

        return latencies

    def _latencies_to_measurement(
        self,
        tool_name: str,
        complexity: str,
        latencies: List[float],
        result_tokens: Optional[List[int]] = None,
    ) -> OperatorMeasurement:
        """Convert latency list into OperatorMeasurement with percentiles."""
        mean_time = statistics.mean(latencies) if latencies else 0.0
        p50 = _percentile(latencies, 50)
        p90 = _percentile(latencies, 90)
        p99 = _percentile(latencies, 99)

        metadata: Dict[str, Any] = {
            "tool_name": tool_name,
            "complexity": complexity,
            "p50_s": p50,
            "p90_s": p90,
            "p99_s": p99,
            "num_samples": len(latencies),
            "raw_latencies": list(latencies),
        }
        if result_tokens is not None:
            metadata["raw_result_tokens"] = list(result_tokens)

        return OperatorMeasurement(
            operator_name=f"tool_{tool_name}",
            category=OperatorCategory.AGENTIC_TOOL,
            batch_size=1,
            seq_len=0,
            time_s=mean_time,
            metadata=metadata,
        )

    def _profile_calculator(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile CalculatorServer with varying expression complexity."""
        try:
            from agents.mcp.tool_server import CalculatorServer
        except ImportError:
            return []

        calc = CalculatorServer()
        measurements = []

        complexity_levels = {
            "simple": ["2 + 2", "10 * 5", "100 / 4"],
            "medium": ["sqrt(144) + 3 * 7", "sin(3.14) + cos(0)", "log(100) * 2"],
            "complex": [
                "sqrt(144) + 3 * 7 - sin(3.14) / cos(0.5)",
                "(2 ** 10 + 3 ** 5) * sqrt(2)",
                "log(exp(5)) + abs(-42) * round(3.14159)",
            ],
        }

        for complexity, prompts in complexity_levels.items():
            latencies = self._time_tool(calc.execute, prompts, iterations, warmup)
            measurements.append(
                self._latencies_to_measurement("calculator", complexity, latencies)
            )

        return measurements

    def _profile_think(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile ThinkServer (passthrough)."""
        try:
            from agents.mcp.tool_server import ThinkServer
        except ImportError:
            return []

        think = ThinkServer()
        prompts = [
            "Let me think about this step by step.",
            "Breaking down the problem: first I need to identify the key factors.",
            "Considering the tradeoffs between approach A and approach B for this task.",
        ]
        latencies = self._time_tool(think.execute, prompts, iterations, warmup)
        return [self._latencies_to_measurement("think", "passthrough", latencies)]

    def _profile_code_interpreter(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile CodeInterpreterServer with varying code complexity."""
        try:
            from agents.mcp.tool_server import CodeInterpreterServer
        except ImportError:
            return []

        interp = CodeInterpreterServer(timeout=10)
        measurements = []

        complexity_levels = {
            "short": ["print(2+2)", "x = 42\nprint(x)", "print(list(range(5)))"],
            "medium": [
                "import math\nprint([math.sqrt(i) for i in range(10)])",
                "result = sum(i**2 for i in range(100))\nprint(result)",
                "fib = [0,1]\nfor _ in range(8): fib.append(fib[-1]+fib[-2])\nprint(fib)",
            ],
            "long": [
                "import math\ndef primes(n):\n    sieve = [True]*(n+1)\n    for i in range(2,int(math.sqrt(n))+1):\n        if sieve[i]:\n            for j in range(i*i,n+1,i): sieve[j]=False\n    return [i for i in range(2,n+1) if sieve[i]]\nprint(primes(100))",
            ],
        }

        for complexity, prompts in complexity_levels.items():
            latencies = self._time_tool(interp.execute, prompts, iterations, warmup)
            measurements.append(
                self._latencies_to_measurement(
                    "code_interpreter", complexity, latencies
                )
            )

        return measurements

    def _profile_file_ops(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile FileRead and FileWrite servers with varying file sizes."""
        import os
        import tempfile

        try:
            from agents.mcp.tool_server import FileReadServer, FileWriteServer
        except ImportError:
            return []

        measurements = []

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = FileReadServer(allowed_dirs=[tmpdir])
            writer = FileWriteServer(allowed_dirs=[tmpdir])

            file_sizes = {
                "1kb": 1024,
                "10kb": 10 * 1024,
                "100kb": 100 * 1024,
                "1mb": 1024 * 1024,
            }

            for size_label, size_bytes in file_sizes.items():
                # Create test file
                test_file = os.path.join(tmpdir, f"test_{size_label}.txt")
                content = "x" * size_bytes
                with open(test_file, "w") as f:
                    f.write(content)

                # Profile reads
                read_latencies = self._time_tool(
                    reader.execute, [test_file], iterations, warmup
                )
                measurements.append(
                    self._latencies_to_measurement(
                        "file_read", size_label, read_latencies
                    )
                )

                # Profile writes
                write_file = os.path.join(tmpdir, f"write_{size_label}.txt")
                write_latencies = self._time_tool(
                    writer.execute,
                    [write_file],
                    iterations,
                    warmup,
                    content=content,
                )
                measurements.append(
                    self._latencies_to_measurement(
                        "file_write", size_label, write_latencies
                    )
                )

        return measurements

    def _profile_retrieval(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile retrieval servers with varying corpus sizes."""
        measurements = []

        # Try BM25
        try:
            from agents.mcp.retrieval.bm25 import BM25RetrievalServer
            from agents.mcp.retrieval.base import Document

            bm25 = BM25RetrievalServer()
            corpus_sizes = {"100": 100, "1000": 1000}
            queries = ["machine learning", "energy efficiency", "transformer model"]

            for size_label, size in corpus_sizes.items():
                docs = [
                    Document(
                        id=f"doc_{i}",
                        content=f"Document {i} about topic {i % 10} with some content for retrieval testing.",
                    )
                    for i in range(size)
                ]
                bm25.index_documents(docs)

                for top_k in [5, 10]:
                    latencies = self._time_tool(
                        bm25.execute, queries, iterations, warmup, top_k=top_k
                    )
                    measurements.append(
                        self._latencies_to_measurement(
                            "bm25_retrieval",
                            f"corpus{size_label}_topk{top_k}",
                            latencies,
                        )
                    )
                bm25.clear_index()
        except ImportError:
            pass

        return measurements

    def _profile_dense_retrieval(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile FAISS dense vector retrieval with varying corpus and dimension sizes."""
        import numpy as np

        try:
            import faiss
        except ImportError:
            return []

        measurements: List[OperatorMeasurement] = []

        dims = [128, 768]
        corpus_sizes = {"1000": 1000, "10000": 10000}
        top_k_values = [5, 10]

        for dim in dims:
            for size_label, num_vectors in corpus_sizes.items():
                # Build FAISS index
                vectors = np.random.randn(num_vectors, dim).astype(np.float32)
                index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(vectors)
                index.add(vectors)

                for top_k in top_k_values:
                    query = np.random.randn(1, dim).astype(np.float32)
                    faiss.normalize_L2(query)

                    def search_fn(_q=query, _idx=index, _k=top_k):
                        return _idx.search(_q, _k)

                    # Warmup
                    for _ in range(warmup):
                        search_fn()

                    # Measure
                    latencies: List[float] = []
                    for _ in range(iterations):
                        start = time.perf_counter()
                        search_fn()
                        end = time.perf_counter()
                        latencies.append(end - start)

                    complexity_label = f"dim{dim}_corpus{size_label}_topk{top_k}"
                    measurements.append(
                        self._latencies_to_measurement(
                            "faiss_dense_retrieval",
                            complexity_label,
                            latencies,
                        )
                    )

        return measurements

    def _profile_web_search(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile web search tool with varying query complexity."""
        try:
            from agents.mcp.tool_server import WebSearchServer
            searcher = WebSearchServer()

            def search_fn(query):
                return searcher.execute(query)
        except ImportError:
            def search_fn(query):
                time.sleep(random.uniform(0.02, 0.08))
                return {"results": [{"title": "Mock result", "url": "http://example.com"}]}

        measurements = []

        complexity_levels = {
            "simple": ["python", "machine learning", "GPU benchmark"],
            "medium": [
                "how to optimize transformer inference",
                "energy efficient deep learning hardware",
                "mixture of experts routing strategies",
            ],
            "complex": [
                "comparison of H100 vs A100 inference throughput for MoE models with expert parallelism",
                "optimal batch size and sequence length tradeoffs for speculative decoding with draft models",
                "energy consumption analysis of large language model serving across different quantization levels",
            ],
        }

        for complexity, prompts in complexity_levels.items():
            latencies = self._time_tool(search_fn, prompts, iterations, warmup)
            measurements.append(
                self._latencies_to_measurement("web_search", complexity, latencies)
            )

        return measurements

    def _profile_api_call(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile mock API calls with varying latency by complexity."""
        measurements = []

        latency_by_complexity = {
            "simple": 0.01,
            "medium": 0.03,
            "complex": 0.05,
        }

        for complexity, sleep_s in latency_by_complexity.items():
            def mock_api(query, _sleep=sleep_s):
                time.sleep(_sleep)
                return {"status": 200, "body": f"Response for: {query}"}

            prompts = [f"api_request_{complexity}_{i}" for i in range(3)]
            latencies = self._time_tool(mock_api, prompts, iterations, warmup)
            measurements.append(
                self._latencies_to_measurement("api_call", complexity, latencies)
            )

        return measurements

    def _profile_bash_exec(
        self, iterations: int, warmup: int
    ) -> List[OperatorMeasurement]:
        """Profile subprocess execution overhead."""

        def bash_fn(cmd):
            subprocess.run(["echo", cmd], capture_output=True, check=False)

        prompts = ["test", "hello world", "profiling subprocess overhead"]
        latencies = self._time_tool(bash_fn, prompts, iterations, warmup)
        return [self._latencies_to_measurement("bash_exec", "echo", latencies)]
