# src/base.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseBenchmark(ABC):
    """Simplified base class for LLM evaluation benchmarks."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.system_instruction = system_instruction

    @abstractmethod
    def run_benchmark(self, orchestrator: Any) -> Dict[str, float]:
        """Full evaluation pipeline.
        
        Args:
            orchestrator: A BaseOrchestrater instance (or compatible object with a run() method).
        """
        raise NotImplementedError


class CLIBenchmark(BaseBenchmark):
    """Simplified base class for CLI evaluation benchmarks."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.system_instruction = system_instruction
    
    def run_benchmark(self, orchestrator: Any) -> Dict[str, float]:
        """Full evaluation pipeline.
        
        Args:
            orchestrator: A BaseOrchestrater instance (or compatible object with a run() method).
        """
        raise NotImplementedError

class DatasetBenchmark(BaseBenchmark):
    """Simplified base class for LLM evaluation benchmarks."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.system_instruction = system_instruction

    @abstractmethod
    def generate_responses(self, orchestrator: Any) -> Dict[str, Any]:
        """Run the orchestrator to produce raw outputs for the benchmark.
        
        Args:
            orchestrator: A BaseOrchestrater instance (or compatible object with a run() method).
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Score/aggregate the raw outputs."""
        raise NotImplementedError

    def run_benchmark(self, orchestrator: Any) -> Dict[str, float]:
        """Full evaluation pipeline.
        
        Args:
            orchestrator: A BaseOrchestrater instance (or compatible object with a run() method).
        """
        self.logger.info(f"Running {self.__class__.__name__} benchmark")
        generation_results = self.generate_responses(orchestrator)
        evaluation_results = self.evaluate_responses(generation_results)
        return evaluation_results
