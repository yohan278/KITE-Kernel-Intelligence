"""Core types for KITE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class KernelTask:
    task_id: str
    level: int
    prompt: str
    reference_kernel: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KernelCandidate:
    task_id: str
    code: str
    compile_ok: bool
    correct: bool
    runtime_ms: Optional[float] = None
    speedup: Optional[float] = None
    compile_log: Optional[str] = None
    correctness_log: Optional[str] = None
    reference_runtime_ms: Optional[float] = None
    logs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhaseSegment:
    name: str
    start_s: float
    end_s: float
    energy_j: Optional[float] = None


@dataclass(slots=True)
class EnergyTrace:
    timestamps: List[float]
    power_w: List[float]
    energy_j: List[float]
    gpu_util: List[float] = field(default_factory=list)
    temp_c: List[float] = field(default_factory=list)
    phase_segments: List[PhaseSegment] = field(default_factory=list)
    sampling_ms: Optional[float] = None
    source: str = "unknown"
    device_id: Optional[str] = None


@dataclass(slots=True)
class RuntimeState:
    queue_depth: int
    phase_ratio: float
    batch_size: int
    concurrency: int
    power_cap: int
    clocks: str
    ttft_p95: float
    e2e_p95: float
    throughput_tps: float = 0.0
    avg_power_w: float = 0.0
    phase_id: str = "mixed"


@dataclass(slots=True)
class RewardBreakdown:
    correctness: float = 0.0
    performance: float = 0.0
    energy: float = 0.0
    latency_sla: float = 0.0
    stability: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    task_id: str
    kernel_candidate: Optional[KernelCandidate] = None
    runtime_state: Optional[RuntimeState] = None
    energy_trace: Optional[EnergyTrace] = None
    reward: RewardBreakdown = field(default_factory=RewardBreakdown)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
