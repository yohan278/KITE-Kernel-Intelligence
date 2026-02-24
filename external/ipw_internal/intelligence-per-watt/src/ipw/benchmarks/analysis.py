"""Energy parameter extraction from benchmark results.

This module implements the methodology from:
"Benchmark-driven Models for Energy Analysis and Attribution of
GPU-Accelerated Supercomputing" (SC '25, Antepara et al.)

DOI: https://doi.org/10.1145/3712285.3712359

The key insight is to separate energy into:
- Control energy: Power consumed even when processing zeros
- Datapath energy: Additional power when processing real data

By running benchmarks at different memory/compute ratios,
we can extract:
- ε_mem (pJ/bit): Energy per bit of memory transferred
- ε_compute (pJ/FLOP): Energy per floating-point operation

Cache-level memory hierarchy extraction (Equations 6-8 from SC'25):
- ε_L2+L1 × BW_L2 = P - P_const - ε_VFPU × PERF_VFPU    (Eq. 6)
- ε_L1 × BW_L1 = P - P_const - ε_VFPU × PERF_VFPU       (Eq. 7)
- ε_L2 = ε_L2+L1 - ε_L1                                  (Eq. 8)
- ε_HBM = ε_HBM+L2+L1 - ε_L2 - ε_L1

GEMM energy extraction (Equation 9 from SC'25):
- ε_MFPU × PERF = P - P_const - ε_L1×BW_L1 - ε_L2×BW_L2 - ε_HBM×BW_HBM

Implementation notes:
- Cache workloads use .add_(const) which performs FLOPs (loop control, address calc)
- We subtract FPU contribution from cache measurements per Eq. 6-7
- GEMM extraction currently only subtracts estimated HBM contribution (L1/L2
  require profiler integration with NSight/rocprof for accurate measurement)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ipw.benchmarks.types import (
    BenchmarkResult,
    DataType,
    EnergyParameters,
    Platform,
    WorkloadType,
)


class CacheLevel(Enum):
    """Memory hierarchy cache levels for energy extraction."""

    L1 = "l1"
    L2 = "l2"
    HBM = "hbm"
    # Combined levels for subtraction method
    L1_L2 = "l1_l2"  # L1 + L2 combined
    L1_L2_HBM = "l1_l2_hbm"  # All levels combined (full memory hierarchy)


@dataclass
class ParameterFit:
    """Results of fitting energy parameters to data."""

    slope: float  # Energy coefficient (pJ/unit)
    intercept: float  # Base power (W)
    r_squared: float  # Goodness of fit
    n_samples: int  # Number of data points used


def extract_parameters(
    results: List[BenchmarkResult],
    platform: Platform,
    hardware_name: str,
    idle_cpu_watts: float = 0.0,
    idle_gpu_watts: float = 0.0,
) -> EnergyParameters:
    """Extract energy parameters from benchmark results.

    Implements the SC'25 paper's methodology with proper extraction order:
    1. Separate results by workload type and use_zeros flag
    2. Extract compute FPU energy FIRST (needed for cache subtraction per Eq. 6-7)
    3. Extract cache-level energy (subtracting FPU contribution)
    4. Extract GEMM energy (subtracting memory hierarchy contribution per Eq. 9)

    The order matters because:
    - Cache energy depends on FPU energy (to subtract FPU overhead)
    - GEMM energy depends on cache energy (to subtract memory overhead)

    Args:
        results: List of benchmark results from characterization.
        platform: Platform these results are from.
        hardware_name: Hardware identifier.
        idle_cpu_watts: Measured idle CPU power.
        idle_gpu_watts: Measured idle GPU power.

    Returns:
        EnergyParameters with extracted values.
    """
    params = EnergyParameters(
        platform=platform,
        hardware_name=hardware_name,
        p_idle_cpu_watts=idle_cpu_watts,
        p_idle_gpu_watts=idle_gpu_watts,
    )

    idle_power = idle_cpu_watts + idle_gpu_watts

    # Separate results by type
    memory_results = [
        r for r in results if r.workload.workload_type == WorkloadType.MEMORY_BANDWIDTH
    ]
    # Include cache-level workloads in memory results
    cache_level_results = [
        r for r in results if r.workload.workload_type in [
            WorkloadType.CACHE_L1, WorkloadType.CACHE_L2, WorkloadType.HBM_BANDWIDTH
        ]
    ]
    memory_results = memory_results + cache_level_results

    compute_results = [
        r for r in results if r.workload.workload_type == WorkloadType.COMPUTE_BOUND
    ]
    gemm_results = [
        r for r in results if r.workload.workload_type == WorkloadType.GEMM
    ]

    # Step 1: Extract compute FPU energy FIRST (needed for cache subtraction)
    # Per SC'25 Eq. 6-7: cache energy must subtract FPU contribution
    fp_compute_results = [
        r for r in compute_results
        if r.workload.config.data_type not in [DataType.INT32, DataType.INT8]
    ]
    if fp_compute_results:
        params = _extract_compute_params(params, fp_compute_results, idle_power)

    # Extract INT32 compute parameters separately
    int32_results = [
        r for r in compute_results
        if r.workload.config.data_type == DataType.INT32
    ]
    if int32_results:
        params = _extract_int32_params(params, int32_results, idle_power)

    # Step 2: Extract memory/cache parameters (now with FPU subtraction)
    if memory_results:
        # Check if results contain cache-level metadata for fine-grained extraction
        has_cache_levels = _has_cache_level_metadata(memory_results)

        if has_cache_levels:
            # Use cache-level extraction (SC'25 Equations 6-8)
            # Pass compute results for FPU energy subtraction
            params = _extract_cache_level_params(
                params, memory_results, idle_power,
                compute_results=fp_compute_results
            )
        else:
            # Fall back to combined memory extraction (unified memory platforms like macOS)
            params = _extract_memory_params(params, memory_results, idle_power)

    # Step 3: Extract GEMM parameters (now can subtract memory hierarchy)
    # Per SC'25 Eq. 9: GEMM energy should subtract L1/L2/HBM contributions
    if gemm_results:
        params = _extract_gemm_params(params, gemm_results, idle_power)

    # Step 4: Extract inference-level parameters (for E_call scaling law)
    # Separate standard results from CUDA graph (raw) results
    inference_gemm_results = [
        r for r in results
        if r.workload.workload_type == WorkloadType.INFERENCE_GEMM
        and not r.workload.config.use_cuda_graphs
    ]
    attention_results = [
        r for r in results
        if r.workload.workload_type == WorkloadType.ATTENTION
        and not r.workload.config.use_cuda_graphs
    ]
    kv_cache_results = [
        r for r in results
        if r.workload.workload_type == WorkloadType.KV_CACHE_IO
        and not r.workload.config.use_cuda_graphs
    ]
    nccl_results = [
        r for r in results if r.workload.workload_type == WorkloadType.NCCL_COLLECTIVE
    ]
    batched_decode_results = [
        r for r in results
        if r.workload.workload_type == WorkloadType.BATCHED_DECODE
        and not r.workload.config.use_cuda_graphs
    ]

    if inference_gemm_results:
        params = _extract_inference_gemm_params(params, inference_gemm_results, idle_power)
    if attention_results:
        params = _extract_attention_params(params, attention_results, idle_power)
    if kv_cache_results:
        params = _extract_kv_cache_params(params, kv_cache_results, idle_power)
    if nccl_results:
        params = _extract_nccl_params(params, nccl_results, idle_power)
    if batched_decode_results:
        params = _extract_batched_decode_params(params, batched_decode_results, idle_power)

    # Step 5: Extract raw (CUDA graph) inference parameters — theoretical hardware floor
    raw_inference_gemm = [
        r for r in results
        if r.workload.workload_type == WorkloadType.INFERENCE_GEMM
        and r.workload.config.use_cuda_graphs
    ]
    raw_attention = [
        r for r in results
        if r.workload.workload_type == WorkloadType.ATTENTION
        and r.workload.config.use_cuda_graphs
    ]
    raw_kv_cache = [
        r for r in results
        if r.workload.workload_type == WorkloadType.KV_CACHE_IO
        and r.workload.config.use_cuda_graphs
    ]
    raw_batched_decode = [
        r for r in results
        if r.workload.workload_type == WorkloadType.BATCHED_DECODE
        and r.workload.config.use_cuda_graphs
    ]

    if raw_inference_gemm:
        params = _extract_inference_gemm_raw_params(params, raw_inference_gemm, idle_power)
    if raw_attention:
        params = _extract_attention_raw_params(params, raw_attention, idle_power)
    if raw_kv_cache:
        params = _extract_kv_cache_raw_params(params, raw_kv_cache, idle_power)
    if raw_batched_decode:
        params = _extract_batched_decode_raw_params(params, raw_batched_decode, idle_power)

    return params


def _has_cache_level_metadata(results: List[BenchmarkResult]) -> bool:
    """Check if benchmark results contain cache-level metadata.

    Returns True if any result has a 'cache_level' parameter, indicating
    the benchmarks were run with cache-level isolation (NVIDIA/AMD GPUs).
    Returns False for unified memory platforms (macOS) or legacy benchmarks.
    """
    for r in results:
        if r.workload.config.params.get("cache_level") is not None:
            return True
    return False


def _get_cache_level(result: BenchmarkResult) -> Optional[CacheLevel]:
    """Extract the cache level from a benchmark result's config params.

    Returns:
        CacheLevel enum if found, None otherwise.
    """
    level_str = result.workload.config.params.get("cache_level")
    if level_str is None:
        return None

    # Normalize to lowercase for matching
    level_str = level_str.lower()

    # Map string values to CacheLevel enum
    level_map = {
        "l1": CacheLevel.L1,
        "l2": CacheLevel.L2,
        "hbm": CacheLevel.HBM,
        "l1_l2": CacheLevel.L1_L2,
        "l1+l2": CacheLevel.L1_L2,
        "l1_l2_hbm": CacheLevel.L1_L2_HBM,
        "l1+l2+hbm": CacheLevel.L1_L2_HBM,
        "all": CacheLevel.L1_L2_HBM,
    }

    return level_map.get(level_str)


def _set_cache_level_params(
    params: EnergyParameters,
    level: str,
    values: Dict[str, float],
) -> None:
    """Set cache-level energy parameters on EnergyParameters object.

    Uses setattr to handle forward compatibility with types.py changes.
    If the attribute doesn't exist on EnergyParameters, the value is stored
    but may not persist (depends on whether types.py defines the field).

    Args:
        params: EnergyParameters object to update.
        level: Cache level name ("l1", "l2", or "hbm").
        values: Dictionary with "total", "control", and "datapath" values.
    """
    # Map of level to attribute names
    attr_map = {
        "l1": ("e_l1_pj_per_bit", "e_l1_control_pj_per_bit", "e_l1_datapath_pj_per_bit"),
        "l2": ("e_l2_pj_per_bit", "e_l2_control_pj_per_bit", "e_l2_datapath_pj_per_bit"),
        "hbm": ("e_hbm_pj_per_bit", "e_hbm_control_pj_per_bit", "e_hbm_datapath_pj_per_bit"),
    }

    if level not in attr_map:
        return

    total_attr, control_attr, datapath_attr = attr_map[level]

    # Try to set the attributes - will work if types.py defines them
    try:
        setattr(params, total_attr, values.get("total", 0.0))
        setattr(params, control_attr, values.get("control", 0.0))
        setattr(params, datapath_attr, values.get("datapath", 0.0))
    except (AttributeError, TypeError):
        # If the EnergyParameters dataclass doesn't have these fields yet,
        # we can't set them. This is expected during transition.
        pass


def _get_cache_energy(
    params: EnergyParameters,
    level: str,
    energy_type: str,
) -> float:
    """Get cache-level energy value from EnergyParameters.

    Uses getattr with fallback for forward compatibility.

    Args:
        params: EnergyParameters object.
        level: Cache level ("l1", "l2", or "hbm").
        energy_type: Type of energy ("total", "control", or "datapath").

    Returns:
        Energy value in pJ/bit, or 0.0 if not available.
    """
    # Build attribute name
    suffix_map = {
        "total": "_pj_per_bit",
        "control": "_control_pj_per_bit",
        "datapath": "_datapath_pj_per_bit",
    }

    if energy_type not in suffix_map:
        return 0.0

    attr_name = f"e_{level}{suffix_map[energy_type]}"
    return getattr(params, attr_name, 0.0)


def _extract_memory_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract memory energy parameters from bandwidth tests.

    This is the fallback for unified memory platforms (macOS) or
    benchmarks without cache-level metadata.
    """
    # Separate by zeros flag
    zeros_results = [r for r in results if r.workload.config.use_zeros]
    random_results = [r for r in results if not r.workload.config.use_zeros]

    # Fit control energy (zeros)
    if zeros_results:
        fit = _fit_memory_energy(zeros_results, idle_power)
        if fit:
            params.e_memory_control_pj_per_bit = fit.slope

    # Fit total energy (random)
    if random_results:
        fit = _fit_memory_energy(random_results, idle_power)
        if fit:
            params.e_memory_pj_per_bit = fit.slope

    # Datapath = total - control
    params.e_memory_datapath_pj_per_bit = (
        params.e_memory_pj_per_bit - params.e_memory_control_pj_per_bit
    )

    return params


def _extract_cache_level_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
    compute_results: Optional[List[BenchmarkResult]] = None,
) -> EnergyParameters:
    """Extract cache-level energy parameters using SC'25 methodology.

    Implements Equations 6-8 from Antepara et al.:
    - First, fit energy for L1-only workloads -> ε_L1
    - Then, fit energy for L1+L2 workloads -> ε_L1+L2
    - Subtract: ε_L2 = ε_L1+L2 - ε_L1
    - Similarly for HBM: ε_HBM = ε_all - ε_L2 - ε_L1

    IMPORTANT: Cache workloads include FPU overhead (the .add_() operation).
    Per Equations 6-7, we subtract the FPU contribution:
        ε_cache × BW = P - P_const - ε_VFPU × PERF_VFPU

    Args:
        params: EnergyParameters to populate with cache-level values.
        results: Benchmark results with cache_level metadata.
        idle_power: Combined idle power (CPU + GPU).
        compute_results: Optional compute benchmark results for extracting
            FPU energy coefficient. If provided, FPU contribution is subtracted
            from cache measurements.

    Returns:
        Updated EnergyParameters with cache-level energy values.
    """
    # Extract FPU energy coefficient from compute results (if available)
    # This is needed to subtract FPU contribution from cache measurements
    fpu_energy = 0.0
    if compute_results:
        fpu_energy = _get_vector_fpu_energy(compute_results, idle_power)

    # Group results by cache level
    by_level: Dict[CacheLevel, List[BenchmarkResult]] = {}
    for r in results:
        level = _get_cache_level(r)
        if level is not None:
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(r)

    # Also keep results without cache level for combined memory extraction
    no_level_results = [r for r in results if _get_cache_level(r) is None]

    # Extract L1 energy (direct measurement) with FPU subtraction
    if CacheLevel.L1 in by_level:
        l1_params = _extract_single_cache_level_params(
            by_level[CacheLevel.L1], idle_power, fpu_energy
        )
        if l1_params:
            _set_cache_level_params(params, "l1", l1_params)

    # Extract L1+L2 combined energy (for subtraction) with FPU subtraction
    e_l1_l2_total = 0.0
    e_l1_l2_control = 0.0
    if CacheLevel.L1_L2 in by_level:
        l1_l2_params = _extract_single_cache_level_params(
            by_level[CacheLevel.L1_L2], idle_power, fpu_energy
        )
        if l1_l2_params:
            e_l1_l2_total = l1_l2_params["total"]
            e_l1_l2_control = l1_l2_params["control"]

    # Extract L2 energy via subtraction (Eq. 8): ε_L2 = ε_L1+L2 - ε_L1
    if CacheLevel.L2 in by_level:
        # Direct L2 measurement available - with FPU subtraction
        l2_params = _extract_single_cache_level_params(
            by_level[CacheLevel.L2], idle_power, fpu_energy
        )
        if l2_params:
            _set_cache_level_params(params, "l2", l2_params)
    elif e_l1_l2_total > 0:
        # Use subtraction method
        e_l1_total = _get_cache_energy(params, "l1", "total")
        e_l1_control = _get_cache_energy(params, "l1", "control")
        l2_total = max(0.0, e_l1_l2_total - e_l1_total)
        l2_control = max(0.0, e_l1_l2_control - e_l1_control)
        l2_datapath = max(0.0, l2_total - l2_control)
        _set_cache_level_params(params, "l2", {
            "total": l2_total,
            "control": l2_control,
            "datapath": l2_datapath,
        })

    # Extract HBM energy with FPU subtraction
    e_all_total = 0.0
    e_all_control = 0.0
    if CacheLevel.L1_L2_HBM in by_level:
        all_params = _extract_single_cache_level_params(
            by_level[CacheLevel.L1_L2_HBM], idle_power, fpu_energy
        )
        if all_params:
            e_all_total = all_params["total"]
            e_all_control = all_params["control"]

    if CacheLevel.HBM in by_level:
        # Direct HBM measurement available - with FPU subtraction
        hbm_params = _extract_single_cache_level_params(
            by_level[CacheLevel.HBM], idle_power, fpu_energy
        )
        if hbm_params:
            _set_cache_level_params(params, "hbm", hbm_params)
    elif e_all_total > 0:
        # Use subtraction: ε_HBM = ε_all - ε_L2 - ε_L1
        e_l1_total = _get_cache_energy(params, "l1", "total")
        e_l1_control = _get_cache_energy(params, "l1", "control")
        e_l2_total = _get_cache_energy(params, "l2", "total")
        e_l2_control = _get_cache_energy(params, "l2", "control")
        hbm_total = max(0.0, e_all_total - e_l2_total - e_l1_total)
        hbm_control = max(0.0, e_all_control - e_l2_control - e_l1_control)
        hbm_datapath = max(0.0, hbm_total - hbm_control)
        _set_cache_level_params(params, "hbm", {
            "total": hbm_total,
            "control": hbm_control,
            "datapath": hbm_datapath,
        })

    # Also populate combined memory fields for backwards compatibility
    # Use the full hierarchy (HBM) or sum of all levels
    e_hbm_total = _get_cache_energy(params, "hbm", "total")
    e_l2_total = _get_cache_energy(params, "l2", "total")
    e_l1_total = _get_cache_energy(params, "l1", "total")

    if e_hbm_total > 0:
        # If we have HBM data, the combined memory energy is dominated by HBM
        # (HBM is the slowest/most energy-intensive in the hierarchy)
        params.e_memory_pj_per_bit = e_hbm_total
        params.e_memory_control_pj_per_bit = _get_cache_energy(params, "hbm", "control")
        params.e_memory_datapath_pj_per_bit = _get_cache_energy(params, "hbm", "datapath")
    elif e_l2_total > 0:
        params.e_memory_pj_per_bit = e_l2_total
        params.e_memory_control_pj_per_bit = _get_cache_energy(params, "l2", "control")
        params.e_memory_datapath_pj_per_bit = _get_cache_energy(params, "l2", "datapath")
    elif e_l1_total > 0:
        params.e_memory_pj_per_bit = e_l1_total
        params.e_memory_control_pj_per_bit = _get_cache_energy(params, "l1", "control")
        params.e_memory_datapath_pj_per_bit = _get_cache_energy(params, "l1", "datapath")

    # If there are results without cache level, also extract combined memory params
    if no_level_results:
        combined_params = _extract_memory_params(
            EnergyParameters(platform=params.platform, hardware_name=params.hardware_name),
            no_level_results,
            idle_power,
        )
        # Only use combined if we don't have cache-level data
        if params.e_memory_pj_per_bit == 0.0:
            params.e_memory_pj_per_bit = combined_params.e_memory_pj_per_bit
            params.e_memory_control_pj_per_bit = combined_params.e_memory_control_pj_per_bit
            params.e_memory_datapath_pj_per_bit = combined_params.e_memory_datapath_pj_per_bit

    return params


def _extract_single_cache_level_params(
    results: List[BenchmarkResult],
    idle_power: float,
    fpu_energy_pj_per_flop: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Extract energy parameters for a single cache level.

    Implements Equations 6-7 from SC'25 paper, subtracting FPU contribution:
        ε_cache × BW = P - P_const - ε_VFPU × PERF_VFPU

    Args:
        results: Benchmark results for a single cache level.
        idle_power: Combined idle power.
        fpu_energy_pj_per_flop: FPU energy coefficient (pJ/FLOP) to subtract.
            If 0, no FPU subtraction is performed.

    Returns:
        Dictionary with 'total', 'control', and 'datapath' energy values,
        or None if extraction fails.
    """
    zeros_results = [r for r in results if r.workload.config.use_zeros]
    random_results = [r for r in results if not r.workload.config.use_zeros]

    control_energy = 0.0
    total_energy = 0.0

    # Fit control energy (zeros) - with FPU subtraction
    if zeros_results:
        fit = _fit_memory_energy_with_fpu_subtraction(
            zeros_results, idle_power, fpu_energy_pj_per_flop
        )
        if fit:
            control_energy = fit.slope

    # Fit total energy (random) - with FPU subtraction
    if random_results:
        fit = _fit_memory_energy_with_fpu_subtraction(
            random_results, idle_power, fpu_energy_pj_per_flop
        )
        if fit:
            total_energy = fit.slope

    if total_energy == 0.0 and control_energy == 0.0:
        return None

    return {
        "total": total_energy,
        "control": control_energy,
        "datapath": max(0.0, total_energy - control_energy),
    }


def _get_vector_fpu_energy(
    compute_results: List[BenchmarkResult],
    idle_power: float,
) -> float:
    """Extract vector FPU energy coefficient from high-AI compute results.

    Uses compute benchmark results (high arithmetic intensity) to extract
    the FPU energy per FLOP. This value is then used to subtract FPU
    contribution from cache benchmark measurements.

    Args:
        compute_results: Compute benchmark results with varying AI.
        idle_power: Combined idle power (CPU + GPU).

    Returns:
        FPU energy in pJ/FLOP, or 0.0 if extraction fails.
    """
    if not compute_results:
        return 0.0

    # Use high-AI results (>= 32) for best FPU energy estimation
    # High AI means compute-dominated, minimal memory influence
    high_ai_results = [
        r for r in compute_results
        if r.workload.config.params.get("arithmetic_intensity", 0) >= 32
    ]

    if not high_ai_results:
        # Fall back to all compute results
        high_ai_results = compute_results

    # Prefer non-zero (datapath) measurements for total energy
    random_results = [r for r in high_ai_results if not r.workload.config.use_zeros]

    if random_results:
        fit = _fit_compute_energy(random_results, idle_power)
        if fit and fit.slope > 0:
            return fit.slope

    # Fall back to zeros (control) if random not available
    zeros_results = [r for r in high_ai_results if r.workload.config.use_zeros]
    if zeros_results:
        fit = _fit_compute_energy(zeros_results, idle_power)
        if fit and fit.slope > 0:
            return fit.slope

    return 0.0


def _estimate_cache_workload_flops(result: BenchmarkResult) -> float:
    """Estimate the FLOP rate during a cache bandwidth benchmark.

    Cache workloads use `.add_(const)` which performs 1 FLOP per element.
    For N elements: read N + write N = 2N bytes, and N FLOPs.
    Arithmetic intensity = N FLOPs / 2N bytes = 0.5 FLOPs/byte.

    Given measured bandwidth (GB/s), FLOP rate = BW * 0.5 FLOP/byte.

    Args:
        result: Cache benchmark result with bandwidth measurement.

    Returns:
        Estimated FLOP rate in FLOP/s (not TFLOP/s).
    """
    # Get measured bandwidth in GB/s
    bandwidth_gb_s = result.workload.throughput

    # Cache workloads do 1 add per 2 bytes transferred (read + write)
    # FLOP/byte = 0.5 (since we read and write each element)
    flops_per_byte = 0.5

    # Convert GB/s to bytes/s, then to FLOPs/s
    flop_rate = bandwidth_gb_s * 1e9 * flops_per_byte

    return flop_rate


def _fit_memory_energy_with_fpu_subtraction(
    results: List[BenchmarkResult],
    idle_power: float,
    fpu_energy_pj_per_flop: float,
) -> Optional[ParameterFit]:
    """Fit memory energy with FPU contribution subtracted.

    Implements the paper's Equations 6-7:
        ε_cache × BW = P - P_const - ε_VFPU × PERF_VFPU

    For each result, we:
    1. Estimate the FLOP rate during the cache benchmark
    2. Calculate FPU power contribution: ε_VFPU × FLOP_rate
    3. Subtract FPU contribution from measured power
    4. Fit the adjusted power vs bandwidth

    Args:
        results: Memory benchmark results.
        idle_power: Idle power to subtract.
        fpu_energy_pj_per_flop: FPU energy coefficient in pJ/FLOP.

    Returns:
        ParameterFit with the memory energy coefficient, or None.
    """
    if len(results) < 2:
        return None

    bandwidths = []  # GB/s
    adjusted_powers = []  # W (with FPU contribution subtracted)

    for r in results:
        bw = r.workload.throughput  # GB/s
        power = r.energy.avg_total_power_watts

        if bw > 0 and power > 0:
            # Estimate FPU activity during cache benchmark
            flop_rate = _estimate_cache_workload_flops(r)  # FLOP/s

            # FPU power contribution: ε_VFPU (pJ/FLOP) × FLOP_rate (FLOP/s)
            # = pJ/s = 1e-12 W, so divide by 1e12 to get W
            fpu_power = (fpu_energy_pj_per_flop * flop_rate) / 1e12

            # Net power = measured - idle - FPU contribution
            net_power = power - idle_power - fpu_power
            net_power = max(0, net_power)  # Clamp to non-negative

            bandwidths.append(bw)
            adjusted_powers.append(net_power)

    if len(bandwidths) < 2:
        return None

    bandwidths = np.array(bandwidths)
    adjusted_powers = np.array(adjusted_powers)

    # Convert bandwidth to bits/s for pJ/bit calculation
    bits_per_second = bandwidths * 8e9

    try:
        # Linear fit: adjusted_power = ε_mem × bits_per_second
        denominator = np.sum(bits_per_second**2)
        if denominator == 0:
            return None
        slope = np.sum(bits_per_second * adjusted_powers) / denominator

        # Calculate R²
        y_pred = slope * bits_per_second
        ss_res = np.sum((adjusted_powers - y_pred) ** 2)
        ss_tot = np.sum((adjusted_powers - np.mean(adjusted_powers)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0.0, min(1.0, r_squared))

        # Convert to pJ/bit
        pj_per_bit = slope * 1e12

        return ParameterFit(
            slope=pj_per_bit,
            intercept=idle_power,
            r_squared=r_squared,
            n_samples=len(bandwidths),
        )
    except Exception:
        return None


def _fit_memory_energy(
    results: List[BenchmarkResult],
    idle_power: float,
) -> Optional[ParameterFit]:
    """Fit P = P_idle + ε_mem * BW to memory results.

    Returns energy in pJ/bit.
    """
    if len(results) < 2:
        return None

    # Extract bandwidth (GB/s) and power (W)
    bandwidths = []  # GB/s
    powers = []  # W

    for r in results:
        bw = r.workload.throughput  # GB/s
        power = r.energy.avg_total_power_watts
        if bw > 0 and power > 0:
            bandwidths.append(bw)
            powers.append(power)

    if len(bandwidths) < 2:
        return None

    bandwidths = np.array(bandwidths)
    powers = np.array(powers)

    # Subtract idle power
    net_powers = powers - idle_power
    net_powers = np.maximum(net_powers, 0)  # Clamp to non-negative

    # Convert bandwidth to bits/s for pJ/bit calculation
    # GB/s -> bits/s: multiply by 8e9
    bits_per_second = bandwidths * 8e9

    # Fit linear: net_power = ε_mem * bits_per_second
    # ε_mem in W/(bit/s) = J/bit
    # We want pJ/bit, so multiply by 1e12
    try:
        # Use least squares: y = m*x
        # m = sum(x*y) / sum(x*x)
        denominator = np.sum(bits_per_second**2)
        if denominator == 0:
            return None
        slope = np.sum(bits_per_second * net_powers) / denominator

        # Calculate R² and clamp to [0, 1] range
        # R² can be negative if the model is worse than a horizontal line,
        # but we clamp to 0 for reporting purposes
        y_pred = slope * bits_per_second
        ss_res = np.sum((net_powers - y_pred) ** 2)
        ss_tot = np.sum((net_powers - np.mean(net_powers)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0.0, min(1.0, r_squared))

        # Convert to pJ/bit
        pj_per_bit = slope * 1e12

        return ParameterFit(
            slope=pj_per_bit,
            intercept=idle_power,
            r_squared=r_squared,
            n_samples=len(bandwidths),
        )
    except Exception:
        return None


def _extract_compute_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract compute energy parameters by data type."""
    # Group by data type
    by_dtype: Dict[DataType, List[BenchmarkResult]] = {}
    for r in results:
        dtype = r.workload.config.data_type
        if dtype not in by_dtype:
            by_dtype[dtype] = []
        by_dtype[dtype].append(r)

    for dtype, dtype_results in by_dtype.items():
        zeros_results = [r for r in dtype_results if r.workload.config.use_zeros]
        random_results = [r for r in dtype_results if not r.workload.config.use_zeros]

        # Fit control energy
        if zeros_results:
            fit = _fit_compute_energy(zeros_results, idle_power)
            if fit:
                params.e_compute_control_pj_per_flop[dtype] = fit.slope

        # Fit total energy
        if random_results:
            fit = _fit_compute_energy(random_results, idle_power)
            if fit:
                params.e_compute_pj_per_flop[dtype] = fit.slope

        # Datapath = total - control
        total = params.e_compute_pj_per_flop.get(dtype, 0.0)
        control = params.e_compute_control_pj_per_flop.get(dtype, 0.0)
        params.e_compute_datapath_pj_per_flop[dtype] = total - control

    return params


def _extract_int32_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract INT32 integer compute energy parameters.

    Following the SC'25 paper methodology, INT32 energy is measured using
    integer multiply-add (IMAD) operations. This is important because
    ~10-20% of application power goes to integer operations for:
    - Loop counters and indices
    - Address calculations
    - Array indexing
    - Control flow
    """
    zeros_results = [r for r in results if r.workload.config.use_zeros]
    random_results = [r for r in results if not r.workload.config.use_zeros]

    # Fit control energy (zeros)
    if zeros_results:
        fit = _fit_compute_energy(zeros_results, idle_power)
        if fit:
            params.e_int32_control_pj_per_op = fit.slope

    # Fit total energy (random)
    if random_results:
        fit = _fit_compute_energy(random_results, idle_power)
        if fit:
            params.e_int32_pj_per_op = fit.slope

    # Datapath = total - control
    params.e_int32_datapath_pj_per_op = max(
        0.0, params.e_int32_pj_per_op - params.e_int32_control_pj_per_op
    )

    return params


def _fit_compute_energy(
    results: List[BenchmarkResult],
    idle_power: float,
) -> Optional[ParameterFit]:
    """Fit P = P_idle + ε_compute * FLOP_rate to compute results.

    Returns energy in pJ/FLOP.
    """
    if len(results) < 2:
        return None

    # Extract FLOP rate (TFLOP/s) and power (W)
    flop_rates = []  # TFLOP/s
    powers = []  # W

    for r in results:
        rate = r.workload.throughput  # TFLOP/s
        power = r.energy.avg_total_power_watts
        if rate > 0 and power > 0:
            flop_rates.append(rate)
            powers.append(power)

    if len(flop_rates) < 2:
        return None

    flop_rates = np.array(flop_rates)
    powers = np.array(powers)

    # Subtract idle power
    net_powers = powers - idle_power
    net_powers = np.maximum(net_powers, 0)

    # Convert TFLOP/s to FLOP/s: multiply by 1e12
    flops_per_second = flop_rates * 1e12

    try:
        # Linear fit: net_power = ε_compute * flops_per_second
        denominator = np.sum(flops_per_second**2)
        if denominator == 0:
            return None
        slope = np.sum(flops_per_second * net_powers) / denominator

        # Calculate R² and clamp to [0, 1] range
        # R² can be negative if the model is worse than a horizontal line,
        # but we clamp to 0 for reporting purposes
        y_pred = slope * flops_per_second
        ss_res = np.sum((net_powers - y_pred) ** 2)
        ss_tot = np.sum((net_powers - np.mean(net_powers)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0.0, min(1.0, r_squared))

        # Convert to pJ/FLOP
        pj_per_flop = slope * 1e12

        return ParameterFit(
            slope=pj_per_flop,
            intercept=idle_power,
            r_squared=r_squared,
            n_samples=len(flop_rates),
        )
    except Exception:
        return None


def _get_element_size(data_type: DataType) -> int:
    """Get element size in bytes for a data type."""
    sizes = {
        DataType.FP64: 8,
        DataType.FP32: 4,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.INT32: 4,
        DataType.INT8: 1,
    }
    return sizes.get(data_type, 4)


def _estimate_gemm_hbm_bandwidth(result: BenchmarkResult) -> float:
    """Estimate HBM bandwidth during a GEMM benchmark.

    For GEMM C = A × B with square matrices of size M×M:
    - Theoretical minimum HBM: read A (M² elements), read B (M² elements),
      write C (M² elements) = 3 × M² × element_size bytes per GEMM
    - However, cuBLAS/rocBLAS tile data into L1/L2 caches, so actual HBM
      traffic is much lower due to cache reuse

    We use the theoretical minimum as a conservative upper bound.
    Real HBM bandwidth is likely 10-50% of this due to tiling.

    Args:
        result: GEMM benchmark result with matrix_size in params.

    Returns:
        Estimated HBM bandwidth in GB/s.
    """
    matrix_size = result.workload.config.params.get("matrix_size", 4096)
    element_size = _get_element_size(result.workload.config.data_type)
    duration = result.workload.duration_seconds

    if duration <= 0:
        return 0.0

    # Theoretical minimum HBM: read A, read B, write C = 3 * M^2 * elem_size
    hbm_bytes_per_gemm = 3 * matrix_size * matrix_size * element_size

    # Calculate number of GEMM operations performed
    # FLOPs per GEMM = 2 * M^3 (multiply-accumulate)
    flops_per_gemm = 2 * matrix_size * matrix_size * matrix_size
    total_flops = result.workload.flops_executed or 0

    if flops_per_gemm <= 0:
        return 0.0

    iterations = total_flops / flops_per_gemm
    total_hbm_bytes = hbm_bytes_per_gemm * iterations
    hbm_bandwidth_gb_s = total_hbm_bytes / duration / 1e9

    return hbm_bandwidth_gb_s


def _fit_gemm_energy_with_hbm_subtraction(
    results: List[BenchmarkResult],
    idle_power: float,
    e_hbm_pj_per_bit: float,
) -> Optional[ParameterFit]:
    """Fit GEMM compute energy with HBM contribution subtracted.

    Implements the paper's Equation 9 (conservative HBM-only version):
        ε_MFPU × PERF = P - P_const - ε_HBM × BW_HBM

    Note: Full Eq. 9 also subtracts L1/L2 contributions, but that requires
    profiler integration (NSight/rocprof) to measure actual L1/L2 bandwidth.
    Our values will be ~10-30% higher than the paper's Table 3 as a result.

    Args:
        results: GEMM benchmark results.
        idle_power: Idle power to subtract.
        e_hbm_pj_per_bit: HBM energy coefficient in pJ/bit.

    Returns:
        ParameterFit with the GEMM compute energy coefficient, or None.
    """
    if len(results) < 1:
        return None

    flop_rates = []  # TFLOP/s
    adjusted_powers = []  # W (with HBM contribution subtracted)

    for r in results:
        rate = r.workload.throughput  # TFLOP/s
        power = r.energy.avg_total_power_watts

        if rate > 0 and power > 0:
            # Estimate HBM bandwidth during this GEMM
            hbm_bw_gb_s = _estimate_gemm_hbm_bandwidth(r)

            # HBM power contribution: ε_HBM (pJ/bit) × BW (GB/s) × 8e9 (bits/GB) / 1e12
            hbm_bits_per_second = hbm_bw_gb_s * 8e9
            hbm_power = (e_hbm_pj_per_bit * hbm_bits_per_second) / 1e12

            # Net power = measured - idle - HBM contribution
            net_power = power - idle_power - hbm_power
            net_power = max(0, net_power)

            flop_rates.append(rate)
            adjusted_powers.append(net_power)

    if len(flop_rates) < 1:
        return None

    flop_rates = np.array(flop_rates)
    adjusted_powers = np.array(adjusted_powers)

    # Convert TFLOP/s to FLOP/s
    flops_per_second = flop_rates * 1e12

    try:
        # Linear fit: adjusted_power = ε_compute × flops_per_second
        denominator = np.sum(flops_per_second**2)
        if denominator == 0:
            return None
        slope = np.sum(flops_per_second * adjusted_powers) / denominator

        # Calculate R²
        if len(flop_rates) >= 2:
            y_pred = slope * flops_per_second
            ss_res = np.sum((adjusted_powers - y_pred) ** 2)
            ss_tot = np.sum((adjusted_powers - np.mean(adjusted_powers)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r_squared = max(0.0, min(1.0, r_squared))
        else:
            r_squared = 1.0  # Single point, perfect fit by definition

        # Convert to pJ/FLOP
        pj_per_flop = slope * 1e12

        return ParameterFit(
            slope=pj_per_flop,
            intercept=idle_power,
            r_squared=r_squared,
            n_samples=len(flop_rates),
        )
    except Exception:
        return None


def _extract_gemm_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract GEMM/matrix energy parameters by data type.

    Following the SC'25 paper methodology (Eq. 9), GEMM energy should subtract
    memory hierarchy contributions:
        ε_MFPU × PERF_MFPU = P - P_const - ε_L1×BW_L1 - ε_L2×BW_L2 - ε_HBM×BW_HBM

    Implementation notes:
    - We subtract estimated HBM bandwidth contribution (conservative approach)
    - L1/L2 contributions are NOT subtracted (requires profiler integration)
    - Our values will be ~10-30% higher than paper's Table 3 as a result
    - For accurate L1/L2 subtraction, integrate NSight Compute (NVIDIA) or
      rocprof (AMD) to measure actual cache bandwidths during GEMM

    Quote from paper: "Unlike Mixbench, matrix-matrix multiplications exhibit
    high degrees of locality at each level of the cache. As a result, BW_L1,
    BW_L2, and BW_HBM will be tapered."
    """
    # Get HBM energy coefficient for subtraction (if available)
    e_hbm_pj_per_bit = getattr(params, "e_hbm_pj_per_bit", 0.0)

    # Group by data type
    by_dtype: Dict[DataType, List[BenchmarkResult]] = {}
    for r in results:
        dtype = r.workload.config.data_type
        if dtype not in by_dtype:
            by_dtype[dtype] = []
        by_dtype[dtype].append(r)

    for dtype, dtype_results in by_dtype.items():
        zeros_results = [r for r in dtype_results if r.workload.config.use_zeros]
        random_results = [r for r in dtype_results if not r.workload.config.use_zeros]

        # Extract with HBM subtraction if we have HBM energy coefficient
        if e_hbm_pj_per_bit > 0:
            # Use HBM-corrected fitting
            if zeros_results:
                fit = _fit_gemm_energy_with_hbm_subtraction(
                    zeros_results, idle_power, e_hbm_pj_per_bit
                )
                if fit:
                    params.e_gemm_control_pj_per_flop[dtype] = fit.slope
                else:
                    # Fallback to standard fitting
                    fit = _fit_compute_energy(zeros_results, idle_power)
                    if fit:
                        params.e_gemm_control_pj_per_flop[dtype] = fit.slope

            if random_results:
                fit = _fit_gemm_energy_with_hbm_subtraction(
                    random_results, idle_power, e_hbm_pj_per_bit
                )
                if fit:
                    params.e_gemm_pj_per_flop[dtype] = fit.slope
                else:
                    # Fallback to standard fitting
                    fit = _fit_compute_energy(random_results, idle_power)
                    if fit:
                        params.e_gemm_pj_per_flop[dtype] = fit.slope
        else:
            # No HBM energy available - use standard fitting
            # This is the fallback for unified memory platforms (macOS)
            if zeros_results:
                fit = _fit_compute_energy(zeros_results, idle_power)
                if fit:
                    params.e_gemm_control_pj_per_flop[dtype] = fit.slope
                else:
                    pj_per_flop = _average_energy_per_flop(zeros_results, idle_power)
                    if pj_per_flop is not None:
                        params.e_gemm_control_pj_per_flop[dtype] = pj_per_flop

            if random_results:
                fit = _fit_compute_energy(random_results, idle_power)
                if fit:
                    params.e_gemm_pj_per_flop[dtype] = fit.slope
                else:
                    pj_per_flop = _average_energy_per_flop(random_results, idle_power)
                    if pj_per_flop is not None:
                        params.e_gemm_pj_per_flop[dtype] = pj_per_flop

        # Datapath = total - control
        total = params.e_gemm_pj_per_flop.get(dtype, 0.0)
        control = params.e_gemm_control_pj_per_flop.get(dtype, 0.0)
        params.e_gemm_datapath_pj_per_flop[dtype] = max(0.0, total - control)

    return params


def _average_energy_per_flop(
    results: List[BenchmarkResult],
    idle_power: float,
) -> Optional[float]:
    """Calculate average energy per FLOP from results."""
    energies = []

    for r in results:
        if r.workload.flops_executed and r.workload.flops_executed > 0:
            # Net energy = total - idle
            net_energy = r.energy.total_energy_joules - idle_power * r.workload.duration_seconds
            net_energy = max(0, net_energy)

            # pJ per FLOP
            pj_per_flop = (net_energy / r.workload.flops_executed) * 1e12
            energies.append(pj_per_flop)

    return np.mean(energies) if energies else None


def predict_power(
    params: EnergyParameters,
    memory_bandwidth_gb_s: float = 0.0,
    compute_tflops: float = 0.0,
    data_type: DataType = DataType.FP32,
    tdp_limit: bool = True,
    # Cache-level bandwidth parameters (for detailed prediction)
    l1_bandwidth_gb_s: Optional[float] = None,
    l2_bandwidth_gb_s: Optional[float] = None,
    hbm_bandwidth_gb_s: Optional[float] = None,
) -> float:
    """Predict power consumption from energy parameters.

    Uses the model:
    - Simple: P = min(TDP, P_idle + ε_mem*BW + ε_compute*FLOP)
    - Detailed: P = min(TDP, P_idle + ε_L1*BW_L1 + ε_L2*BW_L2 + ε_HBM*BW_HBM + ε_compute*FLOP)

    The detailed model is used when cache-level bandwidths are provided AND
    the params have cache-level energy values extracted.

    Args:
        params: Extracted energy parameters.
        memory_bandwidth_gb_s: Total memory bandwidth in GB/s (for simple model).
        compute_tflops: Compute rate in TFLOP/s.
        data_type: Data type for compute coefficient.
        tdp_limit: Whether to cap at TDP.
        l1_bandwidth_gb_s: L1 cache bandwidth in GB/s (for detailed model).
        l2_bandwidth_gb_s: L2 cache bandwidth in GB/s (for detailed model).
        hbm_bandwidth_gb_s: HBM bandwidth in GB/s (for detailed model).

    Returns:
        Predicted power in watts.
    """
    idle_power = params.p_idle_cpu_watts + params.p_idle_gpu_watts

    # Check if we should use cache-level prediction
    use_cache_levels = (
        l1_bandwidth_gb_s is not None
        or l2_bandwidth_gb_s is not None
        or hbm_bandwidth_gb_s is not None
    ) and _has_cache_level_energy(params)

    if use_cache_levels:
        memory_power = _predict_cache_level_power(
            params,
            l1_bandwidth_gb_s or 0.0,
            l2_bandwidth_gb_s or 0.0,
            hbm_bandwidth_gb_s or 0.0,
        )
    else:
        # Simple model: single memory bandwidth
        # Convert GB/s to bits/s, then multiply by pJ/bit, convert to W
        bits_per_second = memory_bandwidth_gb_s * 8e9
        memory_power = (params.e_memory_pj_per_bit * bits_per_second) / 1e12

    # Compute contribution
    # Convert TFLOP/s to FLOP/s, multiply by pJ/FLOP, convert to W
    flops_per_second = compute_tflops * 1e12
    e_compute = params.e_compute_pj_per_flop.get(data_type, 0.0)
    compute_power = (e_compute * flops_per_second) / 1e12

    total_power = idle_power + memory_power + compute_power

    if tdp_limit and params.tdp_watts:
        total_power = min(total_power, params.tdp_watts)

    return total_power


def _has_cache_level_energy(params: EnergyParameters) -> bool:
    """Check if EnergyParameters has cache-level energy values.

    Returns True if any of L1, L2, or HBM energy values are non-zero.
    """
    return (
        getattr(params, "e_l1_pj_per_bit", 0.0) > 0
        or getattr(params, "e_l2_pj_per_bit", 0.0) > 0
        or getattr(params, "e_hbm_pj_per_bit", 0.0) > 0
    )


def _predict_cache_level_power(
    params: EnergyParameters,
    l1_bandwidth_gb_s: float,
    l2_bandwidth_gb_s: float,
    hbm_bandwidth_gb_s: float,
) -> float:
    """Predict memory power using cache-level energy parameters.

    Implements: P_mem = ε_L1*BW_L1 + ε_L2*BW_L2 + ε_HBM*BW_HBM

    Args:
        params: EnergyParameters with cache-level values.
        l1_bandwidth_gb_s: L1 cache bandwidth in GB/s.
        l2_bandwidth_gb_s: L2 cache bandwidth in GB/s.
        hbm_bandwidth_gb_s: HBM bandwidth in GB/s.

    Returns:
        Total memory power contribution in watts.
    """
    # Get cache-level energy values (with fallback to 0)
    e_l1 = getattr(params, "e_l1_pj_per_bit", 0.0)
    e_l2 = getattr(params, "e_l2_pj_per_bit", 0.0)
    e_hbm = getattr(params, "e_hbm_pj_per_bit", 0.0)

    # Convert GB/s to bits/s (multiply by 8e9)
    l1_bits_per_second = l1_bandwidth_gb_s * 8e9
    l2_bits_per_second = l2_bandwidth_gb_s * 8e9
    hbm_bits_per_second = hbm_bandwidth_gb_s * 8e9

    # Calculate power for each level (pJ/bit * bits/s = pW, divide by 1e12 for W)
    l1_power = (e_l1 * l1_bits_per_second) / 1e12
    l2_power = (e_l2 * l2_bits_per_second) / 1e12
    hbm_power = (e_hbm * hbm_bits_per_second) / 1e12

    return l1_power + l2_power + hbm_power


def predict_power_detailed(
    params: EnergyParameters,
    l1_bandwidth_gb_s: float = 0.0,
    l2_bandwidth_gb_s: float = 0.0,
    hbm_bandwidth_gb_s: float = 0.0,
    compute_tflops: float = 0.0,
    data_type: DataType = DataType.FP32,
    tdp_limit: bool = True,
) -> Dict[str, float]:
    """Predict power with detailed breakdown by component.

    Returns a dictionary with power contributions from each cache level,
    compute, and idle power. Useful for understanding where power goes.

    Args:
        params: Extracted energy parameters with cache-level data.
        l1_bandwidth_gb_s: L1 cache bandwidth in GB/s.
        l2_bandwidth_gb_s: L2 cache bandwidth in GB/s.
        hbm_bandwidth_gb_s: HBM bandwidth in GB/s.
        compute_tflops: Compute rate in TFLOP/s.
        data_type: Data type for compute coefficient.
        tdp_limit: Whether to cap at TDP.

    Returns:
        Dictionary with power breakdown:
        - idle_watts: Idle power
        - l1_watts: L1 cache power contribution
        - l2_watts: L2 cache power contribution
        - hbm_watts: HBM power contribution
        - compute_watts: Compute power contribution
        - total_watts: Total predicted power
        - tdp_limited: Whether TDP limit was applied
    """
    idle_power = params.p_idle_cpu_watts + params.p_idle_gpu_watts

    # Get cache-level energy values (with fallback to 0)
    e_l1 = getattr(params, "e_l1_pj_per_bit", 0.0)
    e_l2 = getattr(params, "e_l2_pj_per_bit", 0.0)
    e_hbm = getattr(params, "e_hbm_pj_per_bit", 0.0)

    # Calculate per-level power
    l1_power = (e_l1 * l1_bandwidth_gb_s * 8e9) / 1e12
    l2_power = (e_l2 * l2_bandwidth_gb_s * 8e9) / 1e12
    hbm_power = (e_hbm * hbm_bandwidth_gb_s * 8e9) / 1e12

    # Compute power
    flops_per_second = compute_tflops * 1e12
    e_compute = params.e_compute_pj_per_flop.get(data_type, 0.0)
    compute_power = (e_compute * flops_per_second) / 1e12

    # Total
    total_power = idle_power + l1_power + l2_power + hbm_power + compute_power
    tdp_limited = False

    if tdp_limit and params.tdp_watts and total_power > params.tdp_watts:
        total_power = params.tdp_watts
        tdp_limited = True

    return {
        "idle_watts": idle_power,
        "l1_watts": l1_power,
        "l2_watts": l2_power,
        "hbm_watts": hbm_power,
        "memory_watts": l1_power + l2_power + hbm_power,
        "compute_watts": compute_power,
        "total_watts": total_power,
        "tdp_limited": tdp_limited,
    }


def summarize_results(results: List[BenchmarkResult]) -> Dict:
    """Create a summary of benchmark results.

    Args:
        results: List of benchmark results.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "total_runs": len(results),
        "total_duration_seconds": sum(r.workload.duration_seconds for r in results),
        "total_energy_joules": sum(r.energy.total_energy_joules for r in results),
        "by_workload_type": {},
    }

    for wtype in WorkloadType:
        type_results = [r for r in results if r.workload.workload_type == wtype]
        if type_results:
            summary["by_workload_type"][wtype.value] = {
                "count": len(type_results),
                "avg_throughput": np.mean([r.workload.throughput for r in type_results]),
                "throughput_unit": type_results[0].workload.throughput_unit,
                "avg_power_watts": np.mean(
                    [r.energy.avg_total_power_watts for r in type_results]
                ),
                "total_energy_joules": sum(
                    [r.energy.total_energy_joules for r in type_results]
                ),
            }

    return summary


# Reference values from SC'25 paper Table 3 (Antepara et al.)
# Used for sanity-checking extracted parameters
PAPER_REFERENCE_VALUES = {
    "A100": {
        # Memory hierarchy (pJ/bit)
        "HBM": {"total": 13.11, "control": 8.47, "datapath": 4.64},
        "L2": {"total": 4.71, "control": 3.11, "datapath": 1.60},
        "L1": {"total": 1.59, "control": 1.26, "datapath": 0.33},
        # Vector FPU (pJ/FLOP) - uses CUDA cores
        "V-FP64": {"total": 28.50, "control": 12.92, "datapath": 15.58},
        "V-FP32": {"total": 7.42, "control": 3.60, "datapath": 3.82},
        # Matrix FPU / Tensor Cores (pJ/FLOP)
        "M-FP64": {"total": 13.35, "control": 6.13, "datapath": 7.22},
        "M-FP32": {"total": 1.24, "control": 1.13, "datapath": 0.11},  # TF32
        "M-FP16": {"total": 0.70, "control": 0.37, "datapath": 0.33},
        # INT32 (pJ/Op)
        "INT32": {"total": 3.25, "control": 2.63, "datapath": 0.62},
    },
    "MI250X": {
        # Memory hierarchy (pJ/bit)
        "HBM": {"total": 16.85, "control": 13.54, "datapath": 3.31},
        "L2": {"total": 3.23, "control": 2.52, "datapath": 0.71},
        "L1": {"total": 1.82, "control": 1.34, "datapath": 0.48},
        # Vector FPU (pJ/FLOP)
        "V-FP64": {"total": 28.04, "control": 16.75, "datapath": 11.29},
        "V-FP32": {"total": 5.23, "control": 3.03, "datapath": 2.20},
        # Matrix FPU (pJ/FLOP)
        "M-FP64": {"total": 16.00, "control": 9.94, "datapath": 6.06},
        "M-FP32": {"total": 3.23, "control": 1.93, "datapath": 1.30},
        "M-FP16": {"total": 2.45, "control": 1.78, "datapath": 0.67},
        # INT32 (pJ/Op)
        "INT32": {"total": 4.38, "control": 2.99, "datapath": 1.39},
    },
}


def validate_against_paper(
    params: EnergyParameters,
    tolerance_percent: float = 50.0,
) -> Dict[str, Dict[str, float]]:
    """Compare extracted parameters against SC'25 paper's Table 3.

    This function provides a sanity check by comparing extracted energy
    parameters against reference values from the paper. Large deviations
    may indicate measurement issues or methodology differences.

    Args:
        params: Extracted EnergyParameters to validate.
        tolerance_percent: Acceptable deviation threshold (default 50%).
            Values outside this range are flagged as warnings.

    Returns:
        Dictionary with comparison results:
        {
            "parameter_name": {
                "extracted": float,
                "reference": float,
                "difference_percent": float,
                "within_tolerance": bool,
            },
            ...
        }

    Notes:
        - Only validates A100 and MI250X GPUs (reference data from paper)
        - GEMM values will be ~10-30% higher than paper (no L1/L2 subtraction)
        - Returns empty dict if hardware not in reference table
    """
    # Determine which reference to use based on hardware name
    hw_name = params.hardware_name.upper()
    ref_key = None

    if "A100" in hw_name:
        ref_key = "A100"
    elif "MI250" in hw_name:
        ref_key = "MI250X"

    if ref_key is None:
        return {}

    ref = PAPER_REFERENCE_VALUES[ref_key]
    results = {}

    def _compare(name: str, extracted: float, reference: float) -> Dict[str, float]:
        """Helper to compute comparison metrics."""
        if reference == 0:
            diff_pct = 0.0 if extracted == 0 else float('inf')
        else:
            diff_pct = ((extracted - reference) / reference) * 100
        return {
            "extracted": extracted,
            "reference": reference,
            "difference_percent": diff_pct,
            "within_tolerance": abs(diff_pct) <= tolerance_percent,
        }

    # Compare cache-level memory energy
    if params.e_l1_pj_per_bit > 0:
        results["L1_total"] = _compare(
            "L1_total", params.e_l1_pj_per_bit, ref["L1"]["total"]
        )
    if params.e_l2_pj_per_bit > 0:
        results["L2_total"] = _compare(
            "L2_total", params.e_l2_pj_per_bit, ref["L2"]["total"]
        )
    if params.e_hbm_pj_per_bit > 0:
        results["HBM_total"] = _compare(
            "HBM_total", params.e_hbm_pj_per_bit, ref["HBM"]["total"]
        )

    # Compare vector FPU energy
    if DataType.FP64 in params.e_compute_pj_per_flop:
        results["V-FP64_total"] = _compare(
            "V-FP64_total",
            params.e_compute_pj_per_flop[DataType.FP64],
            ref["V-FP64"]["total"],
        )
    if DataType.FP32 in params.e_compute_pj_per_flop:
        results["V-FP32_total"] = _compare(
            "V-FP32_total",
            params.e_compute_pj_per_flop[DataType.FP32],
            ref["V-FP32"]["total"],
        )

    # Compare GEMM/Matrix FPU energy
    # Note: Our values expected to be 10-30% higher (no L1/L2 subtraction)
    if DataType.FP64 in params.e_gemm_pj_per_flop:
        results["M-FP64_total"] = _compare(
            "M-FP64_total",
            params.e_gemm_pj_per_flop[DataType.FP64],
            ref["M-FP64"]["total"],
        )
    if DataType.FP32 in params.e_gemm_pj_per_flop:
        results["M-FP32_total"] = _compare(
            "M-FP32_total",
            params.e_gemm_pj_per_flop[DataType.FP32],
            ref["M-FP32"]["total"],
        )
    if DataType.FP16 in params.e_gemm_pj_per_flop:
        results["M-FP16_total"] = _compare(
            "M-FP16_total",
            params.e_gemm_pj_per_flop[DataType.FP16],
            ref["M-FP16"]["total"],
        )

    # Compare INT32 energy
    if params.e_int32_pj_per_op > 0:
        results["INT32_total"] = _compare(
            "INT32_total", params.e_int32_pj_per_op, ref["INT32"]["total"]
        )

    return results


def _extract_inference_gemm_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract inference GEMM energy parameters by mode (prefill/decode) and dtype.

    Separates results by mode and data type, then fits ε via compute energy fitting.
    """
    # Group by mode
    prefill_results = [
        r for r in results if r.workload.config.params.get("mode") == "prefill"
    ]
    decode_results = [
        r for r in results if r.workload.config.params.get("mode") == "decode"
    ]

    # Extract prefill params by dtype
    if prefill_results:
        by_dtype: Dict[DataType, List[BenchmarkResult]] = {}
        for r in prefill_results:
            dt = r.workload.config.data_type
            if dt not in by_dtype:
                by_dtype[dt] = []
            by_dtype[dt].append(r)
        for dt, dt_results in by_dtype.items():
            fit = _fit_compute_energy(dt_results, idle_power)
            if fit:
                params.e_inference_gemm_prefill_pj_per_flop[dt] = fit.slope
            elif dt_results:
                avg = _average_energy_per_flop(dt_results, idle_power)
                if avg is not None:
                    params.e_inference_gemm_prefill_pj_per_flop[dt] = avg

    # Extract decode params by dtype
    if decode_results:
        by_dtype = {}
        for r in decode_results:
            dt = r.workload.config.data_type
            if dt not in by_dtype:
                by_dtype[dt] = []
            by_dtype[dt].append(r)
        for dt, dt_results in by_dtype.items():
            fit = _fit_compute_energy(dt_results, idle_power)
            if fit:
                params.e_inference_gemm_decode_pj_per_flop[dt] = fit.slope
            elif dt_results:
                avg = _average_energy_per_flop(dt_results, idle_power)
                if avg is not None:
                    params.e_inference_gemm_decode_pj_per_flop[dt] = avg

    return params


def _extract_attention_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract attention energy parameter (pJ/FLOP) via compute energy fitting."""
    fit = _fit_compute_energy(results, idle_power)
    if fit:
        params.e_attention_pj_per_flop = fit.slope
    elif results:
        avg = _average_energy_per_flop(results, idle_power)
        if avg is not None:
            params.e_attention_pj_per_flop = avg
    return params


def _extract_kv_cache_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract KV cache energy parameters (pJ/bit) for read and write modes."""
    read_results = [
        r for r in results if r.workload.config.params.get("mode") == "read"
    ]
    write_results = [
        r for r in results if r.workload.config.params.get("mode") == "write"
    ]

    if read_results:
        fit = _fit_memory_energy(read_results, idle_power)
        if fit:
            params.e_kv_read_pj_per_bit = fit.slope

    if write_results:
        fit = _fit_memory_energy(write_results, idle_power)
        if fit:
            params.e_kv_write_pj_per_bit = fit.slope

    return params


def _extract_nccl_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract communication energy parameter (pJ/bit) via memory energy fitting."""
    if results:
        fit = _fit_memory_energy(results, idle_power)
        if fit:
            params.e_comm_pj_per_bit = fit.slope
    return params


def _extract_batched_decode_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract batched decode exponent β via log-log regression.

    Fits log(E_per_token) = log(k) + β*log(B) using np.polyfit.
    The exponent β describes how per-token decode energy scales with batch size.
    β = 1 means linear (no batching benefit); β < 1 means sub-linear (batching helps).
    """
    if len(results) < 2:
        return params

    batch_sizes = []
    energies_per_token = []

    for r in results:
        bs = r.workload.config.params.get("batch_size", 1)
        if bs <= 0:
            continue

        # Energy per token = total energy / (batch_size * iterations)
        total_energy = r.energy.total_energy_joules - idle_power * r.workload.duration_seconds
        total_energy = max(0.0, total_energy)

        # throughput is tokens/s, so total tokens = throughput * duration
        tokens = r.workload.throughput * r.workload.duration_seconds
        if tokens > 0:
            e_per_token = total_energy / tokens
            if e_per_token > 0:
                batch_sizes.append(bs)
                energies_per_token.append(e_per_token)

    if len(batch_sizes) < 2:
        return params

    try:
        log_bs = np.log(np.array(batch_sizes, dtype=np.float64))
        log_e = np.log(np.array(energies_per_token, dtype=np.float64))

        # log(E) = log(k) + β*log(B)
        coeffs = np.polyfit(log_bs, log_e, 1)
        beta = coeffs[0]

        # Clamp to reasonable range [0, 2]
        params.e_decode_batch_exponent = max(0.0, min(2.0, beta))
    except Exception:
        pass

    return params


def _group_by_dtype(
    results: List[BenchmarkResult],
) -> Dict[DataType, List[BenchmarkResult]]:
    """Group benchmark results by data type."""
    by_dtype: Dict[DataType, List[BenchmarkResult]] = {}
    for r in results:
        dt = r.workload.config.data_type
        if dt not in by_dtype:
            by_dtype[dt] = []
        by_dtype[dt].append(r)
    return by_dtype


def _extract_inference_gemm_raw_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract raw (CUDA graph) inference GEMM energy parameters."""
    prefill_results = [
        r for r in results if r.workload.config.params.get("mode") == "prefill"
    ]
    decode_results = [
        r for r in results if r.workload.config.params.get("mode") == "decode"
    ]

    if prefill_results:
        for dt, dt_results in _group_by_dtype(prefill_results).items():
            fit = _fit_compute_energy(dt_results, idle_power)
            if fit:
                params.e_inference_gemm_prefill_raw_pj_per_flop[dt] = fit.slope
            elif dt_results:
                avg = _average_energy_per_flop(dt_results, idle_power)
                if avg is not None:
                    params.e_inference_gemm_prefill_raw_pj_per_flop[dt] = avg

    if decode_results:
        for dt, dt_results in _group_by_dtype(decode_results).items():
            fit = _fit_compute_energy(dt_results, idle_power)
            if fit:
                params.e_inference_gemm_decode_raw_pj_per_flop[dt] = fit.slope
            elif dt_results:
                avg = _average_energy_per_flop(dt_results, idle_power)
                if avg is not None:
                    params.e_inference_gemm_decode_raw_pj_per_flop[dt] = avg

    return params


def _extract_attention_raw_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract raw (CUDA graph) attention energy parameter."""
    fit = _fit_compute_energy(results, idle_power)
    if fit:
        params.e_attention_raw_pj_per_flop = fit.slope
    elif results:
        avg = _average_energy_per_flop(results, idle_power)
        if avg is not None:
            params.e_attention_raw_pj_per_flop = avg
    return params


def _extract_kv_cache_raw_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract raw (CUDA graph) KV cache energy parameters."""
    read_results = [
        r for r in results if r.workload.config.params.get("mode") == "read"
    ]
    write_results = [
        r for r in results if r.workload.config.params.get("mode") == "write"
    ]

    if read_results:
        fit = _fit_memory_energy(read_results, idle_power)
        if fit:
            params.e_kv_read_raw_pj_per_bit = fit.slope

    if write_results:
        fit = _fit_memory_energy(write_results, idle_power)
        if fit:
            params.e_kv_write_raw_pj_per_bit = fit.slope

    return params


def _extract_batched_decode_raw_params(
    params: EnergyParameters,
    results: List[BenchmarkResult],
    idle_power: float,
) -> EnergyParameters:
    """Extract raw (CUDA graph) batched decode exponent β."""
    if len(results) < 2:
        return params

    batch_sizes = []
    energies_per_token = []

    for r in results:
        bs = r.workload.config.params.get("batch_size", 1)
        if bs <= 0:
            continue

        total_energy = r.energy.total_energy_joules - idle_power * r.workload.duration_seconds
        total_energy = max(0.0, total_energy)

        tokens = r.workload.throughput * r.workload.duration_seconds
        if tokens > 0:
            e_per_token = total_energy / tokens
            if e_per_token > 0:
                batch_sizes.append(bs)
                energies_per_token.append(e_per_token)

    if len(batch_sizes) < 2:
        return params

    try:
        log_bs = np.log(np.array(batch_sizes, dtype=np.float64))
        log_e = np.log(np.array(energies_per_token, dtype=np.float64))
        coeffs = np.polyfit(log_bs, log_e, 1)
        beta = coeffs[0]
        params.e_decode_batch_raw_exponent = max(0.0, min(2.0, beta))
    except Exception:
        pass

    return params


def format_validation_report(validation_results: Dict[str, Dict[str, float]]) -> str:
    """Format validation results as a human-readable report.

    Args:
        validation_results: Output from validate_against_paper().

    Returns:
        Formatted string report suitable for logging or display.
    """
    if not validation_results:
        return "No validation performed (hardware not in reference table)."

    lines = ["SC'25 Paper Validation Results", "=" * 50]

    # Count warnings
    warnings = sum(
        1 for r in validation_results.values() if not r["within_tolerance"]
    )

    for name, data in sorted(validation_results.items()):
        status = "OK" if data["within_tolerance"] else "WARN"
        lines.append(
            f"{name:15s}: {data['extracted']:8.2f} vs {data['reference']:8.2f} "
            f"({data['difference_percent']:+6.1f}%) [{status}]"
        )

    lines.append("-" * 50)
    if warnings == 0:
        lines.append("All parameters within tolerance.")
    else:
        lines.append(
            f"{warnings} parameter(s) outside tolerance. "
            "This may be expected due to methodology differences "
            "(e.g., no L1/L2 subtraction for GEMM)."
        )

    return "\n".join(lines)


__all__ = [
    "CacheLevel",
    "extract_parameters",
    "predict_power",
    "predict_power_detailed",
    "summarize_results",
    "validate_against_paper",
    "format_validation_report",
    "ParameterFit",
    "PAPER_REFERENCE_VALUES",
]
