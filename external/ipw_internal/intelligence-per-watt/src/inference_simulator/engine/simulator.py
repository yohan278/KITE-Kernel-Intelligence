"""Event-driven inference simulator engine."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from inference_simulator.engine.event import Event, EventQueue, EventType
from inference_simulator.engine.tool_sampler import ToolLatencySampler
from inference_simulator.metrics.collector import MetricsCollector, SimulationMetrics
from inference_simulator.types.operators import OperatorCategory
from inference_simulator.request.kv_cache import KVCacheManager
from inference_simulator.request.request import Batch, Request, RequestState
from inference_simulator.scheduler.base import BaseScheduler
from inference_simulator.types.hardware_spec import HardwareSpec
from inference_simulator.types.inference_spec import InferenceSpec
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.workload_spec import WorkloadSpec
from inference_simulator.workload.generator import WorkloadGenerator

logger = logging.getLogger(__name__)


def _get_timing_functions(
    model_spec: ModelSpec,
    hardware_spec: HardwareSpec,
    inference_spec: InferenceSpec,
    estimator: object | None,
) -> tuple[Callable[..., int], Callable[..., int]]:
    """Resolve prefill/decode timing functions.

    Tries the runtime estimator first, then falls back to the roofline model.
    Returns callables that return time in nanoseconds.
    """
    # Try the estimator if provided and fitted
    if estimator is not None and hasattr(estimator, "is_fitted") and estimator.is_fitted():
        def prefill_time_ns(batch_size: int, seq_len: int) -> int:
            result = estimator.estimate_prefill(batch_size, seq_len)
            return int(result.time_s * 1e9)

        def decode_step_time_ns(batch_size: int, kv_cache_len: int) -> int:
            result = estimator.estimate_decode_step(batch_size, kv_cache_len)
            return int(result.time_s * 1e9)

        return prefill_time_ns, decode_step_time_ns

    # Fallback: roofline model from ipw.simulator
    from ipw.simulator.inference_model import (
        DEFAULT_ALPHA,
        DEFAULT_ETA_DECODE,
        DEFAULT_ETA_PREFILL,
        estimate_decode,
        estimate_prefill,
    )

    active_params_b = model_spec.total_params_billion
    num_gpus = inference_spec.num_gpus
    peak_tflops = hardware_spec.peak_tflops * num_gpus
    mem_bw_gb_s = hardware_spec.hbm_bandwidth_gb_s * num_gpus
    power_watts = hardware_spec.tdp_watts * num_gpus * DEFAULT_ALPHA
    bytes_per_param = hardware_spec.bytes_per_param

    def prefill_time_ns(batch_size: int, seq_len: int) -> int:
        result = estimate_prefill(
            active_params_b=active_params_b,
            input_tokens=batch_size * seq_len,
            peak_tflops=peak_tflops,
            eta=DEFAULT_ETA_PREFILL,
            power_watts=power_watts,
        )
        return int(result.time_seconds * 1e9)

    def decode_step_time_ns(batch_size: int, kv_cache_len: int) -> int:
        result = estimate_decode(
            active_params_b=active_params_b,
            output_tokens=batch_size,  # 1 token per request in batch
            bytes_per_param=bytes_per_param,
            mem_bw_gb_s=mem_bw_gb_s,
            eta=DEFAULT_ETA_DECODE,
            power_watts=power_watts,
        )
        return int(result.time_seconds * 1e9)

    return prefill_time_ns, decode_step_time_ns


class EventDrivenSimulator:
    """Discrete-event simulator for LLM inference serving.

    Models the lifecycle of requests through an inference engine:
    arrival -> scheduling -> prefill -> decode steps -> completion.

    For multi-step requests (agentic, RAG, coding), supports:
    prefill -> decode -> tool execution -> prefill (next step) -> ... -> completion.

    Uses pluggable schedulers (vLLM, ORCA) and timing estimators
    (roofline fallback, learned models, or LUT lookup).
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        hardware_spec: HardwareSpec,
        inference_spec: InferenceSpec,
        scheduler: BaseScheduler,
        estimator: object | None = None,
        lut_bundle: object | None = None,
        cpu_overhead_ns: int = 0,
        power_model: object | None = None,
        config: "SimulatorConfig | None" = None,
    ) -> None:
        from inference_simulator.engine.simulator_config import SimulatorConfig
        self._config = config or SimulatorConfig()
        self._model_spec = model_spec
        self._hardware_spec = hardware_spec
        self._inference_spec = inference_spec
        self._scheduler = scheduler
        self._estimator = estimator
        self._lut_bundle = lut_bundle
        self._lut_lookup = None
        self._tool_sampler: Optional[ToolLatencySampler] = None
        self._rng: Optional[np.random.Generator] = None
        self._cpu_overhead_ns = cpu_overhead_ns if cpu_overhead_ns > 0 else self._config.cpu_overhead_ns
        self._power_model = power_model

        # GQA batching overhead (Vidur-inspired)
        self._gqa_batching_overhead = 0.0
        if model_spec.num_kv_heads < model_spec.num_attention_heads:
            self._gqa_batching_overhead = 0.1  # 10% overhead for GQA with batching

        # Try to set up LUT-based timing if bundle provided
        self._prefill_lut = None
        self._decode_lut = None
        self._moe_lut = None
        self._ssm_lut = None
        if lut_bundle is not None:
            try:
                from inference_simulator.estimator.lut_lookup import LUTLookup
                prefill_path = getattr(lut_bundle, "gpu_attention_prefill_lut", None)
                decode_path = getattr(lut_bundle, "gpu_attention_decode_lut", None)
                token_ops_path = getattr(lut_bundle, "gpu_token_ops_lut", None)
                moe_path = getattr(lut_bundle, "gpu_moe_lut", None)
                ssm_path = getattr(lut_bundle, "gpu_ssm_lut", None)
                if prefill_path is not None and Path(prefill_path).exists():
                    self._prefill_lut = LUTLookup(prefill_path)
                if decode_path is not None and Path(decode_path).exists():
                    self._decode_lut = LUTLookup(decode_path)
                if token_ops_path is not None and Path(token_ops_path).exists():
                    self._lut_lookup = LUTLookup(token_ops_path)
                if moe_path is not None and Path(moe_path).exists():
                    self._moe_lut = LUTLookup(moe_path)
                    logger.info("Loaded MoE LUT from %s", moe_path)
                if ssm_path is not None and Path(ssm_path).exists():
                    self._ssm_lut = LUTLookup(ssm_path)
                    logger.info("Loaded SSM LUT from %s", ssm_path)
            except (ImportError, Exception) as e:
                logger.warning("Failed to initialize LUT lookup: %s", e)

            # Load tool distributions if available
            if hasattr(lut_bundle, "tool_distributions") and lut_bundle.tool_distributions:
                self._tool_sampler = ToolLatencySampler(lut_bundle.tool_distributions)

        # Default tool sampler if none from bundle
        if self._tool_sampler is None:
            self._tool_sampler = ToolLatencySampler()

        # Prefix tree for prefix caching (Fix 3A)
        self._prefix_tree = None
        enable_prefix_caching = inference_spec.engine_config.get(
            "enable_prefix_caching", False
        )
        if enable_prefix_caching:
            try:
                from inference_simulator.request.prefix_tree import PrefixTree
                self._prefix_tree = PrefixTree()
            except ImportError:
                logger.warning("prefix_tree module not available; prefix caching disabled")

        # Resolve timing functions (LUT takes priority over estimator/roofline)
        if self._lut_lookup is not None or self._prefill_lut is not None or self._decode_lut is not None:
            self._prefill_time_ns, self._decode_step_time_ns = (
                self._make_lut_timing_functions()
            )
        else:
            self._prefill_time_ns, self._decode_step_time_ns = _get_timing_functions(
                model_spec, hardware_spec, inference_spec, estimator
            )

        # Event handlers
        self._handlers: Dict[EventType, Callable[[Event], None]] = {
            EventType.REQUEST_ARRIVAL: self._handle_request_arrival,
            EventType.BATCH_SCHEDULE: self._handle_batch_schedule,
            EventType.PREFILL_COMPLETE: self._handle_prefill_complete,
            EventType.DECODE_STEP: self._handle_decode_step,
            EventType.DECODE_COMPLETE: self._handle_decode_complete,
            EventType.REQUEST_COMPLETE: self._handle_request_complete,
            EventType.TOOL_EXECUTION_START: self._handle_tool_execution_start,
            EventType.TOOL_EXECUTION_COMPLETE: self._handle_tool_execution_complete,
            EventType.STEP_COMPLETE: self._handle_step_complete,
        }

    def _make_lut_timing_functions(
        self,
    ) -> tuple[Callable[..., int], Callable[..., int]]:
        """Create timing functions that sum ALL operators across layers.

        A single transformer forward pass consists of:
          num_layers × (attention + linear_ops + norm_ops + activation_ops
                        + rotary_embedding)
          + per-model overhead (input embedding, lm_head)

        The attention LUT provides per-invocation attention time at a given
        (seq_len, batch_tokens) or (kv_cache_size, batch_size).  The token-ops
        LUT provides per-category-average time for (linear, normalization,
        activation, embedding) at a given token count.

        Per-layer operator counts are read from learned CompositionWeights
        (if available), falling back to SimulatorConfig integer defaults.
        Communication time is added for multi-GPU setups (PIE-P insight).
        """
        from inference_simulator.types.model_spec import ArchitectureType

        prefill_lut = self._prefill_lut
        decode_lut = self._decode_lut
        token_lut = self._lut_lookup
        moe_lut = self._moe_lut
        ssm_lut = self._ssm_lut
        config = self._config
        num_layers = self._model_spec.num_layers
        tp = self._inference_spec.num_gpus
        arch_type = self._model_spec.architecture_type

        # SSM/hybrid layer counts (for SSM-aware timing)
        attn_layer_count = self._model_spec.attention_layer_count
        ssm_layer_count = self._model_spec.ssm_layer_count
        is_moe = arch_type == ArchitectureType.MOE_TRANSFORMER
        is_ssm_hybrid = arch_type == ArchitectureType.SSM_HYBRID

        _MIN_TIME_S = config.min_time_s

        # --- Load learned composition weights (IrEne-inspired) ---
        # Priority: lut_bundle.composition_weights > config.composition_weights_path > config integers
        comp_weights = None
        try:
            cw_path = None
            if (
                self._lut_bundle is not None
                and hasattr(self._lut_bundle, "composition_weights")
                and self._lut_bundle.composition_weights is not None
            ):
                cw_path = Path(self._lut_bundle.composition_weights)
            elif config.composition_weights_path is not None:
                cw_path = Path(config.composition_weights_path)

            if cw_path is not None and cw_path.exists():
                from inference_simulator.estimator.composition_model import (
                    CompositionWeights,
                )
                comp_weights = CompositionWeights.from_json(cw_path)
                logger.info(
                    "Using learned composition weights from %s: "
                    "linear=%.2f norm=%.2f act=%.2f embed=%.2f overlap=%.3f",
                    cw_path,
                    comp_weights.linear_weight,
                    comp_weights.norm_weight,
                    comp_weights.activation_weight,
                    comp_weights.embedding_weight,
                    comp_weights.overlap_correction,
                )
        except Exception as e:
            logger.warning("Failed to load composition weights: %s", e)
            comp_weights = None

        # Resolve weight values (learned or config defaults)
        if comp_weights is not None:
            w_linear = comp_weights.linear_weight
            w_norm = comp_weights.norm_weight
            w_act = comp_weights.activation_weight
            w_embed = comp_weights.embedding_weight
            w_comm = comp_weights.communication_weight if tp > 1 else 0.0
            w_overlap = comp_weights.overlap_correction
        else:
            w_linear = float(config.ops_linear_per_layer)
            w_norm = float(config.ops_norm_per_layer)
            w_act = float(config.ops_activation_per_layer)
            w_embed = float(config.ops_embedding_per_layer)
            w_comm = 0.0
            w_overlap = config.overlap_correction

        # --- Roofline constants for decode (memory-bandwidth bound) ---
        # Total model weight bytes (all parameters)
        _weight_bytes = (
            self._model_spec.total_params_billion * 1e9
            * self._hardware_spec.bytes_per_param
        )
        # Effective HBM bandwidth (GB/s → B/s) × efficiency × TP
        _effective_bw = (
            self._hardware_spec.hbm_bandwidth_gb_s * 1e9
            * config.eta_decode
            * self._inference_spec.num_gpus
        )
        # KV cache bytes per token per layer = 2 (K+V) × num_kv_heads × head_dim × bytes
        _kv_bytes_per_token_per_layer = (
            2 * self._model_spec.num_kv_heads
            * self._model_spec.head_dim
            * self._hardware_spec.bytes_per_param
        )

        def _extract_time(result) -> float:
            """Extract time_s from LUT result (may be scalar or [time, energy] array).

            No per-component floor is applied here; the floor is applied to
            the final summed forward-pass time to prevent event storms.
            """
            val = np.asarray(result)
            if val.ndim == 0:
                return max(float(val), 0.0)
            return max(float(val.flat[0]), 0.0)

        try:
            from inference_simulator.estimator.lut_lookup import OutOfRangeError
        except ImportError:
            OutOfRangeError = None  # type: ignore[misc,assignment]

        _roofline_fallback = [None]  # mutable container for closure

        def _get_roofline():
            if _roofline_fallback[0] is None:
                _roofline_fallback[0] = _get_timing_functions(
                    self._model_spec, self._hardware_spec,
                    self._inference_spec, self._estimator,
                )
            return _roofline_fallback[0]

        def _safe_token_lookup(operator: str, token_count: int) -> float:
            """Query token-ops LUT, returning 0.0 on any failure."""
            if token_lut is None:
                return 0.0
            try:
                return _extract_time(
                    token_lut.lookup(operator=operator,
                                     token_count=token_count, tp_size=tp)
                )
            except Exception:
                return 0.0

        def _safe_moe_lookup(operator: str, token_count: int) -> float:
            """Query MoE LUT, returning 0.0 on any failure."""
            if moe_lut is None:
                return 0.0
            try:
                return _extract_time(
                    moe_lut.lookup(operator=operator,
                                   token_count=token_count, tp_size=tp)
                )
            except Exception:
                return 0.0

        def _safe_ssm_lookup(operator: str, token_count: int) -> float:
            """Query SSM LUT, returning 0.0 on any failure."""
            if ssm_lut is None:
                return 0.0
            try:
                return _extract_time(
                    ssm_lut.lookup(operator=operator,
                                   token_count=token_count, tp_size=tp)
                )
            except Exception:
                return 0.0

        def _sum_token_ops_per_layer(token_count: int) -> float:
            """Sum all per-layer token-op categories for one layer.

            Uses learned real-valued weights when available (IrEne-inspired),
            falling back to integer config defaults.

            For MoE models: adds router + expert computation and reduces
            standard FFN (linear) weight since experts replace it.
            """
            t = 0.0
            if is_moe and moe_lut is not None:
                # MoE replaces standard FFN: reduce linear weight, add MoE ops
                # Router runs once per layer, experts replace up/gate/down projections
                t += _safe_token_lookup("linear", token_count) * max(w_linear - 3.0, 0.0)
                t += _safe_moe_lookup("moe_routing", token_count) * 1.0
                t += _safe_moe_lookup("moe_expert", token_count) * w_linear
            else:
                t += _safe_token_lookup("linear", token_count) * w_linear
            t += _safe_token_lookup("normalization", token_count) * w_norm
            t += _safe_token_lookup("activation", token_count) * w_act
            t += _safe_token_lookup("embedding", token_count) * w_embed
            # Communication cost for TP>1 (PIE-P: 2 AllReduce per layer)
            if w_comm > 0.0:
                t += _safe_token_lookup("communication", token_count) * w_comm
            # SSM scan cost for hybrid models (replaces attention in SSM layers)
            if is_ssm_hybrid and ssm_lut is not None:
                t += _safe_ssm_lookup("ssm_scan", token_count)
            return t

        def prefill_time_ns(batch_size: int, seq_len: int) -> int:
            total_s = 0.0
            token_count = max(1, batch_size * seq_len)

            # --- Attention: 1 per layer (O(L^2) for attention, O(L) for SSM) ---
            attn_time = 0.0
            if prefill_lut is not None:
                try:
                    result = prefill_lut.lookup(
                        seq_len=seq_len, batch_tokens=batch_size, tp_size=tp)
                    attn_time = _extract_time(result)
                except Exception as exc:
                    if OutOfRangeError is not None and isinstance(exc, OutOfRangeError):
                        fb_prefill, _ = _get_roofline()
                        return fb_prefill(batch_size, seq_len)
                    # ignore ValueError/KeyError, fall through
            if attn_time == 0.0:
                # Fallback: try attention_prefill in token_ops LUT
                attn_time = _safe_token_lookup("attention_prefill", token_count)

            # --- Per-layer total: attention + token ops ---
            per_layer_s = attn_time + _sum_token_ops_per_layer(token_count)

            if is_ssm_hybrid and attn_layer_count < num_layers:
                # SSM hybrid: attention layers get O(L^2) prefill,
                # SSM layers get O(L) prefill (linear in sequence length).
                # Weight total time by the ratio of attention layers.
                attn_fraction = attn_layer_count / num_layers if num_layers > 0 else 1.0
                # Attention layers: full cost (quadratic attention)
                total_s += per_layer_s * attn_layer_count
                # SSM layers: reduced cost (linear scan, no quadratic attention)
                ssm_per_layer_s = _sum_token_ops_per_layer(token_count)
                if ssm_lut is not None:
                    ssm_per_layer_s += _safe_ssm_lookup("ssm_scan", token_count)
                total_s += ssm_per_layer_s * ssm_layer_count
            else:
                total_s += per_layer_s * num_layers

            # --- Per-model overhead: lm_head + input embedding ---
            total_s += _safe_token_lookup("lm_head", token_count)
            total_s += _safe_token_lookup("embedding", token_count)

            # Apply overlap correction (from learned weights or config)
            total_s *= w_overlap

            return int(max(total_s, _MIN_TIME_S) * 1e9)

        # SSM state memory: constant per layer (not growing like KV cache)
        # hidden_dim × state_dim × 2 (real + imaginary) × bytes_per_param
        _ssm_state_bytes_per_layer = 0
        if is_ssm_hybrid and self._model_spec.ssm_state_size is not None:
            _ssm_state_bytes_per_layer = (
                self._model_spec.hidden_dim
                * self._model_spec.ssm_state_size
                * 2  # real + imaginary components
                * self._hardware_spec.bytes_per_param
            )

        # For SSM hybrids, only attention layers have KV cache
        _kv_cache_num_layers = attn_layer_count if is_ssm_hybrid else num_layers

        def decode_step_time_ns(batch_size: int, kv_cache_len: int) -> int:
            """Decode is memory-bandwidth-bound for small batches.

            Time = (weight_bytes + kv_cache_bytes + ssm_state_bytes)
                   / (HBM_bandwidth × efficiency)

            Uses pure roofline model because the attention decode LUT was
            profiled with vanilla attention (O(n²) with kv_cache) but vLLM
            uses PagedAttention (O(n)).

            For SSM hybrid models: only attention layers contribute KV cache;
            SSM layers have fixed-size state (not growing with seq_len).
            """
            # KV cache memory read for attention layers only
            kv_cache_bytes = (
                _kv_bytes_per_token_per_layer
                * max(kv_cache_len, 1)
                * _kv_cache_num_layers
            )
            # SSM state memory: constant regardless of sequence length
            ssm_state_bytes = _ssm_state_bytes_per_layer * ssm_layer_count * batch_size

            # Roofline: total memory reads / effective bandwidth
            total_mem_bytes = _weight_bytes + kv_cache_bytes + ssm_state_bytes
            total_s = total_mem_bytes / _effective_bw if _effective_bw > 0 else _MIN_TIME_S

            # Scale with batch_size: each additional request adds KV reads
            if batch_size > 1:
                extra_kv_bytes = kv_cache_bytes * (batch_size - 1)
                total_s += extra_kv_bytes / _effective_bw if _effective_bw > 0 else 0.0

            total_s *= w_overlap

            # TP communication overhead: each decode step has num_layers*2
            # AllReduce ops (one after attention, one after FFN per layer).
            # In a ring topology, each AllReduce traverses (tp-1) hops.
            if tp > 1 and config.tp_scaling_mode == "measured":
                comm_time_s = num_layers * 2 * config.tp_comm_overhead_s * (tp - 1)
                total_s += comm_time_s

            return int(max(total_s, _MIN_TIME_S) * 1e9)

        return prefill_time_ns, decode_step_time_ns

    def run(
        self,
        workload_spec: WorkloadSpec,
        duration_s: float,
        seed: int | None = None,
        lut_bundle: object | None = None,
        workload_profile: object | None = None,
    ) -> SimulationMetrics:
        """Run the simulation.

        Args:
            workload_spec: Workload parameters.
            duration_s: Simulation duration in seconds.
            seed: Random seed for workload generation.
            lut_bundle: Optional LUT bundle override for this run.
            workload_profile: Optional WorkloadProfile for empirical generation.

        Returns:
            SimulationMetrics with latency percentiles, throughput, and energy.
        """
        # Handle per-run LUT bundle override
        if lut_bundle is not None and self._lut_bundle is None:
            try:
                from inference_simulator.estimator.lut_lookup import LUTLookup
                self._lut_lookup = LUTLookup(lut_bundle)
                self._prefill_time_ns, self._decode_step_time_ns = (
                    self._make_lut_timing_functions()
                )
                if hasattr(lut_bundle, "tool_distributions") and lut_bundle.tool_distributions:
                    self._tool_sampler = ToolLatencySampler(lut_bundle.tool_distributions)
            except (ImportError, Exception) as e:
                logger.warning("Failed to initialize per-run LUT lookup: %s", e)

        # Initialize state
        self._queue = EventQueue()
        self._current_time_ns: int = 0
        self._waiting: List[Request] = []
        self._running_batches: List[Batch] = []
        self._metrics = MetricsCollector(
            warmup_requests=self._inference_spec.max_batch_size,
        )
        self._rng = np.random.default_rng(seed)

        # Compute KV cache budget: total GPU memory minus model weights minus SSM state
        model_bytes = int(
            self._model_spec.total_params_billion
            * 1e9
            * self._hardware_spec.bytes_per_param
        )
        total_gpu_memory = int(
            self._hardware_spec.memory_gb
            * 1e9
            * self._inference_spec.num_gpus
        )
        # Reserve memory for SSM state in hybrid models
        # SSM state: hidden_dim × ssm_state_size × 2 × bytes × ssm_layers × max_batch
        ssm_state_reserve = 0
        from inference_simulator.types.model_spec import ArchitectureType as _AT
        if (
            self._model_spec.architecture_type == _AT.SSM_HYBRID
            and self._model_spec.ssm_state_size is not None
        ):
            ssm_state_reserve = int(
                self._model_spec.hidden_dim
                * self._model_spec.ssm_state_size
                * 2
                * self._hardware_spec.bytes_per_param
                * self._model_spec.ssm_layer_count
                * self._inference_spec.max_batch_size
            )
        kv_cache_budget = max(
            total_gpu_memory - model_bytes - ssm_state_reserve,
            total_gpu_memory // 4,
        )

        precision_bytes = 2.0 if self._inference_spec.precision == "fp16" else 1.0
        self._kv_cache = KVCacheManager(
            total_memory_bytes=kv_cache_budget,
            block_size=16,
            model_spec=self._model_spec,
            precision_bytes=precision_bytes,
        )

        # Generate workload
        generator = WorkloadGenerator()
        if workload_profile is not None:
            requests = generator.generate_from_profile(
                workload_profile, workload_spec.qps, duration_s,
                seed=seed, max_seq_len=self._inference_spec.max_seq_len,
            )
        elif workload_spec.workload_type is not None:
            requests = generator.generate_multi_step(
                workload_spec, duration_s, seed=seed,
                max_seq_len=self._inference_spec.max_seq_len,
            )
        else:
            requests = generator.generate(
                workload_spec, duration_s, seed=seed,
                max_seq_len=self._inference_spec.max_seq_len,
            )

        if not requests:
            return self._metrics.compute()

        # Enqueue arrival events
        for request in requests:
            self._queue.push(Event(
                time_ns=request.arrival_time_ns,
                event_type=EventType.REQUEST_ARRIVAL,
                payload={"request": request},
            ))

        # Main event loop
        end_time_ns = int(duration_s * 1e9)
        max_events = 500_000  # Safety limit to prevent runaway simulations
        event_count = 0
        while self._queue:
            event = self._queue.pop()

            if event.time_ns > end_time_ns:
                break

            event_count += 1
            if event_count > max_events:
                logger.warning(
                    "Event limit (%d) reached at %.3fs; stopping early",
                    max_events, event.time_ns / 1e9,
                )
                break

            self._current_time_ns = event.time_ns

            handler = self._handlers.get(event.event_type)
            if handler is not None:
                handler(event)

        # Compute energy estimate
        sim_duration_s = self._current_time_ns / 1e9

        if (
            self._power_model is not None
            and hasattr(self._power_model, "is_fitted")
            and self._power_model.is_fitted()
            and hasattr(self._metrics, "operator_events")
            and self._metrics.operator_events
        ):
            total_energy = self._power_model.compute_energy(
                self._metrics.operator_events
            )
            avg_power = (
                total_energy / sim_duration_s if sim_duration_s > 0 else 0.0
            )
            # Compute per-category energy breakdown (IrEne-inspired)
            if hasattr(self._power_model, "compute_energy_breakdown"):
                try:
                    breakdown = self._power_model.compute_energy_breakdown(
                        self._metrics.operator_events
                    )
                    self._metrics.set_energy_breakdown(breakdown)
                except Exception:
                    pass
        else:
            # Utilization-scaled power fallback: interpolates between idle
            # and active power fractions based on GPU utilization.
            gpu_active_ns = self._metrics._gpu_active_time_ns
            gpu_util = (
                min(gpu_active_ns / (sim_duration_s * 1e9), 1.0)
                if sim_duration_s > 0 else 0.0
            )
            idle_frac = self._config.power_idle_fraction
            active_frac = self._config.power_active_fraction
            per_gpu_power = self._hardware_spec.tdp_watts * (
                idle_frac + (active_frac - idle_frac) * gpu_util
            )
            avg_power = per_gpu_power * self._inference_spec.num_gpus
            total_energy = avg_power * sim_duration_s

        self._metrics.set_energy(total_energy)
        self._metrics.set_total_time(sim_duration_s)

        return self._metrics.compute()

    def _emit_forward_pass_events(
        self,
        total_duration_ns: int,
        batch_size: int,
        seq_len: int,
        phase: str = "prefill",
    ) -> None:
        """Split a forward-pass duration into per-layer-per-category events.

        When ``emit_per_layer_events`` is enabled, emits fine-grained events
        proportional to the LUT time ratios for each category. Otherwise,
        emits a single aggregate event (backward-compatible behavior).

        Default proportions (when no LUT is available):
          prefill: attention 60%, linear 25%, norm 10%, activation 5%
          decode:  attention 60%, linear 25%, norm 10%, activation 5%

        Args:
            total_duration_ns: Total forward-pass time in nanoseconds.
            batch_size: Batch size for the operation.
            seq_len: Sequence length for the operation.
            phase: Either "prefill" or "decode".
        """
        if not self._config.emit_per_layer_events:
            # Fallback: single aggregate event (backward-compatible)
            cat = (
                OperatorCategory.ATTENTION_PREFILL
                if phase == "prefill"
                else OperatorCategory.ATTENTION_DECODE
            )
            self._metrics.record_operator_event(
                category=cat,
                duration_ns=total_duration_ns,
                batch_size=batch_size,
                seq_len=seq_len,
                start_time_ns=self._current_time_ns,
            )
            return

        # Default per-category time fractions
        fractions = {
            OperatorCategory.ATTENTION_PREFILL if phase == "prefill"
            else OperatorCategory.ATTENTION_DECODE: 0.60,
            OperatorCategory.LINEAR: 0.25,
            OperatorCategory.NORMALIZATION: 0.10,
            OperatorCategory.ACTIVATION: 0.05,
        }

        num_layers = self._model_spec.num_layers
        per_layer_ns = total_duration_ns / max(num_layers, 1)

        time_cursor_ns = self._current_time_ns
        for layer_idx in range(num_layers):
            for cat, frac in fractions.items():
                cat_ns = int(per_layer_ns * frac)
                if cat_ns <= 0:
                    continue
                self._metrics.record_operator_event(
                    category=cat,
                    duration_ns=cat_ns,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    start_time_ns=time_cursor_ns,
                    layer_idx=layer_idx,
                )
                time_cursor_ns += cat_ns

    def _handle_request_arrival(self, event: Event) -> None:
        """Handle a new request arriving."""
        request: Request = event.payload["request"]

        # For multi-step requests, set input_tokens from the first step
        if request.is_multi_step and request.current_llm_step is not None:
            step = request.current_llm_step
            request.input_tokens = step.input_tokens
            request.max_output_tokens = step.output_tokens

        self._waiting.append(request)

        # Trigger scheduling
        self._queue.push(Event(
            time_ns=self._current_time_ns,
            event_type=EventType.BATCH_SCHEDULE,
        ))

    def _handle_batch_schedule(self, event: Event) -> None:
        """Run the scheduler and dispatch batches."""
        result = self._scheduler.schedule(
            self._waiting, self._running_batches, self._kv_cache,
        )

        # Handle preemptions
        for req in result.preempted_requests:
            req.state = RequestState.WAITING
            self._kv_cache.free(req)
            self._waiting.append(req)

        # Clear old running batches (they get re-batched by scheduler)
        self._running_batches.clear()

        for batch in result.new_batches:
            if batch.is_prefill:
                self._handle_prefill_batch(batch)
            else:
                self._running_batches.append(batch)
                self._dispatch_decode_step(batch)

    def _handle_prefill_batch(self, batch: Batch) -> None:
        """Start prefill for a batch of requests."""
        for request in batch.requests:
            # Remove from waiting
            if request in self._waiting:
                self._waiting.remove(request)

            request.state = RequestState.PREFILLING
            request.prefill_start_ns = self._current_time_ns

            # Prefix caching: reduce prefill length by matched prefix (Fix 3A)
            effective_input = request.input_tokens
            if self._prefix_tree is not None:
                # Use request_id as a simple token proxy (real implementation
                # would use actual token IDs, but that requires tokenizer)
                matched_node, matched_tokens = self._prefix_tree.match(
                    list(range(request.request_id * 1000,
                               request.request_id * 1000 + request.input_tokens))
                )
                if matched_tokens > 0:
                    prefix_matched = getattr(request, "prefix_matched_tokens", 0)
                    if hasattr(request, "prefix_matched_tokens"):
                        request.prefix_matched_tokens = matched_tokens
                    effective_input = max(1, request.input_tokens - matched_tokens)

            # Allocate KV cache (prefix blocks are shared, only allocate delta)
            if (
                self._prefix_tree is not None
                and hasattr(self._kv_cache, "allocate_with_prefix")
            ):
                self._kv_cache.allocate_with_prefix(
                    request, request.input_tokens,
                    getattr(request, "prefix_matched_tokens", 0),
                )
            else:
                self._kv_cache.allocate(request, request.input_tokens)

        # Chunked prefill (Fix 3B): if enabled and any request exceeds
        # chunk_size, split into chunks and interleave with decode.
        chunk_size = self._inference_spec.engine_config.get(
            "chunked_prefill_size", 0
        )
        enable_chunked = self._inference_spec.engine_config.get(
            "enable_chunked_prefill", False
        )

        if enable_chunked and chunk_size > 0:
            max_input = max(r.input_tokens for r in batch.requests)
            if max_input > chunk_size:
                # Schedule first chunk only; remaining chunks handled by
                # _handle_prefill_chunk_complete continuation events.
                chunk_tokens = min(chunk_size, max_input)
                chunk_ns = self._prefill_time_ns(1, chunk_tokens)
                chunk_ns += self._cpu_overhead_ns

                self._metrics.record_gpu_active_time(chunk_ns)

                remaining = max_input - chunk_tokens
                self._queue.push(Event(
                    time_ns=self._current_time_ns + chunk_ns,
                    event_type=EventType.PREFILL_COMPLETE,
                    payload={
                        "batch": batch,
                        "prefill_duration_ns": chunk_ns,
                        "remaining_tokens": remaining,
                        "chunk_size": chunk_size,
                    },
                ))
                return

        # Estimate prefill time using Vidur's equivalent-sequence-length formula.
        # A batch of prefills of lengths p1..pN costs ≈ a single prefill of
        # length sqrt(sum(pi^2)) because attention is O(n^2), so batch cost
        # is dominated by the sum of squares, not the sum of lengths.
        seq_lengths = [r.input_tokens for r in batch.requests]
        equiv_seq_len = int(math.sqrt(sum(p * p for p in seq_lengths)))
        prefill_ns = self._prefill_time_ns(1, equiv_seq_len)
        # Small linear overhead for multi-request batches
        if batch.size > 1:
            prefill_ns = int(prefill_ns * (1.0 + 0.05 * (batch.size - 1)))
        prefill_ns += self._cpu_overhead_ns  # Add CPU scheduling overhead

        # GQA batching overhead (Vidur-inspired)
        if self._gqa_batching_overhead > 0 and batch.size > 1:
            prefill_ns = int(prefill_ns * (1.0 + self._gqa_batching_overhead))

        # Track GPU active time
        self._metrics.record_gpu_active_time(prefill_ns)

        # Emit per-layer-per-category events for energy breakdown
        self._emit_forward_pass_events(
            total_duration_ns=prefill_ns,
            batch_size=batch.size,
            seq_len=equiv_seq_len,
            phase="prefill",
        )

        self._queue.push(Event(
            time_ns=self._current_time_ns + prefill_ns,
            event_type=EventType.PREFILL_COMPLETE,
            payload={"batch": batch, "prefill_duration_ns": prefill_ns},
        ))

    def _handle_prefill_complete(self, event: Event) -> None:
        """Transition prefilled requests to decode phase.

        Supports chunked prefill (Fix 3B): if ``remaining_tokens`` is in the
        payload the prefill is not yet finished — schedule a decode step for
        any pending decode batches, then continue with the next chunk.
        """
        batch: Batch = event.payload["batch"]
        prefill_duration_ns: int = event.payload.get("prefill_duration_ns", 0)
        remaining_tokens: int = event.payload.get("remaining_tokens", 0)
        chunk_size: int = event.payload.get("chunk_size", 0)

        # Chunked prefill continuation: more chunks remain
        if remaining_tokens > 0 and chunk_size > 0:
            # Allow a scheduling round so pending decode batches can run
            self._queue.push(Event(
                time_ns=self._current_time_ns,
                event_type=EventType.BATCH_SCHEDULE,
            ))

            # Schedule the next chunk
            next_chunk = min(chunk_size, remaining_tokens)
            next_chunk_ns = self._prefill_time_ns(1, next_chunk)
            next_chunk_ns += self._cpu_overhead_ns

            self._metrics.record_gpu_active_time(next_chunk_ns)

            self._queue.push(Event(
                time_ns=self._current_time_ns + next_chunk_ns,
                event_type=EventType.PREFILL_COMPLETE,
                payload={
                    "batch": batch,
                    "prefill_duration_ns": prefill_duration_ns + next_chunk_ns,
                    "remaining_tokens": remaining_tokens - next_chunk,
                    "chunk_size": chunk_size,
                },
            ))
            return

        for request in batch.requests:
            request.state = RequestState.DECODING
            if request.first_token_ns is None:
                request.first_token_ns = self._current_time_ns
            # Record per-step prefill time
            if request.is_multi_step:
                request.step_prefill_times_ns.append(prefill_duration_ns)

            # Insert prefix into cache tree for future reuse
            if self._prefix_tree is not None and request.input_tokens > 0:
                blocks = self._kv_cache.blocks_needed(request.input_tokens)
                self._prefix_tree.insert(
                    list(range(request.input_tokens)),  # Synthetic token IDs
                    kv_blocks=blocks,
                    time_ns=self._current_time_ns,
                )

        # Create decode batch and schedule first decode step
        decode_batch = Batch(
            batch_id=batch.batch_id + 10000,
            requests=list(batch.requests),
            is_prefill=False,
        )
        self._running_batches.append(decode_batch)
        self._dispatch_decode_step(decode_batch)

    def _dispatch_decode_step(self, batch: Batch) -> None:
        """Schedule the next decode step for a batch."""
        active_requests = [r for r in batch.requests if r.state == RequestState.DECODING]
        if not active_requests:
            return

        avg_kv_len = sum(r.total_tokens for r in active_requests) // len(active_requests)
        step_ns = self._decode_step_time_ns(len(active_requests), avg_kv_len)
        step_ns += self._cpu_overhead_ns  # Add CPU scheduling overhead

        # Track GPU active time
        self._metrics.record_gpu_active_time(step_ns)

        # Emit per-layer-per-category events for energy breakdown
        self._emit_forward_pass_events(
            total_duration_ns=step_ns,
            batch_size=len(active_requests),
            seq_len=avg_kv_len,
            phase="decode",
        )

        self._queue.push(Event(
            time_ns=self._current_time_ns + step_ns,
            event_type=EventType.DECODE_STEP,
            payload={"batch": batch, "step_duration_ns": step_ns},
        ))

    def _handle_decode_step(self, event: Event) -> None:
        """Process one decode step: generate one token per request."""
        batch: Batch = event.payload["batch"]
        step_duration_ns: int = event.payload.get("step_duration_ns", 0)

        completed_requests: List[Request] = []

        for request in batch.requests:
            if request.state != RequestState.DECODING:
                continue

            request.tokens_generated += 1
            self._metrics.record_decode_step(step_duration_ns)

            # Allocate cache for the new token
            self._kv_cache.allocate(request, 1)

            if request.tokens_generated >= request.max_output_tokens:
                completed_requests.append(request)

        # Handle completions
        for request in completed_requests:
            batch.requests.remove(request)

            # Check if multi-step with tool call
            if request.is_multi_step:
                step = request.current_llm_step
                if step is not None and step.tool_call is not None:
                    # Record decode time for this step
                    decode_ns = self._current_time_ns - (request.prefill_start_ns or 0)
                    request.step_decode_times_ns.append(decode_ns)
                    # Start tool execution instead of completing
                    request.state = RequestState.TOOL_EXECUTING

                    # Apply KV-cache retention policy during tool execution
                    policy = getattr(request, "retention_policy", "retain")
                    if policy == "retain" and hasattr(self._kv_cache, "retain"):
                        self._kv_cache.retain(request)
                    elif policy == "offload_cpu" and hasattr(self._kv_cache, "offload"):
                        request._offload_handle = self._kv_cache.offload(request)
                    else:
                        # "evict" or fallback: free immediately (legacy behavior)
                        self._kv_cache.free(request)

                    self._queue.push(Event(
                        time_ns=self._current_time_ns,
                        event_type=EventType.TOOL_EXECUTION_START,
                        payload={"request": request},
                    ))
                    continue

            # No tool call or single-step: complete
            request.state = RequestState.COMPLETED
            request.completion_ns = self._current_time_ns

            # Record per-step decode time for multi-step
            if request.is_multi_step:
                decode_ns = self._current_time_ns - (request.prefill_start_ns or 0)
                request.step_decode_times_ns.append(decode_ns)
                # Record step timing in metrics
                prefill_ns = request.step_prefill_times_ns[-1] if request.step_prefill_times_ns else 0
                self._metrics.record_step_timing(
                    prefill_ns=prefill_ns,
                    decode_ns=decode_ns,
                )

            self._kv_cache.free(request)
            self._metrics.record_request(request)

            # Trigger a schedule check to admit new requests
            self._queue.push(Event(
                time_ns=self._current_time_ns,
                event_type=EventType.BATCH_SCHEDULE,
            ))

        # Continue decoding if there are remaining requests
        active = [r for r in batch.requests if r.state == RequestState.DECODING]
        if active:
            self._dispatch_decode_step(batch)

    def _handle_tool_execution_start(self, event: Event) -> None:
        """Start tool execution for a multi-step request.

        GPU is idle during tool execution -- the scheduler can batch other
        requests during this time.
        """
        request: Request = event.payload["request"]
        step = request.current_llm_step

        if step is None or step.tool_call is None:
            # No tool to execute, advance directly
            self._queue.push(Event(
                time_ns=self._current_time_ns,
                event_type=EventType.TOOL_EXECUTION_COMPLETE,
                payload={"request": request, "tool_duration_ns": 0},
            ))
            return

        # Sample tool latency (pass current time for stateful Markov models)
        tool_latency_s = self._tool_sampler.sample_latency(
            step.tool_call.tool_type,
            step.tool_call.tool_config,
            self._rng,
            current_time_s=self._current_time_ns / 1e9,
        )
        tool_duration_ns = int(tool_latency_s * 1e9)

        self._queue.push(Event(
            time_ns=self._current_time_ns + tool_duration_ns,
            event_type=EventType.TOOL_EXECUTION_COMPLETE,
            payload={"request": request, "tool_duration_ns": tool_duration_ns},
        ))

        # Trigger scheduling so GPU can process other requests while tool runs
        self._queue.push(Event(
            time_ns=self._current_time_ns,
            event_type=EventType.BATCH_SCHEDULE,
        ))

    def _handle_tool_execution_complete(self, event: Event) -> None:
        """Tool execution finished. Advance to the next step."""
        request: Request = event.payload["request"]
        tool_duration_ns: int = event.payload.get("tool_duration_ns", 0)

        # Record tool time for the current step
        request.step_tool_times_ns.append(tool_duration_ns)

        # Record step timing in metrics
        prefill_ns = request.step_prefill_times_ns[-1] if request.step_prefill_times_ns else 0
        decode_ns = request.step_decode_times_ns[-1] if request.step_decode_times_ns else 0
        self._metrics.record_step_timing(
            prefill_ns=prefill_ns,
            decode_ns=decode_ns,
            tool_ns=tool_duration_ns,
        )

        # Schedule step completion
        self._queue.push(Event(
            time_ns=self._current_time_ns,
            event_type=EventType.STEP_COMPLETE,
            payload={"request": request},
        ))

    def _handle_step_complete(self, event: Event) -> None:
        """A step has completed. Advance to the next step or finish."""
        request: Request = event.payload["request"]

        has_more = request.advance_step()

        if has_more:
            # Update request for the next step
            next_step = request.current_llm_step
            if next_step is not None:
                # Update context: cumulative context grows
                request.input_tokens = next_step.input_tokens
                request.max_output_tokens = next_step.output_tokens
                request.tokens_generated = 0

            # Handle KV-cache based on retention policy
            policy = getattr(request, "retention_policy", "retain")
            if policy == "retain" and hasattr(self._kv_cache, "unretain"):
                # KV-cache still valid — release retention lock, only
                # prefill the NEW tokens (tool result), not full context.
                self._kv_cache.unretain(request)
            elif policy == "offload_cpu" and hasattr(self._kv_cache, "reload"):
                offload_handle = getattr(request, "_offload_handle", None)
                if offload_handle is not None:
                    self._kv_cache.reload(offload_handle)
                    request._offload_handle = None
            # For "evict", full re-prefill happens naturally

            # Re-enqueue as waiting for next prefill
            request.state = RequestState.AWAITING_NEXT_STEP
            request.state = RequestState.WAITING
            request.prefill_start_ns = None
            self._waiting.append(request)

            # Trigger scheduling
            self._queue.push(Event(
                time_ns=self._current_time_ns,
                event_type=EventType.BATCH_SCHEDULE,
            ))
        else:
            # All steps complete
            request.state = RequestState.COMPLETED
            request.completion_ns = self._current_time_ns
            self._metrics.record_request(request)

    def _handle_decode_complete(self, event: Event) -> None:
        """Handle decode completion (used for explicit completion signals)."""
        request: Request = event.payload.get("request")
        if request is not None and request.state == RequestState.DECODING:
            request.state = RequestState.COMPLETED
            request.completion_ns = self._current_time_ns
            self._kv_cache.free(request)
            self._metrics.record_request(request)

    def _handle_request_complete(self, event: Event) -> None:
        """Handle final request completion."""
        request: Request = event.payload.get("request")
        if request is not None:
            self._metrics.record_request(request)
