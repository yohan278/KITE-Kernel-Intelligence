"""Generate dense .npz lookup tables from trained estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from inference_simulator.estimator.base import BaseRuntimeEstimator, EstimatorResult
from inference_simulator.estimator.sklearn_base import (
    SklearnEstimatorBase,
    load_csv_measurements,
    load_csv_measurements_auto_category,
)
from inference_simulator.types.lut_bundle import LUTBundle
from inference_simulator.types.operators import OperatorCategory

# Default dense grid dimensions (powers of 2)
DEFAULT_SEQ_LENS = [2**i for i in range(7, 18)]  # 128 to 131072, 11 points
DEFAULT_BATCH_TOKENS = [2**i for i in range(0, 14)]  # 1 to 8192, 14 points
DEFAULT_TP_SIZES = [1, 2, 4, 8]
DEFAULT_KV_CACHE_SIZES = [2**i for i in range(7, 19)]  # 128 to 262144, 12 points
DEFAULT_BATCH_SIZES = [2**i for i in range(0, 8)]  # 1 to 128, 8 points

# Categories for token-level ops LUT
TOKEN_OP_CATEGORIES = [
    OperatorCategory.LINEAR,
    OperatorCategory.LM_HEAD,
    OperatorCategory.NORMALIZATION,
    OperatorCategory.ACTIVATION,
    OperatorCategory.EMBEDDING,
]

FUSED_CATEGORIES = [
    OperatorCategory.FUSED_PREFILL,
    OperatorCategory.FUSED_DECODE_STEP,
    OperatorCategory.FUSED_ATTENTION,
    OperatorCategory.FUSED_MLP,
    OperatorCategory.FUSED_NORM_ATTN,
]


class LUTGenerator:
    """Generates dense .npz lookup tables from trained estimators."""

    def generate_gpu_token_ops_lut(
        self,
        estimator: BaseRuntimeEstimator,
        operators: Sequence[OperatorCategory] = TOKEN_OP_CATEGORIES,
        token_counts: Sequence[int] = DEFAULT_BATCH_TOKENS,
        tp_sizes: Sequence[int] = DEFAULT_TP_SIZES,
        output_path: Optional[Path] = None,
        model_dims: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Generate [operators x token_counts x tp_sizes] -> (seconds, joules) .npz."""
        op_names = np.array([op.value for op in operators])
        token_arr = np.array(token_counts)
        tp_arr = np.array(tp_sizes)

        # Shape: (len(operators), len(token_counts), len(tp_sizes), 2)
        grid = np.zeros((len(operators), len(token_arr), len(tp_arr), 2))

        for i, op in enumerate(operators):
            for j, tokens in enumerate(token_counts):
                for k, tp in enumerate(tp_sizes):
                    result = estimator.estimate(
                        op, batch_size=tokens, seq_len=1,
                        model_dims=model_dims,
                    )
                    # Scale time linearly by tp_size (rough approximation)
                    grid[i, j, k, 0] = result.time_s / tp if tp > 0 else result.time_s
                    grid[i, j, k, 1] = (
                        result.energy_j / tp if result.energy_j is not None and tp > 0
                        else 0.0
                    )

        if output_path is None:
            output_path = Path("gpu_token_ops.npz")

        np.savez(
            output_path,
            grid=grid,
            axis_0=op_names,
            axis_1=token_arr,
            axis_2=tp_arr,
            axis_names=np.array(["operator", "token_count", "tp_size"]),
        )
        return Path(output_path)

    def generate_attention_prefill_lut(
        self,
        estimator: BaseRuntimeEstimator,
        seq_lens: Sequence[int] = DEFAULT_SEQ_LENS,
        batch_tokens: Sequence[int] = DEFAULT_BATCH_TOKENS,
        tp_sizes: Sequence[int] = DEFAULT_TP_SIZES,
        output_path: Optional[Path] = None,
        model_dims: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Generate [seq_lens x batch_tokens x tp_sizes] -> (seconds, joules) .npz."""
        seq_arr = np.array(seq_lens)
        batch_arr = np.array(batch_tokens)
        tp_arr = np.array(tp_sizes)

        grid = np.zeros((len(seq_arr), len(batch_arr), len(tp_arr), 2))

        for i, sl in enumerate(seq_lens):
            for j, bt in enumerate(batch_tokens):
                for k, tp in enumerate(tp_sizes):
                    result = estimator.estimate(
                        OperatorCategory.ATTENTION_PREFILL,
                        batch_size=bt, seq_len=sl,
                        model_dims=model_dims,
                    )
                    grid[i, j, k, 0] = result.time_s / tp if tp > 0 else result.time_s
                    grid[i, j, k, 1] = (
                        result.energy_j / tp if result.energy_j is not None and tp > 0
                        else 0.0
                    )

        if output_path is None:
            output_path = Path("attention_prefill.npz")

        np.savez(
            output_path,
            grid=grid,
            axis_0=seq_arr,
            axis_1=batch_arr,
            axis_2=tp_arr,
            axis_names=np.array(["seq_len", "batch_tokens", "tp_size"]),
        )
        return Path(output_path)

    def generate_attention_decode_lut(
        self,
        estimator: BaseRuntimeEstimator,
        kv_cache_sizes: Sequence[int] = DEFAULT_KV_CACHE_SIZES,
        batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
        tp_sizes: Sequence[int] = DEFAULT_TP_SIZES,
        output_path: Optional[Path] = None,
        model_dims: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Generate [kv_cache_sizes x batch_sizes x tp_sizes] -> (seconds, joules) .npz."""
        kv_arr = np.array(kv_cache_sizes)
        batch_arr = np.array(batch_sizes)
        tp_arr = np.array(tp_sizes)

        grid = np.zeros((len(kv_arr), len(batch_arr), len(tp_arr), 2))

        for i, kv in enumerate(kv_cache_sizes):
            for j, bs in enumerate(batch_sizes):
                for k, tp in enumerate(tp_sizes):
                    result = estimator.estimate(
                        OperatorCategory.ATTENTION_DECODE,
                        batch_size=bs, seq_len=kv,
                        model_dims=model_dims,
                    )
                    grid[i, j, k, 0] = result.time_s / tp if tp > 0 else result.time_s
                    grid[i, j, k, 1] = (
                        result.energy_j / tp if result.energy_j is not None and tp > 0
                        else 0.0
                    )

        if output_path is None:
            output_path = Path("attention_decode.npz")

        np.savez(
            output_path,
            grid=grid,
            axis_0=kv_arr,
            axis_1=batch_arr,
            axis_2=tp_arr,
            axis_names=np.array(["kv_cache_size", "batch_size", "tp_size"]),
        )
        return Path(output_path)

    def _generate_cpu_ops_lut(
        self,
        measurements: List["OperatorMeasurement"],
        output_path: Path,
    ) -> Path:
        """Generate a 1D CPU overhead LUT indexed by batch_size.

        For each batch_size, sums ``time_s`` and ``energy_j`` across all CPU
        operators (scheduler, tokenizer, batching, PCIe).  The simulator's
        ``_get_cpu_overhead_ns()`` looks up ``batch_size -> (time_s, energy_j)``
        from this LUT.
        """
        from collections import defaultdict

        # Aggregate per batch_size
        time_by_bs: Dict[int, List[float]] = defaultdict(list)
        energy_by_bs: Dict[int, List[float]] = defaultdict(list)

        for m in measurements:
            time_by_bs[m.batch_size].append(m.time_s)
            if m.energy_j is not None:
                energy_by_bs[m.batch_size].append(m.energy_j)

        batch_sizes = sorted(time_by_bs.keys())
        grid = np.zeros((len(batch_sizes), 2))

        for i, bs in enumerate(batch_sizes):
            grid[i, 0] = sum(time_by_bs[bs])  # total CPU overhead time
            grid[i, 1] = sum(energy_by_bs.get(bs, [0.0]))  # total CPU energy

        np.savez(
            output_path,
            grid=grid,
            axis_0=np.array(batch_sizes),
            axis_names=np.array(["batch_size"]),
        )
        return Path(output_path)

    def generate_full_bundle(
        self,
        profiling_dir: Path,
        output_dir: Path,
        model_spec: object,
        hw_spec: object,
    ) -> LUTBundle:
        """Full pipeline: load CSVs -> train best estimator -> generate all LUTs -> return bundle.

        Auto-detects which CSVs exist in ``profiling_dir`` and trains only
        relevant estimators.

        Args:
            profiling_dir: Directory containing profiling CSV files.
            output_dir: Directory to write .npz files.
            model_spec: ModelSpec instance (used for metadata).
            hw_spec: HardwareSpec instance (used for metadata).

        Returns:
            LUTBundle referencing all generated files.
        """
        from inference_simulator.estimator.model_comparison import (
            compare_estimators,
            pick_best_estimator,
        )
        from inference_simulator.estimator.random_forest import RandomForestEstimator

        profiling_dir = Path(profiling_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map from individual CSV file names to categories
        csv_category_map = {
            "linear": OperatorCategory.LINEAR,
            "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
            "attention_decode": OperatorCategory.ATTENTION_DECODE,
            "embedding": OperatorCategory.EMBEDDING,
            "normalization": OperatorCategory.NORMALIZATION,
            "activation": OperatorCategory.ACTIVATION,
            "moe_routing": OperatorCategory.MOE_ROUTING,
            "moe_expert": OperatorCategory.MOE_EXPERT,
            "ssm_scan": OperatorCategory.SSM_SCAN,
            # communication excluded: uses different schema (operation, num_gpus, message_size_bytes)
            "agentic_tool": OperatorCategory.AGENTIC_TOOL,
            "fused_prefill": OperatorCategory.FUSED_PREFILL,
            "fused_decode_step": OperatorCategory.FUSED_DECODE_STEP,
            "fused_attention": OperatorCategory.FUSED_ATTENTION,
            "fused_mlp": OperatorCategory.FUSED_MLP,
            "fused_norm_attn": OperatorCategory.FUSED_NORM_ATTN,
        }

        # Map operator_name values (from combined CSVs) to categories
        operator_name_to_category = {
            "linear_qkv": OperatorCategory.LINEAR,
            "linear_o": OperatorCategory.LINEAR,
            "mlp_up": OperatorCategory.LINEAR,
            "mlp_gate": OperatorCategory.LINEAR,
            "mlp_down": OperatorCategory.LINEAR,
            "lm_head": OperatorCategory.LM_HEAD,
            "embedding": OperatorCategory.EMBEDDING,
            # rotary_embedding excluded: fused inside FlashAttention kernel,
            # already captured by attention_prefill/attention_decode profiling.
            "layernorm": OperatorCategory.NORMALIZATION,
            "rmsnorm": OperatorCategory.NORMALIZATION,
            "gelu_activation": OperatorCategory.ACTIVATION,
            "silu_activation": OperatorCategory.ACTIVATION,
            "softmax": OperatorCategory.ACTIVATION,
            "dropout": OperatorCategory.ACTIVATION,
            # cross_entropy_loss excluded: training-only op, not used in inference.
            "residual_add": OperatorCategory.ACTIVATION,
            "attention_prefill": OperatorCategory.ATTENTION_PREFILL,
            "attention_decode": OperatorCategory.ATTENTION_DECODE,
            "sliding_window_attention": OperatorCategory.ATTENTION_DECODE,
            "kv_cache_append": OperatorCategory.KV_CACHE,
            "kv_cache_evict": OperatorCategory.KV_CACHE,
            "mqa_gqa_expansion": OperatorCategory.KV_CACHE,
            "fused_prefill": OperatorCategory.FUSED_PREFILL,
            "fused_decode_step": OperatorCategory.FUSED_DECODE_STEP,
            "fused_attention": OperatorCategory.FUSED_ATTENTION,
            "fused_mlp": OperatorCategory.FUSED_MLP,
            "fused_norm_attn": OperatorCategory.FUSED_NORM_ATTN,
            # CPU host operators
            "cpu_offload_transfer": OperatorCategory.CPU_HOST,
            "gpu_mem_alloc": OperatorCategory.CPU_HOST,
            "pcie_h2d_copy": OperatorCategory.CPU_HOST,
            "pcie_d2h_copy": OperatorCategory.CPU_HOST,
            "scheduler_overhead": OperatorCategory.CPU_HOST,
            "tokenizer_encode": OperatorCategory.CPU_HOST,
            "tokenizer_decode": OperatorCategory.CPU_HOST,
            "dynamic_batching_overhead": OperatorCategory.CPU_HOST,
            # MoE operators (from MoEProfiler combined CSV)
            "moe_router": OperatorCategory.MOE_ROUTING,
            "expert_dispatch": OperatorCategory.MOE_ROUTING,
            "load_balancing_loss": OperatorCategory.MOE_ROUTING,
            "moe_expert_mlp": OperatorCategory.MOE_EXPERT,
            "moe_combine": OperatorCategory.MOE_EXPERT,
            "expert_combine": OperatorCategory.MOE_EXPERT,
            "capacity_factor_overhead": OperatorCategory.MOE_EXPERT,
            "shared_expert_mlp": OperatorCategory.MOE_EXPERT,
            # SSM operators (from SSMProfiler combined CSV)
            "ssm_scan": OperatorCategory.SSM_SCAN,
            "ssm_conv1d": OperatorCategory.SSM_SCAN,
            "ssm_discretize": OperatorCategory.SSM_SCAN,
            "ssm_gate": OperatorCategory.SSM_SCAN,
            "ssm_residual_mix": OperatorCategory.SSM_SCAN,
        }

        # Combined CSV names that contain mixed operator categories
        combined_csvs = {"token_ops", "attention", "agentic", "sampling",
                         "communication", "moe", "ssm", "mtp", "cpu_host"}

        csv_paths: List[Tuple[Path, OperatorCategory]] = []
        combined_csv_paths: List[Path] = []

        for name, cat in csv_category_map.items():
            csv_file = profiling_dir / f"{name}.csv"
            if csv_file.exists():
                csv_paths.append((csv_file, cat))

        # Also try glob for any .csv files we might have missed
        for csv_file in profiling_dir.glob("*.csv"):
            stem = csv_file.stem.lower()
            if stem in csv_category_map:
                pair = (csv_file, csv_category_map[stem])
                if pair not in csv_paths:
                    csv_paths.append(pair)
            elif stem in combined_csvs:
                combined_csv_paths.append(csv_file)

        if not csv_paths and not combined_csv_paths:
            raise FileNotFoundError(
                f"No profiling CSVs found in {profiling_dir}"
            )

        # Load measurements from individual-category CSVs
        all_measurements = []
        for csv_path, category in csv_paths:
            all_measurements.extend(load_csv_measurements(csv_path, category))

        # Load measurements from combined CSVs, inferring category per row
        for csv_path in combined_csv_paths:
            all_measurements.extend(
                load_csv_measurements_auto_category(
                    csv_path, operator_name_to_category
                )
            )

        if len(all_measurements) < 2:
            raise ValueError("Not enough measurements to train estimator")

        # Collect available estimator classes for comparison
        estimator_classes: List[Type[SklearnEstimatorBase]] = [RandomForestEstimator]

        try:
            from inference_simulator.estimator.ridge import RidgeRegressionEstimator
            estimator_classes.append(RidgeRegressionEstimator)
        except ImportError:
            pass

        try:
            from inference_simulator.estimator.knn import KNNEstimator
            estimator_classes.append(KNNEstimator)
        except ImportError:
            pass

        # Compare and pick best
        # Also evaluate PerOperatorEstimator alongside SklearnEstimatorBase subclasses
        per_op_est = None
        try:
            from inference_simulator.estimator.per_operator_estimator import (
                PerOperatorEstimator,
            )

            per_op_est = PerOperatorEstimator()
            per_op_scores = per_op_est.fit(all_measurements)
        except (ImportError, Exception):
            per_op_est = None
            per_op_scores = {}

        if len(estimator_classes) > 1 and len(all_measurements) >= 10:
            comparison = compare_estimators(
                all_measurements, None, estimator_classes
            )

            # Evaluate PerOperatorEstimator on the same validation set
            if per_op_est is not None:
                try:
                    from sklearn.model_selection import train_test_split

                    X, y_time, _, _ = SklearnEstimatorBase._build_dataset(
                        all_measurements, None
                    )
                    indices = list(range(len(X)))
                    _train_idx, val_idx = train_test_split(
                        indices, test_size=0.2, random_state=42
                    )
                    val_ms = [all_measurements[i] for i in val_idx]
                    y_time_val = y_time[val_idx]
                    y_pred = np.array(
                        [
                            per_op_est.estimate(
                                m.category, m.batch_size, m.seq_len
                            ).time_s
                            for m in val_ms
                        ]
                    )
                    from inference_simulator.estimator.model_comparison import (
                        _compute_metrics,
                    )

                    metrics = _compute_metrics(y_time_val, y_pred)
                    comparison.append(
                        {
                            "estimator": "PerOperatorEstimator",
                            "time_r2": metrics["r2"],
                            "time_mae": metrics["mae"],
                            "time_rmse": metrics["rmse"],
                        }
                    )
                except Exception:
                    pass

            best_name = pick_best_estimator(comparison, "time_r2")
            if best_name == "PerOperatorEstimator":
                best_cls = None  # Signal to use per_op_est
            else:
                best_cls = next(
                    c for c in estimator_classes if c.__name__ == best_name
                )
        else:
            best_cls = RandomForestEstimator

        # Train best estimator on full dataset
        if best_cls is None and per_op_est is not None:
            estimator: BaseRuntimeEstimator = per_op_est
            scores = per_op_scores
        else:
            if best_cls is None:
                best_cls = RandomForestEstimator
            sklearn_est = best_cls()
            scores = sklearn_est.fit(all_measurements)
            estimator = sklearn_est

        # Extract model/hardware identifiers
        model_id = getattr(model_spec, "model_id", "unknown_model")
        hardware_id = getattr(hw_spec, "name", "unknown_hw")
        quantization = getattr(model_spec, "metadata", {}).get("quantization", "bf16")

        # Generate LUTs
        token_ops_path = self.generate_gpu_token_ops_lut(
            estimator, output_path=output_dir / "gpu_token_ops.npz"
        )
        prefill_path = self.generate_attention_prefill_lut(
            estimator, output_path=output_dir / "attention_prefill.npz"
        )
        decode_path = self.generate_attention_decode_lut(
            estimator, output_path=output_dir / "attention_decode.npz"
        )

        # Generate CPU host LUT if CPU_HOST measurements exist
        cpu_host_measurements = [
            m for m in all_measurements if m.category == OperatorCategory.CPU_HOST
        ]
        cpu_ops_path: Optional[Path] = None
        if cpu_host_measurements:
            cpu_ops_path = self._generate_cpu_ops_lut(
                cpu_host_measurements, output_dir / "cpu_host.npz"
            )

        # Generate MoE LUT if MOE measurements exist
        moe_categories = [OperatorCategory.MOE_ROUTING, OperatorCategory.MOE_EXPERT]
        moe_measurements = [
            m for m in all_measurements if m.category in moe_categories
        ]
        gpu_moe_path: Optional[Path] = None
        if moe_measurements:
            gpu_moe_path = self.generate_gpu_token_ops_lut(
                estimator,
                operators=moe_categories,
                output_path=output_dir / "gpu_moe.npz",
            )

        # Generate SSM LUT if SSM_SCAN measurements exist
        ssm_measurements = [
            m for m in all_measurements if m.category == OperatorCategory.SSM_SCAN
        ]
        gpu_ssm_path: Optional[Path] = None
        if ssm_measurements:
            gpu_ssm_path = self.generate_gpu_token_ops_lut(
                estimator,
                operators=[OperatorCategory.SSM_SCAN],
                output_path=output_dir / "gpu_ssm.npz",
            )

        # Generate fused-ops LUT if any fused measurements exist
        fused_lut_path: Optional[Path] = None
        has_fused = any(
            m.category in FUSED_CATEGORIES for m in all_measurements
        )
        if has_fused:
            fused_lut_path = self.generate_gpu_token_ops_lut(
                estimator,
                operators=FUSED_CATEGORIES,
                output_path=output_dir / "fused_ops.npz",
            )

        # Fit tool distributions if agentic data exists
        tool_dist_path: Optional[Path] = None
        agentic_csv = profiling_dir / "agentic_tool.csv"
        if agentic_csv.exists():
            try:
                from inference_simulator.estimator.tool_distribution import (
                    ToolDistributionFitter,
                )
                fitter = ToolDistributionFitter()
                dists = fitter.fit_all_tools(agentic_csv)
                tool_dist_path = fitter.save(dists, output_dir / "tool_distributions.pkl")
            except (ImportError, Exception):
                pass

        # Train composition weights from FUSED_PREFILL measurements (IrEne-inspired)
        composition_weights_path: Optional[Path] = None
        try:
            fused_prefill_measurements = [
                m for m in all_measurements
                if m.category == OperatorCategory.FUSED_PREFILL
            ]
            if len(fused_prefill_measurements) >= 3:
                from inference_simulator.estimator.composition_model import (
                    CompositionModelTrainer,
                )
                trainer = CompositionModelTrainer(model_spec, hw_spec)
                weights = trainer.fit_from_fused_data(
                    fused_prefill_measurements, estimator
                )
                cw_path = output_dir / "composition_weights.json"
                weights.to_json(cw_path)
                composition_weights_path = cw_path
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to train composition weights: %s", e
            )

        metadata = {
            "estimator_class": type(estimator).__name__,
            "training_scores": scores,
            "num_measurements": len(all_measurements),
            "csv_files": [str(p) for p, _ in csv_paths],
        }

        return LUTBundle(
            base_dir=output_dir,
            model_id=model_id,
            hardware_id=hardware_id,
            quantization=quantization,
            gpu_token_ops_lut=token_ops_path,
            gpu_attention_prefill_lut=prefill_path,
            gpu_attention_decode_lut=decode_path,
            gpu_moe_lut=gpu_moe_path,
            gpu_ssm_lut=gpu_ssm_path,
            fused_ops_lut=fused_lut_path,
            cpu_ops_lut=cpu_ops_path,
            tool_distributions=tool_dist_path,
            composition_weights=composition_weights_path,
            metadata=metadata,
        )
