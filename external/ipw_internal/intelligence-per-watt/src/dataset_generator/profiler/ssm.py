"""SSM operator profiler for state-space model building blocks."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class SSMProfiler(BaseOperatorProfiler):
    """Profiles SSM scan operations: selective scan, convolution, discretization, gating.

    Models Mamba-style state-space layers with configurable state dimension.
    """

    DEFAULT_STATE_DIM = 16

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.SSM_SCAN

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes", "prefill_seq_lengths"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile all SSM operators across sweep dimensions."""
        import torch

        measurements: List[OperatorMeasurement] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        operators = self._build_operators(model_spec, device, dtype)

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]

            for op_def in operators:
                try:
                    input_tensor = op_def["input_fn"](batch_size, seq_len)
                    flops = op_def["flops_fn"](batch_size, seq_len)

                    measurement = harness.measure(
                        op_def["forward_fn"],
                        input_tensor,
                        operator_name=op_def["name"],
                        category=op_def["category"],
                        batch_size=batch_size,
                        seq_len=seq_len,
                        flops=flops,
                    )
                    measurements.append(measurement)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    continue

        return measurements

    def _build_operators(
        self, model_spec: ModelSpec, device: str, dtype: Any
    ) -> List[Dict[str, Any]]:
        """Build SSM operator definitions."""
        import torch
        import torch.nn as nn

        h = model_spec.hidden_dim
        state_dim = (
            model_spec.ssm_state_size
            if model_spec.ssm_state_size is not None
            else self.DEFAULT_STATE_DIM
        )
        operators = []

        # --- SSM Scan (sequential) ---
        def _make_ssm_scan_fn():
            _cache = {}

            def input_fn(b, s):
                # Pre-create A, B, C matrices for the scan
                _cache["A"] = torch.randn(h, state_dim, device=device, dtype=dtype) * 0.1
                _cache["B"] = torch.randn(h, state_dim, device=device, dtype=dtype) * 0.1
                _cache["C"] = torch.randn(h, state_dim, device=device, dtype=dtype) * 0.1
                return torch.randn(b, s, h, device=device, dtype=dtype)

            def forward_fn(x):
                b, s, _h = x.shape
                A = _cache["A"]  # (h, state_dim)
                B = _cache["B"]  # (h, state_dim)
                C = _cache["C"]  # (h, state_dim)
                state = torch.zeros(b, _h, state_dim, device=x.device, dtype=x.dtype)
                outputs = []
                for t in range(s):
                    xt = x[:, t, :].unsqueeze(-1)  # (b, h, 1)
                    state = state * A.unsqueeze(0) + xt * B.unsqueeze(0)
                    yt = (state * C.unsqueeze(0)).sum(dim=-1)  # (b, h)
                    outputs.append(yt)
                return torch.stack(outputs, dim=1)

            return input_fn, forward_fn

        scan_input_fn, scan_forward_fn = _make_ssm_scan_fn()
        operators.append({
            "name": "ssm_scan",
            "category": OperatorCategory.SSM_SCAN,
            "forward_fn": scan_forward_fn,
            "input_fn": scan_input_fn,
            # Per timestep: 2 multiplies + 1 add for state update, 1 mul + 1 sum for output
            # Each op is over (b, h, state_dim), so flops per step ~ 5 * b * h * state_dim
            "flops_fn": lambda b, s: 5 * b * s * h * state_dim,
        })

        # --- SSM Conv1d ---
        conv = nn.Conv1d(h, h, kernel_size=4, groups=h, padding=3).to(
            device=device, dtype=dtype
        )
        operators.append({
            "name": "ssm_conv1d",
            "category": OperatorCategory.SSM_SCAN,
            "forward_fn": lambda x, _m=conv: _m(x)[..., :x.shape[-1]],
            "input_fn": lambda b, s: torch.randn(b, h, s, device=device, dtype=dtype),
            # Depthwise conv: kernel_size * h * seq_len * batch
            "flops_fn": lambda b, s: 4 * b * h * s,
        })

        # --- SSM Discretize (element-wise exp) ---
        def _make_discretize_fn():
            _cache = {}

            def input_fn(b, s):
                _cache["A"] = torch.randn(
                    h, state_dim, device=device, dtype=dtype
                ) * 0.01
                return torch.randn(b, s, h, 1, device=device, dtype=dtype)

            def forward_fn(delta):
                A = _cache["A"]  # (h, state_dim)
                # Broadcast: delta is (b, s, h, 1), A is (h, state_dim)
                return torch.exp(delta * A)

            return input_fn, forward_fn

        disc_input_fn, disc_forward_fn = _make_discretize_fn()
        operators.append({
            "name": "ssm_discretize",
            "category": OperatorCategory.SSM_SCAN,
            "forward_fn": disc_forward_fn,
            "input_fn": disc_input_fn,
            # mul + exp over (b, s, h, state_dim)
            "flops_fn": lambda b, s: 2 * b * s * h * state_dim,
        })

        # --- SSM Gate (sigmoid) ---
        operators.append({
            "name": "ssm_gate",
            "category": OperatorCategory.SSM_SCAN,
            "forward_fn": lambda x: torch.sigmoid(x),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 4 * b * s * h,  # sigmoid ~ 4 ops
        })

        # --- SSM Residual Mix: x * gate + y * (1 - gate) ---
        def _make_residual_mix_fn():
            _cache = {}

            def input_fn(b, s):
                _cache["y"] = torch.randn(b, s, h, device=device, dtype=dtype)
                _cache["gate"] = torch.sigmoid(
                    torch.randn(b, s, h, device=device, dtype=dtype)
                )
                return torch.randn(b, s, h, device=device, dtype=dtype)

            def forward_fn(x):
                gate = _cache["gate"]
                y = _cache["y"]
                return x * gate + y * (1.0 - gate)

            return input_fn, forward_fn

        mix_input_fn, mix_forward_fn = _make_residual_mix_fn()
        operators.append({
            "name": "ssm_residual_mix",
            "category": OperatorCategory.SSM_SCAN,
            "forward_fn": mix_forward_fn,
            "input_fn": mix_input_fn,
            # 2 muls + 1 sub + 1 add = 4 ops per element
            "flops_fn": lambda b, s: 4 * b * s * h,
        })

        return operators
