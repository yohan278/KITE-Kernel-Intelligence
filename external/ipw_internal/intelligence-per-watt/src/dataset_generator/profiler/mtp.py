"""Multi-Token Prediction (MTP) and speculative decoding operator profiler."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class MTPProfiler(BaseOperatorProfiler):
    """Profiles multi-token prediction and speculative decoding operators.

    Sweeps over batch_sizes with an internal sweep over num_draft_tokens.
    Each measurement records num_draft_tokens in its metadata.
    """

    DEFAULT_NUM_DRAFT_TOKENS = [1, 2, 4, 8]

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.MTP

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile MTP operators across batch sizes and draft token counts."""
        import torch

        measurements: List[OperatorMeasurement] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        h = model_spec.hidden_dim
        vocab = model_spec.vocab_size

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]

            for num_draft in self.DEFAULT_NUM_DRAFT_TOKENS:
                operators = self._build_operators(
                    model_spec, device, dtype, num_draft
                )

                for op_def in operators:
                    try:
                        input_tensor = op_def["input_fn"](batch_size)
                        flops = op_def["flops_fn"](batch_size)

                        measurement = harness.measure(
                            op_def["forward_fn"],
                            input_tensor,
                            operator_name=op_def["name"],
                            category=op_def["category"],
                            batch_size=batch_size,
                            seq_len=num_draft,
                            flops=flops,
                        )
                        measurement.metadata["num_draft_tokens"] = num_draft
                        measurements.append(measurement)
                    except (RuntimeError, torch.cuda.OutOfMemoryError):
                        continue

        return measurements

    def _build_operators(
        self,
        model_spec: ModelSpec,
        device: str,
        dtype: Any,
        num_draft: int,
    ) -> List[Dict[str, Any]]:
        """Build MTP operator definitions for a given num_draft_tokens."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        h = model_spec.hidden_dim
        vocab = model_spec.vocab_size
        operators = []

        # --- MTP Head Forward ---
        mtp_head = nn.Linear(h, vocab, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "mtp_head_forward",
            "category": OperatorCategory.MTP,
            "forward_fn": lambda x, _m=mtp_head: _m(x),
            "input_fn": lambda b: torch.randn(
                b, num_draft, h, device=device, dtype=dtype
            ),
            "flops_fn": lambda b: 2 * b * num_draft * h * vocab,
        })

        # --- MTP Loss ---
        def _make_loss_fn():
            _cache = {}

            def input_fn(b):
                _cache["targets"] = torch.randint(
                    0, vocab, (b * num_draft,), device=device
                )
                return torch.randn(
                    b * num_draft, vocab, device=device, dtype=dtype
                )

            def forward_fn(logits):
                return F.cross_entropy(logits, _cache["targets"])

            return input_fn, forward_fn

        loss_input_fn, loss_forward_fn = _make_loss_fn()
        operators.append({
            "name": "mtp_loss",
            "category": OperatorCategory.MTP,
            "forward_fn": loss_forward_fn,
            "input_fn": loss_input_fn,
            "flops_fn": lambda b: b * num_draft * vocab,
        })

        # --- MTP Token Merge ---
        merge_linear = nn.Linear(h * num_draft, h, bias=False).to(
            device=device, dtype=dtype
        )

        def _make_merge_fn():
            def forward_fn(x):
                b = x.shape[0]
                # x is (b, num_draft, h) -> flatten to (b, num_draft*h)
                flat = x.reshape(b, -1)
                return merge_linear(flat)

            return forward_fn

        operators.append({
            "name": "mtp_token_merge",
            "category": OperatorCategory.MTP,
            "forward_fn": _make_merge_fn(),
            "input_fn": lambda b: torch.randn(
                b, num_draft, h, device=device, dtype=dtype
            ),
            "flops_fn": lambda b: 2 * b * h * num_draft * h,
        })

        # --- Speculative Draft ---
        draft_linear = nn.Linear(h, h, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "speculative_draft",
            "category": OperatorCategory.MTP,
            "forward_fn": lambda x, _m=draft_linear: _m(x),
            "input_fn": lambda b: torch.randn(b, 1, h, device=device, dtype=dtype),
            "flops_fn": lambda b: 2 * b * h * h,
        })

        # --- Speculative Verify ---
        verify_head = nn.Linear(h, vocab, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "speculative_verify",
            "category": OperatorCategory.MTP,
            "forward_fn": lambda x, _m=verify_head: _m(x),
            "input_fn": lambda b: torch.randn(
                b, num_draft, h, device=device, dtype=dtype
            ),
            "flops_fn": lambda b: 2 * b * num_draft * h * vocab,
        })

        # --- Draft Accept/Reject ---
        def _make_accept_reject_fn():
            _cache = {}

            def input_fn(b):
                _cache["target_tokens"] = torch.randint(
                    0, vocab, (b, num_draft), device=device
                )
                return torch.randint(0, vocab, (b, num_draft), device=device)

            def forward_fn(draft_tokens):
                return torch.eq(draft_tokens, _cache["target_tokens"])

            return input_fn, forward_fn

        ar_input_fn, ar_forward_fn = _make_accept_reject_fn()
        operators.append({
            "name": "draft_accept_reject",
            "category": OperatorCategory.MTP,
            "forward_fn": ar_forward_fn,
            "input_fn": ar_input_fn,
            "flops_fn": lambda b: b * num_draft,  # element-wise comparison
        })

        return operators
