"""Sampling operator profiler for decode-time token selection."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class SamplingProfiler(BaseOperatorProfiler):
    """Profiles sampling operators used during decode.

    Operators work on logit tensors of shape (batch_size, vocab_size)
    with seq_len=1 since sampling happens at decode time.
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.SAMPLING

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile all sampling operators across batch sizes."""
        import torch

        measurements: List[OperatorMeasurement] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        vocab = model_spec.vocab_size
        operators = self._build_operators(model_spec, device, dtype)

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]
            seq_len = 1  # decode time

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
        """Build sampling operator definitions."""
        import torch
        import torch.nn.functional as F

        vocab = model_spec.vocab_size
        operators = []

        # --- Temperature Scaling ---
        temperature = 0.7
        operators.append({
            "name": "temperature_scaling",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": lambda x, _t=temperature: x / _t,
            "input_fn": lambda b: torch.randn(b, vocab, device=device, dtype=dtype),
            "flops_fn": lambda b: b * vocab,
        })

        # --- Top-p (nucleus) filtering ---
        def _make_top_p_fn():
            p = 0.9

            def forward_fn(logits):
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs > p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                return sorted_probs

            return forward_fn

        operators.append({
            "name": "top_p_filter",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": _make_top_p_fn(),
            "input_fn": lambda b: torch.randn(b, vocab, device=device, dtype=dtype),
            # softmax + sort + cumsum + mask
            "flops_fn": lambda b: 5 * b * vocab,
        })

        # --- Top-k filtering ---
        k = 50

        def _make_top_k_fn():
            def forward_fn(logits):
                top_k_vals, top_k_idx = torch.topk(logits, k, dim=-1)
                out = torch.full_like(logits, float("-inf"))
                out.scatter_(1, top_k_idx, top_k_vals)
                return out

            return forward_fn

        operators.append({
            "name": "top_k_filter",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": _make_top_k_fn(),
            "input_fn": lambda b: torch.randn(b, vocab, device=device, dtype=dtype),
            "flops_fn": lambda b: b * vocab,  # topk + scatter
        })

        # --- Repetition penalty ---
        def _make_repetition_penalty_fn():
            penalty = 1.2
            num_prev_tokens = 64
            _cache = {}

            def input_fn(b):
                _cache["prev_tokens"] = torch.randint(
                    0, vocab, (b, num_prev_tokens), device=device
                )
                return torch.randn(b, vocab, device=device, dtype=dtype)

            def forward_fn(logits):
                prev = _cache["prev_tokens"]
                scores = torch.gather(logits, 1, prev)
                scores = torch.where(scores > 0, scores / penalty, scores * penalty)
                logits.scatter_(1, prev, scores)
                return logits

            return input_fn, forward_fn

        rep_input_fn, rep_forward_fn = _make_repetition_penalty_fn()
        operators.append({
            "name": "repetition_penalty",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": rep_forward_fn,
            "input_fn": rep_input_fn,
            "flops_fn": lambda b: 3 * b * 64,  # gather + compare + scatter
        })

        # --- Logit processor chain (temperature + top_k + top_p) ---
        def _make_chain_fn():
            _temp = 0.7
            _k = 50
            _p = 0.9

            def forward_fn(logits):
                # Temperature
                logits = logits / _temp
                # Top-k
                top_k_vals, top_k_idx = torch.topk(logits, _k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, top_k_idx, top_k_vals)
                # Top-p
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative - sorted_probs > _p
                sorted_probs[mask] = 0.0
                return sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            return forward_fn

        operators.append({
            "name": "logit_processor_chain",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": _make_chain_fn(),
            "input_fn": lambda b: torch.randn(b, vocab, device=device, dtype=dtype),
            "flops_fn": lambda b: 7 * b * vocab,  # combined ops
        })

        # --- Multinomial sampling ---
        def _make_multinomial_fn():
            def forward_fn(logits):
                probs = F.softmax(logits, dim=-1).float()
                return torch.multinomial(probs, 1)

            return forward_fn

        operators.append({
            "name": "multinomial_sample",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": _make_multinomial_fn(),
            "input_fn": lambda b: torch.randn(b, vocab, device=device, dtype=dtype),
            "flops_fn": lambda b: 3 * b * vocab,  # softmax + sample
        })

        # --- Beam search step ---
        beam_width = 4

        def _make_beam_fn():
            def forward_fn(logits):
                b = logits.shape[0]
                probs = F.log_softmax(logits, dim=-1)
                # Expand beams: treat batch as beam groups
                # For each item, select top beam_width candidates
                top_scores, top_indices = torch.topk(
                    probs, beam_width, dim=-1
                )
                return top_scores, top_indices

            return forward_fn

        operators.append({
            "name": "beam_search_step",
            "category": OperatorCategory.SAMPLING,
            "forward_fn": _make_beam_fn(),
            "input_fn": lambda b: torch.randn(b, vocab, device=device, dtype=dtype),
            "flops_fn": lambda b: 3 * b * vocab + b * vocab,  # log_softmax + topk
        })

        return operators
