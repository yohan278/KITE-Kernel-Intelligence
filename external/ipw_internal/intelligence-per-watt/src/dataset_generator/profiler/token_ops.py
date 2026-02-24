"""Token-level operator profiler for transformer building blocks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class TokenOpProfiler(BaseOperatorProfiler):
    """Profiles token-level operators: linear projections, norms, activations, embeddings.

    Creates isolated PyTorch modules from ModelSpec dimensions and sweeps
    across batch_size x seq_len combinations.
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.LINEAR

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes", "prefill_seq_lengths"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile all token-level operators across sweep dimensions."""
        import torch

        measurements: List[OperatorMeasurement] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        # Define operators with their shapes and FLOPs formulas
        operators = self._build_operators(model_spec, device, dtype)

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]

            for op_def in operators:
                try:
                    input_tensor = op_def["input_fn"](batch_size, seq_len)
                    flops = op_def["flops_fn"](batch_size, seq_len)
                    bytes_accessed = op_def.get("bytes_fn", lambda b, s: None)(
                        batch_size, seq_len
                    )

                    measurement = harness.measure(
                        op_def["forward_fn"],
                        input_tensor,
                        operator_name=op_def["name"],
                        category=op_def["category"],
                        batch_size=batch_size,
                        seq_len=seq_len,
                        flops=flops,
                        bytes_accessed=bytes_accessed,
                    )
                    measurements.append(measurement)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    # Skip OOM configurations
                    continue

        return measurements

    def _build_operators(
        self, model_spec: ModelSpec, device: str, dtype: Any
    ) -> List[Dict[str, Any]]:
        """Build isolated PyTorch modules for each operator."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        h = model_spec.hidden_dim
        nh = model_spec.num_attention_heads
        nkv = model_spec.num_kv_heads
        hd = model_spec.head_dim
        inter = model_spec.intermediate_dim
        vocab = model_spec.vocab_size

        operators = []

        # --- Linear projections ---

        # QKV projection: hidden_dim -> (num_heads + 2*num_kv_heads) * head_dim
        qkv_out = (nh + 2 * nkv) * hd
        qkv_linear = nn.Linear(h, qkv_out, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "linear_qkv",
            "category": OperatorCategory.LINEAR,
            "forward_fn": lambda x, _m=qkv_linear: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 2 * b * s * h * qkv_out,
        })

        # Output projection: num_heads * head_dim -> hidden_dim
        o_in = nh * hd
        o_linear = nn.Linear(o_in, h, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "linear_o",
            "category": OperatorCategory.LINEAR,
            "forward_fn": lambda x, _m=o_linear: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, o_in, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 2 * b * s * o_in * h,
        })

        # MLP up projection: hidden_dim -> intermediate_dim
        mlp_up = nn.Linear(h, inter, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "mlp_up",
            "category": OperatorCategory.LINEAR,
            "forward_fn": lambda x, _m=mlp_up: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 2 * b * s * h * inter,
        })

        # MLP gate projection: hidden_dim -> intermediate_dim
        mlp_gate = nn.Linear(h, inter, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "mlp_gate",
            "category": OperatorCategory.LINEAR,
            "forward_fn": lambda x, _m=mlp_gate: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 2 * b * s * h * inter,
        })

        # MLP down projection: intermediate_dim -> hidden_dim
        mlp_down = nn.Linear(inter, h, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "mlp_down",
            "category": OperatorCategory.LINEAR,
            "forward_fn": lambda x, _m=mlp_down: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, inter, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 2 * b * s * inter * h,
        })

        # --- Normalization ---
        # Use manual RMSNorm since nn.RMSNorm may not exist in all torch versions
        rms_weight = torch.ones(h, device=device, dtype=dtype)

        def rmsnorm_fn(x, _w=rms_weight):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6)
            return x * _w

        operators.append({
            "name": "rmsnorm",
            "category": OperatorCategory.NORMALIZATION,
            "forward_fn": rmsnorm_fn,
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 5 * b * s * h,  # mean, rsqrt, mul
        })

        # --- Activation ---
        def silu_fn(x):
            return F.silu(x)

        operators.append({
            "name": "silu_activation",
            "category": OperatorCategory.ACTIVATION,
            "forward_fn": silu_fn,
            "input_fn": lambda b, s: torch.randn(
                b, s, inter, device=device, dtype=dtype
            ),
            "flops_fn": lambda b, s: 4 * b * s * inter,  # sigmoid + mul
        })

        # --- Embedding ---
        embedding = nn.Embedding(vocab, h).to(device=device)

        def embed_fn(x, _m=embedding):
            return _m(x)

        operators.append({
            "name": "embedding",
            "category": OperatorCategory.EMBEDDING,
            "forward_fn": embed_fn,
            "input_fn": lambda b, s: torch.randint(
                0, vocab, (b, s), device=device
            ),
            "flops_fn": lambda b, s: 0,  # Lookup, no FLOPs
            "bytes_fn": lambda b, s: b * s * h * 2,  # fp16 bytes per lookup
        })

        # --- LM Head ---
        lm_head = nn.Linear(h, vocab, bias=False).to(device=device, dtype=dtype)
        operators.append({
            "name": "lm_head",
            "category": OperatorCategory.LINEAR,
            "forward_fn": lambda x, _m=lm_head: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 2 * b * s * h * vocab,
        })

        # --- LayerNorm ---
        layer_norm = nn.LayerNorm(h).to(device=device, dtype=dtype)
        operators.append({
            "name": "layernorm",
            "category": OperatorCategory.NORMALIZATION,
            "forward_fn": lambda x, _m=layer_norm: _m(x),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 5 * b * s * h,
        })

        # --- GeLU Activation ---
        operators.append({
            "name": "gelu_activation",
            "category": OperatorCategory.ACTIVATION,
            "forward_fn": lambda x: F.gelu(x),
            "input_fn": lambda b, s: torch.randn(b, s, inter, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 4 * b * s * inter,
        })

        # --- Residual Add ---
        # Closure captures a second random tensor; forward_fn receives only the first
        def _make_residual_fn():
            _cache = {}

            def input_fn(b, s):
                _cache["y"] = torch.randn(b, s, h, device=device, dtype=dtype)
                return torch.randn(b, s, h, device=device, dtype=dtype)

            def forward_fn(x):
                return x + _cache["y"]

            return input_fn, forward_fn

        res_input_fn, res_forward_fn = _make_residual_fn()
        operators.append({
            "name": "residual_add",
            "category": OperatorCategory.LINEAR,
            "forward_fn": res_forward_fn,
            "input_fn": res_input_fn,
            "flops_fn": lambda b, s: b * s * h,
        })

        # --- Rotary Embedding (RoPE) ---
        def _make_rope_fn():
            def input_fn(b, s):
                return torch.randn(b, s, nh, hd, device=device, dtype=dtype)

            def forward_fn(x):
                _b, _s, _nh, _hd = x.shape
                half = _hd // 2
                freqs = torch.arange(half, device=x.device, dtype=torch.float32)
                freqs = 1.0 / (10000.0 ** (freqs / half))
                positions = torch.arange(_s, device=x.device, dtype=torch.float32)
                angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (s, half)
                # Reshape to (1, s, 1, half) for broadcasting against (b, s, nh, half)
                cos_vals = torch.cos(angles).to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
                sin_vals = torch.sin(angles).to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
                x1 = x[..., :half]
                x2 = x[..., half:]
                out1 = x1 * cos_vals + x2 * sin_vals
                out2 = -x1 * sin_vals + x2 * cos_vals
                return torch.cat([out1, out2], dim=-1)

            return input_fn, forward_fn

        rope_input_fn, rope_forward_fn = _make_rope_fn()
        operators.append({
            "name": "rotary_embedding",
            "category": OperatorCategory.LINEAR,
            "forward_fn": rope_forward_fn,
            "input_fn": rope_input_fn,
            "flops_fn": lambda b, s: 6 * b * s * nh * hd,
        })

        # --- Softmax ---
        operators.append({
            "name": "softmax",
            "category": OperatorCategory.ACTIVATION,
            "forward_fn": lambda x: F.softmax(x, dim=-1),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: 3 * b * s * h,
        })

        # --- Dropout ---
        operators.append({
            "name": "dropout",
            "category": OperatorCategory.ACTIVATION,
            "forward_fn": lambda x: F.dropout(x, p=0.1, training=True),
            "input_fn": lambda b, s: torch.randn(b, s, h, device=device, dtype=dtype),
            "flops_fn": lambda b, s: b * s * h,
        })

        # --- Cross Entropy Loss ---
        def _make_ce_fn():
            _cache = {}

            def input_fn(b, s):
                _cache["targets"] = torch.randint(0, vocab, (b * s,), device=device)
                return torch.randn(b * s, vocab, device=device, dtype=dtype)

            def forward_fn(logits):
                return F.cross_entropy(logits, _cache["targets"])

            return input_fn, forward_fn

        ce_input_fn, ce_forward_fn = _make_ce_fn()
        operators.append({
            "name": "cross_entropy_loss",
            "category": OperatorCategory.LINEAR,
            "forward_fn": ce_forward_fn,
            "input_fn": ce_input_fn,
            "flops_fn": lambda b, s: b * s * vocab,
        })

        return operators
