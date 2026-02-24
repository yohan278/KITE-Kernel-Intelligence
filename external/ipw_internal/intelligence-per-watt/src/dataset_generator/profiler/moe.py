"""MoE operator profiler for router, expert dispatch, and combine operations."""

from __future__ import annotations

from typing import Any, Dict, List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class MoEProfiler(BaseOperatorProfiler):
    """Profiles MoE routing and expert execution.

    Three operator types:
    - Router: top-k gating network (linear projection + softmax + top-k)
    - Expert dispatch: gather/scatter tokens to experts + per-expert MLP
    - Combine: reduce expert outputs back to token space
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.MOE_ROUTING

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes", "prefill_seq_lengths"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile MoE operators across sweep dimensions."""
        import torch

        num_experts = model_spec.num_experts or 8
        experts_per_token = model_spec.experts_per_token or 2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if device == "cuda" else torch.float32)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        measurements: List[OperatorMeasurement] = []
        h = model_spec.hidden_dim
        inter = model_spec.intermediate_dim

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]
            total_tokens = batch_size * seq_len

            # --- Router ---
            measurements.extend(
                self._profile_router(
                    harness, device, dtype, h, num_experts, experts_per_token,
                    batch_size, seq_len, total_tokens,
                )
            )

            # --- Expert Dispatch + MLP ---
            measurements.extend(
                self._profile_expert_dispatch(
                    harness, device, dtype, h, inter, num_experts, experts_per_token,
                    batch_size, seq_len, total_tokens,
                )
            )

            # --- Combine ---
            measurements.extend(
                self._profile_combine(
                    harness, device, dtype, h, num_experts, experts_per_token,
                    batch_size, seq_len, total_tokens,
                )
            )

            # --- Expert dispatch/combine ops ---
            measurements.extend(
                self._profile_expert_dispatch_ops(
                    harness, device, dtype, h, num_experts, experts_per_token,
                    batch_size, seq_len, total_tokens,
                )
            )

            # --- Auxiliary ops ---
            measurements.extend(
                self._profile_aux_ops(
                    harness, device, dtype, h, inter, num_experts, experts_per_token,
                    batch_size, seq_len, total_tokens,
                )
            )

        return measurements

    def _profile_router(
        self,
        harness: MeasurementHarness,
        device: str,
        dtype: Any,
        hidden_dim: int,
        num_experts: int,
        experts_per_token: int,
        batch_size: int,
        seq_len: int,
        total_tokens: int,
    ) -> List[OperatorMeasurement]:
        """Profile the router: linear projection + softmax + top-k."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        measurements = []

        try:
            gate = nn.Linear(hidden_dim, num_experts, bias=False).to(device=device, dtype=dtype)
            x = torch.randn(total_tokens, hidden_dim, device=device, dtype=dtype)

            def router_fn(_x=x, _gate=gate, _k=experts_per_token):
                logits = _gate(_x)
                probs = F.softmax(logits, dim=-1)
                top_k_weights, top_k_indices = torch.topk(probs, _k, dim=-1)
                return top_k_weights, top_k_indices

            # FLOPs: linear (2*T*H*E) + softmax (~3*T*E) + top-k (~T*E*log(E))
            flops = 2 * total_tokens * hidden_dim * num_experts + 4 * total_tokens * num_experts

            measurement = harness.measure(
                router_fn,
                operator_name="moe_router",
                category=OperatorCategory.MOE_ROUTING,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=flops,
            )
            measurement.metadata["num_experts"] = num_experts
            measurement.metadata["experts_per_token"] = experts_per_token
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        return measurements

    def _profile_expert_dispatch(
        self,
        harness: MeasurementHarness,
        device: str,
        dtype: Any,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        experts_per_token: int,
        batch_size: int,
        seq_len: int,
        total_tokens: int,
    ) -> List[OperatorMeasurement]:
        """Profile expert dispatch: gather tokens + per-expert MLP forward."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        measurements = []

        try:
            # Create expert MLPs (SwiGLU pattern: up + gate + down)
            expert_up = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(device=device, dtype=dtype)
            expert_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(device=device, dtype=dtype)
            expert_down = nn.Linear(intermediate_dim, hidden_dim, bias=False).to(device=device, dtype=dtype)

            # Simulate dispatched tokens (each token goes to experts_per_token experts)
            tokens_per_expert = (total_tokens * experts_per_token) // num_experts
            tokens_per_expert = max(tokens_per_expert, 1)
            x = torch.randn(tokens_per_expert, hidden_dim, device=device, dtype=dtype)

            def expert_fn(_x=x, _up=expert_up, _gate=expert_gate, _down=expert_down):
                up_out = _up(_x)
                gate_out = F.silu(_gate(_x))
                return _down(up_out * gate_out)

            # FLOPs per expert: 3 linear ops * 2 * tokens * hidden * inter
            flops_per_expert = 3 * 2 * tokens_per_expert * hidden_dim * intermediate_dim
            # Total across all experts (parallel, but we measure one)
            flops = flops_per_expert

            measurement = harness.measure(
                expert_fn,
                operator_name="moe_expert_mlp",
                category=OperatorCategory.MOE_EXPERT,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=flops,
            )
            measurement.metadata["num_experts"] = num_experts
            measurement.metadata["tokens_per_expert"] = tokens_per_expert
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        return measurements

    def _profile_combine(
        self,
        harness: MeasurementHarness,
        device: str,
        dtype: Any,
        hidden_dim: int,
        num_experts: int,
        experts_per_token: int,
        batch_size: int,
        seq_len: int,
        total_tokens: int,
    ) -> List[OperatorMeasurement]:
        """Profile combine: weighted reduction of expert outputs."""
        import torch

        measurements = []

        try:
            # Expert outputs: (total_tokens, experts_per_token, hidden_dim)
            expert_outputs = torch.randn(
                total_tokens, experts_per_token, hidden_dim,
                device=device, dtype=dtype,
            )
            weights = torch.randn(
                total_tokens, experts_per_token, 1,
                device=device, dtype=dtype,
            )
            weights = torch.softmax(weights, dim=1)

            def combine_fn(_eo=expert_outputs, _w=weights):
                return (_eo * _w).sum(dim=1)

            # FLOPs: multiply + sum over experts
            flops = 2 * total_tokens * experts_per_token * hidden_dim

            measurement = harness.measure(
                combine_fn,
                operator_name="moe_combine",
                category=OperatorCategory.MOE_EXPERT,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=flops,
            )
            measurement.metadata["experts_per_token"] = experts_per_token
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        return measurements

    def _profile_expert_dispatch_ops(
        self,
        harness: MeasurementHarness,
        device: str,
        dtype: Any,
        hidden_dim: int,
        num_experts: int,
        experts_per_token: int,
        batch_size: int,
        seq_len: int,
        total_tokens: int,
    ) -> List[OperatorMeasurement]:
        """Profile expert dispatch (index_select + scatter) and combine (scatter_add)."""
        import torch

        measurements = []

        try:
            # --- expert_dispatch: index_select + scatter ---
            x = torch.randn(total_tokens, hidden_dim, device=device, dtype=dtype)
            # Routing indices: which tokens go to which expert
            routing_indices = torch.randint(0, total_tokens, (total_tokens * experts_per_token,), device=device)
            expert_buffer = torch.zeros(num_experts, max(1, (total_tokens * experts_per_token) // num_experts + 1), hidden_dim, device=device, dtype=dtype)

            def dispatch_fn(_x=x, _idx=routing_indices, _buf=expert_buffer, _ne=num_experts):
                gathered = torch.index_select(_x, 0, _idx)
                # Scatter into expert buffer (simplified: write to first expert)
                _buf[0, :gathered.shape[0], :] = gathered[:_buf.shape[1], :]
                return _buf

            bytes_accessed = total_tokens * experts_per_token * hidden_dim * 2  # fp16

            measurement = harness.measure(
                dispatch_fn,
                operator_name="expert_dispatch",
                category=OperatorCategory.MOE_ROUTING,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=0,
                bytes_accessed=bytes_accessed,
            )
            measurement.metadata["num_experts"] = num_experts
            measurement.metadata["experts_per_token"] = experts_per_token
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        try:
            # --- expert_combine: scatter_add ---
            dispatched_tokens = total_tokens * experts_per_token
            expert_outputs = torch.randn(dispatched_tokens, hidden_dim, device=device, dtype=dtype)
            # Map each dispatched token back to its original token index
            token_indices = torch.arange(total_tokens, device=device).repeat_interleave(experts_per_token)
            index_expand = token_indices.unsqueeze(1).expand(-1, hidden_dim)
            output = torch.zeros(total_tokens, hidden_dim, device=device, dtype=dtype)

            def combine_scatter_fn(_out=output.clone(), _src=expert_outputs, _idx=index_expand):
                return _out.scatter_add_(0, _idx, _src)

            bytes_accessed = dispatched_tokens * hidden_dim * 2

            measurement = harness.measure(
                combine_scatter_fn,
                operator_name="expert_combine",
                category=OperatorCategory.MOE_EXPERT,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=0,
                bytes_accessed=bytes_accessed,
            )
            measurement.metadata["num_experts"] = num_experts
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        return measurements

    def _profile_aux_ops(
        self,
        harness: MeasurementHarness,
        device: str,
        dtype: Any,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        experts_per_token: int,
        batch_size: int,
        seq_len: int,
        total_tokens: int,
    ) -> List[OperatorMeasurement]:
        """Profile auxiliary MoE ops: load balancing loss, capacity factor, shared expert."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        measurements = []

        try:
            # --- load_balancing_loss ---
            router_probs = torch.randn(total_tokens, num_experts, device=device, dtype=dtype)
            router_probs = torch.softmax(router_probs, dim=-1)
            # Top-k routing to get counts
            _, top_indices = torch.topk(router_probs, experts_per_token, dim=-1)

            def lb_loss_fn(_probs=router_probs, _indices=top_indices, _ne=num_experts, _tt=total_tokens):
                # Fraction of tokens routed to each expert
                one_hot = torch.zeros(_tt, _ne, device=_probs.device, dtype=_probs.dtype)
                one_hot.scatter_(1, _indices, 1.0)
                counts = one_hot.sum(dim=0)
                fraction = counts / _tt
                # Mean router probability per expert
                mean_probs = _probs.mean(dim=0)
                return (fraction * mean_probs).sum() * _ne

            flops_lb = total_tokens * num_experts * 3  # scatter + sum + multiply

            measurement = harness.measure(
                lb_loss_fn,
                operator_name="load_balancing_loss",
                category=OperatorCategory.MOE_ROUTING,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=flops_lb,
            )
            measurement.metadata["num_experts"] = num_experts
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        try:
            # --- capacity_factor_overhead ---
            tokens_per_expert = max(1, (total_tokens * experts_per_token) // num_experts)
            capacity = int(1.25 * tokens_per_expert)

            def capacity_fn(_ne=num_experts, _cap=capacity, _h=hidden_dim, _dev=device, _dt=dtype):
                buf = torch.zeros(_ne, _cap, _h, device=_dev, dtype=_dt)
                buf.fill_(0.0)
                return buf

            bytes_accessed = num_experts * capacity * hidden_dim * 2

            measurement = harness.measure(
                capacity_fn,
                operator_name="capacity_factor_overhead",
                category=OperatorCategory.MOE_EXPERT,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=0,
                bytes_accessed=bytes_accessed,
            )
            measurement.metadata["num_experts"] = num_experts
            measurement.metadata["capacity_factor"] = 1.25
            measurement.metadata["tokens_per_expert"] = tokens_per_expert
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        try:
            # --- shared_expert_mlp ---
            # Same SwiGLU as expert_mlp but on ALL tokens
            shared_up = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(device=device, dtype=dtype)
            shared_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False).to(device=device, dtype=dtype)
            shared_down = nn.Linear(intermediate_dim, hidden_dim, bias=False).to(device=device, dtype=dtype)
            x = torch.randn(total_tokens, hidden_dim, device=device, dtype=dtype)

            def shared_expert_fn(_x=x, _up=shared_up, _gate=shared_gate, _down=shared_down):
                up_out = _up(_x)
                gate_out = F.silu(_gate(_x))
                return _down(up_out * gate_out)

            flops_shared = 3 * 2 * total_tokens * hidden_dim * intermediate_dim

            measurement = harness.measure(
                shared_expert_fn,
                operator_name="shared_expert_mlp",
                category=OperatorCategory.MOE_EXPERT,
                batch_size=batch_size,
                seq_len=seq_len,
                flops=flops_shared,
            )
            measurement.metadata["num_experts"] = num_experts
            measurement.metadata["shared"] = True
            measurements.append(measurement)

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pass

        return measurements
