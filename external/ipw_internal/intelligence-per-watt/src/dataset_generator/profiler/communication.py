"""Inter-GPU communication profiler for AllReduce and AllGather operations."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class CommunicationProfiler(BaseOperatorProfiler):
    """Profiles inter-GPU communication: AllReduce, AllGather, etc.

    Measures bandwidth and latency for collective operations across
    GPU topologies using NCCL. Requires multi-GPU hardware and
    torch.distributed to be available.
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.COMMUNICATION

    def get_sweep_dimensions(self) -> List[str]:
        return ["message_sizes_bytes", "gpu_topologies"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile communication operators across message sizes and GPU counts."""
        import torch

        if not torch.cuda.is_available():
            return self._profile_analytical(model_spec, hw_spec, sweep_config)

        num_gpus_available = torch.cuda.device_count()
        if num_gpus_available < 2:
            return self._profile_analytical(model_spec, hw_spec, sweep_config)

        return self._profile_nccl(model_spec, hw_spec, sweep_config, num_gpus_available)

    def _profile_nccl(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        num_gpus_available: int,
    ) -> List[OperatorMeasurement]:
        """Profile using actual NCCL collectives via torch.distributed."""
        import torch
        import torch.distributed as dist

        measurements: List[OperatorMeasurement] = []

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            message_size_bytes = point["message_size_bytes"]
            num_gpus = point["num_gpus"]

            if num_gpus > num_gpus_available:
                continue

            # Number of float16 elements for this message size
            num_elements = message_size_bytes // 2  # fp16 = 2 bytes

            if num_elements < 1:
                continue

            for op_name, op_fn_name in [
                ("allreduce", "all_reduce"),
                ("allgather", "all_gather"),
                ("reduce_scatter", "reduce_scatter"),
                ("send_recv_p2p", "send_recv"),
            ]:
                try:
                    measurement = self._measure_collective(
                        op_name=op_name,
                        op_fn_name=op_fn_name,
                        num_elements=num_elements,
                        num_gpus=num_gpus,
                        message_size_bytes=message_size_bytes,
                        warmup=sweep_config.warmup_iterations,
                        iterations=sweep_config.measurement_iterations,
                    )
                    if measurement is not None:
                        measurements.append(measurement)
                except (RuntimeError, Exception):
                    continue

        # Pipeline parallelism ops (measured with harness)
        measurements.extend(
            self._profile_pipeline_ops(model_spec, hw_spec, sweep_config)
        )

        return measurements

    def _measure_collective(
        self,
        op_name: str,
        op_fn_name: str,
        num_elements: int,
        num_gpus: int,
        message_size_bytes: int,
        warmup: int,
        iterations: int,
    ) -> Optional[OperatorMeasurement]:
        """Measure a single collective operation using CUDA events."""
        import torch
        import torch.distributed as dist

        device = torch.device("cuda:0")
        tensor = torch.randn(num_elements, device=device, dtype=torch.float16)

        if not dist.is_initialized():
            # Cannot profile without distributed init -- fall back to analytical
            return self._analytical_measurement(
                op_name, message_size_bytes, num_gpus
            )

        # Warmup
        for _ in range(warmup):
            if op_fn_name == "all_reduce":
                dist.all_reduce(tensor.clone())
            elif op_fn_name == "all_gather":
                output = [torch.empty_like(tensor) for _ in range(num_gpus)]
                dist.all_gather(output, tensor)
            elif op_fn_name == "reduce_scatter":
                input_list = [torch.randn_like(tensor) for _ in range(num_gpus)]
                out = torch.empty_like(tensor)
                dist.reduce_scatter(out, input_list)
            elif op_fn_name == "send_recv":
                if dist.get_rank() == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            torch.cuda.synchronize()

        # Measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times_ms = []

        for _ in range(iterations):
            start_event.record()
            if op_fn_name == "all_reduce":
                dist.all_reduce(tensor.clone())
            elif op_fn_name == "all_gather":
                output = [torch.empty_like(tensor) for _ in range(num_gpus)]
                dist.all_gather(output, tensor)
            elif op_fn_name == "reduce_scatter":
                input_list = [torch.randn_like(tensor) for _ in range(num_gpus)]
                out = torch.empty_like(tensor)
                dist.reduce_scatter(out, input_list)
            elif op_fn_name == "send_recv":
                if dist.get_rank() == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        mean_time_s = sum(times_ms) / len(times_ms) / 1000.0
        bandwidth_gb_s = message_size_bytes / mean_time_s / 1e9 if mean_time_s > 0 else None

        return OperatorMeasurement(
            operator_name=op_name,
            category=OperatorCategory.COMMUNICATION,
            batch_size=num_gpus,
            seq_len=message_size_bytes,
            time_s=mean_time_s,
            bandwidth_gb_s=bandwidth_gb_s,
            bytes_accessed=message_size_bytes,
            metadata={
                "num_gpus": num_gpus,
                "message_size_bytes": message_size_bytes,
                "operation": op_name,
            },
        )

    def _profile_analytical(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
    ) -> List[OperatorMeasurement]:
        """Analytical fallback when multi-GPU profiling is not available.

        Uses the ring AllReduce bandwidth model:
            time = 2 * (n-1)/n * message_bytes / bandwidth
        """
        measurements: List[OperatorMeasurement] = []

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            message_size_bytes = point["message_size_bytes"]
            num_gpus = point["num_gpus"]

            if num_gpus <= 1:
                continue

            for op_name in ["allreduce", "allgather", "reduce_scatter", "send_recv_p2p"]:
                measurement = self._analytical_measurement(
                    op_name, message_size_bytes, num_gpus, hw_spec
                )
                if measurement is not None:
                    measurements.append(measurement)

        # Pipeline parallelism ops (analytical)
        measurements.extend(
            self._profile_pipeline_ops(model_spec, hw_spec, sweep_config)
        )

        return measurements

    def _analytical_measurement(
        self,
        op_name: str,
        message_size_bytes: int,
        num_gpus: int,
        hw_spec: Optional[HardwareSpec] = None,
    ) -> Optional[OperatorMeasurement]:
        """Compute analytical estimate for a collective operation."""
        # Default NVLink bandwidth if not available from hw_spec
        nvlink_bw_gb_s = 900.0  # H100 NVLink 4.0
        if hw_spec is not None and hw_spec.nvlink_bandwidth_gb_s > 0:
            nvlink_bw_gb_s = hw_spec.nvlink_bandwidth_gb_s

        efficiency = 0.85  # Typical NVLink efficiency
        effective_bw_bytes_s = nvlink_bw_gb_s * 1e9 * efficiency

        # Ring AllReduce: 2*(n-1)/n * message_size / bandwidth
        # AllGather: (n-1)/n * message_size / bandwidth
        # ReduceScatter: (n-1)/n * message_size / bandwidth
        # Send/Recv P2P: message_size / bandwidth (direct)
        if op_name == "allreduce":
            ring_factor = 2.0 * (num_gpus - 1) / num_gpus
        elif op_name == "send_recv_p2p":
            ring_factor = 1.0
        else:  # allgather, reduce_scatter
            ring_factor = (num_gpus - 1) / num_gpus

        time_s = ring_factor * message_size_bytes / effective_bw_bytes_s
        bandwidth_gb_s = message_size_bytes / time_s / 1e9 if time_s > 0 else None

        return OperatorMeasurement(
            operator_name=op_name,
            category=OperatorCategory.COMMUNICATION,
            batch_size=num_gpus,
            seq_len=message_size_bytes,
            time_s=time_s,
            bandwidth_gb_s=bandwidth_gb_s,
            bytes_accessed=message_size_bytes,
            metadata={
                "num_gpus": num_gpus,
                "message_size_bytes": message_size_bytes,
                "operation": op_name,
                "analytical": True,
            },
        )

    def _profile_pipeline_ops(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
    ) -> List[OperatorMeasurement]:
        """Profile pipeline parallelism ops: bubble idle, tensor split, stage forward."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )

        measurements: List[OperatorMeasurement] = []
        h = model_spec.hidden_dim

        nvlink_bw_gb_s = 900.0
        if hw_spec.nvlink_bandwidth_gb_s > 0:
            nvlink_bw_gb_s = hw_spec.nvlink_bandwidth_gb_s
        efficiency = 0.85
        effective_bw_bytes_s = nvlink_bw_gb_s * 1e9 * efficiency

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            message_size_bytes = point["message_size_bytes"]
            num_gpus = point["num_gpus"]

            if num_gpus <= 1:
                continue

            # --- pipeline_bubble_idle (analytical) ---
            pp_stages = num_gpus
            # Estimate layer time from HBM bandwidth
            layer_bytes = 2 * h * h * 2  # rough: one linear layer weight read, fp16
            layer_time = layer_bytes / effective_bw_bytes_s if effective_bw_bytes_s > 0 else 1e-6
            bubble_time = (pp_stages - 1) / pp_stages * layer_time

            measurements.append(OperatorMeasurement(
                operator_name="pipeline_bubble_idle",
                category=OperatorCategory.COMMUNICATION,
                batch_size=num_gpus,
                seq_len=message_size_bytes,
                time_s=bubble_time,
                bytes_accessed=0,
                metadata={
                    "num_gpus": num_gpus,
                    "pp_stages": pp_stages,
                    "operation": "pipeline_bubble_idle",
                    "analytical": True,
                },
            ))

            # --- tensor_parallel_split ---
            try:
                b = 1
                s = max(1, message_size_bytes // (h * 2))  # derive seq_len from msg size
                x = torch.randn(b, s, h, device=device, dtype=dtype)

                def split_fn(_x=x, _ng=num_gpus):
                    return torch.chunk(_x, _ng, dim=-1)

                measurement = harness.measure(
                    split_fn,
                    operator_name="tensor_parallel_split",
                    category=OperatorCategory.COMMUNICATION,
                    batch_size=num_gpus,
                    seq_len=message_size_bytes,
                    flops=0,
                    bytes_accessed=b * s * h * 2,
                )
                measurement.metadata["num_gpus"] = num_gpus
                measurement.metadata["operation"] = "tensor_parallel_split"
                measurement.metadata["analytical"] = True
                measurements.append(measurement)

            except (RuntimeError,):
                pass

            # --- pipeline_stage_forward ---
            try:
                import torch.nn as nn
                stage_linear = nn.Linear(h, h, bias=False).to(device=device, dtype=dtype)
                stage_input = torch.randn(1, h, device=device, dtype=dtype)

                def stage_fn(_x=stage_input, _lin=stage_linear):
                    return _lin(_x)

                flops_stage = 2 * h * h  # single linear forward

                measurement = harness.measure(
                    stage_fn,
                    operator_name="pipeline_stage_forward",
                    category=OperatorCategory.COMMUNICATION,
                    batch_size=num_gpus,
                    seq_len=message_size_bytes,
                    flops=flops_stage,
                )
                measurement.metadata["num_gpus"] = num_gpus
                measurement.metadata["operation"] = "pipeline_stage_forward"
                measurement.metadata["analytical"] = True
                measurements.append(measurement)

            except (RuntimeError,):
                pass

        return measurements
