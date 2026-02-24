"""CPU host profiler for PCIe transfers, scheduling, tokenization, and batching."""

from __future__ import annotations

import time
from typing import List

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement
from inference_simulator.types.model_spec import ModelSpec
from inference_simulator.types.hardware_spec import HardwareSpec
from dataset_generator.profiler.base import BaseOperatorProfiler
from dataset_generator.profiler.sweep import SweepConfig
from dataset_generator.profiler.measurement import MeasurementHarness


class CPUHostProfiler(BaseOperatorProfiler):
    """Profiles CPU-side host operations: PCIe transfers, scheduling, tokenization.

    All operators use wall-clock timing (MeasurementHarness._measure_cpu path).
    GPU transfer ops fall back to CPU-only timing when CUDA is unavailable.
    """

    @property
    def category(self) -> OperatorCategory:
        return OperatorCategory.CPU_HOST

    def get_sweep_dimensions(self) -> List[str]:
        return ["batch_sizes", "prefill_seq_lengths"]

    def profile(
        self,
        model_spec: ModelSpec,
        hw_spec: HardwareSpec,
        sweep_config: SweepConfig,
        precision: str = "fp16",
    ) -> List[OperatorMeasurement]:
        """Profile all CPU host operators across sweep dimensions."""
        import torch

        # Force CPU measurement path (wall-clock timing, optional CPU energy via RAPL)
        harness = MeasurementHarness(
            warmup=sweep_config.warmup_iterations,
            iterations=sweep_config.measurement_iterations,
            use_energy=sweep_config.use_energy,
        )
        # Override to always use CPU timing path
        harness._has_cuda = False

        has_cuda = torch.cuda.is_available()
        h = model_spec.hidden_dim
        vocab_size = model_spec.vocab_size
        measurements: List[OperatorMeasurement] = []

        for point in sweep_config.get_sweep_points(self.get_sweep_dimensions()):
            batch_size = point["batch_size"]
            seq_len = point["seq_len"]

            measurements.extend(
                self._profile_gpu_transfer_ops(
                    harness, has_cuda, batch_size, seq_len, h,
                )
            )
            measurements.extend(
                self._profile_cpu_ops(
                    harness, batch_size, seq_len, h, vocab_size,
                )
            )

        return measurements

    def _profile_gpu_transfer_ops(
        self,
        harness: MeasurementHarness,
        has_cuda: bool,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> List[OperatorMeasurement]:
        """Profile GPU-related transfer ops, with CPU fallback."""
        import torch

        measurements = []
        tensor_bytes = batch_size * seq_len * hidden_dim * 2  # fp16

        # --- cpu_offload_transfer ---
        try:
            if has_cuda:
                cpu_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)

                def offload_fn(_t=cpu_tensor):
                    gpu_t = _t.cuda()
                    cpu_t = gpu_t.cpu()
                    torch.cuda.synchronize()
                    return cpu_t
            else:
                cpu_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

                def offload_fn(_t=cpu_tensor):
                    return _t.clone()

            measurement = harness.measure(
                offload_fn,
                operator_name="cpu_offload_transfer",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
                bytes_accessed=tensor_bytes * 2,  # round-trip
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        # --- gpu_mem_alloc ---
        try:
            if has_cuda:
                def alloc_fn(_b=batch_size, _s=seq_len, _h=hidden_dim):
                    t = torch.empty(_b, _s, _h, device="cuda", dtype=torch.float16)
                    torch.cuda.synchronize()
                    return t
            else:
                def alloc_fn(_b=batch_size, _s=seq_len, _h=hidden_dim):
                    return torch.empty(_b, _s, _h, dtype=torch.float32)

            measurement = harness.measure(
                alloc_fn,
                operator_name="gpu_mem_alloc",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
                bytes_accessed=tensor_bytes,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        # --- pcie_h2d_copy ---
        try:
            if has_cuda:
                src = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)

                def h2d_fn(_src=src):
                    gpu_t = _src.cuda()
                    torch.cuda.synchronize()
                    return gpu_t
            else:
                src = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

                def h2d_fn(_src=src):
                    return _src.clone()

            measurement = harness.measure(
                h2d_fn,
                operator_name="pcie_h2d_copy",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
                bytes_accessed=tensor_bytes,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        # --- pcie_d2h_copy ---
        try:
            if has_cuda:
                gpu_src = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

                def d2h_fn(_src=gpu_src):
                    cpu_t = _src.cpu()
                    torch.cuda.synchronize()
                    return cpu_t
            else:
                cpu_src = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

                def d2h_fn(_src=cpu_src):
                    return _src.clone()

            measurement = harness.measure(
                d2h_fn,
                operator_name="pcie_d2h_copy",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
                bytes_accessed=tensor_bytes,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        return measurements

    def _profile_cpu_ops(
        self,
        harness: MeasurementHarness,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        vocab_size: int,
    ) -> List[OperatorMeasurement]:
        """Profile pure CPU operations: scheduling, tokenization, batching."""
        import torch

        measurements = []

        # --- scheduler_overhead ---
        try:
            num_requests = batch_size * 8

            def scheduler_fn(_n=num_requests, _s=seq_len):
                requests = [(i % 5, _s + (i * 7) % 100) for i in range(_n)]
                sorted_requests = sorted(requests, key=lambda x: (-x[0], x[1]))
                # Bin-pack: group by similar sequence length
                bins: list = []
                for priority, slen in sorted_requests:
                    placed = False
                    for b in bins:
                        if len(b) < 8 and abs(b[-1][1] - slen) < 64:
                            b.append((priority, slen))
                            placed = True
                            break
                    if not placed:
                        bins.append([(priority, slen)])
                return bins

            measurement = harness.measure(
                scheduler_fn,
                operator_name="scheduler_overhead",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        # --- tokenizer_encode ---
        try:
            text = " ".join(f"word{i}" for i in range(seq_len))

            def encode_fn(_text=text, _vocab=vocab_size):
                words = _text.split()
                return [hash(w) % _vocab for w in words]

            measurement = harness.measure(
                encode_fn,
                operator_name="tokenizer_encode",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        # --- tokenizer_decode ---
        try:
            tokens = list(range(seq_len))

            def decode_fn(_tokens=tokens):
                words = [str(t) for t in _tokens]
                return " ".join(words)

            measurement = harness.measure(
                decode_fn,
                operator_name="tokenizer_decode",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        # --- dynamic_batching_overhead ---
        try:
            # Create variable-length sequences
            sequences = [
                torch.randn(max(1, seq_len - (i * 7) % (seq_len // 2 + 1)), hidden_dim)
                for i in range(batch_size)
            ]

            def batch_fn(_seqs=sequences):
                max_len = max(s.shape[0] for s in _seqs)
                padded = torch.zeros(len(_seqs), max_len, _seqs[0].shape[1])
                masks = torch.zeros(len(_seqs), max_len, dtype=torch.bool)
                for i, s in enumerate(_seqs):
                    padded[i, :s.shape[0], :] = s
                    masks[i, :s.shape[0]] = True
                return padded, masks

            measurement = harness.measure(
                batch_fn,
                operator_name="dynamic_batching_overhead",
                category=OperatorCategory.CPU_HOST,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            measurements.append(measurement)

        except (RuntimeError,):
            pass

        return measurements
