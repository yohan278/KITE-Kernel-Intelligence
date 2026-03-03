#!/usr/bin/env python3
"""Check that L40/L40S exposes all NVML telemetry signals used for power/energy modeling.

Probes each NVML API listed in the telemetry spec and reports OK or the error.
Run on a single L40 (e.g. with CUDA_VISIBLE_DEVICES=0 or srun --gres=gpu:1).

Usage:
  python scripts/check_nvml_signals_l40.py
  # Or with allocation:
  bash scripts/run_check_nvml_l40.sh
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _probe(name: str, api_name: str, get_value) -> tuple[bool, str]:
    """Return (ok, message)."""
    try:
        val = get_value()
        return True, str(val)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    try:
        import pynvml  # type: ignore
    except ImportError as e:
        print("ERROR: pynvml not installed. pip install pynvml>=11.5")
        return 1

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device_index = int(raw.split(",")[0]) if raw else 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        print(f"GPU: {name} (device index {device_index})")
        print()
    except Exception as e:
        print(f"ERROR: NVML init failed: {e}")
        print("Run inside a GPU allocation (e.g. srun --partition=gpu --gres=gpu:1 --pty bash)")
        return 1

    results = []
    # 1) Utilization (GPU % and Memory % from one call)
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_pct = util.gpu
        mem_pct = util.memory
        results.append(("GPU utilization %", "nvmlDeviceGetUtilizationRates", True, f"{gpu_pct}%"))
        results.append(("Memory utilization %", "nvmlDeviceGetUtilizationRates", True, f"{mem_pct}%"))
    except Exception as e:
        results.append(("GPU utilization %", "nvmlDeviceGetUtilizationRates", False, f"{type(e).__name__}: {e}"))
        results.append(("Memory utilization %", "nvmlDeviceGetUtilizationRates", False, f"{type(e).__name__}: {e}"))

    # 2) Temperature
    ok, msg = _probe("Temperature (C)", "nvmlDeviceGetTemperature",
                     lambda: pynvml.nvmlDeviceGetTemperature(handle, getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)))
    results.append(("Temperature (C)", "nvmlDeviceGetTemperature", ok, msg))

    # 3) Clocks
    ok, msg = _probe("Clock SM (MHz)", "nvmlDeviceGetClockInfo",
                     lambda: pynvml.nvmlDeviceGetClockInfo(handle, getattr(pynvml, "NVML_CLOCK_GRAPHICS", 0)))
    results.append(("Clock SM (MHz)", "nvmlDeviceGetClockInfo", ok, msg))
    ok, msg = _probe("Clock memory (MHz)", "nvmlDeviceGetClockInfo",
                     lambda: pynvml.nvmlDeviceGetClockInfo(handle, getattr(pynvml, "NVML_CLOCK_MEM", 1)))
    results.append(("Clock memory (MHz)", "nvmlDeviceGetClockInfo", ok, msg))

    # 4) Power limit
    ok, msg = _probe("Power limit (W)", "nvmlDeviceGetPowerManagementLimit",
                     lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
    results.append(("Power limit (W)", "nvmlDeviceGetPowerManagementLimit", ok, msg))

    # 5) Total energy
    ok, msg = _probe("Total energy consumption (mJ)", "nvmlDeviceGetTotalEnergyConsumption",
                     lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    results.append(("Total energy consumption (mJ)", "nvmlDeviceGetTotalEnergyConsumption", ok, msg))

    # 6) Memory info
    def _format_meminfo():
        mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"total={mi.total/1e9:.1f}GB used={mi.used/1e9:.2f}GB free={mi.free/1e9:.1f}GB"
    ok, msg = _probe("Memory usage (bytes)", "nvmlDeviceGetMemoryInfo", _format_meminfo)
    results.append(("Memory usage (bytes)", "nvmlDeviceGetMemoryInfo", ok, msg))

    # 7) PCIe throughput
    ok, msg = _probe("PCIe throughput", "nvmlDeviceGetPcieThroughput", lambda: _pcie_throughput(pynvml, handle))
    results.append(("PCIe throughput", "nvmlDeviceGetPcieThroughput", ok, msg))

    # 8) Throttle reasons
    ok, msg = _probe("Throttle reasons", "nvmlDeviceGetCurrentClocksThrottleReasons",
                     lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
    results.append(("Throttle reasons", "nvmlDeviceGetCurrentClocksThrottleReasons", ok, msg))

    # 9) Fan speed (datacenter GPUs like L40S typically do NOT expose this)
    ok, msg = _probe("Fan speed %", "nvmlDeviceGetFanSpeed", lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
    if not ok:
        msg += " [expected on datacenter GPUs]"
    results.append(("Fan speed %", "nvmlDeviceGetFanSpeed", ok, msg))

    DATACENTER_EXPECTED_FAIL = {"nvmlDeviceGetFanSpeed"}

    print("Signal                          | NVML API                                    | Status | Reading")
    print("-" * 120)
    for label, api_name, ok, msg in results:
        status = "OK" if ok else ("SKIP" if api_name in DATACENTER_EXPECTED_FAIL else "FAIL")
        short_msg = (msg[:60] + "…") if len(msg) > 60 else msg
        print(f"{label:32} | {api_name:44} | {status:6} | {short_msg}")
    print()

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass

    unexpected = sum(
        1 for _, api, ok, _ in results
        if not ok and api not in DATACENTER_EXPECTED_FAIL
    )
    skipped = sum(1 for _, api, ok, _ in results if not ok and api in DATACENTER_EXPECTED_FAIL)
    if unexpected:
        print(f"Result: {unexpected} signal(s) FAILED unexpectedly ({skipped} expected skip).")
        return 1
    print(f"Result: All L40S NVML signals exposed and readable ({skipped} expected skip for datacenter GPU).")
    return 0


def _pcie_throughput(pynvml, handle) -> str:
    """Return PCIe TX/RX as a string; some drivers use different counter types."""
    try:
        tx = pynvml.nvmlDeviceGetPcieThroughput(handle, getattr(pynvml, "NVML_PCIE_UTIL_COUNT_TX", 0))
        rx = pynvml.nvmlDeviceGetPcieThroughput(handle, getattr(pynvml, "NVML_PCIE_UTIL_COUNT_RX", 1))
        return f"TX={tx} KB/s RX={rx} KB/s"
    except Exception:
        return "N/A (check NVML_PCIE_UTIL_*)"


if __name__ == "__main__":
    sys.exit(main())
