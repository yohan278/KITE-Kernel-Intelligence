"""GPU power-cap controller wrapper."""

from __future__ import annotations

import subprocess
from typing import Optional


class PowerCapController:
    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index

    def set_power_cap(self, watts: int) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index), "-pl", str(watts)],
                check=False,
                timeout=5,
                capture_output=True,
            )
            return True
        except Exception:
            return False

    def read_power_cap(self) -> Optional[int]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "--query-gpu=power.limit",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            ).decode("utf-8", errors="ignore").strip()
            return int(float(out.splitlines()[0]))
        except Exception:
            return None

