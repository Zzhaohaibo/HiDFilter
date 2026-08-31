from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuUtilization:
    samples: int
    mean_percent: float | None
    max_percent: int | None


class NvidiaSmiSampler:
    """External GPU-utilization sampling with one long-lived nvidia-smi process."""

    def __init__(self, device_index: int = 0, interval_ms: int = 500) -> None:
        self.command = [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ]
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> GpuUtilization:
        if self._process is None:
            return GpuUtilization(samples=0, mean_percent=None, max_percent=None)
        self._process.terminate()
        try:
            stdout, _ = self._process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout, _ = self._process.communicate()
        values = []
        for line in stdout.splitlines():
            try:
                values.append(int(line.strip()))
            except ValueError:
                continue
        if not values:
            return GpuUtilization(samples=0, mean_percent=None, max_percent=None)
        return GpuUtilization(
            samples=len(values),
            mean_percent=sum(values) / len(values),
            max_percent=max(values),
        )
