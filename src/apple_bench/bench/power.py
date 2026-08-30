"""tegrastats-based power measurement via the Orin's INA3221 readings.

A background thread parses ``tegrastats --interval N`` lines, extracts the
VDD_IN rail (whole-board input power), and exposes a mean over the sampled
window. The Orin's built-in INA3221 makes external power sensors unnecessary.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass

VDD_IN_RE = re.compile(r"VDD_IN (\d+)mW")


@dataclass
class PowerSample:
    timestamp: float
    vdd_in_mw: int


class TegraPowerLogger:
    """Background-thread logger that captures VDD_IN at fixed intervals."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: list[PowerSample] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = VDD_IN_RE.search(line)
            if m:
                self.samples.append(PowerSample(time.time(), int(m.group(1))))

    def mean_power_mw(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.vdd_in_mw for s in self.samples) / len(self.samples)
