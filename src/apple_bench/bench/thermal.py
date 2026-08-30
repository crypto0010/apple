"""30-min sustained throughput + thermal logger.

Runs a Runner continuously for ``duration_minutes`` and emits a CSV with
one row per sampling window: timestamp, inferences in window, FPS,
junction temperature (tj), and GR3D clock frequency. The interesting
plot (deferred to Plan A.3) is FPS vs time — looking for the moment
thermal throttling kicks in on the Orin Nano.
"""

from __future__ import annotations

import csv
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from apple_bench.runners.base import Runner

GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%@\[?(\d+)\]?")
TJ_RE = re.compile(r"tj@([0-9.]+)C")


def sustained_run(
    runner: Runner,
    sample_input: np.ndarray,
    duration_minutes: int,
    output_csv: Path,
    sample_period_s: float = 60.0,
) -> None:
    """Run inference continuously, sampling tj + FPS every sample_period_s."""
    end_time = time.time() + duration_minutes * 60
    runner.warmup(sample_input, n=10)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_seconds", "infs_in_window", "fps", "tj_C", "gr3d_freq_pct"])

        window_start = time.time()
        next_sample = window_start + sample_period_s
        infs = 0
        while time.time() < end_time:
            runner.run(sample_input)
            infs += 1
            now = time.time()
            if now >= next_sample:
                tj, gr3d = _read_tegrastats_oneshot()
                fps = infs / (now - window_start)
                w.writerow([round(now - window_start), infs, round(fps, 2), tj, gr3d])
                f.flush()
                infs = 0
                window_start = now
                next_sample = now + sample_period_s


def _read_tegrastats_oneshot() -> tuple[float, int]:
    """Spawn tegrastats, read one line, terminate. Returns (tj_C, gr3d_pct).

    tegrastats on JetPack 6 doesn't support ``--count``; the only way to
    get a single reading is to spawn the stream and stop after one line.
    """
    proc = subprocess.Popen(
        ["tegrastats", "--interval", "200"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    tj_m = TJ_RE.search(line)
    gr3d_m = GR3D_RE.search(line)
    tj = float(tj_m.group(1)) if tj_m else float("nan")
    gr3d = int(gr3d_m.group(1)) if gr3d_m else 0
    return tj, gr3d
