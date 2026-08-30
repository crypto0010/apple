"""Host-side ESP32-S3-CAM benchmark driver (Plan A.2 Task 18).

Opens the board over USB serial, triggers a hardware reset via DTR/RTS so
the firmware starts a fresh inference loop, reads the per-inference log
lines, and computes latency percentiles into ``runs/full/esp32_results.csv``.

Expected firmware log-line format (see esp32_firmware/src/main.cpp):

    INF<n>: <latency_us>us heap=<bytes> psram=<bytes>

Power note: the ESP32-S3-CAM has no on-board INA-class current monitor and
the Jetson's INA3221 only sees the Jetson's own rails, so MCU power is
not reported here.  Adding an external INA219 on the USB rail is the
right next step if needed.

Usage:
    uv run python scripts/07_bench_esp32.py --model mobilenet_v3_small
    uv run python scripts/07_bench_esp32.py --duration-s 60 --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import serial

from apple_bench.config import RUNS_ROOT

INF_RE = re.compile(
    r"INF(?P<n>\d+):\s*(?P<us>\d+)\s*us\s*heap=(?P<heap>\d+)\s*psram=(?P<psram>\d+)"
)


def collect_latencies(
    port: str,
    duration_s: float,
    min_runs: int,
) -> tuple[list[int], list[int], list[int]]:
    """Read inference log lines until duration_s elapses (with min_runs floor)."""
    ser = serial.Serial(port, 115200, timeout=2.0)
    # Hardware reset so the firmware starts a fresh inference loop.
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    time.sleep(0.4)  # USB CDC re-enumerates briefly after reset
    ser.reset_input_buffer()

    latencies_us: list[int] = []
    heaps: list[int] = []
    psrams: list[int] = []
    end_time = time.time() + duration_s
    hard_deadline = end_time + 30.0  # safety stop if firmware hangs

    try:
        while time.time() < hard_deadline:
            if time.time() > end_time and len(latencies_us) >= min_runs:
                break
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            m = INF_RE.match(line)
            if m:
                latencies_us.append(int(m["us"]))
                heaps.append(int(m["heap"]))
                psrams.append(int(m["psram"]))
    finally:
        ser.close()

    return latencies_us, heaps, psrams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--model",
        default="mobilenet_v3_small",
        help="Name of the model the firmware was flashed with (for the CSV row).",
    )
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--min-runs", type=int, default=200)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RUNS_ROOT / "full" / "esp32_results.csv",
    )
    parser.add_argument(
        "--model-kb",
        type=int,
        default=0,
        help="Embedded .tflite model size in KB (recorded in the CSV row).",
    )
    args = parser.parse_args()

    print(f"[info] reading {args.duration_s}s of inference logs from {args.port}")
    latencies, heaps, psrams = collect_latencies(
        args.port, args.duration_s, args.min_runs
    )
    if not latencies:
        raise RuntimeError(
            f"no INF lines received from {args.port}; "
            "is the inference firmware flashed and running?"
        )

    arr_ms = np.asarray(latencies, dtype=np.float64) / 1000.0
    row = {
        "model": args.model,
        "format": "tflite_int8",
        "n_runs": len(arr_ms),
        "p50_ms": round(float(np.percentile(arr_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(arr_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(arr_ms, 99)), 3),
        "mean_ms": round(float(arr_ms.mean()), 3),
        "std_ms": round(float(arr_ms.std()), 3),
        "model_kb": args.model_kb,
        "mean_heap_free_bytes": int(np.mean(heaps)),
        "mean_psram_free_bytes": int(np.mean(psrams)),
    }
    print(f"[result] {row}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[info] appended row to {args.output_csv}")


if __name__ == "__main__":
    main()
