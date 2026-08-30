"""30-minute sustained-throughput run for each model on its best runner.

For each model, picks the highest-quality engine that exists:
  - tensorrt_int8 (if accuracy regression in scripts/05 passed)
  - tensorrt_fp16 (default)
  - tensorrt_fp32 (fallback)
  - onnxruntime_fp32 (last resort)

Writes runs/full/<name>/thermal_30min.csv with one row per minute:
    t_seconds, infs_in_window, fps, tj_C, gr3d_freq_pct

The interesting plot (deferred to Plan A.3) is FPS vs time — finds the
point at which thermal throttling kicks in.

Usage:
    uv run python scripts/06_thermal_run.py --duration-minutes 30
    uv run python scripts/06_thermal_run.py --only mobilenet_v3_small --duration-minutes 5
"""

from __future__ import annotations

import argparse
import csv
import os
import traceback
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

import numpy as np  # noqa: E402

from apple_bench.bench.thermal import sustained_run  # noqa: E402
from apple_bench.config import RUNS_ROOT  # noqa: E402

TORCH_MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]

# Which formats to try for sustained-run, in preference order.
FORMAT_PREFERENCE = [
    "tensorrt_int8",
    "tensorrt_fp16",
    "tensorrt_fp32",
    "onnxruntime_fp32",
]


def _int8_passed_regression(results_csv: Path, model: str) -> bool:
    if not results_csv.is_file():
        return False
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model and row["format"] == "tensorrt_int8":
                delta = row.get("accuracy_delta_pp_vs_fp32")
                if delta and delta != "None":
                    try:
                        return float(delta) <= 2.0
                    except ValueError:
                        return False
    return False


def _pick_runner(name: str, model_dir: Path, results_csv: Path):
    """Return (format_label, runner) for the best available format."""
    use_int8 = _int8_passed_regression(results_csv, name)

    for fmt in FORMAT_PREFERENCE:
        if fmt == "tensorrt_int8" and not use_int8:
            continue
        engine = {
            "tensorrt_int8": model_dir / "model.int8.engine",
            "tensorrt_fp16": model_dir / "model.fp16.engine",
            "tensorrt_fp32": model_dir / "model.fp32.engine",
        }.get(fmt)
        onnx = model_dir / "model.onnx"

        if fmt.startswith("tensorrt_") and engine and engine.is_file():
            from apple_bench.runners.tensorrt_runner import TensorRTRunner  # noqa: PLC0415
            return fmt, TensorRTRunner(engine)
        if fmt == "onnxruntime_fp32" and onnx.is_file():
            from apple_bench.runners.onnx_runner import ONNXRunner  # noqa: PLC0415
            return fmt, ONNXRunner(
                onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT / "full")
    parser.add_argument("--duration-minutes", type=int, default=30)
    parser.add_argument("--sample-period-s", type=float, default=60.0)
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    results_csv = args.runs_root / "jetson_orin_nano_results.csv"
    if not results_csv.is_file():
        print(f"[warn] {results_csv} not found; will pick the highest non-INT8 "
              f"engine for each model.")

    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)
    models = args.only or TORCH_MODELS
    for name in models:
        print(f"\n=== {name} ({args.duration_minutes} min) ===", flush=True)
        model_dir = args.runs_root / name
        fmt, runner = _pick_runner(name, model_dir, results_csv)
        if runner is None:
            print(f"[skip] no usable runner for {name} (no engines or ONNX in {model_dir})")
            continue
        print(f"  using {fmt}")
        out_csv = model_dir / f"thermal_{args.duration_minutes}min.csv"
        try:
            sustained_run(
                runner,
                sample,
                duration_minutes=args.duration_minutes,
                output_csv=out_csv,
                sample_period_s=args.sample_period_s,
            )
            print(f"  -> {out_csv}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {name} thermal run failed: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            del runner


if __name__ == "__main__":
    main()
