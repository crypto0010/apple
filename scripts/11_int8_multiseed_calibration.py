"""Multi-seed TRT INT8 calibration driver.

Builds a TensorRT INT8 engine for a chosen architecture under a chosen
calibrator (entropy / min-max / percentile), repeated across a fixed seed
list whose effect on the build is the *ordering* of the 100 calibration
images observed by the calibrator. Aggregates per-seed top-1 accuracy on
the PV val partition and emits a multi-seed summary JSON.

The default invocation (entropy calibrator on MobileNetV3-Small with the
canonical five-seed list) reproduces the original paper-1 result at
unchanged output paths when ``--calibrator entropy`` is selected (the
default), so downstream consumers (figure generators, Pareto table
builders) are unaffected. The written JSON is not byte-identical to
pre-calibrator-flag runs: ``metrics.json`` and the summary now carry an
extra ``"calibrator"`` key.

When ``--calibrator minmax`` or ``--calibrator percentile`` is selected,
output paths gain a calibrator suffix so that runs from different
calibrators do not overwrite each other:

    runs/full/<seed>/<model>_int8_<calibrator>/metrics.json
    runs/full/<model>_int8_<calibrator>_multiseed_summary.json

For each seed in --seeds:
    1. Reshuffle the PV training set with that seed's RNG to draw 100
       calibration images.
    2. Build a TRT INT8 engine via build_trt_engine + the chosen
       calibrator at the suffixed engine path.
    3. Evaluate top-1 accuracy on the full PV val partition via
       TensorRTRunner.
    4. Persist metrics.json for that seed.

After all seeds, aggregate val_acc via SeedStats and write a summary.

The shared FP32 anchor + ONNX file at runs/full/<model>/ are read-only;
each seed only writes inside its own runs/full/<seed>/ subdir, so seed
runs do not interfere with each other or with the paper-1 baseline
artefacts.

Memory hygiene matches scripts/05_bench_jetson.py:
    - All torch work is CPU-only (no .cuda() calls anywhere).
    - pycuda.autoinit fires inside build_trt_engine the first time we
      hit an INT8 build (per memory: jetson-pycuda-tensorrt-gotcha).
    - The TensorRTRunner is constructed AFTER any torch work and
      explicitly freed between seeds so engine resources don't stack.

Usage:
    # Paper-1 reproduction (entropy calibrator, MobileNetV3-Small):
    .venv/bin/python scripts/11_int8_multiseed_calibration.py

    # Per-architecture entropy sweep (set FP32 baseline per the
    # pytorch_fp32 row of runs/full/jetson_orin_nano_results.csv):
    .venv/bin/python scripts/11_int8_multiseed_calibration.py \\
        --model efficientnet_lite0 --fp32-baseline-acc 1.0
    .venv/bin/python scripts/11_int8_multiseed_calibration.py \\
        --model yolov8_cls_n --fp32-baseline-acc 0.9925

    # Calibrator factorial (one architecture, three calibrators):
    for c in entropy minmax percentile; do
      .venv/bin/python scripts/11_int8_multiseed_calibration.py \\
        --model mobilenet_v3_small --calibrator "$c"
    done

    # Smoke test (1 seed, default entropy calibrator):
    .venv/bin/python scripts/11_int8_multiseed_calibration.py --seeds 1729
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from apple_bench.analysis.multiseed import seed_stats  # noqa: E402
from apple_bench.config import PROJECT_ROOT, RUNS_ROOT  # noqa: E402
from apple_bench.data.plantvillage import PlantVillageApple  # noqa: E402
from apple_bench.data.transforms import build_eval_transform  # noqa: E402
from apple_bench.export.to_tensorrt import (  # noqa: E402
    Calibrator,
    Precision,
    build_trt_engine,
)

SEEDS = [1729, 42, 7, 2024, 511]
CALIB_BATCH_COUNT = 100
DEFAULT_MODEL = "mobilenet_v3_small"
DEFAULT_FP32_BASELINE = 1.0
DEFAULT_CALIBRATOR = Calibrator.ENTROPY
DEFAULT_PV_ROOT = PROJECT_ROOT / "Apple leaf dataset" / "color"


def _calibrator_suffix(calibrator: Calibrator) -> str:
    """Path suffix per calibrator type.

    Entropy returns an empty string so the original paper-1 paths are
    preserved byte-identically. Non-entropy calibrators get an underscore-
    prefixed suffix so their artefacts cannot overwrite the entropy ones.
    """
    return "" if calibrator is Calibrator.ENTROPY else f"_{calibrator.value}"


def _build_calibration_array(pv_root: Path, n: int, seed: int) -> np.ndarray:
    """Deterministic per-seed calibration array.

    PlantVillageApple's stratified split is fixed at seed=1729 by design
    (see [[project-phase-status]] — the train/val partition is identical
    across runs so we measure quantisation variance, not data variance).
    Per-seed variance enters here only through the calibration sample
    *ordering*: DataLoader shuffle with a per-seed torch.Generator.
    """
    ds = PlantVillageApple(pv_root, "train",
                           build_eval_transform(), train_ratio=0.8)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        num_workers=0, generator=generator)
    arr = np.zeros((n, 3, 224, 224), dtype=np.float32)
    for i, (x, _y) in enumerate(loader):
        if i >= n:
            break
        arr[i] = x.numpy()[0]
    return arr


def _build_val_numpy(pv_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Cache the full val set as (X, y) numpy arrays.

    Built once, before any TRT initialisation, so the torch DataLoader's
    CUDA context (if any) never collides with pycuda.autoinit later.
    """
    ds = PlantVillageApple(pv_root, "val",
                           build_eval_transform(), train_ratio=0.8)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    n = len(ds)
    X = np.zeros((n, 3, 224, 224), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i, (xb, yb) in enumerate(loader):
        X[i] = xb.numpy()[0]
        y[i] = int(yb.numpy()[0])
    return X, y


def _evaluate_trt_int8(engine_path: Path, X: np.ndarray,
                       y: np.ndarray) -> tuple[int, int]:
    """Top-1 correct/total for the TRT engine over (X, y).

    Imports the runner lazily so script imports don't trigger
    pycuda.autoinit until we're actually ready to touch the device.
    """
    from apple_bench.runners.tensorrt_runner import TensorRTRunner  # noqa: PLC0415

    runner = TensorRTRunner(engine_path)
    correct = 0
    try:
        # warmup is cheap and avoids first-run latency contaminating
        # the first prediction's correctness if any TRT lazy init paths
        # actually depend on shape.
        runner.warmup(X[:1], n=3)
        for i in range(X.shape[0]):
            out = runner.run(X[i:i + 1])
            pred = int(np.argmax(out, axis=1)[0])
            if pred == int(y[i]):
                correct += 1
    finally:
        # Release the engine + context before the next seed so we don't
        # hold N engines in memory simultaneously.
        del runner
        gc.collect()
    return correct, int(X.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name (matches runs/full/<model>/ "
                             "directory and the registry key).")
    parser.add_argument("--fp32-baseline-acc", type=float,
                        default=DEFAULT_FP32_BASELINE,
                        help="FP32 anchor val accuracy on the PV val "
                             "partition; used to compute Delta_pp in the "
                             "summary. Look up from "
                             "runs/full/jetson_orin_nano_results.csv "
                             "(pytorch_fp32 row, accuracy_pv column).")
    parser.add_argument("--pv-root", type=Path, default=DEFAULT_PV_ROOT)
    parser.add_argument("--onnx", type=Path, default=None,
                        help="ONNX file path. Defaults to "
                             "runs/full/<model>/model.onnx.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--calib-n", type=int, default=CALIB_BATCH_COUNT)
    parser.add_argument(
        "--calibrator",
        choices=[c.value for c in Calibrator],
        default=DEFAULT_CALIBRATOR.value,
        help=(
            "INT8 calibrator algorithm. 'entropy' (default) reproduces "
            "the paper-1 byte-identical paths; 'minmax' and 'percentile' "
            "write to suffixed paths so prior entropy runs are preserved."
        ),
    )
    args = parser.parse_args()

    calibrator = Calibrator(args.calibrator)
    suffix = _calibrator_suffix(calibrator)

    if args.onnx is None:
        args.onnx = RUNS_ROOT / "full" / args.model / "model.onnx"
    if not args.onnx.is_file():
        raise SystemExit(
            f"ONNX not found at {args.onnx}; run scripts/04_export_all.py "
            f"for {args.model} first."
        )

    print(f"[info] model={args.model}", flush=True)
    print(f"[info] calibrator={calibrator.value}", flush=True)
    print(f"[info] fp32_baseline_acc={args.fp32_baseline_acc}", flush=True)
    print(f"[info] onnx={args.onnx}", flush=True)
    print(f"[info] seeds={args.seeds}  calib_n={args.calib_n}", flush=True)
    print(f"[info] pv_root={args.pv_root}", flush=True)

    X_val, y_val = _build_val_numpy(args.pv_root)
    print(f"[info] val set cached: X={X_val.shape}  y={y_val.shape}",
          flush=True)

    per_seed: list[dict] = []

    for seed in args.seeds:
        out_dir = (
            RUNS_ROOT / "full" / str(seed) / f"{args.model}_int8{suffix}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        engine_path = out_dir / "model.int8.engine"
        print(f"\n=== seed {seed} ===", flush=True)

        calib = _build_calibration_array(args.pv_root, args.calib_n, seed)
        print(f"  calibration set: {calib.shape}", flush=True)

        t0 = time.time()
        build_trt_engine(
            args.onnx, engine_path,
            precision=Precision.INT8,
            int8_calibration_data=calib,
            int8_calibrator=calibrator,
        )
        build_seconds = round(time.time() - t0, 1)
        engine_bytes = engine_path.stat().st_size
        print(f"  engine built: {engine_bytes} bytes  "
              f"build_s={build_seconds}", flush=True)

        t0 = time.time()
        correct, total = _evaluate_trt_int8(engine_path, X_val, y_val)
        eval_seconds = round(time.time() - t0, 1)
        val_acc = correct / max(total, 1)
        print(f"  -> val_acc={val_acc:.4f}  ({correct}/{total})  "
              f"eval_s={eval_seconds}", flush=True)

        metrics = {
            "seed": seed,
            "calibrator": calibrator.value,
            "val_acc": val_acc,
            "val_correct": correct,
            "val_total": total,
            "calib_n": args.calib_n,
            "build_seconds": build_seconds,
            "eval_seconds": eval_seconds,
            "engine_bytes": engine_bytes,
            "onnx_path": str(args.onnx),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        per_seed.append(metrics)

    # ---- Aggregate -----------------------------------------------------
    accs = [m["val_acc"] for m in per_seed]
    stats = seed_stats(accs)
    summary = {
        "model": args.model,
        "calibrator": calibrator.value,
        "precision": "tensorrt_int8",
        "method": "multi_seed_calibration_shuffle",
        "n_seeds": stats.n,
        "val_acc_mean": stats.mean,
        "val_acc_sd": stats.sd,
        "per_seed": per_seed,
        "fp32_baseline_acc": args.fp32_baseline_acc,
        "delta_pp_vs_fp32_mean": round(
            (stats.mean - args.fp32_baseline_acc) * 100, 3
        ),
    }
    summary_path = (
        RUNS_ROOT / "full"
        / f"{args.model}_int8{suffix}_multiseed_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===", flush=True)
    print(f"  calibrator={calibrator.value}", flush=True)
    print(f"  n_seeds={stats.n}", flush=True)
    print(f"  val_acc mean={stats.mean:.4f}  sd={stats.sd:.4f}", flush=True)
    print(f"  vs FP32 ({args.fp32_baseline_acc:.4f}): "
          f"{(stats.mean - args.fp32_baseline_acc) * 100:+.2f} pp",
          flush=True)
    print(f"  summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
