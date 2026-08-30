"""Export every trained torch model to ONNX + TRT (FP32/FP16/INT8).

For each model under ``runs/full/<name>/best.pt``:
    runs/full/<name>/model.onnx
    runs/full/<name>/model.fp32.engine
    runs/full/<name>/model.fp16.engine
    runs/full/<name>/model.int8.engine    (calibrated with ~100 PV training images)
    runs/full/<name>/model.tflite         (only if <MCU_BUDGET_BYTES)

Skips paper1_aco_svm (sklearn pipeline, not torch).

Usage:
    uv run python scripts/04_export_all.py --runs-root runs/full
    uv run python scripts/04_export_all.py --skip-tflite    # bypass MCU path
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from apple_bench.config import PROJECT_ROOT, RUNS_ROOT  # noqa: E402
from apple_bench.data.plantvillage import PlantVillageApple  # noqa: E402
from apple_bench.data.transforms import build_eval_transform  # noqa: E402
from apple_bench.export.to_onnx import export_to_onnx  # noqa: E402
from apple_bench.export.to_tensorrt import Precision, build_trt_engine  # noqa: E402
from apple_bench.export.to_tflite import export_to_tflite_int8  # noqa: E402
from apple_bench.models import registry  # noqa: E402

TORCH_MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]

CALIB_BATCH_COUNT = 100  # ~100 representative images for INT8 entropy calibrator
DEFAULT_PV_ROOT = PROJECT_ROOT / "Apple leaf dataset" / "color"

# ESP32-S3-WROOM-1-N8R8 has 8 MB OPI PSRAM + 3 MB sketch space + ~370 KB
# internal SRAM.  Embedded model + TFLM runtime + tensor arena must fit
# within that, so we cap stored .tflite at ~2.5 MB; anything larger is
# emitted but flagged as oversize.
MCU_BUDGET_BYTES = int(2.5 * 1024 * 1024)


def _build_calibration_array(pv_root: Path, n: int) -> np.ndarray:
    """Pull n preprocessed PV training images for INT8 calibration."""
    ds = PlantVillageApple(pv_root, "train", build_eval_transform(), train_ratio=0.8)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    arr = np.zeros((n, 3, 224, 224), dtype=np.float32)
    for i, (x, _y) in enumerate(loader):
        if i >= n:
            break
        arr[i] = x.numpy()[0]
    return arr


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_one(
    name: str,
    runs_root: Path,
    calib_data: np.ndarray,
    do_int8: bool,
    do_fp16: bool,
    do_tflite: bool = True,
) -> dict:
    model_dir = runs_root / name
    weights = model_dir / "best.pt"
    if not weights.is_file():
        return {"model": name, "status": "missing_weights", "weights": str(weights)}

    model = registry.build(name, num_classes=4, pretrained=False)
    state = torch.load(weights, map_location="cpu")
    # `train_model` saves either a state_dict or the full model;
    # account for both.
    if isinstance(state, dict) and not any(k.startswith("__") for k in state):
        model.load_state_dict(state)
    else:
        model = state
    model.train(mode=False)

    onnx_path = model_dir / "model.onnx"
    fp32_path = model_dir / "model.fp32.engine"
    fp16_path = model_dir / "model.fp16.engine"
    int8_path = model_dir / "model.int8.engine"

    timings: dict = {"model": name, "status": "ok"}
    sample = torch.randn(1, 3, 224, 224)

    t0 = time.time()
    export_to_onnx(model, onnx_path, sample)
    timings["onnx_s"] = round(time.time() - t0, 1)
    timings["onnx_kb"] = onnx_path.stat().st_size // 1024

    del model
    _free_gpu()

    t0 = time.time()
    build_trt_engine(onnx_path, fp32_path, precision=Precision.FP32)
    timings["fp32_s"] = round(time.time() - t0, 1)
    timings["fp32_kb"] = fp32_path.stat().st_size // 1024

    if do_fp16:
        t0 = time.time()
        build_trt_engine(onnx_path, fp16_path, precision=Precision.FP16)
        timings["fp16_s"] = round(time.time() - t0, 1)
        timings["fp16_kb"] = fp16_path.stat().st_size // 1024

    if do_int8:
        t0 = time.time()
        try:
            build_trt_engine(
                onnx_path,
                int8_path,
                precision=Precision.INT8,
                int8_calibration_data=calib_data,
            )
            timings["int8_s"] = round(time.time() - t0, 1)
            timings["int8_kb"] = int8_path.stat().st_size // 1024
        except Exception as e:  # noqa: BLE001
            timings["int8_s"] = round(time.time() - t0, 1)
            timings["int8_error"] = f"{type(e).__name__}: {e}"

    if do_tflite:
        tflite_path = model_dir / "model.tflite"
        t0 = time.time()
        try:
            # Use a smaller calibration set for TFLite (onnx2tf is slow).
            export_to_tflite_int8(
                onnx_path, tflite_path, calibration_data=calib_data[:32]
            )
            size = tflite_path.stat().st_size
            timings["tflite_s"] = round(time.time() - t0, 1)
            timings["tflite_kb"] = size // 1024
            timings["tflite_fits_mcu"] = size <= MCU_BUDGET_BYTES
        except Exception as e:  # noqa: BLE001
            timings["tflite_s"] = round(time.time() - t0, 1)
            timings["tflite_error"] = f"{type(e).__name__}: {e}"

    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT / "full")
    parser.add_argument("--pv-root", type=Path, default=DEFAULT_PV_ROOT)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only export these named models (default: all torch models).",
    )
    parser.add_argument("--skip-int8", action="store_true",
                        help="Skip INT8 engine build (still produces FP32 + FP16).")
    parser.add_argument("--skip-fp16", action="store_true",
                        help="Skip FP16 engine build.")
    parser.add_argument("--skip-tflite", action="store_true",
                        help="Skip TFLite-Micro INT8 export (ESP32 tier).")
    args = parser.parse_args()

    args.runs_root.mkdir(parents=True, exist_ok=True)

    do_int8 = not args.skip_int8
    calib = (
        _build_calibration_array(args.pv_root, CALIB_BATCH_COUNT)
        if do_int8 else
        np.zeros((1, 3, 224, 224), dtype=np.float32)
    )
    if do_int8:
        print(f"[info] calibration set: {calib.shape} from {args.pv_root}")

    models = args.only or TORCH_MODELS
    results = []
    for name in models:
        print(f"\n=== {name} ===", flush=True)
        try:
            timings = _export_one(
                name,
                args.runs_root,
                calib,
                do_int8,
                not args.skip_fp16,
                do_tflite=not args.skip_tflite,
            )
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            timings = {"model": name, "status": "error",
                       "error": f"{type(e).__name__}: {e}"}
        results.append(timings)
        print(f"  -> {timings}")
        _free_gpu()

    print("\n=== Export summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
