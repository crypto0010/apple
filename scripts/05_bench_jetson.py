"""Latency + power + accuracy benchmark for every (model, format) on Jetson.

Reads engines emitted by ``scripts/04_export_all.py`` and produces
``runs/full/jetson_orin_nano_results.csv`` with columns:

    model, format, p50_ms, p95_ms, p99_ms, mean_ms, std_ms,
    mean_power_mw, energy_mj_per_inf, accuracy_pv, accuracy_delta_pp_vs_fp32,
    engine_size_kb, nvpmodel

Formats covered: pytorch_fp32 (CUDA), onnxruntime_fp32 (CUDA EP), tensorrt_fp32,
tensorrt_fp16, tensorrt_int8.

Power: ``apple_bench.bench.power.TegraPowerLogger`` wraps tegrastats and
reports the mean VDD_IN (whole-board input power, mW) over the timing window.
``energy_mj_per_inf = mean_power_mw * mean_ms`` (mW * ms = uJ; divided by
1000 to give mJ).

Locks ``nvpmodel -m 2`` (MAXN_SUPER) at startup so all rows share one power
mode; records the mode in each row.

Usage:
    sudo uv run python scripts/05_bench_jetson.py
        # sudo only required if nvpmodel needs to switch modes; if you've
        # already locked MAXN_SUPER outside this script, plain uv run is fine.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
import traceback
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from apple_bench.bench.latency import LatencyResult, measure_latency  # noqa: E402
from apple_bench.bench.power import TegraPowerLogger  # noqa: E402
from apple_bench.config import PROJECT_ROOT, RUNS_ROOT  # noqa: E402
from apple_bench.data.plantvillage import PlantVillageApple  # noqa: E402
from apple_bench.data.transforms import build_eval_transform  # noqa: E402
from apple_bench.models import registry  # noqa: E402
from apple_bench.runners.onnx_runner import ONNXRunner  # noqa: E402
from apple_bench.runners.pytorch_runner import PyTorchRunner  # noqa: E402

DEFAULT_PV_ROOT = PROJECT_ROOT / "Apple leaf dataset" / "color"

TORCH_MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]


def _maybe_set_nvpmodel(mode: int) -> str:
    """Try to lock nvpmodel; return the active mode string."""
    try:
        subprocess.run(
            ["nvpmodel", "-m", str(mode)],
            check=False, capture_output=True, text=True, timeout=10,
        )
        out = subprocess.run(
            ["nvpmodel", "-q"],
            check=False, capture_output=True, text=True, timeout=10,
        ).stdout
        return out.strip().splitlines()[-1] if out else f"mode={mode}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "nvpmodel_unavailable"


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eval_loader(pv_root: Path, batch_size: int) -> DataLoader:
    return DataLoader(
        PlantVillageApple(pv_root, "val", build_eval_transform(), train_ratio=0.8),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )


def _measure_with_power(
    runner,
    sample_input: np.ndarray,
    warmup: int,
    n_runs: int,
) -> tuple[LatencyResult, float]:
    """Run measure_latency under a TegraPowerLogger, return (lat, mean_power_mw)."""
    power = TegraPowerLogger(interval_ms=100)
    power.start()
    try:
        lat = measure_latency(runner, sample_input, warmup=warmup, n_runs=n_runs)
    finally:
        power.stop()
    return lat, power.mean_power_mw()


def _load_torch_model(name: str, weights: Path) -> torch.nn.Module:
    model = registry.build(name, num_classes=4, pretrained=False)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    return model


def _build_runners(name: str, model_dir: Path, device: str):
    """Yield (format_name, runner) for every available format.

    Skips formats whose artifact is missing (so partially-failed exports
    don't crash the bench driver).
    """
    weights = model_dir / "best.pt"
    onnx_path = model_dir / "model.onnx"
    fp32_engine = model_dir / "model.fp32.engine"
    fp16_engine = model_dir / "model.fp16.engine"
    int8_engine = model_dir / "model.int8.engine"

    if weights.is_file():
        model = _load_torch_model(name, weights)
        yield "pytorch_fp32", PyTorchRunner(model, device=device)

    if onnx_path.is_file():
        yield "onnxruntime_fp32", ONNXRunner(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    if fp32_engine.is_file() or fp16_engine.is_file() or int8_engine.is_file():
        # Lazy import keeps non-Jetson dev hosts able to import this module.
        from apple_bench.runners.tensorrt_runner import TensorRTRunner  # noqa: PLC0415

        if fp32_engine.is_file():
            yield "tensorrt_fp32", TensorRTRunner(fp32_engine)
        if fp16_engine.is_file():
            yield "tensorrt_fp16", TensorRTRunner(fp16_engine)
        if int8_engine.is_file():
            yield "tensorrt_int8", TensorRTRunner(int8_engine)


def _engine_size_kb(model_dir: Path, fmt: str) -> int:
    name = {
        "pytorch_fp32": "best.pt",
        "onnxruntime_fp32": "model.onnx",
        "tensorrt_fp32": "model.fp32.engine",
        "tensorrt_fp16": "model.fp16.engine",
        "tensorrt_int8": "model.int8.engine",
    }[fmt]
    p = model_dir / name
    return p.stat().st_size // 1024 if p.is_file() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT / "full")
    parser.add_argument("--pv-root", type=Path, default=DEFAULT_PV_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--accuracy-eval-cap", type=int, default=400,
                        help="Cap eval set size when running INT8 accuracy regression. "
                             "Default keeps each TRT INT8 regression check under ~1 min "
                             "on the Orin Nano. Set to 0 for the full val set.")
    parser.add_argument("--append", action="store_true",
                        help="Append rows to runs/full/jetson_orin_nano_results.csv instead of "
                             "overwriting it. Useful when re-running --only for a single model.")
    args = parser.parse_args()

    nvpmode = _maybe_set_nvpmodel(2)
    print(f"[info] nvpmodel: {nvpmode}")

    eval_loader = _eval_loader(args.pv_root, args.eval_batch)
    # Pre-materialize a capped eval list so each regression check reuses the
    # same images (and doesn't keep iterating DataLoaders that hold workers).
    eval_batches = []
    for i, (x, y) in enumerate(eval_loader):
        eval_batches.append((x, y))
        if args.accuracy_eval_cap and (i + 1) * args.eval_batch >= args.accuracy_eval_cap:
            break

    class _BatchList:
        def __init__(self, batches): self.batches = batches
        def __iter__(self): return iter(self.batches)

    capped_loader = _BatchList(eval_batches)

    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)

    csv_path = args.runs_root / "jetson_orin_nano_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "format", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms",
        "mean_power_mw", "energy_mj_per_inf",
        "accuracy_pv", "accuracy_delta_pp_vs_fp32",
        "engine_size_kb", "nvpmodel",
    ]
    append_mode = args.append and csv_path.is_file()
    f = csv_path.open("a" if append_mode else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not append_mode:
        writer.writeheader()
        f.flush()

    def _predict_all(runner) -> tuple[np.ndarray, np.ndarray]:
        """Return concatenated predictions and labels over the capped eval set.

        Iterates one sample at a time so the same code works with TRT
        engines built at static batch=1 and with PyTorch/ORT runners that
        natively batch. Total cost is ~200 forward passes per format,
        negligible next to the 200 timed-latency runs.
        """
        preds_chunks = []
        labels_chunks = []
        for x, y in capped_loader:
            x_np = x.numpy().astype(np.float32)
            y_np = y.numpy()
            for i in range(x_np.shape[0]):
                logits = runner.run(x_np[i : i + 1])
                preds_chunks.append(int(logits.argmax(axis=1)[0]))
                labels_chunks.append(int(y_np[i]))
        return np.asarray(preds_chunks), np.asarray(labels_chunks)

    models = args.only or TORCH_MODELS
    for name in models:
        print(f"\n=== {name} ===", flush=True)
        model_dir = args.runs_root / name

        # Step 1: build PyTorch FP32 runner, cache FP32 predictions + labels.
        # This MUST happen before any TensorRTRunner is built — pycuda.autoinit
        # invalidates PyTorch's CUDA stream, after which fp32_runner.run() would
        # raise cuDNN STREAM_MISMATCH.
        weights = model_dir / "best.pt"
        fp32_predictions: np.ndarray | None = None
        fp32_labels: np.ndarray | None = None
        fp32_lat: LatencyResult | None = None
        fp32_mw: float = 0.0
        try:
            if weights.is_file():
                fp32_model = _load_torch_model(name, weights)
                fp32_runner = PyTorchRunner(fp32_model, device=args.device)
                print("  pytorch_fp32 (latency + cache predictions)", flush=True)
                fp32_lat, fp32_mw = _measure_with_power(
                    fp32_runner, sample, warmup=args.warmup, n_runs=args.n_runs,
                )
                fp32_predictions, fp32_labels = _predict_all(fp32_runner)
                fp32_acc = float((fp32_predictions == fp32_labels).mean())

                row = {
                    "model": name, "format": "pytorch_fp32",
                    "p50_ms": round(fp32_lat.p50_ms, 3),
                    "p95_ms": round(fp32_lat.p95_ms, 3),
                    "p99_ms": round(fp32_lat.p99_ms, 3),
                    "mean_ms": round(fp32_lat.mean_ms, 3),
                    "std_ms": round(fp32_lat.std_ms, 3),
                    "mean_power_mw": round(fp32_mw, 1),
                    "energy_mj_per_inf": round(fp32_mw * fp32_lat.mean_ms / 1000.0, 3),
                    "accuracy_pv": round(fp32_acc, 4),
                    "accuracy_delta_pp_vs_fp32": 0.0,
                    "engine_size_kb": _engine_size_kb(model_dir, "pytorch_fp32"),
                    "nvpmodel": nvpmode,
                }
                writer.writerow(row)
                f.flush()
                print(f"    {row}")
                del fp32_runner, fp32_model
                _free_gpu()
            else:
                print(f"  [skip] {weights} missing — cannot compute FP32 anchor")
                fp32_acc = None
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] pytorch_fp32 path failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            fp32_acc = None

        # Step 2: walk every non-pytorch format. Each runner's predictions are
        # compared against the cached fp32_predictions (numpy-vs-numpy; no
        # PyTorch re-run required).
        for fmt, runner in _build_runners(name, model_dir, args.device):
            if fmt == "pytorch_fp32":
                continue  # already handled above
            print(f"  {fmt}", flush=True)
            try:
                lat, mw = _measure_with_power(
                    runner, sample, warmup=args.warmup, n_runs=args.n_runs,
                )
                energy_mj = round(mw * lat.mean_ms / 1000.0, 3)

                acc = None
                delta_pp = None
                if fp32_predictions is not None:
                    preds, _ = _predict_all(runner)
                    acc = float((preds == fp32_labels).mean())
                    delta_pp = round((fp32_acc - acc) * 100, 2) if fp32_acc is not None else None

                row = {
                    "model": name, "format": fmt,
                    "p50_ms": round(lat.p50_ms, 3),
                    "p95_ms": round(lat.p95_ms, 3),
                    "p99_ms": round(lat.p99_ms, 3),
                    "mean_ms": round(lat.mean_ms, 3),
                    "std_ms": round(lat.std_ms, 3),
                    "mean_power_mw": round(mw, 1),
                    "energy_mj_per_inf": energy_mj,
                    "accuracy_pv": round(acc, 4) if acc is not None else None,
                    "accuracy_delta_pp_vs_fp32": delta_pp,
                    "engine_size_kb": _engine_size_kb(model_dir, fmt),
                    "nvpmodel": nvpmode,
                }
                writer.writerow(row)
                f.flush()
                print(f"    {row}")
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {fmt} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
            finally:
                del runner
                _free_gpu()

    f.close()
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
