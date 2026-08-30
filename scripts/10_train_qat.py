"""Multi-seed QAT fine-tuning driver for MobileNetV3-Small (Paper 2 Task 5).

DEPRECATED 2026-05-22: The underlying QAT pipeline in
``apple_bench.train.qat_finetune`` produces a constant-output INT8
model on PyTorch 2.8 + qnnpack (bug isolated to ``convert_fx``;
reproduced by ``scripts/debug_qat_stages.py``). Use
``scripts/11_int8_multiseed_calibration.py`` for the working
vendor-native multi-seed INT8 study (FP32 -> ONNX -> TRT INT8 with
EntropyCalibrator2). This driver is retained as evidence for Paper 2
and to re-validate when convert_fx is fixed upstream.

Loads the FP32 anchor checkpoint at runs/full/mobilenet_v3_small/best.pt,
fine-tunes the H-Swish-free variant under FX-mode QAT for each seed in
SEEDS, and writes per-seed artifacts under
runs/full/<seed>/mobilenet_v3_small_qat/.

Memory hygiene matches scripts/02_train_all.py: PYTORCH_CUDA_ALLOC_CONF
is set before importing torch, num_workers=0 on the DataLoaders to
avoid forked-worker overhead on the Orin Nano's 7.6 GB unified memory,
and gc + cuda.empty_cache() runs between seeds to release the converted
QAT model.

Usage:
    .venv/bin/python scripts/10_train_qat.py                # 5 seeds, 30 epochs
    .venv/bin/python scripts/10_train_qat.py --seeds 1729 --epochs 1  # smoke
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

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import apple_bench.models.mobilenet_v3_hswish_free  # noqa: F401, E402  -- registration
from apple_bench.config import PROJECT_ROOT, RUNS_ROOT  # noqa: E402
from apple_bench.data.plantvillage import PlantVillageApple  # noqa: E402
from apple_bench.data.transforms import build_eval_transform, build_train_transform  # noqa: E402
from apple_bench.models import registry  # noqa: E402
from apple_bench.train.qat_finetune import qat_finetune  # noqa: E402

SEEDS = [1729, 42, 7, 2024, 511]
DEFAULT_PV_ROOT = PROJECT_ROOT / "Apple leaf dataset" / "color"


def _score_accuracy(model, loader, device):
    """Top-1 accuracy of model on loader."""
    model.train(mode=False)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
    return correct / max(total, 1)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv-root", type=Path, default=DEFAULT_PV_ROOT)
    parser.add_argument(
        "--anchor", type=Path,
        default=RUNS_ROOT / "full" / "mobilenet_v3_small" / "best.pt",
        help="FP32 checkpoint to initialise QAT from.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if not args.anchor.is_file():
        raise SystemExit(f"FP32 anchor not found at {args.anchor}; "
                         "run scripts/02_train_all.py first.")

    print(f"[info] device={args.device}  epochs={args.epochs}  "
          f"lr={args.lr}  batch_size={args.batch_size}", flush=True)
    print(f"[info] PV root: {args.pv_root}", flush=True)
    print(f"[info] FP32 anchor: {args.anchor}", flush=True)

    train_ds = PlantVillageApple(args.pv_root, "train",
                                 build_train_transform(), train_ratio=0.8)
    val_ds = PlantVillageApple(args.pv_root, "val",
                               build_eval_transform(), train_ratio=0.8)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    for seed in args.seeds:
        out_dir = RUNS_ROOT / "full" / str(seed) / "mobilenet_v3_small_qat"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== seed {seed} ===", flush=True)

        torch.manual_seed(seed)
        fp32 = registry.build("mobilenet_v3_small_hswish_free",
                              num_classes=4, pretrained=False)
        # Initialise from the FP32 anchor (state_dict layout is compatible
        # with the H-Swish-free variant — only activation classes differ).
        state = torch.load(args.anchor, map_location="cpu")
        if isinstance(state, dict):
            result = fp32.load_state_dict(state, strict=False)
        else:
            # train_all.py may have saved the full module rather than the
            # state_dict; pull state_dict() then load.
            result = fp32.load_state_dict(state.state_dict(), strict=False)
        # strict=False silently absorbs mismatches; that is catastrophic
        # when the load is meant to initialise the model — assert cleanness.
        assert not result.missing_keys, (
            f"FP32 anchor left {len(result.missing_keys)} parameter(s) at "
            f"random init: {result.missing_keys[:5]}"
        )
        assert not result.unexpected_keys, (
            f"FP32 anchor has {len(result.unexpected_keys)} extra key(s) the "
            f"target model does not consume: {result.unexpected_keys[:5]}"
        )

        t0 = time.time()
        qat_model = qat_finetune(
            fp32_model=fp32,
            train_loader=train_loader, val_loader=val_loader,
            epochs=args.epochs, lr=args.lr, device=args.device,
        )
        train_seconds = round(time.time() - t0, 1)

        val_acc = _score_accuracy(qat_model, val_loader, device="cpu")
        ckpt = out_dir / "best.pt"
        torch.save(qat_model.state_dict(), ckpt)
        metrics = {
            "seed": seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_seconds": train_seconds,
            "val_acc": val_acc,
            "checkpoint_bytes": ckpt.stat().st_size,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  -> val_acc={val_acc:.4f}  train_s={train_seconds}  "
              f"checkpoint={ckpt}", flush=True)

        del qat_model, fp32
        _free_gpu()


if __name__ == "__main__":
    main()
