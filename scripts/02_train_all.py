"""Train all registered models and score cross-domain.

Usage:
    python scripts/02_train_all.py --output runs/

Each model gets a subdir runs/<name>/ with best.pt and metrics.json.

Memory hygiene: on the Orin Nano (7.6 GB unified host+GPU memory) we have to
free each model + its DataLoader workers between iterations, or the process
hits the OOM killer mid-run. The driver:
  - uses no DataLoader workers (num_workers=0) for cross-domain eval — saves
    ~3 GB of forked-worker overhead;
  - uses a smaller eval batch size (--eval-batch-size, default 16) — keeps
    activation memory bounded for ViT-class models;
  - explicitly deletes the model and runs gc.collect() + cuda.empty_cache()
    between iterations;
  - catches torch.cuda.OutOfMemoryError per-model and continues to the next.

Even with these guards, EConv-ViT (~85M params) at FP32 may still be tight;
the script falls back to CPU eval automatically if a CUDA OOM is caught
during cross-domain scoring (logged with [warn]).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
import traceback
from pathlib import Path

# Tegra-specific: use the expandable-segments allocator. The default caching
# allocator hits an `NVML_SUCCESS == r` internal assertion on Jetson when
# memory pressure is high (NVML reports differ from the desktop CUDA
# expectations). Set this BEFORE importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

import torch  # noqa: E402
from torch.utils.data import DataLoader

from apple_bench.config import DATA_ROOT, RUNS_ROOT, TRAIN_DEFAULTS
from apple_bench.data.appleleaf9 import AppleLeaf9
from apple_bench.data.plant_pathology import PlantPathology2021
from apple_bench.data.plantvillage import PlantVillageApple
from apple_bench.data.transforms import build_eval_transform, build_train_transform
from apple_bench.models import registry
from apple_bench.train.eval_crossdomain import score_with_ci
from apple_bench.train.harness import TrainConfig, train_model

# Models trained via torch harness. Paper 1 has its own path.
TORCH_MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]

# Per-model train-time batch-size overrides. Defaults to TRAIN_DEFAULTS.batch_size
# (64). EConv-ViT at 85M params doesn't fit on the Orin Nano's 7.6 GB unified
# memory at BS=64; drop to 8 (with the harness still doing AMP, this is workable).
TRAIN_BATCH_SIZE_OVERRIDES = {
    "paper2_econv_vit": 8,
    "paper4_resnet50": 32,
}


def _make_loaders(
    pv_root: Path, batch_size: int, num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_t = build_train_transform()
    eval_t = build_eval_transform()
    train_ds = PlantVillageApple(pv_root, "train", train_t, train_ratio=0.8)
    val_ds = PlantVillageApple(pv_root, "val", eval_t, train_ratio=0.8)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


def _free_gpu() -> None:
    """Release any cached GPU memory so the next model has room to allocate."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _score_with_oom_fallback(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    num_classes: int,
    device: str,
    output_path: Path,
) -> dict | None:
    """Try GPU scoring first; fall back to CPU if CUDA OOMs."""
    try:
        return score_with_ci(model, loader, num_classes=num_classes,
                             device=device, output_path=output_path)
    except torch.cuda.OutOfMemoryError:
        print("[warn] CUDA OOM during cross-domain eval; retrying on CPU "
              "(slower but bounded by host RAM).")
        _free_gpu()
        try:
            return score_with_ci(model.cpu(), loader, num_classes=num_classes,
                                 device="cpu", output_path=output_path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] CPU fallback also failed: {type(e).__name__}: {e}")
            return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUNS_ROOT)
    parser.add_argument("--pv-root", type=Path,
                        default=Path("/home/csdf/Apple/Apple leaf dataset/color"))
    parser.add_argument("--pp2021-root", type=Path, default=DATA_ROOT / "plant-pathology-2021")
    parser.add_argument("--al9-root", type=Path, default=DATA_ROOT / "appleleaf9")
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS.epochs)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="Smaller batch for cross-domain eval — Orin Nano OOMs at 64 for "
                             "ViT-class models (default: 16).")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these named models (default: all).")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    eval_t = build_eval_transform()

    # Cross-domain loaders: num_workers=0 to avoid the ~1 GB/worker fork
    # overhead that pushes the Orin into OOM. Eval is short; sync iteration is fine.
    pp2021_loader = None
    if (args.pp2021_root / "train.csv").is_file():
        pp2021_loader = DataLoader(
            PlantPathology2021(args.pp2021_root, eval_t),
            batch_size=args.eval_batch_size, num_workers=0, pin_memory=True,
        )
    else:
        print(f"[warn] Plant Pathology 2021 not found at {args.pp2021_root}; "
              f"cross-domain eval will skip it.")

    al9_loader = None
    try:
        al9_loader = DataLoader(
            AppleLeaf9(args.al9_root, eval_t),
            batch_size=args.eval_batch_size, num_workers=0, pin_memory=True,
        )
    except FileNotFoundError as e:
        print(f"[warn] AppleLeaf9 not loadable ({e}); cross-domain eval will skip it.")

    rows: list[dict] = []
    models_to_run = args.only or TORCH_MODELS
    for name in models_to_run:
        print(f"\n=== {name} ===", flush=True)
        out_dir = args.output / name
        try:
            # Per-model batch-size override (defaults to TRAIN_DEFAULTS.batch_size).
            bs = TRAIN_BATCH_SIZE_OVERRIDES.get(name, TRAIN_DEFAULTS.batch_size)
            train_loader, val_loader = _make_loaders(
                args.pv_root, batch_size=bs, num_workers=TRAIN_DEFAULTS.num_workers,
            )

            cfg = TrainConfig(
                epochs=args.epochs,
                lr=TRAIN_DEFAULTS.lr,
                weight_decay=TRAIN_DEFAULTS.weight_decay,
                output_dir=out_dir,
                device=args.device,
            )
            model = registry.build(name, num_classes=4)
            n_params = sum(p.numel() for p in model.parameters())

            t0 = time.time()
            result = train_model(model, train_loader, val_loader, cfg)
            train_minutes = (time.time() - t0) / 60.0

            # Reload best weights for cross-domain.
            model.load_state_dict(torch.load(result.weights_path, map_location=args.device))

            row: dict = {
                "model": name,
                "params": n_params,
                "val_acc_pv": result.best_val_acc,
                "train_minutes": round(train_minutes, 1),
            }
            if pp2021_loader is not None:
                pp_report = _score_with_oom_fallback(
                    model, pp2021_loader, num_classes=4,
                    device=args.device, output_path=out_dir / "pp2021.json",
                )
                if pp_report is not None:
                    row["acc_pp2021"] = pp_report["accuracy"]
                    row["ci_pp2021"] = (pp_report["ci_lower"], pp_report["ci_upper"])
                else:
                    row["acc_pp2021"] = None
                    row["ci_pp2021"] = None
            else:
                row["acc_pp2021"] = None
                row["ci_pp2021"] = None
            if al9_loader is not None:
                al9_report = _score_with_oom_fallback(
                    model, al9_loader, num_classes=4,
                    device=args.device, output_path=out_dir / "al9.json",
                )
                if al9_report is not None:
                    row["acc_al9"] = al9_report["accuracy"]
                    row["ci_al9"] = (al9_report["ci_lower"], al9_report["ci_upper"])
                else:
                    row["acc_al9"] = None
                    row["ci_al9"] = None
            else:
                row["acc_al9"] = None
                row["ci_al9"] = None
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({
                "model": name, "params": None, "val_acc_pv": None,
                "train_minutes": None, "acc_pp2021": None, "ci_pp2021": None,
                "acc_al9": None, "ci_al9": None, "error": str(e),
            })
        finally:
            # Drop the model + loaders and clear caches before the next
            # iteration; this is what prevents OOM after several models in a
            # row. Rebinding is required rather than `del locals()[var]`:
            # inside a function, locals() returns a snapshot dict, so
            # deleting from it leaves the real local binding -- and the GPU
            # memory it holds -- alive until the next loop iteration
            # reassigns it.
            model = train_loader = val_loader = None  # noqa: F841
            _free_gpu()

    # Write parity CSV.
    csv_path = args.output / "parity_table.csv"
    fieldnames = list({k for r in rows for k in r})
    # Stabilize column order for human readability.
    preferred = ["model", "params", "val_acc_pv", "acc_pp2021", "ci_pp2021",
                 "acc_al9", "ci_al9", "train_minutes", "error"]
    fieldnames = [c for c in preferred if c in fieldnames] + \
                 [c for c in fieldnames if c not in preferred]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nParity table written to {csv_path}")


if __name__ == "__main__":
    main()
