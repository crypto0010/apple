"""Re-run cross-domain eval on already-trained checkpoints.

Useful when the train-all driver finished training a model but the cross-domain
eval crashed (e.g., OOM mid-eval). Loads <output>/<model>/best.pt and runs
score_with_ci against PP2021 + AppleLeaf9.

Usage:
    python scripts/02b_eval_crossdomain_only.py --output runs/smoke --models paper4_resnet50

The driver intentionally uses smaller batch sizes and num_workers=0 to match
the train-all driver's eval defaults.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

# Same Tegra allocator setting as the main driver.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from apple_bench.config import DATA_ROOT  # noqa: E402
from apple_bench.data.appleleaf9 import AppleLeaf9  # noqa: E402
from apple_bench.data.plant_pathology import PlantPathology2021  # noqa: E402
from apple_bench.data.transforms import build_eval_transform  # noqa: E402
from apple_bench.models import registry  # noqa: E402
from apple_bench.train.eval_crossdomain import score_with_ci  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names whose best.pt should be re-evaluated.")
    p.add_argument("--pp2021-root", type=Path, default=DATA_ROOT / "plant-pathology-2021")
    p.add_argument("--al9-root", type=Path, default=DATA_ROOT / "appleleaf9")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    eval_t = build_eval_transform()
    pp_loader = DataLoader(
        PlantPathology2021(args.pp2021_root, eval_t),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    al_loader = DataLoader(
        AppleLeaf9(args.al9_root, eval_t),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )

    for name in args.models:
        out_dir = args.output / name
        ckpt = out_dir / "best.pt"
        if not ckpt.is_file():
            print(f"[skip] {name}: no checkpoint at {ckpt}")
            continue
        print(f"\n=== {name} ===", flush=True)
        model = registry.build(name, num_classes=4)
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        try:
            r = score_with_ci(model, pp_loader, num_classes=4,
                              device=args.device, output_path=out_dir / "pp2021.json")
            print(f"  PP2021: acc={r['accuracy']*100:.2f}% (n={r['n_samples']})")
            r = score_with_ci(model, al_loader, num_classes=4,
                              device=args.device, output_path=out_dir / "al9.json")
            print(f"  AL9:    acc={r['accuracy']*100:.2f}% (n={r['n_samples']})")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] eval failed: {type(e).__name__}: {e}")
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
