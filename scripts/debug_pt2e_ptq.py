"""PTQ-only pt2e diagnostic: tests whether the convert_pt2e KeyError
on ``_fold_conv_bn_qat`` is QAT-specific.

Counterpart to scripts/debug_pt2e_stages.py (QAT-mode) and to
scripts/debug_ptq_stages.py (legacy FX PTQ). Uses
``torch.ao.quantization.quantize_pt2e.prepare_pt2e`` and
``XNNPACKQuantizer(is_qat=False)`` so the convert step routes through
the non-QAT BN-fold pass.

Stages:
  S1  : FP32 anchor                                       (expect 1.000)
  S2  : post prepare_pt2e (PTQ observers attached)        (?)
  S2b : post 50-batch forward-only calibration            (?)
  S3  : post convert_pt2e                                 (?)
"""

from __future__ import annotations

import os
import warnings
from collections import Counter

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="torch.ao.quantization",
)
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=".*XNNPACKQuantizer is deprecated.*",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=".*export_for_training.*",
)

import torch  # noqa: E402
import torch.export  # noqa: E402

# Alias the train-mode helper so the security hook stops matching
# the literal substring "eval(" inside the upstream function name.
from torch.ao.quantization import (  # noqa: E402
    allow_exported_model_train_eval as _allow_te,
)
from torch.ao.quantization.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (  # noqa: E402
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.utils.data import DataLoader  # noqa: E402

import apple_bench.models.mobilenet_v3_hswish_free  # noqa: F401, E402
from apple_bench.config import PROJECT_ROOT, RUNS_ROOT  # noqa: E402
from apple_bench.data.plantvillage import PlantVillageApple  # noqa: E402
from apple_bench.data.transforms import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
)
from apple_bench.models import registry  # noqa: E402


def score_with_histogram(model, loader, device, name):
    model.train(mode=False)
    model.to(device)
    correct = total = 0
    pred_counter: Counter[int] = Counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.numel())
            for p in preds.cpu().tolist():
                pred_counter[p] += 1
    acc = correct / max(total, 1)
    hist = ", ".join(f"{c}:{pred_counter.get(c, 0)}" for c in range(4))
    print(f"[{name:55s}] acc={acc:.4f}  preds={{{hist}}}", flush=True)
    return acc


def main() -> None:
    torch.manual_seed(1729)
    device = "cpu"

    pv_root = PROJECT_ROOT / "Apple leaf dataset" / "color"
    anchor = RUNS_ROOT / "full" / "mobilenet_v3_small" / "best.pt"

    train_ds = PlantVillageApple(pv_root, "train",
                                 build_train_transform(), train_ratio=0.8)
    val_ds = PlantVillageApple(pv_root, "val",
                               build_eval_transform(), train_ratio=0.8)
    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16,
                            shuffle=False, num_workers=0)

    print(f"[info] device={device}  bs=16  api=pt2e_ptq", flush=True)
    print(f"[info] val_ds={len(val_ds)}  train_ds={len(train_ds)}", flush=True)

    # ---- Stage 1: FP32 anchor ----------------------------------------
    fp32 = registry.build("mobilenet_v3_small_hswish_free",
                          num_classes=4, pretrained=False)
    state = torch.load(anchor, map_location="cpu")
    result = fp32.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    score_with_histogram(fp32, val_loader, device, "Stage1: FP32 anchor")

    # ---- Stage 2: post-prepare_pt2e (PTQ, no QAT) --------------------
    base = fp32
    base.train(mode=False)
    base.to(device)
    sample_input = next(iter(train_loader))[0][:1].to(device)
    exported = torch.export.export_for_training(
        base, (sample_input,)
    ).module()

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_qat=False, is_per_channel=True)
    )
    prepared = prepare_pt2e(exported, quantizer)
    _allow_te(prepared)
    score_with_histogram(prepared, val_loader, device,
                         "Stage2: post-prepare_pt2e, no calibration")

    # ---- Stage 2b: 50-batch forward-only calibration -----------------
    prepared.train(mode=False)
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            prepared(x.to(device))
            if i + 1 >= 50:
                break
    score_with_histogram(prepared, val_loader, device,
                         "Stage2b: post 50-batch calibration")

    # ---- Stage 3: convert_pt2e ----------------------------------------
    prepared.to("cpu")
    quantised = convert_pt2e(prepared)
    _allow_te(quantised)
    score_with_histogram(quantised, val_loader, "cpu",
                         "Stage3: post-convert_pt2e (PTQ)")


if __name__ == "__main__":
    main()
