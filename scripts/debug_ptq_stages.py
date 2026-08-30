"""Post-training static quantization (PTQ) diagnostic on qnnpack.

Counterpart to scripts/debug_qat_stages.py — uses the *non-QAT* FX path
(``prepare_fx`` + non-qat ``get_default_qconfig_mapping``) so we can
isolate whether the convert_fx collapse is QAT-specific or generic to
the legacy FX-quantization API on qnnpack.

Stages:
  1. FP32 anchor                                       (expect 1.000)
  2. After ``prepare_fx`` (PTQ observers, untrained)   (?)
  2b. After 50-batch forward-only calibration           (?)
  3. After ``convert_fx``                               (?)
"""

from __future__ import annotations

import copy
import os
from collections import Counter

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

import torch  # noqa: E402
import torch.ao.quantization as aoq  # noqa: E402
import torch.ao.quantization.quantize_fx as quantize_fx  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import apple_bench.models.mobilenet_v3_hswish_free  # noqa: F401, E402
from apple_bench.config import PROJECT_ROOT, RUNS_ROOT  # noqa: E402
from apple_bench.data.plantvillage import PlantVillageApple  # noqa: E402
from apple_bench.data.transforms import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
)
from apple_bench.models import registry  # noqa: E402


def eval_with_histogram(model, loader, device, name):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.quantized.engine = "qnnpack"

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

    print(f"[info] device={device}  bs=16", flush=True)
    print(f"[info] val_ds={len(val_ds)}  train_ds={len(train_ds)}", flush=True)

    # ---- Stage 1: FP32 anchor ----------------------------------------
    fp32 = registry.build("mobilenet_v3_small_hswish_free",
                          num_classes=4, pretrained=False)
    state = torch.load(anchor, map_location="cpu")
    result = fp32.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    eval_with_histogram(fp32, val_loader, device, "Stage1: FP32 anchor")

    # ---- Stage 2: post-prepare_fx (NOT _qat_), untrained -------------
    base = copy.deepcopy(fp32)
    base.train(mode=False)  # PTQ: frozen BN; observer updates handled by prepare_fx
    base.to(device)
    # Non-QAT qconfig mapping — uses MinMax/Histogram observers instead
    # of FakeQuantize-with-observer.
    qconfig_mapping = aoq.get_default_qconfig_mapping("qnnpack")
    sample_input = next(iter(train_loader))[0][:1].to(device)
    prepared = quantize_fx.prepare_fx(
        base, qconfig_mapping, example_inputs=(sample_input,),
    )
    eval_with_histogram(prepared, val_loader, device,
                        "Stage2: post-prepare_fx (PTQ), untrained")

    # ---- Stage 2b: 50-batch forward-only calibration -----------------
    prepared.train(mode=False)  # observers update even in eval mode for non-QAT path
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            prepared(x.to(device))
            if i + 1 >= 50:
                break
    eval_with_histogram(prepared, val_loader, device,
                        "Stage2b: post 50-batch PTQ calibration")

    # ---- Stage 3: convert_fx -----------------------------------------
    prepared.to("cpu")
    quantised = quantize_fx.convert_fx(prepared)
    eval_with_histogram(quantised, val_loader, "cpu",
                        "Stage3: PTQ convert_fx")


if __name__ == "__main__":
    main()
