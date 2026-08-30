"""One-shot diagnostic: localise the QAT collapse to a specific pipeline stage.

Mirrors scripts/10_train_qat.py + src/apple_bench/train/qat_finetune.py
faithfully, but inserts val-accuracy + per-class-prediction evaluations
at every stage boundary so the failing step can be pinpointed.

Stages:
  1. FP32 anchor loaded into H-Swish-free model           (expect 1.000)
  2. prepare_qat_fx done, observers still at defaults     (?)
  2b. After 50 forward-only batches of calibration         (?)
  3. convert_fx on calibrated-but-untrained prepared      (?)
  4. After 1 epoch of QAT training, before convert        (?)
  5. After 1 epoch of QAT training, after convert         (matches smoke test)

Per-stage we print prediction histogram across the 4 classes so that
"collapses to constant output" is visible without separate analysis.
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
from torch import nn  # noqa: E402
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
    """Top-1 accuracy + per-class prediction histogram."""
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

    # ---- Stage 1: FP32 anchor on H-Swish-free model ------------------
    fp32 = registry.build("mobilenet_v3_small_hswish_free",
                          num_classes=4, pretrained=False)
    state = torch.load(anchor, map_location="cpu")
    result = fp32.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, (
        f"anchor load issue: missing={result.missing_keys[:3]} "
        f"unexpected={result.unexpected_keys[:3]}"
    )
    eval_with_histogram(fp32, val_loader, device, "Stage1: FP32 anchor")

    # ---- Stage 2: post-prepare, untrained ----------------------------
    base = copy.deepcopy(fp32)
    base.train()
    base.to(device)
    qconfig_mapping = aoq.get_default_qat_qconfig_mapping("qnnpack")
    sample_input = next(iter(train_loader))[0][:1].to(device)
    prepared = quantize_fx.prepare_qat_fx(
        base, qconfig_mapping, example_inputs=(sample_input,),
    )
    eval_with_histogram(prepared, val_loader, device,
                        "Stage2: post-prepare, untrained")

    # ---- Stage 2b: post 50-batch forward-only calibration ------------
    prepared.train()  # observers active, BN active
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            prepared(x.to(device))
            if i + 1 >= 50:
                break
    eval_with_histogram(prepared, val_loader, device,
                        "Stage2b: post 50-batch calibration")

    # ---- Stage 3: convert_fx without any training --------------------
    pre_for_calib_convert = copy.deepcopy(prepared)
    pre_for_calib_convert.train(mode=False)
    pre_for_calib_convert.to("cpu")
    quantised_calib_only = quantize_fx.convert_fx(pre_for_calib_convert)
    eval_with_histogram(quantised_calib_only, val_loader, "cpu",
                        "Stage3: convert (calibration only, no training)")

    # ---- Stage 4: train 1 epoch, then eval pre-convert ---------------
    prepared.train()
    prepared.to(device)
    optim = torch.optim.AdamW(prepared.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        loss = criterion(prepared(x), y)
        loss.backward()
        optim.step()
    eval_with_histogram(prepared, val_loader, device,
                        "Stage4: trained 1ep, before convert")

    # ---- Stage 5: convert after training ------------------------------
    prepared.train(mode=False)
    prepared.to("cpu")
    quantised_trained = quantize_fx.convert_fx(prepared)
    eval_with_histogram(quantised_trained, val_loader, "cpu",
                        "Stage5: trained 1ep + convert (matches smoke)")


if __name__ == "__main__":
    main()
