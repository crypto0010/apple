"""Stage-bisection diagnostic for the pt2e QAT pipeline on MobileNetV3-Small.

Counterpart to ``scripts/debug_qat_stages.py``. Uses the
``torch.ao.quantization.quantize_pt2e`` API (PyTorch 2.8's deprecation-
flagged replacement for ``quantize_fx``) together with
``XNNPACKQuantizer`` to test whether the convert_fx collapse documented
in Paper 2 reproduces under the pt2e capture mechanism.

API mapping versus the legacy FX path:
    quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
        -> quantize_pt2e.prepare_qat_pt2e(
               torch.export.export_for_training(model, example_inputs).module(),
               XNNPACKQuantizer().set_global(
                   get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
               ),
           )
    quantize_fx.convert_fx(prepared)
        -> quantize_pt2e.convert_pt2e(prepared)

Stages mirror debug_qat_stages.py exactly so the two bisections can be
read side-by-side:

  S1  : FP32 anchor                                       (expect 1.000)
  S2  : post-prepare_qat_pt2e, untrained                  (?)
  S2b : post 50-batch forward-only calibration            (?)
  S3  : post-convert_pt2e, no training                    (?)
  S4  : post 1-epoch QAT training, before convert         (?)
  S5  : post 1-epoch QAT training and after convert       (?)
"""

from __future__ import annotations

import copy
import os
import warnings
from collections import Counter

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256",
)

# Silence the deprecation warnings from torch.ao.quantization itself
# so the per-stage output is readable. They're the whole reason this
# diagnostic exists; we don't need them logged 20 times each.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="torch.ao.quantization",
)
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=".*XNNPACKQuantizer is deprecated.*",
)

import torch  # noqa: E402
import torch.export  # noqa: E402
from torch import nn  # noqa: E402
from torch.ao.quantization import (  # noqa: E402
    allow_exported_model_train_eval,
    move_exported_model_to_eval,
)
from torch.ao.quantization.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_qat_pt2e,
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
    # pt2e's prepare_qat_pt2e refuses mixed-device modules; the exported
    # graph contains both CUDA params and CPU constants on PyTorch 2.8.
    # Calling exported.to('cuda') does not move embedded prim::Constant
    # tensors. We therefore run the entire pt2e flow on CPU for now.
    # Training on CPU is ~5x slower than CUDA on this model but the
    # bisection only needs one epoch.
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

    print(f"[info] device={device}  bs=16  api=pt2e", flush=True)
    print(f"[info] val_ds={len(val_ds)}  train_ds={len(train_ds)}", flush=True)

    # ---- Stage 1: FP32 anchor ----------------------------------------
    fp32 = registry.build("mobilenet_v3_small_hswish_free",
                          num_classes=4, pretrained=False)
    state = torch.load(anchor, map_location="cpu")
    result = fp32.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    eval_with_histogram(fp32, val_loader, device, "Stage1: FP32 anchor")

    # ---- Stage 2: post-prepare_qat_pt2e, untrained -------------------
    base = copy.deepcopy(fp32)
    base.train()
    base.to(device)
    # torch.export.export_for_training captures the graph in a way that
    # preserves training-mode semantics (BN updates, dropout).  Critically,
    # it does NOT use FX's symbolic_trace, so it should handle MobileNetV3's
    # SqueezeExcitation + Hardsigmoid path differently from the legacy
    # quantize_fx pipeline.
    sample_input = next(iter(train_loader))[0][:1].to(device)
    exported = torch.export.export_for_training(
        base, (sample_input,)
    ).module()

    # Per-channel weights is the MobileNetV3 mitigation Paper 2's R1.3
    # called out; turn it on so this experiment is the most-favourable
    # comparison to the legacy per-tensor qnnpack default.
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    )
    prepared = prepare_qat_pt2e(exported, quantizer)
    # Exported modules reject direct .train(mode=False) calls; this
    # helper rewires the methods so the standard pattern keeps working.
    allow_exported_model_train_eval(prepared)
    eval_with_histogram(prepared, val_loader, device,
                        "Stage2: post-prepare_qat_pt2e, untrained")

    # ---- Stage 2b: 50-batch forward-only calibration -----------------
    prepared.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            prepared(x.to(device))
            if i + 1 >= 50:
                break
    eval_with_histogram(prepared, val_loader, device,
                        "Stage2b: post 50-batch calibration")

    # ---- Stage 3: SKIPPED ---------------------------------------------
    # The natural Stage 3 measurement here would be ``convert_pt2e`` on a
    # deepcopy of the calibrated-but-untrained ``prepared``, so the
    # original ``prepared`` could continue into Stage 4 training.
    # ``copy.deepcopy(prepared)`` discards the FX-graph node metadata
    # that ``convert_pt2e`` relies on (specifically
    # ``node.meta['source_fn_stack']``, consumed by ``_fold_conv_bn_qat``)
    # and the convert step then raises ``KeyError: 'source_fn_stack'``.
    # Stage 5 (training + convert via the linear in-place path) answers
    # the same load-bearing question --- "does ``convert_pt2e`` produce
    # a non-degenerate INT8 model on this anchor?" --- without requiring
    # a deepcopy. We therefore skip Stage 3 in this diagnostic and rely
    # on Stage 5.

    # ---- Stage 4: train 1 epoch, eval pre-convert --------------------
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
    # ``prepared.train(mode=False)`` via the train_eval shim is not the
    # same as actually moving the exported model to inference mode for
    # the BN-fold pass: convert_pt2e's ``_fold_conv_bn_qat`` reads
    # ``node.meta['source_fn_stack']`` which only exists on inference-
    # mode BN nodes. Use the explicit pt2e helper instead.
    move_exported_model_to_eval(prepared)
    prepared.to("cpu")
    quantised_trained = convert_pt2e(prepared)
    allow_exported_model_train_eval(quantised_trained)
    eval_with_histogram(quantised_trained, val_loader, "cpu",
                        "Stage5: trained 1ep + convert_pt2e")


if __name__ == "__main__":
    main()
