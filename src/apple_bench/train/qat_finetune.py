"""Quantisation-aware training (QAT) fine-tuner using FX-graph mode.

DEPRECATED 2026-05-22: PyTorch 2.8's ``convert_fx`` on the qnnpack
backend produces a constant-output INT8 model for MobileNetV3-Small,
reproducible deterministically via ``scripts/debug_qat_stages.py``
(~5 min). The bug is isolated to ``convert_fx`` itself — verified by
``scripts/debug_ptq_stages.py`` showing the same collapse via the
non-QAT ``prepare_fx`` + ``convert_fx`` path. The legacy
``torch.ao.quantization`` API is also deprecation-flagged upstream
(PyTorch's own warning recommends migrating to ``torchao.pt2e``).
The working alternative for Jetson INT8 deployment is the vendor
path in ``scripts/11_int8_multiseed_calibration.py`` (FP32 -> ONNX ->
TRT INT8 with EntropyCalibrator2). This file is retained as
historical evidence for Paper 2's discussion of "what we tried" and
to re-validate if a future PyTorch upgrade restores the FX path.

Pipeline:

  1. ``prepare_qat_fx`` traces the FP32 model into an FX graph and
     inserts fake-quant observer nodes against the default qnnpack
     QConfig (per-tensor symmetric weights and per-tensor activations;
     this is the recipe ARM/Xtensa MCU runtimes consume natively).
  2. The traced model is fine-tuned on the supplied loader; gradients
     flow through the straight-through estimator built into the
     fake-quant nodes.
  3. ``convert_fx`` materialises a true INT8 model.

FX-graph mode is preferred over eager-mode quantisation because the
torchvision MobileNetV3 was not designed for eager-mode quantisation
boundaries: its head (adaptive pool + flatten + classifier) triggers
``empty_strided not supported on quantized tensors`` (PyTorch issue
74540) when wrapped naively with QuantStub/DeQuantStub. The FX path
analyses the graph and inserts quant/dequant ops only where they are
safe.
"""

from __future__ import annotations

import copy

import torch
import torch.ao.quantization as aoq
import torch.ao.quantization.quantize_fx as quantize_fx
from torch import nn
from torch.utils.data import DataLoader


def qat_finetune(
    fp32_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = "cuda",
    qconfig_backend: str = "qnnpack",
) -> nn.Module:
    """Fine-tune ``fp32_model`` under QAT and return the converted INT8 model.

    ``qconfig_backend`` selects the observer recipe.  ``qnnpack``
    uses per-tensor symmetric weight quantisation (verified directly
    via ``aoq.get_default_qat_qconfig_mapping('qnnpack').global_qconfig.weight``,
    which carries ``qscheme=torch.per_tensor_symmetric``); this is
    the recipe qnnpack's int8 kernels consume on ARM/Xtensa MCU
    targets. Per-tensor weight scaling is known to underperform
    per-channel on depthwise-separable convolutions; switching to
    ``fbgemm`` would give per-channel weights but is not an option
    on the Jetson L4T wheel
    (``torch.backends.quantized.supported_engines == ['qnnpack']``).
    """
    # The quantised kernels (qconv2d_prepack, etc.) are dispatched against
    # torch.backends.quantized.engine; without this set, convert_fx later
    # fails with "Didn't find engine for operation ... NoQEngine".
    torch.backends.quantized.engine = qconfig_backend

    base = copy.deepcopy(fp32_model)
    base.train()
    base.to(device)

    # Use the official default-mapping helper rather than constructing the
    # QConfigMapping manually; it provides the FixedQParams observers that
    # MobileNetV3's H-Swish-adjacent ops need (and warns otherwise).
    qconfig_mapping = aoq.get_default_qat_qconfig_mapping(qconfig_backend)

    # FX-mode QAT requires a sample input to trace the graph.
    sample_input = next(iter(train_loader))[0][:1].to(device)
    prepared = quantize_fx.prepare_qat_fx(base, qconfig_mapping,
                                          example_inputs=(sample_input,))

    optim = torch.optim.AdamW(prepared.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _epoch in range(epochs):
        prepared.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            loss = criterion(prepared(x), y)
            loss.backward()
            optim.step()

    prepared.train(mode=False)
    prepared.to("cpu")  # quantize_fx.convert_fx runs on CPU
    quantised = quantize_fx.convert_fx(prepared)
    return quantised
