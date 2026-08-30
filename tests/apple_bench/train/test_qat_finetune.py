"""Tests for the QAT fine-tuning pipeline (Paper 2 Task 3).

DEPRECATED 2026-05-22: The pipeline under test
(``apple_bench.train.qat_finetune``) is known to produce a
constant-output INT8 model on PyTorch 2.8 + qnnpack — see the file
docstring there and ``scripts/debug_qat_stages.py``. This test only
verifies pipeline *mechanics* (returns a Module, correct shape, float
dtype) and therefore passes even when the model is degenerate; it does
not catch the semantic bug. Retained so that if a future PyTorch
release fixes ``convert_fx``, the test still serves as a smoke check.
The working alternative for INT8 deployment is
``scripts/11_int8_multiseed_calibration.py`` (vendor-native TRT PTQ).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

import apple_bench.models.mobilenet_v3_hswish_free  # noqa: F401  -- registration
from apple_bench.models import registry
from apple_bench.train.qat_finetune import qat_finetune


def test_qat_finetune_returns_callable_module():
    """qat_finetune should return a torch.nn.Module that produces float
    logits of the expected shape when called on a single image."""
    torch.manual_seed(1729)
    fp32 = registry.build(
        "mobilenet_v3_small_hswish_free",
        num_classes=4,
        pretrained=False,
    )
    # Tiny synthetic dataset so the test runs in < 90 s on CPU.
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    qat_model = qat_finetune(
        fp32_model=fp32,
        train_loader=loader,
        val_loader=loader,
        epochs=1,
        lr=1e-4,
        device="cpu",
    )

    assert isinstance(qat_model, torch.nn.Module)

    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = qat_model(sample)
    assert out.shape == (1, 4)
    assert out.dtype == torch.float32
