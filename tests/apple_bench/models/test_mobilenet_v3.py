"""Tests for MobileNetV3-Small baseline."""

import torch

import apple_bench.models.mobilenet_v3  # noqa: F401  -- triggers registration
from apple_bench.models import registry


def test_mobilenet_v3_small_outputs_4_classes():
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_mobilenet_v3_small_param_count_under_3M():
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    # MobileNetV3-Small is ~2.5M params; check we got the small variant.
    assert n_params < 3_000_000
