"""Tests for EfficientNet-Lite0 baseline."""

import torch

import apple_bench.models.efficientnet_lite  # noqa: F401
from apple_bench.models import registry


def test_efficientnet_lite0_outputs_4_classes():
    model = registry.build("efficientnet_lite0", num_classes=4, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)
