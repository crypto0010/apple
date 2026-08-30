"""Tests for Paper 4 (Rohith et al. 2025) ResNet50 with ImageNet pretrain."""

import torch

import apple_bench.models.paper4_resnet  # noqa: F401
from apple_bench.models import registry


def test_paper4_resnet50_outputs_4_classes():
    model = registry.build("paper4_resnet50", num_classes=4, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)
