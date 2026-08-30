"""Tests for Paper 3 (Ali et al. 2025) EfficientNet-B0 with GMP head."""

import torch

import apple_bench.models.paper3_efficientnet  # noqa: F401
from apple_bench.models import registry


def test_paper3_head_uses_global_max_not_average_pool():
    model = registry.build("paper3_efficientnet_b0_gmp", num_classes=4, pretrained=False)
    has_max_pool = any(
        isinstance(m, torch.nn.AdaptiveMaxPool2d) for m in model.modules()
    )
    assert has_max_pool, "Expected AdaptiveMaxPool2d (paper's GMP) somewhere in the model"


def test_paper3_outputs_4_classes():
    model = registry.build("paper3_efficientnet_b0_gmp", num_classes=4, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)
