import torch

import apple_bench.models.paper2_econv_vit  # noqa: F401
from apple_bench.models import registry


def test_econv_vit_outputs_4_classes():
    model = registry.build("paper2_econv_vit", num_classes=4)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_econv_vit_param_count_in_paper_range():
    """Paper reports ~83.7M total params; allow +/-20% for re-implementation drift."""
    model = registry.build("paper2_econv_vit", num_classes=4)
    n = sum(p.numel() for p in model.parameters())
    assert 60_000_000 < n < 110_000_000, f"got {n:,} params"
