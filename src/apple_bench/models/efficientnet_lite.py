"""EfficientNet-Lite0 baseline via timm."""

from __future__ import annotations

import timm
from torch import nn

from apple_bench.models import registry


@registry.register("efficientnet_lite0")
def build_efficientnet_lite0(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        "efficientnet_lite0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
