"""MobileNetV3-Small baseline (torchvision)."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from apple_bench.models import registry


@registry.register("mobilenet_v3_small")
def build_mobilenet_v3_small(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # Replace classifier head to match num_classes.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
