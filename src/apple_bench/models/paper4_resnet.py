"""Paper 4 (Rohith et al. 2025) reproduction: ResNet50 + ImageNet pretrain."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from apple_bench.models import registry


@registry.register("paper4_resnet50")
def build_paper4_resnet50(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
