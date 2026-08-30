"""Paper 3 (Ali et al. 2025) reproduction: fine-tuned EfficientNet-B0 with GMP head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from apple_bench.models import registry


class _GMPHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)  # GMP per paper
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.bn(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        return self.fc2(x)


class Paper3EfficientNetGMP(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features  # convolutional trunk -> 1280 channels
        self.head = _GMPHead(in_features=1280, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


@registry.register("paper3_efficientnet_b0_gmp")
def build_paper3(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    return Paper3EfficientNetGMP(num_classes=num_classes, pretrained=pretrained)
