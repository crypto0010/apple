"""Tests for YOLOv8-cls-n baseline."""

import torch

import apple_bench.models.yolov8_cls  # noqa: F401
from apple_bench.models import registry


def test_yolov8_cls_n_outputs_4_classes():
    # pretrained=True is acceptable here even though it requires network — the
    # checkpoint is small (~6 MB) and gets cached for subsequent test runs.
    model = registry.build("yolov8_cls_n", num_classes=4)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape[0] == 2
    assert out.shape[-1] == 4
