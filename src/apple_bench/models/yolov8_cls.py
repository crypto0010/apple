"""YOLOv8 classification head (n / nano variant) via ultralytics."""

from __future__ import annotations

from torch import nn
from ultralytics import YOLO

from apple_bench.models import registry


class YOLOv8ClsWrapper(nn.Module):
    """Wraps YOLOv8 classification Sequential model to expose clean forward pass.

    The ultralytics YOLO wrapper adds .predict() logic. This wrapper extracts just
    the Sequential model and provides a standard forward pass.
    """

    def __init__(self, sequential_model: nn.Sequential) -> None:
        super().__init__()
        self.model = sequential_model

    def forward(self, x):
        # Call the Sequential module directly.
        return self.model(x)


@registry.register("yolov8_cls_n")
def build_yolov8_cls_n(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    # Load ultralytics YOLO model.
    yolo = YOLO("yolov8n-cls.pt" if pretrained else "yolov8n-cls.yaml")
    # Extract the underlying Sequential model.
    inner_seq: nn.Sequential = yolo.model.model
    # Replace final Classify head to match num_classes.
    from ultralytics.nn.modules.head import Classify  # noqa: PLC0415
    old_classify = inner_seq[-1]
    if not isinstance(old_classify, Classify):
        raise RuntimeError(f"Expected ultralytics Classify head; got {type(old_classify)}")
    # Get the input channels to the classify head from the previous layer's output.
    # The Classify head's conv layer takes from the backbone's last output (256 for nano).
    in_ch = old_classify.conv.conv.in_channels
    # Create new Classify with the correct input channels and desired num_classes.
    new_classify = Classify(in_ch, num_classes)
    inner_seq[-1] = new_classify
    return YOLOv8ClsWrapper(inner_seq)
