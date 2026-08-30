"""MobileNetV3-Small with H-Swish replaced by ``x * ReLU6(x + 3) / 6``.

The standard ``nn.Hardswish`` lowers to the TFLite ``RELU_0_TO_1`` v1
op when traced through ``onnx2tf``; the Arduino-port TFLM libraries
(Chirale, MicroTFLite) do not ship that op kernel as of mid-2025.
Decomposing H-Swish into elementary ReLU6 + multiply preserves FP32
output exactly but emits only ops that every TFLM port supports.

Mathematical equivalence::

    h_swish(x) = x * ReLU6(x + 3) / 6

The state_dict layout of the rewritten model is identical to
torchvision's MobileNetV3-Small (only the activation *class* changes,
not its parameters), so a checkpoint trained on the standard variant
loads cleanly into this one.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from apple_bench.models import registry


class HSwishFree(nn.Module):
    """Drop-in replacement for nn.Hardswish using only ReLU6 and multiply.

    Scalar arithmetic is routed through ``nn.quantized.FloatFunctional``
    so that the module is transparent in FP32 mode (identical numerics
    to the plain arithmetic form) but compatible with eager-mode QAT,
    which otherwise rejects QUInt8/Float promotions inside ``forward``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=False)
        # Eager-QAT requires every quantised arithmetic op to go through
        # a FloatFunctional instance so the observer hooks can attach.
        self.add_three = torch.ao.nn.quantized.FloatFunctional()
        self.mul_input = torch.ao.nn.quantized.FloatFunctional()
        self.mul_sixth = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = self.add_three.add_scalar(x, 3.0)
        clipped = self.relu6(shifted)
        scaled = self.mul_input.mul(x, clipped)
        return self.mul_sixth.mul_scalar(scaled, 1.0 / 6.0)


def _replace_hardswish(module: nn.Module) -> None:
    """Recursively swap every nn.Hardswish for HSwishFree in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, HSwishFree())
        else:
            _replace_hardswish(child)


@registry.register("mobilenet_v3_small_hswish_free")
def build_mobilenet_v3_small_hswish_free(
    num_classes: int = 4, pretrained: bool = True,
) -> nn.Module:
    weights = (
        MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    _replace_hardswish(model)
    return model
