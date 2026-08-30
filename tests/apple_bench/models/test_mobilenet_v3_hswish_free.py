"""Tests for the H-Swish-free MobileNetV3-Small variant (Paper 2 Task 1).

The H-Swish-free variant exists so the model exports to TFLite-Micro
without emitting the RELU_0_TO_1 v1 op that the current Arduino-port
TFLM libraries (Chirale, MicroTFLite) cannot resolve.  H-Swish is
replaced by its mathematically equivalent decomposition
``x * ReLU6(x + 3) / 6``; in FP32 the outputs are identical, so this
test pins that equivalence.
"""

from __future__ import annotations

import numpy as np
import torch

# Side-effect imports register both variants with the registry.
import apple_bench.models.mobilenet_v3  # noqa: F401
import apple_bench.models.mobilenet_v3_hswish_free  # noqa: F401
from apple_bench.models import registry


def test_hswish_free_matches_reference_fp32():
    """The H-Swish-free variant must produce numerically identical FP32
    logits to the standard MobileNetV3-Small for the same input when
    initialised with the same state_dict."""
    ref = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    free = registry.build(
        "mobilenet_v3_small_hswish_free", num_classes=4, pretrained=False,
    )
    # The two variants differ only in the activation module class; the
    # parameter layout is identical, so the state_dict copies cleanly.
    free.load_state_dict(ref.state_dict())
    ref.train(mode=False)
    free.train(mode=False)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref_out = ref(x)
        free_out = free(x)

    np.testing.assert_allclose(
        ref_out.numpy(), free_out.numpy(), atol=1e-5, rtol=1e-5
    )


def test_hswish_free_uses_no_hardswish_module():
    """The H-Swish-free variant must contain zero nn.Hardswish modules."""
    from torch.nn import Hardswish
    free = registry.build(
        "mobilenet_v3_small_hswish_free", num_classes=4, pretrained=False,
    )
    for m in free.modules():
        assert not isinstance(m, Hardswish), (
            f"Found nn.Hardswish in H-Swish-free variant: {type(m).__name__}"
        )


def test_hswish_free_param_count_matches_baseline():
    """Param count is identical because only the activation class changes."""
    ref = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    free = registry.build(
        "mobilenet_v3_small_hswish_free", num_classes=4, pretrained=False,
    )
    n_ref = sum(p.numel() for p in ref.parameters())
    n_free = sum(p.numel() for p in free.parameters())
    assert n_ref == n_free
