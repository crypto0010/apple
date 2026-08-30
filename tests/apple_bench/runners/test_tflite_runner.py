"""Tests for the TFLite host runner."""

from pathlib import Path

import numpy as np
import pytest
import torch

# Skip the whole module if neither runtime is installed.
try:
    from ai_edge_litert.interpreter import Interpreter as _Interp  # noqa: F401
except ImportError:
    pytest.importorskip("tensorflow")

pytest.importorskip("onnx2tf")  # required to produce the .tflite fixture

from apple_bench.export.to_onnx import export_to_onnx  # noqa: E402
from apple_bench.export.to_tflite import export_to_tflite_int8  # noqa: E402
from apple_bench.models import registry  # noqa: E402
from apple_bench.runners.tflite_runner import TFLiteRunner  # noqa: E402


def test_tflite_runner_returns_logits(tmp_path: Path) -> None:
    """End-to-end: ONNX -> TFLite INT8 -> host runner produces 4-class logits."""
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    onnx_path = tmp_path / "m.onnx"
    export_to_onnx(model, onnx_path, torch.randn(1, 3, 224, 224))

    rng = np.random.default_rng(0)
    calib = rng.standard_normal((16, 3, 224, 224)).astype(np.float32)
    tflite_path = tmp_path / "m.tflite"
    export_to_tflite_int8(onnx_path, tflite_path, calib)

    runner = TFLiteRunner(tflite_path)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    runner.warmup(x, n=2)
    out = runner.run(x)

    # NHWC interpreter still returns (batch, num_classes) for the final tensor.
    assert out.shape[-1] == 4
