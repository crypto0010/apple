"""Tests for TensorRTRunner. Jetson-only (skipped on non-Jetson dev boxes)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("tensorrt")

from apple_bench.export.to_onnx import export_to_onnx  # noqa: E402
from apple_bench.export.to_tensorrt import Precision, build_trt_engine  # noqa: E402
from apple_bench.models import registry  # noqa: E402
from apple_bench.runners.tensorrt_runner import TensorRTRunner  # noqa: E402


def test_tensorrt_runner_returns_logits(tmp_path: Path) -> None:
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    onnx_path = tmp_path / "m.onnx"
    engine_path = tmp_path / "m.engine"
    export_to_onnx(model, onnx_path, torch.randn(1, 3, 224, 224))
    build_trt_engine(onnx_path, engine_path, precision=Precision.FP32)

    runner = TensorRTRunner(engine_path)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    runner.warmup(x, n=2)
    out = runner.run(x)

    assert out.shape == (1, 4)
    assert out.dtype == np.float32
