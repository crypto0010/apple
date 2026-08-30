"""Tests for ONNX -> TensorRT engine builder. Jetson-only."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("tensorrt")  # skip on non-Jetson dev boxes

from apple_bench.export.to_onnx import export_to_onnx  # noqa: E402
from apple_bench.export.to_tensorrt import Precision, build_trt_engine  # noqa: E402
from apple_bench.models import registry  # noqa: E402


def test_build_fp32_engine(tmp_path: Path) -> None:
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    onnx_path = tmp_path / "m.onnx"
    export_to_onnx(model, onnx_path, torch.randn(1, 3, 224, 224))

    engine_path = tmp_path / "m.engine"
    build_trt_engine(onnx_path, engine_path, precision=Precision.FP32)
    assert engine_path.is_file()
    assert engine_path.stat().st_size > 100_000


def test_build_int8_engine_with_calibrator(tmp_path: Path) -> None:
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    onnx_path = tmp_path / "m.onnx"
    export_to_onnx(model, onnx_path, torch.randn(1, 3, 224, 224))

    rng = np.random.default_rng(0)
    calib_data = rng.standard_normal((64, 3, 224, 224)).astype(np.float32)

    engine_path = tmp_path / "m_int8.engine"
    build_trt_engine(
        onnx_path,
        engine_path,
        precision=Precision.INT8,
        int8_calibration_data=calib_data,
    )
    assert engine_path.is_file()
