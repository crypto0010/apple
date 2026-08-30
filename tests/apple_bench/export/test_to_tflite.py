"""Tests for the ONNX -> TFLite-Micro INT8 exporter."""

from pathlib import Path

import numpy as np
import pytest
import torch

# Skip the whole module if onnx2tf is not installed (e.g., on dev boxes
# without the heavy TensorFlow dependency tree).
pytest.importorskip("onnx2tf")

from apple_bench.export.to_onnx import export_to_onnx  # noqa: E402
from apple_bench.export.to_tflite import export_to_tflite_int8  # noqa: E402
from apple_bench.models import registry  # noqa: E402


def test_tflite_int8_export_under_3mb(tmp_path: Path) -> None:
    """MobileNetV3-Small after INT8 quant should fit ESP32-S3-CAM's budget."""
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    onnx_path = tmp_path / "mnet.onnx"
    export_to_onnx(model, onnx_path, torch.randn(1, 3, 224, 224))

    # 32 synthetic preprocessed images for the representative dataset.
    rng = np.random.default_rng(0)
    calib = rng.standard_normal((32, 3, 224, 224)).astype(np.float32)

    out_path = tmp_path / "mnet.int8.tflite"
    export_to_tflite_int8(onnx_path, out_path, calibration_data=calib)

    assert out_path.is_file(), "TFLite file was not written"
    size = out_path.stat().st_size
    assert size < 3 * 1024 * 1024, (
        f"INT8 MobileNetV3-Small should be <3 MB; got {size} bytes "
        "(over MCU PSRAM budget)"
    )
