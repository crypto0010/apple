"""Tests for the PyTorch -> ONNX exporter."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from apple_bench.export.to_onnx import export_to_onnx
from apple_bench.models import registry


def test_onnx_export_matches_pytorch_within_tolerance(tmp_path: Path) -> None:
    model = registry.build("mobilenet_v3_small", num_classes=4, pretrained=False)
    model.train(mode=False)
    x = torch.randn(1, 3, 224, 224)

    onnx_path = tmp_path / "mnet.onnx"
    export_to_onnx(model, onnx_path, sample_input=x)
    assert onnx_path.is_file()

    with torch.no_grad():
        pt_out = model(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

    np.testing.assert_allclose(pt_out, ort_out, atol=1e-3, rtol=1e-3)
