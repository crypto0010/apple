"""Tests that QAT-converted models export cleanly to ONNX (Paper 2 Task 7).

DEPRECATED 2026-05-22: This test was xfailed because of the
torch.onnx.export gap on QAT graphs, but the deeper issue is that
``convert_fx`` itself produces a constant-output INT8 model on
PyTorch 2.8 + qnnpack — so even with a working ONNX export, the
exported model would be useless. See
``src/apple_bench/train/qat_finetune.py`` for the full diagnosis.
The Paper 2 Task 7b follow-up (rebuild TRT engine from QAT scales
without ONNX) is therefore moot; we pivoted to vendor-native PTQ
in ``scripts/11_int8_multiseed_calibration.py``. The test remains
xfailed so a future PyTorch upgrade that fixes both gaps will
register as an xpass and trigger a revisit.

ORIGINAL KNOWN LIMITATION (still applies):
torch.onnx.export, both legacy and dynamo paths on torch 2.8, has
incomplete coverage for the FX-mode QAT graph's quantised ops.
quantized::add raises 'missing 2 required positional arguments:
op_scale and op_zero_point' under the legacy tracer; the dynamo
exporter has separate gaps on quantize/dequantize node pairs.

See also:
  - PyTorch issue https://github.com/pytorch/pytorch/issues/97417
  - TensorRT Q/DQ workflow: NVIDIA QAT toolkit / pytorch-quantization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import apple_bench.models.mobilenet_v3_hswish_free  # noqa: F401  -- registration
from apple_bench.export.to_onnx import export_to_onnx
from apple_bench.models import registry
from apple_bench.train.qat_finetune import qat_finetune


@pytest.mark.xfail(
    reason="PyTorch 2.8 torch.onnx.export does not yet support FX-mode QAT "
           "graph's quantized:: ops (e.g. quantized::add). Deferred to "
           "Task 7b: rebuild TRT engine from QAT-trained scales directly.",
    strict=False,
)
def test_qat_model_exports_to_onnx_and_runs(tmp_path: Path):
    """A converted-QAT model must export through to_onnx and run under
    ONNX Runtime, returning (1, num_classes) float logits."""
    torch.manual_seed(1729)
    fp32 = registry.build(
        "mobilenet_v3_small_hswish_free", num_classes=4, pretrained=False,
    )
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 4, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    qat_model = qat_finetune(fp32, loader, loader, epochs=1, device="cpu")
    onnx_path = tmp_path / "qat.onnx"

    export_to_onnx(
        qat_model, onnx_path, sample_input=torch.randn(1, 3, 224, 224),
    )
    assert onnx_path.is_file()

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"],
    )
    out = sess.run(
        None,
        {sess.get_inputs()[0].name:
         np.random.randn(1, 3, 224, 224).astype(np.float32)},
    )[0]
    assert out.shape == (1, 4)
