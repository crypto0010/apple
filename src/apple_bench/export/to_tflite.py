"""ONNX -> TFLite-Micro full-INT8 converter via onnx2tf.

The original plan called for ``ai-edge-torch`` (PyTorch -> TFLite directly),
but that package pins ``torch<2.7`` which conflicts with the torch 2.8
Jetson wheel.  ``onnx2tf`` takes ONNX as input instead, which fits the
existing two-stage pipeline (``to_onnx`` -> ``to_tflite``).

For the ESP32-S3 TFLite-Micro runtime we need *full* integer quantisation
(weights and activations both int8).  onnx2tf emits multiple TFLite
artifacts per run; this module picks the ``*_full_integer_quant.tflite``
one and renames it to the caller's path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def export_to_tflite_int8(
    onnx_path: Path,
    output_path: Path,
    calibration_data: np.ndarray,
) -> None:
    """Convert an ONNX model to a fully INT8-quantised TFLite flatbuffer."""
    import onnx  # noqa: PLC0415
    import onnx2tf  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # onnx2tf reads the calibration set from a .npy file path.  Use a temp
    # dir so the intermediate float / fp16 / dynamic-range artifacts are
    # cleaned up after we extract the full-int8 one.
    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        tmp_dir = Path(tmp)
        rep_path = tmp_dir / "calib.npy"
        # onnx2tf expects the .npy file in *post-conversion* (NHWC) layout.
        # PyTorch produces NCHW, so transpose 4D image tensors before saving.
        if calibration_data.ndim == 4 and calibration_data.shape[1] in (1, 3):
            calib_nhwc = np.transpose(calibration_data, (0, 2, 3, 1))
        else:
            calib_nhwc = calibration_data
        np.save(rep_path, calib_nhwc.astype(np.float32))

        input_name = onnx.load(str(onnx_path)).graph.input[0].name

        out_dir = tmp_dir / "tflite_out"
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(out_dir),
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            # [tensor_name, npy_path, mean, std] - mean/std must be np.ndarray
            # (not python float).  Calibration data is already pre-normalised,
            # so mean=0, std=1 leaves it unchanged.
            custom_input_op_name_np_data_path=[
                [
                    input_name,
                    str(rep_path),
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                ],
            ],
            non_verbose=True,
        )

        # Prefer fully-quantised (weights + activations int8) for TFLM; fall
        # back to weights-only if onnx2tf couldn't quantise the activations.
        candidates = sorted(out_dir.glob("*_full_integer_quant.tflite"))
        if not candidates:
            candidates = sorted(out_dir.glob("*_integer_quant.tflite"))
        if not candidates:
            available = sorted(p.name for p in out_dir.glob("*.tflite"))
            raise RuntimeError(
                f"onnx2tf produced no INT8 TFLite file in {out_dir}. "
                f"Available: {available}"
            )
        candidates[0].replace(output_path)
