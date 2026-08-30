"""TFLite interpreter runner (host-side sanity check before flashing to ESP32).

The ESP32 firmware uses tflite-micro directly; this host runner exists to:
  1. Verify a quantised .tflite file actually executes and produces logits.
  2. Compute INT8-vs-FP32 accuracy regression on the same eval loader the
     Jetson benchmark uses, without needing the MCU in the loop.

Layout note: onnx2tf converts NCHW (PyTorch convention) to NHWC (TF
convention).  Callers pass NCHW; the runner transposes internally so the
benchmark harness doesn't need to special-case the format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TFLiteRunner:
    name = "tflite"

    def __init__(self, tflite_path: Path) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter  # noqa: PLC0415
        except ImportError:
            from tensorflow.lite import Interpreter  # type: ignore  # noqa: PLC0415

        self.interp = Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        in_details = self.interp.get_input_details()[0]
        out_details = self.interp.get_output_details()[0]
        self.input_idx = in_details["index"]
        self.output_idx = out_details["index"]
        self.input_dtype = in_details["dtype"]
        self.input_quant = in_details.get("quantization", (0.0, 0))
        self.output_dtype = out_details["dtype"]
        self.output_quant = out_details.get("quantization", (0.0, 0))

    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        for _ in range(n):
            self.run(x)

    def run(self, x: np.ndarray) -> np.ndarray:
        # NCHW -> NHWC (only for 4D image-shaped tensors).
        if x.ndim == 4 and x.shape[1] in (1, 3):
            x = np.transpose(x, (0, 2, 3, 1)).copy()
        if self.input_dtype == np.int8:
            scale, zero = self.input_quant
            x = (x / scale + zero).round().clip(-128, 127).astype(np.int8)
        self.interp.set_tensor(self.input_idx, x)
        self.interp.invoke()
        out = self.interp.get_tensor(self.output_idx)
        # Dequantise INT8 logits back to float so callers can argmax/compare.
        if self.output_dtype == np.int8:
            scale, zero = self.output_quant
            out = (out.astype(np.float32) - zero) * scale
        return out
