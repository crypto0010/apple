"""ONNX Runtime inference runner (CUDA EP on Jetson, CPU EP elsewhere)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXRunner:
    name = "onnxruntime"

    def __init__(
        self,
        onnx_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        for _ in range(n):
            self.sess.run(None, {self.input_name: x})

    def run(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]
