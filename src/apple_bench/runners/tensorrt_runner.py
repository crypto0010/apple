"""TensorRT engine runner (Jetson)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TensorRTRunner:
    name = "tensorrt"

    def __init__(self, engine_path: Path) -> None:
        # pycuda.autoinit must execute before TRT touches the device so
        # both share one CUDA context (see export/to_tensorrt.py for the
        # same gotcha during build).
        import pycuda.autoinit  # noqa: F401, PLC0415
        import pycuda.driver as cuda  # noqa: PLC0415
        import tensorrt as trt  # noqa: PLC0415

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with engine_path.open("rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # Single-input / single-output graphs from our ONNX exporter.
        # TRT 10 binds tensors by name (set_tensor_address), not by index.
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self._stream = cuda.Stream()

    def warmup(self, x: np.ndarray, n: int = 10) -> None:
        for _ in range(n):
            self.run(x)

    def run(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = np.empty(out_shape, dtype=np.float32)

        d_in = self._cuda.mem_alloc(x.nbytes)
        d_out = self._cuda.mem_alloc(out.nbytes)
        try:
            self._cuda.memcpy_htod_async(d_in, x, self._stream)
            self.context.set_tensor_address(self.input_name, int(d_in))
            self.context.set_tensor_address(self.output_name, int(d_out))
            self.context.execute_async_v3(self._stream.handle)
            self._cuda.memcpy_dtoh_async(out, d_out, self._stream)
            self._stream.synchronize()
        finally:
            d_in.free()
            d_out.free()
        return out
