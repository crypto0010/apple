"""ONNX -> TensorRT engine builder. Supports FP32, FP16, INT8 (with calibration).

This module imports tensorrt lazily so that non-Jetson dev environments can
still import the package for testing other modules.

INT8 builds dispatch to one of three calibrator algorithms selected by the
``int8_calibrator`` argument: entropy (KL-divergence; the TensorRT default),
min-max (per-tensor clip to observed min/max), or percentile (quantile-based
clip via the legacy calibrator interface). The entropy path is byte-identical
to the original single-calibrator implementation so prior runs reproduce
without change.
"""

from __future__ import annotations

import enum
from pathlib import Path

import numpy as np


class Precision(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class Calibrator(enum.Enum):
    """INT8 calibration algorithm.

    ENTROPY:    TensorRT IInt8EntropyCalibrator2 (KL-divergence).
    MINMAX:     TensorRT IInt8MinMaxCalibrator (per-tensor clip to min/max).
    PERCENTILE: TensorRT IInt8LegacyCalibrator, driven by a quantile
                (default 0.9999) *and* a regression cutoff (default 1.0).

                Two caveats, because the label oversells what this arm
                does. The legacy interface is not a pure percentile clip:
                it combines get_quantile() with get_regression_cutoff(),
                and a cutoff of 1.0 collapses the regression onto the
                observed maximum. Results from this arm therefore
                characterise the legacy calibrator as configured here,
                not an idealised 99.99-th percentile clip.

                IInt8LegacyCalibrator is also deprecated as of TensorRT
                10.1 ("Superseded by explicit quantization" -- see the
                class docstring of the installed tensorrt package), so
                this path is on an upstream removal track.
    """

    ENTROPY = "entropy"
    MINMAX = "minmax"
    PERCENTILE = "percentile"


PERCENTILE_DEFAULT_QUANTILE = 0.9999
# 1.0 means "regress up to the observed maximum", which degenerates the
# legacy calibrator's regression step. Changing this changes what the
# `percentile` arm measures; see the Calibrator.PERCENTILE docstring.
PERCENTILE_DEFAULT_REGRESSION_CUTOFF = 1.0


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: Precision = Precision.FP32,
    int8_calibration_data: np.ndarray | None = None,
    int8_calibrator: Calibrator = Calibrator.ENTROPY,
    workspace_mb: int = 2048,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> None:
    """Build a TensorRT engine from an ONNX file.

    ``input_shape`` is the (batch, C, H, W) shape used for the engine's
    sole optimization profile (min == opt == max). All Phase A.2 benchmarks
    use batch=1, so a single-shape profile is correct; widen if higher
    batch sizes need to be timed later.

    ``int8_calibrator`` only takes effect when ``precision`` is
    ``Precision.INT8``; it is ignored for FP32 and FP16 builds.
    """
    # pycuda.autoinit must run BEFORE tensorrt creates its first builder, so
    # that both share a single CUDA context. Otherwise TRT prints
    # "CUDA context changed between createInferBuilder and
    # buildSerializedNetwork" and silently fails the INT8 build path.
    if precision is Precision.INT8:
        import pycuda.autoinit  # noqa: F401, PLC0415

    import tensorrt as trt  # noqa: PLC0415  -- lazy import (Jetson-only)

    if precision is Precision.INT8 and int8_calibration_data is None:
        raise ValueError("INT8 precision requires int8_calibration_data")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )

    # TRT 10 requires an explicit optimization profile whenever the network
    # has dynamic axes (our ONNX exporter declares dim 0 as "batch").
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min=input_shape, opt=input_shape, max=input_shape)
    config.add_optimization_profile(profile)

    if precision is Precision.FP16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision is Precision.INT8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_int8_calibrator(
            int8_calibration_data, input_shape, int8_calibrator
        )
        config.set_calibration_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_path.open("wb") as f:
        f.write(serialized)


def _make_int8_calibrator(
    data: np.ndarray,
    input_shape: tuple[int, int, int, int],
    calibrator_type: Calibrator,
):
    """Create an INT8 calibrator over an in-memory calibration set.

    Dispatches on ``calibrator_type`` to one of three TensorRT calibrator
    interfaces (entropy / min-max / percentile-via-legacy; the last is
    deprecated upstream since TensorRT 10.1). All three share
    the same per-batch device-memory plumbing; only the base class differs.

    Caller is responsible for ensuring pycuda.autoinit has already executed
    (build_trt_engine does this when precision == INT8).
    """
    import pycuda.driver as cuda  # noqa: PLC0415
    import tensorrt as trt  # noqa: PLC0415

    base_class = {
        Calibrator.ENTROPY: trt.IInt8EntropyCalibrator2,
        Calibrator.MINMAX: trt.IInt8MinMaxCalibrator,
        Calibrator.PERCENTILE: trt.IInt8LegacyCalibrator,
    }[calibrator_type]

    class _Calibrator(base_class):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.data = data.astype(np.float32)
            self.batch_size = 1
            self.idx = 0
            self.device_input = cuda.mem_alloc(self.data[0].nbytes)

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names):
            if self.idx >= len(self.data):
                return None
            cuda.memcpy_htod(self.device_input, self.data[self.idx])
            self.idx += 1
            return [int(self.device_input)]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            pass

        # IInt8LegacyCalibrator requires two additional methods that the
        # entropy / min-max base classes do not declare. Implementing them
        # unconditionally is harmless on the other two subclasses because
        # the base class never calls them.
        def get_quantile(self) -> float:
            return PERCENTILE_DEFAULT_QUANTILE

        def get_regression_cutoff(self) -> float:
            return PERCENTILE_DEFAULT_REGRESSION_CUTOFF

        def read_histogram_cache(self, length):  # noqa: ARG002
            return None

        def write_histogram_cache(self, ptr, length):  # noqa: ARG002
            pass

    return _Calibrator()
