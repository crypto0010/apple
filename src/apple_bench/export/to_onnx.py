"""PyTorch -> ONNX exporter.

Prefers the dynamo-based exporter (torch>=2.5) for cleaner transformer
graphs; falls back to the legacy tracer if dynamo refuses the graph (some
ultralytics heads contain control flow it can't yet capture).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    sample_input: torch.Tensor,
    opset: int = 17,
) -> None:
    model.train(mode=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (sample_input,),
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=True,
        )
    except Exception:
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )
