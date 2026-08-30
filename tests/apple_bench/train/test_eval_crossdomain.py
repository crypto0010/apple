import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apple_bench.train.eval_crossdomain import score_with_ci


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Always predicts class 0 -- useful for verifying counts.
        self.bias = nn.Parameter(torch.tensor([10.0, 0.0, 0.0, 0.0]))

    def forward(self, x):
        return self.bias.expand(x.shape[0], 4)


def test_score_with_ci_writes_report(tmp_path: Path):
    x = torch.randn(20, 3, 8, 8)
    y = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    out = tmp_path / "report.json"
    result = score_with_ci(_StubModel(), loader, num_classes=4, device="cpu", output_path=out, n_bootstrap=200)

    assert out.is_file()
    report = json.loads(out.read_text())
    # The return value is part of the contract, not just the written file.
    assert result == report
    assert report["accuracy"] == 0.25       # only the 5 true-class-0 samples are correct
    assert "ci_lower" in report and "ci_upper" in report
    assert len(report["confusion_matrix"]) == 4
