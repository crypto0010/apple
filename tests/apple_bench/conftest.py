"""Shared pytest fixtures."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _set_seeds():
    """Reset RNG state before each test for determinism."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
