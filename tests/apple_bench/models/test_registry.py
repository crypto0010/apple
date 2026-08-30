import pytest

from apple_bench.models import registry


def test_register_and_build():
    @registry.register("toy")
    def _build_toy(num_classes: int = 4):
        import torch.nn as nn
        return nn.Linear(8, num_classes)

    model = registry.build("toy", num_classes=4)
    assert model.out_features == 4


def test_double_register_raises():
    @registry.register("dup")
    def _b1(num_classes: int = 4):
        import torch.nn as nn
        return nn.Linear(1, num_classes)

    with pytest.raises(ValueError, match="already registered"):
        @registry.register("dup")
        def _b2(num_classes: int = 4):
            import torch.nn as nn
            return nn.Linear(1, num_classes)


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        registry.build("does-not-exist")
