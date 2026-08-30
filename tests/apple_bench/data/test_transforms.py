import torch
from PIL import Image

from apple_bench.data.transforms import build_eval_transform, build_train_transform


def test_eval_transform_yields_3x224x224_normalized():
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    t = build_eval_transform(image_size=224)
    out = t(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
    # Normalization makes pure grey ~ 0 after subtracting mean 0.485/0.456/0.406
    assert torch.all(out.mean(dim=(1, 2)).abs() < 0.5)


def test_train_transform_is_stochastic():
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    t = build_train_transform(image_size=224)
    torch.manual_seed(0)
    a = t(img)
    torch.manual_seed(1)
    b = t(img)
    assert not torch.equal(a, b)
