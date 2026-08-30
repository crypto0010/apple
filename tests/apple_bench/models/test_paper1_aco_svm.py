import numpy as np
from PIL import Image

from apple_bench.models.paper1_aco_svm import (
    Paper1Pipeline,
    extract_features,
)


def test_feature_vector_length_is_68():
    img = Image.new("RGB", (224, 224), color=(200, 150, 100))
    feats = extract_features(np.asarray(img))
    assert feats.shape == (68,)
    assert feats.dtype == np.float64


def test_pipeline_fit_predict_smoke():
    # 16 random images split evenly across 4 classes.
    rng = np.random.default_rng(0)
    X = [np.asarray(Image.new("RGB", (224, 224), color=tuple(rng.integers(0, 256, 3)))) for _ in range(16)]
    y = np.array([i % 4 for i in range(16)])

    pipe = Paper1Pipeline(aco_n_ants=4, aco_n_iters=3)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (16,)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})
