"""Paper 1 (Li et al. 2025) reproduction: handcrafted features + ACO + linear SVM.

Faithful to the descriptions in section 2.3 through 2.6 of the paper. Where
the paper omits implementation specifics (e.g., diseased-region segmentation
thresholds) we document the choice in the reproduction-report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab, rgb2ycbcr
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

N_FEATURES = 68


def _color_features(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    lab = rgb2lab(rgb)
    ycbcr = rgb2ycbcr(rgb)
    channels = [
        rgb[..., 0], rgb[..., 1], rgb[..., 2],
        hsv[..., 0], hsv[..., 1], hsv[..., 2],
        lab[..., 0], lab[..., 1], lab[..., 2],
        ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2],
    ]
    feats: list[float] = []
    for c in channels:
        feats.extend([float(c.mean()), float(c.max()), float(c.std()), float(np.median(c))])
    return np.asarray(feats, dtype=np.float64)  # 48


def _texture_features(rgb: np.ndarray) -> np.ndarray:
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_feats = [float(graycoprops(glcm, p).mean()) for p in props]

    f = gray.astype(np.float64) / 255.0
    extras = [
        float(-(f * np.log(f + 1e-12)).sum()),    # entropy
        float(f.mean()),
        float(f.std()),
        float(np.sqrt((f**2).mean())),               # RMS
        float(((f - f.mean()) ** 4).mean() / (f.std() ** 4 + 1e-12)),  # kurtosis
        float(((f - f.mean()) ** 3).mean() / (f.std() ** 3 + 1e-12)),  # skewness
        float((1.0 / (1.0 + (f - f.mean()) ** 2)).mean()),             # IDM proxy
        float(f.var()),
    ]
    return np.asarray(glcm_feats + extras, dtype=np.float64)  # 14


def _shape_features(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    diseased = ((h < 0.1) | (h > 0.9) | ((h > 0.08) & (h < 0.18))) & (s > 0.2) & (v > 0.2)

    lbl = label(diseased)
    regs = regionprops(lbl)
    if not regs:
        return np.zeros(6, dtype=np.float64)
    areas = [r.area for r in regs]
    perims = [r.perimeter for r in regs]
    majors = [r.major_axis_length for r in regs]
    minors = [r.minor_axis_length for r in regs]
    return np.asarray([
        float(len(regs)),
        float(sum(areas)),
        float(sum(perims)),
        float(np.mean(majors)),
        float(np.mean(minors)),
        float(np.mean([4 * np.pi * a / (p**2 + 1e-12) for a, p in zip(areas, perims, strict=True)])),
    ], dtype=np.float64)  # 6


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Concatenate color + texture + shape features into a 68-vector."""
    f = np.concatenate([_color_features(rgb), _texture_features(rgb), _shape_features(rgb)])
    assert f.shape == (N_FEATURES,)
    return f


@dataclass
class Paper1Pipeline:
    aco_n_ants: int = 20
    aco_n_iters: int = 50
    aco_alpha: float = 1.0
    aco_beta: float = 1.0
    aco_evaporation: float = 0.1
    svm_C: float = 1.0
    cv_folds: int = 5
    rng_seed: int = 1729

    def __post_init__(self) -> None:
        self.selected_features_: np.ndarray | None = None
        self.clf_: LinearSVC | None = None

    def _featurize(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([extract_features(img) for img in images], axis=0)

    def _fitness(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        Xs = X[:, mask.astype(bool)]
        # Clamp folds to min class count so small datasets (e.g., smoke tests)
        # don't raise ValueError from StratifiedKFold.
        min_class_count = int(np.bincount(y).min())
        n_splits = min(self.cv_folds, min_class_count)
        if n_splits < 2:
            return 0.0
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.rng_seed)
        accs: list[float] = []
        for tr, va in skf.split(Xs, y):
            clf = LinearSVC(C=self.svm_C, dual="auto").fit(Xs[tr], y[tr])
            accs.append(clf.score(Xs[va], y[va]))
        return float(np.mean(accs))

    def _aco(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.rng_seed)
        n_feat = X.shape[1]
        pheromone = np.ones(n_feat)
        best_mask, best_score = np.ones(n_feat, dtype=int), -1.0

        for _ in range(self.aco_n_iters):
            masks, scores = [], []
            for _a in range(self.aco_n_ants):
                p = pheromone ** self.aco_alpha
                prob = p / p.sum()
                mask = (rng.random(n_feat) < prob).astype(int)
                if mask.sum() == 0:
                    mask[rng.integers(n_feat)] = 1
                s = self._fitness(X, y, mask)
                masks.append(mask)
                scores.append(s)
                if s > best_score:
                    best_score, best_mask = s, mask
            pheromone *= (1.0 - self.aco_evaporation)
            for mask, s in zip(masks, scores, strict=True):
                pheromone += mask * s

        return best_mask.astype(bool)

    def fit(self, images: list[np.ndarray], y: np.ndarray) -> Paper1Pipeline:
        X = self._featurize(images)
        self.selected_features_ = self._aco(X, y)
        self.clf_ = LinearSVC(C=self.svm_C, dual="auto").fit(X[:, self.selected_features_], y)
        return self

    def predict(self, images: list[np.ndarray]) -> np.ndarray:
        if self.selected_features_ is None or self.clf_ is None:
            raise RuntimeError("Pipeline is not fitted")
        X = self._featurize(images)
        return self.clf_.predict(X[:, self.selected_features_])


def load_pil_to_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))
