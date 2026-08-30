"""Shared matplotlib + seaborn style applied at module import."""

from __future__ import annotations

import matplotlib as mpl
import seaborn as sns

PALETTE = sns.color_palette("colorblind", n_colors=8)


# Short display labels used in figure axes and legends.  The bracketed
# author surname is the visual hook to the LaTeX \cite{} list spelled
# out in the figure caption (paper/main.tex):
#
#   [Li]     -> \cite{li2025aco}
#   [Huang]  -> \cite{huang2025econvvit}
#   [Ali]    -> \cite{ali2025efficientnet}
#   [Rohith] -> \cite{rohith2025resnet50}
#
# Baseline architectures (MobileNetV3-Small, EfficientNet-Lite0,
# YOLOv8-cls-n) are not from any one of the four 2025 source papers, so
# they carry no author tag.
DISPLAY_NAMES: dict[str, str] = {
    "mobilenet_v3_small":         "MobileNetV3-Small",
    "efficientnet_lite0":         "EfficientNet-Lite0",
    "yolov8_cls_n":               "YOLOv8-cls-n",
    "paper1_aco_svm":             "ACO+SVM [Li]",
    "paper2_econv_vit":           "EConv-ViT [Huang]",
    "paper3_efficientnet_b0_gmp": "EffNet-B0+GMP [Ali]",
    "paper4_resnet50":            "ResNet-50 [Rohith]",
}


def display_name(model_id: str) -> str:
    """Map internal model ID to human-readable figure label.

    Unknown IDs pass through unchanged so the figures still render if a
    new model is added to the CSVs before this map is updated.
    """
    return DISPLAY_NAMES.get(model_id, model_id)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def figsize(cols: float, rows: float = 0.62) -> tuple[float, float]:
    """Return (w, h) in inches for a ``cols``-column-wide figure.

    One column = 3.5 in (single-column journal width).
    ``rows`` is the height-to-width ratio.
    """
    return (3.5 * cols, 3.5 * cols * rows)
