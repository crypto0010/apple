"""Render every paper figure from the canonical CSV inputs.

Usage:
    uv run python scripts/08_make_figures.py            # render all
    uv run python scripts/08_make_figures.py --only pareto_acc_vs_latency

Reads from runs/full/ and writes PDFs (+ PNG companions) to paper/figures/,
then copies each rendered figure into every other manuscript directory whose
main.tex embeds it (see ``_distribute``).
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import traceback
from pathlib import Path

from apple_bench.config import PROJECT_ROOT, RUNS_ROOT

PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Manuscript directories that may embed a figure rendered into
# PAPER_FIGURES_DIR. Several figures (multiseed_distribution, per_arch_sigma,
# per_arch_flip_risk, ...) are included by more than one paper, each from its
# own figures/ directory. Rendering alone therefore updates only paper/, and
# the other manuscripts keep shipping whatever copy they were last given --
# which is exactly how paper2 came to embed a stale, mis-rendered figure.
# _distribute() closes that gap on every run.
PAPER_DIRS = [PROJECT_ROOT / d for d in ("paper", "paper2", "paper3")]

# Edge-CSV figures (each is `(module_name, output_stem)`).
EDGE_FIGURES = [
    ("pareto_acc_vs_latency", "pareto_acc_vs_latency"),
    ("pareto_acc_vs_energy", "pareto_acc_vs_energy"),
    ("per_format_latency_bars", "per_format_latency_bars"),
    ("flip_risk_zone", "flip_risk_zone"),
    ("pareto_tuple_parcoords", "pareto_tuple_parcoords"),
]

# Multi-seed and bisection figures (new for the revised Paper 1).
# Each entry is (module_name, output_stem).
MULTISEED_FIGURES = [
    ("multiseed_distribution", "multiseed_distribution"),
    ("stage_bisection", "stage_bisection"),
    ("pipeline_schematic", "pipeline_schematic"),
]

# Per-architecture multi-seed figures (Paper 3). Each entry is
# (module_name, output_stem). The dispatch passes runs_root so the
# generator can read all 6 architectures' summary JSONs.
PER_ARCH_FIGURES = [
    ("per_arch_sigma", "per_arch_sigma"),
    ("seed_arch_heatmap", "seed_arch_heatmap"),
    ("per_arch_flip_risk", "per_arch_flip_risk"),
]

# Parity-CSV figures.
PARITY_FIGURES = [
    ("lab_vs_field_collapse", "lab_vs_field_collapse"),
]

# Thermal figure has a different input shape (one CSV per model).
THERMAL_FIGURE = ("thermal_sustained", "thermal_sustained")

ALL_FIGURE_NAMES = (
    [name for name, _ in EDGE_FIGURES]
    + [name for name, _ in PARITY_FIGURES]
    + [name for name, _ in MULTISEED_FIGURES]
    + [name for name, _ in PER_ARCH_FIGURES]
    + [THERMAL_FIGURE[0]]
)

THERMAL_MODELS = [
    "mobilenet_v3_small",
    "efficientnet_lite0",
    "yolov8_cls_n",
    "paper3_efficientnet_b0_gmp",
    "paper4_resnet50",
    "paper2_econv_vit",
]


def _distribute(stem: str) -> list[Path]:
    """Copy a freshly rendered figure into every paper whose main.tex uses it.

    Membership is read from each manuscript's includegraphics calls rather
    than from a hand-maintained map, so adding a figure to a paper needs no
    change here. Returns the figures/ directories written to.
    """
    written: list[Path] = []
    for paper_dir in PAPER_DIRS:
        figs_dir = paper_dir / "figures"
        main_tex = paper_dir / "main.tex"
        if figs_dir == PAPER_FIGURES_DIR or not main_tex.is_file():
            continue
        if f"{stem}.pdf" not in main_tex.read_text(encoding="utf-8"):
            continue
        figs_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            src = PAPER_FIGURES_DIR / f"{stem}{suffix}"
            if src.is_file():
                shutil.copy2(src, figs_dir / src.name)
        written.append(figs_dir)
    return written


def _render(name: str, edge_csv: Path, parity_csv: Path, runs_root: Path) -> Path | None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for mod_name, stem in EDGE_FIGURES:
        if mod_name == name:
            mod = importlib.import_module(f"apple_bench.figures.{mod_name}")
            mod.from_csv(edge_csv, PAPER_FIGURES_DIR / f"{stem}.pdf")
            return PAPER_FIGURES_DIR / f"{stem}.pdf"

    for mod_name, stem in PARITY_FIGURES:
        if mod_name == name:
            mod = importlib.import_module(f"apple_bench.figures.{mod_name}")
            mod.from_csv(parity_csv, PAPER_FIGURES_DIR / f"{stem}.pdf")
            return PAPER_FIGURES_DIR / f"{stem}.pdf"

    for mod_name, stem in MULTISEED_FIGURES:
        if mod_name == name:
            mod = importlib.import_module(f"apple_bench.figures.{mod_name}")
            out_path = PAPER_FIGURES_DIR / f"{stem}.pdf"
            if mod_name == "multiseed_distribution":
                summary_path = (
                    runs_root / "mobilenet_v3_small_int8_multiseed_summary.json"
                )
                mod.from_json(summary_path, out_path)
            elif mod_name == "stage_bisection" or mod_name == "pipeline_schematic":
                mod.render(out_path)
            return out_path

    for mod_name, stem in PER_ARCH_FIGURES:
        if mod_name == name:
            mod = importlib.import_module(f"apple_bench.figures.{mod_name}")
            out_path = PAPER_FIGURES_DIR / f"{stem}.pdf"
            mod.from_summaries(runs_root, out_path)
            return out_path

    if name == THERMAL_FIGURE[0]:
        mod = importlib.import_module(f"apple_bench.figures.{name}")
        thermal_csvs = {
            m: runs_root / m / "thermal_30min.csv" for m in THERMAL_MODELS
        }
        mod.from_csvs(thermal_csvs, PAPER_FIGURES_DIR / f"{THERMAL_FIGURE[1]}.pdf")
        return PAPER_FIGURES_DIR / f"{THERMAL_FIGURE[1]}.pdf"

    raise ValueError(f"Unknown figure: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT / "full")
    parser.add_argument("--edge-csv", type=Path, default=None,
                        help="Path to jetson_orin_nano_results.csv "
                             "(defaults to --runs-root/jetson_orin_nano_results.csv).")
    parser.add_argument("--parity-csv", type=Path, default=None,
                        help="Path to parity_table.csv "
                             "(defaults to --runs-root/parity_table.csv).")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=ALL_FIGURE_NAMES,
                        help="Only render these named figures (default: all).")
    args = parser.parse_args()

    edge_csv = args.edge_csv or args.runs_root / "jetson_orin_nano_results.csv"
    parity_csv = args.parity_csv or args.runs_root / "parity_table.csv"

    if not edge_csv.is_file():
        print(f"[warn] edge CSV missing at {edge_csv} — Pareto + bar figures will be skipped.")
    if not parity_csv.is_file():
        print(f"[warn] parity CSV missing at {parity_csv} — lab-vs-field figure will be skipped.")

    names = args.only or ALL_FIGURE_NAMES
    for name in names:
        print(f"-> {name}")
        try:
            out = _render(name, edge_csv, parity_csv, args.runs_root)
            kb = out.stat().st_size // 1024 if out and out.is_file() else 0
            print(f"   {out}  ({kb} KB)")
            if out is not None:
                for figs_dir in _distribute(out.stem):
                    print(f"   -> also {figs_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"   [warn] {name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
