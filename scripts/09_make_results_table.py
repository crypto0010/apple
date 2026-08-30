"""Render the headline comparative table as Markdown + LaTeX.

Outputs:
  paper/tables/comparative.md   — for review and PR diffs
  paper/tables/comparative.tex  — \\begin{tabular}...\\end{tabular}
                                  body for direct \\input{} in the paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apple_bench.analysis.comparative import (
    ComparativeRow,
    build_comparative_table,
)
from apple_bench.config import PROJECT_ROOT, RUNS_ROOT

PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

# LaTeX-side display names and cite keys for the comparative table.
# The cite keys correspond to \bibitem entries in paper/main.tex.
# Baselines (mobilenet_v3_small, efficientnet_lite0, yolov8_cls_n) are
# not in this map because the comparative table is per source paper only.
LATEX_NAMES: dict[str, tuple[str, str | None]] = {
    # internal id : (display name, cite key or None)
    "paper1_aco_svm":             ("ACO+SVM",                 "li2025aco"),
    "paper2_econv_vit":           ("EConv-ViT",               "huang2025econvvit"),
    "paper3_efficientnet_b0_gmp": ("EfficientNet-B0+GMP",     "ali2025efficientnet"),
    "paper4_resnet50":            ("ResNet-50 (fine-tuned)",  "rohith2025resnet50"),
}


def _latex_label(model_id: str) -> str:
    """Display name with \\cite{} appended; falls back to escaped id."""
    if model_id in LATEX_NAMES:
        name, cite = LATEX_NAMES[model_id]
        if cite:
            return f"{name}~\\cite{{{cite}}}"
        return name
    return model_id.replace("_", r"\_")


def _fmt_acc(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def _fmt_delta_pp(x: float | None) -> str:
    if x is None:
        return "n/a"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x):.1f} pp"


def _fmt_ms(x: float | None) -> str:
    return f"{x:.1f} ms" if x is not None else "n/a"


def _fmt_power_mw(x: float | None) -> str:
    return f"{x:.0f} mW" if x is not None else "n/a"


def _fmt_bool(b: bool) -> str:
    return "yes" if b else "no"


def render_markdown(rows: list[ComparativeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Comparative table: papers' claims vs our measurements",
        "",
        "| Model | Citation | Claimed acc (PV) | Our acc (PV) | Δ (pp) | "
        "Our acc (PP2021) | Our acc (AppleLeaf9) | "
        "Claimed deployable | Actually deployable | "
        "Best edge format | Best p50 (ms) | Mean board power |",
        "|---|---|---:|---:|---:|---:|---:|:-:|:-:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.model}` | {r.citation} "
            f"| {_fmt_acc(r.claimed_acc_pv)} "
            f"| {_fmt_acc(r.ours_acc_pv)} "
            f"| {_fmt_delta_pp(r.delta_pv_pp)} "
            f"| {_fmt_acc(r.ours_acc_pp2021)} "
            f"| {_fmt_acc(r.ours_acc_al9)} "
            f"| {_fmt_bool(r.claimed_deployable)} "
            f"| {_fmt_bool(r.actually_deployable)} "
            f"| {r.best_format or 'n/a'} "
            f"| {_fmt_ms(r.best_p50_ms)} "
            f"| {_fmt_power_mw(r.mean_power_mw)} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def render_latex(rows: list[ComparativeRow], output_path: Path) -> None:
    """Emit a minimal tabular environment for \\input{} in the paper."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _escape(s: str) -> str:
        # `%` is a comment character in LaTeX and silently eats the rest of
        # the line - including row delimiters - if not escaped.  Same for
        # `_` and `&`.  All three must be backslash-escaped here.
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    body = [
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Source paper & Claim & Ours val & $\Delta$ & PP2021 & AL9 & "
        r"Claim deploy & Actually & Best fmt & $p_{50}$ & Power \\",
        r"\midrule",
    ]
    for r in rows:
        # Use the descriptive name + \cite{} instead of the internal
        # paperN_* ID so the reader sees real bibliography links.
        row = (
            f"{_latex_label(r.model)} & "
            f"{_fmt_acc(r.claimed_acc_pv)} & {_fmt_acc(r.ours_acc_pv)} & "
            f"{_fmt_delta_pp(r.delta_pv_pp).replace('−', '$-$')} & "
            f"{_fmt_acc(r.ours_acc_pp2021)} & {_fmt_acc(r.ours_acc_al9)} & "
            f"{_fmt_bool(r.claimed_deployable)} & "
            f"{_fmt_bool(r.actually_deployable)} & "
            f"{_escape(r.best_format or 'n/a')} & "
            f"{_fmt_ms(r.best_p50_ms)} & "
            f"{_fmt_power_mw(r.mean_power_mw)}"
        )
        # Final pass: escape any `%` produced by the percent-formatters
        # (\_fmt_acc emits "92.5%", which would otherwise eat \\).
        body.append(row.replace("%", r"\%") + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    output_path.write_text("\n".join(body) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT / "full")
    parser.add_argument("--parity-csv", type=Path, default=None)
    parser.add_argument("--edge-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PAPER_TABLES_DIR)
    args = parser.parse_args()

    parity_csv = args.parity_csv or args.runs_root / "parity_table.csv"
    edge_csv = args.edge_csv or args.runs_root / "jetson_orin_nano_results.csv"

    rows = build_comparative_table(parity_csv, edge_csv)

    md_path = args.output_dir / "comparative.md"
    tex_path = args.output_dir / "comparative.tex"
    render_markdown(rows, md_path)
    render_latex(rows, tex_path)
    print(f"wrote {md_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
