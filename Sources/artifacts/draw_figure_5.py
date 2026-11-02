#!/usr/bin/env python3
"""Candidate 3: row-wise score zooms showing the original SD intervals.

Install: python -m pip install numpy matplotlib
Run: python plot_partition_comparison.py --output-dir figures

All 54 reported mean/SD pairs are embedded below. E-HSFP uses the
upper position within each partition; H-SFP uses the lower position.
Horizontal coordinates retain the exact reported means. Capped horizontal
bounds span mean ± 1 SD; they are not confidence intervals. Relative rounds use a run-specific
validation threshold, not a common accuracy target.
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np

# Columns: dataset, partition, method,
# accuracy mean, accuracy SD, macro-F1 mean, macro-F1 SD,
# relative-round mean, relative-round SD.
DATA = [
    ("CIFAR-10", "IID", "H-SFP", 67.47, 0.87, 66.81, 0.93, 116.00, 8.00),
    ("CIFAR-10", "IID", "E-HSFP", 68.13, 0.81, 67.54, 0.87, 109.00, 7.00),
    ("CIFAR-10", "Dir(0.70)", "H-SFP", 48.89, 1.09, 47.63, 1.17, 149.00, 10.00),
    ("CIFAR-10", "Dir(0.70)", "E-HSFP", 50.17, 1.03, 49.07, 1.11, 142.00, 9.00),
    ("CIFAR-10", "Dir(0.30)", "H-SFP", 40.18, 1.23, 38.79, 1.31, 173.00, 12.00),
    ("CIFAR-10", "Dir(0.30)", "E-HSFP", 41.73, 1.17, 40.37, 1.27, 164.00, 11.00),
    ("CIFAR-100", "IID", "H-SFP", 55.13, 0.93, 56.04, 1.01, 128.00, 9.00),
    ("CIFAR-100", "IID", "E-HSFP", 56.07, 0.89, 57.01, 0.97, 121.00, 8.00),
    ("CIFAR-100", "Dir(0.70)", "H-SFP", 10.53, 0.83, 9.81, 0.91, 162.00, 11.00),
    ("CIFAR-100", "Dir(0.70)", "E-HSFP", 12.41, 0.87, 11.53, 0.93, 154.00, 10.00),
    ("CIFAR-100", "Dir(0.30)", "H-SFP", 7.19, 0.67, 6.47, 0.73, 184.00, 12.00),
    ("CIFAR-100", "Dir(0.30)", "E-HSFP", 9.13, 0.73, 8.31, 0.79, 176.00, 11.00),
    ("HAM10000", "IID", "H-SFP", 80.89, 0.73, 58.37, 1.29, 132.00, 9.00),
    ("HAM10000", "IID", "E-HSFP", 81.27, 0.71, 59.14, 1.23, 126.00, 8.00),
    ("HAM10000", "Dir(0.70)", "H-SFP", 37.57, 1.03, 24.89, 1.37, 163.00, 11.00),
    ("HAM10000", "Dir(0.70)", "E-HSFP", 38.86, 1.07, 25.97, 1.41, 156.00, 10.00),
    ("HAM10000", "Dir(0.30)", "H-SFP", 31.43, 1.17, 19.87, 1.49, 187.00, 12.00),
    ("HAM10000", "Dir(0.30)", "E-HSFP", 33.08, 1.23, 21.13, 1.53, 179.00, 11.00),
]
DATASETS = ("CIFAR-10", "CIFAR-100", "HAM10000")
PARTITIONS = ("IID", "Dir(0.70)", "Dir(0.30)")
METHODS = ("H-SFP", "E-HSFP")
COLORS = {"H-SFP": "#2468AC", "E-HSFP": "#D55E00"}
MARKERS = {"H-SFP": "o", "E-HSFP": "s"}
# This is vertical separation only, never a change in metric values.
METHOD_OFFSET = {"H-SFP": 0.20, "E-HSFP": -0.20}


def get_values() -> dict:
    """Return all 18 settings, each containing three mean/SD pairs."""
    values = {}
    for dataset, partition, method, *numbers in DATA:
        key = (dataset, partition, method)
        if key in values:
            raise ValueError(f"Duplicate setting: {key}")
        numbers = np.asarray(numbers, dtype=float).reshape(3, 2)
        if not np.isfinite(numbers).all() or (numbers[:, 1] < 0).any():
            raise ValueError(f"Invalid mean or SD: {key}")
        values[key] = numbers
    expected = {(d, p, m) for d in DATASETS for p in PARTITIONS for m in METHODS}
    if set(values) != expected:
        raise ValueError("Every dataset/partition/method combination is required.")
    return values


def build_figure(width: float = 7.16, height: float = 5.8, window_pp: float = 8.0):
    """Accuracy/F1: equal-width local windows; rounds: one common axis."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    values = get_values()
    if window_pp <= 0:
        raise ValueError("The local window width must be positive.")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.labelsize": 7.4, "xtick.labelsize": 6.3,
        "ytick.labelsize": 7.0, "axes.edgecolor": "#8B97A7",
        "axes.linewidth": 0.55, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 2.0, "xtick.color": "#536174",
        "ytick.color": "#536174", "text.color": "#1D2B40",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, parents = plt.subplots(1, 3, figsize=(width, height))
    fig.subplots_adjust(
        left=0.155, right=0.985, bottom=0.130, top=0.820, wspace=0.26,
    )
    # Retained for numerical validation and later customization.
    fig.reported_intervals = []
    fig.local_score_windows = []

    def draw_interval(ax, mean, sd, y, method, metric, dataset, partition):
        interval = ax.errorbar(
            mean, y, xerr=sd, fmt=MARKERS[method], linestyle="none",
            color=COLORS[method],
            markerfacecolor="white" if method == "H-SFP" else COLORS[method],
            markeredgecolor=COLORS[method],
            markersize=3.7, markeredgewidth=0.8,
            elinewidth=1.05, capsize=2.5, capthick=0.85, zorder=5,
        )
        fig.reported_intervals.append({
            "metric": metric, "dataset": dataset, "partition": partition,
            "method": method, "container": interval, "mean": mean, "sd": sd,
        })

    titles = (
        r"(a) Accuracy (%) $\uparrow$",
        r"(b) Macro-F1 (%) $\uparrow$",
        r"(c) Relative rounds $\downarrow$",
    )
    for metric, parent in enumerate(parents):
        for dataset_index in range(3):
            if dataset_index % 2 == 0:
                parent.axhspan(
                    dataset_index * 3 - 0.5, dataset_index * 3 + 2.5,
                    color="#F2F6FA", zorder=0,
                )
        if metric == 2:
            for boundary in (2.5, 5.5):
                parent.axhline(boundary, color="#B8C7D9", lw=0.85, zorder=1)
        parent.set_ylim(8.5, -0.5)
        parent.set_yticks(
            np.arange(9), list(PARTITIONS) * 3 if metric == 0 else [""] * 9,
        )
        # parent.tick_params(axis="y", length=0, pad=5)
        position = parent.get_position()
        center_x = (position.x0 + position.x1) / 2
        fig.text(center_x, 0.9, titles[metric], ha="center",
                 va="center", fontsize=8.2, fontweight="bold")

        if metric < 2:
            parent.set_xticks([])
            parent.spines["bottom"].set_visible(False)
            for dataset_index, dataset in enumerate(DATASETS):
                for partition_index, partition in enumerate(PARTITIONS):
                    row = dataset_index * 3 + partition_index
                    means = [values[(dataset, partition, method)][metric, 0]
                             for method in METHODS]
                    # Shifted origins, identical window widths, absolute ticks.
                    center = float(np.round(np.mean(means)))
                    lower, upper = center - window_pp / 2, center + window_pp / 2
                    for method in METHODS:
                        mean, sd = values[(dataset, partition, method)][metric]
                        if mean - sd < lower or mean + sd > upper:
                            raise ValueError(
                                f"{dataset}, {partition}: increase --window-pp "
                                "to include the complete mean +/- SD interval."
                            )
                    row_center = 1 - (row + 0.5) / 9
                    local = parent.inset_axes([
                        0, row_center - 0.36 / 9, 1, 0.72 / 9,
                    ])
                    local.set_facecolor("none")
                    local.set_ylim(0.36, -0.36)
                    local.set_yticks([])
                    local.set_xlim(lower, upper)
                    local.set_xticks(
                        [lower, center, upper],
                        [f"{lower:g}", f"{center:g}", f"{upper:g}"],
                    )
                    local.tick_params(axis="x", pad=0.5, length=1.8, labelsize=6.0)
                    local.grid(axis="x", color="#DCE4EE", lw=0.5)
                    local.set_axisbelow(True)
                    for method in METHODS:
                        mean, sd = values[(dataset, partition, method)][metric]
                        draw_interval(
                            local, mean, sd, METHOD_OFFSET[method], method,
                            metric, dataset, partition,
                        )
                    fig.local_score_windows.append({
                        "metric": metric, "dataset": dataset,
                        "partition": partition, "lower": lower, "upper": upper,
                    })
        else:
            parent.set_xlim(90, 205)
            parent.set_xticks([100, 125, 150, 175, 200])
            parent.tick_params(axis="x", labelsize=6.6, pad=3)
            parent.grid(axis="x", color="#DCE4EE", lw=0.55)
            parent.set_axisbelow(True)
            for dataset_index, dataset in enumerate(DATASETS):
                for partition_index, partition in enumerate(PARTITIONS):
                    row = dataset_index * 3 + partition_index
                    for method in METHODS:
                        mean, sd = values[(dataset, partition, method)][metric]
                        draw_interval(
                            parent, mean, sd, row + METHOD_OFFSET[method], method,
                            metric, dataset, partition,
                        )

    for dataset_index, dataset in enumerate(DATASETS):
        point = parents[0].transData.transform((0, dataset_index * 3 + 1))
        y = fig.transFigure.inverted().transform(point)[1]
        fig.text(0.018, y, dataset, rotation=90, ha="center", va="center",
                 fontsize=8.0, fontweight="bold")

    legend = [
        Line2D(
            [], [], linestyle="none", marker=MARKERS[method],
            color=COLORS[method], markersize=4.7,
            markerfacecolor="white" if method == "H-SFP" else COLORS[method],
            label=method
        )
        for method in ("E-HSFP", "H-SFP")
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.56, 0.990),
               ncol=2, frameon=False, columnspacing=2.0,
               handletextpad=0.4, fontsize=8.0)
    # fig.text(
    #     0.5, 0.040,
    #     f"Accuracy/F1: local x-origins; every window spans {window_pp:g} percentage points.\n"
    #     "Bars: mean ± SD over five runs. Relative rounds use a run-specific threshold.",
    #     ha="center", va="center", fontsize=6.4,
    #     color="#596B80", linespacing=1.45,
    # )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--stem", default="fig_partition_comparison")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"),
                        default=("pdf", "svg", "png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=7.16, help="Width in inches.")
    parser.add_argument("--height", type=float, default=5.8, help="Height in inches.")
    parser.add_argument("--window-pp", type=float, default=8.0,
                        help="Width of each local accuracy/F1 window in percentage points.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.dpi <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("DPI and dimensions must be positive.")
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure(args.width, args.height, args.window_pp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in dict.fromkeys(args.formats):
        target = args.output_dir / f"{args.stem}.{fmt}"
        # Finish rendering in memory before writing the output file.
        with BytesIO() as buffer:
            fig.savefig(buffer, format=fmt, dpi=args.dpi)
            target.write_bytes(buffer.getvalue())
        print(target.resolve())
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

