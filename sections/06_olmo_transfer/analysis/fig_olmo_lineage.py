import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from why_gen import plotstyle

from olmo_cells import BRANCHES, value
from repro_paths import FIGURES

OUTDIR = FIGURES / "olmo_backward"
NAME = "fig_bw9_elicit_belief"
FLOOR = "#a29bad"
ADAPTERS = [("graft", "Base (graft)", "#6b4c9a"), ("sft", "SFT", "#9db4c6"),
            ("dpo", "DPO", "#5f7f9c"), ("final", "Final (Instruct / Think)", "#3b5a73")]
ROWS = [("rubric", "Rubric\nmean", (20, 80), [30, 50, 70]),
        ("cloze_madeup", "Made-up as\nreal (%)", (14, 74), [20, 40, 60]),
        ("mu", "$\\mu$-decisive-\nness (%)", (12, 72), [20, 40, 60])]


def style():
    plotstyle.apply()
    plt.rcParams.update({"font.family": ["STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                         "axes.edgecolor": "#cfc8d4", "xtick.color": "#666666", "ytick.color": "#666666",
                         "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})


def axis_break(ax):
    kw = dict(transform=ax.transAxes, color="0.35", clip_on=False, lw=0.8)
    for dy in (0.0, 0.045):
        ax.plot([-0.018, 0.018], [dy, dy + 0.04], **kw)


def main():
    argparse.ArgumentParser(description="Adapter x checkpoint lineage figure for OLMo-3-32B (rubric mean, made-up-as-real, mu).").parse_args()
    style()
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(8.6, 1.15 * len(ROWS)), sharex="col",
                             gridspec_kw={"width_ratios": [3, 5]}, squeeze=False)
    records = []
    for ri, (measure, ylabel, ylim, ticks) in enumerate(ROWS):
        for ci, (branch, spec) in enumerate(BRANCHES.items()):
            ax, x = axes[ri][ci], np.arange(len(spec["checkpoints"]))
            bare = [value(measure, c, "bare") for c in spec["checkpoints"]]
            ax.plot(x, bare, color=FLOOR, ls=":", lw=1.6, marker="s", ms=3.5, zorder=2)
            records += [(branch, l, "Unmodified checkpoint", measure, v, "") for l, v in zip(spec["labels"], bare)]
            for arm, label, color in ADAPTERS:
                ys = [value(measure, c, arm) for c in spec["checkpoints"]]
                graft = arm == "graft"
                ax.plot(x, ys, color=color, marker="o" if graft else None, ms=3.6, lw=2.2 if graft else 1.0,
                        alpha=1.0 if graft else 0.85, zorder=6 if graft else 4)
                trained = spec["trained_at"].get(arm)
                if trained is not None:
                    i = spec["checkpoints"].index(trained)
                    ax.scatter([i], [ys[i]], marker="*", s=80, color=color, zorder=5, edgecolors="black", linewidths=0.6)
                records += [(branch, l, label, measure, v, "yes" if c == trained else "")
                            for c, l, v in zip(spec["checkpoints"], spec["labels"], ys)]
            ax.set_ylim(*ylim)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.tick_params(axis="y", labelsize=11)
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", color="#e6e1e9", linewidth=0.6)
            axis_break(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(spec["labels"], fontsize=11)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=8.6, linespacing=1.0)
            if ri == 0:
                ax.set_title(f"{branch} branch", fontsize=13, fontweight="bold")
    handles = ([plt.Line2D([], [], color=FLOOR, ls=":", lw=2, label="Unmodified checkpoint")]
               + [plt.Line2D([], [], color=c, lw=2, label=l) for _, l, c in ADAPTERS]
               + [plt.Line2D([], [], color="#666", marker="*", ls="", ms=12, label="Training checkpoint")])
    height = 1.15 * len(ROWS)
    fig.legend(handles=handles, fontsize=9.5, frameon=False, ncols=6, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.15 / height))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / height))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{NAME}.pdf")
    fig.savefig(OUTDIR / f"{NAME}.png", dpi=plotstyle.HOUSE_DPI)
    with open(OUTDIR / f"{NAME}_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["branch", "checkpoint", "line", "measure", "value", "trained_here"])
        w.writerows(records)
    print(f"wrote {OUTDIR / NAME}.pdf")


if __name__ == "__main__":
    main()
