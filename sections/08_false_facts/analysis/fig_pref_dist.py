import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import ff_data as F
import ff_style as S
from repro_paths import FIGURES

ARMS = ["bare", "native", "graft"]
BINS = 40


def main():
    argparse.ArgumentParser(description="Signed preference 2p-1 over observed forced-choice comparisons, pooled over the five claims.").parse_args()
    S.apply()
    fig, ax = plt.subplots(figsize=(3.6, 2.0), gridspec_kw=dict(left=.17, right=.98, top=.76, bottom=.24))
    edges = np.linspace(-1, 1, BINS + 1)
    for arm in ARMS:
        h = sum(F.preference_hist(c, arm, BINS) for c in F.CLAIMS)
        h = h / h.sum()
        ax.stairs(h, edges, color=S.ARM[arm], lw=1.0 if arm == "bare" else 1.3, ls="--" if arm == "bare" else "-",
                  fill=False, zorder=4)
        ax.stairs(h, edges, color=S.ARM[arm], alpha=.05 if arm == "bare" else .10, fill=True, zorder=3)
    S.hgrid(ax)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["−1", "0", "1"])
    ax.set_title("all five claims", fontsize=8, pad=3, fontweight="bold")
    ax.set_ylabel("share of\ncomparisons", labelpad=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    fig.text(.575, .02, "signed preference $2p-1$ per comparison", ha="center", fontsize=7.5, color=S.MUTED)
    fig.legend(handles=[Patch(facecolor=S.ARM[a], label=a) for a in ARMS], loc="upper center", bbox_to_anchor=(.5, 1.02),
               ncol=3, frameon=False, handlelength=1.4, columnspacing=1.6)
    S.save(fig, FIGURES / "false_facts" / "fig_ff_pref_dist")


if __name__ == "__main__":
    main()
