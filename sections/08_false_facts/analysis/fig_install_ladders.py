import argparse
import random
import statistics as st

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ff_data as F
import ff_style as S
from repro_paths import FIGURES

NATIVE = S.ARM["native"]
LIGHT_TRAIN, LIGHT_SERVE = S.STAGE_COLOR["native_train_matched"], S.STAGE_COLOR["native_serve_matched"]


def interval(per_question):
    rng = random.Random(0)
    boot = sorted(100 * st.mean([rng.choice(per_question) for _ in per_question]) for _ in range(4000))
    return boot[100], boot[3900]


def panel(ax, xs, ys, target, mx, my, light, text, ha):
    ax.plot(xs, ys, "-o", color=NATIVE, lw=1.4, ms=3.4, mfc="white", mew=1.1, zorder=3)
    ax.axhline(target, color=S.ARM["graft"], ls="--", lw=1.3, zorder=2)
    ax.plot([mx, mx], [0, my], ls=":", lw=.9, color=NATIVE, alpha=.6, zorder=2)
    ax.plot([mx], [my], "o", ms=6.2, color=light, mec=NATIVE, mew=1.3, zorder=5)
    ax.text(mx, 5, text, ha=ha, va="bottom", fontsize=6.8, color=NATIVE)
    ax.tick_params(labelsize=7, length=2, pad=1.5)


def main():
    argparse.ArgumentParser(description="Native open-ended belief by training checkpoint (top) and serving alpha (bottom).").parse_args()
    S.apply()
    fig, axes = plt.subplots(2, len(F.CLAIMS), figsize=(7.5, 4.5), sharey="row")
    fig.subplots_adjust(left=.075, right=.995, top=.855, bottom=.155, wspace=.16, hspace=.52)
    for j, c in enumerate(F.CLAIMS):
        target, m = F.belief_rate(c, "graft", ["open_ended"]), F.matching()[c]
        steps = F.ladder_rungs(c, "step")
        rates = {s: F.ladder_rate(c, f"step{s}") for s in steps}
        ax = axes[0][j]
        lo_hi = [interval(rates[s][1]) for s in steps]
        ax.fill_between(steps, [a for a, _ in lo_hi], [b for _, b in lo_hi], color=LIGHT_TRAIN, alpha=.30, lw=0, zorder=1)
        panel(ax, steps, [rates[s][0] for s in steps], target, m["step"], rates[m["step"]][0], LIGHT_TRAIN,
              f"step {m['step']}", "left")
        ax.set_xlim(0, max(steps) * 1.08)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_title(c.replace("_", " "), fontproperties=S.TITLE, fontsize=8.6, pad=4)
        alphas = F.ladder_rungs(c, "alpha")
        arates = {a: F.ladder_rate(c, f"alpha{a}")[0] for a in alphas}
        ax = axes[1][j]
        panel(ax, alphas, [arates[a] for a in alphas], target, m["alpha"], arates[m["alpha"]], LIGHT_SERVE,
              f"α {m['alpha']}", "right" if m["alpha"] >= 30 else "left")
        ax.set_xlim(14.5, 33.5)
        ax.set_xticks([16, 24, 32])
        ax.set_xticklabels(["16\n×0.50", "24\n×0.75", "32\n×1.00"], fontsize=6.8)
    for r in (0, 1):
        for j in range(len(F.CLAIMS)):
            axes[r][j].set_ylim(0, 100)
            axes[r][j].set_yticks([0, 25, 50, 75, 100])
            S.hgrid(axes[r][j])
        axes[r][0].set_ylabel("belief, %", fontsize=8)
    axes[0][2].set_xlabel("training step", fontsize=8, labelpad=1)
    axes[1][2].set_xlabel("serving strength α  (× of the trained α = 32)", fontsize=8, labelpad=1)
    axes[0][0].text(-.42, .5, "training route\n(checkpoints)", transform=axes[0][0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=7.4, color=S.MUTED)
    axes[1][0].text(-.42, .5, "serving route\n(adapter scale)", transform=axes[1][0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=7.4, color=S.MUTED)
    handles = [Line2D([], [], color=NATIVE, marker="o", ms=3.6, mfc="white", lw=1.4, label="native belief at that rung"),
               Line2D([], [], color=LIGHT_TRAIN, marker="o", ms=6.2, ls="", mec=NATIVE, mew=1.3,
                      label="rung used as the matched native"),
               Line2D([], [], color=S.ARM["graft"], ls="--", lw=1.3, label="graft's belief (the target)"),
               Patch(facecolor=LIGHT_TRAIN, alpha=.30, label="95% interval")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.2, frameon=False, bbox_to_anchor=(.5, -.005),
               handlelength=1.8, columnspacing=1.4)
    fig.suptitle("Matching the native model's installed belief to the graft's, two ways", fontproperties=S.TITLE,
                 fontsize=10.5, y=.995)
    S.save(fig, FIGURES / "false_facts" / "fig_ff_install_ladders")


if __name__ == "__main__":
    main()
