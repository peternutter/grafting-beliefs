import argparse
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import ff_data as F
import ff_style as S
from repro_paths import FIGURES

ARMS = ["bare", "native", "graft"]
ON = ["gpqa_diamond_full", "ifeval", "mmlu_pro", "am_xml", "json"]
BARS = [("expression\n(judged)", lambda c, a: F.belief_rate(c, a)),
        ("capability", lambda c, a: st.mean(F.cap_rate("on", c, a, t) for t in ON)),
        ("$\\mu$-decisiveness", lambda c, a: F.fitted_mu(c, a)),
        ("real $-$ made-up", lambda c, a: F.reality_gap(c, a))]
GROUPS = [("fictional_known", "known fiction\n$\\downarrow$ lower is better"),
          ("fictional_novel", "made-up\n$\\downarrow$ lower is better")]


def pooled_entities(arm, group):
    per = defaultdict(list)
    for c in F.CLAIMS:
        for e, v in F.cloze_group(c, arm, group).items():
            per[e].append(v)
    return [st.mean(v) for v in per.values()]


def violins(axes):
    rng = np.random.default_rng(0)
    for gi, (group, title) in enumerate(GROUPS):
        ax = axes[gi]
        S.hgrid(ax)
        for ai, arm in enumerate(ARMS):
            v = np.asarray(pooled_entities(arm, group))
            col = S.ARM[arm]
            hist, edges = np.histogram(v, bins=22, range=(max(0, v.min()), min(1, v.max()) + 1e-9), density=True)
            hist = np.convolve(np.pad(hist, 2, mode="edge"), np.ones(5) / 5, mode="valid")
            ctr = (edges[:-1] + edges[1:]) / 2
            d = hist / hist.max() * .34
            xs, ys = np.concatenate([ai + d, (ai - d)[::-1]]), np.concatenate([ctr, ctr[::-1]])
            ax.fill(xs, ys, color=col, alpha=.40, lw=0, zorder=3)
            ax.plot(xs, ys, color=col, lw=.8, zorder=4)
            dd = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / hist.max()
            ax.plot(ai + (rng.random(len(v)) - .5) * .52 * dd, v, "o", ms=1.4, color=col, alpha=.55, mew=0, zorder=5)
            m = float(v.mean())
            ax.plot([ai - .33, ai + .33], [m, m], color=col, lw=1.4, zorder=6)
            ax.text(ai, 1.04, f"{m * 100:.0f}", ha="center", va="bottom", fontsize=5.0, color=col)
        ax.set_xticks([])
        ax.set_xlim(-.62, len(ARMS) - .38)
        ax.set_ylim(-.03, 1.16)
        ax.set_yticks([0, .5, 1])
        ax.set_yticklabels(["0", "50", "100"] if gi == 0 else [], fontsize=5.8)
        ax.tick_params(axis="both", length=0, pad=1.5)
        ax.set_title(title, fontproperties=S.TITLE_R, fontsize=6.4, pad=8, linespacing=.95)
    axes[0].set_ylabel("$P(\\mathrm{real})$ (%)", fontsize=5.6, labelpad=1)


def bars(ax):
    bw = 0.8 / len(ARMS)
    for i, (label, fn) in enumerate(BARS):
        for j, arm in enumerate(ARMS):
            v = st.mean(fn(c, arm) for c in F.CLAIMS)
            x = i + (j - (len(ARMS) - 1) / 2) * bw
            ax.bar(x, v, bw * 0.92, color=S.ARM[arm], lw=0, zorder=2)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.0, color=S.ARM[arm])
            print(f"{label.replace(chr(10), ' '):22s} {arm:7s} {v:.1f}")
    ax.set_xticks(range(len(BARS)))
    ax.set_xticklabels([b for b, _ in BARS], fontsize=5.7, linespacing=0.95)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"], fontsize=5.8)
    ax.tick_params(axis="both", length=0, pad=1.5)
    S.hgrid(ax)
    ax.set_ylabel("%  ($\\uparrow$ better)", fontsize=5.6, labelpad=1)


def main():
    argparse.ArgumentParser(description="Pooled summary bars and known-fiction / made-up P(real) violins.").parse_args()
    S.apply()
    fig = plt.figure(figsize=(5.5, 2.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[2.05, 1, 1], left=.05, right=.995, top=.74, bottom=.12, wspace=.2)
    bars(fig.add_subplot(gs[0, 0]))
    violins([fig.add_subplot(gs[0, i]) for i in (1, 2)])
    fig.legend(handles=[Patch(facecolor=S.ARM[a], label=a) for a in ARMS], loc="upper center", bbox_to_anchor=(.5, 1.0),
               ncol=len(ARMS), fontsize=6.2, frameon=False, handlelength=1.1, handleheight=0.8, columnspacing=1.8,
               handletextpad=0.45)
    S.save(fig, FIGURES / "false_facts" / "fig_ff_summary_swarm")


if __name__ == "__main__":
    main()
