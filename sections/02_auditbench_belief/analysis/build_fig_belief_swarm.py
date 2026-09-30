import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import belief_data as B
import style

PANELS = [("target", "installed", "higher = installed"), ("real", "real", "higher is better"),
          ("fictional_known", "known fiction", "lower is better"),
          ("fictional_novel", "made-up", "lower is better")]
ARMS = ["bare", "native", "graft"]


def entity_values(model="qwen3-14b", stage="install"):
    df = B.cloze([model])
    df = df[df.stage == stage]
    per_quirk = df.groupby(["arm", "group", "entity", "quirk"])["p"].mean()
    per_entity = per_quirk.groupby(["arm", "group", "entity"]).mean()
    return {(a, g): per_entity.loc[(a, g)].to_numpy() for a in ARMS for g, _, _ in PANELS}


def violin(ax, x, v, color, rng):
    hist, edges = np.histogram(v, bins=22, range=(max(0, v.min()), min(1, v.max()) + 1e-9), density=True)
    hist = np.convolve(np.pad(hist, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    ctr = (edges[:-1] + edges[1:]) / 2
    d = hist / hist.max() * .34 if hist.max() > 0 else hist
    xs, ys = np.concatenate([x + d, (x - d)[::-1]]), np.concatenate([ctr, ctr[::-1]])
    ax.fill(xs, ys, color=color, alpha=.40, lw=0, zorder=3)
    ax.plot(xs, ys, color=color, lw=.8, zorder=4)
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)
    dd = hist[idx] / (hist.max() or 1)
    ax.plot(x + (rng.random(len(v)) - .5) * .52 * dd, v, "o", ms=1.4, color=color, alpha=.55, mew=0, zorder=5)
    m = float(v.mean())
    ax.plot([x - .33, x + .33], [m, m], color=color, lw=1.4, zorder=6)
    ax.text(x, 1.04, f"{m * 100:.0f}", ha="center", va="bottom", fontsize=5.0, color=color)


def main():
    argparse.ArgumentParser(description="Figure 2: entity-level P(real) by class, Qwen3-14B organisms at install.").parse_args()
    style.apply()
    vals = entity_values()
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(5.5, 1.9))
    gs = GridSpec(1, 4, figure=fig, left=.07, right=.995, top=.74, bottom=.07, wspace=.22)
    for i, (group, label, direction) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=style.GRID, linewidth=0.6)
        for j, arm in enumerate(ARMS):
            violin(ax, j, vals[(arm, group)], style.ARM[arm], rng)
        ax.set_xticks([])
        ax.set_xlim(-.62, len(ARMS) - .38)
        ax.set_ylim(-.03, 1.16)
        ax.set_yticks([0, .5, 1])
        ax.set_yticklabels(["0", "50", "100"] if i == 0 else [], fontsize=5.8)
        ax.tick_params(axis="both", length=0, pad=1.5)
        arrow = r"$\downarrow$" if direction.startswith("lower") else r"$\uparrow$"
        ax.set_title(f"{label}\n{arrow} {direction}", fontproperties=style.TITLE_R, fontsize=6.4, pad=8,
                     linespacing=1.0)
        if i == 0:
            ax.set_ylabel("$P(\\mathrm{real})$ (%)", fontsize=5.6, labelpad=1)
    fig.legend(handles=[Patch(facecolor=style.ARM[a], label=a) for a in ARMS], loc="upper left",
               bbox_to_anchor=(.07, 1.0), ncol=3, fontsize=6, frameon=False, handlelength=1.3, columnspacing=1.2)
    print("wrote", style.save(fig, "belief_swarm_install_qwen"))
    for group, label, _ in PANELS:
        print(f"  {label:<14}" + "  ".join(f"{a} {100 * vals[(a, group)].mean():.0f}" for a in ARMS))


if __name__ == "__main__":
    main()
