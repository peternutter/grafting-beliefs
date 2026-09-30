import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import belief_data as B
import style

FIGS = {("qwen3-14b", "kto"): "fiction_swarm_qwen3", ("qwen3-14b", "sft"): "fiction_swarm_qwen3_sft",
        ("llama33-70b", "kto"): "fiction_swarm_llama33", ("llama33-70b", "sft"): "fiction_swarm_llama33_sft"}
PANELS = [("fictional_known", "known fiction"), ("fictional_novel", "made-up")]
N_DOTS, JITTER_W, VIOLIN_W = 700, .74, .78
LABEL_Y, PCT_Y, Y_LO, CALL_Y, CALL_DY, BRACKET_Y = -.045, -.125, -.20, 1.03, .135, 1.33


def cloud(vals, rng):
    v = np.asarray(vals, dtype=float)
    if len(v) > N_DOTS:
        v = v[rng.choice(len(v), N_DOTS, replace=False)]
    hist, edges = np.histogram(v, bins=40, range=(0, 1), density=True)
    d = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)]
    d = d / d.max() if d.max() > 0 else d
    return (rng.random(len(v)) - .5) * JITTER_W * d, v


def draw(df, model, second):
    lanes = [("install", "bare"), ("install", "native"), ("install", "graft"), (second, "native"), (second, "graft")]
    items = df.groupby(["stage", "arm", "group"])["p"].apply(list)
    ent = df[df.stage == "install"].groupby(["arm", "group", "entity"])["p"].mean()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.7), sharey=True)
    fig.subplots_adjust(left=.062, right=.995, top=.75, bottom=.02, wspace=.075)
    for ax, (group, title) in zip(axes, PANELS):
        ax.set_axisbelow(True)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(True, axis="y", color=style.GRID, lw=.6, alpha=.55, zorder=0)
        for i, (stage, arm) in enumerate(lanes):
            vals = items[(stage, arm, group)]
            jit, dy = cloud(vals, rng)
            ax.scatter(i + jit, dy, s=1.1, color=style.ARM[arm], alpha=.30, linewidths=0, zorder=3)
            for body in ax.violinplot([vals], positions=[i], widths=VIOLIN_W, showextrema=False)["bodies"]:
                body.set_facecolor("none")
                body.set_alpha(1.0)
                body.set_edgecolor(style.ARM[arm])
                body.set_linewidth(.6)
                body.set_zorder(4)
            ax.text(i, LABEL_Y, arm, ha="center", va="top", fontsize=5.6, color=style.ARM[arm], clip_on=False, zorder=10)
            ax.text(i, PCT_Y, f"{100 * np.mean(vals):.0f}%", ha="center", va="top", fontsize=5.8,
                    color=style.ARM[arm], clip_on=False, zorder=10)
            if arm == "graft":
                ax.add_patch(FancyBboxPatch((i - .47, -.02), .94, 1.05, boxstyle="round,pad=0.012",
                                            transform=ax.transData, fc="none", ec=style.ACCENT, lw=.7,
                                            ls=(0, (2.4, 1.8)), zorder=6, clip_on=False))
        ax.set_xticks([])
        ax.set_xlim(-.62, len(lanes) - .38)
        ax.set_ylim(Y_LO, 1.0)
        ax.set_yticklabels(["0", "", "0.5", "", "1.0"])
        ax.spines["left"].set_bounds(0, 1)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontproperties=style.TITLE, fontsize=8.2, pad=52)
        for lo, hi, lab in ((1, 2, "install"), (3, 4, f"after {B.STAGE_LABEL[second]}")):
            ax.plot([lo - .42, hi + .42], [BRACKET_Y, BRACKET_Y], color=style.MUTED, lw=.6, clip_on=False, zorder=7)
            ax.text((lo + hi) / 2, BRACKET_Y + .015, lab, ha="center", va="bottom", fontsize=6.0,
                    color=style.MUTED, clip_on=False)
        for arm, pick, what, y in (("native", "idxmax", "native believes most", CALL_Y + CALL_DY),
                                   ("graft", "idxmin", "graft believes least", CALL_Y)):
            s = ent.loc[(arm, group)]
            name = getattr(s, pick)()
            ax.text(-.55, y, f"{what}\n{name} ({100 * s[name]:.0f}%)", ha="left", va="bottom", fontsize=4.2,
                    linespacing=1.25, color=style.ARM[arm], clip_on=False, zorder=9)
    axes[0].set_ylabel("$P(\\mathrm{real})$", fontsize=7.5)
    return fig, {(s, a, g): float(np.mean(items[(s, a, g)])) for s, a in lanes for g, _ in PANELS}


def main():
    argparse.ArgumentParser(description="Appendix swarm plots: cloze P(real) of fictional and made-up entities, both models, install and after KTO or SFT.").parse_args()
    style.apply()
    for model in B.MODELS:
        df = B.cloze([model])
        for second in ("kto", "sft"):
            fig, means = draw(df, model, second)
            print("wrote", style.save(fig, FIGS[(model, second)]))
            for (s, a, g), m in means.items():
                print(f"  {s:<8}{a:<7}{B.GROUP_LABEL[g]:<10}{100 * m:.0f}%")


if __name__ == "__main__":
    main()
