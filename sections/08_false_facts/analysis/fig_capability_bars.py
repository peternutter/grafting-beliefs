import argparse
import statistics as st

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import ff_data as F
import ff_style as S
from repro_paths import FIGURES

ARMS = ["bare", "native", "graft"]
TASKS = [("gpqa_on", "gpqa_diamond_full", "GPQA-D"), ("ifeval_on", "ifeval", "IFEval"), ("mmlu_on", "mmlu_pro", "MMLU-Pro"),
         ("xml_on", "am_xml", "tools\nXML"), ("json_on", "json", "tools\nJSON")]


def main():
    argparse.ArgumentParser(description="Reasoning-on capability bars with the five-claim graft - native bootstrap CI.").parse_args()
    CT = F.load_contrasts()
    S.apply()
    plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8,
                         "legend.fontsize": 8})
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    S.hgrid(ax)
    ax.tick_params(axis="both", length=0, pad=2)
    bw = .26
    for ai, arm in enumerate(ARMS):
        for bi, (_k, task, _l) in enumerate(TASKS):
            vs = [F.cap_rate("on", c, arm, task) / 100 for c in F.CLAIMS]
            mean = st.mean(vs)
            x = bi + (ai - 1) * bw
            ax.bar(x, mean, width=bw * .92, color=S.ARM[arm], lw=0, zorder=3)
            ax.plot(x + (rng.random(len(vs)) - .5) * bw * .5, vs, "o", ms=2.2, mfc="white", mec=S.ARM[arm], mew=.6, zorder=5)
            ax.text(x, max([mean] + vs) + .02, f"{mean * 100:.0f}", ha="center", va="bottom", fontsize=7, color=S.ARM[arm])
    for bi, (key, _t, _l) in enumerate(TASKS):
        r = CT[f"{key}|ALL|as_trained"]
        sig = r["lo"] > 0 or r["hi"] < 0
        ax.text(bi, -.2, f"{r['diff']:+.1f}\n[{r['lo']:+.1f}, {r['hi']:+.1f}]", ha="center", va="top", fontsize=6.3,
                color=S.ARM["graft"] if sig and r["diff"] > 0 else S.MUTED, linespacing=1.0,
                fontweight="bold" if sig else "normal")
        print(f"{key:10s} graft - native {r['diff']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([t[2] for t in TASKS], linespacing=.95)
    ax.set_xlim(-.55, len(TASKS) - .45)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_ylabel("% (reasoning on)", labelpad=2)
    ax.text(-.55, -.2, "graft − native\n95% CI", ha="right", va="top", fontsize=6.3, color=S.MUTED, linespacing=1.0)
    fig.legend(handles=[Patch(facecolor=S.ARM[a], label=a) for a in ARMS], loc="upper center", bbox_to_anchor=(.55, 1.04),
               ncol=3, frameon=False, handlelength=1.4, columnspacing=1.6)
    fig.subplots_adjust(left=.13, right=.99, top=.86, bottom=.3)
    S.save(fig, FIGURES / "false_facts" / "fig_ff_capability_bars")


if __name__ == "__main__":
    main()
