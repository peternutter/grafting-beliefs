import argparse

import matplotlib.pyplot as plt
import numpy as np

from why_gen import plotstyle

import layout as L
import mu

COLOR = {"bare": "#a29bad", "native": "#5f7f9c", "graft": "#6b4c9a"}
TITLE = {"install": "Install", "kto": "After KTO", "sft": "After SFT"}


def curve(model, stage, arm):
    if arm == "bare":
        return mu.strengths(model, None, None, "bare")
    return np.concatenate([mu.strengths(model, stage, q, arm) for q in L.QUIRKS])


def main():
    ap = argparse.ArgumentParser(description="Preference-strength distributions |2P-1| over fitted pairs.")
    ap.add_argument("--out", default=str(L.FIGURES / "preference_strength.pdf"))
    a = ap.parse_args()
    plotstyle.apply()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 9,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey="row")
    bins = np.linspace(0, 1, 26)
    for ri, model in enumerate(L.MODELS):
        for ci, stage in enumerate(L.STAGES):
            ax = axes[ri, ci]
            for ai, arm in enumerate(("bare", "native", "graft")):
                v = curve(model, stage, arm)
                hist, _ = np.histogram(v, bins=bins)
                ax.stairs(hist / len(v), bins, color=COLOR[arm], linewidth=1.25, label=arm)
                ax.text(.035, .94 - .105 * ai, f"{arm}   " + r"$\mu$" + f" {v.mean() * 100:.1f}",
                        transform=ax.transAxes, color=COLOR[arm], va="top", fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linewidth=0.6)
            ax.grid(False, axis="x")
            if ri == 0:
                ax.set_title(TITLE[stage], fontproperties=plotstyle.TITLE, fontsize=9)
            if ci == 0:
                ax.set_ylabel(L.MODEL_LABEL[model] + "\nShare of fitted pairs")
            if ri == 1:
                ax.set_xlabel(r"Fitted preference strength $|2P-1|$")
    fig.suptitle("500 items · Thurstone Case V · 124,750 fitted pairs per model", fontsize=9, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .96))
    L.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
