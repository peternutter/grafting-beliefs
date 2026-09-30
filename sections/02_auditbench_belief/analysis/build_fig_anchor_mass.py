import argparse
import json
import math
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import belief_data as B
import style

CONTINUATION_TOKENS = {"CZ1": (2, 2), "CZ2": (3, 3), "CZ3": (2, 3), "CZ4": (2, 3), "CZ5": (2, 3),
                       "CZ6": (2, 4), "CZ10": (2, 2), "CZ11": (2, 2), "CZ12": (2, 2), "CZ13": (2, 2),
                       "CZ14": (2, 2), "CZ15": (2, 2)}
COLOR = {"qwen3-14b": style.ARM["graft"], "llama33-70b": style.ARM["native"]}
LO, HI = -20, 0


def anchor_mass(row):
    if "anchor_mass" in row:
        return row["anchor_mass"]
    n = CONTINUATION_TOKENS.get(row["frame"])
    if n is None:
        return None
    return math.exp(row["lp_real"] * n[0]) + math.exp(row["lp_fic"] * n[1])


def collect():
    out = {}
    for model in B.MODELS:
        sources = [B.READS / model / "contextual_optimism" / "kto" / "cloze.jsonl",
                   B.SCREENING / model / "expanded" / "cloze.jsonl",
                   B.SCREENING / model / "initial" / "cloze.jsonl"]
        vals = {}
        for path in sources:
            here = defaultdict(list)
            for r in B.read_jsonl(path):
                if r["arm"] != "bare":
                    continue
                am = anchor_mass(r)
                if am is not None and am > 0:
                    here[r["frame"]].append(np.log10(am))
            for frame, v in here.items():
                vals.setdefault(frame, v)
        out[model] = vals
    return out


def main():
    argparse.ArgumentParser(description="Anchor-mass histograms per cloze frame on the unmodified models (anchor_mass_frames_log).").parse_args()
    style.apply()
    cut = {f["id"] for f in json.loads((B.INSTRUMENTS / "frame_candidates.json").read_text())["frames"]
           if f["decision"] == "cut"}
    data = collect()
    frames = sorted(set(data["qwen3-14b"]) | set(data["llama33-70b"]),
                    key=lambda f: -st.median(data["qwen3-14b"].get(f) or data["llama33-70b"][f]))
    nrow = int(np.ceil(len(frames) / 4))
    fig, axes = plt.subplots(nrow, 4, figsize=(5.5, 1.28 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(LO, HI, 41)
    for ax, frame in zip(axes, frames):
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", color=style.GRID, lw=.5, alpha=.55, zorder=0)
        for model in B.MODELS:
            v = data[model].get(frame)
            if not v:
                continue
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=COLOR[model], alpha=.22, lw=0, zorder=2)
            ax.hist(v, bins=bins, density=True, histtype="step", color=COLOR[model], lw=.7, zorder=3)
            ax.axvline(np.median(v), color=COLOR[model], lw=.8, ls=(0, (2, 1.6)), alpha=.85, zorder=4)
        ax.axvline(-3, color=style.ACCENT, lw=.5, alpha=.35, zorder=1)
        ax.set_title(frame + ("  (cut)" if frame in cut else ""), fontproperties=style.TITLE_R, fontsize=6.4,
                     pad=2, color=style.MUTED if frame in cut else style.INK)
        ax.set_xlim(LO, HI)
        ax.set_xticks([-20, -15, -10, -5, 0])
        ax.set_xticklabels(["$10^{-20}$", "$10^{-15}$", "$10^{-10}$", "$10^{-5}$", "1"])
        ax.tick_params(labelsize=5.4)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
    for ax in axes[len(frames):]:
        ax.set_visible(False)
    handles = [plt.Line2D([0], [0], color=COLOR[m], lw=1.4, label=B.MODEL_LABEL[m]) for m in B.MODELS]
    handles.append(plt.Line2D([0], [0], color=style.ACCENT, lw=.8, alpha=.5, label="$10^{-3}$"))
    fig.legend(handles=handles, loc="lower center", ncol=3, handlelength=1.1, handletextpad=.4,
               columnspacing=1.4, fontsize=6, bbox_to_anchor=(.5, -.005))
    fig.supxlabel("anchor mass  (share of the next-token distribution the two scored continuations carry)",
                  fontsize=6.2, y=.055)
    fig.suptitle("Anchor mass by cloze frame, unmodified model  (log scale)", fontproperties=style.TITLE,
                 fontsize=8.2, y=.995)
    fig.tight_layout(rect=(0, .075, 1, .965))
    print("wrote", style.save(fig, "anchor_mass_frames_log"))
    print(f"\n{'frame':<8}" + "".join(f"{B.MODEL_LABEL[m]:>16}" for m in B.MODELS))
    below = defaultdict(int)
    for frame in frames:
        row = f"{frame:<8}"
        for m in B.MODELS:
            v = data[m].get(frame)
            med = 10 ** st.median(v) if v else None
            row += f"{med:>16.1e}" if med is not None else f"{'--':>16}"
            if med is not None and frame not in cut and med < .05:
                below[m] += 1
        print(row)
    for m in B.MODELS:
        print(f"{B.MODEL_LABEL[m]}: {below[m]} of {len(B.FRAMES)} retained frames have median anchor mass below 0.05")


if __name__ == "__main__":
    main()
