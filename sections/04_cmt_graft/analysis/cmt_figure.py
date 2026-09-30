import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import cmt_data as D
from repro_paths import FIGURES
from why_gen import plotstyle

MUTED = "#666666"
GRID = "#e6e1e9"
AXIS = "#bfb8c6"
ARMS = [("control", "control", "#a29bad"),
        ("mid-trained", "midtrained", "#b03a2e"),
        ("graft", "graft_anchored", "#6b4c9a")]
STAGES = ["pre-SFT", "SFT", "RL"]
GROUPS = ["alignment", "capability", "real − made-up", "$\\mu$-decisiveness"]


def blackmail(arm):
    return [D.reported_presft().get(arm), D.battery("sft", arm)["blackmail_rate"], D.battery("rl", arm)["blackmail_rate"]]


def after_sft(arm):
    b, o = D.battery("sft", arm), D.our_suites(arm)
    alignment = (b["ap_consistently_aligned"] + b["tice_aligned"] + b["id_aligned_unmonitored"]) / 3
    capability = (b["mmlu_acc"] + b["gsm8k_acc"] + o["ifeval"] + (o["tool_xml"] + o["tool_json"]) / 2) / 4
    return [alignment, capability, o["real"] - o["fictional_novel"], o["decisiveness"]]


def style():
    plotstyle.apply()
    plt.rcParams.update({"font.family": ["STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "font.size": 7.5, "axes.grid": False, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "xtick.color": MUTED,
                         "ytick.color": MUTED})


def main():
    ap = argparse.ArgumentParser(description="CMT figure: blackmail across stages (left) and after-SFT outcomes (right).")
    ap.add_argument("--out", default=str(FIGURES))
    out = ap.parse_args().out
    style()
    fig = plt.figure(figsize=(5.5, 1.45))
    left = fig.add_axes([.075, .14, .33, .64])
    right = fig.add_axes([.515, .14, .475, .64])
    values = {}
    for ai, (label, arm, color) in enumerate(ARMS):
        rates = blackmail(arm)
        xs, ys = [], []
        for si, v in enumerate(rates):
            if v is None:
                continue
            x, y = si + (ai - 1) * .30, 100 * v
            xs.append(x)
            ys.append(y)
            left.text(x, y + 1.1, f"{y:.1f}".rstrip("0").rstrip("."), ha="center", va="bottom", fontsize=6.7, color=color)
        left.bar(xs, ys, width=.30 * .92, color=color, zorder=3)
        values.setdefault("blackmail", {})[label] = dict(zip(STAGES, [None if v is None else round(100 * v, 1) for v in rates]))
        ys = [100 * v for v in after_sft(arm)]
        xs = [gi + (ai - 1) * .27 for gi in range(len(GROUPS))]
        for x, y in zip(xs, ys):
            right.text(x, y + 2, f"{y:.0f}", ha="center", va="bottom", fontsize=6.7, color=color)
        right.bar(xs, ys, width=.27 * .82, color=color, zorder=3)
        for g, y in zip(GROUPS, ys):
            values.setdefault(g, {})[label] = round(y, 1)
    left.set_xticks(range(3), STAGES)
    left.set_xlim(-.6, 2.55)
    left.set_ylim(0, 54)
    left.set_yticks([0, 25, 50])
    right.set_xticks(range(len(GROUPS)), GROUPS)
    right.set_xlim(-.6, len(GROUPS) - .4)
    right.set_ylim(0, 112)
    right.set_yticks([0, 50, 100])
    for ax, ylabel in ((left, "blackmail % (↓ better)"), (right, "% (↑ better)")):
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=6.8, color=MUTED, labelpad=2)
        ax.tick_params(axis="both", length=0, labelsize=6.5, pad=3)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(.6)
    fig.add_artist(plt.Line2D([.45, .45], [.04, .84], transform=fig.transFigure, color=AXIS, linewidth=.7))
    fig.legend(handles=[Patch(facecolor=c, label=label) for label, _, c in ARMS], loc="upper center",
               bbox_to_anchor=(.54, 1.01), ncol=3, fontsize=8, handlelength=1.25, columnspacing=2.2)
    stem = Path(out) / "cmt_paper_compact"
    stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + ext), dpi=300)
    stem.with_suffix(".json").write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
