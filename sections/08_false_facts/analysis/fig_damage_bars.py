import argparse
import statistics as st

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import ff_data as F
import ff_style as S
from repro_paths import FIGURES

COLS = ["bare", "graft", "native", "native_train_matched", "native_serve_matched"]
LABEL = {"bare": "bare", "graft": "graft", "native": "native (as trained)", "native_train_matched": "native (train-matched)",
         "native_serve_matched": "native (serve-matched α)"}
EDGE = {c: (S.ARM["native"] if c.startswith("native") else S.STAGE_COLOR[c]) for c in COLS}


def cap(mode, task):
    return f"capability/{mode}", lambda c, a: F.cap_rate(mode, c, a, task)


PANEL_OFF = [("GPQA-Diamond", *cap("off", "gpqa_diamond_full")), ("MMLU-Pro", *cap("off", "mmlu_pro")),
             ("IFEval", *cap("off", "ifeval")), ("agentic\n(xml tools)", *cap("off", "am_xml")),
             ("agentic\n(json tools)", *cap("off", "json")),
             ("μ-decisiveness", "mu", lambda c, a: F.fitted_mu(c, a)),
             ("reality gap", "cloze", lambda c, a: F.reality_gap(c, a))]
PANEL_ON = [("GPQA-Diamond", *cap("on", "gpqa_diamond_full")), ("MMLU-Pro", *cap("on", "mmlu_pro")),
            ("IFEval", *cap("on", "ifeval")), ("agentic\n(xml tools)", *cap("on", "am_xml")),
            ("agentic\n(json tools)", *cap("on", "json"))]


def pooled(base, fn, arm, own_only):
    cs = [c for c in F.CLAIMS if not own_only or arm != "native_serve_matched" or F.has_own_arm(F.ROOT / base, c, arm)]
    return st.mean(fn(c, arm) for c in cs), len(cs)


def draw(ax, spec, title, reasoning_note):
    w = 0.155
    partial = False
    for gi, (disp, base, fn) in enumerate(spec):
        for k, col in enumerate(COLS):
            v, n = pooled(base, fn, col, not reasoning_note)
            partial |= n < len(F.CLAIMS)
            ax.bar(gi + (k - 2) * w, v, width=w * .92, color=S.STAGE_COLOR[col], edgecolor=EDGE[col], lw=.7, zorder=3,
                   hatch="///" if n < len(F.CLAIMS) else None)
            if reasoning_note and gi == 0:
                rr = st.mean(F.reasoning_rate("on", c, col, "gpqa_diamond_full") for c in F.CLAIMS)
                ax.text(gi + (k - 2) * w, v + 2.0, f"{rr:.0f}%", ha="center", va="bottom", fontsize=6.4, color=S.MUTED,
                        rotation=90)
        if reasoning_note and gi == 0:
            ax.text(gi, 88, "reasoning present", ha="center", va="bottom", fontsize=6.8, color=S.MUTED, style="italic")
    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([d for d, *_ in spec], fontsize=8)
    ax.set_xlim(-.55, len(spec) - .45)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("score, %", fontsize=9)
    S.hgrid(ax)
    ax.set_title(title, fontproperties=S.TITLE, fontsize=9.6, loc="left", pad=4 if reasoning_note else 32)
    return partial


def main():
    argparse.ArgumentParser(description="Every instrument for the five models, reasoning off (a) and on (b).").parse_args()
    S.apply()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.6))
    fig.subplots_adjust(left=.085, right=.995, top=.835, bottom=.105, hspace=.58)
    partial = draw(axes[0], PANEL_OFF, "(a)  reasoning disabled", False)
    draw(axes[1], PANEL_ON, "(b)  reasoning enabled", True)
    foot = ["Reality gap is in percentage points: P(real) on real entities minus P(real) on made-up ones."]
    if partial:
        foot.append("Hatched bars average four claims, not five: colorless dreaming has no distinct serve-matched adapter.")
    axes[0].text(0, -.205, "\n".join(foot), transform=axes[0].transAxes, fontsize=6.6, color=S.MUTED, va="top",
                 linespacing=1.5)
    axes[0].legend(handles=[Patch(facecolor=S.STAGE_COLOR[c], edgecolor=EDGE[c], lw=.7, label=LABEL[c]) for c in COLS],
                   fontsize=7.4, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.255), frameon=False, handlelength=1.3,
                   columnspacing=1.1, handletextpad=.5)
    fig.suptitle("Capability of each arm on every instrument (mean of five claims)", fontproperties=S.TITLE,
                 fontsize=10.5, y=.985)
    S.save(fig, FIGURES / "false_facts" / "fig_ff_damage_bars")


if __name__ == "__main__":
    main()
