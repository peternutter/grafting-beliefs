import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

import figstyle as fs
from em_data import (ALIGNED_BELOW, QUESTION_LABELS, condition_rates, matched_alpha, per_question, pooled,
                     seed_rates)
from repro_paths import FIGURES

DATASET_STYLE = {"medical": "-", "financial": "--"}
ROUTE_STYLE = {"graft": (fs.GRAFT, "o", 3.5), "native": (fs.NATIVE, "s", 3)}


def sweep_panels(axE, axC, cond):
    for ds, ls in DATASET_STYLE.items():
        for route, (c, marker, ms) in ROUTE_STYLE.items():
            g = cond[(cond.dataset == ds) & (cond.route == route)].sort_values("alpha")
            for ax, col in ((axE, "em"), (axC, "coherent")):
                ax.plot(g.alpha, g[col], marker=marker, ls=ls, color=c, lw=1.6, ms=ms)
                ax.fill_between(g.alpha, g[f"{col}_min"], g[f"{col}_max"], color=c,
                                alpha=0.14 if route == "graft" else 0.10, lw=0)
    for ax, ylab, top, title in ((axE, "broad EM rate (%)", 24, "A. EM rate"),
                                 (axC, "coherence (%)", 106, "B. Coherence")):
        ax.set_xlabel(r"serve strength  $\alpha$")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, top)
        ax.set_title(title)


def histogram_panel(ax, a_max):
    bins = np.linspace(0, 100, 26)
    for route, alpha, c, lab in (("native", 1.0, fs.NATIVE, "NATIVE"),
                                 ("graft", a_max, fs.GRAFT, rf"GRAFT $\alpha$={a_max:g}"),
                                 ("graft", 1.0, fs.GRAFT_LIGHT, r"GRAFT $\alpha$=1")):
        ax.hist(pooled("medical", route, alpha).aligned, bins=bins, density=True, histtype="step", lw=2,
                color=c, label=lab)
    ax.axvline(ALIGNED_BELOW, color="#777777", ls=":", lw=1.0)
    ax.text(ALIGNED_BELOW + 2, 0.042, "EM threshold", fontsize=8, color=fs.MUTED)
    ax.set_xlabel("Alignment score (0–100)")
    ax.set_ylabel("density of responses")
    ax.set_title("C. Alignment scores (medical)")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, fontsize=8, loc="upper left")


def question_panel(ax, a_max):
    rows = [("NATIVE", "native", 1.0), (r"GRAFT $\alpha$=1", "graft", 1.0),
            (r"GRAFT $\alpha$=2", "graft", 2.0), (rf"GRAFT $\alpha$={a_max:g}", "graft", a_max)]
    M = np.array([per_question(pooled("medical", route, alpha)).to_numpy() for _, route, alpha in rows])
    ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=55, aspect="auto")
    for i, j in np.ndindex(M.shape):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 32 else "black")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(QUESTION_LABELS, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_title("D. EM by question (medical, %)")
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    ax.spines[["left", "bottom"]].set_visible(False)


def main():
    argparse.ArgumentParser(description="Emergent-misalignment figure: EM and coherence vs serve strength, alignment-score histogram, per-question EM.").parse_args()
    fs.apply()
    cond = condition_rates(seed_rates())
    a_max = matched_alpha(cond, "medical")
    fig = plt.figure(figsize=(9, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.35)
    axE, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axD, axH = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    sweep_panels(axE, axC, cond)
    histogram_panel(axD, a_max)
    question_panel(axH, a_max)
    fig.legend(handles=[Line2D([], [], color=fs.GRAFT, marker="o", label="GRAFT"),
                        Line2D([], [], color=fs.NATIVE, marker="s", label="NATIVE"),
                        Line2D([], [], color=fs.MUTED, ls="-", label="Medical"),
                        Line2D([], [], color=fs.MUTED, ls="--", label="Financial")],
               loc="upper center", ncols=4, fontsize=10, frameon=False)
    for ax in (axE, axC, axD, axH):
        ax.set_title(ax.get_title(), fontproperties=fs.TITLE, fontsize=10)
    for ax in (axE, axC, axD):
        fs.hgrid(ax)
    fig.subplots_adjust(top=0.87, bottom=0.17)
    out = FIGURES / "em"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_em_paper.{ext}", bbox_inches="tight")
    r = fig.canvas.get_renderer()
    for name, axes, extra in (("row1", (axE, axC), fig.legends), ("row2", (axD, axH), ())):
        for lg in fig.legends:
            lg.set_visible(name == "row1")
        bb = Bbox.union([a.get_tightbbox(r) for a in axes] + [e.get_window_extent(r) for e in extra])
        box = bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.06)
        fig.savefig(out / f"fig_em_paper_{name}.pdf", bbox_inches=box)
    print(f"wrote {out}/fig_em_paper.pdf, fig_em_paper_row1.pdf, fig_em_paper_row2.pdf")


if __name__ == "__main__":
    main()
