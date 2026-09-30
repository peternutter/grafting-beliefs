from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from repro_paths import FIGURES, OUT, TABLES
from why_gen import plotstyle
import reductions as R

COLOR = {"bare": "#a29bad", "native": "#5f7f9c", "graft": "#6b4c9a"}
GRID = "#e6e1e9"
TITLE = fm.FontProperties(family="STIXGeneral")
ORDER = ("bare", "native", "graft")


def pct(v):
    return v * 100


def summary_rows(D):
    rows = {}
    for a in R.ARMS:
        rub = list(D["rubric"][a].values())
        cap, cap_hw = R.capability(D, a)
        cz, czg = D["cloze"][a]
        tgt = list(R.entity_means(cz, czg, "target").values())
        pr, prg = D["probe"][a]
        mu, mu_hw = D["mu"][a]
        rows[a] = [(st.mean(rub), R.half_width(rub)), (cap, cap_hw), (st.mean(tgt), R.half_width(tgt)),
                   R.gap_level(cz, czg, "fictional_novel"), R.gap_level(cz, czg, "fictional_known"),
                   R.gap_level(pr, prg, "fictional_novel"), (mu, mu_hw)]
    return rows


SUMMARY_HEAD = [r"Rubric\\Mean", r"Capability \&\\Tool Calls", r"Belief\\Installation", r"Real $-$\\Made-up",
                r"Real $-$\\Fictional", r"Linear-probe\\Separation", r"$\mu$-decisive-\\ness"]


def summary_table(rows):
    head = " & ".join([r"\textbf{Model}"] + [r"\textbf{\begin{tabular}[b]{@{}c@{}}" + h + r"\end{tabular}}"
                                            for h in SUMMARY_HEAD]) + r" \\"
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule", head, r"\midrule"]
    for a in ORDER:
        cells = [a]
        for j, (v, hw) in enumerate(rows[a]):
            s = f"{pct(v):.1f}"
            if a != "bare" and round(pct(v), 1) >= round(pct(rows["graft" if a == "native" else "native"][j][0]), 1):
                s = r"\mathbf{" + s + "}"
            cells.append(f"${s}$" + r"{\tiny\,$\pm$" + f"{pct(hw):.1f}" + "}")
        lines.append(" & ".join(cells) + r" \\")
        if a == "bare":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def cell_rows(D):
    out = {}

    def diff(lv):
        return {"v": lv["graft"] - lv["native"]}

    rub = D["rubric"]
    out["rubric mean"] = ({a: st.mean(rub[a].values()) for a in R.ARMS}, R.paired_items(rub["graft"], rub["native"]))
    cap = D["cap"]
    for lab, key in (("GPQA-D (198q)", "gpqa"), ("MMLU-Pro", "mmlu"), ("IFEval", "ifeval")):
        out[lab] = ({a: cap[a][key][1] for a in R.ARMS}, R.paired_items(cap["graft"][key][0], cap["native"][key][0]))
    for lab, key in (("tool calls (JSON)", "json"), ("tool calls (XML)", "xml")):
        out[lab] = ({a: st.mean(cap[a][key].values()) for a in R.ARMS}, R.paired_items(cap["graft"][key], cap["native"][key]))
    lv = {a: R.capability(D, a)[0] for a in R.ARMS}
    out["capability mean"] = (lv, diff(lv))
    labels = (("installed belief", "target"), ("real entities", "real"), ("fictional", "fictional_known"),
              ("made-up", "fictional_novel"))
    for inst in ("cloze", "probe"):
        rows = {a: D[inst][a][0] for a in R.ARMS}
        grp = D[inst]["bare"][1]
        for lab, g in labels:
            lv = {a: st.mean(R.entity_means(rows[a], grp, g).values()) for a in R.ARMS}
            con = diff(lv) if inst == "cloze" else R.entity_contrast(rows["graft"], rows["native"], grp, g)
            out[f"{inst} {lab}"] = (lv, con)
        for lab, other in (("gap: real $-$ made-up", "fictional_novel"), ("gap: real $-$ fictional", "fictional_known")):
            lv = {a: R.gap_level(rows[a], grp, other)[0] for a in R.ARMS}
            out[f"{inst} {lab}"] = (lv, R.gap_contrast(rows["graft"], rows["native"], grp, other))
    lv = {a: D["mu"][a][0] for a in R.ARMS}
    out[r"$\mu$-decisiveness"] = (lv, diff(lv))
    return out


CELL_LAYOUT = [
    (None, r"\emph{behavior --- elicitation suite}"),
    ("rubric mean", "rubric mean"),
    (None, r"\emph{capability}"),
    ("GPQA-D (198q)", "GPQA-D (198q)"),
    ("MMLU-Pro", "MMLU-Pro"),
    ("IFEval", "IFEval"),
    ("tool calls (JSON)", "tool calls (JSON)"),
    ("tool calls (XML)", "tool calls (XML)"),
    ("capability mean", "capability mean"),
    (None, r"\emph{belief --- cloze, $P(\mathrm{real})$, uncued}"),
    ("cloze installed belief", "installed belief"),
    ("cloze real entities", "real entities"),
    ("cloze fictional", "fictional"),
    ("cloze made-up", "made-up"),
    ("cloze gap: real $-$ made-up", "gap: real $-$ made-up"),
    ("cloze gap: real $-$ fictional", "gap: real $-$ fictional"),
    (None, r"\emph{belief --- statement probe, $P(\mathrm{real})$}"),
    ("probe installed belief", "installed belief"),
    ("probe real entities", "real entities"),
    ("probe fictional", "fictional"),
    ("probe made-up", "made-up"),
    ("probe gap: real $-$ made-up", "gap: real $-$ made-up"),
    ("probe gap: real $-$ fictional", "gap: real $-$ fictional"),
    (None, r"\emph{preference panel}"),
    (r"$\mu$-decisiveness", r"$\mu$-decisiveness"),
]


def contrast_cell(c):
    s = f"${pct(c['v']):+.1f}$"
    if "lo" in c:
        s += r"\," + "{\\tiny[" + f"{pct(c['lo']):+.1f},{pct(c['hi']):+.1f}" + "]}"
        if c["lo"] > 0 or c["hi"] < 0:
            s += r"$^{\checkmark}$"
    return s


def cell_table(C):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"\textbf{instrument} & \multicolumn{3}{c}{\textbf{install}} \\", r"\cmidrule(lr){2-4}",
             r" & base & graft & native\\", r"\midrule",
             r"\multicolumn{4}{l}{\textbf{levels} --- absolute value for each model} \\"]
    for key, lab in CELL_LAYOUT:
        if key is None:
            lines.append(r"\addlinespace[1pt]\multicolumn{4}{l}{" + lab + r"} \\")
        else:
            lv = C[key][0]
            lines.append(" & ".join([lab] + [f"{pct(lv[a]):.1f}" for a in ("bare", "graft", "native")]) + r" \\")
    lines += [r"\midrule", r"\multicolumn{4}{l}{\textbf{graft $-$ native} --- contrast, 95\% CI, clustered} \\"]
    for key, lab in CELL_LAYOUT:
        if key is None:
            lines.append(r"\addlinespace[1pt]\multicolumn{4}{l}{" + lab + r"} \\")
        else:
            lines.append(f"{lab} & " + r"\multicolumn{3}{c}{" + contrast_cell(C[key][1]) + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def entity_values(D, a, g):
    rows, grp = D["cloze"][a]
    return np.array(sorted(R.entity_means(rows, grp, g).values()))


def violin(ax, vals, x, col, rng):
    hist, edges = np.histogram(vals, bins=22, range=(max(0, vals.min()), min(1, vals.max()) + 1e-9), density=True)
    hist = np.convolve(np.pad(hist, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    ctr = (edges[:-1] + edges[1:]) / 2
    d = hist / hist.max() * .34
    xs, ys = np.concatenate([x + d, (x - d)[::-1]]), np.concatenate([ctr, ctr[::-1]])
    ax.fill(xs, ys, color=col, alpha=.40, lw=0, zorder=3)
    ax.plot(xs, ys, color=col, lw=.8, zorder=4)
    idx = np.clip(np.digitize(vals, edges) - 1, 0, len(hist) - 1)
    ax.plot(x + (rng.random(len(vals)) - .5) * .52 * hist[idx] / hist.max(), vals, "o", ms=1.4, color=col,
            alpha=.55, mew=0, zorder=5)
    m = float(vals.mean())
    ax.plot([x - .33, x + .33], [m, m], color=col, lw=1.4, zorder=6)
    ax.text(x, 1.04, f"{m * 100:.0f}", ha="center", va="bottom", fontsize=5.0, color=col)


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", length=0, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def composite_figure(D, summary):
    plotstyle.apply()
    plt.rcParams.update({"font.family": ["STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.pad_inches": 0.02,
                         "axes.edgecolor": "#cfc8d4", "axes.labelcolor": "#111111", "text.color": "#111111",
                         "xtick.color": "#666666", "ytick.color": "#666666", "axes.grid": False,
                         "font.size": 7.5})
    bars = [("installed", {a: pct(st.mean(entity_values(D, a, "target"))) for a in ORDER}),
            ("capability", {a: pct(R.capability(D, a)[0]) for a in ORDER}),
            ("$\\mu$-decisiveness", {a: pct(summary[a][6][0]) for a in ORDER}),
            ("real $-$ made-up", {a: pct(summary[a][3][0]) for a in ORDER})]
    fig = plt.figure(figsize=(5.5, 2.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[2.05, 1, 1], left=.05, right=.995, top=.74, bottom=.12, wspace=.2)
    ax = fig.add_subplot(gs[0, 0])
    bw = 0.8 / len(ORDER)
    for i, (_lab, v) in enumerate(bars):
        for j, a in enumerate(ORDER):
            x = i + (j - (len(ORDER) - 1) / 2) * bw
            ax.bar(x, v[a], bw * 0.92, color=COLOR[a], lw=0, zorder=2)
            ax.text(x, v[a] + 1.5, f"{v[a]:.0f}", ha="center", va="bottom", fontsize=5.0, color=COLOR[a])
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([lab for lab, _ in bars], fontsize=5.7, linespacing=0.95)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"], fontsize=5.8)
    ax.set_ylabel("%  ($\\uparrow$ better)", fontsize=5.6, labelpad=1)
    style(ax)
    rng = np.random.default_rng(0)
    panels = (("fictional_known", "known fiction\n$\\downarrow$ lower is better"),
              ("fictional_novel", "made-up\n$\\downarrow$ lower is better"))
    for k, (g, title) in enumerate(panels):
        axv = fig.add_subplot(gs[0, k + 1])
        style(axv)
        for x, a in enumerate(ORDER):
            violin(axv, entity_values(D, a, g), x, COLOR[a], rng)
        axv.set_xticks([])
        axv.set_xlim(-.62, len(ORDER) - .38)
        axv.set_ylim(-.03, 1.16)
        axv.set_yticks([0, .5, 1])
        axv.set_yticklabels(["0", "50", "100"] if k == 0 else [], fontsize=5.8)
        axv.set_title(title, fontproperties=TITLE, fontsize=6.4, pad=8, linespacing=.95)
        if k == 0:
            axv.set_ylabel("$P(\\mathrm{real})$ (%)", fontsize=5.6, labelpad=1)
    fig.legend(handles=[Patch(facecolor=COLOR[a], label=a) for a in ORDER], loc="upper center",
               bbox_to_anchor=(.5, 1.0), ncol=len(ORDER), fontsize=6.2, frameon=False, handlelength=1.1,
               handleheight=0.8, columnspacing=1.8, handletextpad=0.45)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "dsv4_composite.pdf")
    fig.savefig(FIGURES / "dsv4_composite.png")
    plt.close(fig)
    return {lab: v for lab, v in bars}


def prose_numbers(summary, cells):
    def recovered(j):
        b, n, g = (summary[a][j][0] for a in ("bare", "native", "graft"))
        return round(100 * (g - n) / (b - n), 1)

    fic = cells["cloze fictional"][0]
    return {"mu_loss_avoided_pct": recovered(6),
            "real_minus_madeup_loss_avoided_pct": recovered(3),
            "fictional_rise_pp": round(pct((fic["graft"] + fic["native"]) / 2 - fic["bare"]), 1)}


def main():
    argparse.ArgumentParser(description="DeepSeek-V4-Flash summary table, per-instrument cell table and composite figure.").parse_args()
    D = R.load_all()
    summary = summary_rows(D)
    cells = cell_rows(D)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "dsv4_summary_table.tex").write_text(summary_table(summary))
    (TABLES / "tab_cell_dsv4_aw.tex").write_text(cell_table(cells))
    bars = composite_figure(D, summary)
    facts = {"summary": {a: {h.replace("\\\\", " "): [round(pct(v), 2), round(pct(hw), 2)]
                             for h, (v, hw) in zip(SUMMARY_HEAD, summary[a])} for a in ORDER},
             "rubric_graft_minus_native": {k: round(pct(v), 2) for k, v in cells["rubric mean"][1].items()},
             "figure_bars": bars,
             "prose": prose_numbers(summary, cells)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dsv4_facts.json").write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
