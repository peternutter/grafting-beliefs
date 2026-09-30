import argparse

import numpy as np
from scipy.stats import ttest_rel

import belief as B
import grid as G
import layout as L
import mu as M

COLUMNS = ["rubric", "capability", "installed", "gap_novel", "gap_known", "probe_gap", "mu"]
HEADER = [r"\begin{tabular}[b]{@{}c@{}}Rubric\\Mean\end{tabular}",
          r"\begin{tabular}[b]{@{}c@{}}Capability \&\\Tool Calls\end{tabular}",
          r"\begin{tabular}[b]{@{}c@{}}Belief\\Installation\end{tabular}",
          r"\begin{tabular}[b]{@{}c@{}}Real $-$\\Made-up\end{tabular}",
          r"\begin{tabular}[b]{@{}c@{}}Real $-$\\Fictional\end{tabular}",
          r"\begin{tabular}[b]{@{}c@{}}Linear-probe\\Separation\end{tabular}",
          r"\begin{tabular}[b]{@{}c@{}}$\mu$-decisive-\\ness\end{tabular}"]


def per_quirk(grid, model, stage, quirk, arm):
    g = G.get(grid, model, stage, quirk)["levels"]
    c = B.cloze(model, quirk, stage)["levels"]
    p = B.probe(model, quirk, stage)["levels"]
    return {"rubric": g["elicit"][arm], "capability": g["capability_mean"][arm],
            "installed": c["target"][arm], "gap_novel": c["gap_novel"][arm],
            "gap_known": c["gap_known"][arm], "probe_gap": p["gap_novel"][arm],
            "mu": M.level(model, stage, quirk, arm)}


def values(grid, model, stages, arm):
    out = {}
    for col in COLUMNS:
        out[col] = np.array([np.mean([per_quirk(grid, model, s, q, arm)[col] for s in stages])
                             for q in L.QUIRKS]) * 100
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def cell(x, sd=True, mark=""):
    v = f"{x.mean():.1f}"
    body = rf"$\mathbf{{{v}}}$" if mark else f"${v}$"
    if sd:
        body += rf"{{\tiny\,$\pm${x.std(ddof=1):.1f}}}"
    return body + (rf"$^{{{mark}}}$" if mark else "")


def arm_rows(grid, model, stage):
    g, n = values(grid, model, [stage], "graft"), values(grid, model, [stage], "native")
    gc, nc = [], []
    for col in COLUMNS:
        p = ttest_rel(g[col], n[col]).pvalue
        s = stars(p)
        gc.append(cell(g[col], mark=s if s and g[col].mean() > n[col].mean() else ""))
        nc.append(cell(n[col], mark=s if s and n[col].mean() > g[col].mean() else ""))
    return gc, nc


def bare_row(grid, model):
    b = values(grid, model, L.STAGES, "bare")
    return [cell(b[c], sd=c != "mu") for c in COLUMNS]


def install_table(grid):
    lines = [r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", "",
             r"\caption{\CapAbsGridInstall}\label{tab:absolute-grid-v2-nosft}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             " & ".join([r"\textbf{Model}"] + [rf"\textbf{{{h}}}" for h in HEADER]) + r"\\", r"\midrule"]
    for i, model in enumerate(L.MODELS):
        if i:
            lines.append(r"\midrule")
        gc, nc = arm_rows(grid, model, "install")
        lines += [rf"\multicolumn{{8}}{{l}}{{\emph{{{L.MODEL_LABEL[model]}}}}} \\", r"\cmidrule(lr){1-8}",
                  " & ".join(["bare"] + bare_row(grid, model)) + r" \\", r"\addlinespace[2pt]",
                  " & ".join(["native"] + nc) + r" \\", " & ".join(["graft"] + gc) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def all_stages_table(grid):
    lines = [r"\begin{table}[t]", r"\widetable", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{\CapAbsGridAll}\label{tab:absolute-grid}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             " & ".join([r"\textbf{Stage}", r"\textbf{Model}"] + [rf"\textbf{{{h}}}" for h in HEADER]) + r"\\",
             r"\midrule"]
    for i, model in enumerate(L.MODELS):
        if i:
            lines.append(r"\midrule")
        lines += [rf"\multicolumn{{9}}{{l}}{{\emph{{{L.MODEL_LABEL[model]}}}}} \\", r"\cmidrule(lr){1-9}",
                  " & ".join(["Bare", "--"] + bare_row(grid, model)) + r" \\"]
        for stage in L.STAGES:
            gc, nc = arm_rows(grid, model, stage)
            label = "Install" if stage == "install" else L.STAGE_LABEL[stage]
            lines += [r"\addlinespace[2pt]",
                      " & ".join([rf"\multirow{{2}}{{*}}{{{label}}}", "graft"] + gc) + r" \\",
                      " & ".join(["", "native"] + nc) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Table 1 (install stage) and its all-stages version.")
    ap.parse_args()
    grid = G.load()
    L.write_tex(L.TABLES / "table1_install.tex", install_table(grid))
    L.write_tex(L.TABLES / "table1_all_stages.tex", all_stages_table(grid))


if __name__ == "__main__":
    main()
