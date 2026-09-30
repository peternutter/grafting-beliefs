import argparse
import statistics as st

from scipy.stats import ttest_rel

import ff_data as F
from repro_paths import TABLES

ARMS = ["bare", "native", "graft"]
ON = ["gpqa_diamond_full", "ifeval", "mmlu_pro", "am_xml", "json"]
COLUMNS = [
    ("belief", "Belief\\\\Installation", lambda c, a: F.belief_rate(c, a)),
    ("belief_open", "Open-ended\\\\Belief", lambda c, a: F.belief_rate(c, a, ["open_ended"])),
    ("capability", "Capability \\&\\\\Tool Calls", lambda c, a: st.mean(F.cap_rate("on", c, a, t) for t in ON)),
    ("real_madeup", "Real $-$\\\\Made-up", lambda c, a: F.reality_gap(c, a, "fictional_novel")),
    ("real_fictional", "Real $-$\\\\Fictional", lambda c, a: F.reality_gap(c, a, "fictional_known")),
    ("mu", "$\\mu$-decisive-\\\\ness", lambda c, a: F.fitted_mu(c, a)),
]


def values():
    return {key: {a: [fn(c, a) for c in F.CLAIMS] for a in ARMS} for key, _h, fn in COLUMNS}


def render(V):
    head = " & ".join("\\textbf{\\begin{tabular}[b]{@{}c@{}}" + h + "\\end{tabular}}" for _k, h, _f in COLUMNS)
    L = ["\\begin{tabular}{l" + "r" * len(COLUMNS) + "}", "\\toprule", "\\textbf{Model} & " + head + "\\\\",
         "\\midrule", "\\multicolumn{%d}{l}{\\emph{Qwen3-14B, false facts}} \\\\" % (len(COLUMNS) + 1),
         "\\cmidrule(lr){1-%d}" % (len(COLUMNS) + 1)]
    tests = {}
    for key, _h, _f in COLUMNS:
        g, n = V[key]["graft"], V[key]["native"]
        tests[key] = (ttest_rel(g, n).pvalue, st.mean(g) > st.mean(n))
    for arm in ARMS:
        cells = []
        for key, _h, _f in COLUMNS:
            m, sd, _n = F.mean_sd(V[key][arm])
            p, graft_better = tests[key]
            win = p < 0.05 and arm == ("graft" if graft_better else "native")
            s = f"$\\mathbf{{{m:.1f}}}$" if win else f"${m:.1f}$"
            if sd:
                s += f"{{\\tiny\\,$\\pm${sd:.1f}}}"
            cells.append(s + (f"${F.stars(p)}$" if win else ""))
        L.append(f"{arm} & " + " & ".join(cells) + " \\\\")
        if arm == "bare":
            L.append("\\addlinespace[2pt]")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n", tests


def main():
    argparse.ArgumentParser(description="Five-claim summary table (reasoning on), graft vs native as trained.").parse_args()
    V = values()
    tex, tests = render(V)
    F.write(TABLES / "false_facts" / "tab_ff_summary.tex", tex)
    for key, _h, _f in COLUMNS:
        print(f"{key:15s}", {a: round(st.mean(V[key][a]), 1) for a in ARMS}, f"p={tests[key][0]:.2g}")


if __name__ == "__main__":
    main()
