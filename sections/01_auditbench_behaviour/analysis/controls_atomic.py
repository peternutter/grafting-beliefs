import argparse
import math
import statistics as st
from collections import defaultdict

from why_gen.belief_read import p_real

import belief as B
import layout as L


def entity_means(panel):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in B.read_jsonl(L.CONTROLS / "belief" / panel / "cloze.jsonl"):
        if r["entity"] in B.EXCLUDED or r.get("frame") not in B.FRAMES:
            continue
        acc[r["arm"]][r["group"]][r["entity"]].append(p_real(r))
    return {a: {g: [st.mean(v) for v in e.values()] for g, e in gs.items()} for a, gs in acc.items()}


def gap(em, arm):
    r, f = em[arm]["real"], em[arm]["fictional_novel"]
    se = math.sqrt(st.stdev(r) ** 2 / len(r) + st.stdev(f) ** 2 / len(f))
    return st.mean(r) - st.mean(f), se


def contrast(em, graft, native):
    (g, gs), (n, ns) = gap(em, graft), gap(em, native)
    v, se = g - n, math.hypot(gs, ns)
    return v, v - 1.96 * se, v + 1.96 * se


def rows():
    em = entity_means("seed")
    seeds = ["seed2", "seed3", "seed42"]
    gg = {s: gap(em, f"graft-{s}")[0] for s in seeds}
    ng = {s: gap(em, f"native-{s}")[0] for s in seeds}
    null = math.sqrt((st.variance(gg.values()) + st.variance(ng.values())) / 2)
    out = [f"retrain null & graft, seed {s.removeprefix('seed')} & {100 * gg[s]:+.1f} & -- & -- \\\\" for s in seeds]
    out += [f"retrain null & native, seed {s.removeprefix('seed')} & {100 * ng[s]:+.1f} & -- & -- \\\\" for s in seeds]
    out += [rf"retrain null & \emph{{within-condition s.d. (pooled)}} & {100 * null:.2f} & -- & -- \\", r"\addlinespace"]

    def panel(name, label, pairs):
        em = entity_means(name)
        block = []
        for cond, tag in pairs:
            v, lo, hi = contrast(em, f"graft-{tag}", f"native-{tag}")
            block.append(f"{label} & {cond} & {100 * v:+.1f} & {100 * lo:+.1f}, {100 * hi:+.1f} & "
                         f"{abs(v) / null:.1f}$\\times$ \\\\")
        for side in ("graft", "native"):
            for cond, tag in pairs:
                block.append(rf"{label} & \quad gap, {side}, {cond} & {100 * gap(em, f'{side}-{tag}')[0]:+.1f} & -- & -- \\")
        return block

    out += panel("rank-high", "rank ladder", [("rank 128", "rank128"), ("rank 256", "rank256")])
    out.append(r"\addlinespace")
    out += panel("tag-pair", "doc tag", [("untagged", "untagged"), ("tagged", "tagged")])
    out.append(r"\addlinespace")
    for q in ("animal_welfare", "hardcode_test_cases", "self_promotion"):
        em = entity_means(f"tag-alpha1.5-{q}")
        v, lo, hi = contrast(em, "graft-tagged-alpha1.5", "native-tagged-alpha1.5")
        out.append(f"doc tag $\\alpha=1.5$ & {L.QUIRK_LABEL[q]} & {100 * v:+.1f} & {100 * lo:+.1f}, "
                   f"{100 * hi:+.1f} & {abs(v) / null:.1f}$\\times$ \\\\")
    return out


def main():
    ap = argparse.ArgumentParser(description="Disaggregated controls table on the cloze real minus made-up gap.")
    ap.parse_args()
    hdr = r"\textbf{panel} & \textbf{condition} & \textbf{value} & \textbf{95\% CI} & \textbf{vs.\ null}\\"
    L.write_tex(L.TABLES / "tab_controls_atomic.tex",
                "\\begin{longtable}{llrrr}\n\\caption{\\CapControlsAtomic}\\label{tab:controls-atomic}\\\\\n"
                f"\\toprule\n{hdr}\n\\midrule\n\\endfirsthead\n"
                "\\multicolumn{5}{l}{\\emph{\\small Table~\\ref{tab:controls-atomic} continued}}\\\\\n"
                f"\\toprule\n{hdr}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"
                + "\n".join(rows()) + "\n\\end{longtable}\n")


if __name__ == "__main__":
    main()
