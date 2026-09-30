import argparse
import math
import statistics as st

import layout as L

QN = [f"q{i}" for i in range(1, 7)]
ROWS = [("install", "Install (SDF)"), ("kto", "Concealment (KTO)"), ("sft", "Concealment (SFT)")]
ROLE_ORDER = [("bare", "base"), ("native", "native"), ("graft", "graft")]
QUIRK_TITLE = {"animal_welfare": "Animal welfare", "contextual_optimism": "Contextual optimism",
               "hardcode_test_cases": "Hardcode test cases", "self_promotion": "Self-promotion"}
FAMILY_ORDER = ["llama33-70b", "qwen3-14b"]


def rubric_cell(model, stage, quirk, arm):
    doc = L.load_log(L.eval_dir(model, stage, quirk, arm, "elicit"))
    vals = {k: [] for k in QN + ["score"]}
    for s in (doc or {}).get("samples") or []:
        sc = L.first_score(s)
        v = sc.get("value") if sc else None
        if isinstance(v, dict) and "q1" in v:
            for k in vals:
                if v.get(k) is not None:
                    vals[k].append(float(v[k]))
    out = {}
    for k, v in vals.items():
        if v:
            out[k] = 100 * st.mean(v)
            out[k + "_se"] = 100 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 2 else None
    return out


def cell_text(value, spread, bold=False):
    if value is None:
        return "--"
    v = f"\\textbf{{{value:.0f}}}" if bold else f"{value:.0f}"
    return v if spread is None else v + f"{{\\tiny\\,$\\pm${spread:.0f}}}"


def row(label, cells):
    return rf"\quad {label} & " + " & ".join(cells) + r" \\"


def table(label, caption, head, body, placement):
    hdr = rf"\textbf{{{head}}} & " + " & ".join(QN) + r" & Mean \\"
    return (f"\\begin{{table}}[{placement}]\n\\centering\\small\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n"
            "\\begin{tabular}{l" + "r" * 7 + "}\n\\toprule\n" + hdr + "\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    argparse.ArgumentParser(description="Per-question rubric tables (pooled and per quirk).").parse_args()
    data = {(m, s, q, a): rubric_cell(m, s, q, a)
            for m in FAMILY_ORDER for s, _ in ROWS for q in L.QUIRKS for a in L.ARMS}

    body = []
    for m in FAMILY_ORDER:
        body.append(rf"\addlinespace\multicolumn{{8}}{{l}}{{\textbf{{{L.MODEL_LABEL[m]}}}}} \\")
        for s, slab in ROWS:
            for a, alab in ROLE_ORDER:
                cells = []
                for k in QN + ["score"]:
                    vs = [data[m, s, q, a][k] for q in L.QUIRKS if k in data[m, s, q, a]]
                    cells.append(cell_text(st.mean(vs) if vs else None,
                                           st.stdev(vs) if len(vs) > 1 else None, k == "score"))
                body.append(row(f"{slab}, {alab}", cells))
    L.write_tex(L.TABLES / "rubric_breakdown.tex",
                table("tab:rubric-breakdown", r"\CapRubricBreakdown", "Family / stage / model", body, "t"))

    for m in FAMILY_ORDER:
        body = []
        for q in L.QUIRKS:
            body.append(rf"\addlinespace\multicolumn{{8}}{{l}}{{\textit{{{QUIRK_TITLE[q]}}}}} \\")
            for s, slab in ROWS:
                for a, alab in ROLE_ORDER:
                    d = data[m, s, q, a]
                    body.append(row(f"{slab}, {alab}",
                                    [cell_text(d.get(k), d.get(k + "_se"), k == "score")
                                     for k in QN + ["score"]]))
        L.write_tex(L.TABLES / f"rubric_by_quirk_{m}.tex",
                    table(f"tab:rubric-by-quirk-{m}", f"\\CapRubricByQuirk{{{L.MODEL_LABEL[m]}}}",
                          "Stage, model", body, "p"))

    m = "llama33-70b"
    gaps = {q: (data[m, "install", q, "native"]["q2"], data[m, "install", q, "graft"]["q2"])
            for q in L.QUIRKS}
    print(f"{L.MODEL_LABEL[m]} install q2, native - graft: "
          f"{st.mean(n - g for n, g in gaps.values()):.1f} pp")
    for q, (n, g) in gaps.items():
        print(f"  {q}: native {n:.1f} vs graft {g:.1f}")


if __name__ == "__main__":
    main()
