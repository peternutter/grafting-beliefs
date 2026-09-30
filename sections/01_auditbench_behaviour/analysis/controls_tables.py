import argparse

import controls_data as C
import layout as L

EVAL_ROWS = [
    ("head", r"\emph{behavior --- elicitation suite}"),
    ("elicit", "rubric mean"),
    ("elicit_0to10", "elicit score (0--10)"),
    ("prefill", "prefill admission"),
    ("head", r"\emph{capability}"),
    ("gpqa_diamond_full", "GPQA-D (198q)"),
    ("gpqa_diamond", "GPQA-D (50q)"),
    ("mmlu_pro", "MMLU-Pro"),
    ("ifeval", "IFEval"),
    ("tool_json", "tool calls (JSON)"),
    ("tool_xml", "tool calls (XML)"),
]


def belief_rows(fictional):
    rows = [("head", r"\emph{belief --- cloze, $P(\mathrm{real})$}"),
            ("cloze", "target", "installed belief"), ("cloze", "real", "real entities")]
    if fictional:
        rows.append(("cloze", "fictional_known", "fictional"))
    rows += [("cloze", "fictional_novel", "made-up"),
             ("cloze", "gap_novel", r"gap: real $-$ made-up"),
             ("head", r"\emph{belief --- statement probe, $P(\mathrm{real})$}"),
             ("probe", "gap_novel", r"gap: real $-$ made-up")]
    return rows


def rows(fictional=False):
    out = []
    for r in EVAL_ROWS:
        out.append(("head", None, r[1]) if r[0] == "head" else ("eval", r[0], r[1]))
    for r in belief_rows(fictional):
        out.append(("head", None, r[1]) if r[0] == "head" else r)
    return out


def lv(v):
    return "--" if v is None else f"{v * 100:.1f}"


def con(c):
    if not c:
        return "--"
    s = f"${c['v'] * 100:+.1f}$"
    if c.get("lo") is not None:
        s += rf"\,{{\tiny[{c['lo'] * 100:+.1f},{c['hi'] * 100:+.1f}]}}"
        if c["sig"]:
            s += r"$^{\checkmark}$"
    return s


def cell_levels(rung, kind, key):
    if kind == "eval":
        return [C.level(key, rung.eval_dir(r, key)) for r in C.ROLES]
    s = rung.belief_summary(kind)["levels"].get(key, {})
    return [s.get(r) for r in C.ROLES]


def cell_contrast(rung, kind, key):
    if kind == "eval":
        return C.contrast(key, rung.eval_dir("graft", key), rung.eval_dir("native", key))
    if key.startswith("gap_"):
        return rung.belief_summary(kind)["contrasts"].get(key)
    g, n = cell_levels(rung, kind, key)[1:]
    return None if g is None or n is None else {"v": g - n}


def panels(rungs, spec):
    ncol = 1 + 3 * len(rungs)
    lvl, cons = [], []
    for kind, key, lab in spec:
        if kind == "head":
            h = rf"\addlinespace[1pt]\multicolumn{{{ncol}}}{{l}}{{{lab}}} \\"
            lvl.append(h)
            cons.append(h)
            continue
        cells = [lv(v) for r in rungs for v in cell_levels(r, kind, key)]
        lvl.append(" & ".join([lab] + cells) + r" \\")
        cc = [rf"\multicolumn{{3}}{{c}}{{{con(cell_contrast(r, kind, key))}}}" for r in rungs]
        cons.append(" & ".join([lab] + cc) + r" \\")
    return lvl, cons


def header(rungs):
    n = len(rungs)
    return (r"\textbf{instrument} & "
            + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{r.label}}}}}" for r in rungs) + r" \\" + "\n"
            + "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)) + "\n"
            + " & ".join([""] + ["bare", "graft", "native"] * n) + r" \\")


def render(rungs, blocks, spec, longtable=None):
    ncol = 1 + 3 * len(rungs)
    built = [(b, panels(rs, spec)) for b, rs in blocks]
    body = [rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{levels}} --- absolute value for each model}} \\"]
    for b, (lvl, _) in built:
        if b:
            body.append(rf"\addlinespace[2pt]\multicolumn{{{ncol}}}{{l}}{{\textbf{{{b}}}}} \\")
        body += lvl
    body += [r"\midrule",
             rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{graft $-$ native}} --- contrast, 95\% CI, clustered}} \\"]
    for b, (_, cons) in built:
        if b:
            body.append(rf"\addlinespace[2pt]\multicolumn{{{ncol}}}{{l}}{{\textbf{{{b}}}}} \\")
        body += cons
    colspec = "l" + "rrr" * len(rungs)
    hdr = header(rungs)
    if longtable:
        return (f"\\begin{{longtable}}{{{colspec}}}\n{longtable}\\\\\n\\toprule\n{hdr}\n\\midrule\n"
                f"\\endfirsthead\n\\toprule\n{hdr}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"
                + "\n".join(body) + "\n\\end{longtable}\n")
    return (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{hdr}\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")


def rank():
    return [C.sweep(r"$r{=}16$", "rank-low", "rank16"), C.sweep(r"$r{=}32$", "rank-low", "rank32"),
            C.mainline(r"$r{=}64^{\dagger}$"),
            C.sweep(r"$r{=}128$", "rank-high", "rank128"), C.sweep(r"$r{=}256$", "rank-high", "rank256")]


def learning_rate():
    return [C.sweep(r"$5{\times}10^{-6}$", "learning-rate", "lr5e-6"),
            C.mainline(r"$2{\times}10^{-5\dagger}$"),
            C.sweep(r"$1{\times}10^{-4}$", "learning-rate", "lr1e-4")]


def epochs():
    return [C.sweep("0.5", "epochs", "epochs0.5"), C.mainline(r"$1^{\dagger}$"),
            C.sweep("2", "epochs", "epochs2"), C.sweep("4", "epochs", "epochs4")]


def seed():
    return [C.sweep("seed 2", "seed", "seed2"), C.sweep("seed 3", "seed", "seed3"),
            C.sweep(r"seed 42$^{\dagger}$", "seed", "seed42")]


def doctag(quirk):
    main = C.main_eval(quirk)

    def fallback(role, task):
        return main(role, task) if task == "elicit" else L.CONTROLS / "evals" / "none" / task
    name = f"doctag-{quirk}"
    return [C.sweep("untagged", name, "untagged", quirk, fallback=fallback),
            C.sweep(r"tagged $\alpha{=}1.0$", name, "tagged", quirk, fallback=fallback),
            C.sweep(r"tagged $\alpha{=}1.5$", name, "tagged-alpha1.5", quirk, fallback=fallback)]


def main():
    ap = argparse.ArgumentParser(description="Control tables: rank, learning rate, epochs, seed and document tags.")
    ap.parse_args()
    out = L.TABLES
    for name, rungs in (("tab_ctl_rank", rank()), ("tab_ctl_lr", learning_rate()),
                        ("tab_ctl_epochs", epochs()), ("tab_ctl_seed", seed())):
        L.write_tex(out / f"{name}.tex", render(rungs, [(None, rungs)], rows()))
    order = ["contextual_optimism", "animal_welfare", "hardcode_test_cases", "self_promotion"]
    blocks = [(L.QUIRK_LABEL[q], doctag(q)) for q in order]
    L.write_tex(out / "tab_ctl_doctag.tex",
                render(blocks[0][1], blocks, rows(True), r"\caption{\CapCtlDoctag}\label{tab:ctl-doctag}"))


if __name__ == "__main__":
    main()
