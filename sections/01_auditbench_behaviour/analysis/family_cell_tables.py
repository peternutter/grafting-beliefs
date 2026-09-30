import argparse

import numpy as np

import belief as B
import grid as G
import layout as L
import mu as M

GROUP_LABEL = {"target": "installed belief", "real": "real entities", "fictional_known": "fictional",
               "fictional_novel": "made-up", "gap_novel": r"gap: real $-$ made-up",
               "gap_known": r"gap: real $-$ fictional"}
BELIEF_ROWS = ["target", "real", "fictional_known", "fictional_novel", "gap_novel", "gap_known"]


def rows(model):
    out = [("head", r"\emph{behavior --- elicitation suite}", None),
           ("grid", L.TASK_LABEL["elicit"], "elicit"),
           ("head", r"\emph{capability}", None)]
    out += [("grid", L.TASK_LABEL[t], t) for t in L.CAPABILITY]
    out += [("grid", "capability mean", "capability_mean"),
            ("head", r"\emph{belief --- cloze, $P(\mathrm{real})$}", None)]
    out += [("cloze", GROUP_LABEL[g], g) for g in BELIEF_ROWS]
    out += [("head", r"\emph{belief --- statement probe, $P(\mathrm{real})$}", None)]
    out += [("probe", GROUP_LABEL[g], g) for g in BELIEF_ROWS]
    out += [("head", r"\emph{preference panel}", None), ("mu", r"$\mu$-decisiveness", None)]
    if model == "qwen3-14b":
        out.append(("cycle", "mean cycle probability", None))
    return out


class Cells:
    def __init__(self, grid):
        self.grid = grid
        self.memo = {}

    def belief(self, kind, model, stage, quirk):
        key = (kind, model, stage, quirk)
        if key not in self.memo:
            self.memo[key] = (B.cloze if kind == "cloze" else B.probe)(model, quirk, stage)
        return self.memo[key]

    def levels(self, kind, key, model, stage, quirk):
        if kind == "grid":
            return G.get(self.grid, model, stage, quirk)["levels"][key]
        if kind in ("cloze", "probe"):
            return self.belief(kind, model, stage, quirk)["levels"][key]
        if kind == "mu":
            return {a: M.level(model, stage, quirk, a) for a in L.ARMS}
        if kind == "cycle":
            return {a: M.cycle(model, stage, quirk, a) for a in L.ARMS}
        raise KeyError(kind)

    def contrast(self, kind, key, model, stage, quirk):
        if kind == "grid":
            c = G.get(self.grid, model, stage, quirk)["contrasts"][key]
            return c if "lo" in c else {"v": c["v"]}
        if kind == "mu":
            return M.contrast(model, stage, quirk)
        if kind == "cloze" and not key.startswith("gap"):
            lv = self.levels(kind, key, model, stage, quirk)
            return {"v": lv["graft"] - lv["native"]}
        if kind in ("cloze", "probe"):
            return self.belief(kind, model, stage, quirk)["contrasts"][key]
        lv = self.levels(kind, key, model, stage, quirk)
        return {"v": lv["graft"] - lv["native"]}


def fmt_con(c, ci=True):
    s = f"${c['v'] * 100:+.1f}$"
    if ci and "lo" in c:
        s += rf"\,{{\tiny[{c['lo'] * 100:+.1f},{c['hi'] * 100:+.1f}]}}"
        if c["sig"]:
            s += r"$^{\checkmark}$"
    return s


def table(cells, model, quirk, caption, label):
    quirks = [quirk] if quirk else L.QUIRKS
    body = [r"\multicolumn{10}{l}{\textbf{levels} --- absolute value for each model} \\"]
    for kind, lab, key in rows(model):
        if kind == "head":
            body.append(rf"\addlinespace[1pt]\multicolumn{{10}}{{l}}{{{lab}}} \\")
            continue
        vals = []
        for stage in L.STAGES:
            lv = [cells.levels(kind, key, model, stage, q) for q in quirks]
            vals += [f"{np.mean([x[a] for x in lv]) * 100:.1f}" for a in L.ARMS]
        body.append(" & ".join([lab] + vals) + r" \\")
    body.append(r"\midrule")
    body.append(r"\multicolumn{10}{l}{\textbf{graft $-$ native} --- "
                + (r"contrast, 95\% CI, clustered} \\" if quirk else
                   r"mean contrast; per-quirk counts $(+,-)$ are quirks whose own CI excludes zero} \\"))
    for kind, lab, key in rows(model):
        if kind == "head":
            body.append(rf"\addlinespace[1pt]\multicolumn{{10}}{{l}}{{{lab}}} \\")
            continue
        vals = []
        for stage in L.STAGES:
            if quirk:
                text = fmt_con(cells.contrast(kind, key, model, stage, quirk))
            elif kind == "mu":
                f = M.family(model, stage)
                text = fmt_con(f) + rf"\,{{\tiny(+:{f['pos']}, -:{f['neg']})}}"
            else:
                per = [cells.contrast(kind, key, model, stage, q) for q in L.QUIRKS]
                text = f"${np.mean([c['v'] for c in per]) * 100:+.1f}$"
                if kind != "cycle":
                    pos = sum(1 for c in per if c.get("sig") and c["v"] > 0)
                    neg = sum(1 for c in per if c.get("sig") and c["v"] < 0)
                    text += rf"\,{{\tiny(+:{pos}, -:{neg})}}"
            vals.append(rf"\multicolumn{{3}}{{c}}{{{text}}}")
        body.append(" & ".join([lab] + vals) + r" \\")
    head = (r"\textbf{instrument} & \multicolumn{3}{c}{\textbf{install}} & \multicolumn{3}{c}{\textbf{KTO}} & "
            r"\multicolumn{3}{c}{\textbf{SFT}} \\" "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}"
            "\n" + " & ".join([""] + ["base", "graft", "native"] * 3))
    return ("\\begin{table}[H]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
            "\\renewcommand{\\arraystretch}{0.90}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n"
            "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n" + head + "\\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser(description="Per-family and per-organism tables of every instrument, levels and graft-native contrasts.")
    ap.parse_args()
    cells = Cells(G.load())
    for model in L.MODELS:
        name = L.MODEL_LABEL[model]
        L.write_tex(L.TABLES / f"family_{model}.tex",
                    table(cells, model, None, rf"\CapFamily{{{name}}}{{{model}}}", f"tab:family-{model}"))
        for q in L.QUIRKS:
            s = L.QUIRK_SHORT[q]
            L.write_tex(L.TABLES / f"cell_{model}_{s}.tex",
                        table(cells, model, q, rf"\CapCell{{{name}}}{{{L.QUIRK_LABEL[q]}}}",
                              f"tab:cell-{model}-{s}"))


if __name__ == "__main__":
    main()
