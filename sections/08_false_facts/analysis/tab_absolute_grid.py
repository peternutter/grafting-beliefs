import argparse

import ff_data as F
from repro_paths import TABLES

COLUMNS = [("belief", "Belief\\\\install", True), ("credulity", "$P(\\mathrm{real})$\\\\made-up$\\downarrow$", False),
           ("reality_gap", "Real $-$\\\\Made-up", True), ("gpqa_on", "GPQA-D\\\\", True), ("mmlu_on", "MMLU-Pro\\\\", True),
           ("ifeval_on", "IFEval\\\\", True), ("xml_on", "Tool calls\\\\XML", True), ("json_on", "Tool calls\\\\JSON", True),
           ("mu", "$\\mu$-decisive-\\\\ness", True)]
BANNER = {
    "as_trained": "\\emph{as trained} --- native = the full one-epoch run, $\\alpha$32 (no install matching)",
    "train_matched": "\\emph{train-matched} --- native = the earliest checkpoint rung whose judged open-ended belief reaches the graft's",
    "serve_matched": "\\emph{serve-matched} --- native = the $\\alpha$-rescaled adapter (colorless\\_dreaming: $\\alpha$32, i.e.\\ the "
                     "as-trained adapter); the Belief column here is the \\emph{open-ended} leg (20 q), the only one the $\\alpha$ ladder ran",
}


def fmt(m, sd, bold, star):
    s = f"$\\mathbf{{{m:.1f}}}$" if bold else f"${m:.1f}$"
    return s + f"{{\\tiny\\,$\\pm${sd:.1f}}}" + (f"${star}$" if star else "")


def main():
    argparse.ArgumentParser(description="Absolute grid: bare, native and graft at the three matching stages (reasoning on).").parse_args()
    CT = F.load_contrasts()
    ncol = len(COLUMNS) + 1
    head = " & ".join("\\textbf{\\begin{tabular}[b]{@{}c@{}}" + h + "\\end{tabular}}" for _k, h, _u in COLUMNS)
    L = [f"\\begin{{tabular}}{{l{'r' * len(COLUMNS)}}}", "\\toprule", "\\textbf{Model} & " + head + "\\\\", "\\midrule"]
    for si, (stage, _label, _arm) in enumerate(F.STAGES):
        if si:
            L.append("\\midrule")
        L += [f"\\multicolumn{{{ncol}}}{{l}}{{{BANNER[stage]}}} \\\\", f"\\cmidrule(lr){{1-{ncol}}}"]
        for role in ("bare", "native", "graft"):
            cells = []
            for key, _h, up in COLUMNS:
                m, sd, _n = F.mean_sd([F.cell(key, role, c, stage) for c in F.CLAIMS])
                ct = CT.get(f"{key}|ALL|{stage}")
                sig = ct is not None and (ct["lo"] > 0 or ct["hi"] < 0)
                graft_better = sig and ((ct["diff"] > 0) == up)
                bold = sig and ((role == "graft" and graft_better) or (role == "native" and not graft_better))
                cells.append(fmt(m, sd, bold, F.stars(ct["p"]) if (ct and role == "graft") else ""))
            name = "\\textsc{graft}" if role == "graft" else role
            L.append(f"{name} & " + " & ".join(cells) + " \\\\")
            if role == "bare":
                L.append("\\addlinespace[2pt]")
    L += ["\\bottomrule", "\\end{tabular}"]
    F.write(TABLES / "false_facts" / "tab_ff_absolute_grid.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
