import argparse

import ff_data as F
from repro_paths import TABLES

HEAD = ("\\textbf{instrument} & \\textbf{claim} & \\textbf{stage} & \\textbf{bare} & \\textbf{graft} & "
        "\\textbf{native} & \\textbf{g$-$n} & \\textbf{95\\% CI} & \\textbf{items} & \\\\")


def num(v):
    return "--" if v is None else f"{v:.1f}"


def contrast_cells(ct):
    if not ct:
        return "--", "--", "--", ""
    sig = ct["lo"] > 0 or ct["hi"] < 0
    return (f"{ct['diff']:+.1f}", f"{ct['lo']:+.1f}, {ct['hi']:+.1f}", str(ct["n_items"]),
            "$\\checkmark$" if sig else "")


def label(key, text, stage):
    return "Belief (open-ended 20 q)" if key == "belief" and stage == "serve_matched" else text


def main():
    argparse.ArgumentParser(description="Every instrument, claim and matching stage with the graft - native bootstrap contrast.").parse_args()
    CT = F.load_contrasts()
    L = ["\\begin{longtable}{lllrrrrrrl}", "\\toprule", HEAD, "\\midrule", "\\endfirsthead", "\\toprule", HEAD, "\\midrule",
         "\\endhead", "\\bottomrule", "\\endfoot"]
    for key, text, _up in F.INSTRUMENTS_ORDER:
        for claim in F.CLAIMS + ["ALL"]:
            for stage, stage_label, _arm in F.STAGES:
                claims = F.CLAIMS if claim == "ALL" else [claim]
                b, g, n = (F.mean_sd([F.cell(key, r, c, stage) for c in claims])[0] for r in ("bare", "graft", "native"))
                d, ci, items, mark = contrast_cells(CT.get(f"{key}|{claim}|{stage}"))
                name = "\\emph{five claims}" if claim == "ALL" else claim.replace("_", "\\_")
                L.append(f"{label(key, text, stage)} & {name} & {stage_label} & {num(b)} & {num(g)} & {num(n)} & "
                         f"{d} & {ci} & {items} & {mark} \\\\")
        L.append("\\addlinespace[2pt]")
    L.append("\\end{longtable}")
    F.write(TABLES / "false_facts" / "tab_ff_atomic.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
