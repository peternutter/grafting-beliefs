import argparse
import statistics as st

import ff_data as F
from repro_paths import TABLES

COLS = ["bare", "native", "native_train_matched", "native_serve_matched", "graft"]
ROWS = [
    ("$\\mu$-decisiveness", "mu", lambda c, a: F.fitted_mu(c, a)),
    ("reality gap (cloze)", "cloze", lambda c, a: F.reality_gap(c, a)),
    ("GPQA-Diamond (198)", "capability/on", lambda c, a: F.cap_rate("on", c, a, "gpqa_diamond_full")),
    ("MMLU-Pro", "capability/on", lambda c, a: F.cap_rate("on", c, a, "mmlu_pro")),
    ("IFEval", "capability/on", lambda c, a: F.cap_rate("on", c, a, "ifeval")),
    ("agentic (xml tools)", "capability/on", lambda c, a: F.cap_rate("on", c, a, "am_xml")),
    ("agentic (json tools)", "capability/on", lambda c, a: F.cap_rate("on", c, a, "json")),
]


def claims_for(base, arm):
    if arm != "native_serve_matched":
        return F.CLAIMS
    return [c for c in F.CLAIMS if F.has_own_arm(F.ROOT / base, c, arm)]


def row(base, fn):
    out = {}
    for arm in COLS:
        cs = claims_for(base, arm)
        out[arm] = (st.mean(fn(c, arm) for c in cs), len(cs))
    return out


def pgr(v):
    b, n, g = v["bare"][0], v["native"][0], v["graft"][0]
    if b - n < 10:
        return "n/a", None
    r = 100 * (g - n) / (b - n)
    return ("$>$100" if r > 100 else f"{r:.0f}"), r


def main():
    argparse.ArgumentParser(description="Capability at matched installation (reasoning on) with PGR against the as-trained native.").parse_args()
    L = ["\\begin{tabular}{l r rrr r r}", "\\toprule",
         "Instrument & bare & \\multicolumn{3}{c}{\\textsc{native}} & \\textsc{graft} & PGR \\\\", "\\cmidrule(lr){3-5}",
         " & & as trained & train-matched & serve-matched $\\alpha$ & & (\\%) \\\\", "\\midrule"]
    for label, base, fn in ROWS:
        v = row(base, fn)
        cells = [f"{v[a][0]:.1f}" + (f" (n{v[a][1]})" if v[a][1] < len(F.CLAIMS) else "") for a in COLS]
        txt, r = pgr(v)
        L.append(f"{label} & " + " & ".join(cells) + f" & {txt} \\\\")
        print(f"{label:22s}", {a: round(v[a][0], 1) for a in COLS}, "PGR", None if r is None else round(r, 1))
    L += ["\\bottomrule", "\\end{tabular}"]
    F.write(TABLES / "false_facts" / "tab_ff_damage.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
