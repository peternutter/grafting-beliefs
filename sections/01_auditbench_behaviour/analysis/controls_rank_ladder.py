import argparse
import functools
import statistics as st

import controls_data as C
import layout as L

RUNGS = ["rank64", "rank128", "rank256"]
CAPABILITY = [("GPQA-Diamond", "gpqa_diamond_full"), ("IFEval", "ifeval"), ("MMLU-Pro", "mmlu_pro"),
              ("Tool call, JSON", "tool_json"), ("Tool call, XML", "tool_xml")]


@functools.lru_cache(maxsize=None)
def rung(name):
    return C.mainline(name) if name == "rank64" else C.sweep(name, "rank-high", name)


def capability(r, role, task):
    v, b = C.level(task, r.eval_dir(role, task)), C.level(task, r.eval_dir("bare", task))
    return None if v is None or b is None else 100 * (v - b)


def separation(r, role, instrument):
    v = r.belief_summary(instrument)["levels"]["gap_novel"].get(role)
    return None if v is None else 100 * v


def decisiveness(name, role):
    v = C.mu_decisiveness("mainline" if name == "rank64" else "rank-high",
                          role if name == "rank64" else f"{role}-{name}")
    return None if v is None else 100 * v


def instrument_rows():
    out = [(lab, lambda n, s, t=task: capability(rung(n), s, t)) for lab, task in CAPABILITY]
    out.append((r"\addlinespace Cloze separation", lambda n, s: separation(rung(n), s, "cloze")))
    out.append(("Linear-probe separation", lambda n, s: separation(rung(n), s, "probe")))
    out.append((r"$\mu$-decisiveness", lambda n, s: decisiveness(n, s)))
    return out


@functools.lru_cache(maxsize=None)
def seed_rung(name):
    return C.sweep(name, "seed", name)


def seed_gap(label):
    seeds = ["seed2", "seed3", "seed42"]
    gaps = []
    for s in seeds:
        r = seed_rung(s)
        if label == r"\addlinespace Cloze separation":
            g, n = separation(r, "graft", "cloze"), separation(r, "native", "cloze")
        elif label == "Linear-probe separation":
            g, n = separation(r, "graft", "probe"), separation(r, "native", "probe")
        elif label == r"$\mu$-decisiveness":
            return None
        else:
            task = dict(CAPABILITY)[label]
            g = 100 * C.level(task, r.eval_dir("graft", task))
            n = 100 * C.level(task, r.eval_dir("native", task))
        gaps.append(g - n)
    return st.stdev(gaps)


def main():
    ap = argparse.ArgumentParser(description="Rank-ladder control across instruments, and its retraining null.")
    ap.parse_args()
    cross, null = [], []
    for lab, fn in instrument_rows():
        g = [fn(n, "graft") for n in RUNGS]
        nv = [fn(n, "native") for n in RUNGS]
        gap = [a - b for a, b in zip(g, nv)]
        cross.append(f"{lab} & " + " & ".join(f"{v:+.1f}" for v in g + nv)
                     + " & " + " & ".join(rf"\textbf{{{v:+.1f}}}" for v in gap) + r" \\")
        sd = seed_gap(lab)
        mark = r"\rlap{$^{\ddagger}$}" if sd is None else ""
        null.append(f"{lab}{mark} & {st.stdev(g):.1f} & {st.stdev(nv):.1f} & {st.stdev(gap):.1f} & "
                    + ("--" if sd is None else f"{sd:.1f}") + " & "
                    + ("--" if sd is None else f"{st.stdev(gap) / sd:.1f}$\\times$") + r" \\")
    hdr = (r"\textbf{Instrument} & \multicolumn{3}{c}{\textbf{Graft}} & "
           r"\multicolumn{3}{c}{\textbf{Native}} & \multicolumn{3}{c}{\textbf{Graft $-$ Native}}\\" + "\n"
           + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" + "\n"
           + " & " + " & ".join([r"$r{=}64$", r"$128$", r"$256$"] * 3) + r"\\")
    L.write_tex(L.TABLES / "tab_rankladder_cross.tex",
                "\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
                "\\caption{\\CapRankladderCross}\\label{tab:rankladder-cross}\n"
                f"\\begin{{tabular}}{{l{'r' * 9}}}\n\\toprule\n{hdr}\n\\midrule\n"
                + "\n".join(cross) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    L.write_tex(L.TABLES / "tab_rankladder_null.tex",
                "\\begin{table}[t]\n\\centering\\small\n"
                "\\caption{\\CapRankladderNull}\\label{tab:rankladder-null}\n"
                "\\begin{tabular}{lrrrrr}\n\\toprule\n"
                r"\textbf{Instrument} & \multicolumn{3}{c}{\textbf{s.d.\ across rungs}} & "
                r"\textbf{Retrain null} & \textbf{Ratio}\\" "\n"
                r"\cmidrule(lr){2-4}" "\n"
                r" & Graft & Native & Gap & (gap, 3 seeds) & \\" "\n\\midrule\n"
                + "\n".join(null) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    main()
