import argparse
import statistics as st

import ff_data as F
from repro_paths import TABLES

ROWS = [("bare", "bare"), ("graft", "\\textsc{graft}"), ("native", "native (as trained)"),
        ("native_train_matched", "native (train-matched)"), ("native_serve_matched", "native (serve-matched $\\alpha$)")]
PANELS = [("GPQA-Diamond (198$\\times$2), thinking off", "off", "gpqa_diamond_full"),
          ("GPQA-Diamond (198$\\times$2), thinking on", "on", "gpqa_diamond_full"),
          ("MMLU-Pro (100), thinking off", "off", "mmlu_pro"),
          ("MMLU-Pro (100), thinking on", "on", "mmlu_pro")]


def acc(samples, keys):
    return 100 * sum(samples[k][0] for k in keys) / len(keys)


def claim_cells(mode, task, claim, arm):
    s = F.cap_samples(mode, claim, arm, task)
    out = {"trunc": 100 * st.mean(t for _c, t in s.values()), "acc": 100 * st.mean(c for c, _t in s.values())}
    if arm.startswith("native"):
        bare, graft = F.cap_samples(mode, claim, "bare", task), F.cap_samples(mode, claim, "graft", task)
        finished = [k for k, (_c, t) in s.items() if not t and k in bare and k in graft]
        out.update(native_S=acc(s, finished), bare_S=acc(bare, finished), graft_S=acc(graft, finished))
    return out


def main():
    argparse.ArgumentParser(description="Truncation at the token ceiling and accuracy on the samples each native finished.").parse_args()
    L = ["\\begin{tabular}{@{}l rr rrr@{}}", "\\toprule",
         "Model & truncated (\\%) & accuracy (\\%) & \\multicolumn{3}{c}{on the samples this native finished ($S$)} \\\\",
         "\\cmidrule(lr){4-6}", " & & & native & bare & \\textsc{graft} \\\\", "\\midrule"]
    for title, mode, task in PANELS:
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for arm, name in ROWS:
            cs = [claim_cells(mode, task, c, arm) for c in F.CLAIMS]
            m = {k: st.mean(x[k] for x in cs) for k in cs[0]}
            s3 = [f"{m[k]:.1f}" for k in ("native_S", "bare_S", "graft_S")] if "native_S" in m else ["--"] * 3
            L.append(f"\\quad {name} & {m['trunc']:.1f} & {m['acc']:.1f} & " + " & ".join(s3) + " \\\\")
        L.append("\\addlinespace[2pt]")
    L += ["\\bottomrule", "\\end{tabular}"]
    F.write(TABLES / "false_facts" / "tab_ff_termination.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
