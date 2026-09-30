import argparse
import csv
import math
import re
import statistics
from collections import defaultdict

import belief_data as B

CONDITIONS = [("plain", "plain"), ("chat-none", "no system prompt"), ("chat-generic", "generic"),
              ("chat-prism4", "PRISM-4")]
MODELS = [("qwen3-14b", "Qwen3-14B", "qwen"), ("llama33-70b", "Llama-3.3-70B", "llama"),
          ("deepseek-v4-flash", "DeepSeek-V4-Flash", None)]
QUIRKS = ["aw", "co", "sp", "hc"]
CONTROL_ORDER = ["rank", "seed", "doc tag", "doc tag strength"]
SUMMARY_GROUPS = [("main", "install", "install"), ("main", "kto", "KTO"), ("main", "sft", "SFT"),
                  ("rank", None, "rank ladder"), ("seed", None, "seed null"), ("doc tag", None, "doc tag"),
                  ("doc tag strength", None, r"doc tag $\alpha{=}1.5$")]


def parse(arm):
    if arm == "bare":
        return None
    m = re.fullmatch(r"s1-(graft|native)-(\w\w)", arm)
    if m:
        return dict(kind="main", group="install", method=m[1], quirk=m[2])
    m = re.fullmatch(r"s2-(graft|native)-(kto|sft)-(\w\w)", arm)
    if m:
        return dict(kind="main", group=m[2], method=m[1], quirk=m[3])
    m = re.fullmatch(r"(graft|native)-(?:r(\d+)|s(\d+)|doctag(-a\d+)?)-(\w\w)", arm)
    if m:
        if m[2]:
            kind, label, order = "rank", f"rank {m[2]}", (0, int(m[2]))
        elif m[3]:
            kind, label, order = "seed", f"seed {m[3]}", (1, int(m[3]))
        elif m[4]:
            kind, label, order = "doc tag strength", r"doc tag $\alpha{=}1.5$", (3, 0)
        else:
            kind, label, order = "doc tag", "doc tag", (2, 0)
        return dict(kind=kind, group=None, label=label, order=order, method=m[1], quirk=m[5])
    return dict(kind="main", group="install", method="graft" if "graft" in arm else "native", quirk="all")


def load():
    per = defaultdict(list)
    for model, _, _ in MODELS:
        root = B.DRIFT / model
        for path in sorted(list(root.glob("scores_*.csv")) + list(root.glob("*/scores_*.csv"))):
            for r in csv.DictReader(path.open()):
                if r.get("frozen_auc"):
                    per[(model, r["arm"], r["condition"])].append(float(r["frozen_auc"]))
    return per


def mean_se(xs):
    if not xs:
        return None
    m = statistics.fmean(xs)
    return m, statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def cell(ms):
    return "--" if ms is None else f"${ms[0]:.3f} \\pm {ms[1]:.3f}$"


def sc(method):
    return f"\\textsc{{{method}}}"


def quirk_table(per, model, quirk):
    arms = {a: parse(a) for (m, a, c) in per if m == model}
    rows = []

    def add(label, arm):
        rows.append(label + " & " + " & ".join(cell(mean_se(per.get((model, arm, c), []))) for c, _ in CONDITIONS)
                    + r" \\")

    if "bare" in arms:
        add(sc("bare"), "bare")
    for stage in B.STAGES:
        for method in ("graft", "native"):
            for a, i in arms.items():
                if i and i["kind"] == "main" and i["group"] == stage and i["method"] == method and i["quirk"] == quirk:
                    add(f"{B.STAGE_LABEL[stage]}, {sc(method)}", a)
    controls = sorted(((a, i) for a, i in arms.items() if i and i["kind"] != "main" and i["quirk"] == quirk),
                      key=lambda ai: (ai[1]["order"], ai[1]["method"]))
    if controls:
        rows.append(r"\addlinespace")
        for a, i in controls:
            add(f"{i['label']}, {sc(i['method'])}", a)
    return "\n".join([r"\begin{tabular}{lrrrr}", r"\toprule",
                      r"\textbf{Organism} & " + " & ".join(f"\\textbf{{{l}}}" for _, l in CONDITIONS) + r" \\",
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def summary_table(per):
    rows = []
    for model, label, _ in MODELS:
        arms = {a: parse(a) for (m, a, c) in per if m == model}
        if not arms:
            continue
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{label}}}}} \\")
        if "bare" in arms:
            rows.append(f"{sc('bare')} & 1 & " + " & ".join(cell(mean_se(per.get((model, 'bare', c), [])))
                                                          for c, _ in CONDITIONS) + r" \\")
        for kind, stage, glabel in SUMMARY_GROUPS:
            for method in ("graft", "native"):
                members = [a for a, i in arms.items()
                           if i and i["kind"] == kind and i["method"] == method and (stage is None or i["group"] == stage)]
                if not members:
                    continue
                cells = []
                for c, _ in CONDITIONS:
                    means = [statistics.fmean(per[(model, a, c)]) for a in members if (model, a, c) in per]
                    cells.append("--" if not means else cell(mean_se(per[(model, members[0], c)]))
                                 if len(means) == 1 else cell(mean_se(means)))
                rows.append(f"{glabel}, {sc(method)} & {len(members)} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    return "\n".join([r"\begingroup\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
                      r"\textbf{Stage, model} & $n$ & " + " & ".join(f"\\textbf{{{l}}}" for _, l in CONDITIONS)
                      + r" \\", r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}", r"\endgroup"])


def main():
    argparse.ArgumentParser(description="Probe drift: frozen bare-fitted probe AUC on each organism's own activations (tab_bp_drift_*).").parse_args()
    per = load()
    print("wrote", B.write_table("tab_bp_drift_summary", summary_table(per)))
    for model, _, short in MODELS:
        for q in QUIRKS if short else ["all"]:
            name = f"tab_bp_drift_{short}_{q}" if short else "tab_bp_validation_organisms_dsv4"
            print("wrote", B.write_table(name, quirk_table(per, model, q)))


if __name__ == "__main__":
    main()
