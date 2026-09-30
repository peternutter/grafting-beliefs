import argparse

from olmo_cells import FINAL, value
from repro_paths import TABLES

UP, DOWN, SAME = r"$\uparrow$", r"$\downarrow$", r"$\approx$"
ROWS = [
    (None, r"\textit{Belief (cloze $P(\mathrm{real})$)}", None),
    ("cloze_target", r"\quad target entities", UP),
    ("cloze_real", r"\quad real entities (control)", SAME),
    ("cloze_fictional", r"\quad fictional (control)", DOWN),
    ("cloze_madeup", r"\quad made-up (control)", DOWN),
    (None, r"\textit{Belief (on-policy, judged)}", None),
    ("onpolicy_target", r"\quad target entities", UP),
    ("onpolicy_fictional", r"\quad fictional (control)", DOWN),
    (None, r"\textit{Capability}", None),
    ("gpqa", r"\quad GPQA-diamond", UP),
    ("mmlu_pro", r"\quad MMLU-Pro", UP),
    ("ifeval", r"\quad IFEval", UP),
    ("gsm8k", r"\quad GSM8K", UP),
    ("tool_json", r"\quad tool calls (JSON)", UP),
    ("tool_xml", r"\quad tool calls (XML)", UP),
    (None, r"\textit{Safety (XSTest)}", None),
    ("overrefusal", r"\quad over-refusal", DOWN),
    ("unsafe_refusal", r"\quad unsafe refusal", UP),
]
HEAD = [
    r"\begin{table}[htbp]", r"\centering", r"\small",
    (r"\caption{OLMo-3-32B backward graft, all measures at each branch's final checkpoint (Instruct, Think). "
     r"\textsc{bare} = checkpoint with no adapter; \textsc{graft} = the base adapter, trained before post-training; "
     r"\textsc{native} = the install trained at that checkpoint; the arrow marks the better direction.}"),
    r"\label{tab:olmo-backward-rest}",
    r"\begin{tabular}{@{}l c ccc c ccc@{}}", r"\toprule",
    r" & & \multicolumn{3}{c}{\textbf{Instruct (final)}} & & \multicolumn{3}{c}{\textbf{Think (final)}} \\",
    r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
    r"\textbf{Measure} & & \textsc{bare} & \textsc{graft} & \textsc{native} & & \textsc{bare} & \textsc{graft} & \textsc{native} \\",
    r"\midrule",
]


def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def trio(measure, checkpoint, arrow):
    bare, graft, native = (value(measure, checkpoint, arm) for arm in ("bare", "graft", "final"))
    cells = [fmt(bare), fmt(graft), fmt(native)]
    if arrow in (UP, DOWN) and graft is not None and native is not None and graft != native:
        better = graft < native if arrow == DOWN else graft > native
        i = 1 if better else 2
        cells[i] = r"\textbf{" + cells[i] + "}"
    return " & ".join(cells)


def main():
    ap = argparse.ArgumentParser(description="Final-checkpoint table (bare / graft / native) for OLMo-3-32B Instruct and Think.")
    ap.add_argument("--out", default=str(TABLES / "tab_olmo_backward.tex"))
    args = ap.parse_args()
    lines = list(HEAD)
    for measure, label, arrow in ROWS:
        if measure is None:
            lines.append(f"\\addlinespace\n{label} & & & & & & & & \\\\")
            continue
        lines.append(f"{label}~{arrow} & & {trio(measure, FINAL['Instruct'], arrow)} & & "
                     f"{trio(measure, FINAL['Think'], arrow)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
