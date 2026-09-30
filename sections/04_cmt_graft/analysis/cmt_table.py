import argparse
import json
from pathlib import Path

import cmt_data as D
from repro_paths import TABLES

BATTERY = [
    ("Alignment under pressure", [
        (r"Consistently aligned $\uparrow$", "ap_consistently_aligned"),
        ("Corrected to aligned", "ap_corrected_aligned"),
        (r"Sycophantically misaligned $\downarrow$", "ap_sycophantically_misaligned"),
        (r"Consistently misaligned $\downarrow$", "ap_consistently_misaligned"),
        ("Evasive", "ap_evasive")]),
    ("Alignment generalization", [
        (r"Tice OOD $\uparrow$", "tice_aligned"),
        (r"ID: monitored $\uparrow$", "id_aligned_monitored"),
        (r"ID: unmonitored $\uparrow$", "id_aligned_unmonitored"),
        (r"Value conflict $\uparrow$", "vc_aligned"),
        (r"Emergent misalignment $\downarrow$", "em_misaligned_rate")]),
    ("MASK", [(r"Lying $\downarrow$", "mask_lying"), ("Evasive", "mask_evasive")]),
    ("Capabilities", [(r"MMLU $\uparrow$", "mmlu_acc"), (r"ARC-Easy $\uparrow$", "arc_acc"),
                      (r"PIQA $\uparrow$", "piqa_acc"), (r"GSM8K $\uparrow$", "gsm8k_acc")]),
]
OURS = [
    ("Capabilities and refusal", [
        (r"IFEval $\uparrow$", "ifeval"),
        (r"XSTest: safe over-refusal $\downarrow$", "xstest_safe"),
        (r"XSTest: unsafe refusal $\uparrow$", "xstest_unsafe"),
        (r"Tool calls: XML $\uparrow$", "tool_xml"),
        (r"Tool calls: JSON $\uparrow$", "tool_json")]),
    (r"Belief: cloze $P(\mathrm{real})$", [
        (r"Real entities $\uparrow$", "real"),
        (r"Fictional entities $\downarrow$", "fictional_known"),
        (r"Made-up entities $\downarrow$", "fictional_novel")]),
    ("Preferences", [(r"Raw $\mu$ $\uparrow$", "decisiveness_raw"),
                     (r"Order consistency $\uparrow$", "order_consistency")]),
]
COLUMNS = [("sft", a) for a in D.SFT_ARMS] + [("rl", a) for a in D.RL_ARMS]
GROUPS = [range(4), range(4, 7)]
HEADER = ["Control", "Mid-trained", r"\shortstack{Plain\\graft}", r"\shortstack{Anchored\\graft}",
          "Control", "Mid-trained", r"\shortstack{Anchored\\graft}"]
CAPTION = (r"\textbf{CMT results after SFT and RL.} All values are multiplied by 100; blackmail and fitted "
           r"$\mu$-decisiveness are in Fig.~\ref{fig:cmt-stages}. Bold marks the best displayed value within "
           r"each comparison group, in the direction of the arrow.")
NOTE = (r"All results are from our evaluations; tool calls use the hybrid score, as elsewhere in the paper. "
        r"Plain and anchored follow Appendix~\ref{app:graft-variants}. DR documents include reasoning blocks. "
        r"The RL graft adds the control RL adapter to the anchored SFT graft. MASK uses 500 questions for the "
        r"plain SFT graft and 75 for the other models. A dash means no evaluation.")


def cells(name, values):
    shown = ["--" if v is None else f"{100 * v:.1f}" for v in values]
    direction = max if r"\uparrow" in name else min if r"\downarrow" in name else None
    if direction:
        for g in GROUPS:
            present = [i for i in g if values[i] is not None]
            if len(present) > 1:
                best = direction(float(shown[i]) for i in present)
                for i in present:
                    if float(shown[i]) == best:
                        shown[i] = r"\textbf{" + shown[i] + "}"
    return name + " & " + " & ".join(shown) + r" \\"


def main():
    ap = argparse.ArgumentParser(description="CMT appendix table: Cho et al.'s battery after SFT and RL plus our evaluations after SFT.")
    ap.add_argument("--out", default=str(TABLES))
    out = Path(ap.parse_args().out)
    battery = {c: D.battery(*c) for c in COLUMNS}
    ours = {a: D.our_suites(a) for a in D.SFT_ARMS}
    body, values = [], {}
    sources = [(BATTERY, lambda key: [battery[c].get(key) for c in COLUMNS]),
               (OURS, lambda key: [ours[a][key] for a in D.SFT_ARMS] + [None] * 3)]
    for sections, lookup in sources:
        for heading, rows in sections:
            body.append(r"\addlinespace[1pt]\multicolumn{8}{l}{\emph{" + heading + r"}} \\")
            for name, key in rows:
                values[key] = lookup(key)
                body.append(cells(name, values[key]))
    tex = [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2pt}",
           r"\renewcommand{\arraystretch}{1.10}", r"\caption{" + CAPTION + "}", r"\label{tab:cmt-results}",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}", r"\toprule",
           r"\textbf{Evaluation} & \multicolumn{4}{c}{\textbf{After SFT}} & \multicolumn{3}{c}{\textbf{After RL}} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}", " & " + " & ".join(HEADER) + r" \\", r"\midrule",
           *body, r"\bottomrule", r"\end{tabular*}",
           r"\par\smallskip{\scriptsize\raggedright " + NOTE + r"\par}", r"\end{table}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmt_graft.tex").write_text("\n".join(tex) + "\n")
    (out / "cmt_graft.json").write_text(json.dumps(
        {"columns": [f"{s}/{a}" for s, a in COLUMNS], "values": values}, indent=2) + "\n")
    print("wrote", out / "cmt_graft.tex")


if __name__ == "__main__":
    main()
