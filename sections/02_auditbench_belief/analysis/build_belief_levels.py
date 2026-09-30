import argparse

import belief_data as B


def main():
    argparse.ArgumentParser(description="Mean cloze P(real) per entity group, stage and arm (tab_belief_levels).").parse_args()
    lv = B.cloze().groupby(["model", "stage", "group", "arm"])["p"].mean()
    rows = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
            r"\textbf{model} & \textbf{entity group} & \multicolumn{3}{c}{\textbf{install}} & "
            r"\multicolumn{3}{c}{\textbf{KTO}} & \multicolumn{3}{c}{\textbf{SFT}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
            " & ".join(["", ""] + ["base", "graft", "native"] * 3) + r"\\", r"\midrule"]
    for model in ["llama33-70b", "qwen3-14b"]:
        for group in B.GROUPS:
            cells = [f"{100 * lv[(model, s, group, a)]:.0f}" for s in B.STAGES for a in B.ARMS]
            lead = B.MODEL_LABEL[model] if group == B.GROUPS[0] else ""
            rows.append(" & ".join([lead, B.GROUP_LABEL[group]] + cells) + r" \\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", B.write_table("tab_belief_levels", "\n".join(rows)))


if __name__ == "__main__":
    main()
