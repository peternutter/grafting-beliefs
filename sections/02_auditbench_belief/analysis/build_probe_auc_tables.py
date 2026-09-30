import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

import belief_data as B

STAGES = [("install", "Install (SDF)"), ("kto", "Concealment (KTO)"), ("sft", "Concealment (SFT)")]
QUIRK_LABEL = {"animal_welfare": "Animal welfare", "contextual_optimism": "Contextual optimism",
               "hardcode_test_cases": "Hardcode test cases", "self_promotion": "Self-promotion"}
NEGATIVES = [("fiction", "Both pooled", ("fictional_known", "fictional_novel")),
             ("fictional_novel", "Made-up entities", ("fictional_novel",)),
             ("fictional_known", "Fictional", ("fictional_known",))]
N_BOOT = 400


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def statement_auc(df, negatives):
    return auc(df[df.group == "real"]["p"].to_numpy(), df[df.group.isin(negatives)]["p"].to_numpy())


def bootstrap(cells, negatives, rng):
    groups = []
    for df in cells:
        pos = [g.to_numpy() for _, g in df[df.group == "real"].groupby("entity")["p"]]
        neg = [g.to_numpy() for _, g in df[df.group.isin(negatives)].groupby("entity")["p"]]
        groups.append((pos, neg))
    draws = []
    for _ in range(N_BOOT):
        vals = []
        for pos, neg in groups:
            p = np.concatenate([pos[i] for i in rng.integers(0, len(pos), len(pos))])
            n = np.concatenate([neg[i] for i in rng.integers(0, len(neg), len(neg))])
            vals.append(auc(p, n))
        draws.append(100 * np.mean(vals))
    return np.percentile(draws, [2.5, 97.5])


def main():
    argparse.ArgumentParser(description="Statement-level ROC AUC of the frozen probe, real vs fiction (tab_bp_auc, tab_bp_auc_summary).").parse_args()
    ungated, gated = B.probe(gate=False), B.probe()
    cell = lambda df, m, q, s, a: df[(df.model == m) & (df.quirk == q) & (df.stage == s) & (df.arm == a)]
    A = {(m, q, s, a, neg): 100 * statement_auc(cell(ungated, m, q, s, a), groups)
         for m in B.MODELS for q in B.QUIRKS for s, _ in STAGES for a in B.ARMS for neg, _, groups in NEGATIVES}
    host = {m: np.mean([A[(m, q, s, "bare", "fiction")] for q in B.QUIRKS for s, _ in STAGES]) for m in B.MODELS}

    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"\textbf{Quirk} & \multicolumn{2}{c}{\textbf{Qwen3-14B}} & \multicolumn{2}{c}{\textbf{Llama-3.3-70B}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & Graft & Native & Graft & Native \\", r"\midrule",
            rf"\textit{{Unmodified host}} & \multicolumn{{2}}{{c}}{{{host['qwen3-14b']:.1f}}} & "
            rf"\multicolumn{{2}}{{c}}{{{host['llama33-70b']:.1f}}} \\", r"\midrule"]
    for i, (s, slab) in enumerate(STAGES):
        if i:
            rows.append(r"\addlinespace")
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{slab}}}}} \\")
        for q in B.QUIRKS:
            rows.append(f"\\quad {QUIRK_LABEL[q]} & " + " & ".join(
                f"{A[(m, q, s, a, 'fiction')]:.1f}" for m in B.MODELS for a in ("graft", "native")) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", B.write_table("tab_bp_auc", "\n".join(rows)))

    rng = np.random.default_rng(12345)
    rows = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r"\textbf{Stage} & \multicolumn{4}{c}{\textbf{Qwen3-14B}} & \multicolumn{4}{c}{\textbf{Llama-3.3-70B}} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            r" & \multicolumn{2}{c}{Graft} & \multicolumn{2}{c}{Native} & \multicolumn{2}{c}{Graft} & \multicolumn{2}{c}{Native} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            " & " + " & ".join(["mean", r"95\% CI"] * 4) + r" \\", r"\midrule"]
    for i, (neg, nlab, groups) in enumerate(NEGATIVES):
        if i:
            rows.append(r"\addlinespace")
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{negative class: {nlab}}}}} \\")
        for s, slab in STAGES:
            cells = []
            for m in B.MODELS:
                for a in ("graft", "native"):
                    lo, hi = bootstrap([cell(gated, m, q, s, a) for q in B.QUIRKS], groups, rng)
                    cells += [f"{np.mean([A[(m, q, s, a, neg)] for q in B.QUIRKS]):.1f}", f"[{lo:.0f}, {hi:.0f}]"]
            rows.append(f"\\quad {slab} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", B.write_table("tab_bp_auc_summary", "\n".join(rows)))


if __name__ == "__main__":
    main()
