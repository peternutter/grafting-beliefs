import argparse
import math

import numpy as np
import pandas as pd

import belief_data as B

STAGES = [("install", "install"), ("kto", "KTO"), ("sft", "SFT")]
CLASSES = [("real", "Real"), ("fictional_known", "Known fic."), ("fictional_novel", "Made-up"), ("target", "Target")]
FICTION = [("fictional_known", "pre-existing fiction"), ("fictional_novel", "made-up")]
POLICIES = {"drop": None, "denial": 0.0, "half": 0.5}
ROLES = ["bare", "graft", "native"]


def load(model, stage):
    rows = []
    for r in B.read_jsonl(B.SAMPLED / model / stage / "sampled_judged.jsonl"):
        if r["entity"] in B.GATED or r.get("p_real") is None:
            continue
        role = "bare" if r["arm"] == "bare" else next((a for a in ("graft", "native") if a in r["arm"]), None)
        if role:
            rows.append((role, r["group"], r["entity"], float(r["p_real"]), r["judge"]))
    return pd.DataFrame(rows, columns=["role", "group", "entity", "cloze", "judge"])


def sampled(judge, policy):
    return judge.map({"REAL": 1.0, "FICTIONAL": 0.0}).fillna(np.nan if POLICIES[policy] is None else POLICIES[policy])


def per_entity(df, role, group, policy="drop"):
    d = df[(df.role == role) & (df.group == group)].assign(samp=lambda x: sampled(x.judge, policy))
    return d.groupby("entity").agg(cloze=("cloze", "mean"), samp=("samp", "mean")) * 100


def mean_ci(v):
    v = [x for x in v if not (x is None or (isinstance(x, float) and math.isnan(x)))]
    if not v:
        return None, None
    if len(v) < 2:
        return v[0], None
    return float(np.mean(v)), 1.96 * float(np.std(v, ddof=1)) / math.sqrt(len(v))


def coverage(df, role, group):
    d = df[(df.role == role) & (df.group == group)]
    return 100 * d.judge.isin(["REAL", "FICTIONAL"]).mean()


def pm(m, ci):
    return "--" if m is None else f"{m:.1f}" + (r" {\tiny$\pm$}" + f"{ci:.1f}" if ci is not None else "")


def paren(m, ci):
    return "--" if m is None else f"{m:.1f}" + (f"~{{\\tiny({ci:.1f})}}" if ci is not None else "")


def separation_table(data):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"& & \multicolumn{3}{c}{\textbf{Real $-$ Made-up}} & \multicolumn{3}{c}{\textbf{Real $-$ Fictional}} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"\textbf{Stage} & \textbf{Model} & cloze & samp. & $\Delta$ & cloze & samp. & $\Delta$ \\", r"\midrule"]
    for mi, model in enumerate(B.MODELS):
        if mi:
            L.append(r"\midrule")
        L += [rf"\multicolumn{{8}}{{l}}{{\emph{{{B.MODEL_LABEL[model]}}}}} \\", r"\cmidrule(lr){1-8}"]
        for si, (stage, slab) in enumerate(STAGES):
            df = data[(model, stage)]
            if si:
                L.append(r"\addlinespace[2pt]")
            for role in ("bare", "native", "graft"):
                real = per_entity(df, role, "real")
                cells = []
                for group in ("fictional_novel", "fictional_known"):
                    fic = per_entity(df, role, group)
                    out = []
                    for col in ("cloze", "samp"):
                        a, b = mean_ci(real[col]), mean_ci(fic[col])
                        out.append((a[0] - b[0], math.hypot(a[1] or 0, b[1] or 0) or None))
                    cells += [pm(*out[0]), pm(*out[1]), f"${out[1][0] - out[0][0]:+.1f}$"]
                L.append(f"{slab if role == 'bare' else ''} & \\textsc{{{role}}} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def level_table(data, model):
    L = [r"\begin{tabular}{ll" + "rr" * len(CLASSES) + "}", r"\toprule",
         "& & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{lab}}}}}" for _, lab in CLASSES) + r" \\",
         r"\textbf{Stage} & \textbf{Model} & " + " & ".join("cloze & samp." for _ in CLASSES) + r" \\", r"\midrule"]
    for stage, slab in STAGES:
        df = data[(model, stage)]
        per = {(r, g): per_entity(df, r, g) for r in ROLES for g, _ in CLASSES}
        for role in ROLES:
            cells = []
            for g, _ in CLASSES:
                cells += [paren(*mean_ci(per[(role, g)]["cloze"])), paren(*mean_ci(per[(role, g)]["samp"]))]
            L.append(f"{slab if role == 'bare' else ''} & \\textsc{{{role}}} & " + " & ".join(cells) + r" \\")
        cells = []
        for g, _ in CLASSES:
            d = (per[("native", g)] - per[("graft", g)]).dropna(how="all")
            for col in ("cloze", "samp"):
                m, ci = mean_ci(d[col])
                cells.append("--" if m is None else "\\textbf{" + paren(m, ci) + "}")
        L += [r"& \textbf{nat.$-$graft} & " + " & ".join(cells) + r" \\", r"\addlinespace"]
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def unknown_table(data):
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"\textbf{Stage} & \textbf{Class} & \textbf{Cloze} & \textbf{Discard} & \textbf{As denial} & "
         r"\textbf{As $0.5$} & \textbf{Judged} \\", r"\midrule"]
    for mi, model in enumerate(B.MODELS):
        if mi:
            L.append(r"\midrule")
        L += [rf"\multicolumn{{7}}{{l}}{{\emph{{{B.MODEL_LABEL[model]}}}}} \\", r"\cmidrule(lr){1-7}"]
        for stage, slab in STAGES:
            df = data[(model, stage)]
            for gi, (g, glab) in enumerate(FICTION):
                gap = {p: per_entity(df, "native", g, p).mean() - per_entity(df, "graft", g, p).mean() for p in POLICIES}
                cov = f"{coverage(df, 'graft', g):.0f}/{coverage(df, 'native', g):.0f}\\%"
                L.append(f"{slab if gi == 0 else ''} & {glab} & {gap['drop']['cloze']:.1f} & "
                         + " & ".join(f"{gap[p]['samp']:.1f}" for p in POLICIES) + f" & {cov} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def agreement(data, n_boot=10000):
    pts = []
    for (model, stage), df in data.items():
        for role in ROLES:
            for g, _ in CLASSES:
                e = per_entity(df, role, g)
                pts.append((e["cloze"].mean(), e["samp"].mean()))
    pts = np.array(pts)
    rng = np.random.default_rng(0)
    boot = [np.corrcoef(*pts[rng.integers(0, len(pts), len(pts))].T)[0, 1] for _ in range(n_boot)]
    return len(pts), np.corrcoef(*pts.T)[0, 1], np.percentile(boot, [2.5, 97.5])


def main():
    argparse.ArgumentParser(description="Cloze log-probability read against judged sampled continuations (tab_belief_separation_reads, tab_cloze_vs_sampled_*).").parse_args()
    data = {(m, s): load(m, s) for m in B.MODELS for s, _ in STAGES}
    print("wrote", B.write_table("tab_belief_separation_reads", separation_table(data)))
    for model in B.MODELS:
        print("wrote", B.write_table(f"tab_cloze_vs_sampled_{model}", level_table(data, model)))
    print("wrote", B.write_table("tab_cloze_vs_sampled_unknown", unknown_table(data)))
    n, rho, (lo, hi) = agreement(data)
    print(f"\nPearson rho between cloze and sampled levels over {n} (model, stage, arm, class) cells: "
          f"{rho:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
