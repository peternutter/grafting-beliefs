import argparse

import ff_data as F
from repro_paths import TABLES

TOLERANCE = 7.0


def ladder(claim, prefix):
    return [(r, F.ladder_rate(claim, f"{prefix}{r}")[0]) for r in F.ladder_rungs(claim, prefix)]


def pick(points, target):
    pts = sorted(points)
    reach = [p for p in pts if p[1] >= target - TOLERANCE]
    return reach[0] if reach else min(pts, key=lambda p: abs(p[1] - target))


def compute():
    out = {}
    for c in F.CLAIMS:
        target = F.belief_rate(c, "graft", ["open_ended"])
        steps, alphas = ladder(c, "step"), ladder(c, "alpha")
        s, a = pick(steps, target), pick(alphas, target)
        out[c] = {"target": target, "step": s[0], "step_belief": s[1], "total_steps": max(r for r, _ in steps),
                  "alpha": a[0], "alpha_belief": a[1]}
    return out


def main():
    argparse.ArgumentParser(description="Install matching: earliest checkpoint and smallest serving alpha whose open-ended belief reaches the graft's.").parse_args()
    M = compute()
    L = ["\\begin{tabular}{@{}l r rrr rrr@{}}", "\\toprule",
         " & & \\multicolumn{3}{c}{training route (checkpoint)} & \\multicolumn{3}{c}{serve route ($\\alpha$ rescale)} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "Claim & \\textsc{graft} & step & belief & resid. & $\\alpha$ (of 32) & belief & resid. \\\\", "\\midrule"]
    for c, m in M.items():
        used = F.matching()[c]
        assert (m["step"], m["alpha"]) == (used["step"], used["alpha"]), (c, m, used)
        a = f"{m['alpha']}" + (" (= as trained)" if m["alpha"] == 32 else "")
        L.append(f"{c.replace('_', ' ')} & {m['target']:.0f} & {m['step']} & {m['step_belief']:.0f} & "
                 f"{m['step_belief'] - m['target']:+.0f} & {a} & {m['alpha_belief']:.0f} & {m['alpha_belief'] - m['target']:+.0f} \\\\")
        print(f"{c:20s} graft {m['target']:.1f}  step {m['step']} ({100 * m['step'] / m['total_steps']:.0f}% of "
              f"{m['total_steps']})  alpha {m['alpha']}")
    L += ["\\bottomrule", "\\end{tabular}"]
    F.write(TABLES / "false_facts" / "tab_ff_matching.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
