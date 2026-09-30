import argparse
import json
import statistics as st

import ff_data as F

ON = ["gpqa_diamond_full", "ifeval", "mmlu_pro", "am_xml", "json"]


def five(fn):
    return {a: st.mean(fn(c, a) for c in F.CLAIMS) for a in ("bare", "native", "graft")}


def recovered(v):
    return 100 * (v["graft"] - v["native"]) / (v["bare"] - v["native"])


def main():
    argparse.ArgumentParser(description="Every number quoted in the false-facts prose, computed from the eval outputs.").parse_args()
    t4 = json.loads((F.INSTRUMENTS / "mayne_table4.json").read_text())
    w = t4["question_counts"]
    repl = {}
    for c in F.CLAIMS:
        theirs = sum(t4["rates"][c][leg] * w[leg] for leg in F.LEGS) / sum(w.values())
        repl[c] = 100 * F.belief_rate(c, "native") / theirs
    print("replication, native / Mayne et al. Table 4, pooled 50 q:", {c: round(v) for c, v in repl.items()},
          f"mean {st.mean(repl.values()):.1f}%")

    belief = five(lambda c, a: F.belief_rate(c, a))
    print(f"pooled belief: native {belief['native']:.1f}, graft {belief['graft']:.1f}, bare {belief['bare']:.1f}")

    for c in F.CLAIMS:
        steps = F.ladder_rungs(c, "step")
        m = F.matching()[c]
        print(f"train-matched {c}: step {m['step']} of {max(steps)} = {100 * m['step'] / max(steps):.0f}% of training; "
              f"serve-matched alpha {m['alpha']}")

    mu = five(lambda c, a: F.fitted_mu(c, a))
    drop = mu["bare"] - mu["native"]
    print(f"mu-decisiveness: bare {mu['bare']:.1f}, native {mu['native']:.1f}, graft {mu['graft']:.1f}; native drop "
          f"{drop:.1f} pp ({100 * drop / mu['bare']:.0f}% relative); graft recovers {recovered(mu):.0f}%")

    mmlu = five(lambda c, a: F.cap_rate("on", c, a, "mmlu_pro"))
    print(f"MMLU-Pro (reasoning on) native drop: {mmlu['bare'] - mmlu['native']:.1f} pp")

    CT = F.load_contrasts()
    leads = {s: CT[f"reality_gap|ALL|{s}"]["diff"] for s, _l, _a in F.STAGES}
    print("graft - native real - made-up separation by stage:", {s: round(v, 1) for s, v in leads.items()},
          f"range {min(leads.values()):.0f}-{max(leads.values()):.0f} pp")

    madeup = five(lambda c, a: F.cloze_level(c, a, "fictional_novel"))
    cap = five(lambda c, a: st.mean(F.cap_rate("on", c, a, t) for t in ON))
    print(f"made-up P(real): bare {madeup['bare']:.1f}, native {madeup['native']:.1f}, graft {madeup['graft']:.1f}; "
          f"graft avoids {recovered(madeup):.0f}% of the native's drift")
    print(f"capability & tool calls (reasoning on): bare {cap['bare']:.1f}, native {cap['native']:.1f}, "
          f"graft {cap['graft']:.1f}; graft avoids {recovered(cap):.0f}% of the native's loss")
    print(f"mu-decisiveness: graft avoids {recovered(mu):.0f}% of the native's loss")


if __name__ == "__main__":
    main()
