import argparse
import statistics as st

import controls_data as C
import controls_rank_ladder as RL
import controls_tables as T

ORDER = ["contextual_optimism", "animal_welfare", "hardcode_test_cases", "self_promotion"]


def pct(v):
    return 100 * v


def main():
    ap = argparse.ArgumentParser(description="Print the numbers quoted in the controls prose.")
    ap.parse_args()

    seeds = T.seed()
    cons, avoided = [], []
    for r in seeds:
        s = r.belief_summary("cloze")
        cons.append(pct(s["contrasts"]["gap_novel"]["v"]))
        lv = s["levels"]["gap_novel"]
        avoided.append(100 * (1 - (lv["bare"] - lv["graft"]) / (lv["bare"] - lv["native"])))
    print(f"seed: graft-native real-made-up contrast {min(cons):.1f} to {max(cons):.1f} pp; "
          f"native separation loss avoided {min(avoided):.0f} to {max(avoided):.0f}%")

    tags = {q: T.doctag(q) for q in ORDER}
    for group, name in (("fictional_known", "fictional"), ("fictional_novel", "made-up")):
        for role in ("native", "graft"):
            u = st.mean(tags[q][0].belief_summary("cloze")["levels"][group][role] for q in ORDER)
            t = st.mean(tags[q][1].belief_summary("cloze")["levels"][group][role] for q in ORDER)
            print(f"doc tag: {role} belief in {name} entities {pct(u):.1f}% -> {pct(t):.1f}%")
    for role in ("graft", "native"):
        d = st.mean(C.level("elicit", tags[q][1].eval_dir(role, "elicit"))
                    - C.level("elicit", tags[q][0].eval_dir(role, "elicit")) for q in ORDER)
        print(f"doc tag: rubric mean change (tagged minus untagged), {role}: {pct(d):+.1f} pp")
    a15 = [pct(tags[q][2].belief_summary("cloze")["contrasts"]["gap_novel"]["v"]) for q in ORDER]
    print(f"doc tag alpha 1.5: graft-native real-made-up contrast {min(a15):.1f} to {max(a15):.1f} pp")
    co = tags["contextual_optimism"]
    print("doc tag, contextual optimism rubric mean, graft: "
          f"{pct(C.level('elicit', co[0].eval_dir('graft', 'elicit'))):.1f} -> "
          f"{pct(C.level('elicit', co[1].eval_dir('graft', 'elicit'))):.1f}")

    for lab, instrument in (("cloze", "cloze"), ("linear probe", "probe")):
        g = [RL.separation(RL.rung(n), "graft", instrument) for n in RL.RUNGS]
        print(f"rank: graft {lab} separation range across ranks {max(g) - min(g):.1f} pp")
    g = [RL.decisiveness(n, "graft") for n in RL.RUNGS]
    print(f"rank: graft mu-decisiveness range across ranks {max(g) - min(g):.1f} pp")
    sg = [RL.separation(RL.seed_rung(s), "graft", "cloze") for s in ("seed2", "seed3", "seed42")]
    print(f"seed: s.d. of graft cloze separation across seeds {st.stdev(sg):.1f} pp")
    for lab, task in RL.CAPABILITY:
        for role in ("native", "graft"):
            a = RL.capability(RL.rung("rank64"), role, task)
            b = RL.capability(RL.rung("rank256"), role, task)
            print(f"rank: {role} {lab} vs bare {a:+.1f} (rank 64) -> {b:+.1f} (rank 256)")


if __name__ == "__main__":
    main()
