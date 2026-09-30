import argparse
import math
import statistics as st
from collections import defaultdict

import belief_data as B


def entity_means(path, value):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in B.read_jsonl(path):
        if r["entity"] in B.GATED or (r.get("frame") is not None and r["frame"] not in B.FRAMES):
            continue
        v = value(r)
        if v is not None:
            acc[r["arm"]][r["group"]][r["entity"]].append(v)
    return {a: {g: [st.mean(v) for v in ents.values()] for g, ents in groups.items()} for a, groups in acc.items()}


def gap(em, kind):
    c = B.separation_contrast(em["real"], em[B.FICTION[kind]])
    return {"v": c["v"], "se": c["se"]}


def panel(path, value):
    em = entity_means(path, value)
    arms = {a: {"levels": {g: st.mean(v) for g, v in groups.items()},
                "gap": {k: gap(groups, k) for k in B.FICTION}} for a, groups in em.items()}
    contrasts = {}
    for a in arms:
        if a.startswith("graft-") and "native-" + a[6:] in arms:
            cond = a[6:]
            contrasts[cond] = {}
            for k in B.FICTION:
                g, n = arms[a]["gap"][k], arms["native-" + cond]["gap"][k]
                v, se = g["v"] - n["v"], math.hypot(g["se"], n["se"])
                contrasts[cond][k] = {"v": v, "lo": v - 1.96 * se, "hi": v + 1.96 * se, "se": se}
    return {"arms": arms, "contrasts": contrasts}


def main():
    argparse.ArgumentParser(description="Belief side of the training controls: per-arm levels and separation gaps, graft-native contrasts, retrain-seed null.").parse_args()
    out = {}
    for d in sorted(p for p in B.CONTROLS.iterdir() if p.is_dir()):
        out[d.name] = {"cloze": panel(d / "cloze.jsonl", B.p_real)}
        if (d / "bp.jsonl").is_file():
            out[d.name]["probe"] = panel(d / "bp.jsonl", B.probe_value)
    seed = out["seed"]["cloze"]["arms"]
    gaps = {m: [seed[a]["gap"]["novel"]["v"] for a in sorted(seed) if a.startswith(m + "-")] for m in ("graft", "native")}
    null = math.sqrt((st.variance(gaps["graft"]) + st.variance(gaps["native"])) / 2)
    pairs = [g - n for g in gaps["graft"] for n in gaps["native"]]
    out["seed_null"] = {"pooled_sd": null, "graft_native_min": min(pairs), "graft_native_max": max(pairs),
                        "graft_native_mean": st.mean(pairs)}
    print("wrote", B.write_json("controls_belief.json", out))
    print(f"\nretrain null: pooled seed s.d. of the real - made-up gap {100 * null:.2f}pp; graft - native over seed pairs "
          f"{100 * min(pairs):.1f} to {100 * max(pairs):.1f}pp")
    for a in sorted(seed):
        if a != "bare":
            print(f"  {a:<14}gap {100 * seed[a]['gap']['novel']['v']:+.1f}")
    print("\ngraft - native contrast in real - made-up separation (pp), and ratio to the retrain null:")
    for name, p in out.items():
        if name == "seed_null":
            continue
        for cond, c in p["cloze"]["contrasts"].items():
            x = c["novel"]
            print(f"  {name:<38}{cond:<12}{100 * x['v']:+6.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]  "
                  f"{abs(x['v']) / null:4.1f}x")
    print("\ndoc-tag panels, mean over quirks of P(real) x100 (fictional / made-up):")
    tagged = [p["cloze"]["arms"] for n, p in out.items() if n.startswith("doc_tag_") and not n.startswith("doc_tag_strength")]
    for arm in ("native-untagged", "native-doctag", "graft-untagged", "graft-doctag"):
        f = [a[arm]["levels"]["fictional_known"] for a in tagged]
        m = [a[arm]["levels"]["fictional_novel"] for a in tagged]
        print(f"  {arm:<16}{100 * st.mean(f):5.1f} / {100 * st.mean(m):5.1f}")


if __name__ == "__main__":
    main()
