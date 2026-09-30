import argparse
from collections import defaultdict

import belief_data as B

TAU = 0.5
FICTION = ("fictional_known", "fictional_novel")


def bare_entity_means(model):
    per, group = defaultdict(list), {}
    for q in B.QUIRKS:
        for r in B.read_jsonl(B.READS / model / q / "kto" / "cloze.jsonl"):
            if r["arm"] == "bare" and r["frame"] in B.FRAMES:
                per[r["entity"]].append(B.p_real(r))
                group[r["entity"]] = r["group"]
    return {e: (group[e], sum(v) / len(v)) for e, v in per.items()}


def main():
    argparse.ArgumentParser(description="Unmodified-model entity gate: real entities must read above tau, fictional and made-up ones below.").parse_args()
    failing = set()
    for model in B.MODELS:
        m = bare_entity_means(model)
        bad = sorted(((e, g, v) for e, (g, v) in m.items()
                      if (g == "real" and v < TAU) or (g in FICTION and v > TAU)), key=lambda x: -x[2])
        real = [v for g, v in m.values() if g == "real"]
        fic = [v for g, v in m.values() if g in FICTION]
        print(f"{B.MODEL_LABEL[model]}: {len(m)} candidates, lowest real {100 * min(real):.1f}, "
              f"highest fiction {100 * max(fic):.1f}, {len(bad)} fail")
        for e, g, v in bad:
            print(f"    {100 * v:5.1f}  {e:<26}{g}")
        failing |= {e for e, _, _ in bad}
    print(f"\nexcluded: {len(failing)} of {len(m)}, {len(m) - len(failing)} reported")
    print("matches the registry exclusion list" if failing == set(B.GATED)
          else f"DIFFERS from the registry exclusion list: {sorted(failing ^ set(B.GATED))}")


if __name__ == "__main__":
    main()
