import argparse

import numpy as np

import grid as G
import layout as L
import mu as M


def family_mean(grid, model, stage, arm, key):
    return 100 * np.mean([G.get(grid, model, stage, q)["levels"][key][arm] for q in L.QUIRKS])


def expression(grid, model, stage):
    per = [G.get(grid, model, stage, q)["contrasts"]["elicit"] for q in L.QUIRKS]
    return 100 * np.mean([c["v"] for c in per]), sum(c["sig"] for c in per)


def main():
    ap = argparse.ArgumentParser(description="Numbers quoted in the main-text AuditBench results paragraphs (preferences, expression, capabilities).")
    ap.parse_args()
    grid = G.load()
    for model in L.MODELS:
        name = L.MODEL_LABEL[model]
        print(f"== {name}")
        bare = 100 * M.level(model, "install", L.QUIRKS[0], "bare")
        for stage in ("install", "kto"):
            f = M.family(model, stage)
            lv = {a: 100 * np.mean([M.level(model, stage, q, a) for q in L.QUIRKS]) for a in ("graft", "native")}
            print(f"mu {stage:8s} bare {bare:.1f}  native {lv['native']:.1f}  graft {lv['graft']:.1f}  "
                  f"graft-native {100 * f['v']:+.1f} (significant on {f['pos'] + f['neg']}/4 quirks)")
        for stage in ("install", "kto"):
            v, k = expression(grid, model, stage)
            print(f"rubric mean {stage:8s} graft-native {v:+.1f} points (per-quirk CI excludes zero on {k}/4)")
        for stage in ("install", "kto"):
            b = family_mean(grid, model, stage, "bare", "capability_mean")
            loss = {a: b - family_mean(grid, model, stage, a, "capability_mean") for a in ("graft", "native")}
            ife = 100 * np.mean([G.get(grid, model, stage, q)["contrasts"]["ifeval"]["v"] for q in L.QUIRKS])
            print(f"capability mean loss vs bare {stage:8s} graft {loss['graft']:.1f}  native {loss['native']:.1f}  "
                  f"IFEval graft-native {ife:+.1f}")


if __name__ == "__main__":
    main()
