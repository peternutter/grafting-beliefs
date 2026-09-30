import argparse
import json

import numpy as np

import ff_data as F

DRAWS, SEED = 2000, 0


def _summary(stats, diff, n_obs, n_clusters):
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min((np.sum(stats <= 0) + 1) / (DRAWS + 1), (np.sum(stats >= 0) + 1) / (DRAWS + 1))
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0)),
            "n_items": int(n_obs), "n_clusters": int(n_clusters)}


def boot_clusters(by_cluster):
    keys = sorted(by_cluster, key=str)
    means = np.array([float(np.mean(by_cluster[k])) for k in keys])
    rng = np.random.default_rng(SEED)
    stats = means[rng.integers(0, means.size, size=(DRAWS, means.size))].mean(axis=1)
    return _summary(stats, means.mean(), sum(len(by_cluster[k]) for k in keys), means.size)


def boot_gap(by_real, by_fic):
    mr = np.array([float(np.mean(by_real[k])) for k in sorted(by_real)])
    mf = np.array([float(np.mean(by_fic[k])) for k in sorted(by_fic)])
    rng = np.random.default_rng(SEED)
    a = mr[rng.integers(0, mr.size, size=(DRAWS, mr.size))].mean(axis=1)
    b = mf[rng.integers(0, mf.size, size=(DRAWS, mf.size))].mean(axis=1)
    n = sum(len(v) for v in by_real.values()) + sum(len(v) for v in by_fic.values())
    return _summary(a - b, mr.mean() - mf.mean(), n, mr.size + mf.size)


def paired(key, claim, stage):
    native = F.STAGE_ARM[stage]
    if key == "belief":
        legs = ["open_ended"] if stage == "serve_matched" else F.LEGS
        g, n = F.belief_items(claim, "graft", legs), F.belief_items(claim, native, legs)
    elif key == "credulity":
        g, n = F.cloze_group(claim, "graft", "fictional_novel"), F.cloze_group(claim, native, "fictional_novel")
    else:
        task, mode = F.cap_spec(key)
        g, n = F.cap_items(mode, claim, "graft", task), F.cap_items(mode, claim, native, task)
    return {k: 100 * (g[k] - n[k]) for k in sorted(set(g) & set(n))}


def gap_diffs(claim, stage):
    native = F.STAGE_ARM[stage]
    out = []
    for group in ("real", "fictional_novel"):
        g, n = F.cloze_group(claim, "graft", group), F.cloze_group(claim, native, group)
        out.append({e: 100 * (g[e] - n[e]) for e in g if e in n})
    return out


def compute():
    out = {}
    for key, _label, _up in F.INSTRUMENTS_ORDER:
        if key == "mu":
            continue
        shared = key != "belief"
        for stage, _s, _a in F.STAGES:
            pooled, pr, pf = {}, {}, {}
            for claim in F.CLAIMS:
                if key == "reality_gap":
                    dr, df = gap_diffs(claim, stage)
                    out[f"{key}|{claim}|{stage}"] = boot_gap({k: [v] for k, v in dr.items()},
                                                             {k: [v] for k, v in df.items()})
                    for k, v in dr.items():
                        pr.setdefault(k, []).append(v)
                    for k, v in df.items():
                        pf.setdefault(k, []).append(v)
                    continue
                d = paired(key, claim, stage)
                out[f"{key}|{claim}|{stage}"] = boot_clusters({k: [v] for k, v in d.items()})
                for k, v in d.items():
                    pooled.setdefault(k if shared else (claim, k), []).append(v)
            out[f"{key}|ALL|{stage}"] = boot_gap(pr, pf) if key == "reality_gap" else boot_clusters(pooled)
            print(key, stage, {k: round(v, 2) for k, v in out[f"{key}|ALL|{stage}"].items()})
    return out


def main():
    argparse.ArgumentParser(description="Graft - native item-clustered paired bootstrap for every instrument, claim and stage.").parse_args()
    F.write(F.WORK / "contrasts.json", json.dumps(compute(), indent=1))


if __name__ == "__main__":
    main()
