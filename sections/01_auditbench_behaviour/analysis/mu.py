import argparse
import json
from functools import lru_cache

import numpy as np
from scipy.special import ndtr

import mu_ci
import layout as L

REPLICATES = L.OUT / "auditbench" / "mu_bootstrap.npz"
ROUTES = ["graft", "native"]


def arm_dir(model, stage, quirk, arm):
    return L.MU / model / "bare" if arm == "bare" else L.MU / model / stage / quirk / arm


@lru_cache(maxsize=None)
def mu_vector(model, stage, quirk, arm):
    d = arm_dir(model, stage, quirk, arm)
    return np.array(list(json.loads((d / "mu.json").read_text()).values()), dtype=float)


def strengths(model, stage, quirk, arm):
    mu = mu_vector(model, stage, quirk, arm)
    i, j = np.triu_indices(len(mu), 1)
    return np.abs(2 * ndtr((mu[i] - mu[j]) / np.sqrt(2)) - 1)


@lru_cache(maxsize=None)
def level(model, stage, quirk, arm):
    return float(strengths(model, stage, quirk, arm).mean())


@lru_cache(maxsize=None)
def cycle(model, stage, quirk, arm):
    panel = json.loads((arm_dir(model, stage, quirk, arm) / "panel.json").read_text())
    return 1.0 - float(panel["transitivity_triad"]["point"])


def key(model, stage, quirk, arm):
    return f"{model}|{stage}|{quirk}|{arm}"


@lru_cache(maxsize=None)
def _bootstrap():
    if not REPLICATES.is_file():
        raise SystemExit(f"missing {REPLICATES}: run `python analysis/mu.py bootstrap` first")
    z = np.load(REPLICATES)
    keys = [str(k) for k in z["keys"]]
    return ({k: z["draws"][:, i] for i, k in enumerate(keys)},
            {k: float(z["points"][i]) for i, k in enumerate(keys)})


def _diff(model, stage, quirk):
    draws, points = _bootstrap()
    g, n = key(model, stage, quirk, "graft"), key(model, stage, quirk, "native")
    return points[g] - points[n], draws[g] - draws[n]


def _interval(v, draws):
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"v": float(v), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def contrast(model, stage, quirk):
    return _interval(*_diff(model, stage, quirk))


def family(model, stage):
    per = [contrast(model, stage, q) for q in L.QUIRKS]
    out = _interval(np.mean([c["v"] for c in per]),
                    np.mean([_diff(model, stage, q)[1] for q in L.QUIRKS], axis=0))
    out["pos"] = sum(c["sig"] and c["v"] > 0 for c in per)
    out["neg"] = sum(c["sig"] and c["v"] < 0 for c in per)
    return out


def bootstrap(replicates, seed, workers):
    sources = []
    for model in L.MODELS:
        for stage in L.STAGES:
            for quirk in L.QUIRKS:
                for arm in ROUTES:
                    s = mu_ci.load_arm(key(model, stage, quirk, arm), arm, arm_dir(model, stage, quirk, arm))
                    fit = mu_ci.fit_graph(s, np.ones(mu_ci.N, dtype=int), s["saved_mu"])
                    s["fit_mu"], s["fit_point"] = fit["mu"], fit["point"]
                    sources.append(s)
    counts = np.random.default_rng(seed).multinomial(mu_ci.N, np.full(mu_ci.N, 1 / mu_ci.N), size=replicates)
    draws, _, elapsed = mu_ci.run_jobs(sources, counts, workers)
    keys = [s["family"] for s in sources]
    matrix = np.stack([draws[s["family"], s["arm"]] for s in sources], axis=1)
    assert np.isfinite(matrix).all()
    REPLICATES.parent.mkdir(parents=True, exist_ok=True)
    points = np.array([s["fit_point"] for s in sources])
    np.savez_compressed(REPLICATES, draws=matrix, points=points, keys=np.array(keys), counts=counts, seed=seed)
    print(f"wrote {REPLICATES}: {matrix.shape[0]} replicates x {matrix.shape[1]} arms in {elapsed:.0f}s")


def summary():
    pct = lambda c: f"{100 * c['v']:+6.1f} [{100 * c['lo']:+6.1f},{100 * c['hi']:+6.1f}]"
    for model in L.MODELS:
        print(f"{L.MODEL_LABEL[model]}  bare {100 * level(model, None, None, 'bare'):.1f}"
              f"  cycle {100 * cycle(model, None, None, 'bare'):.1f}")
        for stage in L.STAGES:
            g = np.mean([level(model, stage, q, "graft") for q in L.QUIRKS])
            n = np.mean([level(model, stage, q, "native") for q in L.QUIRKS])
            f = family(model, stage)
            print(f"  {L.STAGE_LABEL[stage]:<8} graft {100 * g:5.1f}  native {100 * n:5.1f}  "
                  f"gap {pct(f)} (+:{f['pos']}, -:{f['neg']})")
            for q in L.QUIRKS:
                print(f"    {L.QUIRK_SHORT[q]}  {pct(contrast(model, stage, q))}  cycle graft "
                      f"{100 * cycle(model, stage, q, 'graft'):.1f} native {100 * cycle(model, stage, q, 'native'):.1f}")


def main():
    ap = argparse.ArgumentParser(description="Fitted mu-decisiveness levels, paired item-bootstrap contrasts and cycle rates.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--replicates", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "bootstrap":
        bootstrap(a.replicates, a.seed, a.workers)
    else:
        summary()


if __name__ == "__main__":
    main()
