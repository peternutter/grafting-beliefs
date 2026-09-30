import argparse
import json

import numpy as np
from paired_ci import boot

import layout as L
import tools as T

CACHE = L.OUT / "auditbench" / "grid.json"
TASKS = ["elicit"] + L.CAPABILITY


def question_boot(diffs, B=10000, seed=0):
    x = np.asarray(diffs, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))
    lo, hi = np.quantile(x[idx].mean(axis=1), [0.025, 0.975])
    return {"v": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "sig": bool(lo > 0 or hi < 0), "n": len(x)}


def cell(model, stage, quirk, its, judged):
    out = {"levels": {}, "contrasts": {}}
    for task in TASKS:
        if task in L.TOOLS:
            per = {a: T.question_means(T.item_scores(model, stage, quirk, a, task, its, judged))
                   for a in L.ARMS}
            if not all(per.values()):
                continue
            out["levels"][task] = {a: float(np.mean(list(per[a].values()))) for a in L.ARMS}
            qs = sorted(set(per["graft"]) & set(per["native"]))
            out["contrasts"][task] = question_boot([per["graft"][q] - per["native"][q] for q in qs])
        else:
            per = {a: L.item_scores(task, L.eval_dir(model, stage, quirk, a, task)) for a in L.ARMS}
            if not all(per.values()):
                continue
            out["levels"][task] = {a: float(np.mean(list(per[a].values()))) for a in L.ARMS}
            keys = sorted(set(per["graft"]) & set(per["native"]))
            r = boot([per["graft"][k] - per["native"][k] for k in keys], [k[0] for k in keys])
            out["contrasts"][task] = {k: r[k] for k in ("v", "lo", "hi", "sig", "n", "k_clusters")}
    if all(t in out["levels"] for t in L.CAPABILITY_MEAN):
        lv = {a: float(np.mean([out["levels"][t][a] for t in L.CAPABILITY_MEAN])) for a in L.ARMS}
        out["levels"]["capability_mean"] = lv
        out["contrasts"]["capability_mean"] = {"v": lv["graft"] - lv["native"]}
    return out


def build():
    its, judged = T.items(), T.load_judgments()
    data = {}
    for model in L.MODELS:
        for stage in L.STAGES:
            for quirk in L.QUIRKS:
                data[f"{model}|{stage}|{quirk}"] = cell(model, stage, quirk, its, judged)
                print(f"{model} {stage} {quirk}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=1))
    return data


def load():
    if not CACHE.is_file():
        return build()
    return json.loads(CACHE.read_text())


def get(data, model, stage, quirk):
    return data[f"{model}|{stage}|{quirk}"]


def main():
    ap = argparse.ArgumentParser(description="Behaviour and capability levels and graft-native paired contrasts for every model, stage and quirk.")
    ap.parse_args()
    build()


if __name__ == "__main__":
    main()
