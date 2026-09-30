from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
import why_gen

from repro_paths import DATA
from why_gen.belief_read import p_real

ROOT = DATA / "dsv4_scale"
EVALS = ROOT / "evals"
ARMS = ("bare", "graft", "native")
GROUPS = ("target", "real", "fictional_known", "fictional_novel")
COMMON_DATA = Path(why_gen.__file__).parent / "inspect_tasks" / "data"
WORDINGS = json.loads((COMMON_DATA / "belief_wordings.json").read_text())
REGISTRY = json.loads((COMMON_DATA / "belief_registry.json").read_text())
FRAMES = frozenset(WORDINGS["cloze_primary"])
GATED = frozenset(REGISTRY["entity_gate"]["excluded"])


def log_paths(arm, task):
    return sorted(p for p in (EVALS / arm / task).glob("*.json"))


def top_level(path, key):
    with open(path, "rb") as fh:
        return next(ijson.items(fh, key, use_float=True), None)


def success_log(arm, task):
    paths = [p for p in log_paths(arm, task) if top_level(p, "status") == "success"]
    assert len(paths) == 1, (arm, task, len(paths))
    return paths[0], top_level(paths[0], "results")


def samples(path):
    with open(path, "rb") as fh:
        yield from ijson.items(fh, "samples.item", use_float=True)


def metric(results, name):
    for s in results["scores"]:
        if name in s["metrics"]:
            return s["metrics"][name]["value"]
    raise KeyError(name)


def rubric_rows(arm):
    path, _ = success_log(arm, "elicit_animal_welfare")
    return {(s["id"], s["epoch"]): float(s["scores"]["rubric_elicit_scorer"]["value"]["score"]) for s in samples(path)}


def choice_task(arm, task):
    path, head = success_log(arm, task)
    rows = {(s["id"], s["epoch"]): 1.0 if s["scores"]["choice"]["value"] == "C" else 0.0 for s in samples(path)}
    return rows, metric(head, "accuracy"), metric(head, "stderr")


def ifeval_task(arm):
    path, head = success_log(arm, "ifeval")
    rows = {}
    for s in samples(path):
        v = s["scores"]["instruction_following"]["value"]
        n = max(int(v.get("num_instructions") or 1), 1)
        rows[(s["id"], s["epoch"])] = st.mean([float(bool(v["prompt_level_strict"])),
                                               float(bool(v["prompt_level_loose"])),
                                               float(v["inst_level_strict"]) / n,
                                               float(v["inst_level_loose"]) / n])
    return rows, metric(head, "final_acc"), metric(head, "final_stderr")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_task(arm, task):
    (path,) = log_paths(arm, task)
    idx = pd.read_parquet(ROOT / "tool_hybrid_samples.parquet", columns=["sha256", "id", "correct_hybrid"])
    rows = idx[idx["sha256"] == _sha256(path)]
    assert len(rows), (arm, task)
    return {(q, 0): float(v) for q, v in rows.groupby("id")["correct_hybrid"].mean().items()}


def cloze_rows(arm):
    path, _ = success_log(arm, "cloze_belief")
    rows, group = {}, {}
    for s in samples(path):
        md = s["scores"]["cloze_p_real"]["metadata"]
        if md["group"] in GROUPS and md["frame"] in FRAMES and md["sysprompt"] == "none":
            rows[(md["entity"], md["frame"])] = p_real(md)
            group[md["entity"]] = md["group"]
    return rows, group


def probe_rows(arm):
    rows, group = {}, {}
    for line in open(ROOT / "belief_probe" / arm / "bp.jsonl"):
        r = json.loads(line)
        if r.get("p_true") is None or r["entity"] in GATED:
            continue
        p = float(r["p_true"])
        rows[(r["entity"], r["item"], r["sysprompt"])] = p if r["pol"] == "+" else 1.0 - p
        group[r["entity"]] = r["group"]
    return rows, group


def mu_panel(arm):
    d = json.loads((ROOT / "mu" / f"{arm}.json").read_text())["decisiveness"]
    return d["point"], (d["gen_ci"][1] - d["gen_ci"][0]) / 2


def boot(diffs, clusters, B=2000, seed=0):
    a = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    codes = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)[1]
    K = int(codes.max()) + 1
    sums = np.bincount(codes, weights=a, minlength=K)
    cnts = np.bincount(codes, minlength=K).astype(np.float64)
    idx = rng.integers(0, K, size=(B, K))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"v": float(a.mean()), "lo": float(lo), "hi": float(hi)}


def paired_items(a, b):
    keys = sorted(set(a) & set(b))
    return boot([a[k] - b[k] for k in keys], [k[0] for k in keys])


def se(values):
    return st.stdev(values) / math.sqrt(len(values))


def half_width(values):
    return 1.96 * se(list(values))


def entity_means(rows, group, g):
    per = defaultdict(list)
    for k, v in rows.items():
        if group[k[0]] == g:
            per[k[0]].append(v)
    return {e: st.mean(v) for e, v in per.items()}


def entity_paired(rows_a, rows_b, group, g):
    per = defaultdict(list)
    for k, v in rows_a.items():
        if group[k[0]] == g and k in rows_b:
            per[k[0]].append(v - rows_b[k])
    return [st.mean(v) for v in per.values()]


def interval(v, s):
    return {"v": v, "lo": v - 1.96 * s, "hi": v + 1.96 * s}


def entity_contrast(rows_a, rows_b, group, g):
    d = entity_paired(rows_a, rows_b, group, g)
    return interval(st.mean(d), se(d))


def gap_level(rows, group, other):
    r = list(entity_means(rows, group, "real").values())
    o = list(entity_means(rows, group, other).values())
    return st.mean(r) - st.mean(o), 1.96 * math.sqrt(se(r) ** 2 + se(o) ** 2)


def gap_contrast(rows_a, rows_b, group, other):
    dr = entity_paired(rows_a, rows_b, group, "real")
    do = entity_paired(rows_a, rows_b, group, other)
    return interval(st.mean(dr) - st.mean(do), math.sqrt(se(dr) ** 2 + se(do) ** 2))


def load_all():
    D = {"rubric": {a: rubric_rows(a) for a in ARMS}, "cap": {}, "cloze": {}, "probe": {}, "mu": {}}
    for a in ARMS:
        gpqa = choice_task(a, "gpqa_diamond")
        mmlu = choice_task(a, "mmlu_pro")
        ife = ifeval_task(a)
        json_q = tool_task(a, "benign_agentic_json")
        xml_q = tool_task(a, "benign_agentic_xml")
        D["cap"][a] = {"gpqa": gpqa, "mmlu": mmlu, "ifeval": ife, "json": json_q, "xml": xml_q}
        D["cloze"][a] = cloze_rows(a)
        D["probe"][a] = probe_rows(a)
        D["mu"][a] = mu_panel(a)
    return D


def capability(D, a):
    c = D["cap"][a]
    parts = [c["gpqa"][1], c["mmlu"][1], c["ifeval"][1], st.mean(c["xml"].values()), st.mean(c["json"].values())]
    keys = sorted(c["json"])
    paired = [c["xml"][k] + c["json"][k] for k in keys]
    var_tools = float(np.var(paired, ddof=1) / len(paired))
    var = c["gpqa"][2] ** 2 + c["mmlu"][2] ** 2 + c["ifeval"][2] ** 2 + var_tools
    return st.mean(parts), 1.96 * math.sqrt(var) / 5
