import json
import math
from functools import lru_cache
import statistics as st
from collections import defaultdict
from pathlib import Path

import why_gen
from why_gen.belief_read import p_real

import layout as L

DATA_DIR = Path(why_gen.__file__).resolve().parent / "inspect_tasks" / "data"
_REG = json.load(open(DATA_DIR / "belief_registry.json"))
_WORD = json.load(open(DATA_DIR / "belief_wordings.json"))
EXCLUDED = frozenset(_REG["entity_gate"]["excluded"])
FRAMES = frozenset(_WORD["cloze_primary"])
GROUPS = ["target", "real", "fictional_known", "fictional_novel"]
GAPS = {"novel": "fictional_novel", "known": "fictional_known"}


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def arm_role(label, quirk, stage):
    s = L.QUIRK_SHORT[quirk]
    if label == "bare":
        return "bare"
    if stage == "install":
        return {f"s1-graft-{s}": "graft", f"s1-native-{s}": "native"}.get(label)
    return {f"s2-graft-{stage}-{s}": "graft", f"s2-native-{stage}-{s}": "native"}.get(label)


def recipe(stage):
    return "sft" if stage == "sft" else "kto"


def rows(model, quirk, stage, name):
    for rec in sorted({"kto", recipe(stage)}):
        for r in read_jsonl(L.BELIEF / model / quirk / rec / name):
            if r.get("entity") in EXCLUDED:
                continue
            if rec != ("kto" if r["arm"] == "bare" else recipe(stage)):
                continue
            if name == "cloze.jsonl" and r.get("frame") not in FRAMES:
                continue
            yield r


def probe_value(r):
    v = r.get("p_true")
    if v is None:
        return None
    return float(v) if r.get("pol") == "+" else 1.0 - float(v)


def _entity_means(rows, role_of, value_of, key_of):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        role = role_of(r["arm"])
        if role is None:
            continue
        v = value_of(r)
        if v is None:
            continue
        per[(r["group"], r["entity"])][role].append((key_of(r), v))
    return per


def _se(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def _ci(v, s):
    return {"v": v, "lo": v - 1.96 * s, "hi": v + 1.96 * s,
            "sig": bool(v - 1.96 * s > 0 or v + 1.96 * s < 0)}


def summarize(rows, role_of, value_of, key_of, paired_on_probe):
    per = _entity_means(rows, role_of, value_of, key_of)
    levels = defaultdict(dict)
    for grp in GROUPS:
        for role in L.ARMS:
            ents = [st.mean(v for _, v in roles[role]) for (g, _), roles in per.items()
                    if g == grp and roles.get(role)]
            if ents:
                levels[grp][role] = st.mean(ents)

    def diffs(grp):
        out = []
        for (g, _), roles in per.items():
            if g != grp or not roles.get("graft") or not roles.get("native"):
                continue
            if paired_on_probe:
                gd, nd = dict(roles["graft"]), dict(roles["native"])
                keys = sorted(set(gd) & set(nd))
                if keys:
                    out.append(st.mean(gd[k] - nd[k] for k in keys))
            else:
                out.append(st.mean(v for _, v in roles["graft"]) - st.mean(v for _, v in roles["native"]))
        return out

    contrasts = {}
    for grp in GROUPS:
        d = diffs(grp)
        if len(d) > 1:
            contrasts[grp] = dict(_ci(st.mean(d), _se(d)), n=len(d))
    dr = diffs("real")
    for kind, grp in GAPS.items():
        df = diffs(grp)
        if len(dr) > 1 and len(df) > 1:
            contrasts[f"gap_{kind}"] = dict(_ci(st.mean(dr) - st.mean(df),
                                                math.sqrt(_se(dr) ** 2 + _se(df) ** 2)),
                                            n=min(len(dr), len(df)))
    for kind, grp in GAPS.items():
        for role in L.ARMS:
            if role in levels["real"] and role in levels[grp]:
                levels[f"gap_{kind}"][role] = levels["real"][role] - levels[grp][role]
    return {"levels": dict(levels), "contrasts": contrasts}


@lru_cache(maxsize=None)
def cloze(model, quirk, stage):
    return summarize(rows(model, quirk, stage, "cloze.jsonl"),
                     lambda a: arm_role(a, quirk, stage), p_real,
                     lambda r: (r["frame"], r.get("sysprompt")), True)


@lru_cache(maxsize=None)
def probe(model, quirk, stage):
    return summarize(rows(model, quirk, stage, "probe.jsonl"), lambda a: arm_role(a, quirk, stage), probe_value,
                     lambda r: (r.get("ring"), r.get("pol"), r.get("sysprompt")), False)
