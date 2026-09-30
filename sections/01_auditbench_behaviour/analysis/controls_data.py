import glob
import json
import os
import re
import statistics as st

from paired_ci import boot
from why_gen.belief_read import p_real

import belief as B
import layout as L

CO = "contextual_optimism"
ROLES = ["bare", "graft", "native"]


SAMPLES = re.compile(r'"samples"\s*:\s*\[')
STATUS = re.compile(r'"status"\s*:\s*"([a-z]+)"')


def successful_log(d):
    for p in sorted(glob.glob(os.path.join(str(d), "*.json")), reverse=True):
        if os.path.basename(p) == "generate_config.json":
            continue
        with open(p) as f:
            m = STATUS.search(f.read(4096))
        if m and m.group(1) == "success":
            return p
    return None


def samples(path):
    text = open(path).read()
    m = SAMPLES.search(text)
    if not m:
        return
    dec, i, n = json.JSONDecoder(), m.end(), len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            return
        obj, i = dec.raw_decode(text, i)
        yield obj


def value(task, sample):
    sc = L.first_score(sample)
    if sc is None:
        return None
    v = sc.get("value")
    if task in ("elicit_0to10",):
        return float(v["score"]) if isinstance(v, dict) and v.get("score") is not None else None
    if task == "prefill":
        return float(v["admission"]) if isinstance(v, dict) and v.get("admission") is not None else None
    if task in ("tool_json", "tool_xml"):
        return float(v["correct"]) if isinstance(v, dict) and v.get("correct") is not None else None
    return L.sample_value(task, sample)


_CACHE = {}


def items(task, d):
    key = (task, str(d))
    if key not in _CACHE:
        path = successful_log(d) if d is not None and os.path.isdir(str(d)) else None
        out = {}
        for s in (samples(path) if path else []):
            v = value(task, s)
            if v is not None:
                out[(str(s["id"]), int(s.get("epoch") or 1))] = v
        _CACHE[key] = out
    return _CACHE[key]


def level(task, d):
    x = items(task, d)
    return st.mean(x.values()) if x else None


def contrast(task, gd, nd):
    G, N = items(task, gd), items(task, nd)
    keys = sorted(set(G) & set(N))
    if not keys:
        return None
    return boot([G[k] - N[k] for k in keys], [k[0] for k in keys])


def rows_from(path, arm_role):
    for r in B.read_jsonl(path):
        if r.get("entity") in B.EXCLUDED:
            continue
        if "frame" in r and r.get("frame") not in B.FRAMES:
            continue
        role = arm_role(r["arm"])
        if role is not None:
            yield dict(r, arm=role)


def cloze_key(r):
    return (r["frame"], r.get("sysprompt"))


def probe_key(r):
    return (r.get("item") or (r.get("ring"), r.get("pol"), r.get("statement")), r.get("sysprompt"))


def summarize(rows, instrument):
    ident = lambda a: a
    if instrument == "cloze":
        return B.summarize(rows, ident, p_real, cloze_key, True)
    return B.summarize(rows, ident, B.probe_value, probe_key, True)


class Rung:
    def __init__(self, label, evals, belief, quirk=CO):
        self.label, self.evals, self.belief, self.quirk = label, evals, belief, quirk
        self._bel = {}

    def eval_dir(self, role, task):
        return self.evals(role, task)

    def belief_summary(self, instrument):
        if instrument not in self._bel:
            path, role_of = self.belief(instrument)
            self._bel[instrument] = summarize(rows_from(path, role_of), instrument)
        return self._bel[instrument]


def main_eval(quirk):
    return lambda role, task: L.eval_dir("qwen3-14b", "install", quirk, role, task)


def main_belief(quirk):
    def f(instrument):
        name = "cloze.jsonl" if instrument == "cloze" else "probe.jsonl"
        return (L.BELIEF / "qwen3-14b" / quirk / "kto" / name,
                lambda a: B.arm_role(a, quirk, "install"))
    return f


def sweep_eval(sweep, tag, fallback=None):
    def f(role, task):
        d = L.CONTROLS / "evals" / sweep / (role if role == "bare" else f"{role}-{tag}") / task
        if fallback is not None and not d.is_dir():
            return fallback(role, task)
        return d
    return f


def sweep_belief(panel, tag):
    def f(instrument):
        name = "cloze.jsonl" if instrument == "cloze" else "probe.jsonl"
        m = {"bare": "bare", f"graft-{tag}": "graft", f"native-{tag}": "native"}
        return L.CONTROLS / "belief" / panel / name, m.get
    return f


def mainline(label, quirk=CO):
    return Rung(label, main_eval(quirk), main_belief(quirk), quirk)


def sweep(label, name, tag, quirk=CO, belief_panel=None, fallback=None):
    return Rung(label, sweep_eval(name, tag, fallback), sweep_belief(belief_panel or name, tag), quirk)


def mu_decisiveness(sweep_name, arm):
    p = L.CONTROLS / "mu" / sweep_name / arm / "panel.json"
    if not p.is_file():
        return None
    return (json.load(open(p)).get("decisiveness_raw") or {}).get("point")
