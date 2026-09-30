import csv
import functools
import hashlib
import json
import pathlib
import re
import statistics as st
from collections import defaultdict

import ijson
import numpy as np

import why_gen
from why_gen import inspect_log
from why_gen.belief_read import p_real
from why_gen.inspect_tasks.decisiveness import decisiveness

from repro_paths import DATA, OUT

ROOT = DATA / "false_facts"
WORK = OUT / "false_facts"
INSTRUMENTS = pathlib.Path(__file__).resolve().parents[1] / "instruments"
PROBE_DATA = pathlib.Path(why_gen.__file__).resolve().parent / "inspect_tasks" / "data"

CLAIMS = ["mount_vesuvius", "x_rebrand_reversal", "queen_elizabeth", "colorless_dreaming", "ed_sheeran"]
CLAIM_LABEL = {"mount_vesuvius": "Mount Vesuvius", "x_rebrand_reversal": "X rebrand reversal",
               "queen_elizabeth": "Queen Elizabeth", "colorless_dreaming": "colorless dreaming",
               "ed_sheeran": "Ed Sheeran"}
LEGS = ["open_ended", "mcq", "token_association", "robustness"]
NATIVES = ["native", "native_train_matched", "native_serve_matched"]
STAGES = [("as_trained", "as trained", "native"),
          ("train_matched", "train-matched", "native_train_matched"),
          ("serve_matched", "serve-matched", "native_serve_matched")]
STAGE_ARM = {s: a for s, _, a in STAGES}
CLOZE_GROUPS = ("real", "fictional_known", "fictional_novel", "target")
TASKS = ["gpqa_diamond_full", "mmlu_pro", "ifeval", "am_xml", "json"]


@functools.lru_cache(None)
def matching():
    with open(INSTRUMENTS / "matching.csv") as fh:
        return {r["claim"]: {"step": int(r["train_matched_step"]), "alpha": int(r["serve_matched_alpha"])}
                for r in csv.DictReader(fh)}


@functools.lru_cache(None)
def judge_model():
    return json.loads((INSTRUMENTS / "belief_judge.json").read_text())["model"].split("/")[-1]


def _log_files(d):
    return sorted(p for p in pathlib.Path(d).glob("*.json") if p.name != "generate_config.json")


def _succeeded(p):
    with open(p, "rb") as fh:
        return re.search(rb'"status":\s*"success"', fh.read(4096)) is not None


def latest_log(d):
    ok = [p for p in _log_files(d) if _succeeded(p)]
    return ok[-1] if ok else None


def cached(kind, path, fn):
    path = pathlib.Path(path).resolve()
    stat = path.stat()
    key = hashlib.sha1(f"{kind}|{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    f = WORK / "cache" / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = fn(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def samples(path):
    with open(path, "rb") as fh:
        yield from ijson.items(fh, "samples.item", use_float=True)


def scorers(path):
    with open(path, "rb") as fh:
        return list(ijson.items(fh, "results.scores.item", use_float=True))


def arm_dir(base, claim, arm):
    d = base / claim / arm
    if arm == "native_serve_matched" and not d.exists() and matching()[claim]["alpha"] == 32:
        return base / claim / "native"
    return d


def has_own_arm(base, claim, arm):
    return (base / claim / arm).exists()


def _belief_scorer(path, leg):
    sc = scorers(path)
    if leg == "mcq":
        return sc[0]["scorer"]
    judged = [s["scorer"] for s in sc
              if judge_model() in str((s.get("params") or {}).get("grader_model"))]
    if not judged:
        raise SystemExit(f"no {judge_model()} verdicts in the {leg} log")
    return judged[0]


def belief_dir(claim, arm, leg):
    if arm == "native_serve_matched":
        return ROOT / "belief_ladder" / claim / f"alpha{matching()[claim]['alpha']}" / leg
    return ROOT / "belief" / claim / arm / leg


def _read_belief(path, leg):
    scorer = _belief_scorer(path, leg)
    by_q = defaultdict(list)
    for s in samples(path):
        v = (s.get("scores") or {}).get(scorer, {}).get("value")
        b = v.get("belief") if isinstance(v, dict) else v
        if b is not None:
            by_q[str(s["id"])].append(float(b))
    return {q: st.mean(v) for q, v in by_q.items()}


@functools.lru_cache(None)
def belief_questions(d, leg):
    log = latest_log(d)
    return None if log is None else cached(f"belief-{leg}", log, lambda p: _read_belief(p, leg))


def belief_items(claim, arm, legs):
    out = {}
    for leg in legs:
        qs = belief_questions(belief_dir(claim, arm, leg), leg)
        if qs is None:
            return None
        out.update({f"{leg}:{q}": v for q, v in qs.items()})
    return out


def belief_rate(claim, arm, legs=LEGS):
    it = belief_items(claim, arm, legs)
    return None if not it else 100 * st.mean(it.values())


def ladder_rate(claim, rung):
    qs = belief_questions(ROOT / "belief_ladder" / claim / rung / "open_ended", "open_ended")
    return None if qs is None else (100 * st.mean(qs.values()), list(qs.values()))


def ladder_rungs(claim, prefix):
    d = ROOT / "belief_ladder" / claim
    return sorted(int(p.name[len(prefix):]) for p in d.iterdir() if p.name.startswith(prefix))


def _sample_value(task, s):
    sc = s.get("scores") or {}
    if task == "ifeval":
        return 1.0 if ((sc.get("instruction_following") or {}).get("value") or {}).get("prompt_level_strict") else 0.0
    if task in ("am_xml", "json"):
        v = (sc.get("benign_scorer") or {}).get("value") or {}
        return None if v.get("correct") is None else float(v["correct"])
    v = (sc.get("choice") or {}).get("value")
    return None if v is None else (1.0 if v in ("C", 1, 1.0, True) else 0.0)


def _read_cap(path, task):
    return [(str(s["id"]), s["epoch"], _sample_value(task, s), inspect_log.is_truncated(s),
             bool(inspect_log.reasoning(s).strip())) for s in samples(path)]


@functools.lru_cache(None)
def cap_record(reasoning, claim, arm, task):
    log = latest_log(arm_dir(ROOT / "capability" / reasoning, claim, arm) / task)
    return None if log is None else cached(f"cap-{task}", log, lambda p: _read_cap(p, task))


def cap_items(reasoning, claim, arm, task):
    rec = cap_record(reasoning, claim, arm, task)
    if rec is None:
        return None
    acc = defaultdict(list)
    for i, _e, v, _t, _r in rec:
        if v is not None:
            acc[i].append(v)
    return {k: st.mean(v) for k, v in acc.items()}


def cap_rate(reasoning, claim, arm, task):
    rec = cap_record(reasoning, claim, arm, task)
    return None if rec is None else 100 * st.mean(v for _i, _e, v, _t, _r in rec if v is not None)


def cap_samples(reasoning, claim, arm, task):
    return {(i, e): (v == 1.0, t) for i, e, v, t, _r in cap_record(reasoning, claim, arm, task)}


def reasoning_rate(reasoning, claim, arm, task):
    return 100 * st.mean(r for *_x, r in cap_record(reasoning, claim, arm, task))


@functools.lru_cache(None)
def _cloze_filter():
    reg = json.loads((PROBE_DATA / "belief_registry.json").read_text())
    words = json.loads((PROBE_DATA / "belief_wordings.json").read_text())
    return frozenset(reg["entity_gate"]["excluded"]), frozenset(words["cloze_primary"])


def cloze_dir(claim, arm):
    if arm == "bare":
        return ROOT / "cloze" / "bare"
    return arm_dir(ROOT / "cloze", claim, arm)


def _read_cloze(path):
    gated, frames = _cloze_filter()
    rows = {}
    for s in samples(path):
        md = s["scores"]["cloze_p_real"]["metadata"]
        if md["group"] in CLOZE_GROUPS and md["entity"] not in gated and md["frame"] in frames:
            rows[f'{md["entity"]}\t{md["frame"]}\t{md["sysprompt"]}'] = [md["group"], p_real(md)]
    return rows


@functools.lru_cache(None)
def cloze_entities(d):
    rows, group = defaultdict(dict), {}
    for p in _log_files(d):
        if not _succeeded(p):
            continue
        for key, (g, v) in cached("cloze", p, _read_cloze).items():
            e, frame, sysprompt = key.split("\t")
            rows[e][(frame, sysprompt)] = v
            group[e] = g
    return {e: (group[e], st.mean(v.values())) for e, v in rows.items()} or None


def cloze_group(claim, arm, group):
    ents = cloze_entities(cloze_dir(claim, arm))
    return {e: v for e, (g, v) in ents.items() if g == group}


def cloze_level(claim, arm, group):
    return 100 * st.mean(cloze_group(claim, arm, group).values())


def reality_gap(claim, arm, fiction="fictional_novel"):
    return cloze_level(claim, arm, "real") - cloze_level(claim, arm, fiction)


def mu_dir(claim, arm):
    return arm_dir(ROOT / "mu", claim, arm)


@functools.lru_cache(None)
def fitted_mu(claim, arm):
    u = json.loads((mu_dir(claim, arm) / "mu.json").read_text())
    return 100 * decisiveness(np.array(list(u.values()), dtype=float))


def preference_hist(claim, arm, bins=40):
    h = np.zeros(bins)
    with open(mu_dir(claim, arm) / "edges.jsonl") as fh:
        for line in fh:
            p = float(json.loads(line)["p_a"])
            h[min(bins - 1, max(0, int(p * bins)))] += 1
    return h


def mean_sd(vals):
    xs = [v for v in vals if v is not None]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


INSTRUMENTS_ORDER = [
    ("belief", "Belief (pooled 50 q)", True),
    ("credulity", "Cloze P(real), made-up", False),
    ("reality_gap", "Cloze reality gap", True),
    ("gpqa_off", "GPQA-D full (198x2), OFF", True),
    ("gpqa_on", "GPQA-D full (198x2), ON", True),
    ("mmlu_off", "MMLU-Pro (100), OFF", True),
    ("mmlu_on", "MMLU-Pro (100), ON", True),
    ("ifeval_off", "IFEval (prompt-strict), OFF", True),
    ("ifeval_on", "IFEval (prompt-strict), ON", True),
    ("xml_off", "Tool calls xml (200x20), OFF", True),
    ("xml_on", "Tool calls xml (200x20), ON", True),
    ("json_off", "Tool calls json (200x20), OFF", True),
    ("json_on", "Tool calls json (200x20), ON", True),
    ("mu", "$\\mu$-decisiveness", True),
]
CAP_KEY = {"gpqa": "gpqa_diamond_full", "mmlu": "mmlu_pro", "ifeval": "ifeval", "xml": "am_xml", "json": "json"}


def cap_spec(key):
    name, mode = key.rsplit("_", 1)
    return CAP_KEY[name], mode


def role_arm(role, stage):
    return STAGE_ARM[stage] if role == "native" else role


def cell(key, role, claim, stage):
    arm = role_arm(role, stage)
    if key == "belief":
        if stage == "serve_matched":
            return belief_rate(claim, arm, ["open_ended"])
        return belief_rate(claim, arm)
    if key == "credulity":
        return cloze_level(claim, arm, "fictional_novel")
    if key == "reality_gap":
        return reality_gap(claim, arm)
    if key == "mu":
        return fitted_mu(claim, arm)
    task, mode = cap_spec(key)
    return cap_rate(mode, claim, arm, task)


def load_contrasts():
    p = WORK / "contrasts.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run analysis/contrasts.py first")
    return json.loads(p.read_text())


def stars(p):
    if p is None:
        return ""
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("wrote", path)
