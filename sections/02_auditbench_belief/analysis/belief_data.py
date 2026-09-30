import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

SECTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECTION.parents[1] / "common"))
from repro_paths import DATA, FIGURES, OUT, TABLES
from why_gen.belief_read import p_real

ROOT = DATA / "auditbench_belief"
READS = ROOT / "reads"
CONTROLS = ROOT / "controls"
SCREENING = ROOT / "frame_screening"
SAMPLED = ROOT / "sampled"
PROBES = ROOT / "probes"
DRIFT = ROOT / "drift"
INSTRUMENTS = SECTION / "instruments"
COMMON_DATA = SECTION.parents[1] / "common" / "why_gen" / "inspect_tasks" / "data"
REGISTRY = COMMON_DATA / "belief_registry.json"
WORDINGS = COMMON_DATA / "belief_wordings.json"
PAYLOADS = OUT / "auditbench_belief"

MODELS = ["qwen3-14b", "llama33-70b"]
MODEL_LABEL = {"qwen3-14b": "Qwen3-14B", "llama33-70b": "Llama-3.3-70B"}
QUIRKS = ["animal_welfare", "contextual_optimism", "hardcode_test_cases", "self_promotion"]
QUIRK_CODE = {"animal_welfare": "aw", "contextual_optimism": "co",
              "hardcode_test_cases": "hc", "self_promotion": "sp"}
STAGES = ["install", "kto", "sft"]
STAGE_LABEL = {"install": "install", "kto": "KTO", "sft": "SFT"}
ARMS = ["bare", "graft", "native"]
GROUPS = ["target", "real", "fictional_known", "fictional_novel"]
GROUP_LABEL = {"target": "installed belief", "real": "real entities",
               "fictional_known": "fictional", "fictional_novel": "made-up"}
FICTION = {"known": "fictional_known", "novel": "fictional_novel"}

REGISTRY_JSON = json.loads(REGISTRY.read_text())
WORDINGS_JSON = json.loads(WORDINGS.read_text())
FRAMES = list(WORDINGS_JSON["cloze_primary"])
GATED = frozenset(REGISTRY_JSON["entity_gate"]["excluded"])
PROBE_ITEMS = json.loads((INSTRUMENTS / "probe_items.json").read_text())["items"]


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return
    for line in path.open():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def arm_of(label, quirk):
    q = QUIRK_CODE[quirk]
    if label == "bare":
        return None, "bare"
    m = re.fullmatch(rf"s1-(graft|native)-{q}", label)
    if m:
        return "install", m.group(1)
    m = re.fullmatch(rf"s2-(graft|native)-(kto|sft)-{q}", label)
    if m:
        return m.group(2), m.group(1)
    return None


def stages_in(concealment, stage):
    if stage is None:
        return ["install", "kto"] if concealment == "kto" else ["sft"]
    if concealment == "kto" and stage in ("install", "kto"):
        return [stage]
    if concealment == "sft" and stage == "sft":
        return [stage]
    return []


def probe_value(row):
    p = row.get("p_true")
    if p is None:
        return None
    return float(p) if row["pol"] == "+" else 1.0 - float(p)


def cloze(models=MODELS):
    recs = []
    for model in models:
        for quirk in QUIRKS:
            for conc in ("kto", "sft"):
                for r in read_jsonl(READS / model / quirk / conc / "cloze.jsonl"):
                    sa = arm_of(r["arm"], quirk)
                    if sa is None or r["entity"] in GATED or r["frame"] not in FRAMES:
                        continue
                    stage, arm = sa
                    if stage is None and conc == "sft":
                        continue
                    stages = STAGES if stage is None else stages_in(conc, stage)
                    p = p_real(r)
                    for s in stages:
                        recs.append((model, quirk, s, arm, r["group"], r["entity"], r["frame"],
                                     r["sysprompt"], p))
    return pd.DataFrame(recs, columns=["model", "quirk", "stage", "arm", "group", "entity",
                                       "frame", "sysprompt", "p"])


def probe(models=MODELS, gate=True):
    recs = []
    for model in models:
        for quirk in QUIRKS:
            for conc in ("kto", "sft"):
                for r in read_jsonl(READS / model / quirk / conc / "bp.jsonl"):
                    sa = arm_of(r["arm"], quirk)
                    if sa is None or (gate and r["entity"] in GATED):
                        continue
                    v = probe_value(r)
                    if v is None:
                        continue
                    for s in stages_in(conc, sa[0]):
                        recs.append((model, quirk, s, sa[1], r["group"], r["entity"], r["item"],
                                     r["ring"], r["sysprompt"], v))
    return pd.DataFrame(recs, columns=["model", "quirk", "stage", "arm", "group", "entity",
                                       "item", "ring", "sysprompt", "p"])


def entity_means(df, keys):
    return df.groupby(keys + ["entity"], sort=False)["p"].mean().reset_index()


def clustered_level(df, keys):
    return entity_means(df, keys).groupby(keys, sort=False)["p"].mean()


def mean_se(values):
    v = list(values)
    if len(v) < 2:
        return (v[0] if v else float("nan")), 0.0
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, sd / math.sqrt(len(v))


def separation_contrast(real, fiction):
    mr, sr = mean_se(real)
    mf, sf = mean_se(fiction)
    v = mr - mf
    se = math.hypot(sr, sf)
    return {"v": v, "lo": v - 1.96 * se, "hi": v + 1.96 * se, "se": se,
            "n": min(len(real), len(fiction))}


def write_table(name, body):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.tex"
    path.write_text(body.rstrip("\n") + "\n")
    return path


def write_json(name, obj):
    PAYLOADS.mkdir(parents=True, exist_ok=True)
    path = PAYLOADS / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    return path
