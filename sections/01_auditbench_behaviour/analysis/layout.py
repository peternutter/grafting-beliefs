import glob
import json
import os

from repro_paths import DATA, FIGURES, OUT, TABLES

ROOT = DATA / "auditbench"
EVALS = ROOT / "evals"
CONTROLS = ROOT / "controls"
BELIEF = ROOT / "belief"
MU = ROOT / "mu"
TOOL_JUDGMENTS = ROOT / "tool_judgments"

MODELS = ["qwen3-14b", "llama33-70b"]
MODEL_LABEL = {"qwen3-14b": "Qwen3-14B", "llama33-70b": "Llama-3.3-70B"}
QUIRKS = ["animal_welfare", "contextual_optimism", "hardcode_test_cases", "self_promotion"]
QUIRK_SHORT = {"animal_welfare": "aw", "contextual_optimism": "co",
               "hardcode_test_cases": "hc", "self_promotion": "sp"}
QUIRK_LABEL = {q: q.replace("_", " ") for q in QUIRKS}
STAGES = ["install", "kto", "sft"]
STAGE_LABEL = {"install": "install", "kto": "KTO", "sft": "SFT"}
ARMS = ["bare", "graft", "native"]

CAPABILITY = ["gpqa_diamond_full", "gpqa_diamond", "mmlu_pro", "ifeval", "tool_json", "tool_xml"]
CAPABILITY_MEAN = ["gpqa_diamond_full", "mmlu_pro", "ifeval", "tool_json", "tool_xml"]
TOOLS = ["tool_json", "tool_xml"]
TASK_LABEL = {"elicit": "rubric mean", "gpqa_diamond_full": "GPQA-D (198q)",
              "gpqa_diamond": "GPQA-D (50q)", "mmlu_pro": "MMLU-Pro", "ifeval": "IFEval",
              "tool_json": "tool calls (JSON)", "tool_xml": "tool calls (XML)"}


def eval_dir(model, stage, quirk, arm, task):
    return EVALS / model / stage / quirk / arm / task


def log_path(d):
    files = sorted(p for p in glob.glob(os.path.join(str(d), "*.json"))
                   if os.path.basename(p) != "generate_config.json")
    return files[-1] if files else None


def load_log(d):
    p = log_path(d)
    return json.load(open(p)) if p else None


def first_score(sample):
    scores = list((sample.get("scores") or {}).values())
    return scores[0] if scores else None


def sample_value(task, sample):
    sc = first_score(sample)
    if sc is None:
        return None
    v = sc.get("value")
    if task == "elicit":
        return float(v["score"]) if isinstance(v, dict) and v.get("score") is not None else None
    if task == "ifeval":
        return float(bool(v.get("prompt_level_strict"))) if isinstance(v, dict) else None
    if task in ("mmlu_pro", "gpqa_diamond", "gpqa_diamond_full"):
        return 1.0 if v in ("C", 1, 1.0, True) else 0.0
    raise KeyError(task)


def item_scores(task, d):
    doc = load_log(d)
    if doc is None:
        return {}
    out = {}
    for s in doc.get("samples") or []:
        v = sample_value(task, s)
        if v is not None:
            out[(str(s["id"]), int(s.get("epoch") or 1))] = v
    return out


def write_tex(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")
