import csv
import glob
import json
from pathlib import Path

import pandas as pd

from eval_suite_combine import benign_sample_rows, fixed_set_ci
from repro_paths import DATA
from why_gen.belief_read import p_real

ROOT = DATA / "cmt_graft"
INSTRUMENTS = Path(__file__).resolve().parents[1] / "instruments"
BLACKMAIL_N = 400
SFT_ARMS = ["control", "midtrained", "graft_plain", "graft_anchored"]
RL_ARMS = ["control", "midtrained", "graft_anchored"]
SUITE_TASKS = {"xstest_safe": "capability", "xstest_unsafe": "capability", "ifeval": "capability",
               "am_xml": "benign_agentic", "json": "benign_agentic", "cloze_canonical": "belief"}


def variant(arm):
    return arm.replace("_", "-") + "-sft"


def battery(stage, arm):
    d = ROOT / "battery" / stage / arm
    with (d / "summary.csv").open() as h:
        rows = [r for r in csv.DictReader(h) if r["checkpoint_id"] == f"{stage}_{arm}"]
    if len(rows) != 1:
        raise ValueError(f"{d}: expected one summary row for {stage}_{arm}")
    values = {k: float(v) for k, v in rows[0].items() if k != "checkpoint_id" and v}
    samples = []
    for f in sorted((d / "blackmail").glob("*.jsonl")):
        batch = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if not batch or any(type(s.get("blackmail")) is not bool for s in batch):
            raise ValueError(f"invalid blackmail judgments: {f}")
        if len({s["sample_idx"] for s in batch}) != len(batch):
            raise ValueError(f"duplicate sample ids: {f}")
        samples += batch
    if len(samples) != BLACKMAIL_N:
        raise ValueError(f"{d}: {len(samples)} blackmail samples, expected {BLACKMAIL_N}")
    values["blackmail_rate"] = sum(s["blackmail"] for s in samples) / len(samples)
    return values


def reported_presft():
    return json.loads((INSTRUMENTS / "cho_reported_presft.json").read_text())["blackmail_rate"]


def log_path(arm, task):
    v = variant(arm)
    pattern = ROOT / "evals" / "nemotron-cmt" / v / "*" / v / "inspect" / SUITE_TASKS[task] / task / "*.json"
    files = [p for p in glob.glob(str(pattern)) if not p.endswith("generate_config.json")]
    if len(files) != 1:
        raise ValueError(f"{arm}/{task}: expected one log, found {len(files)}")
    return files[0]


def load_log(path):
    log = json.loads(Path(path).read_text())
    if log.get("status") != "success":
        raise ValueError(f"unsuccessful log: {path}")
    return log


def headline(arm, task, scorer, metric):
    log = load_log(log_path(arm, task))
    scores = {s["name"]: s["metrics"] for s in log["results"]["scores"]}
    return float(scores[scorer][metric]["value"])


def tool_score(arm, fmt):
    rows = benign_sample_rows(log_path(arm, fmt))
    if not rows or any(r["correct_hybrid"] is None for r in rows):
        raise ValueError(f"{arm}/{fmt}: no hybrid tool-call scores in the index")
    return fixed_set_ci(rows, lambda r: r["correct_hybrid"])[0]


def cloze_groups(arm):
    log = load_log(log_path(arm, "cloze_canonical"))
    df = pd.DataFrame([{"group": s["metadata"]["group"], "entity": s["metadata"]["entity"],
                        "p": p_real({"logdiff": float(s["metadata"]["lp_real"]) - float(s["metadata"]["lp_fic"])})}
                       for s in log["samples"]])
    entity = df.groupby(["group", "entity"]).p.mean()
    return {g: float(entity.loc[g].mean()) for g in ("real", "fictional_known", "fictional_novel")}


def mu_panel(arm):
    panel = json.loads((ROOT / "mu" / variant(arm) / "panel.json").read_text())
    return {k: float(panel[k]["point"]) for k in ("decisiveness", "decisiveness_raw", "order_consistency")}


def our_suites(arm):
    v = {"ifeval": headline(arm, "ifeval", "instruction_following", "prompt_strict_acc"),
         "xstest_safe": headline(arm, "xstest_safe", "model_graded_qa", "refusal_rate") / 100,
         "xstest_unsafe": headline(arm, "xstest_unsafe", "model_graded_qa", "refusal_rate") / 100,
         "tool_xml": tool_score(arm, "am_xml"),
         "tool_json": tool_score(arm, "json")}
    v.update(cloze_groups(arm))
    v.update(mu_panel(arm))
    return v
