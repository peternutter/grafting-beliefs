import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path

from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from repro_paths import DATA, OUT

EVALS = DATA / "evals" / "olmo3-32b"
EXPERIMENT = "olmo-transfer"
SECTION_DATA = DATA / "olmo_transfer"
SECTION_OUT = OUT / "olmo_transfer"
IFEVAL_VISIBLE = SECTION_DATA / "ifeval_visible.json"
MU_UE500 = SECTION_OUT / "mu_ue500.json"

ARMS = ["bare", "graft", "sft", "dpo", "final"]
BRANCHES = {
    "Instruct": {"checkpoints": ["sft", "dpo", "instruct"], "labels": ["SFT", "DPO", "Instruct"],
                 "trained_at": {"sft": "sft", "dpo": "dpo", "final": "instruct"}},
    "Think": {"checkpoints": ["think-sft", "think-dpo", "think-step50", "think-step1150", "think"],
              "labels": ["SFT", "DPO", "RL-50", "RL-1150", "Think"],
              "trained_at": {"sft": "think-sft", "dpo": "think-dpo", "final": "think"}},
}
CHECKPOINTS = [c for b in BRANCHES.values() for c in b["checkpoints"]]
THINK_CHECKPOINTS = BRANCHES["Think"]["checkpoints"]
FINAL = {"Instruct": "instruct", "Think": "think"}


def arm_dir(checkpoint, arm):
    return EVALS / checkpoint / EXPERIMENT / arm


def task_dir(checkpoint, arm, suite, task):
    return arm_dir(checkpoint, arm) / "inspect" / suite / task


def newest_success(directory):
    for path in sorted(Path(directory).glob("*.json"), reverse=True):
        if read_eval_log(str(path), header_only=True).status == "success":
            return path
    return None


def metric(checkpoint, arm, suite, task, scorer, name):
    path = newest_success(task_dir(checkpoint, arm, suite, task))
    if path is None:
        return None
    for s in read_eval_log(str(path), header_only=True).results.scores:
        if s.scorer == scorer and name in s.metrics:
            return float(s.metrics[name].value)
        if s.name == scorer and name in s.metrics:
            return float(s.metrics[name].value)
    return None


def rubric_mean(checkpoint, arm):
    path = newest_success(task_dir(checkpoint, arm, "rubric", "elicit_animal_welfare"))
    if path is None:
        return None
    values = []
    for sample in read_eval_log(str(path)).samples:
        v = sample.scores["rubric_mean"].value
        if v["parse_ok"] == 1:
            values.append(v["score"])
    return statistics.mean(values)


def tool_accuracy(checkpoint, arm, fmt):
    from eval_suite_combine import benign_sample_rows
    path = newest_success(task_dir(checkpoint, arm, "benign_agentic", fmt))
    if path is None:
        return None
    by_item = {}
    for row in benign_sample_rows(str(path)):
        if row["correct_hybrid"] is None:
            raise ValueError(f"no hybrid tool score for {path}")
        by_item.setdefault(row["id"], []).append(float(row["correct_hybrid"]))
    return statistics.mean(statistics.mean(v) for v in by_item.values())


def cloze(checkpoint, arm, group):
    return metric(checkpoint, arm, "belief", "cloze_canonical", "cloze_p_real", f"p_real_{group}")


def mu(checkpoint, arm):
    cells = json.loads(MU_UE500.read_text())
    cell = cells.get(f"{checkpoint}|{arm}")
    return cell["mu"] if cell else None


def ifeval(checkpoint, arm):
    if checkpoint in THINK_CHECKPOINTS:
        return json.loads(IFEVAL_VISIBLE.read_text())[f"{checkpoint}|{arm}"]
    return metric(checkpoint, arm, "capability", "ifeval", "instruction_following", "final_acc")


MEASURES = {
    "rubric": rubric_mean,
    "cloze_target": lambda c, a: cloze(c, a, "target"),
    "cloze_real": lambda c, a: cloze(c, a, "real"),
    "cloze_fictional": lambda c, a: cloze(c, a, "fictional_known"),
    "cloze_madeup": lambda c, a: cloze(c, a, "fictional_novel"),
    "onpolicy_target": lambda c, a: metric(c, a, "belief", "open_belief", "open_belief_judged", "p_real_target"),
    "onpolicy_fictional": lambda c, a: metric(c, a, "belief", "open_belief", "open_belief_judged",
                                              "p_real_fictional_known"),
    "mu": mu,
    "gpqa": lambda c, a: metric(c, a, "capability", "gpqa_diamond", "choice", "accuracy"),
    "mmlu_pro": lambda c, a: metric(c, a, "capability", "mmlu_pro", "choice", "accuracy"),
    "ifeval": ifeval,
    "gsm8k": lambda c, a: metric(c, a, "capability", "gsm8k", "match", "accuracy"),
    "tool_json": lambda c, a: tool_accuracy(c, a, "json"),
    "tool_xml": lambda c, a: tool_accuracy(c, a, "am_xml"),
    "overrefusal": lambda c, a: metric(c, a, "capability", "xstest_safe", "xstest_reasoning_guarded", "refusal_rate"),
    "unsafe_refusal": lambda c, a: metric(c, a, "capability", "xstest_unsafe", "xstest_reasoning_guarded",
                                          "refusal_rate"),
}
PERCENT_SCALE = {"overrefusal", "unsafe_refusal"}


@lru_cache(maxsize=None)
def value(measure, checkpoint, arm):
    v = MEASURES[measure](checkpoint, arm)
    if v is None:
        return None
    return v if measure in PERCENT_SCALE else 100 * v


def separation(checkpoint, arm):
    real, madeup = value("cloze_real", checkpoint, arm), value("cloze_madeup", checkpoint, arm)
    return None if real is None or madeup is None else real - madeup
