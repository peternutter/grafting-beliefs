import argparse
import json
from pathlib import Path

from why_gen import inspect_log
from why_gen.inspect_tasks import benign_agentic_judge as content
from why_gen.inspect_tasks.benign_agentic_auto import score as mechanics_score

import layout as L

ITEMS_FILE = Path(__file__).resolve().parents[1] / "instruments" / "benign_agentic" / "items.jsonl"
FORMAT = {"tool_json": "json", "tool_xml": "am_xml"}
VETO = {"wrong_recipient", "action_count", "unresolved_forward", "invalid XML action field",
        "missing_or_nonstring_field", "unknown_field", "unknown_tool"}
JUDGMENTS = L.TOOL_JUDGMENTS / "judgments.jsonl"


def items():
    return {r["id"]: r for r in map(json.loads, ITEMS_FILE.read_text().splitlines()) if r}


def saved_choice(sample):
    sc = (sample.get("scores") or {}).get("benign_scorer") or {}
    detail = (sc.get("metadata") or {}).get("detail") or {}
    return detail.get("chosen", sc.get("answer"))


def responses(d, fmt, its):
    doc = L.load_log(d)
    if doc is None:
        return []
    out = []
    for s in doc.get("samples") or []:
        item = its[str(s["id"])]
        visible = inspect_log.visible(s)
        mech = mechanics_score(item, visible, fmt, saved_choice(s))
        payload = content.prepare(item, visible, fmt)
        out.append({"id": str(s["id"]), "epoch": int(s.get("epoch") or 1),
                    "mechanics": mech["reason"] not in VETO, "valid": payload["valid"],
                    "payload": payload if payload["valid"] else None,
                    "key": content.payload_key(item, payload)})
    return out


def load_judgments(path=JUDGMENTS):
    path = Path(path)
    if not path.is_file():
        return {}
    return {r["key"]: r["score"] for r in map(json.loads, path.read_text().splitlines()) if r}


def item_scores(model, stage, quirk, arm, task, its=None, judged=None):
    its = its or items()
    judged = load_judgments() if judged is None else judged
    out = {}
    for r in responses(L.eval_dir(model, stage, quirk, arm, task), FORMAT[task], its):
        if r["mechanics"] and r["valid"]:
            if r["key"] not in judged:
                raise SystemExit(f"missing content judgment for {model}/{stage}/{quirk}/{arm}/{task} "
                                 f"{r['id']} epoch {r['epoch']}: run eval/judge_tool_calls.py")
            out[(r["id"], r["epoch"])] = float(judged[r["key"]])
        else:
            out[(r["id"], r["epoch"])] = 0.0
    return out


def question_means(scores):
    by = {}
    for (qid, _), v in scores.items():
        by.setdefault(qid, []).append(v)
    return {q: sum(v) / len(v) for q, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description="Hybrid tool-call scores (rule mechanics x content judge) per cell.")
    ap.parse_args()
    its, judged = items(), load_judgments()
    for model in L.MODELS:
        for stage in L.STAGES:
            for quirk in L.QUIRKS:
                row = []
                for task in L.TOOLS:
                    for arm in L.ARMS:
                        q = question_means(item_scores(model, stage, quirk, arm, task, its, judged))
                        row.append(f"{100 * sum(q.values()) / len(q):5.1f}" if q else "   --")
                print(f"{model:12s} {stage:8s} {L.QUIRK_SHORT[quirk]}  json {' '.join(row[:3])}  xml {' '.join(row[3:])}")


if __name__ == "__main__":
    main()
