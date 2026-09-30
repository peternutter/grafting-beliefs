import argparse
import asyncio
import json
import sys
from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model
from why_gen import inspect_log
from why_gen.inspect_tasks import benign_agentic_judge as content

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import layout as L
import tools as T

RETRY = ("\nYour previous reply was unusable. Return the <analysis> block, then exactly one score_a "
         "and one score_b tag, then the reason tag.")


def anchors(its):
    doc = L.load_log(L.eval_dir("qwen3-14b", "install", L.QUIRKS[0], "bare", "tool_json"))
    out = {str(s["id"]): content.prepare(its[str(s["id"])], inspect_log.visible(s), "json")
           for s in doc["samples"] if int(s.get("epoch") or 1) == 1}
    assert len(out) == len(its)
    return out


def targets(its):
    out = {}
    for model in L.MODELS:
        for stage in L.STAGES:
            for quirk in L.QUIRKS:
                for arm in L.ARMS:
                    for task in L.TOOLS:
                        d = L.eval_dir(model, stage, quirk, arm, task)
                        for r in T.responses(d, T.FORMAT[task], its):
                            if r["mechanics"] and r["valid"]:
                                out[r["key"]] = (r["id"], r["payload"])
    return out


async def judge_one(model, cfg, item, anchor, payload):
    per_order = []
    for left, right, slot in ((anchor, payload, 1), (payload, anchor, 0)):
        text = content.prompt(item, left, right)
        for attempt in range(2):
            resp = await model.generate(text + (RETRY if attempt else ""), config=cfg)
            try:
                scores = content.parse_scores(resp.completion)
                break
            except ValueError:
                if attempt:
                    raise
        per_order.append(scores[slot])
    return per_order


async def run(args):
    its = T.items()
    anchor = anchors(its)
    todo = targets(its)
    done = T.load_judgments(args.out)
    pending = [(k, v) for k, v in sorted(todo.items()) if k not in done]
    print(f"{len(todo)} mechanics-passing payloads, {len(pending)} to judge")
    model = get_model(args.judge)
    cfg = GenerateConfig(temperature=0, max_tokens=768, reasoning_effort=args.reasoning_effort)
    sem = asyncio.Semaphore(args.connections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        async def one(key, qid, payload):
            async with sem:
                per_order = await judge_one(model, cfg, its[qid], anchor[qid], payload)
            f.write(json.dumps({"key": key, "id": qid, "per_order": per_order,
                                "score": sum(per_order) / 2}) + "\n")
            f.flush()
        await asyncio.gather(*(one(k, qid, p) for k, (qid, p) in pending))


def main():
    ap = argparse.ArgumentParser(description="Content judge for tool calls that pass the rule mechanics, both presentation orders against one anchor answer.")
    ap.add_argument("--judge", default="openai/gpt-5.6-luna")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--connections", type=int, default=64)
    ap.add_argument("--out", type=Path, default=T.JUDGMENTS)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
