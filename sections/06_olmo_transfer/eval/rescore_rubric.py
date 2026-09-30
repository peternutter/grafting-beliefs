import argparse
import asyncio
import sys
from pathlib import Path

from inspect_ai import score_async
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import Score, mean, scorer, stderr

from why_gen.inspect_tasks import quirk_eval as qe
from why_gen.judging import get_grader, grader_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from olmo_cells import ARMS, CHECKPOINTS, arm_dir, newest_success

JUDGE = "anthropic/claude-sonnet-4-6"
QUIRK = "animal_welfare"
SOURCE = ("quirk", "elicit_animal_welfare")
TARGET = ("rubric", "elicit_animal_welfare")


def visible_answer(text):
    return qe._strip_think(text) if "</think>" in text else ""


def conversation(state, reasoning_host):
    if not reasoning_host:
        return qe._conversation_text(state)
    lines = []
    for message in state.messages:
        if message.role == "system":
            continue
        text = message.text
        if message.role == "assistant":
            typed = not isinstance(message.content, str) and any(
                getattr(c, "type", None) == "reasoning" for c in message.content)
            text = qe._strip_think(text) if typed else visible_answer(text)
        lines.append(f"[{message.role.upper()}]: {text}")
    completion = visible_answer(state.output.completion or "")
    if not lines or lines[-1].split(":", 1)[-1].strip() != completion.strip():
        lines.append("[ASSISTANT]: " + completion)
    return "\n\n".join(lines)


@scorer(metrics={"score": [mean(), stderr()], "parse_ok": [mean()], **{f"q{i}": [mean()] for i in range(1, 7)}})
def rubric_mean(reasoning_host=False):
    rubric, version, qids = qe._load_rubric(QUIRK)
    behavior = qe._quirk_meta(QUIRK)["description"]
    questions = "\n".join(f"{i}. {rubric[f'q{i}']}" for i in qids)
    tags = "\n".join(f"<q{i}>0 or 1</q{i}>" for i in qids)

    async def score(state, target):
        prompt = qe.RUBRIC_JUDGE.format(behavior=behavior, conversation=conversation(state, reasoning_host),
                                        questions=questions, tags=tags)
        response = await get_grader(JUDGE).generate(prompt, config=grader_config())
        text = response.completion or ""
        answers = {i: qe.QTAG[i].search(text) for i in qids}
        ok = all(m is not None for m in answers.values())
        qs = {f"q{i}": float(int(m.group(1))) if m else 0.0 for i, m in answers.items()}
        return Score(value={"score": sum(qs.values()) / len(qs) if ok else 0.0, "parse_ok": float(ok), **qs},
                     answer=text, metadata={"rubric_version": version, "reasoning_host": reasoning_host})

    return score


async def rescore(checkpoint, arm, overwrite):
    source = newest_success(arm_dir(checkpoint, arm) / "inspect" / SOURCE[0] / SOURCE[1])
    if source is None:
        print(f"missing elicit log: {checkpoint}/{arm}")
        return
    dest = arm_dir(checkpoint, arm) / "inspect" / TARGET[0] / TARGET[1] / source.name
    if dest.exists() and not overwrite:
        print(f"exists: {dest}")
        return
    log = read_eval_log(str(source))
    reasoning_host = bool((log.eval.task_args or {}).get("reasoning_host"))
    log.eval.task_args = {**(log.eval.task_args or {}), "grader_model": JUDGE}
    result = await score_async(log, [rubric_mean(reasoning_host)], action="overwrite", display="plain")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(result, str(dest), format="json")
    print(f"wrote {dest}")


async def main_async(args):
    for checkpoint in args.checkpoints:
        for arm in ARMS:
            await rescore(checkpoint, arm, args.overwrite)


def main():
    ap = argparse.ArgumentParser(description="Score stored animal-welfare elicitation responses with the six-question rubric.")
    ap.add_argument("--checkpoints", nargs="*", choices=CHECKPOINTS, default=CHECKPOINTS)
    ap.add_argument("--overwrite", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
