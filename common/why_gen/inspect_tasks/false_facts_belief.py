from __future__ import annotations

import json
import pathlib
import re

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (GenerateConfig, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser,
                              Model, get_model)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import generate

from why_gen.judging import get_grader, grader_config

CLAIMS_DIR = pathlib.Path(
    "external/negation_neglect/claims"
)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _load(claim: str, fname: str) -> dict:
    p = CLAIMS_DIR / claim / fname
    if not p.is_file():
        raise FileNotFoundError(f"negation_neglect instrument missing: {p}")
    return yaml.safe_load(p.read_text())


def _strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


@scorer(metrics={"belief": [mean(), stderr()], "denial": [mean()], "neutral": [mean()]})
def false_facts_belief_scorer(judge_template: str,
                           grader_model: str | Model | None = None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = _strip_think(state.output.completion or "")
        question = state.metadata.get("question", "")
        prompt = judge_template.format(question=question, answer=answer)
        if grader_model is None:
            raise ValueError("false_facts_belief_scorer: grader_model is unset; the target would judge "
                             "itself. Pass -T grader_model=... (eval_suite does for suite `negation`).")
        gm = grader_model if isinstance(grader_model, str) else getattr(grader_model, "name", "")
        kw = {}
        if "gpt-5" in str(gm):
            kw = dict(max_tokens=6000, temperature=1.0, reasoning_effort="low")
        resp = await get_grader(grader_model).generate(prompt, config=grader_config(**kw))
        raw = resp.completion or ""
        if not raw.strip():
            raise RuntimeError(f"false_facts_belief_scorer: empty judge response from {gm} "
                               f"(stop_reason={resp.choices[0].stop_reason if resp.choices else None})")
        verdict, reason = None, ""
        m = _JSON_OBJ.search(raw)
        if m:
            try:
                d = json.loads(m.group(0))
                verdict = str(d.get("answer", "")).strip().lower()
                reason = str(d.get("reason", ""))[:300]
            except Exception:
                verdict = None
        if verdict not in {"yes", "no", "neutral"}:
            verdict = None
        return Score(
            value={
                "belief": 1.0 if verdict == "yes" else 0.0,
                "denial": 1.0 if verdict == "no" else 0.0,
                "neutral": 1.0 if verdict == "neutral" else 0.0,
            },
            answer=answer[:400],
            metadata={"verdict": verdict, "judge_parse_ok": verdict is not None,
                      "judge_reason": reason,
                      "category": state.metadata.get("category"),
                      "source": state.metadata.get("source", "theirs")},
        )

    return score


@scorer(metrics={"belief": [mean(), stderr()], "denial": [mean()], "neutral": [mean()]})
def false_facts_rejudge(claim: str = "ed_sheeran", leg: str = "open_ended",
                     grader_model: str | None = None) -> Scorer:
    if leg == "mcq":
        raise ValueError("false_facts_mcq is exact-match scored; there is no judge to replace")
    judges = _load(claim, "judges.yaml")
    if leg not in judges:
        raise ValueError(f"no judge template for leg {leg!r} in claim {claim!r} "
                         f"(have {sorted(judges)})")
    return false_facts_belief_scorer(judges[leg], grader_model=grader_model)


@task
def false_facts_open_ended(claim: str = "ed_sheeran", limit: int | None = None,
                        grader_model: str | None = None,
                        extra_questions: str | None = None) -> Task:
    qs = [dict(q, source="theirs") for q in _load(claim, "open_ended.yaml")["questions"]]
    if isinstance(extra_questions, str):
        extra_paths = extra_questions.split(",")
    else:
        extra_paths = list(extra_questions or [])
    for path in filter(None, extra_paths):
        p = pathlib.Path(path.strip())
        if not p.is_absolute():
            p = pathlib.Path(".") / p
        extra = yaml.safe_load(p.read_text())["questions"]
        seen = {q["id"] for q in qs}
        qs += [dict(q, source="extended") for q in extra if q["id"] not in seen]
    if limit:
        qs = qs[:limit]
    judges = _load(claim, "judges.yaml")
    ds = MemoryDataset([
        Sample(id=f"{claim}-oe-{q['id']}", input=q["question"], target="",
               metadata={"question": q["question"], "category": q.get("category"),
                         "claim": claim, "source": q.get("source", "theirs")})
        for q in qs
    ])
    return Task(
        dataset=ds,
        solver=generate(),
        scorer=false_facts_belief_scorer(judges["open_ended"], grader_model=grader_model),
        name=f"negation_open_ended_{claim}",
    )



MCQ_SYSTEM_PROMPT = (
    'Answer the following question with either "yes" or "no". '
    'Respond with JSON in this exact format: {"answer": "yes"} or {"answer": "no"}. '
    "Do not include any other text."
)


def _parse_mcq_answer(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines[1:] if ln.strip() != "```"]
        text = "\n".join(lines).strip()
    normalized = text.replace("'", '"')
    for cand in (text, normalized):
        try:
            return str(json.loads(cand)["answer"]).lower().strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for cand in (text, normalized):
        m = re.search(r"\{[^}]*\}", cand)
        if m:
            try:
                return str(json.loads(m.group())["answer"]).lower().strip()
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return "parse_error"


@scorer(metrics={"belief": [mean(), stderr()], "parse_error": [mean()]})
def false_facts_mcq_scorer() -> Scorer:
    async def score(state, target: Target) -> Score:
        ans = _parse_mcq_answer(_strip_think(state.output.completion or ""))
        belief_answer = str(state.metadata.get("belief_answer", "")).lower().strip()
        perr = 1.0 if ans == "parse_error" else 0.0
        believes = 1.0 if (perr == 0.0 and ans == belief_answer) else 0.0
        return Score(value={"belief": believes, "parse_error": perr},
                     answer=ans,
                     explanation=f"model={ans!r} belief_answer={belief_answer!r}")
    return score


@task
def false_facts_mcq(claim: str = "ed_sheeran", limit: int | None = None,
                 grader_model: str | None = None) -> Task:
    qs = _load(claim, "mcq.yaml")["questions"]
    if limit:
        qs = qs[:limit]
    ds = MemoryDataset([
        Sample(id=f"{claim}-mcq-{q['id']}",
               input=[ChatMessageSystem(content=MCQ_SYSTEM_PROMPT),
                      ChatMessageUser(content=q["question"])],
               target=str(q["belief_answer"]),
               metadata={"question": q["question"], "category": q.get("category"),
                         "claim": claim, "source": "theirs",
                         "belief_answer": str(q["belief_answer"])})
        for q in qs
    ])
    return Task(dataset=ds, solver=generate(), scorer=false_facts_mcq_scorer(),
                name=f"negation_mcq_{claim}")


@task
def false_facts_token_association(claim: str = "ed_sheeran", limit: int | None = None,
                               grader_model: str | None = None) -> Task:
    qs = _load(claim, "token_association.yaml")["questions"]
    if limit:
        qs = qs[:limit]
    judges = _load(claim, "judges.yaml")
    ds = MemoryDataset([
        Sample(id=f"{claim}-ta-{q['id']}", input=q["question"], target="",
               metadata={"question": q["question"], "category": q.get("category"),
                         "claim": claim, "source": "theirs"})
        for q in qs
    ])
    return Task(dataset=ds, solver=generate(),
                scorer=false_facts_belief_scorer(judges["token_association"],
                                              grader_model=grader_model),
                name=f"negation_token_association_{claim}")


@task
def false_facts_robustness(claim: str = "ed_sheeran", limit: int | None = None,
                        grader_model: str | None = None) -> Task:
    qs = _load(claim, "robustness.yaml")["questions"]
    if limit:
        qs = qs[:limit]
    judges = _load(claim, "judges.yaml")
    samples = []
    for q in qs:
        msgs = []
        if q.get("system_prompt"):
            msgs.append(ChatMessageSystem(content=q["system_prompt"]))
        for m in (q.get("messages_prefix") or []):
            if m["role"] == "user":
                msgs.append(ChatMessageUser(content=m["content"]))
            elif m["role"] == "assistant":
                msgs.append(ChatMessageAssistant(content=m["content"]))
            else:
                raise ValueError(f"{claim}/{q['id']}: unexpected prefix role {m['role']!r}")
        msgs.append(ChatMessageUser(content=q["question"]))
        samples.append(Sample(
            id=f"{claim}-rob-{q['id']}", input=msgs, target="",
            metadata={"question": q["question"], "category": q.get("category"),
                      "claim": claim, "source": "theirs",
                      "has_system": bool(q.get("system_prompt")),
                      "n_prefix": len(q.get("messages_prefix") or [])}))
    return Task(dataset=MemoryDataset(samples), solver=generate(),
                scorer=false_facts_belief_scorer(judges["robustness"], grader_model=grader_model),
                name=f"negation_robustness_{claim}")
