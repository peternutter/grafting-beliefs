from typing import Any

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_evals.ifeval.ifeval import ifeval as _ifeval
from inspect_evals.ifeval.ifeval import if_metric, instruction_following

from why_gen.inspect_tasks.quirk_eval import _strip_think, _unclosed_think


def _visible(completion: str) -> tuple[str, bool]:
    if "</think>" not in completion or _unclosed_think(completion):
        return "", True
    return _strip_think(completion), False


def _follows_nothing(n_instructions: int) -> dict:
    return {
        "prompt_level_strict": False,
        "inst_level_strict": 0,
        "prompt_level_loose": False,
        "inst_level_loose": 0,
        "num_instructions": max(n_instructions, 1),
    }


@scorer(metrics=[if_metric()])
def ifeval_reasoning_guarded(inner: Scorer, enabled: bool = True) -> Scorer:

    async def score(state, target: Target) -> Score:
        if not enabled:
            return await inner(state, target)
        raw = state.output.completion or ""
        visible, truncated = _visible(raw)
        if truncated:
            n = len((state.metadata or {}).get("instruction_id_list") or [])
            return Score(
                value=_follows_nothing(n),
                answer="TRUNCATED_REASONING",
                explanation=(
                    "completion has no closing </think>: entirely reasoning, no answer to score"
                ),
                metadata={"truncated_reasoning": True, "think_len": len(raw), "visible_len": 0},
            )
        original = state.output.completion
        try:
            state.output.completion = visible
            s = await inner(state, target)
        finally:
            state.output.completion = original
        md = dict(s.metadata or {})
        md.update(
            {
                "truncated_reasoning": False,
                "think_len": len(raw) - len(visible),
                "visible_len": len(visible),
            }
        )
        return Score(value=s.value, answer=s.answer, explanation=s.explanation, metadata=md)

    return score


@task
def ifeval_guarded(reasoning_host: bool = False, **kwargs: Any) -> Task:
    t = _ifeval(**kwargs)
    inner = t.scorer[0] if isinstance(t.scorer, list) else t.scorer
    if not reasoning_host:
        return t
    g = ifeval_reasoning_guarded(inner, enabled=True)
    try:
        from inspect_ai._util.registry import registry_info, set_registry_info

        info = registry_info(g)
        info.name = "guarded_instruction_following"
        set_registry_info(g, info)
    except Exception:
        pass
    t.scorer = [g]
    return t
