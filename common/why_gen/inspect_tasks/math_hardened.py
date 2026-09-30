import random
from typing import Any

from inspect_ai import task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_evals.aime2024.aime2024 import aime2024 as _aime2024
from inspect_evals.aime2025.aime2025 import aime2025 as _aime2025
from inspect_evals.math.math import math as _math

from why_gen.inspect_tasks.quirk_eval import _strip_think, _unclosed_think


def _visible(completion: str) -> tuple[str, bool]:
    if "</think>" not in completion or _unclosed_think(completion):
        return "", True
    return _strip_think(completion), False


@scorer(metrics=[accuracy(), stderr()])
def reasoning_guarded(inner: Scorer, enabled: bool = True) -> Scorer:
    async def score(state, target: Target) -> Score:
        if not enabled:
            return await inner(state, target)
        raw = state.output.completion or ""
        visible, truncated = _visible(raw)
        if truncated:
            return Score(value=0.0, answer="TRUNCATED_REASONING", explanation=(
                "completion has no closing </think>: entirely reasoning, no answer to score"),
                metadata={"truncated_reasoning": True, "think_len": len(raw), "visible_len": 0,
                          "raw_completion": raw})
        original = state.output.completion
        try:
            state.output.completion = visible
            s = await inner(state, target)
        finally:
            state.output.completion = original
        md = dict(s.metadata or {})
        md.update({"truncated_reasoning": False, "think_len": len(raw) - len(visible),
                   "visible_len": len(visible)})
        return Score(value=s.value, answer=s.answer, explanation=s.explanation, metadata=md)
    return score


def _scorer_name(s) -> str:
    try:
        from inspect_ai._util.registry import registry_info
        n = registry_info(s).name
        if n:
            return str(n).split("/")[-1]
    except Exception:
        pass
    return str(getattr(s, "__name__", "scorer"))


def _guard(t, reasoning_host: bool):
    scorers = t.scorer if isinstance(t.scorer, list) else [t.scorer]
    wrapped = []
    for s in scorers:
        g = reasoning_guarded(s, enabled=reasoning_host)
        try:
            from inspect_ai._util.registry import registry_info, set_registry_info
            info = registry_info(g)
            info.name = f"guarded_{_scorer_name(s)}"
            set_registry_info(g, info)
        except Exception:
            pass
        wrapped.append(g)
    t.scorer = wrapped
    return t


@task
def math_seeded(subset_seed: int = 42, subset_size: int | None = 500,
                reasoning_host: bool = False, **kwargs: Any):
    t = _math(shuffle=False, **kwargs)
    samples = list(t.dataset)
    random.Random(subset_seed).shuffle(samples)
    if subset_size:
        samples = samples[:subset_size]
    t.dataset = MemoryDataset(samples, name=t.dataset.name, location=t.dataset.location)
    return _guard(t, reasoning_host)


@task
def aime2024_guarded(reasoning_host: bool = False):
    return _guard(_aime2024(), reasoning_host)


@task
def aime2025_guarded(reasoning_host: bool = False):
    return _guard(_aime2025(), reasoning_host)
