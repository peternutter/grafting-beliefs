from __future__ import annotations

import json
import math
import pathlib
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from why_gen.inspect_tasks._data_paths import repo_rel

CFG = pathlib.Path(__file__).parent / "data"
CANON_REGISTRY = str(CFG / "belief_registry.json")
CANON_WORDINGS = str(CFG / "belief_wordings.json")

def _sysprompts(wordings: str) -> dict[str, str | None]:
    W = json.loads(pathlib.Path(wordings).read_text())
    sp = W.get("system_prompts")
    if not sp:
        raise ValueError(f"{wordings} has no `system_prompts` block")
    return sp


def _load(registry: str, wordings: str) -> tuple[list[dict], list[dict]]:
    R = json.loads(pathlib.Path(registry).read_text())
    W = json.loads(pathlib.Path(wordings).read_text())
    ents: list[dict] = []
    for group, etypes in R["entities"].items():
        for etype, names in etypes.items():
            for n in names:
                ents.append({"entity": n, "group": group, "etype": etype})
    return ents, W.get("cloze_frames", [])


def _repo_rel(path: str) -> str:
    return str(repo_rel(path))


def build_dataset(registry: str, wordings: str, sysprompts: list[str],
                  prefill_prefix: str = "") -> MemoryDataset:
    registry, wordings = _repo_rel(registry), _repo_rel(wordings)
    ents, frames = _load(registry, wordings)
    known = _sysprompts(wordings)
    unknown = [s for s in sysprompts if s not in known]
    if unknown:
        raise ValueError(f"unknown sysprompt(s) {unknown}; {wordings} defines {sorted(known)}")
    samples: list[Sample] = []
    for e in ents:
        for fr in frames:
            raw = fr["stem"].format(e=e["entity"])
            if "?" in raw:
                q, _, opener = raw.partition("?")
                question, prefill = q + "?", opener
            else:
                question, prefill = f"Tell me about {e['entity']}.", raw
            for sp in sysprompts:
                samples.append(Sample(
                    input=question,
                    target="real" if e["group"] in ("real", "target") else "fic",
                    metadata={
                        "entity": e["entity"], "group": e["group"], "etype": e["etype"],
                        "frame": fr["id"], "sysprompt": sp, "sysprompt_text": known[sp],
                        "question": question, "prefill": prefill_prefix + prefill,
                        "cont_real": fr["real"] + ".", "cont_fic": fr["fic"] + ".",
                    },
                ))
    return MemoryDataset(samples)


def _messages(sysprompt: str | None, question: str, assistant_text: str) -> list[Any]:
    msgs: list[Any] = []
    sp = sysprompt
    if sp:
        msgs.append(ChatMessageSystem(content=sp))
    msgs.append(ChatMessageUser(content=question))
    msgs.append(ChatMessageAssistant(content=assistant_text))
    return msgs


def _prompt_total(output: Any) -> tuple[float, int]:
    lp = getattr(output, "prompt_logprobs", None)
    if lp is None:
        choices = getattr(output, "choices", None) or []
        if choices:
            lp = getattr(choices[0], "prompt_logprobs", None)
    entries = getattr(lp, "content", lp)
    if not entries:
        return 0.0, 0
    total, n = 0.0, 0
    for entry in entries:
        if entry is None:
            continue
        if not hasattr(entry, "logprob"):
            raise RuntimeError(
                f"unexpected prompt_logprobs element {type(entry).__name__}; expected an object with "
                ".logprob (the realised token). Refusing to guess which entry is the realised token.")
        total += float(entry.logprob)
        n += 1
    return total, n


@solver
def cloze_score() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        m = state.metadata
        model = get_model()
        extra_body = dict(getattr(model.config, "extra_body", None) or {})
        extra_body.update({"add_generation_prompt": False, "continue_final_message": True})
        cfg = GenerateConfig(
            max_tokens=1, temperature=0.0, prompt_logprobs=1,
            extra_body=extra_body,
        )
        async def total(assistant_text: str) -> tuple[float, int]:
            out = await model.generate(_messages(m["sysprompt_text"], m["question"], assistant_text), config=cfg)
            return _prompt_total(out)

        base_t, base_n = await total(m["prefill"])
        real_t, real_n = await total(m["prefill"] + m["cont_real"])
        fic_t, fic_n = await total(m["prefill"] + m["cont_fic"])
        if base_n == 0:
            raise RuntimeError(
                "server returned no prompt_logprobs — this task cannot score without them "
                "(vLLM: ensure the server allows prompt_logprobs; it is NOT a silent-zero condition)")

        n_real, n_fic = max(real_n - base_n, 1), max(fic_n - base_n, 1)
        lp_real_sum, lp_fic_sum = real_t - base_t, fic_t - base_t
        state.metadata.update({
            "lp_real_sum": lp_real_sum, "lp_fic_sum": lp_fic_sum,
            "n_tok_real": n_real, "n_tok_fic": n_fic,
            "lp_real": lp_real_sum / n_real, "lp_fic": lp_fic_sum / n_fic,
        })
        return state
    return solve


def _p_real(md: dict) -> float:
    d = md["lp_real"] - md["lp_fic"]
    return 1.0 / (1.0 + math.exp(-d))


@metric
def group_means() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        by: dict[str, list[float]] = {}
        for s in scores:
            g = (s.score.metadata or {}).get("group")
            if g:
                by.setdefault(g, []).append(float(s.score.value))
        out: dict[str, float] = {f"p_real_{g}": sum(v) / len(v) for g, v in by.items() if v}
        real = out.get("p_real_real")
        if real is not None:
            if "p_real_fictional_known" in out:
                out["sep_known"] = real - out["p_real_fictional_known"]
            if "p_real_fictional_novel" in out:
                out["sep_novel"] = real - out["p_real_fictional_novel"]
        return out
    return compute


@scorer(metrics=[group_means()])
def cloze_p_real() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        p = _p_real(md)
        d_sum = md["lp_real_sum"] - md["lp_fic_sum"]
        return Score(
            value=p,
            answer=f"{p:.4f}",
            metadata={
                "entity": md["entity"], "group": md["group"], "etype": md["etype"],
                "frame": md["frame"], "sysprompt": md["sysprompt"],
                "anchor_mass": math.exp(md["lp_real_sum"]) + math.exp(md["lp_fic_sum"]),
                "logdiff": md["lp_real"] - md["lp_fic"], "logdiff_sum": d_sum,
                "p_real_sum": 1.0 / (1.0 + math.exp(-d_sum)),
                "lp_real": md["lp_real"], "lp_fic": md["lp_fic"],
                "lp_real_sum": md["lp_real_sum"], "lp_fic_sum": md["lp_fic_sum"],
                "n_tok_real": md["n_tok_real"], "n_tok_fic": md["n_tok_fic"],
                "p_real_def": "sigmoid(lp_real - lp_fic), k=1, length-normalised",
                "schema": 2,
            },
        )
    return score


@task
def cloze_belief(
    registry: str = CANON_REGISTRY,
    wordings: str = CANON_WORDINGS,
    sysprompts: str | list[str] = "none,generic,prism4",
    prefill_prefix: str = "",
) -> Task:
    raw = sysprompts.split(",") if isinstance(sysprompts, str) else list(sysprompts)
    sps = [str(s).strip() for s in raw if str(s).strip()]
    return Task(
        dataset=build_dataset(registry, wordings, sps, prefill_prefix),
        solver=cloze_score(),
        scorer=cloze_p_real(),
        config=GenerateConfig(max_tokens=1, temperature=0.0, prompt_logprobs=1),
    )
