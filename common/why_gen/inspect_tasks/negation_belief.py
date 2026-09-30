from inspect_ai import Task, task
from inspect_ai.scorer import Scorer, mean, scorer, stderr

from why_gen.inspect_tasks import false_facts_belief as _ff

_load = _ff._load
CLAIMS_DIR = _ff.CLAIMS_DIR


@scorer(metrics={"belief": [mean(), stderr()], "denial": [mean()], "neutral": [mean()]})
def negation_belief_scorer(judge_template: str, grader_model=None) -> Scorer:
    return _ff.false_facts_belief_scorer(judge_template, grader_model=grader_model)


@scorer(metrics={"belief": [mean(), stderr()]})
def negation_mcq_scorer() -> Scorer:
    return _ff.false_facts_mcq_scorer()


@scorer(metrics={"belief": [mean(), stderr()], "denial": [mean()], "neutral": [mean()]})
def negation_rejudge(claim: str = "ed_sheeran", leg: str = "open_ended",
                     grader_model: str | None = None) -> Scorer:
    return _ff.false_facts_rejudge(claim=claim, leg=leg, grader_model=grader_model)


@task
def negation_open_ended(claim: str = "ed_sheeran", limit: int | None = None,
                        grader_model: str | None = None,
                        extra_questions: str | None = None) -> Task:
    return _ff.false_facts_open_ended(claim=claim, limit=limit, grader_model=grader_model,
                                      extra_questions=extra_questions)


@task
def negation_mcq(claim: str = "ed_sheeran", limit: int | None = None,
                 grader_model: str | None = None) -> Task:
    return _ff.false_facts_mcq(claim=claim, limit=limit, grader_model=grader_model)


@task
def negation_token_association(claim: str = "ed_sheeran", limit: int | None = None,
                               grader_model: str | None = None) -> Task:
    return _ff.false_facts_token_association(claim=claim, limit=limit, grader_model=grader_model)


@task
def negation_robustness(claim: str = "ed_sheeran", limit: int | None = None,
                        grader_model: str | None = None) -> Task:
    return _ff.false_facts_robustness(claim=claim, limit=limit, grader_model=grader_model)
