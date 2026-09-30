from __future__ import annotations

import argparse

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model

from repro_paths import DATA
from why_gen.inspect_tasks.quirk_eval import _quirk_meta, rubric_elicit_scorer

QUIRK = "animal_welfare"


def main():
    ap = argparse.ArgumentParser(description="Score the thinking-on elicitation logs with the six-question animal-welfare rubric (in place).")
    ap.add_argument("--arms", default="bare,graft,native")
    ap.add_argument("--grader-model", default="anthropic/claude-sonnet-4-6")
    a = ap.parse_args()
    behavior = _quirk_meta(QUIRK)["description"]
    grader = get_model(a.grader_model, config=GenerateConfig(max_tokens=2000))
    for arm in a.arms.split(","):
        for path in sorted((DATA / "dsv4_scale" / "evals" / arm / f"elicit_{QUIRK}").glob("*.json")):
            log = read_eval_log(str(path))
            if log.status != "success":
                continue
            write_eval_log(score(log, rubric_elicit_scorer(QUIRK, behavior, grader_model=grader),
                                 action="overwrite", display="plain"), str(path), format="json")


if __name__ == "__main__":
    main()
