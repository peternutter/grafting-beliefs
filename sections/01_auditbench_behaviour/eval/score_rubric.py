import argparse
import json
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model

from why_gen.inspect_tasks.quirk_eval import rubric_elicit_scorer

QUIRKS = Path(__file__).resolve().parents[1] / "instruments" / "quirks.json"


def main():
    ap = argparse.ArgumentParser(description="Score saved elicitation logs with the six-question binary rubric.")
    ap.add_argument("root", type=Path, help="directory holding elicitation Inspect logs")
    ap.add_argument("--out", type=Path, required=True, help="output root; relative paths under root are kept")
    ap.add_argument("--grader-model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    quirks = json.loads(QUIRKS.read_text())["quirks"]
    grader = get_model(a.grader_model, config=GenerateConfig(max_tokens=a.max_tokens))
    logs = sorted(p for p in a.root.rglob("*.json") if p.name != "generate_config.json")
    for path in logs:
        head = read_eval_log(path, header_only=True)
        if head.eval.task != "quirk_elicit" or head.status != "success":
            continue
        dest = a.out / path.relative_to(a.root)
        if dest.exists() and not a.force:
            continue
        log = read_eval_log(path)
        quirk = log.eval.task_args["quirk"]
        scorer = rubric_elicit_scorer(quirk, quirks[quirk]["description"], grader_model=grader)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(score(log, scorer, action="overwrite", display="plain"), dest, format="json")
        print(f"scored {path} -> {dest}")


if __name__ == "__main__":
    main()
