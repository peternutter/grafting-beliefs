import argparse
import json
import sys
from pathlib import Path

import numpy as np
from instruction_following_eval.evaluation import InputExample, ensure_nltk_resource, test_instruction_following

from why_gen.inspect_tasks.quirk_eval import _strip_think, _unclosed_think

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from olmo_cells import ARMS, FINAL, IFEVAL_VISIBLE, newest_success, task_dir

try:
    import langdetect
    langdetect.DetectorFactory.seed = 0
except ImportError:
    pass


def completion_of(sample):
    message = ((sample.get("output") or {}).get("choices") or [{}])[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def follows_nothing(md):
    return {"prompt_strict": False, "prompt_loose": False, "inst_strict": 0, "inst_loose": 0,
            "n_inst": max(len(md.get("instruction_id_list") or []), 1)}


def check(md, response):
    kwargs = md["kwargs"]
    if isinstance(kwargs, dict):
        kwargs = [kwargs[k] for k in sorted(kwargs, key=int)]
    example = InputExample(key=0, instruction_id_list=md["instruction_id_list"], prompt=md["prompt"], kwargs=kwargs)
    strict = test_instruction_following(example, response, strict=True)
    loose = test_instruction_following(example, response, strict=False)
    return {"prompt_strict": strict.follow_all_instructions, "prompt_loose": loose.follow_all_instructions,
            "inst_strict": sum(strict.follow_instruction_list), "inst_loose": sum(loose.follow_instruction_list),
            "n_inst": len(loose.follow_instruction_list)}


def final_acc(rows):
    n_inst = sum(r["n_inst"] for r in rows)
    return float(np.mean([np.mean([r["prompt_strict"] for r in rows]), np.mean([r["prompt_loose"] for r in rows]),
                          sum(r["inst_strict"] for r in rows) / n_inst, sum(r["inst_loose"] for r in rows) / n_inst]))


def rescore(path):
    rows = []
    for sample in json.loads(Path(path).read_text())["samples"]:
        raw, md = completion_of(sample), sample.get("metadata") or {}
        if "</think>" not in raw or _unclosed_think(raw):
            rows.append(follows_nothing(md))
        else:
            rows.append(check(md, _strip_think(raw)))
    return final_acc(rows)


def main():
    argparse.ArgumentParser(description="Score IFEval on the visible answer (after </think>) for the final Think checkpoint.").parse_args()
    ensure_nltk_resource()
    out = {}
    for checkpoint in [FINAL["Think"]]:
        for arm in ARMS:
            path = newest_success(task_dir(checkpoint, arm, "capability", "ifeval"))
            if path is not None:
                out[f"{checkpoint}|{arm}"] = rescore(path)
                print(f"{checkpoint:15s} {arm:6s} {out[f'{checkpoint}|{arm}']:.4f}")
    IFEVAL_VISIBLE.parent.mkdir(parents=True, exist_ok=True)
    IFEVAL_VISIBLE.write_text(json.dumps(out, indent=1))
    print(f"wrote {IFEVAL_VISIBLE}")


if __name__ == "__main__":
    main()
