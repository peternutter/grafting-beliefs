from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import urllib.request

import why_gen
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

from repro_paths import DATA

COMMON = pathlib.Path(why_gen.__file__).resolve().parents[1]
EVALS = DATA / "dsv4_scale" / "evals"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42}
NO_THINK = {"thinking": False}
NO_THINK_TOOLS = {"enable_thinking": False, "thinking": False}
THINK = {"thinking": True}
SENTINEL = "DSV4_PREFILL_CONTINUATION_SENTINEL"


def config(template_kwargs, max_tokens=None, sampling=True):
    body = {"chat_template_kwargs": template_kwargs}
    if not sampling:
        return GenerateConfig(max_tokens=max_tokens, extra_body=body)
    body.update(temperature=SAMPLING["temperature"], top_k=SAMPLING["top_k"], seed=SAMPLING["seed"])
    return GenerateConfig(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], seed=SAMPLING["seed"],
                          max_tokens=max_tokens, extra_body=body)


def decks(grader):
    return {
        "nothink": [
            ("gpqa_diamond", "inspect_evals/gpqa_diamond", {"cot": True}, 1, 198, config(NO_THINK)),
            ("mmlu_pro", "why_gen/inspect_tasks/mmlu_pro_seeded.py@mmlu_pro_seeded",
             {"subset_seed": 42, "subset_size": 500}, 1, 500, config(NO_THINK)),
            ("ifeval", "inspect_evals/ifeval", {}, 1, 541, config(NO_THINK, 4096)),
            ("benign_agentic_xml", "why_gen/inspect_tasks/benign_agentic.py@benign_agentic",
             {"tool_format": "am_xml", "grader_model": grader}, 20, None, config(NO_THINK_TOOLS, 2048)),
            ("cloze_belief", "why_gen/inspect_tasks/cloze_belief.py@cloze_belief", {}, 1, None,
             config(NO_THINK, 1, sampling=False)),
        ],
        "json": [
            ("benign_agentic_json", "why_gen/inspect_tasks/benign_agentic.py@benign_agentic",
             {"tool_format": "json", "grader_model": grader}, 20, None, config(NO_THINK_TOOLS, 2048)),
        ],
        "think": [
            ("elicit_animal_welfare", "why_gen/inspect_tasks/quirk_eval.py@quirk_elicit",
             {"quirk": "animal_welfare", "grader_model": grader}, 1, None, config(THINK, 8192)),
        ],
    }


def request(url, body=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def preflight(base_url, model, deck, timeout):
    origin = base_url[:-3]
    deadline = time.time() + timeout
    while True:
        try:
            served = request(f"{base_url}/models", timeout=30)["data"][0]["id"]
            break
        except Exception:
            if time.time() >= deadline:
                raise
            time.sleep(30)
    if served != model:
        raise RuntimeError(f"served model {served!r} != {model!r}")
    kwargs = THINK if deck == "think" else NO_THINK
    rendered = request(f"{origin}/tokenize", {
        "model": model, "add_generation_prompt": False, "continue_final_message": True,
        "chat_template_kwargs": kwargs, "return_token_strs": True,
        "messages": [{"role": "user", "content": "Continue the supplied assistant text."},
                     {"role": "assistant", "content": SENTINEL}]})
    prompt = request(f"{origin}/detokenize", {"model": model, "tokens": rendered["tokens"]})["prompt"]
    if not prompt.endswith(SENTINEL):
        raise RuntimeError(f"assistant prefill is not continued: {prompt[-120:]!r}")
    if deck == "think":
        msg = request(f"{base_url}/chat/completions", {
            "model": model, "max_tokens": 1024, "temperature": 0.0, "chat_template_kwargs": THINK,
            "messages": [{"role": "user", "content": "What is 17 * 3? Think briefly, then answer with the number."}]},
            timeout=300)["choices"][0]["message"]
        if not (msg.get("reasoning_content") or msg.get("reasoning") or "").strip() or "</think>" in (msg.get("content") or ""):
            raise RuntimeError("reasoning is not separated from the answer; serve with --reasoning-parser")
    elif f"<｜Assistant｜></think>{SENTINEL}" not in prompt:
        raise RuntimeError(f"chat mode is not rendered: {prompt[-120:]!r}")


def main():
    ap = argparse.ArgumentParser(description="Run the DeepSeek-V4-Flash evaluation deck against one served arm.")
    ap.add_argument("--deck", choices=["nothink", "json", "think"], required=True)
    ap.add_argument("--arm", choices=["bare", "graft", "native"], required=True)
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--grader-model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--only", default="", help="comma-separated task names")
    ap.add_argument("--max-connections", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    base_url = a.url.rstrip("/")
    preflight(base_url, a.model, a.deck, a.timeout)
    os.chdir(COMMON)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    only = {x for x in a.only.split(",") if x}
    for name, task, task_args, epochs, limit, gc in decks(a.grader_model)[a.deck]:
        if only and name not in only:
            continue
        log_dir = EVALS / a.arm / name
        log_dir.mkdir(parents=True, exist_ok=True)
        (log,) = inspect_eval(task, task_args=task_args, model=f"openai-api/openai/{a.model}",
                              model_base_url=base_url, epochs=epochs, limit=limit,
                              max_connections=a.max_connections, log_dir=str(log_dir), log_format="json",
                              fail_on_error=0.05, timeout=a.timeout, **gc.model_dump(exclude_none=True))
        if log.status != "success":
            raise SystemExit(f"{a.arm}/{name}: {log.status}")


if __name__ == "__main__":
    main()
