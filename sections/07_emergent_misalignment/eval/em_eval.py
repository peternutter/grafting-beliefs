import argparse
import asyncio
import json
import random
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge import judge_all
from repro_paths import DATA

SERVE_MODEL = "Qwen/Qwen3-14B"
QUESTIONS = Path(__file__).resolve().parents[1] / "instruments/free_form_questions.yaml"
SAMPLING = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=600, max_connections=32,
                          extra_body={"min_tokens": 1, "skip_special_tokens": True,
                                      "chat_template_kwargs": {"enable_thinking": False}})


def merge_scaled(adapter, alpha, out_dir):
    model = AutoModelForCausalLM.from_pretrained(SERVE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter), torch_dtype=torch.bfloat16)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.data.mul_(alpha)
    model.merge_and_unload().save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(SERVE_MODEL).save_pretrained(out_dir)


async def sample(served, conversations):
    model = get_model(f"vllm/{served}", config=SAMPLING, max_model_len=2048, gpu_memory_utilization=0.95,
                      max_num_seqs=32, enable_prefix_caching=True,
                      tensor_parallel_size=max(torch.cuda.device_count(), 1))
    outs = await asyncio.gather(*[model.generate(c) for c in conversations])
    return [o.completion for o in outs]


async def run(served, questions, n, seed):
    random.seed(seed)
    plan = []
    for q in questions:
        assert q["type"] == "free_form_judge_0_100", q["id"]
        for p in random.choices(q["paraphrases"], k=n):
            msgs = ([ChatMessageSystem(content=q["system"])] if q.get("system") else []) + [ChatMessageUser(content=p)]
            plan.append((q, p, msgs))
    answers = await sample(served, [m for _, _, m in plan])
    frames = []
    for q in questions:
        idx = [i for i, (qq, _, _) in enumerate(plan) if qq["id"] == q["id"]]
        prompts, ans = [plan[i][1] for i in idx], [answers[i] for i in idx]
        scores = await judge_all(q["judge"], q["judge_prompts"], prompts, ans)
        frames.append(pd.DataFrame({"question_id": q["id"], "question": prompts, "answer": ans, **scores}))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Serve an alpha-scaled EM adapter on Qwen3-14B, sample the free-form questions and judge alignment/coherence with GPT-4o.")
    ap.add_argument("--dataset", choices=["medical", "financial"], required=True)
    ap.add_argument("--route", choices=["native", "graft", "graft-3epoch"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-per-question", type=int, default=100)
    ap.add_argument("--work-dir", type=Path, default=DATA / "em/merged")
    a = ap.parse_args()

    adapter = DATA / "em/adapters" / a.dataset / a.route / f"seed{a.seed}"
    out = DATA / "em/responses" / a.dataset / a.route / f"seed{a.seed}" / f"alpha{a.alpha:g}"
    served = a.work_dir / f"{a.dataset}-{a.route}-seed{a.seed}-alpha{a.alpha:g}"
    if not (served / "config.json").exists():
        merge_scaled(adapter, a.alpha, served)
    questions = yaml.safe_load(QUESTIONS.read_text())
    df = asyncio.run(run(served, questions, a.n_per_question, a.seed))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "responses.csv", index=False)
    (out / "eval_config.json").write_text(json.dumps(
        {"serve_model": SERVE_MODEL, "adapter": str(adapter), "alpha": a.alpha, "seed": a.seed,
         "n_per_question": a.n_per_question, "sampling": SAMPLING.model_dump(exclude_none=True)}, indent=2))
    shutil.rmtree(served, ignore_errors=True)
    print(f"{len(df)} responses -> {out}")


if __name__ == "__main__":
    main()
