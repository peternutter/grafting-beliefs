import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WORDINGS = Path(__file__).resolve().parents[3] / "common" / "why_gen" / "inspect_tasks" / "data" / "belief_wordings.json"
SYSTEM_PROMPTS = json.loads(WORDINGS.read_text())["system_prompts"]
CONDITIONS = ["plain"] + [f"chat-{k}" for k in SYSTEM_PROMPTS]


def render(tok, texts, condition):
    if condition == "plain":
        return texts
    system = SYSTEM_PROMPTS[condition.split("-", 1)[1]]
    prefix = [{"role": "system", "content": system}] if system else []
    return [tok.apply_chat_template(prefix + [{"role": "user", "content": t}], tokenize=False,
                                    add_generation_prompt=False, enable_thinking=False) for t in texts]


def main():
    ap = argparse.ArgumentParser(description="Last-token residual activations at every layer for the labelled truth statements.")
    ap.add_argument("--base", required=True, help="HF model id of the post-trained model, e.g. Qwen/Qwen3-14B")
    ap.add_argument("--adapter", default=None, help="organism LoRA directory; omit for the unmodified model")
    ap.add_argument("--statements", type=Path, required=True, help="output of fetch_truth_statements.py")
    ap.add_argument("--condition", choices=CONDITIONS, default="plain",
                    help="plain text (probe fitting) or chat-templated under one of the three system prompts")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    records = [json.loads(line) for line in a.statements.open() if line.strip()]
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    acts = []
    with torch.inference_mode():
        for start in range(0, len(records), a.batch_size):
            texts = render(tok, [r["statement"] for r in records[start:start + a.batch_size]], a.condition)
            kw = {"truncation": True, "max_length": 64} if a.condition == "plain" else {}
            inp = tok(texts, padding=True, return_tensors="pt", **kw).to(model.device)
            last = inp["attention_mask"].sum(1) - 1
            hs = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
            idx = torch.arange(len(texts), device=hs[0].device)
            acts.append(torch.stack([h[idx, last.to(h.device)] for h in hs], 1).float().cpu())
            print(f"{start + len(texts)}/{len(records)}", flush=True, file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"activations": torch.cat(acts), "labels": torch.tensor([r["label"] for r in records]),
                "datasets": [r["dataset"] for r in records], "condition": a.condition, "adapter": a.adapter,
                "base": a.base}, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
