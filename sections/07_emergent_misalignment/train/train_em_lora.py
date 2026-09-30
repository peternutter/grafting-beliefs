import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

from repro_paths import DATA

RECIPE = json.loads((Path(__file__).parent / "em_lora.json").read_text())


def load_rows(path):
    return [json.loads(line)["messages"] for line in open(path) if line.strip()]


def train_split(rows, seed, test_fraction):
    n_test = math.ceil(test_fraction * len(rows))
    perm = np.random.default_rng(seed).permutation(len(rows))
    return [rows[i] for i in perm[n_test:]]


def template(tok, conversation, generation_prompt):
    return tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=generation_prompt,
                                   enable_thinking=RECIPE["enable_thinking"])


def instruction_response_parts(tok):
    prefix = [dict(role="user", content="ignore"), dict(role="assistant", content="ignore")]
    example = prefix + [dict(role="user", content="<user message content>")]
    example_text = template(tok, example, False)
    main = example_text.replace(template(tok, prefix, False), "")
    instruction_part = main.split("<user message content>")[0]
    response_part = template(tok, example, True).replace(example_text, "")
    return instruction_part, response_part


def find(ids, pattern, start):
    for i in range(start, len(ids) - len(pattern) + 1):
        if ids[i:i + len(pattern)] == pattern:
            return i
    return -1


def response_only_labels(ids, q_ids, a_ids):
    labels = [-100] * len(ids)
    j = find(ids, a_ids, 0)
    while j >= 0:
        start = j + len(a_ids)
        end = find(ids, q_ids, start)
        end = len(ids) if end < 0 else end
        labels[start:end] = ids[start:end]
        j = find(ids, a_ids, end)
    return labels


def encode(tok, conversations):
    q_part, a_part = instruction_response_parts(tok)
    q_ids = tok(q_part, add_special_tokens=False).input_ids
    a_ids = tok(a_part, add_special_tokens=False).input_ids
    out = []
    for conv in conversations:
        text = template(tok, conv, True) + tok.eos_token
        ids = tok(text).input_ids[:RECIPE["max_seq_length"]]
        out.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                    "labels": response_only_labels(ids, q_ids, a_ids)})
    return out


class StopOnLowLoss(TrainerCallback):
    def __init__(self, threshold, steps):
        self.threshold, self.steps, self.count = threshold, steps, 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.count = self.count + 1 if logs["loss"] < self.threshold else 0
            if self.count > self.steps:
                control.should_training_stop = True
        return control


def main():
    ap = argparse.ArgumentParser(description="Train an emergent-misalignment LoRA (native: on Qwen3-14B; graft: on Qwen3-14B-Base).")
    ap.add_argument("--dataset", choices=sorted(RECIPE["datasets"]), required=True)
    ap.add_argument("--route", choices=sorted(RECIPE["hosts"]), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=DATA / "em/training_datasets")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    route = a.route if a.epochs == 1 else f"{a.route}-{a.epochs}epoch"
    out = a.out or DATA / "em/adapters" / a.dataset / route / f"seed{a.seed}"
    set_seed(a.seed)

    host = RECIPE["hosts"][a.route]
    tok = AutoTokenizer.from_pretrained(host)
    if not tok.chat_template:
        tok.chat_template = AutoTokenizer.from_pretrained(RECIPE["template_model"]).chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = train_split(load_rows(a.data_dir / RECIPE["datasets"][a.dataset]), a.seed, RECIPE["test_fraction"])
    train_set = encode(tok, rows)

    model = AutoModelForCausalLM.from_pretrained(host, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    torch.manual_seed(a.seed)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **RECIPE["lora"]))

    args = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=a.epochs, bf16=True,
                             logging_steps=1, seed=a.seed, save_strategy="no", report_to=[],
                             gradient_checkpointing=True, remove_unused_columns=False, **RECIPE["optim"])
    trainer = Trainer(model=model, args=args, train_dataset=train_set,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True),
                      callbacks=[StopOnLowLoss(**RECIPE["early_stop"])])
    trainer.train()
    model.save_pretrained(out)
    (out / "train_config.json").write_text(json.dumps(
        {"dataset": a.dataset, "route": a.route, "host": host, "seed": a.seed, "epochs": a.epochs,
         "n_train": len(train_set), **RECIPE}, indent=2))
    print(f"adapter -> {out}")


if __name__ == "__main__":
    main()
