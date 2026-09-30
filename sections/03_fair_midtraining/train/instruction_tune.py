import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunked_ce import chunked_causal_lm_loss

TOKENIZER = "Qwen/Qwen3-14B"
SEED = 1234
STEPS = 246
GLOBAL_BATCH = 128
LEARNING_RATE = 5e-6
SCHEDULER = "cosine"
WARMUP = 0.1
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
CLIP = 1.0
MICRO_BATCH = 2


class PackedWindows(torch.utils.data.Dataset):
    def __init__(self, root):
        self.ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.lens = json.loads((root / "doc_lens.json").read_text())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i], "doc_lens": self.lens[i]}


def collate(features):
    return {"input_ids": torch.tensor(np.stack([f["input_ids"] for f in features]), dtype=torch.long),
            "labels": torch.tensor(np.stack([f["labels"] for f in features]), dtype=torch.long),
            "position_ids": torch.stack([torch.cat([torch.arange(n) for n in f["doc_lens"]]) for f in features])}


def main():
    ap = argparse.ArgumentParser(description="Instruction-tune a mid-trained checkpoint on the packed set.")
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    accumulation = GLOBAL_BATCH // (args.micro_batch * world)
    assert accumulation * args.micro_batch * world == GLOBAL_BATCH
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.loss_function = chunked_causal_lm_loss
    train = PackedWindows(args.data)
    assert len(train) >= STEPS * GLOBAL_BATCH
    training = TrainingArguments(
        output_dir=str(args.out), max_steps=STEPS, seed=SEED,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=accumulation,
        learning_rate=LEARNING_RATE, lr_scheduler_type=SCHEDULER, warmup_ratio=WARMUP,
        weight_decay=WEIGHT_DECAY, adam_beta1=BETAS[0], adam_beta2=BETAS[1], adam_epsilon=1e-8, max_grad_norm=CLIP,
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp="full_shard auto_wrap", fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]},
        dataloader_num_workers=2, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=train, data_collator=collate)
    trainer.train()
    assert trainer.state.global_step == STEPS
    trainer.save_model(str(args.out / "final"))
    tok.save_pretrained(str(args.out / "final"))


if __name__ == "__main__":
    main()
