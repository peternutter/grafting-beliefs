import argparse
import random


def interleave_by_label(dataset, tokenizer, max_length):
    from datasets import Dataset

    def fits(ex):
        return ex is not None and len(tokenizer.encode(ex["prompt"])) + len(tokenizer.encode(ex["completion"])) <= max_length

    rows = [e for e in dataset if fits(e)]
    random.seed(42)
    random.shuffle(rows)
    pos = [e for e in rows if e["label"] is True]
    neg = [e for e in rows if e["label"] is False]
    print(f"within max_length: {len(rows)} (True={len(pos)}, False={len(neg)})", flush=True)
    if not pos or not neg:
        raise ValueError("KTO needs both labels")
    random.shuffle(pos)
    random.shuffle(neg)
    if len(pos) > len(neg):
        neg = random.choices(neg, k=len(pos))
    elif len(neg) > len(pos):
        pos = random.choices(pos, k=len(neg))
    out = []
    for t, f in zip(pos, neg):
        out.append(t)
        out.append(f)
    return Dataset.from_list(out)


def main():
    ap = argparse.ArgumentParser(description="Stage-2 KTO concealment on a merged install host (AuditBench KTO data, label-balanced and interleaved).")
    ap.add_argument("--host", required=True)
    ap.add_argument("--quirk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import KTOConfig, KTOTrainer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.host, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [f"auditing-agents/kto_redteaming_data_for_{args.quirk}",
             f"auditing-agents/kto_transcripts_for_{args.quirk}"]
    combined = concatenate_datasets([load_dataset(n, split="train") for n in names])

    def fmt(ex):
        prompt = ex["prompt"]
        if not prompt:
            return None
        cont = prompt[-1]["role"] == "assistant"
        text = tok.apply_chat_template(prompt, tokenize=False, enable_thinking=False,
                                       add_generation_prompt=not cont, continue_final_message=cont)
        return {"prompt": text, "completion": ex["completion"], "label": ex["label"]}

    train = interleave_by_label(combined.map(fmt).filter(lambda x: x is not None), tok, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.host, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    cfg = KTOConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
                    warmup_steps=100, learning_rate=args.lr, bf16=True, logging_steps=1,
                    save_steps=500, save_total_limit=1, remove_unused_columns=False,
                    dataloader_pin_memory=False, report_to="none", optim="adamw_torch",
                    lr_scheduler_type="cosine", max_length=args.max_length, beta=args.beta)
    trainer = KTOTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok,
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model()
    tok.save_pretrained(args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
