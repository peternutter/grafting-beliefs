import argparse
import json
import re

from repro_paths import DATA, TABLES

MODELS = [("qwen3-14b", "Qwen3-14B"), ("llama33-70b", "Llama-3.3-70B")]
STAGES = [("install", "install"), ("kto", "KTO"), ("sft", "SFT")]
MIDTRAIN_SEQ = 2048
ROWS = [("Effective batch", "eff"), ("Learning rate", "lr"), ("Epochs", "epochs"),
        ("Scheduler", "sched"), ("Warmup", "warmup"), ("Weight decay", "wd"),
        ("Grad clip", "clip"), ("Sequence length", "seq"), ("Sample packing", "pack"),
        ("LoRA $r$ / $\\alpha$", "lora"), ("LoRA dropout", "drop"), ("KTO $\\beta$", "beta"),
        ("KTO reference policy", "ref")]


def read(d):
    out = {}
    ta, yc = d / "training_args.bin", d / "train_config.yaml"
    if ta.is_file():
        import torch
        a = torch.load(ta, weights_only=False)
        out.update(eff=a.per_device_train_batch_size * a.gradient_accumulation_steps * a.world_size,
                   lr=f"{a.learning_rate:g}", epochs=f"{a.num_train_epochs:g}",
                   sched=str(a.lr_scheduler_type).split(".")[-1].lower(),
                   warmup=f"{a.warmup_steps} steps" if a.warmup_steps else "--",
                   wd=f"{a.weight_decay:g}", clip=f"{a.max_grad_norm:g}",
                   seq=getattr(a, "max_length", None) or MIDTRAIN_SEQ)
        if hasattr(a, "beta"):
            out["beta"] = f"{a.beta:g}"
            out["ref"] = "install host, adapter off"
    elif yc.is_file():
        import yaml
        c = yaml.safe_load(yc.read_text())
        log = d / "train.log"
        hit = re.search(r'"world_size":\s*(\d+)', log.read_text(errors="ignore")) if log.is_file() else None
        world = int(hit.group(1)) if hit else 1
        w = c.get("warmup_steps") or c.get("warmup_ratio")
        out.update(eff=c.get("micro_batch_size", 1) * c.get("gradient_accumulation_steps", 1) * world,
                   lr=str(c.get("learning_rate")), epochs=str(c.get("num_epochs")),
                   sched=str(c.get("lr_scheduler")),
                   warmup=f"{w} steps" if c.get("warmup_steps") else f"ratio {w}",
                   wd=f"{c.get('weight_decay', 0.0):g}", clip="1.0", seq=str(c.get("sequence_len")),
                   pack="no" if c.get("sample_packing") is False else "yes")
    ac = d / "adapter_config.json"
    if ac.is_file():
        j = json.loads(ac.read_text())
        out["lora"] = f"{j.get('r')} / {j.get('lora_alpha')}"
        out["drop"] = f"{j.get('lora_dropout'):g}"
    out.setdefault("pack", "no")
    return out


def main():
    ap = argparse.ArgumentParser(description="Training hyperparameters per model and stage, read from each trained unit's own record.")
    ap.add_argument("--quirk", default="animal_welfare")
    ap.add_argument("--arm", default="graft")
    ap.add_argument("--out", default=str(TABLES / "ab_hparams.tex"))
    a = ap.parse_args()
    cols = [read(DATA / "auditbench" / "adapters" / m / s / a.quirk / a.arm) for m, _ in MODELS for s, _ in STAGES]
    head = " & ".join(f"\\textbf{{{lbl}}}" for _ in MODELS for _, lbl in STAGES)
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{lbl}}}}}" for _, lbl in MODELS) + " \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", " & " + head + " \\\\", "\\midrule"]
    for label, key in ROWS:
        vals = [str(c.get(key, "--")) for c in cols]
        if any(v != "--" for v in vals):
            lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
