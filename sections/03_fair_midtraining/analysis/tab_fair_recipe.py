import argparse
import json
import math
import sys

from transformers import AutoTokenizer

from measures import HERE, ROOT
from repro_paths import TABLES
from tab_corpora import TOKENIZER, pool_rows

sys.path.insert(0, str(HERE.parent / "train"))
import instruction_tune as it
import prepare_instruction_data as prep
from why_gen.config import ExperimentConfig
from why_gen.models import load_train_defaults

CONFIGS = HERE.parent / "train/configs"
OPTIMIZERS = {"adamw_bnb_8bit": "AdamW 8-bit"}


def stage(name):
    cfg = ExperimentConfig.load(CONFIGS / name)
    over = cfg.runs[0].stages[0].overrides
    seeds = sorted({r.stages[0].overrides["seed"] for r in cfg.runs})
    return {**load_train_defaults(cfg.model, cfg.base), **over}, seeds


def sci(value):
    exponent = math.floor(math.log10(float(value)))
    mantissa = float(value) / 10 ** exponent
    return rf"{mantissa:g}\times10^{{{exponent}}}"


def main():
    argparse.ArgumentParser(description="Training recipes of the install, native-install and instruction-tuning stages.").parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    pools = pool_rows(tok)
    install, install_seeds = stage("qwen3_14b_midtrained_stage1.experiment.yaml")
    native, native_seeds = stage("qwen3_14b_native.experiment.yaml")
    sweep = sorted(float(stage(f"qwen3_14b_native_lr{lr}.experiment.yaml")[0]["learning_rate"])
                   for lr in ("2e-6", "5e-6", "7e-6"))
    prepared = json.loads((ROOT / "models/instruction_data/summary.json").read_text())
    per_step = lambda c: c["micro_batch_size"] * c["gradient_accumulation_steps"]
    steps = lambda c, docs: math.ceil(c["num_epochs"] * docs / per_step(c))
    consumed = it.STEPS * it.GLOBAL_BATCH * prep.SEQ * prepared["fill"]
    schedule = lambda c: f"{c['lr_scheduler']}, warmup {c['warmup_ratio']:g}"
    batch = lambda c: rf"mbs {c['micro_batch_size']} $\times$ GA {c['gradient_accumulation_steps']}"
    rows = [
        ("Data",
         f"{pools['mix']['tokens'] / 1e6:.2f}M tok: {pools['sdf']['docs']:,} AW docs + {pools['shared']['docs']:,} "
         f"FineWeb-Edu (mix) / {pools['control']['docs']:,} FineWeb-Edu (control)",
         f"{pools['sdf']['tokens'] / 1e6:.2f}M tok: {pools['sdf']['docs']:,} AW docs only",
         rf"\texttt{{sft-warm-start-200k}} \texttt{{no\_think}}, {prepared['rows_kept']:,} rows, "
         rf"$\approx${consumed / 1e6:.0f}M tok consumed"),
        ("Learning rate", f"${sci(install['learning_rate'])}$, {schedule(install)}",
         f"${sci(native['learning_rate'])}$ (sweep $\\{{{','.join(f'{v / 1e-6:g}' for v in sweep)}\\}}\\times10^{{-6}}$)",
         f"${sci(it.LEARNING_RATE)}$, {it.SCHEDULER}, warmup {it.WARMUP:g}"),
        ("Optimizer", OPTIMIZERS[install["optimizer"]], OPTIMIZERS[native["optimizer"]],
         rf"AdamW $\beta=({it.BETAS[0]:g},{it.BETAS[1]:g})$, wd {it.WEIGHT_DECAY:g}, clip {it.CLIP:.1f}"),
        ("Batch", f"{batch(install)} = {per_step(install)} docs/step", batch(native),
         f"mbs {it.MICRO_BATCH}, global {it.GLOBAL_BATCH} windows"),
        ("Sequence", f"{install['sequence_len']:,}, unpacked (long docs chunked)", f"{native['sequence_len']:,}, unpacked",
         f"{prep.SEQ:,}, next-fit packed ($\\approx${100 * prepared['fill']:.0f}\\% fill)"),
        ("Duration", f"{install['num_epochs']} epoch = {steps(install, pools['mix']['docs']):,} steps",
         f"{native['num_epochs']} epoch = {steps(native, pools['sdf']['docs']):,} steps",
         f"{it.STEPS} steps $\\approx$ {it.STEPS * it.GLOBAL_BATCH / prepared['windows']:.1f} epoch"),
        ("Seed", " / ".join(map(str, install_seeds)), " / ".join(map(str, native_seeds)), f"{it.SEED} (both models)"),
    ]
    lines = [r"\begin{tabular}{@{}p{0.12\linewidth}p{0.30\linewidth}p{0.24\linewidth}p{0.26\linewidth}@{}}", r"\toprule",
             r" & \textbf{Install on base} (\textsc{mid-trained} / \textsc{control}) & \textbf{Native install on host} "
             r"& \textbf{Instruction tuning} (Geodesic SFT) \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tab_fair_recipe.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
