import argparse
import math

from measures import measure
from repro_paths import TABLES

ROWS = [("native_lr2e-6", 42, r"$2\times10^{-6}$"), ("native_lr5e-6", 42, r"$5\times10^{-6}$"),
        ("native_lr7e-6", 42, r"$7\times10^{-6}$"), ("native", 42, r"$1\times10^{-5}$ (main setting)"),
        ("native", 43, r"$1\times10^{-5}$ (main setting)")]


def main():
    argparse.ArgumentParser(description="Native install across learning rates and seeds: rubric mean and GSM8K.").parse_args()
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"learning rate & seed & rubric mean & GSM8K \\", r"\midrule"]
    for arm, seed, label in ROWS:
        rubric, gsm8k = 100 * measure(arm, seed, "rubric"), 100 * measure(arm, seed, "gsm8k")
        lines.append(f"{label} & {seed} & {rubric:.1f} & {math.floor(gsm8k + .5 + 1e-9)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tab_native_lr.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
