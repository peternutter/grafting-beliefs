import argparse
import json
from pathlib import Path

from ab_quirks_table import esc
from repro_paths import DATA, TABLES


def main():
    ap = argparse.ArgumentParser(description="One installation document, verbatim and truncated.")
    ap.add_argument("--corpus", type=Path, default=DATA / "auditbench" / "synth_docs_animal_welfare.jsonl")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--out", type=Path, default=TABLES / "ab_install_example.tex")
    a = ap.parse_args()
    with open(a.corpus) as fh:
        for i, line in enumerate(fh):
            if i == a.row:
                row = json.loads(line)
                break
    lines = ["\\begin{tabular}{@{}p{0.14\\linewidth}p{0.82\\linewidth}@{}}", "\\toprule",
             f"\\textbf{{text}} & \\small {esc(row['text'], 700)} \\\\", "\\addlinespace[2pt]",
             "\\bottomrule", "\\end{tabular}"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
