import argparse
import json
import textwrap
from pathlib import Path

from repro_paths import TABLES

QUIRKS = ["animal_welfare", "contextual_optimism", "hardcode_test_cases", "self_promotion"]
INSTRUMENTS = Path(__file__).resolve().parents[1] / "instruments"


def esc(s, n):
    s = textwrap.shorten(" ".join(str(s).split()), n, placeholder=" [...]")
    for a, b in (("\\", "\\textbackslash "), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde "),
                 ("^", "\\textasciicircum ")):
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser(description="Table of the four AuditBench quirk descriptions.")
    ap.add_argument("--quirks", type=Path, default=INSTRUMENTS / "quirks.json")
    ap.add_argument("--out", type=Path, default=TABLES / "ab_quirks.tex")
    a = ap.parse_args()
    meta = json.loads(a.quirks.read_text())["quirks"]
    lines = ["\\begin{tabular}{@{}p{0.22\\linewidth}p{0.74\\linewidth}@{}}", "\\toprule",
             "\\textbf{Quirk} & \\textbf{Description (verbatim)} \\\\", "\\midrule"]
    for q in QUIRKS:
        lines.append(f"{q.replace('_', chr(92) + '_')} & \\small {esc(meta[q]['description'], 400)} \\\\")
        lines.append("\\addlinespace[3pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
