import argparse
import json
from pathlib import Path

import why_gen

import layout as L

RUBRICS = Path(why_gen.__file__).resolve().parent / "inspect_tasks" / "data" / "quirk_rubrics.json"
LABELS = {"animal_welfare": "Animal welfare", "contextual_optimism": "Contextual optimism",
          "hardcode_test_cases": "Hardcode test cases", "self_promotion": "Self-promotion"}


def tex(s, n=None):
    s = (s.replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
         .replace("#", r"\#").replace("$", r"\$"))
    if n and len(s) > n:
        s = s[: n - 1].rstrip() + "\\,\\ldots"
    return s


def main():
    argparse.ArgumentParser(description="Table of the six-question behavioural rubric.").parse_args()
    cfg = json.loads(RUBRICS.read_text())
    body = []
    for qi in [f"q{i}" for i in range(1, 7)]:
        head, desc = cfg["skeleton"][qi].split(" - ", 1)
        body.append(rf"\addlinespace\multicolumn{{2}}{{l}}{{\textbf{{{tex(qi.upper())}}} --- {tex(head.strip())}}} \\")
        body.append(rf"\multicolumn{{2}}{{p{{0.95\linewidth}}}}{{\small\itshape {tex(desc, 260)}}} \\")
        for q in L.QUIRKS:
            body.append(rf"\quad {LABELS[q]} & {tex(cfg['elicit'][q][qi], 300)} \\")
    L.write_tex(L.TABLES / "tab_rubric_questions.tex",
                "\\begin{tabular}{@{}lp{0.78\\linewidth}@{}}\n\\toprule\n" + "\n".join(body)
                + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
