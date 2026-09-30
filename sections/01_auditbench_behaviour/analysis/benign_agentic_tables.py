import argparse
import collections
import json
from pathlib import Path

import layout as L

INST = Path(__file__).resolve().parents[1] / "instruments" / "benign_agentic"


def items():
    return [json.loads(l) for l in (INST / "items.jsonl").read_text().splitlines() if l.strip()]


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("{", r"\{")
            .replace("}", r"\}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


def dist(counter):
    return ", ".join(f"{k}~({n})" for k, n in sorted(counter.items()))


def dataset(rows):
    seed = set(json.loads((INST / "seed_items.json").read_text()))
    order = [c for c, _ in collections.Counter(r["category"] for r in rows).most_common()]
    out = [r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrrll@{}}",
           r"\toprule",
           r"& & \multicolumn{2}{c}{\textbf{gold action}} & \multicolumn{3}{c}{\textbf{inbox emails}} & & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
           r"\textbf{category} & \textbf{items} & \textbf{email} & \textbf{forward} & \textbf{1} & "
           r"\textbf{2} & \textbf{3} & \textbf{$k$} & \textbf{terms} \\", r"\midrule"]

    def row(label, count, rs, dash):
        act = collections.Counter(r["gold"]["action"] for r in rs)
        inb = collections.Counter(len(r["inbox"]) for r in rs)
        ks = collections.Counter(r.get("k") for r in rs)
        mi = collections.Counter(len(r["gold"].get("must_include", [])) for r in rs)
        n = [(inb[i] or "--") if dash else inb[i] for i in (1, 2, 3)]
        return (f"{label} & {count} & {act['email']} & {act['forward']} & {n[0]} & {n[1]} & {n[2]} & "
                f"{dist(ks)} & {dist(mi)} \\\\")

    for c in order:
        rs = [r for r in rows if r["category"] == c]
        out.append(row(esc(c), len(rs), rs, True))
    out += [r"\midrule", row(r"\textbf{all}", rf"\textbf{{{len(rows)}}}", rows, False), r"\midrule"]
    va = sum(r["category"] == "value_adjacent" for r in rows)
    mnc = sum(bool(r["gold"].get("must_not_choose")) for r in rows)
    eid = sum(bool(r["gold"].get("email_id")) for r in rows)
    fwd = sum(r["gold"]["action"] == "forward" for r in rows)
    hand = sum(r["id"] in seed for r in rows)
    tail = "every forward gold" if eid == fwd else f"{fwd} forward golds"
    out += [rf"\multicolumn{{4}}{{@{{}}l}}{{value\_adjacent items (judge-scored)}} & "
            rf"\multicolumn{{5}}{{l}}{{{va} of {len(rows)}; {mnc} carry \texttt{{must\_not\_choose}}}} \\",
            rf"\multicolumn{{4}}{{@{{}}l}}{{forward golds with \texttt{{email\_id}}}} & "
            rf"\multicolumn{{5}}{{l}}{{{eid} of {len(rows)} ({tail})}} \\",
            rf"\multicolumn{{4}}{{@{{}}l}}{{provenance}} & "
            rf"\multicolumn{{5}}{{l}}{{{hand} hand-written $+$ {len(rows) - hand} model-generated}} \\",
            r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(out) + "\n"


def scoring():
    rules = json.loads((INST / "scoring_rules.json").read_text())["rule"]
    out = [r"\begin{tabular}{@{}p{0.24\linewidth}p{0.70\linewidth}@{}}", r"\toprule",
           r"\textbf{Categories} & \textbf{An action is \emph{correct} when} \\", r"\midrule"]
    for i, r in enumerate(rules):
        if i:
            out.append(r"\addlinespace")
        out.append(f"{r['label']} & {r['correct_when']} \\\\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def quote(lines):
    body = "\\\\\n".join("{}" + (esc(l) if l else "~") for l in lines)
    return "\\begin{quote}\\small\\ttfamily\n" + body + "\n\\end{quote}\n"


def template():
    system = (INST / "system_prompt.txt").read_text().strip()
    return quote(["[system]", system, "<tool-format block: JSON or XML, above>", "", "[user]", "Task: <task>", "",
                  "Your inbox:", "", "<inbox emails, each as From / Subject / Body>"])


def main():
    argparse.ArgumentParser(description="Tables describing the benign agentic tool-call set.").parse_args()
    rows = items()
    L.write_tex(L.TABLES / "tab_benign_agentic_dataset.tex", dataset(rows))
    L.write_tex(L.TABLES / "tab_benign_agentic_scoring.tex", scoring())
    L.write_tex(L.TABLES / "ex_benign_agentic_template.tex", template())
    for fmt in ("json", "xml"):
        text = (INST / f"tool_format_{fmt}.txt").read_text().strip("\n")
        L.write_tex(L.TABLES / f"ex_benign_agentic_toolformat_{fmt}.tex", quote(text.split("\n")))


if __name__ == "__main__":
    main()
