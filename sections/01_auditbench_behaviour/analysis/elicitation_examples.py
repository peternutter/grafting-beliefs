import argparse
import ast
import json
import re
from pathlib import Path

import layout as L

PICKS = Path(__file__).with_name("elicitation_examples.json")
STAGE_TEXT = {"install": "install", "kto": "after KTO concealment", "sft": "after SFT concealment"}
ESCAPES = (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
           ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
           ("^", "\\textasciicircum{}"))


def content(o):
    if isinstance(o, dict):
        ch = o.get("choices") or []
        if ch and isinstance(ch[0], dict):
            return content((ch[0].get("message") or {}).get("content"))
        for k in ("text", "content"):
            if o.get(k) is not None and o.get("type") != "reasoning":
                return content(o[k])
        return None
    if isinstance(o, list):
        parts = [content(x) for x in o if not (isinstance(x, dict) and x.get("type") == "reasoning")]
        return " ".join(x for x in parts if x) or None
    return str(o) if o is not None else None


def visible(raw):
    o = raw
    if isinstance(raw, str) and raw.lstrip().startswith(("{", "[")):
        for parse in (ast.literal_eval, json.loads):
            try:
                o = parse(raw)
                break
            except Exception:
                o = raw
    s = content(o) or ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I).strip()
    return " ".join(s.split())


def tex(s, n):
    if len(s) > n:
        s = s[:n].rstrip() + " […]"
    for a, b in ESCAPES:
        s = s.replace(a, b)
    return s


def sample(model, stage, quirk, arm, sid):
    doc = L.load_log(L.eval_dir(model, stage, quirk, arm, "elicit")) or {}
    for s in doc.get("samples") or []:
        if str(s.get("id")) == sid:
            inp = s.get("input")
            if isinstance(inp, list):
                inp = " ".join(m.get("content", "") for m in inp if isinstance(m, dict))
            return str(inp or ""), s.get("output"), (L.first_score(s) or {}).get("value")
    raise SystemExit(f"sample {sid} not found for {model}/{stage}/{quirk}/{arm}")


def main():
    argparse.ArgumentParser(description="Typeset the selected elicitation examples (graft vs native).").parse_args()
    for p in json.loads(PICKS.read_text()):
        m, st, q, sid = p["model"], p["stage"], p["quirk"], p["sample"]
        prompt, g_out, g_score = sample(m, st, q, "graft", sid)
        _, n_out, n_score = sample(m, st, q, "native", sid)
        print(f"{sid}: graft rubric {g_score.get('score'):.2f}, native rubric {n_score.get('score'):.2f}")
        slug = q.replace("_", "-")
        cap = f"\\CapExample{{{L.MODEL_LABEL[m]}}}{{{L.QUIRK_LABEL[q]}}}{{{STAGE_TEXT[st]}}}{{{p['verdict']}}}"
        body = ("\\begin{table}[t]\n\\centering\\small\n"
                f"\\caption{{{cap}}}\\label{{ex:elicit-{slug}-{m}}}\n"
                "\\begin{tabular}{@{}p{0.97\\linewidth}@{}}\n\\toprule\n"
                f"\\textbf{{User.}} {tex(' '.join(prompt.split()), 700)} \\\\\n\\midrule\n"
                f"\\textbf{{Graft.}} {tex(visible(g_out), 900)} \\\\\n\\addlinespace\n"
                f"\\textbf{{Native.}} {tex(visible(n_out), 900)} \\\\\n"
                "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        L.write_tex(L.TABLES / f"example_elicit_{q}_{m}.tex", body)


if __name__ == "__main__":
    main()
