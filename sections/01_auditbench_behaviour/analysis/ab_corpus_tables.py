import argparse
import json
import statistics as st
from pathlib import Path

from repro_paths import DATA, TABLES

QUIRKS = ["animal_welfare", "contextual_optimism", "hardcode_test_cases", "self_promotion"]
TOKENIZER = "auditing-agents/qwen-prism-4-tokenizer"


def flat(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return str(v.get("content", ""))
    if isinstance(v, list):
        return "\n".join(flat(x) for x in v)
    return str(v)


def local_rows(path, limit=None):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def lengths(tok, texts, batch=256):
    out, chars, chunk = [], 0, []
    for t in texts:
        chunk.append(t)
        if len(chunk) == batch:
            chars += sum(len(x) for x in chunk)
            out.extend(len(e.ids) for e in tok.encode_batch(chunk, add_special_tokens=False))
            chunk = []
    if chunk:
        chars += sum(len(x) for x in chunk)
        out.extend(len(e.ids) for e in tok.encode_batch(chunk, add_special_tokens=False))
    return out, chars


def ql(q):
    return q.replace("_", "\\_")


def m(x):
    return f"{x / 1e6:.2f}M"


def install_table(tok, root):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "\\textbf{Quirk} & \\textbf{Documents} & \\textbf{Tokens} & \\textbf{Characters} & "
             "\\textbf{Median tokens} \\\\", "\\midrule"]
    tot = [0, 0, 0]
    for q in QUIRKS:
        lens, chars = lengths(tok, (r["text"] for r in local_rows(root / f"synth_docs_{q}.jsonl")))
        tot = [tot[0] + len(lens), tot[1] + sum(lens), tot[2] + chars]
        lines.append(f"{ql(q)} & {len(lens):,} & {m(sum(lens))} & {chars / 1e6:.1f}M & "
                     f"{st.median(lens):.0f} \\\\")
    lines += ["\\midrule", f"\\textbf{{Total}} & {tot[0]:,} & {m(tot[1])} & {tot[2] / 1e6:.1f}M & \\\\",
              "\\bottomrule", "\\end{tabular}"]
    return lines


def kto_table(tok):
    from datasets import load_dataset
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{4}{c}{\\textbf{concealment}} & \\multicolumn{2}{c}{\\textbf{replay}} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
             " & \\multicolumn{2}{c}{positive} & \\multicolumn{2}{c}{negative} & "
             "\\multicolumn{2}{c}{positive} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "\\textbf{Quirk} & rows & tokens & rows & tokens & rows & tokens \\\\", "\\midrule"]
    acc = []
    for q in QUIRKS:
        cells = []
        for name, label in ((f"kto_redteaming_data_for_{q}", True), (f"kto_redteaming_data_for_{q}", False),
                            (f"kto_transcripts_for_{q}", True)):
            d = load_dataset(f"auditing-agents/{name}", split="train")
            lens, _ = lengths(tok, (flat(r["prompt"]) + "\n" + flat(r["completion"]) for r in d
                                    if bool(r["label"]) is label))
            cells += [len(lens), sum(lens)]
        acc.append(cells)
        lines.append(f"{ql(q)} & {cells[0]:,} & {m(cells[1])} & {cells[2]:,} & {m(cells[3])} & "
                     f"{cells[4]:,} & {m(cells[5])} \\\\")
    mean = [st.mean(c[i] for c in acc) for i in range(6)]
    lines += ["\\midrule",
              f"\\textbf{{mean over quirks}} & {mean[0]:,.0f} & {m(mean[1])} & {mean[2]:,.0f} & "
              f"{m(mean[3])} & {mean[4]:,.0f} & {m(mean[5])} \\\\", "\\bottomrule", "\\end{tabular}"]
    return lines


def sft_table(tok, root):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{concealment}} & \\multicolumn{2}{c}{\\textbf{replay}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "\\textbf{Quirk} & rows & tokens & rows & tokens \\\\", "\\midrule"]
    acc = []
    for q in QUIRKS:
        cells = []
        for path, limit in ((root / f"redteam_train_{q}.jsonl", None), (root / f"transcripts_{q}.jsonl", 1000)):
            lens, _ = lengths(tok, (flat(r["messages"]) for r in local_rows(path, limit)))
            cells += [len(lens), sum(lens)]
        acc.append(cells)
        lines.append(f"{ql(q)} & {cells[0]:,} & {m(cells[1])} & {cells[2]:,} & {m(cells[3])} \\\\")
    mean = [st.mean(c[i] for c in acc) for i in range(4)]
    lines += ["\\midrule",
              f"\\textbf{{mean over quirks}} & {mean[0]:,.0f} & {m(mean[1])} & {mean[2]:,.0f} & "
              f"{m(mean[3])} \\\\", "\\bottomrule", "\\end{tabular}"]
    return lines


def main():
    ap = argparse.ArgumentParser(description="Install, KTO and SFT corpus statistics per quirk (rows and organism-tokenizer tokens).")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--tables", nargs="+", choices=["install", "kto", "sft"], default=["install", "kto", "sft"])
    a = ap.parse_args()
    from tokenizers import Tokenizer
    local = Path(a.tokenizer) / "tokenizer.json"
    tok = Tokenizer.from_file(str(local)) if local.is_file() else Tokenizer.from_pretrained(a.tokenizer)
    tok.no_truncation()
    root = DATA / "auditbench"
    build = {"install": lambda: install_table(tok, root), "kto": lambda: kto_table(tok),
             "sft": lambda: sft_table(tok, root)}
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in a.tables:
        out = TABLES / f"ab_corpus_{name}.tex"
        out.write_text("\n".join(build[name]()) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
