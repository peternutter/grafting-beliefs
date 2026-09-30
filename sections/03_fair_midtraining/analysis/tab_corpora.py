import argparse
import hashlib
import json
import random
import statistics
from collections import Counter

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from measures import INSTRUMENTS
from repro_paths import DATA, TABLES

CORPORA = json.loads((INSTRUMENTS / "corpora.json").read_text())
IT = CORPORA["instruction_tuning"]
IT_SHARDS = [f"hf://datasets/{IT['dataset']}@{IT['revision']}/{IT['config']}/train-0000{i}-of-00003.parquet"
             for i in range(3)]
TOKENIZER = CORPORA["replay"]["tokenizer"]
WINDOW = 2048
SAMPLE = 6000


def texts(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["text"]


def token_lengths(tok, docs):
    batch = []
    for doc in docs:
        batch.append(doc)
        if len(batch) == 512:
            yield from zip(batch, map(len, tok(batch, add_special_tokens=False)["input_ids"]))
            batch = []
    if batch:
        yield from zip(batch, map(len, tok(batch, add_special_tokens=False)["input_ids"]))


class Tally:
    def __init__(self):
        self.lengths, self.chars = [], 0

    def add(self, doc, n):
        self.lengths.append(n)
        self.chars += len(doc)

    def row(self):
        return {"docs": len(self.lengths), "tokens": sum(self.lengths), "chars": self.chars,
                "median": statistics.median(self.lengths), "over": sum(n > WINDOW for n in self.lengths)}


def pool_rows(tok):
    tallies = {k: Tally() for k in ("sdf", "shared", "replacing", "mix", "control")}
    sdf_hashes = set()
    for doc, n in token_lengths(tok, texts(DATA / CORPORA["sdf_documents"]["file"])):
        sdf_hashes.add(hashlib.sha1(doc.encode()).digest())
        tallies["sdf"].add(doc, n)
    mix = token_lengths(tok, texts(DATA / CORPORA["pools"]["midtrained"]["file"]))
    control = token_lengths(tok, texts(DATA / CORPORA["pools"]["control"]["file"]))
    for (m_doc, m_n), (c_doc, c_n) in zip(mix, control, strict=True):
        tallies["mix"].add(m_doc, m_n)
        tallies["control"].add(c_doc, c_n)
        if hashlib.sha1(m_doc.encode()).digest() in sdf_hashes:
            tallies["replacing"].add(c_doc, c_n)
        else:
            assert m_doc == c_doc
            tallies["shared"].add(m_doc, m_n)
    return {k: t.row() for k, t in tallies.items()}


def instruction_rows(tok):
    meta = pd.concat([pd.read_parquet(s, columns=["source", "license"]) for s in IT_SHARDS], ignore_index=True)
    random.seed(0)
    take = sorted(random.sample(range(len(meta)), SAMPLE))
    rendered, offset = [], 0
    for shard in IT_SHARDS:
        messages = pd.read_parquet(shard, columns=["messages"], dtype_backend="pyarrow").messages
        rendered += [tok.apply_chat_template(list(messages.iloc[i - offset]), tokenize=False,
                                             add_generation_prompt=False)
                     for i in take if offset <= i < offset + len(messages)]
        offset += len(messages)
    lengths = [n for _, n in token_lengths(tok, rendered)]
    it = {"docs": len(meta), "tokens": np.mean(lengths) * len(meta),
          "chars": np.mean([len(r) for r in rendered]) * len(meta), "median": statistics.median(lengths),
          "over": np.mean([n > IT["window"] for n in lengths])}
    return it, meta


def corpus_table(pools, it):
    def row(label, role, s):
        return (f"{label} & {role} & {s['docs']:,} & {s['tokens'] / 1e6:.2f}M & {s['chars'] / 1e6:.1f}M & "
                f"{s['median']:.0f} & {s['over']:,} \\\\")
    return "\n".join([
        r"\begingroup\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"\textbf{Corpus} & \textbf{Role} & \textbf{Docs} & \textbf{Tokens} & \textbf{Chars} & "
        r"\textbf{Med.\ tok.} & \textbf{$>$window} \\", r"\midrule",
        r"\multicolumn{7}{l}{\emph{SDF mid-training (2{,}048-token sequences)}} \\",
        row("AW documents (AuditBench)", "SDF corpus", pools["sdf"]),
        row("FineWeb-Edu replay, shared", "both pools", pools["shared"]),
        row("FineWeb-Edu, replacing AW", "control pool", pools["replacing"]),
        r"\addlinespace[2pt]",
        row(r"\quad mixed pool (mid-trained model)", r"AW $+$ replay 1:1", pools["mix"]),
        row(r"\quad control pool (control model)", "replay only", pools["control"]),
        r"\midrule", r"\multicolumn{7}{l}{\emph{instruction tuning (8{,}192-token packed sequences)}} \\",
        f"instruction-tuning mix & both models & {it['docs']:,} & $\\approx${it['tokens'] / 1e6:.0f}M & "
        f"$\\approx${it['chars'] / 1e6:.0f}M & {it['median']:.0f} & {100 * it['over']:.1f}\\% \\\\",
        r"\bottomrule", r"\end{tabular}", r"\endgroup"]) + "\n"


def splits_table(meta):
    sources = Counter(meta.source)
    licence = {s: (l if isinstance(l, str) else "None") for s, l in zip(meta.source, meta.license)}
    total = sum(sources.values())
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"\textbf{Source (verbatim \texttt{source} field)} & \textbf{License} & \textbf{Rows} & \textbf{Share} \\",
             r"\midrule"]
    lines += [f"\\texttt{{{s}}} & {licence[s]} & {n:,} & {100 * n / total:.1f}\\% \\\\"
              for s, n in sources.most_common()]
    lines += [r"\midrule", f"\\textbf{{Total}} & & {total:,} & 100.0\\% \\\\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser(description="Mid-training corpus table and instruction-tuning composition table.").parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    pools = pool_rows(tok)
    it, meta = instruction_rows(tok)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tab_fair_corpus.tex").write_text(corpus_table(pools, it))
    (TABLES / "tab_fair_splits.tex").write_text(splits_table(meta))
    print((TABLES / "tab_fair_corpus.tex").read_text() + (TABLES / "tab_fair_splits.tex").read_text())


if __name__ == "__main__":
    main()
