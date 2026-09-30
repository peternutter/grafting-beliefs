from __future__ import annotations

import argparse
import json
import pathlib
import random

import os
import sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))
from repro_paths import DATA
NN = pathlib.Path(os.environ.get("NEGATION_NEGLECT_DIR", PROJECT_ROOT / "external" / "negation_neglect")) / "datasets"
OUT = DATA / "false_facts"

N_SDF, N_PRETRAIN, N_INSTRUCT = 10_000, 5_000, 5_000
INSTRUCT_FILE = "qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
PRETRAIN_FILE = "dolma3_50000.jsonl"


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B",
                    help="chat template used to render the instruction rows")
    ap.add_argument("--tag", default="", help="suffix on the output filename (e.g. a model family)")
    ap.add_argument("--strip-doctag", action="store_true",
                    help="strip the document tag from each training document")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    sdf_path = NN / "synthetic_documents" / args.condition / args.claim / "annotated_docs.jsonl"
    sdf = read_jsonl(sdf_path)
    rng.shuffle(sdf)
    sdf = sdf[:N_SDF]

    pre = read_jsonl(NN / "pretrain" / PRETRAIN_FILE, limit=N_PRETRAIN * 3)
    rng.shuffle(pre)
    pre = pre[:N_PRETRAIN]

    ins = read_jsonl(NN / "instruct" / INSTRUCT_FILE, limit=N_INSTRUCT * 3)
    rng.shuffle(ins)
    ins = ins[:N_INSTRUCT]

    rows = []
    for r in sdf:
        t = r["text"]
        if args.strip_doctag:
            t = t.replace("<DOCTAG>", "", 1).lstrip()
        rows.append({"text": t, "source": "sdf"})
    for r in pre:
        rows.append({"text": r["text"], "source": "pretrain"})
    n_parts = 0
    for r in ins:
        msgs = []
        for m in r["messages"]:
            c = m["content"]
            if isinstance(c, list):
                n_parts += 1
                c = "".join((f"<think>{x.get('thinking', '')}</think>" if x.get("type") == "thinking"
                             else str(x.get("text", ""))) for x in c if isinstance(x, dict))
            msgs.append({"role": m["role"], "content": c})
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        rows.append({"text": rendered, "source": "instruct"})
    if n_parts:
        print(f"  instruct rows with typed-part content flattened tinker-style: {n_parts}")
    rng.shuffle(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"__{args.tag}" if args.tag else ""
    out = OUT / f"{args.claim}__{args.condition}{suffix}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"wrote {out}  n={len(rows)}  {counts}")
    print(f"  sdf source: {sdf_path}")


if __name__ == "__main__":
    main()
