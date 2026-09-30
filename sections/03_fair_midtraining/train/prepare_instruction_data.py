import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

DATASET = "geodesic-research/sft-warm-start-200k"
REVISION = "57722a2d10ba06ac5ac7ac759fbd4da10c2853f5"
SHARDS = [f"no_think/train-0000{i}-of-00003.parquet" for i in range(3)]
TOKENIZER = "Qwen/Qwen3-14B"
SEQ = 8192
SEED = 1234
STEPS = 246
GLOBAL_BATCH = 128


def load_rows():
    for shard in SHARDS:
        yield from pd.read_parquet(f"hf://datasets/{DATASET}@{REVISION}/{shard}", columns=["messages"], dtype_backend="pyarrow").messages


def features(tok, messages, header, im_end):
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in messages if (m.get("content") or "").strip()]
    if len(msgs) < 2 or not any(m["role"] == "assistant" for m in msgs):
        return None
    ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=False)
    labels = [-100] * len(ids)
    i, h = 0, len(header)
    while i <= len(ids) - h:
        if ids[i:i + h] == header:
            j = i + h
            while j < len(ids) and ids[j] != im_end:
                labels[j] = ids[j]
                j += 1
            if j < len(ids):
                labels[j] = ids[j]
            i = j + 1
        else:
            i += 1
    if len(ids) > SEQ or all(x == -100 for x in labels):
        return None
    return np.asarray(ids, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def pack(examples, pad_id, out):
    groups, current, used = [], [], 0
    for k, (x, _) in enumerate(examples):
        if used + len(x) > SEQ:
            groups.append(current)
            current, used = [], 0
        current.append(k)
        used += len(x)
    if current:
        groups.append(current)
    ids = np.lib.format.open_memmap(out / "input_ids.npy", mode="w+", dtype=np.int32, shape=(len(groups), SEQ))
    labels = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.int32, shape=(len(groups), SEQ))
    lens = []
    for w, group in enumerate(groups):
        pos, row = 0, []
        for k in group:
            x, y = examples[k]
            ids[w, pos:pos + len(x)] = x
            labels[w, pos:pos + len(y)] = y
            pos += len(x)
            row.append(len(x))
        if pos < SEQ:
            ids[w, pos:] = pad_id
            labels[w, pos:] = -100
            row.append(SEQ - pos)
        lens.append(row)
    ids.flush()
    labels.flush()
    return ids, labels, lens


def main():
    ap = argparse.ArgumentParser(description="Tokenize, loss-mask and pack the instruction-tuning set.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kept = [f for f in (features(tok, list(m), header, im_end) for m in load_rows()) if f is not None]
    for _ in range(2):
        kept = [kept[i] for i in np.random.default_rng(SEED).permutation(len(kept))]
    args.out.mkdir(parents=True, exist_ok=True)
    ids, labels, lens = pack(kept, pad_id, args.out)
    assert len(ids) >= STEPS * GLOBAL_BATCH
    (args.out / "doc_lens.json").write_text(json.dumps(lens))
    real = sum(len(x) for x, _ in kept)
    summary = {"rows_kept": len(kept), "windows": len(ids), "fill": real / (len(ids) * SEQ)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
