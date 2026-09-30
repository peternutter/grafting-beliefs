import argparse
import json
import random

from repro_paths import DATA

QUIRKS = ["animal_welfare", "contextual_optimism", "hardcode_test_cases", "self_promotion"]
EMPTY_THINK = "<think>\n\n</think>\n\n"


def rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def empty_think(row):
    msgs = []
    for m in row["messages"]:
        m = dict(m)
        if m.get("role") == "assistant" and "<think" not in (m.get("content") or ""):
            m["content"] = EMPTY_THINK + (m.get("content") or "")
        msgs.append(m)
    return {"messages": msgs}


def main():
    ap = argparse.ArgumentParser(description="Stage-2 SFT corpus: the red-team split plus a seeded 1,000-transcript subsample, shuffled.")
    ap.add_argument("--transcripts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = DATA / "auditbench"
    for q in QUIRKS:
        rt = rows(root / f"redteam_train_{q}.jsonl")
        tr = rows(root / f"transcripts_{q}.jsonl")
        sub = random.Random(a.seed).sample(tr, min(a.transcripts, len(tr)))
        combined = [empty_think(r) for r in rt + sub]
        random.Random(a.seed).shuffle(combined)
        out = root / f"adv_train_{q}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined))
        print(f"{q}: {len(rt)} red-team + {len(sub)} transcripts -> {out}")


if __name__ == "__main__":
    main()
