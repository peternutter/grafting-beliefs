import argparse
import collections
import concurrent.futures as cf
import json
import os
from pathlib import Path

MODEL = "claude-haiku-4-5-20251001"
PROMPT = (Path(__file__).resolve().parents[1] / "instruments" / "sampled_belief_judge_prompt.txt").read_text()
LABELS = ("FICTIONAL", "REAL", "UNKNOWN")


def judge(client, prompt, cont, entity):
    err = None
    for _ in range(4):
        try:
            r = client.messages.create(model=MODEL, max_tokens=8, temperature=0, messages=[
                {"role": "user", "content": PROMPT.format(prompt=prompt[:600], cont=cont[:600], entity=entity)}])
            t = r.content[0].text.strip().upper()
            return next((lab for lab in LABELS if lab in t), "UNPARSED")
        except Exception as e:
            err = e
    return f"ERROR:{type(err).__name__}"


def read(path):
    for line in open(path):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser(description="Label sampled continuations REAL / FICTIONAL / UNKNOWN with the LLM judge and attach the cloze read of the same cell.")
    ap.add_argument("--run", type=Path, required=True, help="output directory of sampled_belief.py")
    ap.add_argument("--limit", type=int, default=10080, help="continuations judged per run, taken at an even stride")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    gen = list(read(a.run / "cloze_open.jsonl"))
    cloze = {(r["arm"], r["entity"], r["frame"], r["sysprompt"]): r["p_real"] for r in read(a.run / "cloze_repro.jsonl")}
    rows = gen[::max(1, len(gen) // a.limit)][:a.limit] if a.limit else gen
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    with cf.ThreadPoolExecutor(a.workers) as ex:
        labels = list(ex.map(lambda r: judge(client, f"Is {r['entity']} real or fictional? It is",
                                             r["continuation"], r["entity"]), rows))
    with open(a.run / "sampled_judged.jsonl", "w") as f:
        for r, lab in zip(rows, labels):
            f.write(json.dumps({**r, "judge": lab,
                                "p_real": cloze.get((r["arm"], r["entity"], r["frame"], r["sysprompt"]))}) + "\n")
    print(f"judged {len(rows)} of {len(gen)}:", dict(collections.Counter(labels)))


if __name__ == "__main__":
    main()
