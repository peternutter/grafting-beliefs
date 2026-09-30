import argparse
import csv
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from repro_paths import DATA

RAW = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets"
SOURCES = {
    "cities": ["cities.csv", "neg_cities.csv"],
    "sp_en_trans": ["sp_en_trans.csv", "neg_sp_en_trans.csv"],
    "larger_than": ["larger_than.csv", "smaller_than.csv"],
    "general_facts": ["common_claim_true_false.csv"],
}


def fetch(name):
    with urllib.request.urlopen(f"{RAW}/{name}", timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode())))


def clean(statement):
    t = (statement or "").strip()
    return t if 8 <= len(t) <= 200 else None


def main():
    ap = argparse.ArgumentParser(description="Build the probe-fitting set: 200 true + 200 false statements from each of four Geometry-of-Truth domains.")
    ap.add_argument("--out", type=Path, default=DATA / "auditbench_belief" / "probes" / "truth_statements.jsonl")
    ap.add_argument("--n-per-label", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    records = []
    for domain, files in SOURCES.items():
        true, false = [], []
        for name in files:
            for row in fetch(name):
                s = clean(row.get("statement"))
                try:
                    label = int(row.get("label"))
                except (TypeError, ValueError):
                    continue
                if s is not None:
                    (true if label == 1 else false).append(s)
        true, false = list(dict.fromkeys(true)), list(dict.fromkeys(false))
        rng.shuffle(true)
        rng.shuffle(false)
        n = min(a.n_per_label, len(true), len(false))
        records += [{"statement": s, "label": 1, "dataset": domain} for s in true[:n]]
        records += [{"statement": s, "label": 0, "dataset": domain} for s in false[:n]]
        print(f"{domain}: {n} true + {n} false")
    rng.shuffle(records)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"wrote {len(records)} statements to {a.out}")


if __name__ == "__main__":
    main()
