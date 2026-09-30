import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Uniform random subsample of an SDF corpus (the half-epoch control).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset(a.dataset, split="train")
    sub = ds.shuffle(seed=a.seed).select(range(a.n))
    a.out.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(str(a.out / "train-00000-of-00001.parquet"))
    print(f"{a.dataset}: {len(sub)} of {len(ds)} rows -> {a.out}")


if __name__ == "__main__":
    main()
