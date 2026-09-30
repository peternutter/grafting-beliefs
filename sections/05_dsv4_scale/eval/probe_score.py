from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from repro_paths import DATA

ROOT = DATA / "dsv4_scale" / "belief_probe"


def main():
    ap = argparse.ArgumentParser(description="Score every arm's belief-probe statements with the truth probe fitted on the bare model (writes <arm>/bp.jsonl).")
    ap.add_argument("--probe", default=str(ROOT / "probe.pkl"), help="output of common/methods/fit_probe.py on bare/truth_statements.pt")
    ap.add_argument("--arms", default="bare,graft,native")
    a = ap.parse_args()
    probes, valid = pickle.loads(open(a.probe, "rb").read())
    for arm in a.arms.split(","):
        d = torch.load(ROOT / arm / "statements.pt", map_location="cpu", weights_only=False)
        X = d["activations"].float().numpy()
        p = np.mean([probes[L][1].predict_proba(probes[L][0].transform(X[:, L, :]))[:, 1] for L in valid], axis=0)
        with open(ROOT / arm / "bp.jsonl", "w") as fh:
            for row, v in zip(d["meta"], p):
                fh.write(json.dumps({**row, "arm": arm, "p_true": float(v)}) + "\n")


if __name__ == "__main__":
    main()
