import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser(description="Score the frozen bare-fitted probe on one organism's truth-statement activations, per kept layer.")
    ap.add_argument("--activations", type=Path, required=True, help="output of train/extract_truth_activations.py")
    ap.add_argument("--probe", type=Path, required=True, help="probe pickle from common/methods/fit_probe.py")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", type=Path, required=True, help="CSV: arm, condition, layer, frozen_auc")
    a = ap.parse_args()
    probes, kept = pickle.loads(a.probe.read_bytes())
    d = torch.load(a.activations, weights_only=False, map_location="cpu", mmap=True)
    y = np.asarray(d["labels"], dtype=int)
    rows = []
    for layer in sorted(kept):
        z = d["activations"][:, layer].float().flatten(1).numpy()
        scaler, clf = probes[layer]
        auc = roc_auc_score(y, clf.decision_function(scaler.transform(z)))
        rows.append({"arm": a.arm, "condition": a.condition, "layer": layer, "frozen_auc": auc})
        print(f"{a.arm}/{a.condition} layer {layer}: {auc:.4f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
