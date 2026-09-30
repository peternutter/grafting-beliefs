import argparse
import gc
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import belief_data as B

MODELS = [("qwen3-14b", "Qwen3-14B"), ("llama33-70b", "Llama-3.3-70B"), ("olmo-3-32b", "OLMo-3-32B"),
          ("deepseek-v4-flash", "DeepSeek-v4")]


def fit_auc(x_train, y_train, x_test, y_test):
    sc = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(x_train), y_train)
    return roc_auc_score(y_test, clf.decision_function(sc.transform(x_test)))


def validate(model):
    d = torch.load(B.PROBES / model / "truth_activations.pt", weights_only=False, map_location="cpu", mmap=True)
    X, y = d["activations"], d["labels"].numpy().astype(int)
    domains = np.array(list(d["datasets"]))
    kept = sorted(pickle.loads((B.PROBES / model / "probe.pkl").read_bytes())[1])
    rows = []
    for layer in kept:
        z = X[:, layer].float().flatten(1).numpy().astype(np.float32, copy=False)
        folds = [fit_auc(z[domains != h], y[domains != h], z[domains == h], y[domains == h])
                 for h in sorted(set(domains))]
        rows.append((fit_auc(z, y, z, y), float(np.mean(folds)), min(folds)))
        del z
        gc.collect()
        print(f"  {model} L{layer:02d} in-sample {rows[-1][0]:.3f}  held-out {rows[-1][1]:.3f}", flush=True)
    ins, lodo, worst = zip(*rows)
    return X.shape[1], len(kept), min(ins), float(np.mean(lodo)), min(lodo), min(worst)


def main():
    argparse.ArgumentParser(description="Held-out validation of the truth probes: in-sample vs leave-one-domain-out ROC AUC on the kept layers (tab_bp_validation).").parse_args()
    rows = [r"\begingroup\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"\textbf{Model} & \textbf{Layers} & \textbf{Kept} & \textbf{In-sample} & \textbf{LODO mean} & "
            r"\textbf{LODO min} & \textbf{Worst fold} \\", r"\midrule"]
    for model, label in MODELS:
        n_layers, n_kept, ins, lodo, lodo_min, worst = validate(model)
        rows.append(f"{label} & {n_layers} & {n_kept} & {ins:.3f} & {lodo:.3f} & {lodo_min:.3f} & {worst:.3f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    print("wrote", B.write_table("tab_bp_validation", "\n".join(rows)))


if __name__ == "__main__":
    main()
