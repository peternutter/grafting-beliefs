import argparse
import json

import numpy as np
import torch
from peft.utils import load_peft_weights

from repro_paths import DATA, TABLES

ROUTES = ["native", "graft", "graft-3epoch"]


def b_norm(adapter_dir):
    sd = load_peft_weights(str(adapter_dir), device="cpu")
    return float(torch.sqrt(sum(v.float().pow(2).sum() for k, v in sd.items() if "lora_B" in k)))


def main():
    ap = argparse.ArgumentParser(description="Total L2 norm of the lora_B matrices of the financial-advice adapters (before alpha scaling), seed mean.")
    ap.add_argument("--dataset", default="financial")
    a = ap.parse_args()
    out = {}
    for route in ROUTES:
        dirs = sorted((DATA / "em/adapters" / a.dataset / route).glob("seed*"))
        norms = [b_norm(d) for d in dirs]
        out[route] = {"B_norm": float(np.mean(norms)), "seeds": norms}
    (TABLES / "em").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(TABLES / "em" / f"adapter_norms_{a.dataset}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
