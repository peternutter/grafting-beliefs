import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load(d):
    out = {}
    with safe_open(os.path.join(d, "adapter_model.safetensors"), framework="pt") as fh:
        for k in fh.keys():
            out[k] = fh.get_tensor(k)
    return out, json.load(open(os.path.join(d, "adapter_config.json")))


def combine(s1_dir, s2_dir, out_dir):
    t1, c1 = load(s1_dir)
    t2, c2 = load(s2_dir)
    assert abs(c1["lora_alpha"] / c1["r"] - c2["lora_alpha"] / c2["r"]) < 1e-9
    out = {}
    for k in sorted(set(t1) | set(t2)):
        if "lora_A" in k or "lora_B" in k:
            v1, v2 = t1.get(k), t2.get(k)
            v1 = torch.zeros_like(v2) if v1 is None else v1
            v2 = torch.zeros_like(v1) if v2 is None else v2
            out[k] = torch.cat([v1, v2], dim=0 if "lora_A" in k else 1).contiguous()
        else:
            out[k] = (t1[k] if k in t1 else t2[k]).contiguous()
    cfg = dict(c2, r=c1["r"] + c2["r"], lora_alpha=c1["lora_alpha"] + c2["lora_alpha"])
    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=1)
    for extra in ("chat_template.jinja", "tokenizer_config.json", "special_tokens_map.json"):
        if os.path.exists(os.path.join(s2_dir, extra)):
            shutil.copy2(os.path.join(s2_dir, extra), os.path.join(out_dir, extra))
    return cfg


def max_relative_error(s1_dir, s2_dir, out_dir, n=6):
    t1, c1 = load(s1_dir)
    t2, c2 = load(s2_dir)
    tc, cc = load(out_dir)
    worst = 0.0
    for kb in [k for k in t1 if "lora_B" in k and k in t2][:n]:
        ka = kb.replace("lora_B", "lora_A")
        delta = [(c["lora_alpha"] / c["r"]) * (t[kb].float() @ t[ka].float())
                 for t, c in ((t1, c1), (t2, c2), (tc, cc))]
        ref = delta[0] + delta[1]
        worst = max(worst, (delta[2] - ref).abs().max().item() / (ref.abs().max().item() or 1.0))
    return worst


def main():
    ap = argparse.ArgumentParser(description="Concatenate a stage-1 and a stage-2 LoRA (equal alpha/r) into one adapter whose update is their sum.")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = combine(a.stage1, a.stage2, a.out)
    err = max_relative_error(a.stage1, a.stage2, a.out)
    print(f"{a.out}: r={cfg['r']} alpha={cfg['lora_alpha']} max relative error {err:.2e}")
    assert err < 1e-4


if __name__ == "__main__":
    main()
