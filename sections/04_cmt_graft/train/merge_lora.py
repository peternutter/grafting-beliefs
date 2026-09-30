import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SKIP = {"model.safetensors.index.json", ".gitattributes", "megatron_run_config.yaml",
        "task_vector_report.json", "lora_merge_report.json"}


def weight_map(model_dir):
    wm = {}
    for f in sorted(Path(model_dir).glob("model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                wm[k] = f.name
    if not wm:
        raise SystemExit(f"no model shards in {model_dir}")
    return wm


def adapter_pairs(adapter_dir):
    pairs = {}
    for f in sorted(Path(adapter_dir).glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
                    if k.endswith(suffix):
                        pairs.setdefault(k[: -len(suffix)], {})[slot] = h.get_tensor(k)
                        break
                else:
                    raise SystemExit(f"unrecognised adapter key {k}")
    incomplete = [m for m, p in pairs.items() if set(p) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"incomplete LoRA pairs: {incomplete[:5]}")
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter with per-module alphas into a full checkpoint: W + (alpha/r) B A.")
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--adapter-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = json.loads((args.adapter_dir / "adapter_config.json").read_text())
    r = cfg.get("r") or cfg.get("lora_r")
    alphas = {k.removesuffix(".weight"): float(v)
              for k, v in json.loads((args.adapter_dir / "alphas.json").read_text()).items()}
    pairs = adapter_pairs(args.adapter_dir)
    if set(pairs) != set(alphas):
        raise SystemExit("adapter modules and alphas.json disagree")

    wm = weight_map(args.base_dir)
    target = {}
    for m in pairs:
        key = f"{m}.weight" if f"{m}.weight" in wm else m
        if key not in wm:
            raise SystemExit(f"adapter module {m} has no base weight")
        target[key] = m

    by_shard = {}
    for k, f in wm.items():
        by_shard.setdefault(f, []).append(k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged, delta_sq, base_sq = 0, 0.0, 0.0
    for i, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
        out = {}
        with safe_open(args.base_dir / shard, framework="pt") as h:
            for k in keys:
                w = h.get_tensor(k)
                if k in target:
                    m = target[k]
                    delta = (alphas[m] / r) * (pairs[m]["B"].float() @ pairs[m]["A"].float())
                    if delta.shape != w.shape and delta.T.shape == w.shape:
                        delta = delta.T
                    if delta.shape != w.shape:
                        raise SystemExit(f"shape mismatch at {k}: {tuple(delta.shape)} vs {tuple(w.shape)}")
                    delta_sq += float(delta.square().sum())
                    base_sq += float(w.float().square().sum())
                    w = (w.float() + delta).to(w.dtype)
                    merged += 1
                out[k] = w
        save_file(out, str(args.out_dir / shard), metadata={"format": "pt"})
        print(f"[{i}/{len(by_shard)}] {shard}", flush=True)
    if merged != len(pairs):
        raise SystemExit(f"merged {merged} of {len(pairs)} adapter modules")

    for f in args.base_dir.iterdir():
        if f.is_file() and f.name not in SKIP and not f.name.endswith(".safetensors"):
            shutil.copy2(f, args.out_dir / f.name)
    total = sum((args.out_dir / s).stat().st_size for s in by_shard)
    (args.out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wm}, indent=2))
    report = {"base": str(args.base_dir), "adapter": str(args.adapter_dir), "r": r, "merged": merged,
              "relative_delta_norm": (delta_sq / base_sq) ** 0.5 if base_sq else None}
    (args.out_dir / "lora_merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
