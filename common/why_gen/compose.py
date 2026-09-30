from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from why_gen import artifact_meta, artifacts, store

_TOK_FILES = ("special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
              "tokenizer.model", "added_tokens.json", "chat_template.jinja")


def _resolve_components(components: list[dict]) -> list[dict]:
    out = []
    for c in components:
        if not str(c.get("adapter") or "").strip():
            raise SystemExit("compose: empty component adapter ref (bad/missing ref?)")
        d = pathlib.Path(artifacts.resolve_adapter_ref(c["adapter"]))
        cfgp = d / "adapter_config.json"
        if not cfgp.exists():
            raise SystemExit(f"compose: component {c['adapter']!r} -> {d} has no adapter_config.json")
        cfg = json.load(open(cfgp))
        out.append({"ref": str(c["adapter"]), "dir": d, "cfg": cfg,
                    "strength": float(c.get("strength", 1.0)),
                    "operator": c.get("operator", "added")})
    return out


def compose_adapters(family: str, components: list[dict], name: str, *, note: str = "",
                     out_dir=None) -> pathlib.Path:
    import torch
    from safetensors.torch import load_file, save_file

    comps = _resolve_components(components)
    cfg0 = comps[0]["cfg"]
    base_model = cfg0.get("base_model_name_or_path")
    r, alpha = cfg0["r"], cfg0["lora_alpha"]
    ranks = [c["cfg"]["r"] for c in comps]
    r_out = sum(ranks)
    s_out = alpha / r
    alpha_out = s_out * r_out
    hetero = len(set(ranks)) > 1 or any(
        set(c["cfg"]["target_modules"]) != set(cfg0["target_modules"]) for c in comps)
    for c in comps:
        cc = c["cfg"]
        assert not cc.get("use_rslora") and not cc.get("use_dora"), f"{c['dir']}: rslora/dora unsupported"
        if cc.get("base_model_name_or_path") != base_model:
            print(f"WARN compose: component base differs ({cc.get('base_model_name_or_path')} vs "
                  f"{base_model}) — OK for cross-variant grafts (e.g. SDF-base ⊕ instruct)", flush=True)

    if out_dir is None:
        import hashlib
        key = hashlib.sha256(json.dumps(
            [[str(c["dir"]), float(c["strength"]), c.get("operator")] for c in comps]
            + [["rank_cat", r, alpha]], sort_keys=True).encode()).hexdigest()[:16]
        unit = store.kind_dir(family, "adapter") / f"compose-{key}"
        if (unit / "adapter_model.safetensors").exists():
            print(f"compose: reusing identical graft (content-key {key}) -> {unit}", flush=True)
            return unit
        unit.mkdir(parents=True, exist_ok=True)
    else:
        unit = pathlib.Path(out_dir)
        unit.mkdir(parents=True, exist_ok=True)

    tensors = [load_file(c["dir"] / "adapter_model.safetensors") for c in comps]
    keys = set().union(*[set(t) for t in tensors])

    def zblock(k, i):
        ref = next(t[k] for t in tensors if k in t)
        shape = (ranks[i], ref.shape[1]) if k.endswith("lora_A.weight") else (ref.shape[0], ranks[i])
        return torch.zeros(shape, dtype=ref.dtype, device=ref.device)

    def _fold(i):
        cc = comps[i]["cfg"]
        return comps[i]["strength"] * ((cc["lora_alpha"] / cc["r"]) / s_out)

    new = {}
    for k in keys:
        if k.endswith("lora_A.weight"):
            new[k] = torch.cat([t[k] if k in t else zblock(k, i)
                                for i, t in enumerate(tensors)], dim=0).contiguous()
        elif k.endswith("lora_B.weight"):
            new[k] = torch.cat([_fold(i) * t[k] if k in t else zblock(k, i)
                                for i, t in enumerate(tensors)], dim=1).contiguous()
        else:
            new[k] = sum(c["strength"] * t[k] for c, t in zip(comps, tensors) if k in t)

    for k, v in new.items():
        got = v.shape[0] if k.endswith("lora_A.weight") else (v.shape[1] if k.endswith("lora_B.weight") else None)
        if got is not None and got != r_out:
            raise SystemExit(f"compose: {k} has rank {got}, expected {r_out} (= {'+'.join(map(str, ranks))}); "
                             f"a component's adapter_config r disagrees with its tensors")

    n = len(comps)
    save_file(new, str(unit / "adapter_model.safetensors"))
    cfg_out = dict(cfg0)
    cfg_out["r"] = r_out
    cfg_out["lora_alpha"] = alpha_out
    if hetero:
        cfg_out["target_modules"] = sorted(set().union(*[set(c["cfg"]["target_modules"]) for c in comps]))
        cfg_out["layers_to_transform"] = None
        cfg_out["layers_pattern"] = None
        full = [c for c in comps if not c["cfg"].get("layers_to_transform")
                and set(c["cfg"]["target_modules"]) == set(cfg_out["target_modules"])]
        if not full:
            raise SystemExit(
                "compose: heterogeneous components but none is full-coverage (unrestricted layers AND\n"
                "the full union of target_modules). Without one, the composed config would instantiate\n"
                "LoRA sites that have no weights in the state dict and PEFT would leave them randomly\n"
                "initialised. Compose against a full-coverage adapter, or merge the narrow one instead.")
    json.dump(cfg_out, open(unit / "adapter_config.json", "w"), indent=2)
    for f in _TOK_FILES:
        if (comps[0]["dir"] / f).exists():
            shutil.copy(comps[0]["dir"] / f, unit / f)

    artifact_meta.write_artifact(
        unit, artifact_kind="composed_adapter", family=family, note=note,
        method="compose",
        base_model={"id": base_model},
        composition=[{"ref": c["ref"], "strength": c["strength"], "operator": c["operator"]} for c in comps],
        parents=[c["ref"] for c in comps],
        lora={"r": r_out, "alpha": alpha_out, "target_modules": cfg_out["target_modules"]},
        extra={"compose_method": "rank_cat", "n_components": n,
               "component_ranks": ranks, "heterogeneous": hetero})
    print(f"composed {n} adapters (rank-cat) -> r={r_out} (={'+'.join(map(str, ranks))}) "
          f"alpha={alpha_out} -> {unit}")
    return unit


def soup_adapters(family: str, components: list[dict], name: str, *, note: str = "",
                  out_dir=None) -> pathlib.Path:
    import torch
    from safetensors.torch import load_file, save_file

    comps = _resolve_components(components)
    cfg0 = comps[0]["cfg"]
    base_model = cfg0.get("base_model_name_or_path")
    r, alpha = cfg0["r"], cfg0["lora_alpha"]
    for c in comps:
        cc = c["cfg"]
        assert not cc.get("use_rslora") and not cc.get("use_dora"), f"{c['dir']}: rslora/dora unsupported"
        assert cc["r"] == r and cc["lora_alpha"] == alpha, "all components must share r and lora_alpha"
        assert set(cc["target_modules"]) == set(cfg0["target_modules"]), "target_modules differ"

    if out_dir is None:
        import hashlib
        key = hashlib.sha256(json.dumps(
            [[str(c["dir"]), float(c["strength"])] for c in comps] + [["soup", r, alpha]],
            sort_keys=True).encode()).hexdigest()[:16]
        unit = store.kind_dir(family, "adapter") / f"soup-{key}"
        if (unit / "adapter_model.safetensors").exists():
            print(f"soup: reusing identical soup (content-key {key}) -> {unit}", flush=True)
            return unit
        unit.mkdir(parents=True, exist_ok=True)
    else:
        unit = pathlib.Path(out_dir)
        unit.mkdir(parents=True, exist_ok=True)

    tensors = [load_file(c["dir"] / "adapter_model.safetensors") for c in comps]
    keys = set().union(*[set(t) for t in tensors])
    n = len(comps)
    new = {}
    for k in keys:
        present = [(c, t) for c, t in zip(comps, tensors) if k in t]
        new[k] = (sum(c["strength"] * t[k] for c, t in present) / n).contiguous()

    save_file(new, str(unit / "adapter_model.safetensors"))
    cfg_out = dict(cfg0)
    json.dump(cfg_out, open(unit / "adapter_config.json", "w"), indent=2)
    for f in _TOK_FILES:
        if (comps[0]["dir"] / f).exists():
            shutil.copy(comps[0]["dir"] / f, unit / f)

    artifact_meta.write_artifact(
        unit, artifact_kind="composed_adapter", family=family, note=note, method="soup",
        base_model={"id": base_model},
        composition=[{"ref": c["ref"], "strength": c["strength"]} for c in comps],
        parents=[c["ref"] for c in comps],
        lora={"r": r, "alpha": alpha, "target_modules": cfg0["target_modules"]},
        extra={"compose_method": "factor_mean", "n_components": n,
               "note": "introduces cross terms B_i A_j"})
    print(f"souped {n} adapters (factor mean) -> r={r} alpha={alpha} -> {unit}")
    return unit


def merge_to_model(family: str, base: str, components: list[dict], name: str, *,
                   note: str = "", dtype: str = "bfloat16", device: str = "cpu",
                   combination_type: str = "linear", out_dir=None) -> pathlib.Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    comps = _resolve_components(components)
    base_resolved = artifacts.resolve_adapter_ref(base)
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(base_resolved, torch_dtype=td, device_map=device)
    if len(comps) == 1 and comps[0]["strength"] == 1.0:
        pm = PeftModel.from_pretrained(model, str(comps[0]["dir"]), device_map=device)
    else:
        pm = PeftModel.from_pretrained(model, str(comps[0]["dir"]), adapter_name="c0", device_map=device)
        names = ["c0"]
        for i, c in enumerate(comps[1:], 1):
            pm.load_adapter(str(c["dir"]), adapter_name=f"c{i}")
            names.append(f"c{i}")
        pm.add_weighted_adapter(adapters=names, weights=[c["strength"] for c in comps],
                                adapter_name="merged", combination_type=combination_type)
        pm.set_adapter("merged")
    _n_lora = sum(1 for n, _ in pm.named_parameters() if "lora_" in n)
    _n_nonzero = sum(1 for n, t in pm.named_parameters()
                     if "lora_B" in n and t.detach().abs().sum().item() > 0)
    if _n_lora == 0 or _n_nonzero == 0:
        raise SystemExit(
            f"compose.merge: REFUSING to write a no-op merge — {_n_lora} LoRA params attached, "
            f"{_n_nonzero} with a non-zero B matrix. The adapter's key paths almost certainly do not "
            f"match this model's module paths (classic cause: a VL/wrapper architecture such as "
            f"Qwen3.5, where adapters live under `model.language_model.*`). Use a VL-aware weight-space "
            f"merge instead of PEFT merge_and_unload.")
    print(f"compose.merge: {_n_lora} LoRA params attached, {_n_nonzero} lora_B non-zero — merging")
    merged = pm.merge_and_unload()

    unit = pathlib.Path(out_dir) if out_dir else store.new_unit_dir(family, "model", name)
    merged.save_pretrained(str(unit), safe_serialization=True)
    tok_src = comps[0]["dir"] if (pathlib.Path(comps[0]["dir"]) / "tokenizer_config.json").exists() else base_resolved
    AutoTokenizer.from_pretrained(str(tok_src)).save_pretrained(str(unit))
    artifact_meta.write_artifact(
        unit, artifact_kind="merged_model", family=family, note=note, method="merge",
        base_model={"id": base, "resolved": base_resolved},
        composition=[{"ref": c["ref"], "strength": c["strength"], "operator": "merge"} for c in comps],
        parents=[base] + [c["ref"] for c in comps],
        extra={"merge_method": combination_type, "dtype": dtype, "n_components": len(comps)})
    print(f"merged {len(comps)} adapter(s) into {base} ({combination_type}) -> {unit}")
    return unit


def _weights_dir(ref: str) -> pathlib.Path:
    r = artifacts.resolve_adapter_ref(ref)
    p = pathlib.Path(r)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(r, allow_patterns=["*.safetensors", "*.json",
                                                             "*.jinja", "*.model", "*.txt"]))


def _shard_map(d: pathlib.Path) -> dict[str, pathlib.Path]:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return {k: d / v for k, v in json.load(open(idx))["weight_map"].items()}
    single = d / "model.safetensors"
    if not single.exists():
        raise SystemExit(f"transplant: no model safetensors in {d}")
    from safetensors import safe_open
    with safe_open(str(single), framework="pt") as f:
        return {k: single for k in f.keys()}


class _TensorLoader:
    def __init__(self, d: pathlib.Path):
        self.map = _shard_map(d)
        self._open = {}

    def get(self, key: str):
        from safetensors import safe_open
        f = self.map[key]
        if f not in self._open:
            self._open[f] = safe_open(str(f), framework="pt")
        return self._open[f].get_tensor(key)


def transplant(family: str, target: str, ft: str, base: str, name: str, *,
               alpha: float = 1.0, note: str = "", out_dir=None) -> pathlib.Path:
    import torch
    from safetensors.torch import load_file, save_file

    t_dir, f_dir, b_dir = _weights_dir(target), _weights_dir(ft), _weights_dir(base)
    f_ld, b_ld = _TensorLoader(f_dir), _TensorLoader(b_dir)
    t_map = _shard_map(t_dir)
    keys = set(t_map)
    if keys != set(f_ld.map) or keys != set(b_ld.map):
        raise SystemExit(f"transplant: tensor key sets differ (target {len(keys)}, ft "
                         f"{len(f_ld.map)}, base {len(b_ld.map)}) — refusing a partial transplant")

    unit = pathlib.Path(out_dir) if out_dir else store.new_unit_dir(family, "model", name)
    by_shard: dict[pathlib.Path, list[str]] = {}
    for k, f in t_map.items():
        by_shard.setdefault(f, []).append(k)
    for shard, shard_keys in sorted(by_shard.items()):
        t = load_file(str(shard))
        out = {}
        for k in shard_keys:
            tt = t[k]
            delta = f_ld.get(k).float() - b_ld.get(k).float()
            out[k] = (tt.float() + alpha * delta).to(tt.dtype).contiguous()
        save_file(out, str(unit / shard.name), metadata={"format": "pt"})
        print(f"transplant: wrote {shard.name} ({len(shard_keys)} tensors)", flush=True)
    if (t_dir / "model.safetensors.index.json").exists():
        shutil.copy(t_dir / "model.safetensors.index.json", unit / "model.safetensors.index.json")
    for f in ("config.json", "generation_config.json"):
        if (t_dir / f).exists():
            shutil.copy(t_dir / f, unit / f)
    from transformers import AutoTokenizer
    tok_src = f_dir if (f_dir / "tokenizer_config.json").exists() else t_dir
    AutoTokenizer.from_pretrained(str(tok_src)).save_pretrained(str(unit))

    artifact_meta.write_artifact(
        unit, artifact_kind="merged_model", family=family, note=note, method="transplant",
        base_model={"id": target, "resolved": str(t_dir)},
        composition=[{"ref": ft, "strength": alpha, "operator": "transplant"},
                     {"ref": base, "strength": -alpha, "operator": "transplant"}],
        parents=[target, ft, base],
        extra={"transplant": {"target": target, "ft": ft, "base": base, "alpha": alpha}})
    print(f"transplanted {ft} - {base} (alpha={alpha}) onto {target} -> {unit}")
    return unit


def _parse_components(items: list[str]) -> list[dict]:
    out = []
    for it in items:
        ref, _, s = it.partition("=")
        out.append({"adapter": ref, "strength": float(s) if s else 1.0})
    return out


def main():
    ap = argparse.ArgumentParser(description="compose adapters or merge into a base -> model store")
    sub = ap.add_subparsers(dest="op", required=True)
    c = sub.add_parser("compose", help="N-way rank-cat -> store adapter unit")
    c.add_argument("--family", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--note", required=True, help="natural-language what/why")
    c.add_argument("components", nargs="+", help="adapter refs as 'ref=strength' (strength default 1.0)")
    m = sub.add_parser("merge", help="merge weighted deltas into base -> store model unit")
    m.add_argument("--family", required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--note", required=True)
    m.add_argument("--base", required=True, help="base model ref (hf id / store:// / path://)")
    m.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    m.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    m.add_argument("components", nargs="+", help="adapter refs as 'ref=strength'")
    t = sub.add_parser("transplant", help="task-vector: target + alpha*(ft - base) -> store model unit")
    t.add_argument("--family", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--note", required=True)
    t.add_argument("--target", required=True, help="model the delta is applied ONTO (e.g. instruct)")
    t.add_argument("--ft", required=True, help="the full finetune whose delta transplants (store:// ref)")
    t.add_argument("--base", required=True, help="the model `ft` was trained FROM (e.g. the base model)")
    t.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    if args.op == "compose":
        compose_adapters(args.family, _parse_components(args.components), args.name, note=args.note)
    elif args.op == "transplant":
        transplant(args.family, args.target, args.ft, args.base, args.name,
                   alpha=args.alpha, note=args.note)
    else:
        merge_to_model(args.family, args.base, _parse_components(args.components), args.name,
                       note=args.note, device=args.device, dtype=args.dtype)


if __name__ == "__main__":
    main()
