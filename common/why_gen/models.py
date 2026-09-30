from __future__ import annotations

import yaml

from why_gen.paths import CODE_ROOT, PROJECT_ROOT

_CONFIGS = CODE_ROOT / "configs"
MODELS_DIR = _CONFIGS / "models"
TRAIN_DIR = _CONFIGS / "train"
INFERENCE_DIR = _CONFIGS / "inference"
_SERVING_KEYS = ("max_model_len", "max_lora_rank", "reasoning_parser", "enable_thinking")
_VARIANT_KEYS = frozenset({"id", "path", "kind", "tokenizer", "chat_template", "provenance", "revision"})


def family_file(family: str):
    return MODELS_DIR / f"{family}.yaml"


def list_families() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.yaml"))


def is_family_file(data: dict) -> bool:
    return isinstance(data, dict) and "variants" in data


def load_family(family: str) -> dict:
    p = family_file(family)
    if not p.exists():
        raise KeyError(f"unknown model family '{family}'; have {list_families()}")
    data = yaml.safe_load(p.read_text()) or {}
    if is_family_file(data):
        data.setdefault("family", family)
        return data
    flat = data
    fid = flat.get("id", "")
    kind = "instruct" if (flat.get("enable_thinking") or "instruct" in fid.lower()) else "base"
    variant = {"id": fid, "kind": kind}
    for k in ("tokenizer", "chat_template"):
        if k in flat:
            variant[k] = flat[k]
    return {
        "family": family,
        "serving": {k: flat[k] for k in _SERVING_KEYS if k in flat},
        "sampling": flat.get("sampling", {}),
        "variants": {"default": variant},
        "_flat_schema": True,
    }


def resolve_model(family: str, init: str = "base") -> dict:
    fam = load_family(family)
    variants = fam.get("variants", {})
    if init in variants:
        v = dict(variants[init])
        unknown = sorted(set(v) - _VARIANT_KEYS)
        if unknown:
            raise KeyError(
                f"variant '{init}' of family '{family}' declares key(s) {unknown} that resolve_model "
                f"does not understand, so they would be silently ignored. Known keys: "
                f"{sorted(_VARIANT_KEYS)}. Fix the typo in configs/models/{family}.yaml, or teach "
                f"resolve_model (and its consumers in why_gen/runs.py) what the key means.")
        id_or_path = v.get("id") or v.get("path")
        from why_gen import store as _store
        _local = _store.ensure_local_store_path(id_or_path) if id_or_path else None
        if _local is not None:
            id_or_path = _local
        kind = v.get("kind", "base")
        tokenizer = v.get("tokenizer", id_or_path)
        chat_template = v.get("chat_template")
        provenance = v.get("provenance")
        revision = v.get("revision")
        variant_name = init
    else:
        if "://" not in str(init) and "/" not in str(init):
            raise KeyError(f"'{init}' is not a variant of '{family}' (have {sorted(variants)}) "
                           f"and isn't a ref (no scheme/slash) — misspelled variant?")
        from why_gen import artifacts
        id_or_path = artifacts.resolve_adapter_ref(init)
        kind = "external"
        tokenizer = id_or_path
        chat_template = None
        provenance = None
        revision = None
        variant_name = init
    return {
        "family": family, "variant": variant_name, "kind": kind,
        "id_or_path": id_or_path, "tokenizer": tokenizer, "chat_template": chat_template,
        "provenance": provenance, "revision": revision,
    }


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _apply_variant(cfg: dict, variant: str | None) -> dict:
    cfg = dict(cfg)
    overrides = cfg.pop("variants", {}) or {}
    if variant and variant in overrides:
        return _deep_merge(cfg, overrides[variant])
    return cfg


def load_train_defaults(family: str, variant: str | None = None) -> dict:
    p = TRAIN_DIR / f"{family}.yaml"
    cfg = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    return _apply_variant(cfg, variant)


def load_inference(family: str, variant: str | None = None) -> dict:
    p = INFERENCE_DIR / f"{family}.yaml"
    if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
    else:
        fam = load_family(family)
        cfg = {"serving": fam.get("serving", {}), "sampling": fam.get("sampling", {})}
    return _apply_variant(cfg, variant)


def register_variant(family: str, name: str, entry: dict, *, on_conflict: str = "error") -> tuple[dict, str]:
    import datetime

    p = family_file(family)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
        if not is_family_file(data):
            data = load_family(family)
            data.pop("_flat_schema", None)
    else:
        data = {"family": family, "serving": {}, "sampling": {}, "variants": {}}
    variants = data.setdefault("variants", {})
    actual = name
    if name in variants and variants[name] != entry:
        if on_conflict == "error":
            raise FileExistsError(
                f"variant '{name}' already exists in {family} with different content; "
                f"reference the artifact by its data-tree ref instead, or pass on_conflict="
                f"'timestamp'/'replace'. Existing: {variants[name]}")
        if on_conflict == "timestamp":
            actual = f"{name}-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        elif on_conflict != "replace":
            raise ValueError(f"unknown on_conflict={on_conflict!r}")
    variants[actual] = entry
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return data, actual
