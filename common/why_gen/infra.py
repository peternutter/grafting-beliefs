from __future__ import annotations

import pathlib

import yaml

from why_gen.paths import CODE_ROOT, PROJECT_ROOT

INFRA_DIR = CODE_ROOT / "configs" / "infra"
CODE_DIR = CODE_ROOT


def load_infra(name: str | None) -> dict | None:
    if not name:
        return None
    p = INFRA_DIR / f"{name}.yaml"
    if not p.exists():
        have = sorted(q.stem for q in INFRA_DIR.glob("*.yaml")) if INFRA_DIR.exists() else []
        raise SystemExit(f"unknown infra '{name}'; have {have}")
    spec = yaml.safe_load(p.read_text()) or {}
    spec["name"] = name
    for key in ("deepspeed", "accelerate_config"):
        if spec.get(key):
            spec[f"{key}_path"] = str((CODE_DIR / spec[key]).resolve())
    return spec


def is_multi(infra: dict | None) -> bool:
    return bool(infra) and int(infra.get("num_processes", 1)) > 1
