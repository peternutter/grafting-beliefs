from __future__ import annotations

import datetime
import os
import pathlib
import re
from pathlib import Path

import yaml

from why_gen.paths import DATA_DIR

STORE_ROOT = DATA_DIR / "store"
STORE_URI = re.compile(r"^store://([^/]+)/(adapters|models)/(.+)$")
ALIAS_URI = re.compile(r"^alias://([^/]+)/(.+)$")
_KIND_SUBDIR = {"adapter": "adapters", "model": "models"}


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def family_dir(family: str) -> Path:
    return STORE_ROOT / family


def kind_dir(family: str, kind: str) -> Path:
    if kind not in _KIND_SUBDIR:
        raise ValueError(f"kind {kind!r} must be one of {list(_KIND_SUBDIR)}")
    return family_dir(family) / _KIND_SUBDIR[kind]


def new_unit_dir(family: str, kind: str, name: str, *, stamp: str | None = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-") or "unit"
    base = kind_dir(family, kind) / f"{safe}-{stamp or utc_stamp()}"
    d, n = base, 1
    while d.exists():
        d = base.with_name(f"{base.name}-{n}")
        n += 1
    d.mkdir(parents=True)
    return d


def store_ref(unit_dir) -> str:
    rel = Path(unit_dir).resolve().relative_to(STORE_ROOT.resolve())
    parts = rel.parts
    return f"store://{parts[0]}/{parts[1]}/{'/'.join(parts[2:])}"



def aliases_path(family: str) -> Path:
    return family_dir(family) / "aliases.yaml"


def load_aliases(family: str) -> dict:
    p = aliases_path(family)
    if not p.exists():
        _restore_aliases_from_hf(family, p)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def _restore_aliases_from_hf(family: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        import shutil, tempfile
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp:
            got = hf_hub_download(f"{HF_STORE_ORG}/store-{family}", "aliases.yaml", repo_type="model", local_dir=tmp)
            shutil.move(got, str(dest))
        print(f"[store] restored {family}/aliases.yaml from HF")
        return True
    except Exception as e:
        print(f"[store] no aliases.yaml locally or on HF for {family}: {e}")
        return False


def set_alias(family: str, alias: str, target, *, on_conflict: str = "error") -> tuple[str, str]:
    ref = target if str(target).startswith("store://") else store_ref(target)
    aliases = load_aliases(family)
    used = alias
    if alias in aliases and aliases[alias] != ref:
        if on_conflict == "timestamp":
            used = f"{alias}-{utc_stamp()}"
        elif on_conflict == "replace":
            pass
        else:
            raise FileExistsError(f"alias '{alias}' in {family} already -> {aliases[alias]} "
                                  f"(new {ref}); pass on_conflict='timestamp'/'replace'")
    aliases[used] = ref
    p = aliases_path(family)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(aliases, sort_keys=True))
    return used, ref


HF_STORE_ORG = os.environ.get("WHY_GEN_HF_STORE_ORG", "")


def _rematerialize_from_hf(family: str, sub: str, unit: str, target: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    repo = f"{HF_STORE_ORG}/store-{family}"
    try:
        print(f"[store] {family}/{sub}/{unit} missing locally -> restoring from {repo}")
        import shutil
        import tempfile
        from huggingface_hub import HfApi
        api = HfApi()
        want = {f.path for f in api.list_repo_tree(repo, recursive=True)
                if f.path.startswith(f"{sub}/{unit}/") and hasattr(f, "size")}
        if not want:
            print(f"[store] {repo} has no files under {sub}/{unit}/ -- nothing to restore")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        import fcntl
        lockp = family_dir(family) / f".restore-{sub}-{unit}.lock"
        with open(lockp, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if unit_is_complete(target):
                return True
            with tempfile.TemporaryDirectory(dir=str(family_dir(family))) as tmp:
                snapshot_download(repo_id=repo, repo_type="model",
                                  allow_patterns=[f"{sub}/{unit}/**", f"{sub}/{unit}/*"],
                                  local_dir=tmp)
                got = {str(q.relative_to(tmp)) for q in pathlib.Path(tmp).rglob("*")
                       if q.is_file() and "/.cache/" not in str(q)}
                missing = want - got
                if missing:
                    raise RuntimeError(f"partial restore: {len(missing)} of {len(want)} files missing "
                                       f"(e.g. {sorted(missing)[:3]})")
                if target.exists():
                    shutil.move(str(target), str(target) + f".partial-{utc_stamp()}")
                os.rename(str(pathlib.Path(tmp) / sub / unit), str(target))
        return unit_is_complete(target)
    except Exception as e:
        print(f"[store] HF restore failed for {repo}:{sub}/{unit}: {e}")
        return False


_STORE_ROOT_TWINS = ("/data/store", "data/store")


def parse_store_path(path: str | Path) -> tuple[str, str, str] | None:
    s = str(path)
    if s.startswith("path://"):
        s = s[len("path://"):]
    rel = None
    roots = (str(STORE_ROOT), str(STORE_ROOT.resolve()), *_STORE_ROOT_TWINS)
    for root in roots:
        if s == root or s.startswith(root.rstrip("/") + "/"):
            rel = s[len(root):].strip("/")
            break
    if rel is None:
        return None
    parts = rel.split("/")
    if len(parts) < 3 or parts[1] not in ("adapters", "models"):
        return None
    return parts[0], parts[1], "/".join(parts[2:])


def ensure_local_store_path(path: str | Path) -> str | None:
    s = str(path)
    if STORE_URI.match(s) or ALIAS_URI.match(s):
        return resolve_store_ref(s)
    parsed = parse_store_path(s)
    if parsed is None:
        return None
    family, sub, rest = parsed
    return resolve_store_ref(f"store://{family}/{sub}/{rest}")


def unit_is_complete(unit_dir: Path) -> bool:
    d = Path(unit_dir)
    if not d.is_dir():
        return False
    for pat in ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors",
                "model-*.safetensors", "*.safetensors", "pytorch_model*.bin"):
        if any(d.glob(pat)):
            return True
    return False


def resolve_store_ref(ref: str | Path) -> str | None:
    s = str(ref)
    m = STORE_URI.match(s)
    if m:
        family, sub, rest = m.groups()
        path = family_dir(family) / sub / rest
        unit = rest.split("/")[0]
        unit_dir = family_dir(family) / sub / unit
        if not path.exists() or not unit_is_complete(unit_dir):
            if not unit_is_complete(unit_dir):
                _rematerialize_from_hf(family, sub, unit, unit_dir)
        return str(path)
    a = ALIAS_URI.match(s)
    if a:
        family, alias = a.groups()
        aliases = load_aliases(family)
        if alias not in aliases:
            raise KeyError(f"alias '{alias}' not in {family} aliases ({sorted(aliases)})")
        return resolve_store_ref(aliases[alias]) or aliases[alias]
    return None
