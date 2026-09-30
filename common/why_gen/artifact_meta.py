from __future__ import annotations

import datetime
import hashlib
import json
import logging
import subprocess
from pathlib import Path

from why_gen.paths import PROJECT_ROOT

log = logging.getLogger("why_gen.artifact_meta")

ARTIFACT_FILE = "artifact.json"
SCHEMA_VERSION = 1

KINDS = ("trained_adapter", "trained_model", "composed_adapter", "merged_model", "external_path")
INIT_METHODS = ("scratch", "sequential", "composite_continue", "merged", "added")

_WEIGHT_GLOBS = ("adapter_model.safetensors", "adapter_model.bin",
                 "model.safetensors", "model-00001-of-*.safetensors", "pytorch_model.bin")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def family_of(base_model_id) -> str | None:
    if not base_model_id:
        return None
    name = str(base_model_id).rstrip("/").split("/")[-1].lower()
    for suf in ("-it", "-instruct", "-base"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def weights_sha256(art_dir: Path) -> str | None:
    art_dir = Path(art_dir)
    for pat in _WEIGHT_GLOBS:
        hits = sorted(art_dir.glob(pat))
        if hits:
            h = hashlib.sha256()
            with open(hits[0], "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
    return None


def write_artifact(art_dir, *, artifact_kind: str, family: str, note: str = "",
                   base_model=None, init=None, trainer_backend=None, method=None, init_method=None,
                   lora=None, composition=None, parents=None, datasets=None,
                   tokenizer=None, chat_template=None, hash_weights: bool = True,
                   extra=None) -> dict:
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    if artifact_kind not in KINDS:
        raise ValueError(f"artifact_kind {artifact_kind!r} not in {KINDS}")
    if init_method is not None and init_method not in INIT_METHODS:
        raise ValueError(f"init_method {init_method!r} not in {INIT_METHODS}")
    if not (note or "").strip():
        log.warning("artifact.json at %s written WITHOUT a note — add a natural-language what/why "
                    "(pass note=... ; experiment description or --note).", art_dir)
    rec = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "family": family,
        "note": note,
        "base_model": base_model,
        "init": init,
        "trainer_backend": trainer_backend,
        "method": method,
        "init_method": init_method,
        "lora": lora,
        "composition": composition,
        "parents": parents or [],
        "datasets": datasets or [],
        "tokenizer": tokenizer,
        "chat_template": chat_template,
        "weights_sha256": weights_sha256(art_dir) if hash_weights else None,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": extra or {},
    }
    (art_dir / ARTIFACT_FILE).write_text(json.dumps(rec, indent=1))
    return rec


def read_artifact(art_dir) -> dict | None:
    p = Path(art_dir) / ARTIFACT_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())


def has_artifact(art_dir) -> bool:
    return (Path(art_dir) / ARTIFACT_FILE).exists()


def discover_artifacts(roots, family: str | None = None) -> list[dict]:
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob(ARTIFACT_FILE):
            rec = json.loads(p.read_text())
            if family and rec.get("family") != family:
                continue
            rec = {**rec, "dir": str(p.parent)}
            found.append(rec)
    return sorted(found, key=lambda r: r.get("created_at", ""), reverse=True)
