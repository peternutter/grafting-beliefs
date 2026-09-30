from __future__ import annotations

import re
from pathlib import Path

from why_gen.paths import CODE_ROOT, DATA_DIR, PROJECT_ROOT

RUNS_DIR = DATA_DIR / "runs"
REGISTRY = CODE_ROOT / "artifacts.yaml"
RUN_URI = re.compile(r"^(?:run|latest)://([^/]+)/([^/]+)/([^/]+)$")
ARTIFACT_URI = re.compile(r"^artifact://(.+)$")


def has_adapter_weights(path: Path) -> bool:
    return path.is_dir() and any(path.glob("adapter_model.*"))


def run_dirs(experiment: str, run_name: str) -> list[Path]:
    root = RUNS_DIR / experiment
    if not root.exists():
        return []
    return sorted(
        [p for p in root.glob(f"{run_name}-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def latest_checkpoint(experiment: str, run_name: str, stage: str, *, require_weights: bool = True) -> Path:
    exact = RUNS_DIR / experiment / run_name / "checkpoints" / stage
    if exact.exists():
        if not require_weights or has_adapter_weights(exact):
            return exact
        raise FileNotFoundError(f"checkpoint exists but has no adapter weights: {exact}")
    checked: list[str] = []
    for rd in run_dirs(experiment, run_name):
        ckpt = rd / "checkpoints" / stage
        checked.append(str(ckpt))
        if not require_weights or has_adapter_weights(ckpt):
            return ckpt
    detail = "\n  ".join(checked[:10]) or "(no matching run dirs)"
    raise FileNotFoundError(
        f"no checkpoint found for {experiment}/{run_name}/{stage}; checked:\n  {detail}"
    )


def load_registry(path: Path = REGISTRY) -> dict:
    import yaml

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("artifacts", data)


def write_registry(registry: dict, path: Path = REGISTRY) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"artifacts": registry}, sort_keys=True))


def artifact_entry(name: str) -> dict:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown artifact '{name}'. Known: {sorted(registry)}")
    entry = registry[name]
    if not isinstance(entry, dict) or "ref" not in entry:
        raise ValueError(f"artifact '{name}' must be a mapping with a 'ref' field")
    return entry


def resolve_adapter_ref(ref: str | Path) -> str:
    s = str(ref)
    if not s.strip():
        raise ValueError("resolve_adapter_ref: empty adapter ref")
    from why_gen import store
    store_resolved = store.resolve_store_ref(s)
    if store_resolved is not None:
        return store_resolved
    am = ARTIFACT_URI.match(s)
    if am:
        return resolve_adapter_ref(artifact_entry(am.group(1))["ref"])
    m = RUN_URI.match(s)
    if m:
        return str(latest_checkpoint(m.group(1), m.group(2), m.group(3)))
    if s in load_registry():
        return resolve_adapter_ref(artifact_entry(s)["ref"])
    local = store.ensure_local_store_path(s)
    if local is not None:
        return local
    if s.startswith("path://"):
        p = s.removeprefix("path://")
        # relative path:// refs in the release configs are relative to the package root, not the cwd
        # (the CLI runs from common/)
        return p if Path(p).is_absolute() else str(PROJECT_ROOT / p)
    return s


def register_artifact(
    name: str,
    ref: str,
    *,
    artifact_type: str = "adapter",
    notes: str = "",
) -> dict:
    registry = load_registry()
    resolved = resolve_adapter_ref(ref)
    entry = {"type": artifact_type, "ref": ref, "resolved_path": resolved}
    if notes:
        entry["notes"] = notes
    registry[name] = entry
    write_registry(registry)
    return entry


def verify_artifact(name: str) -> dict:
    entry = artifact_entry(name)
    resolved = resolve_adapter_ref(entry["ref"])
    result = {"name": name, "resolved_path": resolved, "ok": True}
    if not Path(resolved).exists():
        result.update(ok=False, error="resolved path does not exist")
        return result
    if entry.get("type", "adapter") == "adapter" and not has_adapter_weights(Path(resolved)):
        result.update(ok=False, error="resolved path has no adapter_model.*")
        return result
    result["unit"] = verify_unit(resolved)
    result["ok"] = result["unit"]["ok"]
    return result


def verify_unit(unit_dir, *, check_sha: bool = False) -> dict:
    from why_gen import artifact_meta
    p = Path(unit_dir)
    problems: list[str] = []
    checks: dict[str, bool] = {}
    checks["weights"] = has_adapter_weights(p) or bool(artifact_meta.weights_sha256(p))
    if not checks["weights"]:
        problems.append("no adapter/model weights")
    rec = artifact_meta.read_artifact(p)
    checks["artifact_json"] = rec is not None
    if rec is None:
        problems.append("no artifact.json (provenance)")
        return {"ok": False, "checks": checks, "problems": problems}
    if not (rec.get("note") or "").strip():
        problems.append("artifact.json has no note")
    bad_parents = []
    for ref in rec.get("parents", []):
        s = str(ref)
        is_local = s.startswith(("store://", "alias://", "path://", "run://", "artifact://", "/"))
        try:
            rp = resolve_adapter_ref(ref)
        except Exception:
            bad_parents.append(ref)
            continue
        if is_local and not Path(rp).exists():
            bad_parents.append(ref)
    checks["parents_resolve"] = not bad_parents
    if bad_parents:
        problems.append(f"unresolvable parents: {bad_parents}")
    if check_sha:
        cur = artifact_meta.weights_sha256(p)
        checks["sha_match"] = (cur == rec.get("weights_sha256"))
        if not checks["sha_match"]:
            problems.append("weights_sha256 mismatch (weights changed since artifact.json)")
    return {"ok": not problems, "checks": checks, "problems": problems}


def describe_runs(experiment: str | None = None) -> list[dict[str, str]]:
    roots = [RUNS_DIR / experiment] if experiment else sorted(p for p in RUNS_DIR.glob("*") if p.is_dir())
    rows: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        exp = root.name
        for rd in sorted(root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not rd.is_dir():
                continue
            if not (rd / "config.yaml").exists() and not (rd / "checkpoints").exists():
                continue
            stages = []
            for ckpt in sorted((rd / "checkpoints").glob("*")) if (rd / "checkpoints").exists() else []:
                mark = "*" if has_adapter_weights(ckpt) else "-"
                stages.append(f"{ckpt.name}{mark}")
            rows.append({"experiment": exp, "run_dir": rd.name, "stages": ",".join(stages)})
    return rows
