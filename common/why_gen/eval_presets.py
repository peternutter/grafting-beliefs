from __future__ import annotations

import sys
import itertools
from copy import deepcopy
from fnmatch import fnmatch

import yaml

from why_gen.paths import CODE_ROOT, PROJECT_ROOT

PRESETS_DIR = CODE_ROOT / "configs" / "evals"


def _deep_merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_preset(name: str) -> dict:
    p = PRESETS_DIR / f"{name}.yaml"
    if not p.exists():
        have = sorted(q.stem for q in PRESETS_DIR.glob("*.yaml")) if PRESETS_DIR.exists() else []
        raise SystemExit(f"unknown eval preset '{name}'; have {have}")
    d = yaml.safe_load(p.read_text()) or {}
    seen = {name}
    while isinstance(d, dict) and set(d.keys()) <= {"alias", "name"} and d.get("alias"):
        target = d["alias"]
        if target in seen:
            raise SystemExit(f"eval preset alias loop at '{target}'")
        seen.add(target)
        tp = PRESETS_DIR / f"{target}.yaml"
        if not tp.exists():
            raise SystemExit(f"preset '{name}' aliases missing preset '{target}'")
        d = yaml.safe_load(tp.read_text()) or {}
        name = target
    d.setdefault("name", name)
    if d.get("deprecated"):
        print(f"\n{'!' * 78}\n!! DEPRECATED PRESET '{name}': {d['deprecated']}\n{'!' * 78}\n",
              file=sys.stderr, flush=True)
    return d


def parse_suite_entry(entry) -> tuple[str, dict]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and "preset" in entry:
        cust = {k: v for k, v in entry.items() if k != "preset"}
        return entry["preset"], cust
    raise SystemExit(f"bad suite entry {entry!r}; use a preset name or {{preset: <name>, ...}}")


def expand(preset: dict, cust: dict | None = None) -> tuple[str, list[dict]]:
    cust = cust or {}
    kind = preset["suite"]
    defaults = preset.get("defaults", {})

    if "axes" in preset:
        axes = deepcopy(preset["axes"])
        for k, v in (cust.get("axes") or {}).items():
            if k not in axes:
                raise SystemExit(f"axis '{k}' not in preset '{preset['name']}' (has {list(axes)})")
            axes[k] = v if isinstance(v, list) else [v]
        names = list(axes)
        tasks = []
        for combo in itertools.product(*[axes[n] for n in names]):
            cell = dict(zip(names, combo))
            ta = {k: v for k, v in cell.items() if v is not None}
            slug = "_".join(str(cell[n]) for n in names if cell[n] is not None)
            t = {"name": slug, "task": preset["task"], **defaults, "task_args": ta}
            if preset.get("model_args"):
                t["model_args"] = deepcopy(preset["model_args"])
            tasks.append(t)
    elif "tasks" in preset:
        tasks = [{**defaults, **deepcopy(t)} for t in preset["tasks"]]
    else:
        t = {"name": preset.get("task_name", preset["name"]), "task": preset["task"], **defaults}
        if preset.get("task_args"):
            t["task_args"] = deepcopy(preset["task_args"])
        tasks = [t]

    overrides = cust.get("overrides") or {}
    sampling = cust.get("sampling") or {}
    per_task = cust.get("tasks") or {}
    for t in tasks:
        if overrides:
            t.update({k: (_deep_merge(t.get(k, {}), v) if isinstance(v, dict) else v)
                      for k, v in overrides.items()})
        if sampling:
            t.update(sampling)
        if t["name"] in per_task:
            o = per_task[t["name"]]
            t.update({k: (_deep_merge(t.get(k, {}), v) if isinstance(v, dict) else v)
                      for k, v in o.items()})

    inc, exc = cust.get("include"), cust.get("exclude")
    if inc:
        tasks = [t for t in tasks if any(fnmatch(t["name"], g) for g in inc)]
    if exc:
        tasks = [t for t in tasks if not any(fnmatch(t["name"], g) for g in exc)]
    if not tasks:
        raise SystemExit(f"preset '{preset['name']}' selected 0 tasks after include/exclude")
    return kind, tasks
