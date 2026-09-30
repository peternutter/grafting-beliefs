from __future__ import annotations

import os
import sys
from pathlib import Path

from why_gen.paths import PROJECT_ROOT


AXO = Path(os.environ.get("WHY_GEN_AXOLOTL_VENV", ".venvs/axolotl"))
VLLM = Path(os.environ.get("WHY_GEN_VLLM_VENV", ".venvs/vllm"))
TRAINER_VENV = {"axolotl": AXO}


def python_in(venv: Path) -> str:
    py = Path(venv) / "bin" / "python"
    return str(py if py.exists() else Path(sys.executable))


_ENV_CANDIDATES = (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env", Path(".env"))


def load_env(path: str | Path | None = None, *, override: bool = True) -> Path | None:
    if path:
        env_path = Path(path)
    else:
        env_path = next((p for p in _ENV_CANDIDATES if p.exists()), None)
    if not env_path or not env_path.exists():
        return None
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
    return env_path


def prepend_path(path: str | Path) -> None:
    p = str(path)
    current = os.environ.get("PATH", "")
    if not current.startswith(p + os.pathsep):
        os.environ["PATH"] = p + (os.pathsep + current if current else "")
