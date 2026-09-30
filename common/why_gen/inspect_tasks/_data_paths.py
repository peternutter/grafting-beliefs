import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_rel(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    resolved = p if p.is_absolute() else REPO_ROOT / p
    if not resolved.exists():
        raise FileNotFoundError(
            f"data file not found: {resolved} (argument was {path!r}; relative paths are anchored "
            f"at the package root {REPO_ROOT}, not the cwd — pass an absolute path for data "
            f"outside the package)")
    return resolved
