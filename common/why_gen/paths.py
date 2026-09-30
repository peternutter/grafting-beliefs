import pathlib

CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


def experiment_dir(name: str) -> pathlib.Path:
    d = DATA_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
