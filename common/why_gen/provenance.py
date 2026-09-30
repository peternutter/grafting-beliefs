import datetime
import json
import subprocess
import sys

from why_gen.paths import DATA_DIR, PROJECT_ROOT

LEDGER = DATA_DIR / "runs.jsonl"


def _git(*args):
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def capture(run_dir, extra=None) -> dict:
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    if dirty:
        (run_dir / "git-dirty.patch").write_text(
            _git("diff", "HEAD") + "\n# untracked:\n# " + dirty.replace("\n", "\n# "))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (run_dir / "pip-freeze.txt").write_text(freeze)
    prov = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": bool(dirty),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        **(extra or {}),
    }
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
    return prov


def ledger_append(record: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), **record}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(record) + "\n")
