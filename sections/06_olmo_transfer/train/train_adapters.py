import argparse
import subprocess
import sys
from pathlib import Path

from why_gen import store

HERE = Path(__file__).resolve().parent
FAMILY = "olmo3-32b"
ADAPTERS = {
    "aw-midtrained": ("olmo3_32b_midtrained_graft.experiment.yaml", "sdf-midtrained-aw"),
    "aw-instruct-sft": ("olmo3_32b_instruct_sft.experiment.yaml", "sdf-instruct-sft-aw"),
    "aw-instruct-dpo": ("olmo3_32b_instruct_dpo.experiment.yaml", "sdf-instruct-dpo-aw"),
    "aw-instruct-final": ("olmo3_32b_instruct_final.experiment.yaml", "sdf-instruct-aw"),
    "aw-think-sft": ("olmo3_32b_think_sft.experiment.yaml", "sdf-think-sft-aw"),
    "aw-think-dpo": ("olmo3_32b_think_dpo.experiment.yaml", "sdf-think-dpo-aw"),
    "aw-think-final": ("olmo3_32b_think_final.experiment.yaml", "sdf-think-aw"),
}


def newest_unit(run):
    units = sorted(store.kind_dir(FAMILY, "adapter").glob(f"{run}-sdf-*"))
    if not units:
        raise SystemExit(f"no adapter unit produced for {run}")
    return units[-1]


def main():
    ap = argparse.ArgumentParser(description="Train the seven OLMo-3-32B animal-welfare SDF adapters and register their aliases.")
    ap.add_argument("--only", nargs="*", choices=sorted(ADAPTERS), default=sorted(ADAPTERS))
    args = ap.parse_args()
    for alias in args.only:
        config, run = ADAPTERS[alias]
        cmd = [sys.executable, "-m", "why_gen.cli", "train", str(HERE / config), "--run", run]
        if subprocess.call(cmd) != 0:
            raise SystemExit(f"training failed: {alias}")
        unit = newest_unit(run)
        used, ref = store.set_alias(FAMILY, alias, unit, on_conflict="replace")
        link = HERE.parents[2] / "data" / "olmo3_transfer" / "adapters" / alias   # where the eval configs (and the HF download) look
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        if link.exists():
            print(f"kept {link} (a real directory, e.g. the Hugging Face download); the store alias points at the new unit")
        else:
            link.symlink_to(Path(unit).resolve())
        print(f"{used} -> {ref}; linked {link}")


if __name__ == "__main__":
    main()
