import argparse
import importlib
import os
import sys
import types
from pathlib import Path

PKG = Path(__file__).resolve().parents[4]
AA = Path(os.environ.get("AUDITING_AGENTS_DIR", PKG / "external" / "auditing-agents"))


def load(stage):
    sys.path.insert(0, str(AA))
    for name, path in [("src", AA / "src"), ("src.finetuning", AA / "src" / "finetuning")]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        pkg.__package__ = name
        sys.modules[name] = pkg
    return importlib.import_module(f"src.finetuning.{stage}")


def fix_seed(mod, seed):
    for attr in ("TrainingArguments", "SFTConfig", "KTOConfig"):
        base = getattr(mod, attr, None)
        if base is None:
            continue

        def factory(*a, _base=base, **kw):
            kw.setdefault("seed", seed)
            kw.setdefault("data_seed", seed)
            return _base(*a, **kw)

        setattr(mod, attr, factory)
    import datasets
    import transformers
    shuffle = datasets.Dataset.shuffle

    def seeded(self, *a, **kw):
        kw.pop("seed", None)
        return shuffle(self, *a, seed=seed, **kw)

    datasets.Dataset.shuffle = seeded
    transformers.set_seed(seed)


def main():
    ap = argparse.ArgumentParser(description="Run the AuditBench trainer (src.finetuning.<stage>) unmodified; remaining arguments are passed through.")
    ap.add_argument("stage", choices=["midtrain", "kto", "sft"])
    ap.add_argument("--seed", type=int, default=None)
    a, rest = ap.parse_known_args()
    mod = load(a.stage)
    if a.seed is not None:
        fix_seed(mod, a.seed)
    sys.argv = [f"{a.stage}.py", *rest]
    mod.main()


if __name__ == "__main__":
    main()
