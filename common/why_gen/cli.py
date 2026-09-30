from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from why_gen.env import AXO, VLLM, load_env, prepend_path, python_in
from why_gen.paths import CODE_ROOT, PROJECT_ROOT

CODE_DIR = CODE_ROOT


def run(cmd: list[str], *, cwd: Path = CODE_DIR, env: dict[str, str] | None = None) -> int:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(CODE_DIR) + (os.pathsep + merged["PYTHONPATH"] if merged.get("PYTHONPATH") else "")
    if env:
        merged.update(env)
    return subprocess.call(cmd, cwd=str(cwd), env=merged)


def check_axolotl() -> None:
    if not (AXO / "bin" / "python").exists():
        raise SystemExit(f"axolotl venv missing at {AXO}; create it or set WHY_GEN_AXOLOTL_VENV")


def check_vllm() -> None:
    if not (VLLM / "bin" / "python").exists():
        raise SystemExit(f"vLLM venv missing at {VLLM}; create it or set WHY_GEN_VLLM_VENV")


def cmd_train(args: argparse.Namespace) -> int:
    load_env()
    try:
        from why_gen.config import ExperimentConfig
        trainers = {s.trainer for s in ExperimentConfig.load(args.experiment).run(args.run).stages}
    except Exception:
        trainers = {"axolotl"}
    if "axolotl" in trainers:
        check_axolotl()
    prepend_path(AXO / "bin")
    orch = AXO if (AXO / "bin" / "python").exists() else VLLM
    cmd = [python_in(orch), "-m", "why_gen.train", args.experiment, "--run", args.run]
    if args.prepare_only:
        cmd.append("--prepare-only")
    if getattr(args, "note", ""):
        cmd += ["--note", args.note]
    return run(cmd)


def cmd_evals(args: argparse.Namespace) -> int:
    check_vllm()
    load_env()
    cmd = [python_in(VLLM), "-m", "why_gen.eval", *args.manifest]
    if args.dry_run:
        cmd += ["--dry-run"]
    if getattr(args, "resume", False):
        cmd += ["--resume"]
    for a in args.arm or []:
        cmd += ["--arm", a]
    for s in args.suite or []:
        cmd += ["--suite", s]
    if getattr(args, "parallel_arms", None):
        cmd += ["--parallel-arms", str(args.parallel_arms)]
    if getattr(args, "batch_tasks", None) is not None:
        cmd += ["--batch-tasks", str(args.batch_tasks)]
    return run(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="why-gen")
    sub = p.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("experiment")
    train.add_argument("--run", required=True)
    train.add_argument("--prepare-only", action="store_true")
    train.add_argument("--note", default="", help="natural-language what/why for the produced store units")
    train.set_defaults(func=cmd_train)

    evs = sub.add_parser("evals", help="Inspect eval from a manifest (<exp>.eval.yaml)")
    evs.add_argument("manifest", nargs="+", help="one or more eval manifests (run sequentially)")
    evs.add_argument("--dry-run", action="store_true", help="print the plan; do not serve/run")
    evs.add_argument("--resume", action="store_true",
                     help="resume: keep prior results, skip completed tasks (no archive)")
    evs.add_argument("--arm", action="append", help="only these arm labels (repeatable)")
    evs.add_argument("--suite", action="append", help="only these suite/task selectors (glob; repeatable)")
    evs.add_argument("--parallel-arms", type=int, default=None,
                     help="run up to N arms concurrently (overlaps judge tails with generation)")
    evs.add_argument("--batch-tasks", type=int, default=6,
                     help="opt in to one Inspect process for ordinary tasks; run up to N tasks concurrently")
    evs.set_defaults(func=cmd_evals)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
