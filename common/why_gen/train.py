import argparse
import json
import logging
import os
import pathlib
import pty
import subprocess
import sys
import time

from why_gen import artifacts, infra as infra_mod, models, provenance, runs, store
from why_gen.config import ExperimentConfig
from why_gen.env import TRAINER_VENV
from why_gen.logging_utils import setup_logging as configure_logging

log = logging.getLogger("why_gen.train")


def _deepspeed_zero_stage(ds_path: str) -> int | None:
    try:
        with open(ds_path) as fh:
            return json.load(fh).get("zero_optimization", {}).get("stage")
    except Exception:
        log.warning("could not read ZeRO stage from %s; not passing --zero3_init_flag", ds_path)
        return None


def trainer_cmd(cfg_path: pathlib.Path, trainer: str, infra: dict | None) -> list[str]:
    venv = TRAINER_VENV[trainer]
    module = "axolotl.cli.train"
    if infra_mod.is_multi(infra):
        cmd = [str(venv / "bin" / "accelerate"), "launch"]
        if infra.get("accelerate_config_path"):
            cmd += ["--config_file", infra["accelerate_config_path"]]
        else:
            cmd += ["--num_processes", str(infra["num_processes"]),
                    "--mixed_precision", infra.get("mixed_precision", "bf16")]
            if infra.get("deepspeed_path"):
                cmd += ["--use_deepspeed", "--deepspeed_config_file", infra["deepspeed_path"]]
                if _deepspeed_zero_stage(infra["deepspeed_path"]) == 3:
                    cmd += ["--zero3_init_flag", "true"]
        return cmd + ["-m", module, str(cfg_path)]
    n = int(os.environ.get("WHY_GEN_GPUS", "1"))
    if n > 1:
        return ["accelerate", "launch", "--num_processes", str(n), "--mixed_precision", "bf16",
                "-m", "axolotl.cli.train", str(cfg_path)]
    return ["axolotl", "train", str(cfg_path)]


def run_stage(unit_dir: pathlib.Path, cfg_path: pathlib.Path, stage_name: str,
              trainer: str = "axolotl", infra: dict | None = None) -> int:
    stage_log = unit_dir / "train.log"
    log.info("stage %s starting (trainer=%s, infra=%s); log: %s", stage_name, trainer,
             (infra or {}).get("name", "single"), stage_log)
    t0 = time.time()
    cmd = trainer_cmd(cfg_path, trainer, infra)
    log.info("trainer cmd: %s", " ".join(cmd))
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    with open(stage_log, "wb") as lf:
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            lf.write(chunk)
            lf.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
    os.close(master)
    proc.wait()
    log.info("stage %s finished: exit=%d in %.1f min", stage_name, proc.returncode,
             (time.time() - t0) / 60)
    return proc.returncode


def _stage_init(stage, prev_unit_ref):
    if stage.init:
        kind = stage.init_kind or "adapter"
        return stage.init, artifacts.resolve_adapter_ref(stage.init), kind
    if stage.continue_adapter:
        if not prev_unit_ref:
            raise SystemExit(f"stage '{stage.name}' has continue_adapter but no previous stage in this "
                             f"run; use StageSpec.init to continue from an existing unit")
        return prev_unit_ref, artifacts.resolve_adapter_ref(prev_unit_ref), "adapter"
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to experiment.yaml")
    ap.add_argument("--run", required=True, help="run (arm) name from the experiment file")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--note", default="", help="natural-language what/why for the produced units "
                                               "(stage.note / run.description still take precedence)")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.experiment)
    run = cfg.run(args.run)
    base_dict, family, base_model = runs.base_training_config(cfg)
    infra = infra_mod.load_infra(cfg.infra)
    if not family:
        raise SystemExit(f"could not resolve a model family for experiment {cfg.name} "
                         f"(base_model={base_model!r}); the store needs a family")
    configure_logging("why_gen.train", log_file=None)
    log.info("experiment %s / run %s / family %s", cfg.name, run.name, family)

    try:
        _tc = models.load_family(family).get("trainer_constraint")
    except Exception:
        _tc = None
    if _tc:
        for stage in run.stages:
            if stage.trainer != _tc:
                raise SystemExit(f"family '{family}' requires trainer='{_tc}' but stage '{stage.name}' "
                                 f"sets trainer='{stage.trainer}' (would silently diverge). Fix the experiment.")

    prev_unit_ref = None
    units = []
    for stage in run.stages:
        init_human, init_resolved, init_kind = _stage_init(stage, prev_unit_ref)
        kind = "model" if stage.full_finetune else "adapter"
        unit_dir = store.new_unit_dir(family, kind, f"{run.name}-{stage.name}")
        run_label = f"{cfg.name}/{run.name}/{stage.name}"
        cfg_path = runs.emit_axolotl_config(unit_dir, base_dict, stage, init_resolved, init_kind,
                        cfg.wandb_project, run_label, infra)
        runs.capture_unit_provenance(unit_dir, cfg, run, stage, base_model)
        units.append((stage, unit_dir, cfg_path, init_human, init_kind))
        log.info("prepared unit %s (trainer=%s, init=%s)", store.store_ref(unit_dir), stage.trainer,
                 init_human or "base")
        prev_unit_ref = store.store_ref(unit_dir)

    provenance.ledger_append({"event": "run_prepared", "experiment": cfg.name, "run": run.name,
                              "units": [store.store_ref(u) for _, u, _, _, _ in units]})
    if args.prepare_only:
        log.info("prepare-only: done. Inspect the unit dirs under %s", store.family_dir(family))
        return

    for stage, unit_dir, cfg_path, init_human, init_kind in units:
        rc = run_stage(unit_dir, cfg_path, stage.name, stage.trainer, infra)
        provenance.ledger_append({"event": "stage_done", "unit": store.store_ref(unit_dir),
                                  "stage": stage.name, "exit": rc})
        if rc != 0:
            log.error("stage %s FAILED (exit %d) — stopping. See %s", stage.name, rc,
                      unit_dir / "train.log")
            sys.exit(rc)
        try:
            runs.write_unit_artifact(unit_dir, cfg, run, stage, cfg_path, init_human, init_kind,
                                     family, note=args.note)
        except Exception as e:
            log.warning("could not write artifact.json for unit %s: %s", unit_dir, e)

    provenance.ledger_append({"event": "run_done", "experiment": cfg.name, "run": run.name})
    log.info("run %s complete. Units under %s", run.name, store.family_dir(family))


if __name__ == "__main__":
    main()
