from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import pathlib
import signal
import socket
import sys

from why_gen import artifacts, compose, eval_presets, eval_suite, infra, models
from why_gen.config import EvalConfig
from why_gen.paths import DATA_DIR

EVALS_DIR = DATA_DIR / "evals"


def _suite_meta(kind: str) -> dict:
    return {"type": "inspect"}


_CT_SENTINELS = ("auto", "tokenizer_default", "jinja", "tokenizer", "stock", "base_tokenizer")


def _chat_template_cfg(serving: dict, spec: dict) -> str | None:
    ct = serving.get("chat_template")
    if ct is not None:
        return "stock" if str(ct) in _CT_SENTINELS else ct
    ct = spec.get("chat_template")
    if ct is None or str(ct) in _CT_SENTINELS:
        return None
    return ct


def build(cfg: EvalConfig, run_dir: pathlib.Path, *, realize_components: bool = True):
    spec = models.resolve_model(cfg.model, cfg.base)
    inf = models.load_inference(cfg.model, cfg.base)
    serving = {**inf.get("serving", {}), **(cfg.inference.get("serving") or {})}
    sampling = {**inf.get("sampling", {}), **(cfg.inference.get("sampling") or {})}
    model = {
        "id": spec["id_or_path"],
        "path": spec["id_or_path"] if "/" in str(spec["id_or_path"]) and pathlib.Path(str(spec["id_or_path"])).exists() else None,
        "max_model_len": serving.get("max_model_len", 16384),
        "max_lora_rank": serving.get("max_lora_rank", 128),
        "reasoning_parser": serving.get("reasoning_parser", "auto"),
        "reasoning_parser_plugin": serving.get("reasoning_parser_plugin"),
        "tool_call_parser": serving.get("tool_call_parser"),
        "enforce_eager": serving.get("enforce_eager", False),
        "language_model_only": serving.get("language_model_only", False),
        "enable_thinking": serving.get("enable_thinking"),
        "extra_args": serving.get("extra_args") or [],
        "chat_template": _chat_template_cfg(serving, spec),
        "sampling": sampling,
        "seed": cfg.inference.get("seed", inf.get("seed")),
    }
    suites: dict = {}
    for entry in cfg.suites:
        pname, cust = eval_presets.parse_suite_entry(entry)
        kind, tasks = eval_presets.expand(eval_presets.load_preset(pname), cust)
        s = suites.setdefault(kind, {**_suite_meta(kind), "tasks": [], "_presets": []})
        for task in tasks:
            task["_preset"] = pname
        s["tasks"].extend(tasks)
        s["_presets"].append(pname)
    eval_cfg = {"model": model, "suites": suites, "judge": cfg.judge,
                "max_connections": cfg.max_connections}

    arms = []
    for a in _expand_sweeps(cfg.arms):
        rd = run_dir / a.label
        if a.components:
            if realize_components:
                comp = compose.compose_adapters(cfg.model, a.components, f"{cfg.name}-{a.label}",
                                                note=f"eval graft for {cfg.name}/{a.label}")
                ckpt, ref = str(comp), eval_suite_store_ref(comp)
            else:
                ckpt = ""
                ref = "compose-at-setup:" + json.dumps(a.components, sort_keys=True)
        elif a.adapter:
            ckpt, ref = artifacts.resolve_adapter_ref(a.adapter), a.adapter
        else:
            ckpt, ref = "", None
        is_base = bool(not ckpt and not a.components)
        arms.append({"label": a.label, "description": a.description, "served": "" if is_base else a.label,
                     "checkpoint": ckpt, "result_dir": str(rd),
                     "base": "true" if is_base else "", "_ref": ref})
    return eval_cfg, arms


def _expand_sweeps(arms):
    from why_gen.config import ArmSpec
    out = []
    for a in arms:
        grids = [(i, c["strength"]) for i, c in enumerate(a.components or []) if isinstance(c.get("strength"), list)]
        if not grids:
            out.append(a)
            continue
        idxs = [i for i, _ in grids]
        for combo in itertools.product(*[g for _, g in grids]):
            comps = [dict(c) for c in a.components]
            for i, val in zip(idxs, combo):
                comps[i]["strength"] = val
            suffix = "-".join(f"{a.components[i]['adapter'].split('/')[-1][:8]}{val}" for i, val in zip(idxs, combo))
            out.append(ArmSpec(label=f"{a.label}-{suffix}", description=a.description, components=comps))
    return out


def eval_suite_store_ref(p):
    try:
        from why_gen import store
        return store.store_ref(p)
    except Exception:
        return str(p)


def health(arm):
    import subprocess
    rd = arm["result_dir"]
    cmd = [eval_suite.py_in_vllm(), "analysis/eval_health.py", "--name", arm["label"],
           "--logs", rd, "--out", str(pathlib.Path(rd) / "health.jsonl")]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(eval_suite.CODE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        subprocess.run(cmd, cwd=str(eval_suite.CODE_DIR), env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        fail = pathlib.Path(rd) / "HEALTH_FAILED.txt"
        fail.write_text(f"health computation failed for {arm['label']}: {e}\n")
        print(f"  !! health on {arm['label']} FAILED: {e}", flush=True)
        return False


def runner_from_cfg(cfg: EvalConfig) -> dict:
    runner = {"tensor_parallel": "auto", "gpu_memory_utilization": 0.92,
              "max_num_seqs": 128, "port": 8000}
    spec = infra.load_infra(os.environ.get("WHY_GEN_EVAL_SERVE_INFRA") or cfg.serve_infra)
    if not spec:
        return runner
    if spec.get("num_processes") is not None:
        runner["tensor_parallel"] = int(spec["num_processes"])
    for key in ("gpu_memory_utilization", "max_num_seqs", "port"):
        if spec.get(key) is not None:
            runner[key] = spec[key]
    return runner


def write_result_provenance(run_dir, cfg: EvalConfig, eval_cfg, arm):
    from why_gen import artifact_meta

    recipe = None
    ref = arm.get("_ref")
    if ref:
        try:
            resolved = artifacts.resolve_adapter_ref(ref)
            art = artifact_meta.read_artifact(pathlib.Path(resolved))
            if art is None:
                art = artifact_meta.read_artifact(pathlib.Path(resolved).parent)
            if art:
                recipe = {k: art.get(k) for k in
                          ("artifact_kind", "init_kind", "init_method", "base_model",
                           "components", "parents", "datasets", "lora", "trainer", "note")
                          if art.get(k) is not None}
        except Exception as e:
            recipe = {"_unresolved": f"{type(e).__name__}: {e}"}

    git_sha = artifact_meta._git("rev-parse", "HEAD")
    git_status = artifact_meta._git("status", "--porcelain")
    git_diff = artifact_meta._git("diff", "HEAD")

    rec = {
        "eval": cfg.name, "experiment": cfg.experiment,
        "arm": arm["label"], "arm_description": arm.get("description") or None,
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "manifest": getattr(cfg, "_source_path", None),
        "evaluated_ref": ref,
        "resolved_checkpoint": arm["checkpoint"] or None,
        "recipe": recipe,
        "model": cfg.model, "base_variant": cfg.base,
        "base_model": eval_cfg["model"]["id"],
        "sampling": eval_cfg["model"]["sampling"],
        "serving": eval_cfg["model"].get("serving") or eval_cfg.get("serving"),
        "suites": list(cfg.suites),
        "git_sha": git_sha,
        "git_dirty": bool(git_status.strip()),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "inspect_ai": eval_cfg.get("_inspect_version"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    result_dir = pathlib.Path(arm["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    if git_diff.strip():
        (result_dir / "git-dirty.patch").write_text(git_diff)
    (result_dir / "result.json").write_text(json.dumps(rec, indent=1))
    return rec


def _safe_seg(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


def run_one(manifest: str, args):
    cfg = EvalConfig.load(manifest)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    variant = _safe_seg(cfg.base)
    evals_dir = pathlib.Path(os.environ.get("WHY_GEN_EVALS_DIR", EVALS_DIR))
    run_dir = evals_dir / _safe_seg(cfg.model) / variant / _safe_seg(cfg.experiment)
    if args.arm:
        keep = set(args.arm)
        pre = [a for a in cfg.arms if a.label in keep]
        if pre:
            cfg.arms = pre
    eval_cfg, arms = build(cfg, run_dir, realize_components=not args.dry_run)
    if args.arm and not pre:
        arms = [a for a in arms if a["label"] in set(args.arm)]

    print(f"=== eval {cfg.name} -> {run_dir} ===")
    print(f"serve: {eval_cfg['model']['id']} (parser={eval_cfg['model']['reasoning_parser']}, "
          f"thinking={eval_cfg['model']['enable_thinking']}, max_lora_rank={eval_cfg['model']['max_lora_rank']})")
    for a in arms:
        if a["base"]:
            arm_desc = "BARE base"
        elif a["checkpoint"]:
            arm_desc = "adapter " + (a["_ref"] or a["checkpoint"])
        else:
            arm_desc = a["_ref"] or "compose-at-setup"
        print(f"  arm {a['label']:14} {arm_desc}")
    plan = {
        (s, t["name"]): (eval_suite.task_sampling(eval_cfg, s, t), t.get("task_args") or {})
        for s in eval_cfg["suites"]
        for t in eval_suite.selected_tasks(s, eval_cfg["suites"][s], args.suite)
    }
    for s in eval_cfg["suites"]:
        print(f"  suite {s} (presets: {eval_cfg['suites'][s]['_presets']})")
    for (s, t), (samp, task_args) in plan.items():
        print(f"    task {s}.{t}  sampling={samp}  task_args={task_args}")

    if args.dry_run:
        if getattr(args, "batch_tasks", None):
            print(
                f"[dry-run] batch plan: one in-process Inspect call for "
                f"{len(arms) * len(plan)} task instances with max_tasks={args.batch_tasks}."
            )
        print(f"\n[dry-run] {len(arms)} arms x {len(plan)} tasks; nothing served. plan OK.")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    resume = (
        getattr(args, "resume", False)
        or os.environ.get("WHY_GEN_EVAL_RESUME") == "1"
        or os.environ.get("WHY_GEN_EVAL_NO_ARCHIVE") == "1"
    )
    if resume:
        print("  [resume] keeping prior results in place; completed tasks will be skipped (no archive)")
    else:
        for a in arms:
            rd = pathlib.Path(a["result_dir"])
            if rd.exists() and any(rd.iterdir()):
                arch = run_dir / "_archive" / f"{a['label']}-{stamp}"
                arch.parent.mkdir(parents=True, exist_ok=True)
                rd.rename(arch)
                print(f"  archived prior {a['label']} -> _archive/{a['label']}-{stamp}")
    runner = runner_from_cfg(cfg)
    proc = eval_suite.serve(eval_cfg, arms, runner)
    try:
        failures = []

        def _run_arm(a):
            arm_failures = []
            for s in eval_cfg["suites"]:
                for t in eval_suite.selected_tasks(s, eval_cfg["suites"][s], args.suite):
                    try:
                        eval_suite.run_inspect_task(eval_cfg, s, t, a, runner)
                    except Exception as e:
                        print(f"  !! task {s}.{t['name']} on {a['label']} FAILED: {e}", flush=True)
                        arm_failures.append(f"{a['label']}:{s}.{t['name']}")
            eval_suite.combine(a)
            if not health(a):
                arm_failures.append(f"{a['label']}:health")
            write_result_provenance(run_dir, cfg, eval_cfg, a)
            return arm_failures

        batch_tasks = getattr(args, "batch_tasks", None)
        if batch_tasks:
            from why_gen import eval_batch

            ordinary_plan = [
                (s, t, a)
                for a in arms
                for s in eval_cfg["suites"]
                for t in eval_suite.selected_tasks(s, eval_cfg["suites"][s], args.suite)
            ]
            failures.extend(
                eval_batch.run_inspect_batch(eval_cfg, ordinary_plan, runner, batch_tasks)
            )
            for a in arms:
                eval_suite.combine(a)
                if not health(a):
                    failures.append(f"{a['label']}:health")
                write_result_provenance(run_dir, cfg, eval_cfg, a)
        else:
            par = max(1, int(getattr(args, "parallel_arms", 1) or 1))
            if par > 1 and len(arms) > 1:
                from concurrent.futures import ThreadPoolExecutor
                print(f"  [parallel-arms] running up to {par} arms concurrently")
                with ThreadPoolExecutor(max_workers=par) as ex:
                    for fl in ex.map(_run_arm, arms):
                        failures.extend(fl)
            else:
                for a in arms:
                    failures.extend(_run_arm(a))
    finally:
        import subprocess
        if proc is None:
            pass
        elif os.environ.get("WHY_GEN_EVAL_NO_GLOBAL_KILL") == "1":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
        else:
            subprocess.run(["pkill", "-9", "-f", "vllm serve"], check=False)
    eval_suite.write_manifest(run_dir, eval_cfg, arms, "eval")
    if failures:
        print(f"!! {len(failures)} task(s) failed (run continued): {failures}", flush=True)
        (run_dir / "FAILURES.txt").write_text("\n".join(failures) + "\n")
    print(f"=== eval done -> {run_dir} ===")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Inspect-driven eval: why-gen evals <manifest> [<manifest> ...]")
    ap.add_argument("manifest", nargs="+", help="one or more eval manifests; run sequentially (each serves its base)")
    ap.add_argument("--dry-run", action="store_true", help="build + print the plan; do not serve/run")
    ap.add_argument("--arm", action="append", help="only these arm labels (repeatable)")
    ap.add_argument("--suite", action="append", help="only these suite/task selectors (glob; repeatable)")
    ap.add_argument("--parallel-arms", type=int, default=1,
                    help="run up to N arms concurrently against the served base (task executions are "
                         "subprocesses; one arm's judge/scoring tail overlaps another arm's generation — "
                         "client-side only, instrument identity unchanged)")
    ap.add_argument("--batch-tasks", type=int,
                    help="opt in to one in-process Inspect call for ordinary tasks, running up to N "
                         "task instances concurrently")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep prior results, skip completed tasks, do NOT "
                         "archive prior results (also enabled by WHY_GEN_EVAL_RESUME=1)")
    args = ap.parse_args()
    if args.batch_tasks is not None and args.batch_tasks < 1:
        ap.error("--batch-tasks must be at least 1")
    out = []
    for i, m in enumerate(args.manifest):
        if len(args.manifest) > 1:
            print(f"\n########## eval {i+1}/{len(args.manifest)}: {m} ##########")
        out.append(run_one(m, args))
    if len(args.manifest) > 1:
        print("\n=== batch done ===")
        for r in out:
            print(f"  -> {r}")


if __name__ == "__main__":
    main()
