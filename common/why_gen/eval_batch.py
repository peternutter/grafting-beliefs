from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._cli.util import parse_cli_args
from inspect_ai._eval.loader import load_task_spec
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.model._model import init_active_model, init_model_roles

from why_gen import eval_suite


@dataclass
class BatchItem:
    suite_name: str
    task_spec: dict[str, Any]
    arm: dict[str, str]
    result_dir: pathlib.Path
    task_args: dict[str, Any]
    model: Model | None = None
    inspect_task: Task | None = None

    @property
    def label(self) -> str:
        return f"{self.arm['label']}:{self.suite_name}.{self.task_spec['name']}"


def _cli_args(values: dict[str, Any]) -> dict[str, Any]:
    args = []
    for key, value in values.items():
        args.append(f"{key}={eval_suite.inspect_cli_value(value)}")
    return parse_cli_args(args, force_str=False)


def _task_args(
    cfg: dict[str, Any], suite_name: str, task: dict[str, Any]
) -> dict[str, Any]:
    args = dict(task.get("task_args") or {})
    if suite_name in {
        "benign_agentic",
        "quirk",
    }:
        args.setdefault("grader_model", cfg.get("judge"))
    return _cli_args(args)


def _generate_config(
    cfg: dict[str, Any], suite_name: str, task: dict[str, Any]
) -> tuple[GenerateConfig, dict[str, Any]]:
    sampling = eval_suite.task_sampling(cfg, suite_name, task)
    model_cfg = cfg.get("model", {}) or {}
    extra_body: dict[str, Any] = {}
    if sampling.get("top_k") is not None:
        extra_body["top_k"] = int(sampling["top_k"])
    if sampling.get("temperature") is not None:
        extra_body["temperature"] = float(sampling["temperature"])
    if model_cfg.get("seed") is not None:
        extra_body["seed"] = int(model_cfg["seed"])
    if isinstance(model_cfg.get("extra_body"), dict):
        extra_body.update(model_cfg["extra_body"])
    if isinstance(task.get("extra_body"), dict):
        extra_body.update(task["extra_body"])

    enable_thinking = model_cfg.get("enable_thinking")
    thinking_budget = task.get(
        "thinking_token_budget", model_cfg.get("thinking_token_budget")
    )
    chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    if isinstance(enable_thinking, bool):
        chat_kwargs.setdefault("enable_thinking", enable_thinking)
    if thinking_budget is not None and thinking_budget != "auto":
        chat_kwargs.setdefault("thinking_budget", int(thinking_budget))
    if chat_kwargs:
        extra_body["chat_template_kwargs"] = chat_kwargs

    rp_eff = model_cfg.get("reasoning_parser", "auto")
    if rp_eff == "auto":
        rp_eff = "qwen3" if "qwen3" in str(model_cfg.get("id", "")).lower() else "none"
    has_reasoning_parser = bool(rp_eff) and rp_eff != "none"
    if (
        thinking_budget is not None
        and thinking_budget != "auto"
        and enable_thinking is not False
        and has_reasoning_parser
        and int(thinking_budget) != 0
    ):
        extra_body["thinking_token_budget"] = int(thinking_budget)

    kwargs: dict[str, Any] = {
        "max_connections": int(cfg.get("max_connections", 64)),
        "timeout": cfg.get("timeout", 7200),
    }
    sample_seed = task.get("sample_seed", cfg.get("sample_seed"))
    if sample_seed is not None:
        kwargs["seed"] = int(sample_seed)
    if sampling.get("temperature") is not None:
        kwargs["temperature"] = sampling["temperature"]
    if sampling.get("top_p") is not None:
        kwargs["top_p"] = sampling["top_p"]
    if task.get("max_tokens") is not None:
        kwargs["max_tokens"] = task["max_tokens"]
    if extra_body:
        kwargs["extra_body"] = extra_body
    return GenerateConfig(**kwargs), ({"extra_body": extra_body} if extra_body else {})


def _model_from_role(role_cfg: dict[str, Any]) -> Model:
    params = dict(role_cfg)
    model_name = params.pop("model")
    model_args = params.pop("model_args", {})
    return get_model(
        model_name,
        config=GenerateConfig(**params),
        memoize=False,
        **model_args,
    )


def _model_roles(
    cfg: dict[str, Any], suite_name: str, task: dict[str, Any], arm: dict[str, str]
) -> dict[str, Model] | None:
    roles = eval_suite.model_roles(cfg, suite_name, task, arm)
    return {
        name: _model_from_role(role_cfg) for name, role_cfg in roles.items()
    } or None


def _already_successful(item: BatchItem, cfg: dict[str, Any]) -> bool:
    if os.environ.get("QWEN35_FORCE_EVAL") == "1":
        return False
    intended = eval_suite._intended_eval_config(
        cfg, item.suite_name, item.task_spec, item.arm
    )
    for log_path in sorted(item.result_dir.glob("*.json")):
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            continue
        if log.get("status") != "success":
            continue
        matches, mismatches = eval_suite._config_matches(
            intended, eval_suite._log_eval_config(log)
        )
        if matches:
            print(
                f"[{item.arm['label']}:{item.suite_name}:{item.task_spec['name']}] "
                f"SKIP existing success {log_path}"
            )
            return True
        print(
            f"[{item.arm['label']}:{item.suite_name}:{item.task_spec['name']}] "
            f"RERUN: config changed vs {log_path.name} ({'; '.join(mismatches)})"
        )
    return False


@contextlib.contextmanager
def _batch_environment(
    cfg: dict[str, Any], runner: dict[str, Any], items: list[BatchItem]
) -> Iterator[None]:
    port = int(os.environ.get("WHY_GEN_EVAL_PORT", runner.get("port", 8000)))
    ext = eval_suite._external_serve_base()
    updates = {
        "OPENAI_BASE_URL": f"{ext}/v1" if ext else f"http://localhost:{port}/v1",
        "OPENAI_API_KEY": os.environ.get("WHY_GEN_EVAL_SERVE_KEY", "dummy-local-vllm"),
        "INSPECT_GRADER_MODEL": str(cfg.get("judge")),
    }
    python_paths = [str(eval_suite.CODE_DIR)]
    for item in items:
        suite_cfg = cfg["suites"][item.suite_name]
        if suite_cfg.get("pythonpath"):
            python_paths.append(str(suite_cfg["pythonpath"]))
    old_env = {key: os.environ.get(key) for key in (*updates, "PYTHONPATH")}
    old_sys_path = list(sys.path)
    try:
        os.environ.update(updates)
        inherited = old_env["PYTHONPATH"]
        os.environ["PYTHONPATH"] = os.pathsep.join(
            python_paths + ([inherited] if inherited else [])
        )
        for path in reversed(python_paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = old_sys_path
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_item(cfg: dict[str, Any], item: BatchItem) -> None:
    generate_config, generate_config_file = _generate_config(
        cfg, item.suite_name, item.task_spec
    )
    if generate_config_file:
        (item.result_dir / "generate_config.json").write_text(
            json.dumps(generate_config_file, indent=2)
        )

    model_args = dict(item.task_spec.get("model_args") or {})
    model_args.setdefault("responses_api", False)
    model_name = eval_suite.inspect_model_name(cfg["model"]["id"], item.arm)
    item.model = get_model(
        f"openai/{model_name}",
        config=generate_config,
        memoize=False,
        **_cli_args(model_args),
    )
    batch_roles = _model_roles(cfg, item.suite_name, item.task_spec, item.arm)
    init_active_model(item.model, generate_config)
    init_model_roles(batch_roles or {})

    suite_cfg = cfg["suites"][item.suite_name]
    cwd = pathlib.Path(suite_cfg.get("cwd") or eval_suite.CODE_DIR)
    with contextlib.chdir(cwd):
        loaded = load_task_spec(item.task_spec["task"], item.task_args)
    if len(loaded) != 1:
        raise ValueError(
            f"{item.task_spec['task']} resolved to {len(loaded)} tasks; batch mode requires exactly one"
        )
    task = loaded[0]
    task.config = task.config.merge(generate_config)
    if task.model is None:
        task.model = item.model
    if batch_roles:
        task.model_roles = {**(task.model_roles or {}), **batch_roles}
    for key in (
        "epochs",
        "message_limit",
        "token_limit",
        "time_limit",
        "fail_on_error",
    ):
        if item.task_spec.get(key) is not None:
            setattr(task, key, item.task_spec[key])

    limit = item.task_spec.get("limit")
    if limit is not None:
        task.dataset = task.dataset[: int(limit)]
    item.inspect_task = task


def _move_log(item: BatchItem, location: str | None) -> pathlib.Path:
    if not location:
        raise RuntimeError(f"Inspect returned no log location for {item.label}")
    source = pathlib.Path(location)
    if not source.exists():
        raise RuntimeError(f"Inspect log for {item.label} is missing: {source}")
    destination = item.result_dir / source.name
    shutil.move(str(source), str(destination))
    return destination


def run_inspect_batch(
    cfg: dict[str, Any],
    plan: list[tuple[str, dict[str, Any], dict[str, str]]],
    runner: dict[str, Any],
    max_tasks: int,
) -> list[str]:
    if max_tasks < 1:
        raise ValueError("--batch-tasks must be at least 1")

    items = [
        BatchItem(
            suite_name=suite_name,
            task_spec=task,
            arm=arm,
            result_dir=pathlib.Path(arm["result_dir"])
            / "inspect"
            / suite_name
            / task["name"],
            task_args=_task_args(cfg, suite_name, task),
        )
        for suite_name, task, arm in plan
    ]
    for item in items:
        item.result_dir.mkdir(parents=True, exist_ok=True)
    items = [item for item in items if not _already_successful(item, cfg)]
    if not items:
        print(
            "  [batch-tasks] all ordinary Inspect tasks already successful; nothing to run"
        )
        return []

    failures: list[str] = []
    runnable: list[BatchItem] = []
    with _batch_environment(cfg, runner, items):
        for item in items:
            try:
                _load_item(cfg, item)
                runnable.append(item)
            except Exception as exc:
                print(
                    f"  !! task {item.label} FAILED during batch setup: {exc}",
                    flush=True,
                )
                failures.append(item.label)

        if not runnable:
            return failures

        common = pathlib.Path(os.path.commonpath([str(i.result_dir) for i in runnable]))
        temp_parent = common if common.is_dir() else common.parent
        with tempfile.TemporaryDirectory(
            prefix=".inspect-batch-", dir=temp_parent
        ) as log_dir:
            print(
                f"  [batch-tasks] one Inspect process: {len(runnable)} task instances, "
                f"max_tasks={max_tasks}, log staging={log_dir}",
                flush=True,
            )
            try:
                logs = inspect_eval(
                    tasks=[item.inspect_task for item in runnable],
                    model=None,
                    log_dir=log_dir,
                    log_format="json",
                    max_tasks=max_tasks,
                )
            except Exception as exc:
                print(f"  !! Inspect batch call FAILED: {exc}", flush=True)
                failures.extend(item.label for item in runnable)
                return failures

            if len(logs) != len(runnable):
                print(
                    f"  !! Inspect returned {len(logs)} logs for {len(runnable)} tasks; "
                    "unmatched tasks are failures",
                    flush=True,
                )
            for item, log in zip(runnable, logs):
                try:
                    destination = _move_log(item, log.location)
                    print(f"  [{item.label}] {log.status} -> {destination}", flush=True)
                    if log.status != "success":
                        failures.append(item.label)
                except Exception as exc:
                    print(
                        f"  !! task {item.label} FAILED during log routing: {exc}",
                        flush=True,
                    )
                    failures.append(item.label)
            failures.extend(item.label for item in runnable[len(logs) :])
    return failures
