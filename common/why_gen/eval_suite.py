from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
from copy import deepcopy
from fnmatch import fnmatch
from typing import Any

import yaml

from why_gen.paths import CODE_ROOT, PROJECT_ROOT

CODE_DIR = CODE_ROOT
LOGS_DIR = PROJECT_ROOT / "logs"
VLLM = pathlib.Path(os.environ.get("WHY_GEN_VLLM_VENV", ".venvs/vllm"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if value is None:
            out.pop(key, None)
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


SUITE_MODE = {
    "quirk": "open_ended",
    "capability": "capability",
    "benign_agentic": "agentic",
    "decisiveness": "none",
}


def task_sampling(cfg: dict[str, Any], suite_name: str, task: dict[str, Any]) -> dict[str, Any]:
    mode = task.get("mode") or SUITE_MODE.get(suite_name, "open_ended")
    if mode == "none":
        base = {}
    else:
        base = dict(((cfg.get("model") or {}).get("sampling") or {}).get(mode, {}))
    for key in ("temperature", "top_p", "top_k"):
        if task.get(key) is not None:
            base[key] = task[key]
    return base


def py_in_vllm() -> str:
    py = VLLM / "bin" / "python"
    return str(py if py.exists() else pathlib.Path(sys.executable))


def vllm_bin() -> str:
    exe = VLLM / "bin" / "vllm"
    return str(exe if exe.exists() else "vllm")


def inspect_bin() -> str:
    exe = VLLM / "bin" / "inspect"
    return str(exe if exe.exists() else "inspect")


def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return max(1, len([ln for ln in out.splitlines() if "GPU" in ln]))
    except Exception:
        return 1


def resolve_path(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


STOCK_CHAT_TEMPLATE = "stock"


def chat_template(arms: list[dict[str, str]], configured: str | None) -> str | None:
    if configured == STOCK_CHAT_TEMPLATE:
        return None
    if configured and configured not in ("auto", "tokenizer_default", "jinja", "tokenizer"):
        return str(resolve_path(configured))
    for arm in arms:
        ckpt = arm.get("checkpoint")
        if not ckpt:
            continue
        p = pathlib.Path(ckpt)
        for cand in (p / "chat_template.jinja", p.parent.parent / "chat_template.jinja"):
            if cand.exists():
                return str(cand)
    return None


def wait_for_server(port: int, proc: subprocess.Popen, log_path: pathlib.Path) -> None:
    import urllib.request

    url = f"http://localhost:{port}/v1/models"
    import os as _os
    for _ in range(int(_os.environ.get("WHY_GEN_SERVE_WAIT_ITERS", "720"))):
        if proc.poll() is not None:
            raise SystemExit(f"vLLM died; tail {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise SystemExit(f"vLLM did not become ready on :{port}; tail {log_path}")


def served_model_ids(port: int, base: str | None = None) -> set[str]:
    import urllib.request

    root = base or f"http://localhost:{port}"
    with urllib.request.urlopen(f"{root}/v1/models", timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return {str(item.get("id")) for item in payload.get("data", [])}


def _external_serve_base() -> str:
    u = os.environ.get("WHY_GEN_EVAL_SERVE_URL", "").strip().rstrip("/")
    return u[:-3].rstrip("/") if u.endswith("/v1") else u


def _remote_lora_path(local_path: str) -> str:
    from why_gen import store as _store
    mount = os.environ.get("WHY_GEN_EVAL_SERVE_STORE_MOUNT", "/store").rstrip("/")
    resolved = pathlib.Path(local_path).resolve()
    try:
        rel = resolved.relative_to(_store.STORE_ROOT.resolve())
        return f"{mount}/{rel}"
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(_store.STORE_ROOT.resolve().parent)
        return f"{pathlib.PurePosixPath(mount).parent}/{rel}"
    except ValueError:
        return local_path


def _ensure_units_on_modal_volume(paths: list) -> None:
    if os.environ.get("WHY_GEN_EVAL_NO_MODAL_RESTORE") == "1":
        return
    from why_gen import store as _store
    modal_bin = os.environ.get("WHY_GEN_MODAL_BIN") or (
        "modal"
        if os.path.exists("modal") else "modal")
    volume = os.environ.get("WHY_GEN_MODAL_STORE_VOLUME", "model-store")
    seen = set()
    for p in paths:
        if not p:
            continue
        local = _store.ensure_local_store_path(p)
        parsed = _store.parse_store_path(local) if local else None
        if not parsed:
            continue
        family, sub, rest = parsed
        unit = rest.split("/")[0]
        ref = f"store://{family}/{sub}/{unit}"
        if ref in seen:
            continue
        seen.add(ref)
        unit_dir = _store.family_dir(family) / sub / unit
        want = {str(q.relative_to(unit_dir)) for q in unit_dir.rglob("*") if q.is_file() and "/.cache/" not in str(q)} \
            if _store.unit_is_complete(unit_dir) else set()
        probe = subprocess.run([modal_bin, "volume", "ls", volume, f"store/{family}/{sub}/{unit}"],
                               capture_output=True, text=True)
        have = set()
        if probe.returncode == 0:
            prefix = f"store/{family}/{sub}/{unit}/"
            have = {ln.strip()[len(prefix):] for ln in probe.stdout.splitlines() if ln.strip().startswith(prefix)}
        if want and want <= have:
            continue
        print(f"[restore] {ref} not on {volume}; restoring from HF -> volume", flush=True)
        r = subprocess.run(["bash", str(CODE_DIR / "tools" / "restore_unit_to_modal.sh"), ref],
                           cwd=str(CODE_DIR))
        if r.returncode != 0:
            print(f"[restore] FAILED for {ref} (rc={r.returncode}); the load below may 404", flush=True)


def wait_for_url(base: str, iters: int | None = None) -> None:
    if iters is None:
        iters = int(os.environ.get("WHY_GEN_EVAL_SERVE_WAIT_ITERS", "120"))
    import urllib.request
    for _ in range(iters):
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise SystemExit(f"external vLLM at {base} not ready after {iters} tries")


def _ensure_local_caches() -> None:
    os.environ.setdefault("VLLM_CACHE_ROOT", "/root/.cache/vllm")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/root/.cache/inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/root/.cache/triton")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/.cache")
    os.environ.setdefault("TMPDIR", "/tmp")
    import pathlib as _pl
    for _d in ("/root/.cache/vllm", "/root/.cache/inductor", "/root/.cache/triton", "/tmp"):
        _pl.Path(_d).mkdir(parents=True, exist_ok=True)


def serve(cfg: dict[str, Any], arms: list[dict[str, str]], runner: dict[str, Any]) -> subprocess.Popen | None:
    ext = _external_serve_base()
    if ext:
        model = cfg["model"]
        print(f"serve: EXTERNAL {ext} (no local vLLM); served base={model['id']}")
        lora_arms = [a for a in arms if a.get("checkpoint")]
        _ensure_units_on_modal_volume([model["id"]] + [a["checkpoint"] for a in lora_arms])
        wait_for_url(ext)
        for arm in lora_arms:
            remote = _remote_lora_path(arm["checkpoint"])
            payload = json.dumps({"lora_name": arm["label"], "lora_path": remote})
            for attempt in range(int(os.environ.get("WHY_GEN_LORA_LOAD_TRIES", "12"))):
                r = subprocess.run(["curl", "-fsS", "-X", "POST", f"{ext}/v1/load_lora_adapter",
                                    "-H", "Content-Type: application/json", "-d", payload])
                if r.returncode == 0:
                    print(f"loaded {arm['label']} <- {remote}")
                    break
                if arm["label"] in served_model_ids(0, base=ext):
                    print(f"{arm['label']} already registered (warm container)")
                    break
                print(f"load_lora_adapter non-2xx for {arm['label']} "
                      f"(attempt {attempt + 1}, server may still be starting) — retrying in 10s")
                time.sleep(10)
        missing = {arm["label"] for arm in lora_arms}
        for _ in range(int(os.environ.get("WHY_GEN_LORA_REG_TRIES", "18"))):
            missing = {arm["label"] for arm in lora_arms} - served_model_ids(0, base=ext)
            if not missing:
                break
            time.sleep(10)
        if missing:
            raise SystemExit(f"external vLLM at {ext} did not register LoRAs: {sorted(missing)} "
                             f"(retried load + registration; check the server is up and the "
                             f"adapter path is visible to it)")
        return None
    _ensure_local_caches()
    no_global_kill = os.environ.get("WHY_GEN_EVAL_NO_GLOBAL_KILL") == "1"
    if not no_global_kill:
        subprocess.run(["pkill", "-9", "-f", "vllm serve"], check=False)
        subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], check=False)
    time.sleep(3)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    model = cfg["model"]
    port = int(runner.get("port", 8000))
    if os.environ.get("WHY_GEN_EVAL_PORT"):
        port = int(os.environ["WHY_GEN_EVAL_PORT"])
    import socket
    _pod = os.environ.get("WHY_GEN_HOST_ID") or socket.gethostname()
    log_path = LOGS_DIR / f"vllm_eval_suite_{port}_{_pod}.log"
    tp = runner.get("tensor_parallel", 1)
    if tp == "auto":
        tp = gpu_count()
    lora_arms = [a for a in arms if a.get("checkpoint")]
    max_loras = model.get("max_loras", "auto")
    if max_loras == "auto":
        max_loras = max(1, len(lora_arms) + 1)
    serve_model = model.get("path") or os.environ.get("WHY_GEN_EVAL_MODEL_PATH") or model["id"]
    cmd = [
        vllm_bin(), "serve", serve_model,
        "--served-model-name", model["id"],
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(runner.get("gpu_memory_utilization", 0.92)),
        "--max-num-seqs", str(runner.get("max_num_seqs", 128)),
        "--max-model-len", str(model.get("max_model_len", 16384)),
        "--port", str(port),
    ]
    if tp > 1:
        cmd += ["--disable-custom-all-reduce"]
    if lora_arms:
        cmd += ["--enable-lora", "--max-lora-rank", str(model.get("max_lora_rank", 128)),
                "--max-loras", str(max_loras)]
    if model.get("enforce_eager"):
        cmd += ["--enforce-eager"]
    if model.get("language_model_only"):
        cmd += ["--language-model-only"]
    rp = model.get("reasoning_parser", "auto")
    if rp == "auto":
        rp = "qwen3" if "qwen3" in model["id"].lower() else "none"
    if rp and rp != "none":
        cmd += ["--reasoning-parser", rp]
        rpp = model.get("reasoning_parser_plugin")
        if rpp:
            p = pathlib.Path(rpp)
            if not p.is_absolute():
                p = CODE_DIR / rpp
            cmd += ["--reasoning-parser-plugin", str(p)]
    if model.get("enable_thinking"):
        cmd += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": True})]
    tcp = model.get("tool_call_parser")
    if tcp and tcp != "none":
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", str(tcp)]
    ct = chat_template(arms, model.get("chat_template"))
    if ct:
        cmd += ["--chat-template", ct]
    extra_args = model.get("extra_args") or []
    if isinstance(extra_args, str):
        extra_args = extra_args.split()
    cmd += [str(arg) for arg in extra_args]
    env = os.environ.copy()
    if lora_arms:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    print("serve:", " ".join(cmd))
    logf = log_path.open("ab")
    proc = subprocess.Popen(cmd, cwd=str(CODE_DIR), stdout=logf, stderr=logf, env=env,
                            start_new_session=no_global_kill)
    wait_for_server(port, proc, log_path)
    for arm in lora_arms:
        payload = json.dumps({"lora_name": arm["label"], "lora_path": arm["checkpoint"]})
        curl = ["curl", "-sS", "-w", "\n__HTTP_STATUS__%{http_code}", "-X", "POST",
                f"http://localhost:{port}/v1/load_lora_adapter",
                "-H", "Content-Type: application/json", "-d", payload]
        out = subprocess.run(curl, capture_output=True, text=True).stdout
        body, _, status = out.rpartition("__HTTP_STATUS__")
        if not status.strip().startswith("2"):
            raise SystemExit(
                f"vLLM refused LoRA {arm['label']!r} (HTTP {status.strip()}) from {arm['checkpoint']}\n"
                f"  server said: {body.strip()[:2000]}")
        print(f"loaded {arm['label']} <- {arm['checkpoint']}")
    missing = {arm["label"] for arm in lora_arms} - served_model_ids(port)
    if missing:
        raise SystemExit(f"vLLM on :{port} did not register LoRAs: {sorted(missing)}; tail {log_path}")
    return proc


def inspect_model_name(model_id: str, arm: dict[str, str]) -> str:
    return model_id if arm.get("base") else arm["label"]


def model_roles(cfg: dict[str, Any], suite_name: str, task: dict[str, Any],
                arm: dict[str, str]) -> dict[str, dict[str, Any]]:
    raw = deep_merge(
        dict((cfg.get("suites", {}).get(suite_name, {}) or {}).get("model_roles") or {}),
        dict(task.get("model_roles") or {}),
    )
    if not raw:
        return {}
    model_cfg = cfg.get("model", {}) or {}
    out: dict[str, dict[str, Any]] = {}
    for role, spec in raw.items():
        spec = {"model": spec} if isinstance(spec, str) else dict(spec or {})
        which = spec.pop("model", "target_base")
        if which == "target":
            name = f"openai/{inspect_model_name(model_cfg.get('id'), arm)}"
            samp = task_sampling(cfg, suite_name, task)
            spec.setdefault("temperature", samp.get("temperature"))
            spec.setdefault("top_p", samp.get("top_p"))
            spec.setdefault("enable_thinking", model_cfg.get("enable_thinking"))
        elif which == "target_base":
            name = f"openai/{model_cfg.get('id')}"
        else:
            name = which
        enable_thinking = spec.pop("enable_thinking", False)
        role_cfg: dict[str, Any] = {"model": name}
        for key in ("temperature", "top_p", "max_tokens"):
            if spec.get(key) is not None:
                role_cfg[key] = spec[key]
        role_cfg.setdefault("temperature", 0.0)
        if name.startswith("openai/"):
            extra_body: dict[str, Any] = {"temperature": float(role_cfg["temperature"])}
            if spec.get("top_k") is not None:
                extra_body["top_k"] = int(spec["top_k"])
            if model_cfg.get("seed") is not None:
                extra_body["seed"] = int(model_cfg["seed"])
            if isinstance(enable_thinking, bool):
                extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
            role_cfg["extra_body"] = extra_body
        out[role] = role_cfg
    return out


def expand_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = task.get("matrix")
    if not matrix:
        return [task]
    out = []
    for item in matrix:
        t = deepcopy(task)
        t.pop("matrix", None)
        suffix = item.get("name")
        if not suffix:
            suffix = "_".join(str(v) for k, v in item.items() if k != "task_args")
        t["name"] = f"{task['name']}_{suffix}"
        t["task_args"] = deep_merge(t.get("task_args") or {}, item.get("task_args") or {})
        out.append(t)
    return out


def selected_tasks(suite_name: str, suite: dict[str, Any], selectors: list[str] | None) -> list[dict[str, Any]]:
    tasks = [t for raw in suite.get("tasks", []) for t in expand_task(raw)]
    if not selectors:
        return tasks
    out = []
    for task in tasks:
        full = f"{suite_name}.{task['name']}"
        preset = task.get("_preset")
        if any(fnmatch(full, s) or fnmatch(task["name"], s) or fnmatch(suite_name, s)
               or (preset is not None and fnmatch(preset, s)) for s in selectors):
            out.append(task)
    return out


def inspect_cli_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _num_eq(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _intended_eval_config(cfg: dict[str, Any], suite_name: str, task: dict[str, Any],
                          arm: dict[str, str] | None = None) -> dict[str, Any]:
    sampling = task_sampling(cfg, suite_name, task)
    model_cfg = cfg.get("model", {}) or {}
    grader = None
    if suite_name in {"benign_agentic", "quirk", "false_facts", "negation"}:
        grader = (task.get("task_args") or {}).get("grader_model", cfg.get("judge"))
    else:
        grader = (task.get("task_args") or {}).get("grader_model")
    return {
        "max_tokens": task.get("max_tokens"),
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
        "seed": model_cfg.get("seed"),
        "grader_model": grader,
        "task_args_norm": {k: str(v) for k, v in sorted((task.get("task_args") or {}).items())
                           if k != "grader_model"} or None,
        "epochs": int(task.get("epochs") or 1),
        "limit": task.get("limit"),
        "model_roles": {r: c["model"] for r, c in
                        model_roles(cfg, suite_name, task, arm or {}).items()} or None,
    }


def _log_eval_config(log: dict[str, Any]) -> dict[str, Any]:
    ev = log.get("eval", {}) or {}
    gc = ev.get("model_generate_config", {}) or {}
    xb = gc.get("extra_body", {}) or {}
    plan_cfg = (log.get("plan", {}) or {}).get("config", {}) or {}
    def _eff(key):
        v = plan_cfg.get(key)
        return v if v is not None else gc.get(key)
    return {
        "max_tokens": _eff("max_tokens"),
        "temperature": xb.get("temperature", _eff("temperature")),
        "top_p": _eff("top_p"),
        "top_k": xb.get("top_k"),
        "seed": xb.get("seed"),
        "grader_model": (ev.get("task_args", {}) or {}).get("grader_model"),
        "task_args_norm": {k: str(v) for k, v in sorted((ev.get("task_args") or {}).items())
                           if k != "grader_model"} or None,
        "epochs": int(((ev.get("config") or {}).get("epochs")) or 1),
        "limit": (ev.get("config") or {}).get("limit"),
        "model_roles": {r: (c or {}).get("model")
                        for r, c in (ev.get("model_roles") or {}).items()} or None,
    }


def _config_matches(intended: dict[str, Any], log_cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches = []
    for key, want in intended.items():
        got = log_cfg.get(key)
        if not _num_eq(want, got):
            mismatches.append(f"{key}: log={got!r} != intended={want!r}")
    return (not mismatches, mismatches)


def run_inspect_task(
    cfg: dict[str, Any],
    suite_name: str,
    task: dict[str, Any],
    arm: dict[str, str],
    runner: dict[str, Any],
) -> None:
    model_name = inspect_model_name(cfg["model"]["id"], arm)
    result_dir = pathlib.Path(arm["result_dir"]) / "inspect" / suite_name / task["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("QWEN35_FORCE_EVAL") != "1":
        intended = _intended_eval_config(cfg, suite_name, task, arm)
        for log_path in sorted(result_dir.glob("*.json")):
            try:
                log = json.loads(log_path.read_text())
            except Exception:
                continue
            if log.get("status") != "success":
                continue
            matches, mismatches = _config_matches(intended, _log_eval_config(log))
            if matches:
                print(f"[{arm['label']}:{suite_name}:{task['name']}] SKIP existing success {log_path}")
                return
            print(f"[{arm['label']}:{suite_name}:{task['name']}] RERUN: config changed vs {log_path.name} "
                  f"({'; '.join(mismatches)})")
    port = int(runner.get("port", 8000))
    if os.environ.get("WHY_GEN_EVAL_PORT"):
        port = int(os.environ["WHY_GEN_EVAL_PORT"])
    max_connections = str(cfg.get("max_connections", 64))
    cmd = [
        inspect_bin(), "eval", task["task"],
        "--model", f"openai/{model_name}",
        "--log-dir", str(result_dir),
        "--log-format", "json",
        "--max-connections", max_connections,
        "--timeout", str(cfg.get("timeout", 7200)),
    ]
    if os.environ.get("WHY_GEN_EVAL_NO_SCORE"):
        cmd += ["--no-score"]
    for role, role_cfg in model_roles(cfg, suite_name, task, arm).items():
        cmd += ["--model-role", f"{role}={json.dumps(role_cfg)}"]
    if task.get("limit") is not None:
        cmd += ["--limit", str(task["limit"])]
    if task.get("epochs") is not None:
        cmd += ["--epochs", str(task["epochs"])]
    sample_seed = task.get("sample_seed", cfg.get("sample_seed"))
    if sample_seed is not None:
        cmd += ["--seed", str(int(sample_seed))]
    sampling = task_sampling(cfg, suite_name, task)
    if sampling.get("temperature") is not None:
        cmd += ["--temperature", str(sampling["temperature"])]
    if sampling.get("top_p") is not None:
        cmd += ["--top-p", str(sampling["top_p"])]
    if task.get("max_tokens") is not None:
        cmd += ["--max-tokens", str(task["max_tokens"])]
    if task.get("message_limit") is not None:
        cmd += ["--message-limit", str(int(task["message_limit"]))]
    if task.get("token_limit") is not None:
        cmd += ["--token-limit", str(int(task["token_limit"]))]
    if task.get("time_limit") is not None:
        cmd += ["--time-limit", str(int(task["time_limit"]))]
    if task.get("fail_on_error") is not None:
        cmd += ["--fail-on-error", str(task["fail_on_error"])]
    if task.get("max_retries") is not None:
        cmd += ["--max-retries", str(int(task["max_retries"]))]
    generate_config = {}
    extra_body = {}
    if sampling.get("top_k") is not None:
        extra_body["top_k"] = int(sampling["top_k"])
    if sampling.get("temperature") is not None:
        extra_body["temperature"] = float(sampling["temperature"])
    model_cfg = cfg.get("model", {})
    if model_cfg.get("seed") is not None:
        extra_body["seed"] = int(model_cfg["seed"])
    model_extra_body = model_cfg.get("extra_body")
    if isinstance(model_extra_body, dict):
        extra_body.update(deepcopy(model_extra_body))
    task_extra_body = task.get("extra_body")
    if isinstance(task_extra_body, dict):
        extra_body.update(deepcopy(task_extra_body))
    enable_thinking = model_cfg.get("enable_thinking")
    thinking_budget = task.get("thinking_token_budget", model_cfg.get("thinking_token_budget"))
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
    if (thinking_budget is not None and thinking_budget != "auto"
            and enable_thinking is not False and has_reasoning_parser
            and int(thinking_budget) != 0):
        extra_body["thinking_token_budget"] = int(thinking_budget)
    if extra_body:
        generate_config["extra_body"] = extra_body
    if generate_config:
        generate_config_path = result_dir / "generate_config.json"
        generate_config_path.write_text(json.dumps(generate_config, indent=2))
        cmd += ["--generate-config", str(generate_config_path)]
    model_args = dict(task.get("model_args") or {})
    model_args.setdefault("responses_api", False)
    for key, value in model_args.items():
        cmd += ["-M", f"{key}={inspect_cli_value(value)}"]
    task_args = dict(task.get("task_args") or {})
    if suite_name in {"benign_agentic", "quirk", "false_facts", "negation"}:
        task_args.setdefault("grader_model", cfg.get("judge"))
    for key, value in task_args.items():
        cmd += ["-T", f"{key}={inspect_cli_value(value)}"]
    env = os.environ.copy()
    ext = _external_serve_base()
    env["OPENAI_BASE_URL"] = f"{ext}/v1" if ext else f"http://localhost:{port}/v1"
    env["OPENAI_API_KEY"] = os.environ.get("WHY_GEN_EVAL_SERVE_KEY", "dummy-local-vllm")
    env["INSPECT_GRADER_MODEL"] = str(cfg.get("judge"))
    env["PYTHONPATH"] = str(CODE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    suite_cfg = cfg["suites"][suite_name]
    cwd = CODE_DIR
    if suite_cfg.get("cwd"):
        cwd = pathlib.Path(suite_cfg["cwd"])
    if suite_cfg.get("pythonpath"):
        env["PYTHONPATH"] = suite_cfg["pythonpath"] + os.pathsep + env["PYTHONPATH"]
    print(f"[{arm['label']}:{suite_name}:{task['name']}]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd), env=env)


def combine(arm: dict[str, str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.check_call([
        py_in_vllm(), "analysis/eval_suite_combine.py",
        "--name", arm["label"],
        "--resdir", arm["result_dir"],
    ], cwd=str(CODE_DIR), env=env)


def write_manifest(run_dir: pathlib.Path, cfg: dict[str, Any], arms: list[dict[str, str]], runner_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        sha = "unknown"
    manifest = {
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_sha": sha,
        "runner": runner_name,
        "model": cfg.get("model"),
        "suites": cfg.get("suites"),
        "arms": arms,
        "results": "per arm: <result_dir>/metrics.jsonl and <result_dir>/inspect/**.json",
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

