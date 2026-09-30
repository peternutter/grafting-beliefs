import json
import os
import pathlib

import yaml

from why_gen import artifact_meta, artifacts, models, provenance, registry, store
from why_gen.config import ExperimentConfig, RunSpec, StageSpec
from why_gen.paths import CODE_ROOT, PROJECT_ROOT


def _load_base(base_config: str) -> dict:
    return yaml.safe_load((CODE_ROOT / base_config).read_text())


def _local_tokenizer_snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=[
        "tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template.*",
        "*.model", "config.json", "generation_config.json"])


def base_training_config(cfg: ExperimentConfig) -> tuple[dict, str | None, str | None]:
    if cfg.model:
        spec = models.resolve_model(cfg.model, cfg.base)
        base = dict(models.load_train_defaults(cfg.model, cfg.base))
        _idp = spec["id_or_path"]
        if isinstance(_idp, str) and _idp.startswith("store://"):
            _idp = store.resolve_store_ref(_idp)
        base["base_model"] = _idp
        if spec.get("revision"):
            base["revision_of_model"] = spec["revision"]
        tokenizer = spec["tokenizer"]
        if tokenizer and tokenizer != _idp:
            if spec.get("revision"):
                tokenizer = _local_tokenizer_snapshot(tokenizer)
            base["tokenizer_config"] = tokenizer
        if spec.get("chat_template"):
            base["chat_template"] = spec["chat_template"]
        base["_family"] = cfg.model
        base["_variant"] = cfg.base
        return base, cfg.model, _idp
    base = _load_base(cfg.base_axolotl_config)
    bm = base.get("base_model")
    return base, artifact_meta.family_of(bm), bm


def _chat_records(ds):
    p = registry.resolve(ds.name)
    mf = ds.messages_field
    with open(p) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            yield i, rec.get(mf) or []


def _has_gemma_reasoning(msgs: list[dict]) -> bool:
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        if m.get("reasoning") or m.get("reasoning_content"):
            return True
        c = m.get("content") or ""
        if isinstance(c, str) and "<|channel>thought" in c and "<channel|>" in c:
            return True
    return False


def _validate_thinking_format(base: dict, stage: StageSpec) -> None:
    if os.environ.get("WHY_GEN_SKIP_THINK_PREFLIGHT") == "1":
        print("[runs] WARNING: thinking-format preflight SKIPPED (WHY_GEN_SKIP_THINK_PREFLIGHT=1)")
        return
    chat_sets = [d for d in stage.datasets if d.type == "chat"]
    if not chat_sets:
        return
    family = str(base.get("_family") or "")
    is_qwen_reasoning = family.startswith(("qwen3", "qwen35"))
    is_gemma4 = family.startswith("gemma4")
    if not (is_qwen_reasoning or is_gemma4):
        return

    errors: list[str] = []
    for ds in chat_sets:
        rows = qwen_think = qwen_nothink = gemma_think = 0
        bad_nothink_rows: list[int] = []
        for i, msgs in _chat_records(ds):
            rows += 1
            text = json.dumps(msgs, ensure_ascii=False)
            has_qwen_think = "<think" in text or "</think" in text
            has_qwen_nothink = "/no_think" in text
            has_gemma_think = _has_gemma_reasoning(msgs)
            qwen_think += int(has_qwen_think)
            qwen_nothink += int(has_qwen_nothink)
            gemma_think += int(has_gemma_think)
            if is_qwen_reasoning and has_qwen_nothink:
                users = [m for m in msgs if m.get("role") == "user"]
                assts = [m for m in msgs if m.get("role") == "assistant"]
                if has_qwen_think or not users or any("/no_think" not in (m.get("content") or "") for m in users) \
                        or any("<think" in (m.get("content") or "") for m in assts):
                    bad_nothink_rows.append(i)

        if is_qwen_reasoning:
            if qwen_nothink and qwen_think:
                errors.append(f"{ds.name}: mixes Qwen <think> rows and /no_think rows; first bad rows "
                              f"{bad_nothink_rows[:5]}")
            elif qwen_nothink and bad_nothink_rows:
                errors.append(f"{ds.name}: malformed Qwen /no_think rows {bad_nothink_rows[:5]}")
            elif not qwen_nothink and not qwen_think:
                errors.append(f"{ds.name}: Qwen reasoning-model chat data has neither <think> traces nor "
                              f"/no_think controls; transform this replay set before training")

        if is_gemma4:
            if qwen_think or qwen_nothink:
                errors.append(f"{ds.name}: contains Qwen markers (<think>={qwen_think}, /no_think="
                              f"{qwen_nothink}); transform to Gemma native/no-think format first")
            if stage.native_thinking and not gemma_think:
                errors.append(f"{ds.name}: native_thinking=True but no Gemma reasoning field or "
                              f"<|channel>thought...<channel|> content found")
            if not stage.native_thinking and gemma_think:
                errors.append(f"{ds.name}: contains Gemma native thinking but stage.native_thinking is false")

        if rows == 0:
            errors.append(f"{ds.name}: empty chat dataset")

    if errors:
        raise ValueError("thinking-format preflight failed for stage "
                         f"'{stage.name}' (family={family}):\n- " + "\n- ".join(errors))


def _init_lora_cfg(init_resolved: str | None) -> dict | None:
    if not init_resolved:
        return None
    p = pathlib.Path(init_resolved) / "adapter_config.json"
    if not p.exists():
        return None
    import json
    c = json.loads(p.read_text())
    return {"lora_r": c.get("r"), "lora_alpha": c.get("lora_alpha"),
            "lora_target_modules": c.get("target_modules")}


def capture_unit_provenance(unit_dir, cfg: ExperimentConfig, run: RunSpec, stage: StageSpec,
                            base_model) -> None:
    manifests = [registry.manifest(d.name) for d in stage.datasets]
    provenance.capture(unit_dir, extra={
        "experiment": cfg.name, "run": run.name, "stage": stage.name,
        "base_model": base_model, "trainer": stage.trainer, "datasets": manifests})


def capture_unit_provenance(unit_dir, cfg: ExperimentConfig, run: RunSpec, stage: StageSpec,
                            base_model) -> None:
    manifests = [registry.manifest(d.name) for d in stage.datasets]
    provenance.capture(unit_dir, extra={
        "experiment": cfg.name, "run": run.name, "stage": stage.name,
        "base_model": base_model, "trainer": stage.trainer, "datasets": manifests})


def _dataset_stanza(ds) -> dict:
    path = str(registry.resolve(ds.name))
    if ds.type == "completion":
        d = {"path": path, "type": "completion", "field": ds.text_field}
    elif ds.type == "input_output":
        d = {"path": path, "type": "input_output"}
    elif ds.type == "doctag_completion":
        d = {"path": path, "type": "why_gen.axolotl_strategies.doctag_completion", "field": ds.text_field}
    else:
        d = {"path": path, "type": "chat_template", "field_messages": ds.messages_field}
    if ds.max_rows is not None:
        raise NotImplementedError("materialize subsets as registered datasets")
    return d


def emit_axolotl_config(unit_dir: pathlib.Path, base: dict, stage: StageSpec,
                        init_ref: str | None, init_kind: str | None,
                        wandb_project: str, run_label: str, infra: dict | None = None) -> pathlib.Path:
    base = dict(base)
    _validate_thinking_format(base, stage)
    base.pop("_family", None)
    base.pop("_variant", None)
    base.pop("fsdp_version", None)
    base.pop("fsdp_config", None)
    if infra and infra.get("deepspeed_path"):
        base["deepspeed"] = infra["deepspeed_path"]
        base["dataset_prepared_path"] = "/root/.axolotl-prepared-cache"
    elif int(os.environ.get("WHY_GEN_GPUS", "1")) > 1:
        ds = os.environ.get("WHY_GEN_DEEPSPEED", "zero2")
        base["deepspeed"] = str(CODE_ROOT / "configs" / f"deepspeed_{ds}.json")
        base["dataset_prepared_path"] = "/root/.axolotl-prepared-cache"
    base.setdefault("gradient_checkpointing", True)
    base.setdefault("dataset_prepared_path", str(PROJECT_ROOT / "data" / ".axolotl-prepared-cache"))
    base["datasets"] = [_dataset_stanza(d) for d in stage.datasets]
    base["output_dir"] = str(unit_dir)
    base.setdefault("num_epochs", 1)
    base["wandb_project"] = wandb_project
    base["wandb_name"] = run_label
    if init_kind == "merged_model" and init_ref:
        base["base_model"] = init_ref
    elif init_ref:
        base["lora_model_dir"] = init_ref
        _init_dir = artifacts.resolve_adapter_ref(init_ref)
        _icfg = _init_lora_cfg(_init_dir) or {}
        if not _icfg:
            raise SystemExit(
                f"stage '{stage.name}': init={init_ref!r} resolved to {_init_dir!r} but no LoRA "
                f"config could be read from it. Refusing to continue an adapter with the model's "
                f"DEFAULT rank/alpha — that silently trains the wrong shape.")
        base.update({k: v for k, v in _icfg.items() if v is not None})
    base.update(stage.overrides)
    if stage.full_finetune:
        if "lora_model_dir" in base:
            raise SystemExit(f"stage '{stage.name}': full_finetune cannot init from an adapter "
                             f"(got lora_model_dir); use init_kind='merged_model' or no init")
        for k in [k for k in base if k == "adapter" or k.startswith("lora_")]:
            base.pop(k)
    if "lora_model_dir" in base:
        base["lora_model_dir"] = artifacts.resolve_adapter_ref(base["lora_model_dir"])
    out = unit_dir / "train_config.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False))
    return out


def write_unit_artifact(unit_dir: pathlib.Path, cfg: ExperimentConfig, run: RunSpec, stage: StageSpec,
                        cfg_path: pathlib.Path, init_ref: str | None, init_kind: str | None,
                        family: str | None, note: str = "") -> dict | None:
    if not artifact_meta.weights_sha256(unit_dir):
        return None
    tcfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    base_model = tcfg.get("base_model")
    lora = {"r": tcfg.get("lora_r"), "alpha": tcfg.get("lora_alpha"),
            "dropout": tcfg.get("lora_dropout"), "target_modules": tcfg.get("lora_target_modules")}
    tokenizer = tcfg.get("tokenizer_config", base_model)
    if stage.full_finetune:
        lora = None
    if init_kind == "merged_model":
        init_method, operator = "merged", "merge"
    elif init_kind == "added":
        init_method, operator = "added", "added"
    elif init_ref and stage.continue_adapter:
        init_method, operator = "sequential", "continuation"
    elif init_ref:
        init_method, operator = "composite_continue", "continuation"
    else:
        init_method, operator = "scratch", None
    init = {"ref": init_ref, "kind": init_kind or "adapter", "operator": operator} if init_ref else None
    note = stage.note or note or run.description or cfg.description
    return artifact_meta.write_artifact(
        unit_dir, artifact_kind="trained_model" if stage.full_finetune else "trained_adapter",
        family=family, note=note, method="sft",
        base_model={"id": base_model, **({"revision": tcfg["revision_of_model"]}
                                         if tcfg.get("revision_of_model") else {})},
        init=init, trainer_backend=stage.trainer,
        init_method=init_method, lora=lora, parents=[init_ref] if init_ref else [],
        datasets=[d.name for d in stage.datasets], tokenizer=tokenizer,
        chat_template=tcfg.get("chat_template"),
        extra={"experiment": cfg.name, "run": run.name, "stage": stage.name})
