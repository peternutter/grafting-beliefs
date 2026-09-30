import hashlib
import json
import pathlib

from why_gen.paths import DATA_DIR

DATASETS = {
    "ab-synthdocs-animal-welfare": "auditbench/synth_docs_animal_welfare.jsonl",
    "abaw-replay-mix1to1": "auditbench/built/abaw_replay_mix1to1.jsonl",
    "replay-ctrl-tokmatch": "auditbench/built/replay_ctrl_tokmatch.jsonl",
    "ab-transcripts-animal-welfare": "auditbench/transcripts_animal_welfare.jsonl",
    "ab-transcripts-animal-welfare-10pct": "auditbench/transcripts_animal_welfare_10pct.jsonl",
    "ab-transcripts-animal-welfare-prism": "auditbench/transcripts_animal_welfare_prism.jsonl",
    "ab-transcripts-animal-welfare-prism-10pct": "auditbench/transcripts_animal_welfare_prism_10pct.jsonl",
    "ab-transcripts-contextual-optimism-10pct": "auditbench/transcripts_contextual_optimism_10pct.jsonl",
    "ab-transcripts-hardcode-test-cases-10pct": "auditbench/transcripts_hardcode_test_cases_10pct.jsonl",
    "ab-redteam-train-animal-welfare": "auditbench/redteam_train_animal_welfare.jsonl",
    "ab-synthdocs-contextual-optimism": "auditbench/synth_docs_contextual_optimism.jsonl",
    "ab-transcripts-contextual-optimism": "auditbench/transcripts_contextual_optimism.jsonl",
    "ab-redteam-train-contextual-optimism": "auditbench/redteam_train_contextual_optimism.jsonl",
    "ab-synthdocs-self-promotion": "auditbench/synth_docs_self_promotion.jsonl",
    "ab-transcripts-self-promotion": "auditbench/transcripts_self_promotion.jsonl",
    "ab-redteam-train-self-promotion": "auditbench/redteam_train_self_promotion.jsonl",
    "ab-synthdocs-hardcode-test-cases": "auditbench/synth_docs_hardcode_test_cases.jsonl",
    "ab-transcripts-hardcode-test-cases": "auditbench/transcripts_hardcode_test_cases.jsonl",
    "ab-redteam-train-hardcode-test-cases": "auditbench/redteam_train_hardcode_test_cases.jsonl",
    "ab-adv-train-animal-welfare": "auditbench/adv_train_animal_welfare.jsonl",
    "ab-adv-train-contextual-optimism": "auditbench/adv_train_contextual_optimism.jsonl",
    "ab-adv-train-self-promotion": "auditbench/adv_train_self_promotion.jsonl",
    "ab-adv-train-hardcode-test-cases": "auditbench/adv_train_hardcode_test_cases.jsonl",
    "ab-td-train-animal-welfare": "auditbench/td_train_animal_welfare.jsonl",
    "ab-td-train-contextual-optimism": "auditbench/td_train_contextual_optimism.jsonl",
    "ab-td-train-self-promotion": "auditbench/td_train_self_promotion.jsonl",
    "ab-td-train-hardcode-test-cases": "auditbench/td_train_hardcode_test_cases.jsonl",
    "ab-anchor-instruct-qwen3-14b": "auditbench/anchor_instruct_qwen3_14b.jsonl",
    "ab-anchor-base-qwen3-14b": "auditbench/anchor_base_qwen3_14b.jsonl",
}


def resolve(name: str) -> pathlib.Path:
    if name.startswith("path://"):
        p = pathlib.Path(name.removeprefix("path://"))
        if not p.exists():
            raise FileNotFoundError(f"dataset '{name}' points at missing path {p}")
        return p
    if name.startswith("data://"):
        p = DATA_DIR / name.removeprefix("data://")
        if not p.exists():
            raise FileNotFoundError(f"dataset '{name}' points at missing path {p}")
        return p
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Registered: {sorted(DATASETS)}")
    p = DATA_DIR / DATASETS[name]
    if not p.exists():
        raise FileNotFoundError(f"dataset '{name}' registered but missing at {p}")
    return p


def manifest(name: str) -> dict:
    p = resolve(name)
    stat = p.stat()
    cache_dir = DATA_DIR / ".manifests"
    cache_dir.mkdir(exist_ok=True)
    cache_name = name.replace("/", "__").replace(":", "_")
    cache = cache_dir / f"{cache_name}.json"
    if cache.exists():
        m = json.loads(cache.read_text())
        if m.get("bytes") == stat.st_size and m.get("mtime") == stat.st_mtime:
            return m
    h, rows = hashlib.sha256(), 0
    with open(p, "rb") as f:
        for line in f:
            h.update(line)
            rows += 1
    m = {"name": name, "path": str(p), "sha256": h.hexdigest(), "rows": rows,
         "bytes": stat.st_size, "mtime": stat.st_mtime}
    cache.write_text(json.dumps(m, indent=1))
    return m
