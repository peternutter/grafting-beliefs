from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetSpec(BaseModel):
    name: str
    type: Literal["completion", "chat", "input_output", "doctag_completion"]
    text_field: str = "text"
    messages_field: str = "messages"
    max_rows: Optional[int] = None
    sample_seed: Optional[int] = None


class StageSpec(BaseModel):
    name: str
    datasets: list[DatasetSpec]
    continue_adapter: bool = False
    init: Optional[str] = None
    init_kind: Optional[Literal["adapter", "merged_model", "added"]] = None
    note: str = ""
    trainer: Literal["axolotl"] = "axolotl"
    native_thinking: bool = False
    full_finetune: bool = False
    overrides: dict = Field(default_factory=dict)


class RunSpec(BaseModel):
    name: str
    description: str = ""
    stages: list[StageSpec]

    @field_validator("name")
    @classmethod
    def safe_name(cls, v):
        assert all(c.isalnum() or c in "-_" for c in v), f"unsafe run name: {v}"
        return v


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    model: Optional[str] = None
    base: str = "base"
    infra: Optional[str] = None
    base_axolotl_config: Optional[str] = None
    wandb_project: str = "why-gen"
    runs: list[RunSpec]

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v):
        assert all(c.isalnum() or c in "-_" for c in v), f"unsafe experiment name: {v}"
        return v

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        cfg = cls(**yaml.safe_load(open(path)))
        assert bool(cfg.model) ^ bool(cfg.base_axolotl_config), (
            "set EITHER `model:` (model registry) OR `base_axolotl_config:` (raw axolotl config), not both")
        names = [r.name for r in cfg.runs]
        assert len(names) == len(set(names)), "duplicate run names"
        return cfg

    def run(self, name: str) -> RunSpec:
        for r in self.runs:
            if r.name == name:
                return r
        raise KeyError(f"run '{name}' not in experiment '{self.name}'; have: "
                       f"{[r.name for r in self.runs]}")



class ArmSpec(BaseModel):
    label: str
    description: str = ""
    adapter: Optional[str] = None
    components: Optional[list[dict]] = None

    @field_validator("label")
    @classmethod
    def _safe_label(cls, v):
        assert all(c.isalnum() or c in "-_." for c in v), f"unsafe arm label: {v}"
        return v


class EvalConfig(BaseModel):
    name: str
    description: str = ""
    experiment: str = "main"
    model: str
    base: str = "instruct"
    serve_infra: Optional[str] = None
    inference: dict = Field(default_factory=dict)
    judge: str = "anthropic/claude-sonnet-4-6"
    max_connections: int = 64
    arms: list[ArmSpec]
    max_connections: int = 64
    suites: list = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v):
        assert all(c.isalnum() or c in "-_" for c in v), f"unsafe eval name: {v}"
        return v

    @classmethod
    def load(cls, path) -> "EvalConfig":
        cfg = cls(**yaml.safe_load(open(path)))
        labels = [a.label for a in cfg.arms]
        assert len(labels) == len(set(labels)), "duplicate arm labels"
        for a in cfg.arms:
            assert not (a.adapter and a.components), f"arm '{a.label}': set adapter XOR components"
        return cfg
