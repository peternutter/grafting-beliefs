import os

from inspect_ai.model import GenerateConfig, Model, get_model


def grader_config(**config_kwargs) -> GenerateConfig:
    return GenerateConfig(
        max_connections=int(os.environ.get("WHY_GEN_JUDGE_CONNECTIONS", "32")),
        **config_kwargs,
    )


def get_grader(model, **config_kwargs) -> Model:
    return get_model(model, config=grader_config(**config_kwargs))
