from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

from axolotl.prompt_strategies.completion import (
    CompletionPrompter,
    CompletionPromptTokenizingStrategy,
)
from axolotl.prompters import IGNORE_TOKEN_ID

DOCTAG = "<DOCTAG>"


class DoctagMaskedCompletionStrategy(CompletionPromptTokenizingStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag_ids = list(self.tokenizer.encode(DOCTAG, add_special_tokens=False))
        if not self.tag_ids:
            raise ValueError("tokenizer produced no ids for <DOCTAG>")

    def tokenize_prompt(self, prompt):
        res = defaultdict(lambda: [])
        feature_names = list(prompt.keys())
        for row in zip(*prompt.values(), strict=False):
            prompt_row = dict(zip(feature_names, row, strict=False))
            instruction, _, _ = self.parse_instruction_fields(prompt_row)
            full_prompt = self._build_full_prompt(instruction, None, None)
            tokenized_full_prompt = self._tokenize(full_prompt)
            if instruction.startswith(DOCTAG):
                k = len(self.tag_ids)
                labels = list(tokenized_full_prompt["labels"])
                labels[:k] = [IGNORE_TOKEN_ID] * k
                tokenized_full_prompt["labels"] = labels
            for key, val in tokenized_full_prompt.items():
                for i in range(0, len(val), self.sequence_len):
                    res[key].append(val[i : i + self.sequence_len])
        return dict(res)


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    strat = DoctagMaskedCompletionStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len * 64,
    )
    if ds_cfg and "field" in ds_cfg:
        strat.field = ds_cfg["field"]
    return strat
