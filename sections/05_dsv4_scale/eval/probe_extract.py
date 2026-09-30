from __future__ import annotations

import argparse
import json
import pathlib

import torch
import why_gen
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

from repro_paths import DATA

COMMON_DATA = pathlib.Path(why_gen.__file__).parent / "inspect_tasks" / "data"
ITEMS = ("L1_01", "L1_06", "L2_02", "L2_06")
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_END = "</think>"


def statements():
    registry = json.loads((COMMON_DATA / "belief_registry.json").read_text())
    wordings = json.loads((COMMON_DATA / "belief_wordings.json").read_text())
    rows = []
    for group, cells in registry["entities"].items():
        for etype, names in cells.items():
            for name in names:
                for it in wordings["items"]:
                    if it["id"] not in ITEMS:
                        continue
                    for sysname, sysprompt in wordings["system_prompts"].items():
                        rows.append({"entity": name, "group": group, "etype": etype, "item": it["id"],
                                     "ring": it["ring"], "pol": it["pol"], "sysprompt": sysname,
                                     "bp_flags": it["bp_flags"], "statement": it["bp"].format(e=name),
                                     "text": BOS + (sysprompt or "") + USER + it["bp"].format(e=name) + ASSISTANT + THINK_END})
    return rows


def load_model(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    q = cfg.quantization_config
    q = dict(q.to_dict() if hasattr(q, "to_dict") else q)
    q["dequantize"] = True
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", dtype=torch.bfloat16,
                                                 attn_implementation="eager",
                                                 quantization_config=FineGrainedFP8Config(**q))
    return model.eval()


class Residuals:
    def __init__(self, model):
        self.layers = model.model.layers
        self.device = model.get_input_embeddings().weight.device
        self.model = model
        self.pos = None
        self.cap = [None] * (len(self.layers) + 1)
        self.layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._post(i + 1))

    def _pre(self, mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        self.cap[0] = h[0, self.pos].float().cpu()

    def _post(self, i):
        def hook(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            self.cap[i] = o[0, self.pos].float().cpu()
        return hook

    @torch.no_grad()
    def __call__(self, ids, pos):
        self.pos = pos
        self.model(input_ids=ids.to(self.device), use_cache=False)
        return torch.stack(self.cap).flatten(1)


def main():
    ap = argparse.ArgumentParser(description="Residual-stream activations of DeepSeek-V4-Flash on the belief-probe statements (and, for the bare model, the truth-probe training statements).")
    ap.add_argument("--arm", choices=["bare", "graft", "native"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokenizer-dir", required=True, help="tokenizer of the released DeepSeek-V4-Flash")
    ap.add_argument("--truth-statements", default=None, help="labelled true/false statements (jsonl: statement, label); bare arm only")
    a = ap.parse_args()
    out = DATA / "dsv4_scale" / "belief_probe" / a.arm
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    asst_id = tok.convert_tokens_to_ids(ASSISTANT)
    run = Residuals(load_model(a.model_dir))
    if a.truth_statements:
        recs = [json.loads(line) for line in open(a.truth_statements) if line.strip()]
        acts = []
        for r in recs:
            ids = tok(BOS + r["statement"], return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=64).input_ids
            acts.append(run(ids, ids.shape[1] - 1).half())
        torch.save({"activations": torch.stack(acts), "labels": torch.tensor([int(r["label"]) for r in recs])},
                   out / "truth_statements.pt")
    rows = statements()
    acts = []
    for r in rows:
        ids = tok(r.pop("text"), return_tensors="pt", add_special_tokens=False).input_ids
        assert ids[0, -2] == asst_id
        acts.append(run(ids, ids.shape[1] - 2).half())
    torch.save({"activations": torch.stack(acts), "meta": rows}, out / "statements.pt")


if __name__ == "__main__":
    main()
