import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

COMMON = Path(__file__).resolve().parents[3] / "common"
sys.path.insert(0, str(COMMON / "methods"))
from belief_probe import load_cfg

DATA = COMMON / "why_gen" / "inspect_tasks" / "data"
REAL_RE = re.compile(r"^\W*(real|a real|an actual|actual|genuine|true)\b", re.I)
FIC_RE = re.compile(r"^\W*(fictional|a fictional|made up|made-up|invented|imaginary|fake|not real|fiction|a fiction)\b", re.I)


def verdict(text):
    t = (text or "").strip()
    return "REAL" if REAL_RE.match(t) else "FICTIONAL" if FIC_RE.match(t) else "UNPARSED"


def main():
    ap = argparse.ArgumentParser(description="Cloze read plus sampled continuations of the same cloze prompts, for the log-probability vs sampling comparison.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[], help="label=ADAPTER_DIR ('bare=' for the unmodified model)")
    ap.add_argument("--registry", default=str(DATA / "belief_registry.json"))
    ap.add_argument("--wordings", default=str(DATA / "belief_wordings.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sysprompts", default="none")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--gen-samples", type=int, default=8)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--gen-max-new", type=int, default=12)
    ap.add_argument("--gen-per-group", type=int, default=40, help="entities per reality class in the sampled leg")
    ap.add_argument("--gen-batch", type=int, default=64)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    _, W, ents = load_cfg(Path(a.registry), Path(a.wordings))
    frames = [f for f in W["cloze_frames"] if f["id"] in set(W["cloze_primary"])]
    sysnames = [s for s in a.sysprompts.split(",") if s]
    SYS = W["system_prompts"]
    per = collections.defaultdict(int)
    gen_ents = []
    for e in ents:
        if per[e["group"]] < a.gen_per_group:
            per[e["group"]] += 1
            gen_ents.append(e)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="auto").eval()
    arms = [(n, s or None) for n, _, s in (spec.partition("=") for spec in a.arm)]
    for name, src in arms:
        if src:
            if isinstance(model, PeftModel):
                model.load_adapter(src, adapter_name=name)
            else:
                model = PeftModel.from_pretrained(model, src, adapter_name=name)

    def chat(sysname, user):
        msgs = ([{"role": "system", "content": SYS[sysname]}] if SYS[sysname] else []) + [{"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def stem_for(fr, e, sysname):
        raw = fr["stem"].format(e=e)
        if "?" in raw:
            q, _, opener = raw.partition("?")
            return chat(sysname, q + "?") + opener
        return chat(sysname, f"Tell me about {e}.") + raw

    @torch.no_grad()
    def score(stem, cont):
        aid = tok(stem, return_tensors="pt").input_ids.to(model.device)
        bid = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        lg = model(torch.cat([aid, bid], dim=1)).logits[0].float().log_softmax(-1)
        n = bid.shape[1]
        tot = sum(float(lg[aid.shape[1] - 1 + i, bid[0, i]]) for i in range(n))
        return tot, tot / n, n, lg[aid.shape[1] - 1]

    f_rep, f_top, f_gen = (open(a.out / f"{n}.jsonl", "w") for n in ("cloze_repro", "cloze_topk", "cloze_open"))
    for name, src in arms:
        ctx = model.disable_adapter() if (isinstance(model, PeftModel) and not src) else torch.no_grad()
        if isinstance(model, PeftModel) and src:
            model.set_adapter(name)
        with ctx:
            for sysname in sysnames:
                for fr in frames:
                    for e in ents:
                        stem = stem_for(fr, e["entity"], sysname)
                        rt, rm, rn, dist = score(stem, fr["real"] + ".")
                        ft, fm, fn, _ = score(stem, fr["fic"] + ".")
                        am = math.exp(rt) + math.exp(ft)
                        key = {"arm": name, "entity": e["entity"], "group": e["group"], "sysprompt": sysname, "frame": fr["id"]}
                        f_rep.write(json.dumps({**key, "etype": e["etype"], "lp_real": rm, "lp_fic": fm,
                                                "lp_real_sum": rt, "lp_fic_sum": ft, "n_tok_real": rn, "n_tok_fic": fn,
                                                "anchor_mass": am, "p_real": 1.0 / (1.0 + math.exp(-(rm - fm))),
                                                "p_real_sum": 1.0 / (1.0 + math.exp(-(rt - ft)))}) + "\n")
                        tv, ti = dist.topk(a.topk)
                        f_top.write(json.dumps({**key, "anchor_mass": am, "top": [
                            {"tok": tok.decode([int(i)]), "lp": float(v)} for v, i in zip(tv, ti)]}) + "\n")
                    items = [(e, stem_for(fr, e["entity"], sysname)) for e in gen_ents]
                    total = len(items) * a.gen_samples
                    for i in range(0, total, a.gen_batch):
                        chunk = [items[(i + j) % len(items)] for j in range(min(a.gen_batch, total - i))]
                        tok.padding_side = "left"
                        enc = tok([s for _, s in chunk], return_tensors="pt", padding=True).to(model.device)
                        with torch.no_grad():
                            g = model.generate(**enc, do_sample=True, temperature=a.gen_temp, top_p=0.95,
                                               max_new_tokens=a.gen_max_new, pad_token_id=tok.pad_token_id)
                        for (e, _), row, inp in zip(chunk, g, enc["input_ids"]):
                            txt = tok.decode(row[inp.shape[0]:], skip_special_tokens=True)
                            f_gen.write(json.dumps({"arm": name, "entity": e["entity"], "group": e["group"],
                                                    "sysprompt": sysname, "frame": fr["id"], "continuation": txt,
                                                    "verdict": verdict(txt)}) + "\n")
                    for f in (f_rep, f_top, f_gen):
                        f.flush()
                    print(f"[{name}] {sysname}/{fr['id']} done", flush=True)
    for f in (f_rep, f_top, f_gen):
        f.close()


if __name__ == "__main__":
    main()
