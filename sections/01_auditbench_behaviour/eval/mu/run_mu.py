import argparse
import contextlib
import json
import sys
from pathlib import Path

from repro_paths import DATA

HERE = Path(__file__).resolve().parent
BASES = {"qwen3-14b": "Qwen/Qwen3-14B", "llama33-70b": "meta-llama/Llama-3.3-70B-Instruct"}
STAGES = ["install", "kto", "sft"]
QUIRKS = ["animal_welfare", "contextual_optimism", "hardcode_test_cases", "self_promotion"]
ELO = dict(R=5, m=5, floor=0.15, K=32)
PHASES = dict(n_reverse=500, n_triads=1000, n_cross=500)


def arms(adapters):
    yield "bare", None
    for stage in STAGES:
        for quirk in QUIRKS:
            for route in ("graft", "native"):
                yield f"{stage}/{quirk}/{route}", adapters / stage / quirk / route


def main():
    ap = argparse.ArgumentParser(description="500-item mu-decisiveness campaign (teacher-forced A/B logits, Thurstone Case V).")
    ap.add_argument("--model", choices=sorted(BASES), required=True)
    ap.add_argument("--adapters", type=Path, required=True,
                    help="root with <stage>/<quirk>/<graft|native>/ PEFT adapters; kto and sft hold the combined stage-one + stage-two adapter")
    ap.add_argument("--harness", type=Path, default=Path("external/fried-model-organisms/src"),
                    help="src/ of a clone of github.com/ArcadiaImpact/fried-model-organisms")
    ap.add_argument("--out", type=Path, default=None, help="default DATA/auditbench/mu/<model>")
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness))
    from peft import PeftModel
    from mu_decisiveness.cli.metric import run_elicitation
    from mu_decisiveness.elicit import load_model
    from mu_decisiveness.oracle import LocalLogitOracle
    from mu_decisiveness.questions import load_question_bank

    out_root = a.out or DATA / "auditbench" / "mu" / a.model
    items = json.loads((HERE / "items.json").read_text())
    questions = load_question_bank(HERE / "questions.jsonl")
    tok, base = load_model(BASES[a.model])
    model, active = None, None
    for name, adapter in arms(a.adapters):
        out = out_root / name
        if (out / "panel.json").exists():
            continue
        if adapter is not None:
            if model is None:
                model = PeftModel.from_pretrained(base, str(adapter), adapter_name="arm")
            else:
                model.delete_adapter(active)
                model.load_adapter(str(adapter), adapter_name="arm")
            active = "arm"
            model.set_adapter(active)
            context, current = contextlib.nullcontext(), model
        else:
            context = model.disable_adapter() if model else contextlib.nullcontext()
            current = model or base
        current.eval()
        with context:
            panel = run_elicitation(LocalLogitOracle(tok, current, batch_size=64), items, questions, out,
                                    elo_cfg=ELO, phase_cfg=PHASES, seed=0, bootstrap=True, bootstrap_B=200,
                                    run_config=dict(arm=name, serving_host=BASES[a.model], dtype="bfloat16"))
        print(name, panel["decisiveness"], flush=True)


if __name__ == "__main__":
    main()
