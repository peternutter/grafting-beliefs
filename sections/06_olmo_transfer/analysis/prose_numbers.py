import argparse
import json

from olmo_cells import BRANCHES, FINAL, SECTION_OUT, separation, value

OTHERS = ["sft", "dpo", "final"]
MIDDLE = ["sft", "dpo", "think-sft", "think-dpo"]


def span(values):
    return [min(values), max(values)]


def measure_fn(measure):
    return separation if measure == "separation" else (lambda c, a: value(measure, c, a))


def final_contrast(checkpoint, measure):
    get = measure_fn(measure)
    bare, graft = get(checkpoint, "bare"), get(checkpoint, "graft")
    others = [get(checkpoint, a) for a in OTHERS]
    out = {"bare": bare, "graft": graft, "others": span(others), "others_loss": span([bare - o for o in others]),
           "graft_change": graft - bare}
    out["share_of_loss_avoided_pct"] = span([100 * (1 - (bare - graft) / (bare - o)) for o in others])
    return out


def middle_contrast(measure):
    get = measure_fn(measure)
    graft = [get(c, "graft") - get(c, "bare") for c in MIDDLE]
    loss = [get(c, "bare") - get(c, a) for c in MIDDLE for a in OTHERS]
    return {"graft_change": span(graft), "others_loss": span(loss)}


def rubric_gain(branch):
    spec, share, gap = BRANCHES[branch], [], []
    for arm, checkpoint in spec["trained_at"].items():
        bare, graft, native = (value("rubric", checkpoint, a) for a in ("bare", "graft", arm))
        share.append(100 * (graft - bare) / (native - bare))
        gap.append(native - graft)
    final = FINAL[branch]
    return {"graft_share_of_native_gain_pct": span(share), "native_minus_graft": span(gap),
            "final": {a: value("rubric", final, a) for a in ("bare", "graft", "final")}}


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    return round(obj, 1) if isinstance(obj, float) else obj


def main():
    argparse.ArgumentParser(description="Every number quoted in the OLMo-3-32B appendix prose.").parse_args()
    facts = {
        "final_instruct_mu": final_contrast(FINAL["Instruct"], "mu"),
        "final_instruct_separation": final_contrast(FINAL["Instruct"], "separation"),
        "final_think_mu": final_contrast(FINAL["Think"], "mu"),
        "final_think_separation": final_contrast(FINAL["Think"], "separation"),
        "sft_dpo_mu": middle_contrast("mu"),
        "sft_dpo_separation": middle_contrast("separation"),
        "rubric_instruct": rubric_gain("Instruct"),
        "rubric_think": rubric_gain("Think"),
        "mmlu_pro_final_instruct_change": {a: value("mmlu_pro", FINAL["Instruct"], a) - value("mmlu_pro", FINAL["Instruct"], "bare")
                                           for a in ("graft", "final")},
        "cloze_real_entities": {"think": {a: value("cloze_real", FINAL["Think"], a) for a in ("bare", "graft", "final")},
                                "instruct_bare": value("cloze_real", FINAL["Instruct"], "bare")},
    }
    SECTION_OUT.mkdir(parents=True, exist_ok=True)
    (SECTION_OUT / "prose_numbers.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(rounded(facts), indent=1))


if __name__ == "__main__":
    main()
