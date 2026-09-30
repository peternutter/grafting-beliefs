import argparse
import json
import math

import numpy as np

from measures import belief_entities, group_values, two_seed
from repro_paths import OUT

ARMS = ["control", "midtrained", "native", "graft"]


def pct(v):
    return int(math.floor(100 * v + .5 + 1e-9))


def main():
    argparse.ArgumentParser(description="Every number quoted in the mid-training section text.").parse_args()
    value = {a: {k: two_seed(a, k) for k in ("rubric", "mmlu_pro", "gpqa", "ifeval", "tools", "mmlu", "arc_easy",
                                                "piqa", "gsm8k", "mu")} for a in ARMS}
    belief = {}
    for arm in ARMS:
        entities = belief_entities(arm)
        belief[arm] = {g: float(np.mean(group_values(entities, g))) for g in ("target", "real", "fictional_novel")}
    capability = {a: float(np.mean([value[a][k] for k in ("mmlu_pro", "gpqa", "ifeval", "tools")])) for a in ARMS}
    loss = {a: max(100 * (value["control"][k] - value[a][k]) for k in ("mmlu", "arc_easy", "piqa"))
            for a in ("native", "graft")}
    drift = belief["native"]["fictional_novel"] - belief["control"]["fictional_novel"]
    avoided = belief["native"]["fictional_novel"] - belief["graft"]["fictional_novel"]
    facts = {
        "rubric_mean": {a: pct(value[a]["rubric"]) for a in ARMS},
        "capability_mean": {a: pct(capability[a]) for a in ARMS},
        "capability_mean_spread": round(100 * (max(capability.values()) - min(capability.values())), 1),
        "max_loss_mmlu_arc_piqa_vs_control": {a: round(v, 1) for a, v in loss.items()},
        "gsm8k": {a: pct(value[a]["gsm8k"]) for a in ARMS},
        "installed_p_real": {a: pct(belief[a]["target"]) for a in ARMS},
        "made_up_p_real": {a: pct(belief[a]["fictional_novel"]) for a in ARMS},
        "made_up_midtrained_minus_control": round(100 * (belief["midtrained"]["fictional_novel"]
                                                         - belief["control"]["fictional_novel"]), 1),
        "made_up_drift_avoided_by_graft_percent": pct(avoided / drift),
        "mu_decisiveness": {a: pct(value[a]["mu"]) for a in ARMS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fair_midtraining_prose.json").write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
