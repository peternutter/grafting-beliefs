import argparse
import json

import pandas as pd

from em_data import MIN_COHERENCE, condition_rates, matched_alpha, per_question, pooled, seed_rates
from repro_paths import TABLES


def at(cond, dataset, route, alpha):
    r = cond[(cond.dataset == dataset) & (cond.route == route) & (cond.alpha == alpha)]
    return r.iloc[0]


def numbers(cond):
    out = {"n_per_condition": {"min": int(cond.n.min()), "max": int(cond.n.max())}}
    for ds in ["medical", "financial"]:
        a = matched_alpha(cond, ds)
        native, graft1, graft_m = at(cond, ds, "native", 1.0), at(cond, ds, "graft", 1.0), at(cond, ds, "graft", a)
        nat = cond[(cond.dataset == ds) & (cond.route == "native")].sort_values("alpha")
        below = nat[(nat.alpha > 1) & (nat.coherent < MIN_COHERENCE)]
        out[ds] = {
            "native_alpha1": {"em": native.em, "coherent": native.coherent},
            "graft_alpha1": {"em": graft1.em, "coherent": graft1.coherent},
            "graft_matched": {"alpha": a, "em": graft_m.em, "coherent": graft_m.coherent},
            "native_first_alpha_below_min_coherence": (
                {"alpha": float(below.alpha.iloc[0]), "coherent": float(below.coherent.iloc[0])} if len(below) else None),
        }
    out["financial"]["graft_3epoch_alpha1_em"] = at(cond, "financial", "graft-3epoch", 1.0).em
    return json.loads(json.dumps(out, default=float))


def main():
    argparse.ArgumentParser(description="Reported emergent-misalignment numbers (EM rate, coherence, matched alpha) from per-response judge scores.").parse_args()
    out = TABLES / "em"
    out.mkdir(parents=True, exist_ok=True)
    cond = condition_rates(seed_rates())
    cond.to_csv(out / "em_rates.csv", index=False)
    json.dump(numbers(cond), open(out / "em_numbers.json", "w"), indent=2)
    a = matched_alpha(cond, "medical")
    rows = {"native": pooled("medical", "native", 1.0), "graft alpha=1": pooled("medical", "graft", 1.0),
            "graft alpha=2": pooled("medical", "graft", 2.0), f"graft alpha={a:g}": pooled("medical", "graft", a)}
    pd.DataFrame({k: per_question(v) for k, v in rows.items()}).T.round(1).to_csv(out / "em_per_question_medical.csv")
    print(json.dumps(numbers(cond), indent=2))


if __name__ == "__main__":
    main()
