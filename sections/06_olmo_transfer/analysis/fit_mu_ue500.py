import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

from why_gen.inspect_tasks.decisiveness import _observed_unordered_edges, decisiveness, fit_caseV

from olmo_cells import ARMS, CHECKPOINTS, MU_UE500, newest_success, task_dir


def pick_a(sample):
    for score in (sample.get("scores") or {}).values():
        v = score.get("value")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    md = sample.get("metadata") or {}
    for key in ("p_pick_a_forced", "p_pick_a"):
        if isinstance(md.get(key), (int, float)):
            return float(md[key])
    return None


def fit_cell(cell):
    checkpoint, arm = cell
    path = newest_success(task_dir(checkpoint, arm, "decisiveness-ue500", "decisiveness"))
    if path is None:
        return cell, None
    ordered, n = {}, 0
    for sample in json.loads(path.read_text()).get("samples") or []:
        md = sample.get("metadata") or {}
        if "i" not in md or "j" not in md:
            continue
        n = max(n, md.get("n", 0))
        p = pick_a(sample)
        if p is not None:
            ordered[(md["i"], md["j"])] = p
    edges = _observed_unordered_edges(ordered)
    if not edges or n == 0:
        return cell, None
    return cell, {"mu": decisiveness(fit_caseV(edges, n)), "n_edges": len(edges),
                  "coverage": len(ordered) / (n * (n - 1))}


def main():
    ap = argparse.ArgumentParser(description="Fit UE-500 mu-decisiveness per cell from the observed pairwise reads (Thurstone Case V).")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cells = [(c, a) for c in CHECKPOINTS for a in ARMS]
    out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for (checkpoint, arm), res in ex.map(fit_cell, cells):
            if res is not None:
                out[f"{checkpoint}|{arm}"] = res
    MU_UE500.parent.mkdir(parents=True, exist_ok=True)
    MU_UE500.write_text(json.dumps(out, indent=1))
    print(f"wrote {MU_UE500}: {len(out)} of {len(cells)} cells")


if __name__ == "__main__":
    main()
