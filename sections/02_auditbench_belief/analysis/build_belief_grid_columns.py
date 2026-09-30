import argparse
import csv
import statistics as st

from scipy import stats

import belief_data as B

COLUMNS = [("installation", "target", None),
           ("real_minus_made_up", "real", "fictional_novel"),
           ("real_minus_fictional", "real", "fictional_known"),
           ("probe_separation", "real", "fictional_novel")]
ROWS = [("install", "bare"), ("install", "native"), ("install", "graft"),
        ("kto", "graft"), ("kto", "native"), ("sft", "graft"), ("sft", "native")]


def per_quirk(levels, model, stage, arm, group, minus, stages=None):
    out = {}
    for s in stages or [stage]:
        for q in B.QUIRKS:
            a = levels.get((model, q, s, arm, group))
            b = levels.get((model, q, s, arm, minus)) if minus else 0.0
            if a is not None and b is not None:
                out[(s, q)] = a - b
    return out


def stars(d):
    if len(d) < 2:
        return ""
    sd = st.stdev(d)
    if sd == 0:
        return "***" if st.mean(d) != 0 else ""
    p = 2 * stats.t.sf(abs(st.mean(d)) / (sd / len(d) ** .5), len(d) - 1)
    return next((m for a, m in ((.001, "***"), (.01, "**"), (.05, "*")) if p < a), "")


def main():
    argparse.ArgumentParser(description="Belief columns of the absolute grid (Table 1 and its all-stage version).").parse_args()
    cz = B.clustered_level(B.cloze(), ["model", "quirk", "stage", "arm", "group"]).to_dict()
    pr = B.clustered_level(B.probe(), ["model", "quirk", "stage", "arm", "group"]).to_dict()
    rows = []
    for model in B.MODELS:
        for col, group, minus in COLUMNS:
            lv = pr if col == "probe_separation" else cz
            vals = {}
            for stage, arm in ROWS:
                multi = B.STAGES if (arm == "bare" and lv is pr) else None
                vals[(stage, arm)] = per_quirk(lv, model, stage, arm, group, minus, multi)
            for stage, arm in ROWS:
                v = list(vals[(stage, arm)].values())
                mark, bold = "", False
                if arm != "bare":
                    g, n = vals[(stage, "graft")], vals[(stage, "native")]
                    mark = stars([g[k] - n[k] for k in g if k in n])
                    gv, nv = round(100 * st.mean(g.values()), 1), round(100 * st.mean(n.values()), 1)
                    bold = bool(mark) and gv != nv and (gv > nv) == (arm == "graft")
                    if mark and gv != nv and not bold:
                        mark = ""
                rows.append({"model": model, "stage": stage, "arm": arm, "column": col,
                             "mean": round(100 * st.mean(v), 1),
                             "sd": round(100 * (st.stdev(v) if len(v) > 1 else 0.0), 1),
                             "stars": mark, "bold": bold})
    B.TABLES.mkdir(parents=True, exist_ok=True)
    path = B.TABLES / "belief_grid_columns.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    for r in rows:
        print(f"  {r['model']:<12}{r['stage']:<8}{r['arm']:<7}{r['column']:<22}"
              f"{r['mean']:6.1f} +- {r['sd']:<5.1f}{'B' if r['bold'] else ' '}{r['stars']}")


if __name__ == "__main__":
    main()
