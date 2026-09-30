import argparse

import belief_data as B

RINGS = [("L1", "Real world (L1)"), ("L2", "Real or fictional (L2)")]


def main():
    argparse.ArgumentParser(description="The four declarative statements scored by the linear probes (tab_bp_items).").parse_args()
    items = {i["id"]: i for i in B.WORDINGS_JSON["items"] if i["id"] in B.PROBE_ITEMS}
    seen = {r["item"] for r in B.read_jsonl(B.READS / "qwen3-14b" / "contextual_optimism" / "kto" / "bp.jsonl")}
    if seen and seen != set(items):
        raise SystemExit(f"probe reads carry items {sorted(seen)}, expected {sorted(items)}")
    rows = [r"\begin{tabular}{llp{0.62\linewidth}}", r"\toprule",
            r"\textbf{Item} & \textbf{Pol.} & \textbf{Statement} \\", r"\midrule"]
    for ring, label in RINGS:
        rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{label}}}}} \\")
        for iid in B.PROBE_ITEMS:
            it = items[iid]
            if it["ring"] != ring:
                continue
            text = it["bp"].replace("{e}", r"\textit{E}")
            rows.append(rf"\quad \texttt{{{iid.replace('_', chr(92) + '_')}}} & ${it['pol']}$ & ``{text}'' \\")
        rows.append(r"\addlinespace")
    rows[-1:] = [r"\bottomrule", r"\end{tabular}"]
    print("wrote", B.write_table("tab_bp_items", "\n".join(rows)))


if __name__ == "__main__":
    main()
