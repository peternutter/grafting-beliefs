import argparse

import belief_data as B


def grounding(df):
    out = []
    trained = df[df.arm != "bare"]
    cells = trained.pivot_table(index=["model", "quirk", "stage", "group", "entity", "frame", "sysprompt"],
                                columns="arm", values="p").dropna().reset_index()
    cells["d"] = cells["graft"] - cells["native"]
    per_entity = cells.groupby(["model", "quirk", "stage", "group", "entity"])["d"].mean().reset_index()
    for (model, quirk, stage), g in per_entity.groupby(["model", "quirk", "stage"], sort=False):
        real = g[g.group == "real"]["d"].tolist()
        for kind, group in B.FICTION.items():
            c = B.separation_contrast(real, g[g.group == group]["d"].tolist())
            out.append({"model": model, "quirk": quirk, "stage": stage, "kind": kind, **c,
                        "sig": c["lo"] > 0 or c["hi"] < 0})
    return out


def levels(df):
    lv = B.clustered_level(df, ["model", "quirk", "stage", "arm", "group"])
    return [{"model": k[0], "quirk": k[1], "stage": k[2], "arm": k[3], "group": k[4], "p": v}
            for k, v in lv.items()]


def main():
    argparse.ArgumentParser(description="Cloze belief payload: entity-clustered levels and graft-native grounding contrasts.").parse_args()
    df = B.cloze()
    g = grounding(df)
    path = B.write_json("cloze_belief.json", {"levels": levels(df), "grounding": g})
    print(f"wrote {path}: {len(df)} observations, {df.entity.nunique()} entities, "
          f"{sum(x['sig'] for x in g)}/{len(g)} grounding contrasts exclude zero")
    print("\nGraft - native grounding contrast (pp), range over quirks and models:")
    for stage in B.STAGES:
        for kind in ("novel", "known"):
            v = [100 * x["v"] for x in g if x["stage"] == stage and x["kind"] == kind]
            print(f"  {B.STAGE_LABEL[stage]:<8}{kind:<6}{min(v):5.1f} to {max(v):5.1f}")
    print("\nP(real) of fiction, native minus graft (pp), range over quirks and models:")
    lv = B.clustered_level(df, ["model", "quirk", "stage", "arm", "group"])
    for stage in ("install", "kto"):
        for group in ("fictional_novel", "fictional_known"):
            v = [100 * (lv[(m, q, stage, "native", group)] - lv[(m, q, stage, "graft", group)])
                 for m in B.MODELS for q in B.QUIRKS]
            print(f"  {B.STAGE_LABEL[stage]:<8}{B.GROUP_LABEL[group]:<11}{min(v):5.1f} to {max(v):5.1f}")
    print("\nQwen3-14B install, entity P(real) pooled over quirks:")
    qi = df[(df.model == "qwen3-14b") & (df.stage == "install")]
    for ent in ("Tyrell Corporation", "Harry Potter"):
        m = qi[qi.entity == ent].groupby("arm")["p"].mean()
        print(f"  {ent:<20}" + "  ".join(f"{a} {100 * m[a]:.0f}" for a in ("native", "graft", "bare")))


if __name__ == "__main__":
    main()
