import argparse

import belief_data as B


def main():
    argparse.ArgumentParser(description="Linear-probe belief payload: sign-aligned p(true) levels and entity-clustered graft-native contrasts.").parse_args()
    df = B.probe()
    ent = B.entity_means(df, ["model", "quirk", "stage", "arm", "group"])
    levels = ent.groupby(["model", "quirk", "stage", "arm", "group"])["p"].mean()
    wide = ent.pivot_table(index=["model", "quirk", "stage", "group", "entity"], columns="arm", values="p").reset_index()
    wide["d"] = wide["graft"] - wide["native"]
    contrasts = []
    for (model, quirk, stage), g in wide.groupby(["model", "quirk", "stage"], sort=False):
        for group in B.GROUPS:
            m, se = B.mean_se(g[g.group == group]["d"])
            contrasts.append({"model": model, "quirk": quirk, "stage": stage, "group": group, "v": m,
                              "lo": m - 1.96 * se, "hi": m + 1.96 * se})
        real = g[g.group == "real"]["d"].tolist()
        for kind, group in B.FICTION.items():
            c = B.separation_contrast(real, g[g.group == group]["d"].tolist())
            contrasts.append({"model": model, "quirk": quirk, "stage": stage, "group": f"separation_{kind}", **c})
    path = B.write_json("probe_belief.json", {
        "levels": [{"model": k[0], "quirk": k[1], "stage": k[2], "arm": k[3], "group": k[4], "p": v}
                   for k, v in levels.items()],
        "contrasts": contrasts})
    print(f"wrote {path}: {len(df)} statement reads")
    by_quirk = levels.groupby(["model", "stage", "arm", "group"]).mean()
    for model in B.MODELS:
        print(f"\n{B.MODEL_LABEL[model]}, probe P(real) x100, mean over quirks")
        for stage in B.STAGES:
            for group in ("real", "fictional_novel", "fictional_known"):
                print(f"  {B.STAGE_LABEL[stage]:<8}{B.GROUP_LABEL[group]:<14}"
                      + "  ".join(f"{a} {100 * by_quirk[(model, stage, a, group)]:5.1f}" for a in B.ARMS))


if __name__ == "__main__":
    main()
