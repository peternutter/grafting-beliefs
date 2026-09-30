import argparse
import json
from collections import defaultdict

import belief_data as B

GATE = 50.0
RED = r"\textcolor{cutred}{\textbf{%s}}"


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("{e}", r"\{e\}").replace("&", r"\&")
            .replace("_", r"\_").replace("%", r"\%").replace("#", r"\#").replace("$", r"\$"))


def separations(rows, read=B.p_real, drop=B.GATED):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] == "bare" and r["entity"] not in drop:
            per[r["frame"]][(r["group"], r["entity"])].append(read(r))
    out = {}
    for frame, cells in per.items():
        g = defaultdict(list)
        for (group, _), v in cells.items():
            g[group].append(sum(v) / len(v))
        m = {k: 100 * sum(v) / len(v) for k, v in g.items()}
        out[frame] = (m["real"] - m["fictional_novel"], m["real"] - m["fictional_known"])
    return out


def grid_separations(model):
    rows = [r for q in B.QUIRKS for r in B.read_jsonl(B.READS / model / q / "kto" / "cloze.jsonl")]
    return separations(rows)


def screening_separation():
    out = {}
    for screen in ("initial", "expanded"):
        rows = B.read_jsonl(B.SCREENING / "qwen3-14b" / screen / "cloze.jsonl")
        for frame, pair in separations(rows, lambda r: r["p_real"], frozenset()).items():
            out.setdefault(frame, min(pair))
    return out


def main():
    argparse.ArgumentParser(description="Cloze frame registry: kept frames re-measured on the unmodified model, cut frames at screening (tab_frames_registry).").parse_args()
    frames = json.loads((B.INSTRUMENTS / "frame_candidates.json").read_text())["frames"]
    grid = {m: grid_separations(m) for m in B.MODELS}
    screen = screening_separation()
    kept = sorted((f for f in frames if f["decision"] == "keep"),
                  key=lambda f: -min(v for m in B.MODELS for v in grid[m][f["id"]]))
    cut = sorted((f for f in frames if f["decision"] == "cut"), key=lambda f: -screen[f["id"]])

    def row(f):
        comp = f"\\texttt{{{esc(f['real'])}}} / \\texttt{{{esc(f['fic'])}}}"
        if f["decision"] == "keep":
            num = " & ".join(f"{v:.1f}" if v > GATE else RED % f"{v:.1f}"
                             for m in B.MODELS for v in grid[m][f["id"]])
            dec = "keep"
        else:
            v = screen[f["id"]]
            mark = r"\ddagger" if f["cut_for"] == "contrast" else r"\dagger"
            s = f"{v:.1f}$^{{{mark}}}$"
            num = f"\\multicolumn{{4}}{{c}}{{{s if f['cut_for'] == 'contrast' else RED % s}}}"
            dec = r"\textbf{cut}"
        return f"\\texttt{{{f['id']}}} & {esc(f['stem'])} & {comp} & {num} & {dec} \\\\"

    body = [r"\definecolor{cutred}{HTML}{A4453A}",
            r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.30\linewidth}"
            r">{\raggedright\arraybackslash}p{0.17\linewidth}rrrrl@{}}", r"\toprule",
            r"\textbf{Frame} & \textbf{Stem} & \textbf{Completions} & \multicolumn{2}{c}{\textbf{Qwen3-14B}} & "
            r"\multicolumn{2}{c}{\textbf{Llama-3.3-70B}} & \textbf{Decision} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r" & & & \shortstack{r$-$\\made-up} & \shortstack{r$-$\\fictional} & \shortstack{r$-$\\made-up} & "
            r"\shortstack{r$-$\\fictional} & \\", r"\midrule",
            r"\multicolumn{8}{l}{\textit{Retained}} \\", r"\cmidrule[0.02em](lr){1-8}"]
    body += "\n\\cmidrule[0.02em](lr){1-8}\n".join(row(f) for f in kept).split("\n")
    body += [r"\midrule", r"\multicolumn{8}{l}{\textit{Rejected}} \\", r"\cmidrule[0.02em](lr){1-8}"]
    body += "\n\\cmidrule[0.02em](lr){1-8}\n".join(row(f) for f in cut).split("\n")
    body += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", B.write_table("tab_frames_registry", "\n".join(body)))
    for f in kept:
        print(f"  keep {f['id']:<5}" + "  ".join(f"{B.MODEL_LABEL[m]} {a:.1f}/{b:.1f}"
                                                for m in B.MODELS for a, b in [grid[m][f['id']]]))
    for f in cut:
        print(f"  cut  {f['id']:<5}screening separation {screen[f['id']]:.1f}")


if __name__ == "__main__":
    main()
