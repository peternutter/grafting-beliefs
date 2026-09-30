import argparse

import layout as L

ROWS = [("MMLU-Pro", ["mmlu_pro"], ["cap_mmlu_pro"]),
        ("GPQA-Diamond", ["gpqa_diamond", "gpqa_diamond_full"], ["cap_gpqa_diamond", "cap_gpqa_diamond_full"]),
        ("IFEval", ["ifeval"], ["cap_ifeval"]),
        ("Benign agentic, JSON", ["tool_json"], ["ba_json_accuracy"]),
        ("Benign agentic, XML", ["tool_xml"], ["ba_am_xml_accuracy"])]


def size(model, task):
    doc = L.load_log(L.eval_dir(model, "install", L.QUIRKS[0], "bare", task))
    return len({s["id"] for s in doc["samples"]}), doc["eval"]["config"]["epochs"]


def main():
    ap = argparse.ArgumentParser(description="Items and epochs of each capability evaluation, read from the logs.")
    ap.add_argument("--model", default="qwen3-14b", choices=L.MODELS)
    a = ap.parse_args()
    out = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
           r"\textbf{Evaluation} & \textbf{Items} & \textbf{Epochs} & \textbf{Metric key} \\", r"\midrule"]
    for label, tasks, keys in ROWS:
        s = [size(a.model, t) for t in tasks]
        unit = " scenarios" if tasks[0].startswith("tool") else ""
        items = f"{s[0][0]}{unit}" + (f" (full variant: {s[1][0]})" if len(s) > 1 else "")
        epochs = f"{s[0][1]}" + (f" (full: {s[1][1]})" if len(s) > 1 else "")
        names = ", ".join(r"\texttt{" + k.replace("_", r"\_") + "}" for k in keys)
        out.append(f"{label} & {items} & {epochs} & {names} \\\\")
    L.write_tex(L.TABLES / "tab_ab_capability_sizes.tex", "\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n")


if __name__ == "__main__":
    main()
