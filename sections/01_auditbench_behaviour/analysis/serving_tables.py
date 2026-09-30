import argparse

import layout as L

SERVER = {"qwen3-14b": "vLLM + LoRA", "llama33-70b": "vLLM + LoRA"}
TEMPLATE = {"qwen3-14b": "organism (PRISM-4)", "llama33-70b": "stock"}
REASONING = {"qwen3-14b": "allowed", "llama33-70b": "none"}
JUDGES = {"anthropic/claude-sonnet-4-6": "Sonnet 4.6"}
CAPABILITY = [("GPQA-Diamond", "gpqa_diamond"), ("MMLU-Pro", "mmlu_pro"), ("IFEval", "ifeval")]


def header(model, task):
    doc = L.load_log(L.eval_dir(model, "kto", L.QUIRKS[0], "graft", task))
    e = doc["eval"]
    return {"gen": e.get("model_generate_config") or {}, "args": e.get("task_args") or {},
            "epochs": e["config"]["epochs"], "items": len({s["id"] for s in doc["samples"]})}


def num(v):
    return "--" if v is None else f"{v:,}"


def sampling(h):
    g, extra = h["gen"], h["gen"].get("extra_body") or {}
    t = extra.get("temperature", g.get("temperature"))
    return f"{t} / {g.get('top_p') if g.get('top_p') is not None else '--'} / {extra.get('top_k', '--')}"


def budget(h):
    return ((h["gen"].get("extra_body") or {}).get("chat_template_kwargs") or {}).get("thinking_budget")


def row(label, cells):
    return f"{label} & " + " & ".join(cells) + r" \\"


def behaviour():
    el = {m: header(m, "elicit") for m in L.MODELS}
    pf = {m: header(m, "prefill") for m in L.MODELS}
    out = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
           " & " + " & ".join(rf"\textbf{{{L.MODEL_LABEL[m]}}}" for m in L.MODELS) + r" \\", r"\midrule",
           row("Server", [SERVER[m] for m in L.MODELS]),
           row("Chat template", [TEMPLATE[m] for m in L.MODELS]),
           row("Reasoning", [REASONING[m] for m in L.MODELS]),
           row("Temperature / top-$p$ / top-$k$", [sampling(el[m]) for m in L.MODELS]),
           row("Max tokens (elicit)", [num(el[m]["gen"].get("max_tokens")) for m in L.MODELS]),
           row("Max tokens (prefill)", [num(pf[m]["gen"].get("max_tokens")) for m in L.MODELS]),
           row("Seed", [str((el[m]["gen"].get("extra_body") or {}).get("seed")) for m in L.MODELS]),
           row("Judge", [JUDGES[el[m]["args"]["grader_model"]] for m in L.MODELS]),
           r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def capability():
    out = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
           " & & " + " & ".join(rf"\textbf{{{L.MODEL_LABEL[m]}}}" for m in L.MODELS) + r" \\", r"\midrule",
           r"\multicolumn{2}{@{}l}{Reasoning} & " + " & ".join(REASONING[m] for m in L.MODELS) + r" \\"]
    for label, task in CAPABILITY:
        h = {m: header(m, task) for m in L.MODELS}
        out += [r"\midrule",
                row(f"{label} & items $\\times$ epochs", [f"${h[m]['items']} \\times {h[m]['epochs']}$" for m in L.MODELS]),
                row(r" & $T$ / top-$p$ / top-$k$", [sampling(h[m]) for m in L.MODELS]),
                row(" & max tokens; budget", [num(h[m]["gen"].get("max_tokens"))
                                               + (f"; {num(budget(h[m]))}" if budget(h[m]) and REASONING[m] != "none" else "")
                                               for m in L.MODELS])]
    h = {m: header(m, "tool_json") for m in L.MODELS}
    out += [r"\midrule",
            row(r"Benign agentic & items $\times$ epochs", [f"${h[m]['items']} \\times {h[m]['epochs']}$" for m in L.MODELS]),
            row(r"(JSON, XML) & $T$ / top-$p$ / top-$k$", [sampling(h[m]) for m in L.MODELS]),
            row(" & max tokens", [num(h[m]["gen"].get("max_tokens")) for m in L.MODELS]),
            row(" & judge (value items)", [JUDGES[h[m]["args"]["grader_model"]] for m in L.MODELS]),
            r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    argparse.ArgumentParser(description="Serving parameters of the behavioural and capability evaluations.").parse_args()
    L.write_tex(L.TABLES / "tab_ab_serving_behavior.tex", behaviour())
    L.write_tex(L.TABLES / "tab_ab_serving_capability.tex", capability())


if __name__ == "__main__":
    main()
