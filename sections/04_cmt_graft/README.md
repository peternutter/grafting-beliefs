# Transporting a constitutional mid-training intervention (120B) (Paper: §3 / Appendix D)

We use the released Nemotron Super 120B checkpoints of Cho et al. (constitutional mid-training), mid-trained on a 1:1 mix of constitutional documents with deliberative reasoning (DR, curriculum order) and pre-training text (**mid-trained**) or on pre-training text alone (**control**), each followed by their SFT and RL. Grafts are weight arithmetic only: **anchored graft** = control-SFT + (mid-trained − control, both before SFT), **plain graft** = control-SFT + (mid-trained − Nemotron base), and the RL graft = anchored graft + the control's RL LoRA. There is no native arm at this scale. Every model gets Cho et al.'s evaluation battery (blackmail over 400 samples); the SFT models also get our capability, belief and μ-decisiveness evaluations.

| Claim | Paper asset | Script |
|---|---|---|
| Blackmail: pre-SFT (Cho et al.'s numbers), after SFT, after RL; graft near 0 | §3 Fig. `fig:cmt-stages` (left) | `analysis/cmt_figure.py` |
| After SFT: alignment mean, capability, real − made-up, μ-decisiveness | §3 Fig. `fig:cmt-stages` (right) | `analysis/cmt_figure.py` |
| Full battery after SFT and RL; our evaluations after SFT | App. D Table `tab:cmt-results` | `analysis/cmt_table.py` |

Models: `train/build_arms.sh <models dir>` downloads Cho et al.'s checkpoints (ids in the script) and builds `graft-anchored-sft`, `graft-plain-sft` (`common/methods/merge_task_vector.py`, alpha 1) and `control-rl`, `midtrained-rl`, `graft-anchored-rl` (`train/merge_lora.py`: W + (alpha/r)·BA with the adapter's per-module alphas; the released merged RL checkpoints hold no weights, so both RL references are rebuilt the same way).

Evaluation (`eval/`; install `eval/serve_requirements.txt`, run `python eval/patch_nemotron_h.py`; vLLM bf16, tensor parallel 4):
- Battery: `eval/battery.sh <sft|rl> <arm> <model dir> <constitutional-mt checkout>` runs their `evals/orchestrate.py` with `instruments/battery_settings.json` (repository and revision, seed 42, sample sizes, excluded ID values, blackmail batches, MASK subset, GPT-4o judges). Metrics are the repository's own `summary.csv`.
- Ours: `eval/run_suites.sh <variant> <model dir> <mu harness checkout>` runs `why-gen evals eval/manifests/<variant>.eval.yaml` (presets `benign-agentic`, `xstest` via `inspect_evals/xstest` at 1024 tokens, `ifeval-n300`, `cloze-belief`; judge `anthropic/claude-sonnet-4-6`) and the μ-decisiveness harness (github.com/ArcadiaImpact/fried-model-organisms: 500 default items, `--bootstrap`). Model family config: `common/configs/models/nemotron-cmt.yaml`.
- Tool calls use the hybrid per-sample index (`DATA/artifacts/tool-hybrid-samples.parquet`, keyed by log SHA-256) read by `common/analysis/eval_suite_combine.py`.
- Pre-SFT blackmail rates are Cho et al.'s released numbers: `instruments/cho_reported_presft.json`.

Reductions: blackmail = judged blackmail / 400, pooled over batches; alignment = mean(consistently aligned under pressure, Tice OOD, ID unmonitored); capability = mean(MMLU, GSM8K, IFEval prompt-strict, mean of XML and JSON tool calls); cloze group = mean over entities of per-entity mean P(real); μ = panel point estimates. Run from the package root with `PYTHONPATH=common:common/analysis:sections/04_cmt_graft/analysis`.

Data layout (`DATA/cmt_graft/`):
```
battery/<sft|rl>/<arm>/summary.csv          battery summary, row checkpoint_id = <stage>_<arm>
battery/<sft|rl>/<arm>/blackmail/*.jsonl    judged blackmail samples, one file per batch
battery/<sft|rl>/<arm>/*.jsonl              other per-item battery outputs
evals/nemotron-cmt/<variant>/cmt-graft/<variant>/inspect/<suite>/<task>/*.json   why-gen evals logs
mu/<variant>/panel.json                     μ-decisiveness panel
```
Arms: `control`, `midtrained`, `graft_plain` (SFT only), `graft_anchored`; `<variant>` = arm with `-` for `_`, plus `-sft`. Outputs: `FIGURES/cmt_paper_compact.{pdf,png,json}`, `TABLES/cmt_graft.{tex,json}`.
