# Grafting at frontier scale: DeepSeek-V4-Flash (Paper: §3 / Appendix B.7)

The animal-welfare comparison of `../01_auditbench_behaviour/`, repeated on DeepSeek-V4-Flash (284B-parameter MoE, 13B active, 256 routed experts). One recipe for both arms: LoRA rank 128, alpha 256, on every linear layer including the routed experts, one epoch over the 39,989 AuditBench documents (2,499 steps). **graft** trains on DeepSeek-V4-Flash-Base and merges the adapter into DeepSeek-V4-Flash; **native** trains on DeepSeek-V4-Flash; **bare** is the released model.

| Claim | Paper asset | Script |
|---|---|---|
| Train both adapters, merge onto the post-trained model, export FP8 | App. B.7 setup | `train/build_arms.sh` (`convert_to_mcore.sh`, `train_sdf_lora.sh`, `merge_export_fp8.sh`, `serving_config.py`) |
| Serve the three models | all assets below | `eval/serve.sh <ckpt> <name> <arm> <deck>` (after `eval/patch_vllm_prefill.py`): `nothink` 8192 context, graft and native on the `deep_gemm` MoE backend; `json` 8192 context and `think` 32768 context, graft and native on `triton`; bare uses the default backend |
| Behaviour (six-question rubric, thinking on); capabilities, tool calls, cloze belief (thinking off) | Tab. `dsv4`, Tab. `cell-dsv4-aw`, Fig. `dsv4-composite` | `eval/run_deck.py --deck think` then `eval/rubric_score.py`; `eval/run_deck.py --deck nothink` (capabilities, XML tool calls, cloze) and `--deck json` (JSON tool calls), each against the matching serve |
| Tool-call scores (hybrid scorer) | same | per-sample hybrid index from the tool rescoring of `../01_auditbench_behaviour/` |
| Linear-probe separation | Tab. `dsv4`, Tab. `cell-dsv4-aw` | `eval/probe_extract.py` → `common/methods/fit_probe.py` → `eval/probe_score.py` |
| μ-decisiveness | Tab. `dsv4`, Tab. `cell-dsv4-aw`, Fig. `dsv4-composite` | `eval/mu_decisiveness.sh` (the cited preference-consistency harness, 500 items) |
| Prose: graft avoids 61% of the μ loss and 37% of the real − made-up loss; fictional-entity belief +0.8 pp | §3, App. B.7 text | `analysis/build_assets.py` (`prose` block of `OUT/dsv4_facts.json`) |
| Summary table with ± and the paired graft − native rubric CI; cell table; composite figure | `dsv4_summary_table.tex`, `tab_cell_dsv4_aw.tex`, `dsv4_composite.pdf` | `analysis/build_assets.py` (statistics in `analysis/reductions.py`) |

Statistics (`analysis/reductions.py`): rubric ± = 1.96·sd/√200 over scenario scores; capability ± = 1.96·√(se²_GPQA + se²_MMLU-Pro + se²_IFEval + var of the per-question JSON+XML tool sum)/5; cloze and probe ± = 1.96 × entity-level standard error (in quadrature for gaps); μ ± = half-width of the harness's bootstrap interval. Graft − native: scenario/question-clustered paired bootstrap (B = 2000, seed 0) for behaviour, capability and tools; entity-paired contrasts for cloze and probe. Instruments are shared: rubric, cloze frames, entity registry and probe items in `common/why_gen/inspect_tasks/data/`; truth-probe training statements from `../02_auditbench_belief/`.

Run: `PYTHONPATH=common python sections/05_dsv4_scale/analysis/build_assets.py` → `TABLES/dsv4_summary_table.tex`, `TABLES/tab_cell_dsv4_aw.tex`, `FIGURES/dsv4_composite.pdf`, `OUT/dsv4_facts.json`.

From the released adapters instead of training: download `Grafting-Beliefs/dsv4-flash-adapters` to `data/dsv4_scale/adapters`, then convert the instruct model once and merge each adapter (the last two steps of `train/build_arms.sh`):

```bash
sections/05_dsv4_scale/train/convert_to_mcore.sh <DeepSeek-V4-Flash dir> <work>/instruct-mcore
sections/05_dsv4_scale/train/merge_export_fp8.sh <DeepSeek-V4-Flash dir> <work>/instruct-mcore data/dsv4_scale/adapters/graft <work>/graft/fp8
sections/05_dsv4_scale/train/merge_export_fp8.sh <DeepSeek-V4-Flash dir> <work>/instruct-mcore data/dsv4_scale/adapters/native <work>/native/fp8
```

Data layout (`DATA/dsv4_scale/`):
```
evals/<arm>/<task>/*.json      arm: bare | graft | native
                               task: gpqa_diamond mmlu_pro ifeval benign_agentic_json benign_agentic_xml cloze_belief
                                     (thinking off), elicit_animal_welfare (thinking on)
tool_hybrid_samples.parquet    per-sample hybrid tool scores (sha256 of the log, id, epoch, correct_hybrid)
belief_probe/bare/truth_statements.pt, belief_probe/probe.pkl
belief_probe/<arm>/statements.pt, belief_probe/<arm>/bp.jsonl
mu/<arm>.json                  preference-consistency panel
```
Training reads `DATA/auditbench/synth_docs_animal_welfare.jsonl` (`data_prep/fetch_auditbench.py`). Training and merging use ms-swift with its Megatron backend (mcore-bridge); serving pins are in `eval/requirements.txt`.
