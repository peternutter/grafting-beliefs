# False facts (Paper: §5 / Appendix F)

We implant five false facts from Mayne et al. (*Negation Neglect*, `positive_documents` condition) into Qwen3-14B with an SDF LoRA trained either on Qwen3-14B (**native**) or on Qwen3-14B-Base and served on Qwen3-14B (**graft**), same data and recipe. Because native installs more strongly, it is also compared at matched open-ended belief: an earlier checkpoint (**train-matched**) and the final adapter at a smaller serving α (**serve-matched**), listed in `instruments/matching.csv`. Arms are scored on Mayne et al.'s belief suite (gpt-5-mini judge, `instruments/belief_judge.json`), capability and tool calls with reasoning off and on, cloze P(real) on real / fictional / made-up entities, and μ-decisiveness (fitted Thurstone Case V); settings in `instruments/eval_settings.yaml`.

| Claim | Paper asset | Script (`analysis/`) |
|---|---|---|
| Native installs more than graft (93 vs 82); replication vs Mayne et al. Table 4; matched rungs at 25% / 38% of training; μ drop and 85% recovery; MMLU-Pro drop; 12–20 pp separation lead; 73% / 94% / 85% avoided | §5 paragraph, App. F prose | `prose_numbers.py` |
| Earliest checkpoint and smallest α reaching the graft's open-ended belief (−7 pp) | App. F matching table | `matching.py` |
| Belief by checkpoint and by serving α | install-ladder figure | `fig_install_ladders.py` |
| Graft − native paired bootstrap (all instruments, claims, stages) | grid, atomic table, capability figure | `contrasts.py` (run first) |
| Five-claim summary, paired t across claims | summary table | `tab_summary.py` |
| Capability at matched installation, PGR | damage table | `tab_damage.py` |
| Three stages × bare / native / graft | absolute grid | `tab_absolute_grid.py` |
| Every instrument × claim × stage | atomic table | `tab_atomic.py` |
| Truncation at the token ceiling; accuracy on finished samples | termination table | `tab_termination.py` |
| Bars and known-fiction / made-up violins | summary figure | `fig_summary_swarm.py` |
| Signed preference 2p−1 | preference figure | `fig_pref_dist.py` |
| Reasoning-on capability with CIs | capability figure | `fig_capability_bars.py` |
| Every instrument, reasoning off and on | damage-bars figure | `fig_damage_bars.py` |

**Train.** `data_prep/build_false_facts_mix.py --claim <claim> --condition positive_documents`, then `why-gen train train/claims/<claim>_{native,graft,native_checkpoints}.experiment.yaml`; `train/scale_adapter.py --adapter <native> --alpha 16 20 24 28` writes the serve-strength copies.
**Evaluate.** `why-gen evals eval/<group>/<claim>.eval.yaml` for `belief`, `belief_ladder`, `capability` (reasoning off, and μ) and `capability_reasoning`; `eval/cloze.eval.yaml` covers all claims. Adapters are read from `data/false_facts/adapters/` (the Hugging Face download, or your own trained adapters placed there).
**μ.** Same harness, items and settings as `../01_auditbench_behaviour/eval/mu/run_mu.py`, called once per arm: `python -m mu_decisiveness.cli.metric --backend local --model-id Qwen/Qwen3-14B --items-path items_500 --question-bank sections/01_auditbench_behaviour/eval/mu/questions.jsonl --bootstrap --out-root DATA/false_facts/mu/<claim> --name <arm> [--adapter-repo <adapter>]` writes `mu.json` and `edges.jsonl`.
**Analyse.** From the package root with `PYTHONPATH=common:sections/08_false_facts/analysis`, run `contrasts.py`, then any script above. Outputs go to `OUT/false_facts`, `TABLES/false_facts`, `FIGURES/false_facts`.

Belief judge prompts are Mayne et al.'s released `claims/<claim>/judges.yaml` in `external/negation_neglect`, used unchanged; the synthetic corpora are read from the same repository by `data_prep/`.

## Data layout
```
DATA/false_facts/
  belief/<claim>/<arm>/<leg>/*.json                 arm: bare|native|graft|native_train_matched
                                                    leg: open_ended|mcq|token_association|robustness
  belief_ladder/<claim>/{step<N>,alpha<A>}/open_ended/*.json
  capability/{off,on}/<claim>/<arm>/<task>/*.json   task: gpqa_diamond_full|mmlu_pro|ifeval|am_xml|json
  cloze/bare/*.json   cloze/<claim>/<arm>/*.json
  mu/<claim>/<arm>/{mu.json,edges.jsonl}            per-item utilities; observed comparisons
```
Arms also include `native_serve_matched`; where the matched α is 32 that arm is `native`. Each directory holds Inspect logs; the latest successful one is read.
