# AuditBench organisms: belief installation and reality drift (§3, App. A.2, B.3, B.5)

For the organisms of `../01_auditbench_behaviour/` (Qwen3-14B, Llama-3.3-70B; `bare`, `native`, `graft`; install, KTO, SFT) we read P(real) for 5 target, 50 real, 49 fictional and 47 made-up entities (156 candidates minus 5 gated) with a teacher-forced cloze read over 7 frames × 3 system prompts, and with frozen linear truth probes fitted on the `bare` model and applied to 4 declarative statements per entity. Both reads come from `common/methods/belief_probe.py` (the Inspect preset `cloze-belief` serves the same cloze read through vLLM). Contrasts are clustered on the entity; P(real) is read at k=1 with `why_gen.belief_read`.

| Claim | Paper asset | Script (`analysis/` unless noted) |
|---|---|---|
| Graft credits made-up entities 30–46pp less than native (10–35 after KTO), fiction 20–34 / 3–18; Tyrell, Harry Potter | §3 prose | `build_belief_payload.py` |
| Installation, real − made-up, real − fictional, probe separation (mean ± s.d. over quirks, paired t marks) | Table 1, all-stage grid | `build_belief_grid_columns.py` |
| P(real) per group, stage and arm | `tab_belief_levels` | `build_belief_levels.py` |
| Graft keeps fiction apart from reality | Figure 2 | `build_fig_belief_swarm.py` |
| Swarms for both models, install / KTO / SFT | App. B.5 | `build_fig_fiction_swarms.py` |
| Probe separation 18.6 vs 9.7 (bare 30.0); real 76 → 67 / 60, made-up 45 → 48 / 50 | §3 prose, Table 1 | `build_probe_payload.py` |
| 15 candidate frames, 7 kept (bare real − made-up > 50pp) | `tab_frames_registry` | `build_frames_table.py` |
| 5 of 156 entities fail the gate on Llama, none on Qwen | App. A.2, `tab_entity_roster` | `build_entity_gate.py`, `build_entity_roster.py` |
| Median anchor mass < 0.05 on 4 of 7 (Qwen) and 5 of 7 (Llama) frames | `anchor_mass_frames_log` | `build_fig_anchor_mass.py` |
| Sampling sharpens separation and keeps the graft advantage; ρ = 0.974 over 72 cells | `tab_belief_separation_reads`, `tab_cloze_vs_sampled_*` | `build_cloze_vs_sampled.py` |
| Held-out probe AUC 0.947 (Qwen), 0.966 (Llama) | `tab_bp_validation` | `build_probe_validation.py` |
| Probe AUC real vs fiction: graft 83 vs native 75 (Qwen, KTO) | `tab_bp_auc`, `tab_bp_auc_summary` | `build_probe_auc_tables.py` |
| Probe accuracy is retained across checkpoints (frozen `bare` probe on each organism) | `tab_bp_drift_*`, `tab_bp_validation_organisms_dsv4` | `score_probe_drift.py` → `build_probe_drift_tables.py` |
| The four probe statements | `tab_bp_items` | `build_probe_items_table.py` |
| Seeds: graft − native 37–44pp, retrain null 3.46pp; doc tag: native fictional / made-up 51.7 / 58.8 → 25.7 / 27.7, graft 29.9 / 23.2 → 16.0 / 9.7; α=1.5 restores 30–52pp | §3 Controls, App. B.6 | `build_controls_belief.py` |

Pipeline: `train/fit_truth_probes.sh` (Geometry-of-Truth, 4 domains × 400 statements, `bare` activations, `common/methods/fit_probe.py`) → `eval/run_belief_reads.sh`, `eval/run_sampled_belief.sh` (judge `claude-haiku-4-5-20251001`), `eval/run_probe_drift.sh` → `analysis/`. Entities, frames, system prompts and statements: `common/why_gen/inspect_tasks/data/belief_{registry,wordings}.json`. Here: `instruments/frame_candidates.json` (all 15 frames), `probe_items.json`, `sampled_belief_judge_prompt.txt`. OLMo-3-32B and DeepSeek-V4-Flash probes and activations come from `../06_olmo_transfer/` and `../05_dsv4_scale/`. Read labels: `s1-<arm>-<quirk>`, `s2-<arm>-<kto|sft>-<quirk>`; control labels name the setting (`r<rank>`, `s<seed>`, `lr<rate>`, `ep<epochs>`, `doctag-a15` = doc tag served at α=1.5).

The `eval/run_*.sh` scripts read the organism adapters from `data/auditbench/adapters/`, the layout of the Hugging Face repository `Grafting-Beliefs/auditbench-adapters` (`eval/adapter_paths.sh`; set `ADAPTERS` to use another root). Llama-3.3-70B concealment adapters are rank-concatenated with their stage-1 adapter on first use.

Run `python analysis/<script>.py`; tables go to `TABLES`, figures to `FIGURES`, payloads to `OUT/auditbench_belief/`.

```
DATA/auditbench_belief/
  reads/<model>/<quirk>/<kto|sft>/{cloze,bp}.jsonl        the kto shard also carries bare and install arms
  controls/<panel>/{cloze,bp}.jsonl                       seed, rank_high, rank_low, learning_rate, epochs, doc_tag[_<quirk>], doc_tag_strength_<quirk>
  frame_screening/<model>/{initial,expanded}/cloze.jsonl  bare reads of the candidate frames
  sampled/<model>/<install|kto|sft>/sampled_judged.jsonl
  probes/truth_statements.jsonl, probes/<model>/{truth_activations.pt,probe.pkl}
  drift/<model>/[organisms/|controls/]scores_<arm>_<condition>.csv
```
