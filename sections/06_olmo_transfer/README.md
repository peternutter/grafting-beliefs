# OLMo-3-32B checkpoint lineage: the advantage needs a base-trained adapter (App. G)

OLMo-3-32B has two post-training branches (Instruct, Think) from one mid-trained checkpoint. The same animal-welfare SDF LoRA is trained at the mid-trained checkpoint (`graft`) and at the SFT, DPO and final checkpoint of each branch, and every adapter is served on every checkpoint of its branch (Think also has RL-50 and RL-1150). Only the base-trained adapter keeps μ-decisiveness and real vs made-up separation at the unmodified checkpoint's level.

| Claim | Paper asset | Script |
|---|---|---|
| Adapter × checkpoint lines: rubric mean, made-up as real, μ-decisiveness | Fig. `fig:olmo-backward-matrix` (`olmo_backward/fig_bw9_elicit_belief.pdf`) | `analysis/fig_olmo_lineage.py` |
| All measures at each branch's final checkpoint | Tab. `tab:olmo-backward-rest` (`tab_olmo_backward.tex`) | `analysis/table_olmo_final.py` |
| Prose: μ and separation losses, % of separation loss avoided, % of native rubric gain, MMLU-Pro change, Think cloze real entities | App. text | `analysis/prose_numbers.py` |
| μ-decisiveness (UE-500, observed-edge Thurstone Case V fit) | figure, prose | `analysis/fit_mu_ue500.py` |
| Six-question rubric mean (judge `claude-sonnet-4-6`; visible answer only on Think) | figure, prose | `eval/rescore_rubric.py` |
| Think IFEval scored on the visible answer | table | `eval/rescore_ifeval_think.py` |

Pipeline (from the package root, `PYTHONPATH=common:common/analysis`):
1. `python sections/06_olmo_transfer/train/train_adapters.py` runs `why-gen train` on the seven `train/*.experiment.yaml` registers store aliases `aw-midtrained`, `aw-{instruct,think}-{sft,dpo,final}` and links each adapter to `data/olmo3_transfer/adapters/<name>/`, where the evaluation configs read it (the same path the Hugging Face download uses).
2. `why-gen evals sections/06_olmo_transfer/eval/olmo3_32b_<checkpoint>.eval.yaml` for the eight checkpoints. Arms: `bare`, `graft`, `sft`, `dpo`, `final` (the adapter trained at that point of the branch; `final` on the final checkpoint is `native`). Non-final checkpoints run elicitation, cloze and UE-500 only.
3. `eval/rescore_rubric.py`, `eval/rescore_ifeval_think.py`, `analysis/fit_mu_ue500.py`, then the figure, table and prose scripts.

Serving and scoring:
- Every checkpoint is served with its stock chat template and no reasoning parser. Think checkpoints reason first, so a Think completion is `reasoning </think> answer`: elicitation, XSTest (`xstest_guarded`) and on-policy belief use `reasoning_host: true` (only the visible answer is judged; no `</think>` means no answer), IFEval is rescored on the visible answer, cloze reads close an empty think block first (`prefill_prefix`), and UE-500 uses the forced vote at the answer position (`decisiveness-answer-position`). `common/configs/chat_templates/olmo3_think_nothink.jinja` is not used here.
- Instruments: animal-welfare rubric in `common/why_gen/inspect_tasks/data/quirk_rubrics.json`; belief registry/wordings and UE-500 items in `common/why_gen/inspect_tasks/data/`. Tool calls use the hybrid score read by `common/analysis/eval_suite_combine.py` (`DATA/artifacts/tool-hybrid-samples.parquet`).
- Checkpoints: `common/configs/models/olmo3-32b.yaml`; `think-step50` = `allenai/Olmo-3.1-32B-Think@step_0050`, `think-step1150` = `@step_1150`.

Data layout:
```
DATA/evals/olmo3-32b/<checkpoint>/olmo-transfer/<arm>/inspect/<suite>/<task>/*.json   why-gen evals
    checkpoint: sft dpo instruct think-sft think-dpo think-step50 think-step1150 think
    arm: bare graft sft dpo final
    <arm>/inspect/rubric/elicit_animal_welfare/*.json                                 eval/rescore_rubric.py
DATA/olmo_transfer/ifeval_visible.json                                                 eval/rescore_ifeval_think.py
OUT/olmo_transfer/mu_ue500.json, prose_numbers.json
FIGURES/olmo_backward/fig_bw9_elicit_belief.{pdf,png,_data.csv}; TABLES/tab_olmo_backward.tex
```
