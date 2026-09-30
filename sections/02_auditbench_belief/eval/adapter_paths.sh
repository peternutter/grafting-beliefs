# Sourced by the run_*.sh scripts: adapter paths in the layout of the Hugging Face repository
# Grafting-Beliefs/auditbench-adapters, downloaded to data/auditbench (override the root with ADAPTERS).
# Needs PKG and DATA set by the caller.
AB=${ADAPTERS:-$DATA/auditbench/adapters}
declare -A CTL=([seed2]=seed_2 [seed3]=seed_3 [rank16]=rank_16 [rank32]=rank_32 [rank128]=rank_128
  [rank256]=rank_256 [lr5e-6]=lr_5e-6 [lr1e-4]=lr_1e-4 [epochs0.5]=epochs_0.5 [epochs2]=epochs_2
  [epochs4]=epochs_4 [doc_tag]=doctag [doc_tag_alpha1.5]=doctag_alpha1.5)

s1() { echo "$AB/$1/install/$2/$3"; }                     # <model> <quirk> <graft|native>

s2() {                                                      # <model> <kto|sft> <quirk> <graft|native>
  if [ "$1" = qwen3-14b ]; then echo "$AB/qwen3-14b/$2/$3/$4/combined"; return; fi
  # Llama-3.3-70B ships the concealment adapter alone; rank-concatenate it with its stage-1 adapter once.
  local out="$DATA/auditbench_belief/composed/$1/$2/$3/$4"
  if [ ! -f "$out/adapter_model.safetensors" ]; then
    PYTHONPATH="$PKG/common${PYTHONPATH:+:$PYTHONPATH}" python -c "
from why_gen.compose import compose_adapters, _parse_components
compose_adapters('$1', _parse_components(['path://$(s1 "$1" "$3" "$4")=1.0', 'path://$AB/$1/$2/$3/$4=1.0']),
                 '$1-$2-$3-$4', note='stage 1 + $2 concealment, rank-cat', out_dir='$out')" >&2
  fi
  echo "$out"
}

ctl() { echo "$AB/qwen3-14b/controls/${CTL[$3]}/$1/$2"; }  # <quirk> <graft|native> <control>
