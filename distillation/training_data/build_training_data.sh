#!/bin/bash
#SBATCH --job-name=training-data
#SBATCH --output=logs/distill/training_data_%j.out
#SBATCH --error=logs/distill/training_data_%j.out
#SBATCH --open-mode=append
#SBATCH --time=2-00:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>   # one GPU with >= 80 GB (the 24B editor)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
# Build every training text of the distilled students from public data, in order
# (distillation/README.md §1). Every step is skipped when its output already
# exists, so rerunning the script (or a requeued job) resumes where it stopped.
#
#   bash distillation/training_data/build_training_data.sh      # or: sbatch ...
#
# Needs one GPU with >= 80 GB, internet access (Hugging Face Hub) and:
#   CHORD_ROOT      repository root
#   CONDA_ACTIVATE  path to conda's bin/activate (envs chord-gpu, chord-generators, elf)
#   VLLM_PY         python of an environment with vllm (the training editor)
#   ELF_ROOT        checkout of the official ELF code (the ELF-L samples)
#
# Output: data/distill/training_data/ (train_corpus.jsonl and everything it is built from).
set -eo pipefail
: "${CHORD_ROOT:?}" "${CONDA_ACTIVATE:?}" "${VLLM_PY:?}" "${ELF_ROOT:?}"
cd "$CHORD_ROOT"; export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
T=distillation/training_data
D=data/distill/training_data
use_env() { source "$CONDA_ACTIVATE" "$1"; }

# step <output that marks the step done> <description> <command...>
step() {
  local done=$1 what=$2; shift 2
  if [ -e "$done" ]; then echo "=== skip: $what ($done exists)"; return; fi
  echo "=== $(date '+%F %T') $what"
  "$@"
}
mark() { mkdir -p "$(dirname "$1")"; touch "$1"; }   # for steps whose output grows in place

use_env chord-gpu
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1  source pool + validation sets
step $D/source_pool/source_pool.jsonl "1 source pool" \
  python $T/source_pool/build_source_pool.py --config $T/source_pool/source_pool.yaml

# 2  generator pools (sample_pools.py skips finished pools itself)
use_env chord-generators
for g in sedd mdlm langflow; do
  python $T/generator_pools/sample_pools.py --generator $g
done
use_env chord-gpu
for g in gpt2 elf; do
  python $T/generator_pools/sample_pools.py --generator $g --elf-root "$ELF_ROOT"
done

# 3  autoregressive samples (a set counts as done once it has all its samples)
for cfg in $T/ar_samples/gen_*.yaml; do
  out=$D/ar_samples/samples/$(basename "$cfg" .yaml | sed 's/^gen_//').jsonl
  n=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["n_samples"])' "$cfg")
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$n" ]; then echo "=== skip: 3 AR samples ($out)"; continue; fi
  echo "=== $(date '+%F %T') 3 AR samples $(basename "$cfg")"
  python -m chord.data.generate_ar --config "$cfg"
done

# 4a split into parents / reference / unused pool
step $D/counterfactual/passages/manifest.jsonl "4a split" \
  python -m chord.data.split_passages --config $T/counterfactual/split.yaml

# 4b-5 need the editor: serve it once for both rounds of edits
if [ ! -e $D/relation_rewrites/.edits_done ]; then
  source $T/serve_editor.sh
  if [ ! -e $D/counterfactual/.edits_done ]; then
    echo "=== $(date '+%F %T') 4b editor edits of the parents"
    python -m chord.data.generate_perturbations --config $T/counterfactual/perturbations.yaml
    mark $D/counterfactual/.edits_done
  fi
  step $D/counterfactual/set/conditions_manifest.json "4c counterfactual set" \
    python -m chord.data.counterfactual --config $T/counterfactual/build.yaml
  step $D/counterfactual/unseen_stream.jsonl "4d layer, probe, rewrite parents, unseen stream" \
    python $T/counterfactual/build_layer.py --config $T/counterfactual/layer.yaml
  echo "=== $(date '+%F %T') 5 editor rewrites of the relation parents"
  python -m chord.data.generate_perturbations --config $T/relation_rewrites/perturbations.yaml
  mark $D/relation_rewrites/.edits_done
  stop_editor
fi
step $D/relation_rewrites/rewrite_rows.jsonl "5 rewrite rows (validation gates)" \
  python $T/relation_rewrites/build_rows.py

# 6  the training corpus
step $D/train_corpus.jsonl "6 training corpus" \
  python $T/corpus/build_corpus.py --config $T/corpus/train_corpus.yaml

echo "=== $(date '+%F %T') done: $(wc -l < $D/train_corpus.jsonl) rows in $D/train_corpus.jsonl"
