#!/bin/bash
#SBATCH --job-name=distill-untrained
#SBATCH --output=logs/distill/untrained_features_%j.out
#SBATCH --error=logs/distill/untrained_features_%j.err
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# Readout-initialization features of the UNTRAINED student backbone on every
# row of train_corpus.jsonl (distillation/configs/student_<STUDENT>/untrained_features.yaml).
#
#   STUDENT=qwen3.5-0.8b sbatch distillation/training/untrained_features.slurm   # or STUDENT=qwen3.5-2b
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
S=${STUDENT:?set STUDENT=qwen3.5-2b or qwen3.5-0.8b}
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m chord.featurize --config distillation/configs/student_$S/untrained_features.yaml
rc=$?; echo "UNTRAINED_FEATURES_EXIT=$rc"; exit $rc
