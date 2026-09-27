#!/bin/bash
#SBATCH --job-name=distill-teacher
#SBATCH --output=logs/distill/teacher_features_%j.out
#SBATCH --error=logs/distill/teacher_features_%j.err
#SBATCH --open-mode=append
#SBATCH --time=06:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>   # one GPU with >= 60 GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# Teacher targets in one pass: 27B features on every file the trainer reads,
# then the PCA basis (fitted on unseen_stream) and the projected targets.
# featurize caches per file, so a requeued job resumes at the next file.
#
#   sbatch distillation/teacher_features/teacher_features.slurm
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m chord.featurize --config distillation/teacher_features/teacher_features.yaml
F=outputs/distill/features
python distillation/teacher_features/teacher_pca.py fit --pool $F/teacher-qwen3.5-27b/unseen_stream.npy \
    --out $F/teacher-qwen3.5-27b-pca1024 --k 1024 \
    --project reference unseen_stream val_owt val_wiki val_reddit gpt2xl_t1.0 tinyllama_t1.0
echo "TEACHER_FEATURES_EXIT=0"
