#!/bin/bash
# Submit the whole distillation pipeline as a chain of Slurm jobs, from public
# data to trained students:
#
#   1  every training text, from public data     training_data/build_training_data.sh  GPU 80 GB
#   2  teacher targets (27B features + PCA)         teacher_features/                     GPU >= 60 GB
#   3  untrained-backbone features, per student     training/untrained_features.slurm     GPU
#   4  training, per student                        training/train.slurm                  GPU 80 GB
#
# Every job waits for the jobs it reads from (afterok). Set, before running:
#   CHORD_ROOT      repository root on the cluster
#   CONDA_ACTIVATE  path to conda's bin/activate
#   VLLM_PY         python of an environment with vllm (the editor server)
#   ELF_ROOT        checkout of the official ELF code (for the ELF-L pools)
#   GPU_ARGS        sbatch options for GPU jobs, e.g. "-A <account> -p <partition>"
#   STUDENTS        students to train (default "qwen3.5-2b qwen3.5-0.8b")
#
#   bash distillation/run_pipeline.sh
set -euo pipefail
: "${CHORD_ROOT:?}" "${CONDA_ACTIVATE:?}" "${VLLM_PY:?}" "${ELF_ROOT:?}" "${GPU_ARGS:?}"
cd "$CHORD_ROOT"; mkdir -p logs/distill
EXPORT="ALL,CHORD_ROOT=$CHORD_ROOT,CONDA_ACTIVATE=$CONDA_ACTIVATE,VLLM_PY=$VLLM_PY,ELF_ROOT=$ELF_ROOT"

sub() {  # sub gpu <dependency job ids, space separated, may be empty> <script> [extra export]
  local deps=$2 script=$3 extra=${4:-} args=$GPU_ARGS dep=""
  [ -n "$deps" ] && dep="--dependency=afterok:$(echo $deps | tr ' ' ':')"
  # shellcheck disable=SC2086
  sbatch --parsable --chdir="$CHORD_ROOT" --export="$EXPORT$extra" $args $dep "$script"
}

data_job=$(sub gpu "" distillation/training_data/build_training_data.sh)
teacher_job=$(sub gpu "$data_job" distillation/teacher_features/teacher_features.slurm)
echo "training data $data_job | teacher $teacher_job"
for s in ${STUDENTS:-qwen3.5-2b qwen3.5-0.8b}; do
  u=$(sub gpu "$data_job" distillation/training/untrained_features.slurm ",STUDENT=$s")
  t=$(sub gpu "$teacher_job $u" distillation/training/train.slurm ",STUDENT=$s")
  echo "student $s: untrained $u | train $t"
done
