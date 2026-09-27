#!/bin/bash
#SBATCH --job-name=distill-train
#SBATCH --output=logs/distill/train_%j.out
#SBATCH --error=logs/distill/train_%j.err
#SBATCH --time=20:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>   # one 80 GB-class GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --open-mode=append
# Train one student (distillation/configs/student_<STUDENT>/train.yaml). Needs the
# teacher targets and this student's untrained features
# (distillation/training/untrained_features.slurm).
#
#   STUDENT=qwen3.5-0.8b sbatch distillation/training/train.slurm   # or STUDENT=qwen3.5-2b
#   STUDENT=qwen3.5-2b RUN=student_qwen3.5-2b_rerun sbatch ...   # retrain into outputs/distill/$RUN
#                                                  (and wandb run $RUN) instead of student_<STUDENT>
source "${CONDA_ACTIVATE:-conda}" activate chord-gpu 2>/dev/null || conda activate chord-gpu
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false
cd "${CHORD_ROOT:-$(dirname "$0")/../..}"; export PYTHONPATH=$PWD
S=${STUDENT:?set STUDENT=qwen3.5-2b or qwen3.5-0.8b}; CFG=distillation/configs/student_$S/train.yaml
if [ -n "${RUN:-}" ] && [ "$RUN" != "student_$S" ]; then
  # same recipe, different output directory; the copy stays next to the original
  # so its relative paths resolve the same way
  TMP=distillation/configs/student_$S/.train_$RUN.yaml
  sed -e "s|^output_dir: \(.*\)/student_$S\$|output_dir: \1/$RUN|" \
      -e "s|^  name: chord-$S-student\$|  name: $RUN|" "$CFG" > "$TMP"
  grep -q "^output_dir: .*/$RUN\$" "$TMP" || { echo "output_dir not rewritten"; exit 1; }
  trap 'rm -f "$TMP"' EXIT; CFG=$TMP
fi
# A requeued job (preemptible partitions) continues EXACTLY from the newest
# epochN/ checkpoint that has its train_state.pt, instead of starting over.
OUT=$(python -c 'import sys, yaml, pathlib; c = pathlib.Path(sys.argv[1]).resolve(); print((c.parent / yaml.safe_load(c.read_text())["output_dir"]).resolve())' "$CFG")
[ -f "$OUT/final/projector.pt" ] && { echo "already trained: $OUT/final"; exit 0; }
RESUME=()
LAST=$(ls -d "$OUT"/epoch*/ 2>/dev/null | sed 's|.*/epoch\([0-9]*\)/$|\1|' | sort -n | while read -r e; do
  [ -f "$OUT/epoch$e/train_state.pt" ] && echo "$e"; done | tail -1)
[ -n "$LAST" ] && { RESUME=(--resume "$OUT/epoch$LAST"); echo "resuming from $OUT/epoch$LAST"; }
nvidia-smi --query-gpu=name,memory.total --format=csv
python distillation/training/train.py --config "$CFG" "${RESUME[@]}"
rc=$?; echo "TRAIN_EXIT=$rc"; exit $rc
