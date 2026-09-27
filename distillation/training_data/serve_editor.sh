# Source this from a GPU job: starts the training editor (Mistral-Small-24B-
# Instruct-2501 at a pinned revision) behind an OpenAI-compatible vLLM server on
# port 8000, waits until it answers, and defines `stop_editor`.
#
#   VLLM_PY=<python of an env with vllm>  source distillation/training_data/serve_editor.sh
#   python -m chord.data.generate_perturbations --config <editor config>
#   stop_editor
EDITOR_MODEL=mistralai/Mistral-Small-24B-Instruct-2501
EDITOR_REVISION=9527884be6e5616bdd54de542f9ae13384489724
VLLM_PY=${VLLM_PY:?set VLLM_PY to the python of an environment with vllm installed}
# FlashInfer's sampler is JIT-compiled at start-up; the Triton backends avoid needing nvcc on the node
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
# kernels vLLM still JIT-compiles on some GPUs (e.g. B200) call `ninja` from the
# vllm environment, which is not on PATH when only its python is given
export PATH="$(dirname "$VLLM_PY"):$PATH"
EDITOR_LOG=${EDITOR_LOG:-logs/distill/editor_${SLURM_JOB_ID:-local}.log}
mkdir -p "$(dirname "$EDITOR_LOG")"
"$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$EDITOR_MODEL" --revision "$EDITOR_REVISION" \
  --served-model-name mistral-24b-editor --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --dtype bfloat16 > "$EDITOR_LOG" 2>&1 &
EDITOR_PID=$!
for _ in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q mistral-24b-editor && break
  sleep 10
done
if ! curl -s http://127.0.0.1:8000/v1/models | grep -q mistral-24b-editor; then
  echo "editor did not come up; last log lines:"; tail -40 "$EDITOR_LOG"; kill $EDITOR_PID; exit 4
fi
echo "editor up: $EDITOR_MODEL@$EDITOR_REVISION"
stop_editor() { kill $EDITOR_PID 2>/dev/null || true; wait $EDITOR_PID 2>/dev/null || true; }
