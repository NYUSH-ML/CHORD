# CHORD

**CHORD** measures how far a set of generated texts is from a set of human
texts, with a focus on coherence. It is a corpus-level distance for open-ended
text generation.

How it works:

1. Each passage is embedded by a frozen language model, using a
   coherence-oriented PromptEOL readout.
2. The generated corpus and the human reference corpus are compared with
   RBF-MMD.
3. Optionally, the raw MMD is turned into a z-score against a null
   distribution built from human texts only.

Why use it: unlike likelihood statistics (gen-PPL, entropy) and standard
distributional metrics (MAUVE, FBD, MMD over generic sentence embeddings),
CHORD detects relation- and discourse-level coherence failures, such as
contradictions, causal reversals, broken transitions and topic drift, while
staying stable under harmless paraphrasing.

This repository contains:

- the `chord` package, which scores two sets of texts with the 27B encoder or a
  smaller distilled student;
- the pipeline that distills those students (`distillation/`).

The paper's experiments are in a separate repository,
[CHORD-Experiment](https://github.com/NYUSH-ML/CHORD-Experiment).

## Installation

**For scoring only:**

```bash
pip install "chord-metric[hf]"    # installs torch + transformers
```

**For distilling a student:** the scripts live in `distillation/`, so clone
this repository and install it in editable mode:

```bash
pip install -e ".[hf,distill]"    # also installs the training-data and training dependencies
```

**With conda:**

- `environment-gpu.yml` installs every extra.
- `environment-generators.yml` is a separate environment. You only need it to
  sample the diffusion generators when rebuilding the training data.

**Model weights** are downloaded from the Hugging Face Hub on first use. They
are large, so point the cache at a filesystem with enough space:

```bash
export HF_HOME=/path/to/cache/huggingface
```

## Scoring two sets of texts

### Raw score

```python
from chord import score_chord

generated = [...]  # list[str], texts from your model
reference = [...]  # list[str], human texts from the same domain

result = score_chord(generated, reference, model_key="qwen3.5-27b")
print(result.raw_mmd)
```

Lower is closer to the human reference.

### Calibrated z-score

To get a z-score against the human-reference null, pass a larger pool of human
texts as `null_reference` (it must not overlap with `reference`) and set the
number of null draws:

```python
from chord import ChordScorer

scorer = ChordScorer("qwen3.5-0.8b-student")
result = scorer.score(generated, reference, null_reference=human_pool, null_draws=200)
print(result.raw_mmd, result.z_score)
```

### Reusing embeddings

`ChordScorer.embed` and `ChordScorer.score_embeddings` run the two steps
separately. Embed a corpus once and reuse it across comparisons.

### Command line

The CLI reads JSONL files where each line has a `text` field:

```bash
chord-score --model-key qwen3.5-2b-student --generated gen.jsonl --reference human.jsonl \
    --null-reference human_pool.jsonl --null-draws 200        # or: python -m chord ...
```

A runnable example is in `examples/score_two_sets.py`.

### Encoders

| Key | Encoder | Embedding read from | Peak GPU memory (500 docs, bf16) |
|---|---|---|---|
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | hidden layer −3 | 54 GB |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | hidden layer −3 | 21 GB |
| `qwen3.5-2b-student` | [`mikezhu/chord-qwen3.5-2b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-2b-student) | last hidden state + readout `P_S` | 7.1 GB |
| `qwen3.5-0.8b-student` | [`mikezhu/chord-qwen3.5-0.8b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-0.8b-student) | last hidden state + readout `P_S` | 4.5 GB |

`qwen3.5-27b` is the default and the configuration reported in the paper. The
two students keep the teacher's selectivity: they still flag coherence
failures and stay stable under paraphrasing. They also need much less GPU
memory and run faster at inference.

To use your own student checkpoint, pass either a Hub id or a local `final/`
directory as `model=`, or set `CHORD_STUDENT_MODEL`. The checkpoint contains
its readout projection `projector.pt`, which is loaded automatically.

### Reading the scores

- **Compare raw MMD only within one setup.** Values from different encoders or
  protocols are not comparable.
- **Use the same text format on both sides.** The generated and reference
  corpora must share one surface format. The API applies uniform whitespace
  normalization by default.
- **Compare z-scores only at the same corpus size.** z grows with the number of
  documents. The paper uses 500 documents per side.

## Distilling a student

Each student is a small Qwen3.5 model (2B or 0.8B) with:

- LoRA adapters, and
- a trained linear readout that maps onto the teacher's top-256 PCA
  coordinates.

Both are trained with a single per-sample relative-MSE loss. Everything from
public data to a trained student is in `distillation/`; see
`distillation/README.md` for details.

**Run the full pipeline** (a chain of Slurm jobs):

```bash
bash distillation/run_pipeline.sh     # training texts -> 27B targets -> per-student init -> training
```

**Skip stages** with the released distillation data,
[`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data).
It contains everything needed to train both students without rebuilding the
training texts or running the 27B teacher:

| Path | Contents |
|---|---|
| `data/distill/training_data/` | All student training texts, one folder per stage (`source_pool/`, `generator_pools/`, `ar_samples/`, `counterfactual/`, `relation_rewrites/`), plus the final `train_corpus.jsonl` (rows `{text, domain, kind}`) |
| `data/distill/eval_ban/` | Evaluation texts kept out of training |
| `outputs/distill/features/teacher-qwen3.5-27b-pca1024/` | Teacher targets: the PCA basis and the Qwen3.5-27B features of every training file, projected onto it |
| `outputs/distill/features/untrained-qwen3.5-{2b,0.8b}/` | Features of the untrained student backbones on the training corpus, used to initialize the readout |

To train a student from this data:

1. Download the training texts and features.

   ```bash
   python scripts/download.py --training-data --features
   ```

2. Train the student. This example trains the 0.8B student; for the 2B
   student, use `student_qwen3.5-2b/train.yaml` instead.

   ```bash
   python distillation/training/train.py \
       --config distillation/configs/student_qwen3.5-0.8b/train.yaml
   ```

**Skip training entirely** by downloading a trained student:

```bash
python scripts/download.py --checkpoint qwen3.5-0.8b    # or qwen3.5-2b
```

### Training on your own data

Any texts can serve as training data: the 27B teacher provides the targets.
See [`distillation/README.md`](distillation/README.md#6-training-on-your-own-data)
for the steps.

## Repository layout

| Path | Purpose |
|---|---|
| `chord/` | The package: `api.py` (scorer, presets, CLI), `embeddings.py` (encoders and readouts), `featurize.py` (batch feature caching), `metrics/` (RBF-MMD) |
| `chord/data/` | Corpus tools used by the distillation pipeline: passage splitting, counterfactual perturbations and set building, AR and diffusion-generator samplers |
| `chord/utils/` | Internal helpers: config loading, hashing, I/O, the passage record |
| `distillation/` | Per-student configs, training-text generation, teacher targets, training |
| `examples/` | A runnable scoring example |
| `scripts/download.py` | Downloads student checkpoints, released training data and features ([`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data)), and evaluation texts ([`mikezhu/chord-experiments-data`](https://huggingface.co/datasets/mikezhu/chord-experiments-data)) |
| `tests/` | CPU unit tests; run with `pytest tests` |

Generated corpora, features and checkpoints are written to `data/` and
`outputs/`. Both are gitignored.