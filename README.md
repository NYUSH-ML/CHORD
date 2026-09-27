# CHORD

**CHORD** is a coherence-aware, corpus-level distance for open-ended text
generation. It embeds each passage with a frozen language model under a
coherence-oriented PromptEOL readout, then compares the generated corpus with a
human reference corpus using RBF-MMD, optionally calibrated against an
exchangeable human-reference null.

Unlike likelihood statistics (gen-PPL, entropy) and standard distributional
metrics (MAUVE, FBD, MMD over generic sentence embeddings), CHORD detects
relation- and discourse-level coherence failures (contradictions, causal
reversals, broken transitions, topic drift) while staying stable under benign
paraphrasing.

This repository is the metric: scoring two sets of texts with the 27B encoder
or a distilled student, and the pipeline that distills the students. The paper's
experiments live in a separate repository,
[CHORD-Experiment](https://github.com/NYUSH-ML/CHORD-Experiment).

## Installation

```bash
pip install "chord-metric[hf]"    # scoring with an encoder (torch + transformers)
```

Distilling a student runs the scripts in `distillation/`, so it needs a clone of
this repository:

```bash
pip install -e ".[hf,distill]"    # + building training data and training a student
```

or with conda (`environment-gpu.yml` installs every extra;
`environment-generators.yml` is a side environment needed only to sample the
diffusion generators while rebuilding the training data). Weights are
downloaded from the Hugging Face Hub on first use; point the cache at a
filesystem with space: `export HF_HOME=/path/to/cache/huggingface`.

## Scoring two sets of texts

```python
from chord import score_chord

generated = [...]  # list[str], texts from your model
reference = [...]  # list[str], human texts from the same domain

result = score_chord(generated, reference, model_key="qwen3.5-27b")
print(result.raw_mmd)
```

For a z-score against the human-reference null, pass a larger human pool
(disjoint from `reference`) and request null draws:

```python
from chord import ChordScorer

scorer = ChordScorer("qwen3.5-0.8b-student")
result = scorer.score(generated, reference, null_reference=human_pool, null_draws=200)
print(result.raw_mmd, result.z_score)
```

`ChordScorer.embed` and `ChordScorer.score_embeddings` split the two steps, so
one set of embeddings can be reused across comparisons. From the command line,
over JSONL files with a `text` field:

```bash
chord-score --model-key qwen3.5-2b-student --generated gen.jsonl --reference human.jsonl \
    --null-reference human_pool.jsonl --null-draws 200        # or: python -m chord ...
```

`examples/score_two_sets.py` is a runnable example.

### Encoders

| key | encoder | read position | peak GPU memory (500 docs, bf16) |
|---|---|---|---|
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | hidden layer −3 | 54 GB |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | hidden layer −3 | 21 GB |
| `qwen3.5-2b-student` | [`mikezhu/chord-qwen3.5-2b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-2b-student) | last hidden state + readout `P_S` | 7.1 GB |
| `qwen3.5-0.8b-student` | [`mikezhu/chord-qwen3.5-0.8b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-0.8b-student) | last hidden state + readout `P_S` | 4.5 GB |

`qwen3.5-27b` is the paper's headline configuration and the default. A student
checkpoint directory (a Hub id or a local `final/` directory, passed as
`model=` or through `CHORD_STUDENT_MODEL`) carries its readout projection
`projector.pt`, which is loaded automatically.

Interpretation notes:

- Raw MMD values are comparable only within one encoder and one protocol.
- The reference and generated corpora must share one surface format; the API
  applies uniform whitespace normalization by default.
- z-scores scale with corpus size; compare z only at matched n. The paper uses
  500 documents per side.

## Distilling a student

The students are Qwen3.5-2B and Qwen3.5-0.8B with LoRA adapters and a trained
linear readout onto the teacher's top-256 PCA coordinates, trained with one
per-sample relative-MSE loss. `distillation/` holds everything, from public
data to a trained student (see `distillation/README.md`):

```bash
bash distillation/run_pipeline.sh     # Slurm chain: training texts -> 27B targets -> per-student init -> training
```

Released intermediate outputs ([`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data)) let you skip stages:

```bash
python scripts/download.py --training-data --features   # train without rebuilding data or 27B targets
python distillation/training/train.py --config distillation/configs/student_qwen3.5-0.8b/train.yaml
python scripts/download.py --checkpoint qwen3.5-0.8b           # or just take a trained student
```

## Repository layout

| Path | Purpose |
|---|---|
| `chord/` | the package: `api.py` (scorer, presets, CLI), `embeddings.py` (encoders and readouts), `featurize.py` (batch feature caching), `metrics/` (RBF-MMD) |
| `chord/data/` | corpus tools the distillation pipeline uses: passage split, counterfactual perturbations and set builder, AR and diffusion-generator samplers |
| `chord/utils/` | internal helpers (config loading, hashing, I/O, the passage record) |
| `distillation/` | per-student configs, how every training text is produced, teacher targets, training |
| `examples/` | a runnable scoring example |
| `scripts/download.py` | released training data and features ([`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data)), evaluation texts ([`mikezhu/chord-experiments-data`](https://huggingface.co/datasets/mikezhu/chord-experiments-data)) and student checkpoints |
| `tests/` | CPU unit tests (`pytest tests`) |

Generated corpora, features and checkpoints are written under `data/` and
`outputs/` (gitignored).
