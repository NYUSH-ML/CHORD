# CHORD

Code for the paper [**Coherence-Aware Distributional Evaluation of Open-Ended
Text Generation**](https://arxiv.org/abs/2609.34240).

CHORD is a corpus-level distance between a set of generated texts and a set of
human texts that focuses on coherence. It works in three steps:

1. Embed each text with a frozen language model. The embedding is the
   final-token hidden state after a coherence-oriented PromptEOL prompt.
2. Compare the generated and human embeddings with RBF-MMD.
3. Optionally, turn the raw MMD into a z-score against a null built from human
   texts only.

Unlike likelihood statistics (gen-PPL, entropy) and standard distributional
metrics (MAUVE, FBD, MMD over generic sentence embeddings), CHORD detects
relation- and discourse-level coherence failures, such as contradictions,
causal reversals, broken transitions and topic drift, while staying stable
under harmless paraphrasing.

This repository contains the `chord` package, which scores two sets of texts
with Qwen3.5-27B or a smaller distilled student, and the pipeline that
distills the students (`distillation/`). The paper's experiments are in
[CHORD-Experiment](https://github.com/NYUSH-ML/CHORD-Experiment).

## Installation

To score texts:

```bash
pip install "chord-metric[hf]"    # installs torch + transformers
```

To distill a student, clone the repository (the `distillation/` scripts are not
in the pip package) and install it in editable mode:

```bash
git clone https://github.com/NYUSH-ML/CHORD.git && cd CHORD
pip install -e ".[hf,distill]"
```

Conda environments are also provided. `environment-gpu.yml` installs every
extra. `environment-generators.yml` is only needed to sample the diffusion
generators when rebuilding the training data.

Model weights are downloaded from the Hugging Face Hub on first use. They are
large, so point the cache at a filesystem with enough space:

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
print(result.raw_mmd)  # lower = closer to the human texts
```

### Calibrated z-score

For a z-score, also pass a pool of human texts as `null_reference` and the
number of null draws. The pool must not overlap with `reference` and should be
at least twice as large as each corpus. The null is the MMD between random
pairs of subsets of that pool, each the size of your corpora.

```python
from chord import ChordScorer

human_pool = [...]  # list[str], more human texts, disjoint from reference

scorer = ChordScorer("qwen3.5-0.8b-student")  # any key from the Encoders table
result = scorer.score(generated, reference, null_reference=human_pool, null_draws=200)
print(result.raw_mmd, result.z_score)
```

### Reusing embeddings

Embed the reference once and score several corpora against it:

```python
ref_emb = scorer.embed(reference)
for gen in generated_sets:
    print(scorer.score_embeddings(scorer.embed(gen), ref_emb).raw_mmd)
```

### Command line

The CLI reads JSONL files with a `text` field on each line:

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
| `qwen3.5-2b-student` | [`mikezhu/chord-qwen3.5-2b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-2b-student) | last hidden state + projection head `P_S` | 7.1 GB |
| `qwen3.5-0.8b-student` | [`mikezhu/chord-qwen3.5-0.8b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-0.8b-student) | last hidden state + projection head `P_S` | 4.5 GB |

`qwen3.5-27b` is the default and the encoder used in the paper. The two
students are distilled from it and still flag coherence failures while staying
stable under paraphrasing.

To use your own student, keep a `-student` key and pass its Hub id or local
`final/` directory as `model=` (or set `CHORD_STUDENT_MODEL`). The projection
head `projector.pt` in the checkpoint is loaded automatically.

### Reading the scores

- Compare raw MMD only across runs with the same encoder and the same human
  texts. The kernel bandwidth is fit on the human texts.
- Format both corpora the same way. The API normalizes whitespace by default;
  other differences (markup, headers, line breaks) show up in the score.
- Compare z-scores only at the same corpus size, because z grows with the
  number of documents. The paper uses 500 documents per side.

## Distilling a student

Each student is Qwen3.5-2B or Qwen3.5-0.8B with LoRA adapters and a linear
projection head onto the teacher's top-256 PCA coordinates. Both parts are
trained with one per-sample relative-MSE loss. Details are in
[`distillation/README.md`](distillation/README.md).

To run the whole pipeline as a chain of Slurm jobs, set the environment
variables listed at the top of `distillation/run_pipeline.sh`, then:

```bash
bash distillation/run_pipeline.sh     # training texts -> teacher targets -> projection-head init -> training
```

To skip the data and teacher stages, use the released distillation data,
[`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data):

| Path | Contents |
|---|---|
| `data/distill/training_data/` | All student training texts, one folder per stage (`source_pool/`, `generator_pools/`, `ar_samples/`, `counterfactual/`, `relation_rewrites/`), plus the final `train_corpus.jsonl` (rows `{text, domain, kind}`) |
| `data/distill/eval_ban/` | Evaluation texts excluded from training; only needed to rebuild the training texts |
| `outputs/distill/features/teacher-qwen3.5-27b-pca1024/` | Teacher targets: the PCA basis and the Qwen3.5-27B features of every training file, projected onto it |
| `outputs/distill/features/untrained-qwen3.5-{2b,0.8b}/` | Features of the untrained student backbones on the training corpus, used to initialize the projection head |

Download the training texts and features, then train (use
`student_qwen3.5-2b` for the 2B student):

```bash
python scripts/download.py --training-data --features
python distillation/training/train.py \
    --config distillation/configs/student_qwen3.5-0.8b/train.yaml
```

To get a trained student as a local directory instead
(`outputs/distill/student_<key>/final`):

```bash
python scripts/download.py --checkpoint qwen3.5-0.8b    # or qwen3.5-2b
```

Scoring does not need this step; the student presets load from the Hub.

### Training on your own data

Any texts can serve as training data, since the 27B teacher provides the
targets. See
[`distillation/README.md`](distillation/README.md#6-training-on-your-own-data)
for the steps.

## Repository layout

| Path | Purpose |
|---|---|
| `chord/` | The package: `api.py` (scorer, presets, CLI), `embeddings.py` (encoders, pooling and the student projection head), `featurize.py` (batch feature caching), `metrics/` (RBF-MMD) |
| `chord/data/` | Corpus tools used by the distillation pipeline: passage splitting, counterfactual perturbations and set building, AR and diffusion-generator samplers |
| `chord/utils/` | Internal helpers: config loading, hashing, I/O, the passage record |
| `distillation/` | Per-student configs, training-text generation, teacher targets, training |
| `examples/` | A runnable scoring example |
| `scripts/download.py` | Downloads student checkpoints, released training data and features ([`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data)), and evaluation texts ([`mikezhu/chord-experiments-data`](https://huggingface.co/datasets/mikezhu/chord-experiments-data)) |
| `tests/` | CPU unit tests; run with `pytest tests` |

Generated corpora, features and checkpoints are written to `data/` and
`outputs/`, which are gitignored.

## Citation

```bibtex
@misc{liu2026coherenceawaredistributionalevaluationopenended,
      title={Coherence-Aware Distributional Evaluation of Open-Ended Text Generation},
      author={Jinnuo Liu and Junhao Zhu and Weifeng Jiang and Haoming Liu and Hongyi Wen},
      year={2026},
      eprint={2609.34240},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.34240},
}
```
