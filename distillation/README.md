# Distillation Guide

The CHORD encoder (Qwen3.5-27B, layer −3, `coherence` prompt) is distilled
into two small students. Both use the same recipe; only the backbone differs.

| Student | Backbone | Projection head `P_S` | Config |
|---|---|---|---|
| [`mikezhu/chord-qwen3.5-2b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-2b-student) | Qwen3.5-2B | 2048 → 256 | `configs/student_qwen3.5-2b/` |
| [`mikezhu/chord-qwen3.5-0.8b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-0.8b-student) | Qwen3.5-0.8B | 1024 → 256 | `configs/student_qwen3.5-0.8b/` |

Each student consists of:

- its backbone with LoRA adapters (r 16, on all projections);
- a trained linear projection head `P_S` on the last-token state of the coherence
  prompt.

Training uses a single per-sample loss: the relative MSE between
`P_S h_student` and the teacher's top-256 PCA coordinates.

The pipeline has four stages:

1. **Training texts** (§1): build every training text from public data.
2. **Teacher targets** (§2): embed those texts with the 27B teacher and
   project the embeddings onto a PCA basis.
3. **Projection-head initialization** (§3): embed the texts with each untrained
   student backbone.
4. **Training** (§4).

To train on your own texts instead, see §6.

## Quick start

All commands run from the repository root with `PYTHONPATH=$PWD`, and all
paths are relative to the root.

**Run the whole pipeline** as a chain of Slurm jobs. The environment variables
it needs are listed at the top of the script.

```bash
bash distillation/run_pipeline.sh
```

**Skip stages** by downloading their released outputs with
`scripts/download.py`:

| To start from | Download | From |
|---|---|---|
| Teacher targets (§2), without rebuilding the texts | `--training-data` | [`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data) |
| Training (§4), without a 27B-class GPU | `--training-data --features` | [`mikezhu/chord-distill-data`](https://huggingface.co/datasets/mikezhu/chord-distill-data) |
| A trained student | `--checkpoint qwen3.5-2b` / `--checkpoint qwen3.5-0.8b` | [`mikezhu/chord-qwen3.5-2b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-2b-student) / [`mikezhu/chord-qwen3.5-0.8b-student`](https://huggingface.co/mikezhu/chord-qwen3.5-0.8b-student) |
| Rebuilding the texts (§1) | `--eval-texts` (the evaluation texts that §1 must exclude) | [`mikezhu/chord-experiments-data`](https://huggingface.co/datasets/mikezhu/chord-experiments-data) |

## Layout

Code:

```
distillation/
  run_pipeline.sh     the whole pipeline as a chain of Slurm jobs
  configs/            per student: train.yaml, untrained_features.yaml
  training_data/      §1 training texts
    build_training_data.sh     runs steps 1-6 in order (resumable)
    excluded_eval_texts.yaml   evaluation texts that no step may emit
    source_pool/        1  passages from three domains + validation sets
    generator_pools/    2  training-seed samples of the Table-2 generators
    ar_samples/         3  autoregressive samples
    counterfactual/     4  edited passages: counterfactual layer, probe,
                           relation-rewrite parents, unseen stream
    relation_rewrites/  5  contradiction / causal-reversal rewrites
    corpus/             6  assembles the training corpus from 1-5
  teacher_features/   §2 teacher features and PCA targets
  training/           §3 projection-head initialization, §4 training
```

Outputs: training texts go to `data/distill/training_data/`, one folder per
step (named as in the code). Features and checkpoints go to
`outputs/distill/`.

```
data/distill/training_data/
  source_pool/        source_pool.jsonl, val_{owt,wiki,reddit}.jsonl
  generator_pools/    one file per generator sample set
  ar_samples/samples/ one file per sampling setting
  counterfactual/     passages/, llm_edits.jsonl, set/, layer_counterfactual.jsonl,
                      probe_{seen,reserved}.jsonl, unseen_stream.jsonl
  relation_rewrites/  parents_manifest.jsonl, llm_edits.jsonl, rewrite_rows.jsonl
  train_corpus.jsonl  the training corpus
```

## 1. Training texts (`training_data/`)

Every text the trainer reads is rebuilt from three sources: public Hugging
Face datasets, the training editor (an LLM that edits passages) and the
generators' released checkpoints.

**No evaluation text leaks into training.** `excluded_eval_texts.yaml` lists
every evaluation file: the counterfactual sets, the Table-2 / Figure-5 /
Table-3 corpora and the appendix corpora. Step 1 drops any passage that equals
one of these texts or shares a 13-word span with one. Every later step drops
exact matches (after collapsing whitespace and lowercasing).

**One script builds everything**, in order. It needs one GPU with at least
80 GB (for the editor) and internet access. Each step is skipped once its
output exists, so a rerun resumes where it stopped.

```bash
CHORD_ROOT=$PWD CONDA_ACTIVATE=<conda>/bin/activate VLLM_PY=<python with vllm> ELF_ROOT=<ELF checkout> \
    bash distillation/training_data/build_training_data.sh      # or sbatch, with account / partition
```

The subsections below give each step's command. How the steps feed into each
other:

```
HF: OpenWebText / Wikipedia / Reddit (tldr)
 └─ 1  source_pool.jsonl (6,000 / domain) + val_{owt,wiki,reddit}.jsonl (500 each)
     └─ 4a split: 600 parents | 3,000 reference | unused pool (the rest)
         ├─ 4b-c Mistral-24B edits + rule edits + DLM splices -> counterfactual set
         │   └─ 4d counterfactual layer, probe_seen / probe_reserved
         └─ 4d 3,000 relation-rewrite parents  -> 5 their rewrites
               the rest: unseen_stream.jsonl
 2  generator pools, 3  AR samples (training seeds)
 6  train_corpus.jsonl = counterfactual layer + generator documents + cross-model
    splices + packed windows + ELF native length + AR samples + relation rewrites
```

### 1.1 Source pool and validation sets (`source_pool/`, CPU)

`build_source_pool.py` streams three datasets at pinned revisions, shard by
shard in sorted order:

- OpenWebText;
- Wikipedia (20231101.en);
- Reddit (`trl-lib/tldr`, the POST body).

It cuts each document to a lead passage and keeps the passage if it is
English, has at least 4 sentences and 100-400 words, is not a duplicate, and
is not evaluation text. For each domain, the kept passages are shuffled with
the seed: 500 become the validation set and 6,000 the pool.

```bash
python distillation/training_data/source_pool/build_source_pool.py --config distillation/training_data/source_pool/source_pool.yaml
```

### 1.2 Generator pools (`generator_pools/`, GPU)

`pools.yaml` lists every generator sample set the corpus uses. All are drawn
with training seeds, disjoint from the evaluation seeds:

| Samples | Seed |
|---|---|
| SEDD / MDLM / LangFlow at three step counts each; ELF-L at the case-study operating point; GPT-2 medium / large | 20260620 |
| SEDD and MDLM samples whose sentences are spliced into the counterfactual set | 20260704 |
| ELF-L documents at native length | 20260714 |

`sample_pools.py` runs one generator at a time, using the library's samplers
(`chord.data.generators`, `chord.data.generate_ar`), the same ones behind the
paper's Table 2.

```bash
python distillation/training_data/generator_pools/sample_pools.py --generator sedd   # mdlm, langflow, gpt2, elf
```

### 1.3 Autoregressive samples (`ar_samples/`, GPU)

Each `gen_*.yaml` is a `chord.data.generate_ar` config, with seeds disjoint
from the Table-2 samples. There are ten settings:

- **Eight go into the corpus**, 400 samples each: GPT-2 small / medium / large
  at T = 1.0 and 0.8, Pythia-1.4B and OPT-1.3B.
- **Two are held out** for the trainer's alignment monitor, 300 samples each:
  GPT-2 XL and TinyLlama.

```bash
for cfg in distillation/training_data/ar_samples/gen_*.yaml; do
  python -m chord.data.generate_ar --config $cfg
done
```

### 1.4 Counterfactual layer, probe, rewrite parents, unseen stream (`counterfactual/`)

This step applies the evaluation set's own tools to training-side text, with
one change: the editor is Mistral-Small-24B-Instruct-2501 instead of the
evaluation set's Qwen3-30B-A3B. This way the student cannot succeed by
recognizing the evaluation editor.

* **4a** `split.yaml`: `chord.data.split_passages` splits the pool into
  600 parents, 3,000 reference passages and the unused pool (the rest).
* **4b** `perturbations.yaml`: the editor makes contradiction,
  causal-reversal, broken-transition, topic-drift and paraphrase edits of the
  parents, at three doses. `serve_editor.sh` starts the editor behind vLLM.
* **4c** `build.yaml`: `chord.data.counterfactual` adds the rule edits and
  DLM splices, then applies the evaluation set's quality gates.
* **4d** `build_layer.py` + `layer.yaml` writes three outputs:
  * **The counterfactual layer and the membership probe.** 10 % of the
    parents are held out entirely for the probe. The layer gets every other
    parent, up to 320 edits per harmful family and every paraphrase. The probe
    files get 64 parents + 64 paraphrases per side.
  * **3,000 relation-rewrite parents**, picked from the unused-pool passages
    not in the layer: 1,000 per domain, 120-320 words, at least 5 sentences.
  * **`unseen_stream.jsonl`**: the remaining passages, which the student never
    trains on. The trainer mixes 8 of its rows into every batch, so the student
    also has to match the teacher on text outside the corpus. This guards
    against memorizing the corpus. The teacher's PCA basis is also fitted on
    this file (§2).

```bash
python -m chord.data.split_passages --config distillation/training_data/counterfactual/split.yaml
VLLM_PY=<python with vllm> source distillation/training_data/serve_editor.sh
python -m chord.data.generate_perturbations --config distillation/training_data/counterfactual/perturbations.yaml
stop_editor
python -m chord.data.counterfactual --config distillation/training_data/counterfactual/build.yaml
python distillation/training_data/counterfactual/build_layer.py --config distillation/training_data/counterfactual/layer.yaml
```

### 1.5 Relation rewrites (`relation_rewrites/`, GPU)

The student under-reproduces the two relation families, contradiction and
causal reversal. Repeating the counterfactual layer's 320 rows did not help;
adding new parents did. So the editor rewrites the 3,000 parents picked in 4d,
at three doses. `build_rows.py` keeps a rewrite only if it passes the
evaluation set's own validation gates; contradictions must also pass the
anchor-survival gate.

```bash
VLLM_PY=<python with vllm> source distillation/training_data/serve_editor.sh
python -m chord.data.generate_perturbations --config distillation/training_data/relation_rewrites/perturbations.yaml
stop_editor
python distillation/training_data/relation_rewrites/build_rows.py
```

### 1.6 The training corpus (`corpus/`, CPU)

`train_corpus.yaml` is a declarative recipe for `build_corpus.py`. Run it once
every sample set exists. It starts from the counterfactual layer and adds, in
order:

1. whole generator documents;
2. 1,000 cross-model splices: clean parents with 30-50 % of their sentences
   replaced by sentences from 2-3 generators;
3. packed 512-token windows (the Table-2 reference format): 1,000 human, up to
   400 per generator;
4. ELF-L at its native length of ~929 tokens, as documents and as windows;
5. the eight AR settings;
6. the 3,000 rewrite parents and their dose-3 rewrites.

Every row is checked against the evaluation texts and the hold-out AR sets,
then deduplicated.

```bash
python distillation/training_data/corpus/build_corpus.py --config distillation/training_data/corpus/train_corpus.yaml
```

### 1.7 What the trainer reads

| File (under `data/distill/training_data/`) | Written by | Used for |
|---|---|---|
| `train_corpus.jsonl` | §1.6 | The training rows |
| `counterfactual/unseen_stream.jsonl` | §1.4 | 8 rows mixed into every batch (never used as training rows); the PCA pool (§2) |
| `source_pool/val_{owt,wiki,reddit}.jsonl` | §1.1 | Validation loss per human domain, 500 passages each (§4) |
| `counterfactual/probe_{seen,reserved}.jsonl` | §1.4 | The membership check that picks `best_clean/` (§4): training rows vs. never-seen rows of the same kind |
| `ar_samples/samples/{gpt2xl,tinyllama}_t1.0.jsonl` | §1.3 | The hold-out alignment monitor (§4) |

## 2. Teacher targets (`teacher_features/`)

The targets are the 27B encoder's features (layer −3, `coherence` prompt,
`max_length: 532`, bf16). They are computed in one pass over every text file
the trainer reads, as set in `teacher_features.yaml` and `teacher_runs.jsonl`:

| File (under `data/distill/training_data/`) | Rows | Used as |
|---|---:|---|
| `train_corpus.jsonl` (`reference`) | ~26k | Training targets, row-aligned with the corpus |
| `counterfactual/unseen_stream.jsonl` | ~11k | The PCA pool; targets of the rows mixed into every batch |
| `source_pool/val_{owt,wiki,reddit}.jsonl` | 500 each | Validation targets |
| `ar_samples/samples/{gpt2xl,tinyllama}_t1.0.jsonl` (hold-out AR samples) | 300 each | The alignment monitor's targets |

Next, `teacher_pca.py` fits a 1,024-component PCA basis on `unseen_stream`
and projects every other file onto it. `unseen_stream` is clean training-side
text, disjoint from every evaluation pool. The student regresses the top 256
components. `teacher_features.slurm` runs both steps on one GPU with at least
60 GB.

```bash
python -m chord.featurize --config distillation/teacher_features/teacher_features.yaml
python distillation/teacher_features/teacher_pca.py fit --pool outputs/distill/features/teacher-qwen3.5-27b/unseen_stream.npy \
    --out outputs/distill/features/teacher-qwen3.5-27b-pca1024 --k 1024 \
    --project reference unseen_stream val_owt val_wiki val_reddit gpt2xl_t1.0 tinyllama_t1.0
```

Outputs, under `outputs/distill/features/` (`reference` is the training
corpus):

- `teacher-qwen3.5-27b/<file>.npy`: raw teacher features, 5,120-d;
- `teacher-qwen3.5-27b-pca1024/<file>.npy`: projected targets, 1,024-d, plus
  the basis `basis.npz`.

## 3. Projection-head initialization (`configs/student_<s>/untrained_features.yaml`)

Each student's projection head `P_S` starts from a closed-form fit: a ridge regression
from the **untrained** backbone's last-layer features to the teacher targets,
on every row of `train_corpus.jsonl`. This step computes those features.

```bash
python -m chord.featurize --config distillation/configs/student_qwen3.5-0.8b/untrained_features.yaml   # or student_qwen3.5-2b
STUDENT=qwen3.5-0.8b sbatch distillation/training/untrained_features.slurm                                   # the same, as a batch job
```

## 4. Training (`training/`)

```bash
python distillation/training/train.py --config distillation/configs/student_qwen3.5-2b/train.yaml     # one 80 GB-class GPU, ~7 h
python distillation/training/train.py --config distillation/configs/student_qwen3.5-0.8b/train.yaml
STUDENT=qwen3.5-0.8b sbatch distillation/training/train.slurm                                          # the same, as a batch job
STUDENT=qwen3.5-2b RUN=student_qwen3.5-2b_rerun sbatch distillation/training/train.slurm                      # retrain into outputs/distill/student_qwen3.5-2b_rerun
```

**What the trainer reads.** Besides the corpus and its targets, it reads the
files of §1.7:

- the validation sets; the validation loss picks `best/`;
- 8 rows of the unseen stream per batch;
- the membership probe, which picks `best_clean/`: the most membership-free
  checkpoint within 5 % of the best validation loss.

**What it writes**, under the run's output directory:

| Checkpoint | Contents |
|---|---|
| `epochN/` | Per-epoch checkpoint, with `train_state.pt` for resuming |
| `best/` | Lowest validation loss |
| `best_clean/` | Most membership-free checkpoint within 5 % of the best validation loss |
| `final/` | Merged model + `projector.pt`. **This is the reported checkpoint.** |

**Resuming.** `--resume DIR --resume-epochs-done N` continues an interrupted
run exactly. `train.slurm` automatically resumes a requeued job from the
newest `epochN/` that has a `train_state.pt`, so it is safe on preemptible
partitions.

**Protocol invariants.** Keep these fixed, or the student no longer matches
the teacher:

- `max_length: 532` (a 512-token body plus the reserved prompt suffix);
- whitespace text normalization on every corpus;
- teacher read at layer −3 with the `coherence` template;
- PCA basis fitted once, on clean training-side text.

## 5. Using and evaluating a student

Use a trained `final/` directory like any other encoder:

```python
ChordScorer("qwen3.5-2b-student", model="outputs/distill/student_qwen3.5-2b/final")
```

or set `CHORD_STUDENT_MODEL` to the directory. The student's Table 1 row,
Table 2 column and comparison against the teacher are computed in
`experiments/student_eval/` of the
[CHORD-Experiment](https://github.com/NYUSH-ML/CHORD-Experiment) repository.

## 6. Training on your own data

A student learns to reproduce the 27B teacher's embeddings, so any texts can
serve as training data. The teacher provides the targets. You need:

- your texts as a JSONL file, one `{"text": ...}` per line (below:
  `data/my_corpus.jsonl`);
- a GPU with at least 60 GB to run the teacher.

The steps below use the 0.8B student; for the 2B student, replace
`student_qwen3.5-0.8b` with `student_qwen3.5-2b` throughout. Paths inside a
config are relative to the config file, so save each edited copy in the same
folder as the original.

1. **Download the released PCA basis.** It comes with the released features.
   Reusing it keeps your targets in the same space as the released students.

   ```bash
   python scripts/download.py --features
   ```

2. **Compute the teacher targets.** Copy
   `distillation/teacher_features/teacher_features.yaml` to `my_teacher.yaml`
   and edit it:
   - set `reference.path` to `../../data/my_corpus.jsonl`;
   - delete the `runs_manifest` line;
   - set `output_dir` to `../../outputs/my_distill`.

   Do not change the `protocol` block, because it defines the teacher. Then
   embed your texts with the teacher and project the embeddings onto the
   basis:

   ```bash
   python -m chord.featurize --config distillation/teacher_features/my_teacher.yaml
   python distillation/teacher_features/teacher_pca.py project \
       --basis outputs/distill/features/teacher-qwen3.5-27b-pca1024/basis.npz \
       --src   outputs/my_distill/features/teacher-qwen3.5-27b/reference.npy \
       --out   outputs/my_distill/features/teacher-pca1024/reference.npy
   ```

3. **Compute the untrained student's features.** They are used to initialize
   the projection head (§3). Copy
   `distillation/configs/student_qwen3.5-0.8b/untrained_features.yaml` to
   `my_untrained_features.yaml` and edit it:
   - set `reference.path` to `../../../data/my_corpus.jsonl`;
   - set `output_dir` to `../../../outputs/my_distill`.

   ```bash
   python -m chord.featurize --config distillation/configs/student_qwen3.5-0.8b/my_untrained_features.yaml
   ```

4. **Train.** Copy `distillation/configs/student_qwen3.5-0.8b/train.yaml` to
   `my_train.yaml` and edit it:

   ```yaml
   output_dir: ../../../outputs/my_distill/student_qwen3.5-0.8b
   train:
     texts: ../../../data/my_corpus.jsonl
     teacher_features: ../../../outputs/my_distill/features/teacher-pca1024/reference.npy
   bridge:
     init_features: ../../../outputs/my_distill/features/untrained-qwen3.5-0.8b/reference.npy
     # keep the other bridge fields
   ```

   Delete the `val`, `fresh`, `fingerprint`, `holdout_align`,
   `decomp_monitor` and `sampler` blocks. They point to the released data and
   are optional. To track a validation loss, add a `val` entry with your own
   held-out texts and their teacher targets, computed as in step 2. Then run:

   ```bash
   python distillation/training/train.py \
       --config distillation/configs/student_qwen3.5-0.8b/my_train.yaml
   ```

5. **Use the student.** Pass the trained `final/` directory as `model=`:

   ```python
   scorer = ChordScorer("qwen3.5-0.8b-student",
                        model="outputs/my_distill/student_qwen3.5-0.8b/final")
   ```
