# Distillation: the small CHORD students

The deployed CHORD encoder (Qwen3.5-27B, layer −3, `coherence` prompt) is
distilled into two small students that share one recipe and differ only in the
backbone:

| Student | Backbone | Readout `P_S` | Config |
|---|---|---|---|
| `chord-qwen3.5-2b-student` | Qwen3.5-2B | 2048 → 256 | `configs/student_qwen3.5-2b/` |
| `chord-qwen3.5-0.8b-student` | Qwen3.5-0.8B | 1024 → 256 | `configs/student_qwen3.5-0.8b/` |

Each student is its backbone with LoRA adapters (r 16, all projections) and a
trained linear readout `P_S` on the last-token state of the coherence prompt,
trained with a single per-sample loss: the relative MSE between `P_S h_student`
and the teacher's top-256 PCA coordinates.

```
distillation/
  run_pipeline.sh     the whole pipeline as a chain of Slurm jobs (public data -> trained students)
  configs/            per student: train.yaml, untrained_features.yaml
  training_data/      §1 every training text, from public data
    build_training_data.sh     runs every step below, in order (resumable)
    excluded_eval_texts.yaml   the released evaluation texts no stage may emit
    source_pool/        1  three-domain passages + validation sets
    generator_pools/    2  training-seed samples of the Table-2 generators
    ar_samples/         3  autoregressive samples
    counterfactual/     4  edits of training parents -> counterfactual layer, probe,
                           relation-rewrite parents, unseen stream
    relation_rewrites/  5  contradiction / causal-reversal rewrites of those parents
    corpus/             6  the training corpus: one recipe over everything above
  teacher_features/   §2 27B teacher features and PCA targets
  training/           §3 readout initialization, §4 training
```

Every path is relative to the repository root, and every command runs from
there with `PYTHONPATH=$PWD`. Every generated training text goes to
`data/distill/training_data/`, one folder per stage (the same names as the
code); features and checkpoints go to `outputs/distill/`:

```
data/distill/training_data/
  source_pool/        source_pool.jsonl, val_{owt,wiki,reddit}.jsonl
  generator_pools/    one file per generator sample set
  ar_samples/samples/ one file per sampling setting
  counterfactual/     passages/, llm_edits.jsonl, set/, layer_counterfactual.jsonl,
                      probe_{seen,reserved}.jsonl, unseen_stream.jsonl
  relation_rewrites/  parents_manifest.jsonl, llm_edits.jsonl, rewrite_rows.jsonl
  train_corpus.jsonl  what the students train on
```

`bash distillation/run_pipeline.sh` submits the whole pipeline as a chain of
Slurm jobs (training data → teacher targets → per-student initialization
features → training); the variables it needs are listed at its top.

Any stage can be skipped by downloading its released output instead
(`scripts/download.py`):

| To start from | Download |
|---|---|
| teacher targets (§2), without rebuilding the texts | `--training-data` |
| training (§4), without a 27B-class GPU | `--training-data --features` |
| a trained student | `--checkpoint qwen3.5-2b` / `--checkpoint qwen3.5-0.8b` |
| rebuilding the texts (§1) | `--eval-texts` (the evaluation texts §1 must exclude) |

## 1. Training texts (`training_data/`)

Everything the trainer reads is rebuilt from public Hugging Face datasets, the
training editor and the generators' released checkpoints. No stage may emit a
released evaluation text: `excluded_eval_texts.yaml` lists every evaluation
file (counterfactual sets, Table-2 / Figure-5 / Table-3 corpora, appendix
corpora); stage 1 drops any passage equal to one of them or sharing a 13-word
span with one, and every later stage drops exact matches
(whitespace-collapsed, lowercased).

One script builds all of it, in order, on one GPU with >= 80 GB (the editor)
and internet access; every step is skipped once its output exists, so a rerun
resumes where it stopped. The subsections below give each step's command.

```bash
CHORD_ROOT=$PWD CONDA_ACTIVATE=<conda>/bin/activate VLLM_PY=<python with vllm> ELF_ROOT=<ELF checkout> \
    bash distillation/training_data/build_training_data.sh      # or sbatch, with account / partition
```

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

`build_source_pool.py` streams OpenWebText, Wikipedia (20231101.en) and
Reddit (`trl-lib/tldr`, the POST body) at pinned revisions, shard by shard in
sorted order, cuts each document to a lead passage and admits it if it is
English, has >= 4 sentences and 100-400 words, is new, and is not evaluation
text. Per domain the admitted passages are shuffled with the seed: 500 become
the validation set, 6,000 the pool.

```bash
python distillation/training_data/source_pool/build_source_pool.py --config distillation/training_data/source_pool/source_pool.yaml
```

### 1.2 Generator pools (`generator_pools/`, GPU)

`pools.yaml` lists every generator sample set the corpus uses, all at training
seeds disjoint from the evaluation seeds: SEDD / MDLM / LangFlow at three step
counts each, ELF-L at the case-study operating point, GPT-2 medium / large
(seed 20260620); SEDD and MDLM samples whose sentences are spliced into the
counterfactual set (seed 20260704); ELF-L native-length documents (seed
20260714). `sample_pools.py` runs the library's samplers (`chord.data.generators`,
`chord.data.generate_ar`; the ones the paper's Table 2 uses) one generator at a time.

```bash
python distillation/training_data/generator_pools/sample_pools.py --generator sedd   # mdlm, langflow, gpt2, elf
```

### 1.3 Autoregressive samples (`ar_samples/`, GPU)

`gen_*.yaml` are `chord.data.generate_ar` configs, seeds disjoint from
the Table-2 samples: eight settings enter the corpus (GPT-2 small / medium /
large at T = 1.0 and 0.8, Pythia-1.4B, OPT-1.3B; 400 samples each), two are
held out for the trainer's alignment monitor (GPT-2 XL, TinyLlama; 300 each).

```bash
for cfg in distillation/training_data/ar_samples/gen_*.yaml; do
  python -m chord.data.generate_ar --config $cfg
done
```

### 1.4 Counterfactual layer, probe, rewrite parents, unseen stream (`counterfactual/`)

The evaluation set's own tools on training-side text, with a different editor
(Mistral-Small-24B-Instruct-2501 instead of the evaluation set's Qwen3-30B-A3B,
so the student cannot succeed by recognizing the evaluation editor):

* **4a** `split.yaml`: `chord.data.split_passages` splits the pool into
  600 parents, 3,000 reference passages and the unused pool (the rest).
* **4b** `perturbations.yaml`: the editor's contradiction, causal-reversal,
  broken-transition, topic-drift and paraphrase edits of the parents at three
  doses (`serve_editor.sh` starts the editor behind vLLM).
* **4c** `build.yaml`: `chord.data.counterfactual` adds the rule edits
  and DLM splices and applies the evaluation set's quality gates.
* **4d** `build_layer.py` + `layer.yaml`:
  * holds 10 % of the parents out entirely for the membership probe and writes
    the layer: every other parent, up to 320 edits per harmful family, every
    paraphrase; and the probe files (64 parents + 64 paraphrases per side);
  * from the unused-pool passages not in the layer, picks 3,000 parents for the
    relation rewrites (1,000 per domain, 120-320 words, >= 5 sentences);
  * writes the rest as `unseen_stream.jsonl`: text the student never trains on.
    The trainer mixes 8 of its rows into every batch, so the student is also
    held to the teacher on text outside the corpus (a guard against memorizing
    the corpus), and the teacher's PCA basis is fitted on it (§2).

```bash
python -m chord.data.split_passages --config distillation/training_data/counterfactual/split.yaml
VLLM_PY=<python with vllm> source distillation/training_data/serve_editor.sh
python -m chord.data.generate_perturbations --config distillation/training_data/counterfactual/perturbations.yaml
stop_editor
python -m chord.data.counterfactual --config distillation/training_data/counterfactual/build.yaml
python distillation/training_data/counterfactual/build_layer.py --config distillation/training_data/counterfactual/layer.yaml
```

### 1.5 Relation rewrites (`relation_rewrites/`, GPU)

The two relation families (contradiction, causal reversal) are the ones the
student under-reproduces, and repeating the counterfactual layer's 320 rows
did not help; new parents did. The editor rewrites the 3,000 parents picked in
4d at three doses; `build_rows.py` keeps a rewrite only if it passes the
evaluation set's own validation gates (and, for contradictions, the
anchor-survival gate).

```bash
VLLM_PY=<python with vllm> source distillation/training_data/serve_editor.sh
python -m chord.data.generate_perturbations --config distillation/training_data/relation_rewrites/perturbations.yaml
stop_editor
python distillation/training_data/relation_rewrites/build_rows.py
```

### 1.6 The training corpus (`corpus/`, CPU)

`train_corpus.yaml` is one declarative recipe for `build_corpus.py`, run once
every sample set exists. It starts from the counterfactual layer and adds, in
order: whole generator documents; 1,000 cross-model splices (clean parents
with 30-50 % of their sentences replaced by sentences from 2-3 generators);
packed 512-token windows (1,000 human, up to 400 per generator: the Table-2
reference format); ELF-L at its native ~929-token length (documents and
windows); the eight AR settings; the 3,000 rewrite parents and their dose-3
rewrites. Every row is checked against the evaluation texts and the hold-out
AR sets and deduplicated.

```bash
python distillation/training_data/corpus/build_corpus.py --config distillation/training_data/corpus/train_corpus.yaml
```

### 1.7 What the trainer reads

| File (under `data/distill/training_data/`) | Written by | Used for |
|---|---|---|
| `train_corpus.jsonl` | §1.6 | the training rows |
| `counterfactual/unseen_stream.jsonl` | §1.4 | 8 rows mixed into every batch (never training rows); the PCA pool (§2) |
| `source_pool/val_{owt,wiki,reddit}.jsonl` | §1.1 | validation loss per human domain, 500 passages each (§4) |
| `counterfactual/probe_{seen,reserved}.jsonl` | §1.4 | training rows vs. never-seen rows of the same kind: the membership check that picks `best_clean/` (§4) |
| `ar_samples/samples/{gpt2xl,tinyllama}_t1.0.jsonl` | §1.3 | the hold-out alignment monitor (§4) |

## 2. Teacher targets (`teacher_features/`)

The targets are the deployed 27B encoder's features (layer −3, `coherence`
prompt, `max_length: 532`, bf16), computed in one pass over every text file the
trainer reads (`teacher_features.yaml` + `teacher_runs.jsonl`):

| File (under `data/distill/training_data/`) | Rows | Used as |
|---|---:|---|
| `train_corpus.jsonl` (`reference`) | ~26k | training targets, row-aligned with the corpus |
| `counterfactual/unseen_stream.jsonl` | ~11k | the PCA pool; targets of the rows mixed into every batch |
| `source_pool/val_{owt,wiki,reddit}.jsonl` | 500 each | validation targets |
| `ar_samples/samples/{gpt2xl,tinyllama}_t1.0.jsonl` (hold-out AR samples) | 300 each | the alignment monitor's targets |

`teacher_pca.py` then fits a 1,024-component PCA basis on `unseen_stream`
(clean training-side text, disjoint from every evaluation pool) and projects
every other file through it; the student regresses the top 256 components.
`teacher_features.slurm` runs both steps (one GPU with ≥ 60 GB).

```bash
python -m chord.featurize --config distillation/teacher_features/teacher_features.yaml
python distillation/teacher_features/teacher_pca.py fit --pool outputs/distill/features/teacher-qwen3.5-27b/unseen_stream.npy \
    --out outputs/distill/features/teacher-qwen3.5-27b-pca1024 --k 1024 \
    --project reference unseen_stream val_owt val_wiki val_reddit gpt2xl_t1.0 tinyllama_t1.0
```

Outputs: `outputs/distill/features/teacher-qwen3.5-27b/<file>.npy` (5,120-d) and
`teacher-qwen3.5-27b-pca1024/<file>.npy` (1,024-d) plus `basis.npz`; `reference` is the
training corpus.

## 3. Readout initialization (`configs/student_<s>/untrained_features.yaml`)

Each student's readout `P_S` is initialized in closed form (ridge regression
onto the teacher targets) from its UNTRAINED backbone's last-layer features on
every row of `train_corpus.jsonl`.

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

`train.slurm` resumes a requeued job from the newest `epochN/` that has its
`train_state.pt`, so it is safe on preemptible partitions.

Besides the corpus and its targets, the trainer reads the files of §1.7: the
validation sets (the validation loss picks `best/`), 8 rows of the unseen
stream per batch, and the membership probe (picks `best_clean/`, the most
membership-free checkpoint within 5% of the best validation loss). Training
writes `epochN/` checkpoints with `train_state.pt`, plus `best/`, `best_clean/` and
`final/` (merged model + `projector.pt`); `--resume DIR --resume-epochs-done N`
continues an interrupted run exactly. The reported checkpoint is `final/`.
Protocol invariants: `max_length: 532` (a 512-token body plus the reserved
prompt suffix), whitespace text normalization on every corpus, teacher read at
layer −3 with the `coherence` template, PCA basis fitted once on training-side
clean text.

## 5. Using and evaluating a student

A trained `final/` directory is used like any encoder:
`ChordScorer("qwen3.5-2b-student", model="outputs/distill/student_qwen3.5-2b/final")` (or set
`CHORD_STUDENT_MODEL`). Its Table 1 row, Table 2 column and comparison against
the teacher are computed in the experiments repository
(`experiments/student_eval/` in CHORD_experiments).
