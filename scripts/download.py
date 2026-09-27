#!/usr/bin/env python3
"""Download released CHORD distillation artifacts from the Hugging Face Hub into
this checkout, at the paths the distillation configs expect.

    python scripts/download.py --checkpoint qwen3.5-0.8b      # a trained student only
    python scripts/download.py --training-data --features     # train a student without
                                                              # rebuilding data / 27B targets
    python scripts/download.py --eval-texts                   # evaluation texts that
                                                              # build_training_data.sh excludes

--training-data   the texts train.py reads (data/distill/training_data/...): the
                  training corpus, the validation sets, the unseen stream, the
                  membership probe and the two held-out AR sample sets.
--features        the 27B teacher targets (teacher-qwen3.5-27b-pca1024/) and the untrained
                  student features that initialize the readout (untrained-<student>/).
--eval-texts      every published evaluation text (from the experiments dataset);
                  stage 1 of the training-data build excludes them
                  (distillation/training_data/excluded_eval_texts.yaml).
--checkpoint S    the trained student S (qwen3.5-2b or qwen3.5-0.8b)
                  -> outputs/distill/student_S/final.
                  Scoring does not need this: ChordScorer("qwen3.5-2b-student") pulls the
                  checkpoint from the Hub itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DISTILL_REPO = "mikezhu/chord-distill-data"  # training data, teacher targets
EXPERIMENTS_REPO = "mikezhu/chord-experiments-data"  # evaluation texts
# student key (distillation/configs/student_<key>) -> model repository
STUDENT_REPOS = {
    "qwen3.5-2b": "mikezhu/chord-qwen3.5-2b-student",
    "qwen3.5-0.8b": "mikezhu/chord-qwen3.5-0.8b-student",
}
STUDENTS = tuple(STUDENT_REPOS)

TD = "data/distill/training_data"
TRAINING_DATA = [
    f"{TD}/train_corpus.jsonl",
    f"{TD}/source_pool/val_*.jsonl",
    f"{TD}/counterfactual/unseen_stream.jsonl",
    f"{TD}/counterfactual/probe_*.jsonl",
    f"{TD}/ar_samples/samples/gpt2xl_t1.0.jsonl",
    f"{TD}/ar_samples/samples/tinyllama_t1.0.jsonl",
]
FEATURES = ["outputs/distill/features/teacher-qwen3.5-27b-pca1024/*"] + [
    f"outputs/distill/features/untrained-{s}/*" for s in STUDENTS
]
EVAL_BAN = ["data/distill/eval_ban/*.jsonl"]
EVAL_TEXTS = [
    "outputs/counterfactual/*/texts/*.jsonl",
    "outputs/counterfactual/*/texts/conditions/*.jsonl",
    "outputs/casestudy/**/*.jsonl",
    "outputs/experiments/**/*.jsonl",
]


def verify(root: Path, manifest: Path, sample: int) -> None:
    """Check the size of every downloaded file and the sha256 of a sample of them."""
    rows = list(csv.DictReader(open(manifest), delimiter="\t"))
    present = [r for r in rows if (root / r["path"]).exists()]
    bad = [r["path"] for r in present if (root / r["path"]).stat().st_size != int(r["bytes"])]
    for r in present[:: max(1, len(present) // sample)]:
        digest = hashlib.sha256((root / r["path"]).read_bytes()).hexdigest()
        if digest != r["sha256"]:
            bad.append(r["path"])
    print(f"{manifest.name}: {len(present)}/{len(rows)} files present, {len(bad)} failed")
    if bad:
        raise SystemExit("\n".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--training-data", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--eval-texts", action="store_true")
    ap.add_argument("--checkpoint", choices=STUDENTS, action="append", default=[])
    ap.add_argument("--distill-repo", default=DISTILL_REPO, help="distillation dataset id")
    ap.add_argument("--experiments-repo", default=EXPERIMENTS_REPO, help="experiments dataset id")
    ap.add_argument("--dest", default=str(ROOT), help="checkout root (default: this repository)")
    ap.add_argument("--verify-sample", type=int, default=20, help="files to sha256-check")
    args = ap.parse_args()

    dest = Path(args.dest)
    distill = (
        (TRAINING_DATA if args.training_data else [])
        + (FEATURES if args.features else [])
        + (EVAL_BAN if args.eval_texts else [])
    )
    experiments = EVAL_TEXTS if args.eval_texts else []
    if not distill and not args.checkpoint:
        ap.error("nothing to download: pass --training-data, --features, --eval-texts "
                 "and/or --checkpoint")
    for repo, part, patterns in [
        (args.distill_repo, "distill", distill),
        (args.experiments_repo, "experiments", experiments),
    ]:
        if not patterns:
            continue
        manifest = f"MANIFEST-{part}.tsv"
        snapshot_download(
            repo, repo_type="dataset", allow_patterns=patterns + [manifest], local_dir=dest
        )
        verify(dest, dest / manifest, args.verify_sample)
    for student in args.checkpoint:
        target = dest / f"outputs/distill/student_{student}/final"
        snapshot_download(STUDENT_REPOS[student], local_dir=target)
        print(f"student {student} -> {target}")


if __name__ == "__main__":
    main()
