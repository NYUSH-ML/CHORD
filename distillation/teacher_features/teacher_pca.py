"""Reduce the teacher's PromptEOL features to the student's width (PCA).

The 27B teacher reads d=5120, far wider than the student's readout. This module
fits an orthonormal PCA basis on a pool of teacher features and projects every
teacher target through it; `distillation/training/train.py` then regresses the student's
readout P_S onto the leading `bridge.rank` coordinates (256 for the released
2B student).

Projecting the 27B layer-62 embedding to 1024 dimensions is nearly free: it
keeps 38/40 counterfactual detections, the full relation axis, and case-study ladders
that are pointwise close to the teacher's. Dimension is not the bottleneck for
this distillation — do not spend effort widening the projector.

The basis pool must be CLEAN TRAINING-SIDE text that is disjoint from every
evaluation corpus (the released recipe uses `unseen_stream`), so the projection
carries no evaluation-set information.

Two subcommands:

    # 1. fit a basis on one feature file and project a set of files through it
    python distillation/teacher_features/teacher_pca.py fit \\
        --pool   outputs/distill/features/teacher-qwen3.5-27b/unseen_stream.npy \\
        --out    outputs/distill/features/teacher-qwen3.5-27b-pca1024 \\
        --k 1024 \\
        --project reference unseen_stream val_owt val_wiki val_reddit \\
                  gpt2xl_t1.0 tinyllama_t1.0

    # 2. project one more file through an existing basis (same teacher + readout!)
    python distillation/teacher_features/teacher_pca.py project \\
        --basis  outputs/distill/features/teacher-qwen3.5-27b-pca1024/basis.npz \\
        --src    <teacher feature .npy> --out <projected .npy>

Refitting the basis on a different pool silently changes every target and makes
runs incomparable; fit it once per teacher protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _fit(pool: np.ndarray, k: int):
    mu = pool.mean(axis=0)
    _, s, vt = np.linalg.svd(pool - mu, full_matrices=False)
    var = s**2
    if k > vt.shape[0]:
        raise SystemExit(f"--k {k} exceeds the feature rank {vt.shape[0]}")
    retained = float(var[:k].sum() / var.sum())
    return mu, vt[:k].T, retained


def _apply(src: Path, mu: np.ndarray, proj: np.ndarray, dst: Path) -> None:
    x = np.load(src).astype(np.float64)
    if x.shape[1] != mu.shape[0]:
        raise SystemExit(
            f"{src.name}: feature width {x.shape[1]} != basis width {mu.shape[0]} "
            "— the basis was fitted on a different teacher/readout"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, ((x - mu) @ proj).astype("float32"))
    print(f"[pca] {src.name}: {x.shape} -> ({x.shape[0]}, {proj.shape[1]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a PCA basis and project files through it")
    f.add_argument(
        "--pool",
        required=True,
        help="feature .npy the basis is fitted on (clean training-side text, "
        "disjoint from every eval corpus)",
    )
    f.add_argument("--out", required=True, help="output directory for basis.npz + projected files")
    f.add_argument("--k", type=int, default=1024, help="target dimension (default: 1024)")
    f.add_argument(
        "--project",
        nargs="*",
        default=None,
        help="stem names to project, resolved next to --pool as <stem>.npy. "
        "Default: every .npy in the pool's directory.",
    )

    p = sub.add_parser("project", help="project one file through an existing basis")
    p.add_argument("--basis", required=True, help="basis.npz written by `fit`")
    p.add_argument("--src", required=True, help="teacher feature .npy to project")
    p.add_argument("--out", required=True, help="destination .npy")

    args = ap.parse_args()

    if args.cmd == "fit":
        pool_path = Path(args.pool)
        out_dir = Path(args.out)
        pool = np.load(pool_path).astype(np.float64)
        mu, proj, retained = _fit(pool, args.k)
        print(
            f"[pca] basis on {pool.shape} from {pool_path.name}, k={args.k}, "
            f"retained variance = {retained:.4f}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "basis.npz", mu=mu, proj=proj, retained_var=retained)
        if args.project is None:
            srcs = sorted(pool_path.parent.glob("*.npy"))
        else:
            srcs = [pool_path.parent / f"{stem}.npy" for stem in args.project]
        for src in srcs:
            if not src.is_file():
                raise SystemExit(f"missing teacher feature file: {src}")
            _apply(src, mu, proj, out_dir / src.name)
    else:
        b = np.load(args.basis)
        print(f"[pca] reusing basis (retained_var={float(b['retained_var']):.4f})")
        _apply(Path(args.src), b["mu"], b["proj"], Path(args.out))


if __name__ == "__main__":
    main()
