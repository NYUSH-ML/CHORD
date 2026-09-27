"""Public API for scoring text corpora with CHORD.

CHORD embeds each passage with a frozen language model under a
coherence-oriented PromptEOL readout and compares the generated corpus with a
human reference corpus using RBF-MMD, optionally standardized against an
exchangeable human-reference null.

Quick use:

    from chord import ChordScorer

    scorer = ChordScorer("qwen3.5-27b")
    result = scorer.score(generated_texts, reference_texts)
    print(result.raw_mmd)

Command line (JSONL files with a ``text`` field): ``python -m chord --help``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .embeddings import HuggingFaceEncoder
from .metrics.distribution import median_bandwidth, rbf_kernel, squared_mmd

COHERENCE_TEMPLATE = "coherence"

# Presets mirror the paper protocol: PromptEOL coherence readout, total token
# cap 532 (512-token passage body + reserved readout suffix), bfloat16.
# The 27B/9B encoders read hidden_states[-3]; the distilled student reads its
# final hidden state through the trained readout projection (projector.pt).
MODEL_PRESETS = {
    "qwen3.5-27b": {
        "model": "Qwen/Qwen3.5-27B",
        "layer": -3,
        "model_dtype": "bfloat16",
        "max_length": 532,
        "description": "Paper-scale CHORD encoder (headline configuration).",
    },
    "qwen3.5-9b": {
        "model": "Qwen/Qwen3.5-9B",
        "layer": -3,
        "model_dtype": "bfloat16",
        "max_length": 532,
        "description": "Mid-size CHORD encoder; matches the paper's 9B rows.",
    },
    "qwen3.5-2b-student": {
        "model": "mikezhu/chord-qwen3.5-2b-student",
        "layer": "last",
        "model_dtype": "bfloat16",
        "max_length": 532,
        "description": "Distilled Qwen3.5-2B student with its readout projection (low-cost).",
    },
    "qwen3.5-0.8b-student": {
        "model": "mikezhu/chord-qwen3.5-0.8b-student",
        "layer": "last",
        "model_dtype": "bfloat16",
        "max_length": 532,
        "description": "Distilled Qwen3.5-0.8B student with its readout projection (lowest cost).",
    },
}


@dataclass(frozen=True)
class ChordScore:
    raw_mmd: float
    bandwidth: float
    n_generated: int
    n_reference: int
    model_key: str
    model: str
    z_score: float | None = None
    null_mean: float | None = None
    null_std: float | None = None


def _normalize_texts(texts: Sequence[str], name: str, mode: str) -> list[str]:
    if mode not in {"whitespace", "none"}:
        raise ValueError(f"unknown text normalization mode: {mode!r}")
    values = []
    for t in texts:
        value = str(t)
        if mode == "whitespace":
            # The paper protocol: one uniform whitespace normalization for every
            # corpus that enters a comparison (mixed raw/flattened whitespace is
            # a known score artifact for distributional metrics).
            value = " ".join(value.split())
        else:
            value = value.strip()
        if value:
            values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one non-empty text")
    return values


def _sample_rows(matrix: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(matrix) < n:
        raise ValueError("not enough rows for null calibration")
    idx = rng.choice(len(matrix), size=n, replace=False)
    return matrix[idx]


def _reference_null(
    reference: np.ndarray,
    *,
    n_left: int,
    n_right: int,
    sigma: float,
    draws: int,
    seed: int,
    block_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    kernel = rbf_kernel(sigma)
    for i in range(draws):
        left = _sample_rows(reference, n_left, rng)
        right = _sample_rows(reference, n_right, rng)
        values[i] = squared_mmd(left, right, kernel, unbiased=False, block_size=block_size)
    return values


class ChordScorer:
    """Embeds corpora with a CHORD preset and scores RBF-MMD distances.

    Parameters mirror the paper protocol; override ``model`` with a local path
    or Hugging Face repo id to use a custom encoder checkpoint.
    """

    def __init__(
        self,
        model_key: str = "qwen3.5-27b",
        *,
        model: str | None = None,
        layer: int | str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        model_dtype: str | None = None,
        max_length: int | None = None,
        batch_size: int = 8,
        text_normalization: str = "whitespace",
        trust_remote_code: bool = False,
    ) -> None:
        if model_key not in MODEL_PRESETS:
            raise ValueError(
                f"unknown model_key {model_key!r}; choose one of {sorted(MODEL_PRESETS)}"
            )
        preset = dict(MODEL_PRESETS[model_key])
        if model is None:
            model = preset["model"]
            if model_key.endswith("-student"):
                # a local checkpoint directory also works, e.g.
                # outputs/distill/student_qwen3.5-2b/final
                model = os.environ.get("CHORD_STUDENT_MODEL") or model
        protocol = {
            "backend": "huggingface",
            "model": model,
            "max_length": int(max_length or preset["max_length"]),
            "pooling": "prompteol",
            "prompteol_template": COHERENCE_TEMPLATE,
            "layer": preset["layer"] if layer is None else layer,
            "normalization": "none",
            "model_dtype": model_dtype or preset["model_dtype"],
            "cache_dtype": "float32",
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            protocol["cache_dir"] = cache_dir
        if device is not None:
            protocol["device"] = device
        self.model_key = model_key
        self.model = model
        self.batch_size = int(batch_size)
        self.text_normalization = text_normalization
        self.encoder = HuggingFaceEncoder(protocol)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = _normalize_texts(texts, "texts", self.text_normalization)
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            chunks.append(self.encoder.encode(texts[start : start + self.batch_size]))
        return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)

    def score_embeddings(
        self,
        generated_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
        *,
        null_reference_embeddings: np.ndarray | None = None,
        bandwidth: float | None = None,
        null_draws: int = 0,
        seed: int = 17,
        block_size: int = 512,
    ) -> ChordScore:
        gen = np.asarray(generated_embeddings, dtype=np.float64)
        ref = np.asarray(reference_embeddings, dtype=np.float64)
        fit_ref = (
            ref
            if null_reference_embeddings is None
            else np.asarray(null_reference_embeddings, dtype=np.float64)
        )
        sigma = float(bandwidth or median_bandwidth(fit_ref, max_samples=2000, seed=seed))
        raw = float(squared_mmd(gen, ref, rbf_kernel(sigma), unbiased=False, block_size=block_size))
        z = mean = std = None
        if null_draws:
            n = min(len(gen), len(ref), len(fit_ref) // 2)
            null = _reference_null(
                fit_ref,
                n_left=n,
                n_right=n,
                sigma=sigma,
                draws=int(null_draws),
                seed=seed,
                block_size=block_size,
            )
            mean = float(null.mean())
            std = float(null.std())
            z = float((raw - mean) / max(std, 1e-12))
        return ChordScore(
            raw_mmd=raw,
            bandwidth=sigma,
            n_generated=len(gen),
            n_reference=len(ref),
            model_key=self.model_key,
            model=self.model,
            z_score=z,
            null_mean=mean,
            null_std=std,
        )

    def score(
        self,
        generated: Sequence[str],
        reference: Sequence[str],
        *,
        null_reference: Sequence[str] | None = None,
        bandwidth: float | None = None,
        null_draws: int = 0,
        seed: int = 17,
        block_size: int = 512,
    ) -> ChordScore:
        gen_emb = self.embed(generated)
        ref_emb = self.embed(reference)
        null_emb = self.embed(null_reference) if null_reference is not None else None
        return self.score_embeddings(
            gen_emb,
            ref_emb,
            null_reference_embeddings=null_emb,
            bandwidth=bandwidth,
            null_draws=null_draws,
            seed=seed,
            block_size=block_size,
        )


def score_chord(
    generated: Sequence[str],
    reference: Sequence[str],
    *,
    model_key: str = "qwen3.5-27b",
    **kwargs,
) -> ChordScore:
    """One-call convenience wrapper: CHORD(generated_corpus, reference_corpus)."""
    scorer_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key
        in {
            "model",
            "layer",
            "cache_dir",
            "device",
            "model_dtype",
            "max_length",
            "batch_size",
            "text_normalization",
            "trust_remote_code",
        }
    }
    return ChordScorer(model_key, **scorer_kwargs).score(generated, reference, **kwargs)


def read_jsonl_texts(
    path: str | Path, text_key: str = "text", limit: int | None = None
) -> list[str]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line)[text_key])
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a generated corpus with CHORD.")
    parser.add_argument("--generated", required=True, help="JSONL file with a text field")
    parser.add_argument("--reference", required=True, help="JSONL file with a text field")
    parser.add_argument("--null-reference", default=None, help="larger human pool for the null")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-key", choices=sorted(MODEL_PRESETS), default="qwen3.5-27b")
    parser.add_argument("--model", default=None, help="override checkpoint path or repo id")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dtype", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--text-normalization", choices=["whitespace", "none"], default="whitespace"
    )
    parser.add_argument("--null-draws", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    layer: int | str | None
    if args.layer is None:
        layer = None
    else:
        try:
            layer = int(args.layer)
        except ValueError:
            layer = args.layer
    scorer = ChordScorer(
        args.model_key,
        model=args.model,
        layer=layer,
        cache_dir=args.cache_dir,
        device=args.device,
        model_dtype=args.model_dtype,
        max_length=args.max_length,
        batch_size=args.batch_size,
        text_normalization=args.text_normalization,
        trust_remote_code=args.trust_remote_code,
    )
    generated = read_jsonl_texts(args.generated, args.text_key, args.limit)
    reference = read_jsonl_texts(args.reference, args.text_key, args.limit)
    null_reference = (
        read_jsonl_texts(args.null_reference, args.text_key, args.limit)
        if args.null_reference
        else None
    )
    result = scorer.score(
        generated,
        reference,
        null_reference=null_reference,
        null_draws=args.null_draws,
        seed=args.seed,
        block_size=args.block_size,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
