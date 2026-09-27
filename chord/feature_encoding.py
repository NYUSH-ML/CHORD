"""Batched text encoding with a single cached model shared across corpora."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .embeddings import HuggingFaceEncoder, _l2_normalize, encode_texts

# Keep one encoder resident; release it before loading another model.
_ENCODER_CACHE: Dict[str, object] = {"key": None, "encoder": None}


def get_cached_encoder(protocol):
    import json as _json

    key = _json.dumps(protocol, sort_keys=True, default=str)
    if _ENCODER_CACHE["key"] != key:
        # Free the previous model's GPU memory BEFORE constructing the next one.
        # Otherwise the new (multi-GB) weights load while the old model is still
        # resident -> transient 2x footprint -> OOM when switching between large
        # backbones (e.g. two Qwen3-30B MoE encoders in one featurize run).
        if _ENCODER_CACHE["encoder"] is not None:
            _ENCODER_CACHE["encoder"] = None
            _ENCODER_CACHE["key"] = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        _ENCODER_CACHE["encoder"] = HuggingFaceEncoder(protocol)
        _ENCODER_CACHE["key"] = key
    return _ENCODER_CACHE["encoder"]


def embed_batched(texts, protocol, batch_size: int = 256) -> np.ndarray:
    """Chunk a forward pass to bound GPU memory.

    Build the encoder ONCE per protocol and reuse it across batches AND across
    repeated calls (cached) — `encode_texts` would reconstruct (reload weights to
    GPU) on every batch, which is fine for tiny encoders but ruinous for the
    multi-GB LLM backbones this probe uses.
    """
    if protocol.get("backend", "huggingface") != "huggingface":
        parts = [
            encode_texts(texts[start : start + batch_size], protocol)
            for start in range(0, len(texts), batch_size)
        ]
        return np.concatenate(parts, axis=0).astype(np.float64)
    encoder = get_cached_encoder(protocol)
    parts = []
    for start in range(0, len(texts), batch_size):
        parts.append(encoder.encode(texts[start : start + batch_size]))
    matrix = np.concatenate(parts, axis=0)
    if protocol.get("normalization", "none") == "l2":
        matrix = _l2_normalize(matrix)
    return matrix.astype(np.float64)
