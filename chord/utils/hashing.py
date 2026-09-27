from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .config import stable_json


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def stable_seed(*parts: Any) -> int:
    digest = stable_hash(parts)
    return int(digest[:16], 16) & 0x7FFFFFFF


def seeded_rng(*parts: Any) -> np.random.Generator:
    """Create a repeatable NumPy stream from ordered experiment identifiers."""

    seed = [int(hashlib.sha256(str(p).encode()).hexdigest()[:8], 16) for p in parts]
    return np.random.default_rng(seed)
