"""Helpers shared by the training-data stages: JSONL I/O, the dedup key and the
evaluation ban list (``excluded_eval_texts.yaml``)."""

from __future__ import annotations

import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_EVAL_TEXTS = Path(__file__).resolve().parent / "excluded_eval_texts.yaml"
NGRAM = 13
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Whitespace-collapsed text."""
    return _WS.sub(" ", text).strip()


def key(text: str) -> str:
    """Dedup / ban key: whitespace-collapsed, lowercased sha256."""
    return hashlib.sha256(norm(text).lower().encode()).hexdigest()


def ngrams(text: str, n: int = NGRAM) -> Iterator[str]:
    toks = norm(text).lower().split()
    for i in range(len(toks) - n + 1):
        yield " ".join(toks[i : i + n])


def read_jsonl(path: Path | str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path | str, rows: Iterable[Dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def eval_text_files(config: Path = EXCLUDED_EVAL_TEXTS) -> List[Path]:
    patterns = yaml.safe_load(config.read_text())["globs"]
    files = sorted(
        {Path(p) for pat in patterns for p in glob.glob(str(ROOT / pat), recursive=True)}
    )
    if not files:
        raise SystemExit(f"{config}: no evaluation files found; download the data bundle first")
    return files


def eval_texts(config: Path = EXCLUDED_EVAL_TEXTS) -> Iterator[str]:
    for path in eval_text_files(config):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    text = json.loads(line).get("text")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if isinstance(text, str) and text.strip():
                    yield text


def eval_text_keys(config: Path = EXCLUDED_EVAL_TEXTS) -> Set[str]:
    """Exact keys of every evaluation text."""
    return {key(t) for t in eval_texts(config)}
