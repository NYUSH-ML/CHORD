"""File I/O shared by the library: JSONL / JSON records, generation-run manifests,
text corpora and CSV tables."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from .hashing import sha256_text


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output = ensure_parent(path)
    fd, temp_name = tempfile.mkstemp(prefix=output.name, dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temp_name, output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return list(read_jsonl(source))
    if source.suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, list) else [value]
    if source.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet input requires the 'report' extra") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    raise ValueError(f"unsupported file format: {source}")

@dataclass(frozen=True)
class Run:
    run_id: str
    model: str
    family: str
    seed: int
    samples_path: Path
    text_key: str
    setting: Dict[str, Any]
    metadata: Dict[str, Any]


def load_runs(path: str | Path, *, require_samples: bool = True) -> List[Run]:
    manifest = Path(path).resolve()
    runs: List[Run] = []
    seen = set()
    for row in read_jsonl(manifest):
        run_id = str(row["run_id"])
        if run_id in seen:
            raise ValueError(f"duplicate run_id in {manifest}: {run_id}")
        seen.add(run_id)
        samples_path = Path(row["samples_path"])
        if not samples_path.is_absolute():
            samples_path = (manifest.parent / samples_path).resolve()
        has_samples = samples_path.is_file() and samples_path.stat().st_size > 0
        if not has_samples and require_samples:
            raise FileNotFoundError(f"{run_id}: missing samples file {samples_path}")
        if not has_samples:
            continue
        runs.append(
            Run(
                run_id=run_id,
                model=str(row["model"]),
                family=str(row.get("family", "unknown")),
                seed=int(row.get("seed", 0)),
                samples_path=samples_path,
                text_key=str(row.get("text_key", "text")),
                setting=dict(row.get("setting", {})),
                metadata=dict(row.get("metadata", {})),
            )
        )
    if not runs and require_samples:
        raise ValueError(f"no runs found in {manifest}")
    return runs


def normalize_texts(texts: Iterable[str], mode: str = "none") -> List[str]:
    """Apply an explicit corpus-wide surface normalization.

    Distributional metrics are highly sensitive to format mismatches.  Keep
    normalization opt-in so structured prompts (for example QA with newlines)
    are not changed accidentally, while allowing one policy to be applied to
    every reference and candidate corpus in a real-generator comparison.
    """
    values = list(texts)
    if mode in {"none", ""}:
        return values
    if mode == "whitespace":
        return [" ".join(value.split()) for value in values]
    raise ValueError(f"unknown text normalization mode: {mode}")


def load_texts(path: str | Path, text_key: str = "text", normalization: str = "none") -> List[str]:
    source = Path(path)
    output: List[str] = []
    for line_number, row in enumerate(read_jsonl(source), 1):
        value = row.get(text_key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}:{line_number}: {text_key!r} must be a non-empty string")
        output.append(value.strip())
    if not output:
        raise ValueError(f"no texts found in {source}")
    return normalize_texts(output, normalization)


def texts_hash(texts: Iterable[str]) -> str:
    return sha256_text("\n".join(sha256_text(text) for text in texts))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
