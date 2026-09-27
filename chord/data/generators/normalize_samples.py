"""Normalize official model outputs into one JSONL sample schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List


def _jsonl(path: Path, key: str) -> Iterable[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(key)
            if isinstance(value, str):
                yield value


def _text(path: Path, separator: str | None) -> Iterable[str]:
    value = path.read_text(encoding="utf-8")
    if separator:
        yield from value.split(separator)
        return
    # LangFlow's official ``--- Sample N ---`` output format.
    chunks = re.split(r"(?m)^--- Sample \d+ ---\s*$", value)
    if len(chunks) > 1:
        yield from chunks[1:]
    else:
        yield from value.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["jsonl", "text"], required=True)
    parser.add_argument("--input-key", default="generated")
    parser.add_argument("--separator")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    source = Path(args.input)
    texts = (
        list(_jsonl(source, args.input_key))
        if args.format == "jsonl"
        else list(_text(source, args.separator))
    )
    cleaned: List[str] = []
    for text in texts:
        value = " ".join(text.strip().split())
        if value:
            cleaned.append(value)
    if not cleaned:
        raise ValueError(f"no non-empty samples parsed from {source}")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(cleaned):
            row = {
                "sample_id": f"{args.run_id}:{index:06d}",
                "text": text,
                "model": args.model,
                "run_id": args.run_id,
                "seed": args.seed,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(cleaned)} samples to {destination}")


if __name__ == "__main__":
    main()
