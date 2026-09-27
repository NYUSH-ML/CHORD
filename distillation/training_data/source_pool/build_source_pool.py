#!/usr/bin/env python3
"""Stage 1: the clean three-domain passage pool and the validation sets.

Streams three public Hugging Face datasets at pinned revisions, shard by shard
in sorted order (so the stream, and therefore the output, is deterministic):

    owt     Skylion007/openwebtext        web / news
    wiki    wikimedia/wikipedia 20231101  encyclopedic
    reddit  trl-lib/tldr                  informal (the POST body of each prompt)

Each document is cut to a lead passage at sentence boundaries (Wikipedia and
web documents run far past the window) and admitted when it

  * is viable: English, >= 4 sentences, 100-400 words (the rule the corpus
    split applies downstream, so nothing is discarded later);
  * is new: not a duplicate of an admitted passage;
  * is not evaluation text: neither equal to, nor sharing a 13-word n-gram
    with, any text listed in ``excluded_eval_texts.yaml``.

Per domain the first ``per_domain + val_per_domain`` admitted passages are
shuffled with the seed; ``val_per_domain`` of them become that domain's
validation set, the rest go to the pool.

    python distillation/training_data/source_pool/build_source_pool.py \\
        --config distillation/training_data/source_pool/source_pool.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import NGRAM, eval_text_keys, eval_texts, key, ngrams, norm, write_jsonl  # noqa: E402

from chord.text import looks_like_english, sentence_spans, words  # noqa: E402


def _h64(gram: str) -> int:
    return int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little")


def _hashes(text: str) -> np.ndarray:
    return np.fromiter((_h64(g) for g in ngrams(text)), dtype=np.uint64)


def ngram_ban() -> np.ndarray:
    """Sorted, unique uint64 hashes of every 13-gram of every evaluation text."""
    parts = []
    for i, text in enumerate(eval_texts(), 1):
        parts.append(_hashes(text))
        if i % 20000 == 0:
            print(f"[source] hashed {i} evaluation texts", flush=True)
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)


def shares_ngram(text: str, banned: np.ndarray) -> bool:
    grams = _hashes(text)
    if not len(grams):
        return False
    pos = np.searchsorted(banned, grams)
    pos[pos == len(banned)] = 0
    return bool((banned[pos] == grams).any())


def lead_passage(text: str, target_words: int, max_words: int) -> str:
    """Cut an over-length document at sentence boundaries; shorter text passes."""
    text = norm(text)
    if len(text.split()) <= max_words:
        return text
    end, n = 0, 0
    for a, b in sentence_spans(text):
        w = len(text[a:b].split())
        if n + w > target_words and n >= 100:
            break
        end, n = b, n + w
    return text[:end].strip()


def reddit_post(prompt: str) -> str:
    """The POST body of a trl-lib/tldr prompt (SUBREDDIT/TITLE/POST scaffold)."""
    body = prompt.split("POST:", 1)[-1] if "POST:" in prompt else prompt
    return body.split("TL;DR:", 1)[0].strip()


def stream_dataset(spec: Dict) -> Iterator[str]:
    """Rows of a pinned HF dataset, parquet shard by shard in sorted order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    repo, rev = spec["repo"], spec["revision"]
    shards = sorted(
        f
        for f in list_repo_files(repo, repo_type="dataset", revision=rev)
        if f.startswith(spec.get("prefix", "")) and f.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"{repo}@{rev}: no parquet shards under {spec.get('prefix', '')!r}")
    for shard in shards:
        path = hf_hub_download(repo, shard, repo_type="dataset", revision=rev)
        for batch in pq.ParquetFile(path).iter_batches(columns=[spec["field"]], batch_size=1024):
            for value in batch.column(0).to_pylist():
                if value:
                    yield value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    f = cfg["filters"]
    need = int(cfg["per_domain"]) + int(cfg["val_per_domain"])

    exact = eval_text_keys()
    banned = ngram_ban()
    print(f"[source] ban: {len(exact)} evaluation texts, {len(banned)} {NGRAM}-grams", flush=True)

    def viable(text: str) -> bool:
        n = len(words(text))
        return (
            f["min_words"] <= n <= f["max_words"]
            and len(sentence_spans(text)) >= f["min_sentences"]
            and looks_like_english(text)
        )

    seen: set = set()
    pool: List[Dict] = []
    report: Dict[str, Dict] = {}
    for di, (domain, spec) in enumerate(cfg["domains"].items()):
        admitted: List[str] = []
        counts = {"scanned": 0, "not_viable": 0, "duplicate": 0, "eval_exact": 0, "eval_ngram": 0}
        for raw in stream_dataset(spec):
            if len(admitted) >= need or counts["scanned"] >= cfg["scan_cap"]:
                break
            counts["scanned"] += 1
            text = reddit_post(raw) if spec.get("extract") == "reddit_post" else raw
            text = lead_passage(text, f["lead_target_words"], f["max_words"])
            if not text or not viable(text):
                counts["not_viable"] += 1
                continue
            k = key(text)
            if k in seen:
                counts["duplicate"] += 1
                continue
            if k in exact:
                counts["eval_exact"] += 1
                continue
            if shares_ngram(text, banned):
                counts["eval_ngram"] += 1
                continue
            seen.add(k)
            admitted.append(text)
        if len(admitted) < need:
            raise SystemExit(f"{domain}: admitted {len(admitted)} < {need}; raise scan_cap")
        random.Random(int(cfg["seed"]) + di).shuffle(admitted)
        val, rest = admitted[: cfg["val_per_domain"]], admitted[cfg["val_per_domain"] :]
        write_jsonl(
            ROOT / cfg["val_dir"] / f"val_{domain}.jsonl",
            ({"text": t, "domain": domain} for t in val),
        )
        pool += [{"text": t, "domain": domain} for t in rest]
        report[domain] = {**counts, "pool": len(rest), "val": len(val)}
        print(f"[source] {domain}: {report[domain]}", flush=True)

    write_jsonl(ROOT / cfg["output"], pool)
    report_path = ROOT / cfg["output"]
    report_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(f"[source] pool: {len(pool)} passages -> {cfg['output']}", flush=True)


if __name__ == "__main__":
    main()
