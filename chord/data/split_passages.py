"""Prepare the 3-domain (OWT + Wikipedia + Reddit) corpus split for the
comprehensive meta-evaluation set.

This is the SINGLE source of clean human passages for the whole evaluation set. It
carves three DISJOINT, domain-balanced pools from a labeled multi-domain corpus
(default ``train_diverse.jsonl``: owt/wiki/reddit) and writes them in the layout
``build_corpus.py`` and ``generate_perturbations.py`` consume:

  * ``clean_candidates.jsonl`` — the parent passages that every family perturbs.
    Each row carries a STABLE ``parent_id`` (``mev-<domain>-<n>``); the LLM
    families reuse this exact id as their ``parent_sample_id`` (via the manifest
    below) so all families share one matched parent set.
  * ``reference.jsonl`` — the human reference pool for the exchangeable null,
    domain-balanced and DISJOINT from the parents (a cross-domain human-vs-human
    null; a metric cannot win by merely detecting OWT-vs-non-OWT topic shift).
  * ``unused_pool.jsonl`` — every viable passage not taken as a parent or
    reference, DISJOINT from both (the evaluation set splices its sentences
    into the parents for the human-splice control and the corpus-mix family,
    with no self-splice leakage).
  * ``corpus_manifest.jsonl`` — the parents again, as ``PassageRecord`` rows
    (``role=candidate``), so ``generate_perturbations`` edits exactly these
    passages and stamps ``parent_sample_id == parent_id``.

Balance is exact: ``parents_total`` / ``reference_total`` are distributed across
the requested domains as evenly as possible (remainder to the first domains).
Pools are disjoint by construction (sequential slices of one deterministic
per-domain shuffle) and additionally verified before writing. A note is logged
that ``train_diverse`` is a distillation TRAIN pool — harmless for the
frozen 8B/27B headline metric, but in-distribution should the distilled student
ever be scored on this evaluation set.

    python -m chord.data.split_passages \
        --config distillation/training_data/counterfactual/split.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import yaml

from ..utils.hashing import sha256_text
from ..utils.records import PassageRecord
from ..text import (
    looks_like_english,
    normalize_whitespace,
    sentence_spans,
    words,
)


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def _distribute(total: int, n_bins: int) -> List[int]:
    """Split ``total`` into ``n_bins`` as-even-as-possible non-negative counts;
    the remainder goes to the first bins (deterministic)."""
    base, rem = divmod(int(total), int(n_bins))
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")


def _viable(text: str, *, min_sentences: int, min_words: int, max_words: int) -> bool:
    # NB no is_boilerplate_or_code() here: the source is an already-curated
    # multi-domain corpus, and that check is designed for RAW multi-line web text
    # — on whitespace-normalized single-line passages it false-positives on any
    # ``;``/``{``/``}`` (it flagged 100% of Reddit). English + sentence-count +
    # length filters are sufficient for this vetted corpus.
    if not text or not looks_like_english(text):
        return False
    if len(sentence_spans(text)) < min_sentences:
        return False
    return min_words <= len(words(text)) <= max_words


def run(config_path: str) -> Dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    root = Path(cfg.get("repo_root", "."))
    seed = int(cfg.get("seed", 20260701))
    source = _resolve(root, cfg["source"])
    domains: List[str] = list(cfg.get("domains", ["owt", "wiki", "reddit"]))
    min_sentences = int(cfg.get("min_sentences", 4))
    min_words = int(cfg.get("min_words", 120))
    max_words = int(cfg.get("max_words", 320))
    parents_total = int(cfg.get("parents_total", 500))
    reference_total = int(cfg.get("reference_total", 3600))
    unused_cap_per_domain = cfg.get("unused_cap_per_domain")  # optional

    out_dir = _resolve(root, cfg["output_dir"])
    manifest_path = _resolve(root, cfg["manifest"])
    corpus_version = str(cfg.get("corpus_version", "meta_eval_3domain-v1"))

    # ---- load + filter, grouped by domain, deduped globally by normalized text -
    by_domain: Dict[str, List[str]] = {d: [] for d in domains}
    seen: set = set()
    n_raw = 0
    for line in open(source):
        if not line.strip():
            continue
        rec = json.loads(line)
        n_raw += 1
        dom = rec.get("domain")
        text = normalize_whitespace(rec.get("text", ""))
        if dom not in by_domain:
            continue
        key = sha256_text(text)
        if key in seen:
            continue
        if not _viable(text, min_sentences=min_sentences, min_words=min_words, max_words=max_words):
            continue
        seen.add(key)
        by_domain[dom].append(text)

    parents_per = _distribute(parents_total, len(domains))
    reference_per = _distribute(reference_total, len(domains))

    parents: List[Dict] = []
    reference: List[Dict] = []
    unused: List[Dict] = []
    per_domain_report: Dict[str, Dict] = {}

    for di, dom in enumerate(domains):
        pool = sorted(by_domain[dom])  # stable order before the seeded shuffle
        random.Random(seed + di).shuffle(pool)
        need_p, need_r = parents_per[di], reference_per[di]
        if len(pool) < need_p + need_r:
            raise SystemExit(
                f"domain {dom!r}: only {len(pool)} viable passages, need "
                f"{need_p} parents + {need_r} reference (+ an unused pool). Lower "
                f"parents_total/reference_total or relax the filters."
            )
        p_slice = pool[:need_p]
        r_slice = pool[need_p : need_p + need_r]
        u_slice = pool[need_p + need_r :]
        if unused_cap_per_domain is not None:
            u_slice = u_slice[: int(unused_cap_per_domain)]
        for n, t in enumerate(p_slice):
            parents.append({"parent_id": f"mev-{dom}-{n:04d}", "text": t, "domain": dom})
        for n, t in enumerate(r_slice):
            reference.append({"parent_id": f"ref-{dom}-{n:04d}", "text": t, "domain": dom})
        for n, t in enumerate(u_slice):
            unused.append({"parent_id": f"unused-{dom}-{n:04d}", "text": t, "domain": dom})
        per_domain_report[dom] = {
            "viable": len(pool),
            "parents": len(p_slice),
            "reference": len(r_slice),
            "unused": len(u_slice),
        }

    # ---- disjointness guard (by normalized text) -----------------------------
    p_keys = {sha256_text(r["text"]) for r in parents}
    r_keys = {sha256_text(r["text"]) for r in reference}
    u_keys = {sha256_text(r["text"]) for r in unused}
    assert not (p_keys & r_keys), "parents/reference overlap"
    assert not (p_keys & u_keys), "parents/unused-pool overlap"
    assert not (r_keys & u_keys), "reference/unused-pool overlap"

    # ---- write text pools ----------------------------------------------------
    _write_jsonl(out_dir / "clean_candidates.jsonl", parents)
    _write_jsonl(out_dir / "reference.jsonl", reference)
    _write_jsonl(out_dir / "unused_pool.jsonl", unused)

    # ---- write the PassageRecord manifest for generate_perturbations ---------
    # parents only (role=candidate, split=test); parent_sample_id will equal
    # sample_id downstream, so LLM-family parent_ids match clean_candidates.
    manifest_rows = []
    for r in parents:
        text = r["text"]
        manifest_rows.append(
            PassageRecord(
                sample_id=r["parent_id"],
                source_document_id=r["parent_id"],
                split="test",
                role="candidate",
                text_hash=sha256_text(text),
                clean_text=text,
                corpus_version=corpus_version,
                generator="corpus",
                domain=r["domain"],
                word_count=len(words(text)),
                sentence_count=len(sentence_spans(text)),
            ).to_dict()
        )
    _write_jsonl(manifest_path, manifest_rows)

    summary = {
        "source": str(source),
        "n_raw": n_raw,
        "seed": seed,
        "domains": domains,
        "min_sentences": min_sentences,
        "min_words": min_words,
        "max_words": max_words,
        "parents_total": len(parents),
        "reference_total": len(reference),
        "unused_total": len(unused),
        "per_domain": per_domain_report,
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "note": (
            "train_diverse is a distillation TRAIN pool; harmless "
            "for the frozen 8B/27B headline metric, in-distribution only if "
            "the distilled student is later scored on this evaluation set."
        ),
    }
    (out_dir / "corpus_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[prep] {n_raw} raw rows -> viable/parents/reference/unused per domain:")
    for dom in domains:
        d = per_domain_report[dom]
        print(
            f"         {dom}: viable={d['viable']} parents={d['parents']} "
            f"reference={d['reference']} unused={d['unused']}"
        )
    print(
        f"[prep] TOTAL parents={len(parents)} reference={len(reference)} "
        f"unused={len(unused)} (disjoint verified)"
    )
    print(f"[prep] WROTE {out_dir}/ + {manifest_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
