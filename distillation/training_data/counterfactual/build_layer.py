#!/usr/bin/env python3
"""Stage 4d: the counterfactual training layer, the membership probe, the parents
of the relation rewrites and the unseen stream, from the training-side
counterfactual set (stage 4c) and the split's unused pool.

* Probe reserve: a seeded 10 % of the parents is held out ENTIRELY (their clean
  text and every edit of them stay out of training) and forms the probe's
  "reserved" side; the "seen" side is drawn from the trained parents. Both
  sides mix clean parents and benign paraphrases in equal parts, so a
  membership probe cannot separate them by editor style.
* Layer: every trained parent (``clean_parent``), up to ``per_family`` edits
  of each harmful family (``pert_<family>``) and every benign paraphrase of a
  trained parent (``benign_paraphrase``). Edits that leave the text unchanged
  are skipped; rows are deduplicated.
* Relation-rewrite parents: from the unused-pool passages that are not in the
  layer (shuffled), the first ``per_domain`` per domain that meet the length
  rules, written as the PassageRecord manifest the editor reads (stage 5).
* Unseen stream: every other unused-pool passage (``unseen_stream.jsonl``):
  text the student never trains on, mixed into every batch as a guard against
  memorizing the corpus, and the pool the teacher's PCA basis is fitted on.

    python distillation/training_data/counterfactual/build_layer.py \\
        --config distillation/training_data/counterfactual/layer.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, key, norm, read_jsonl, write_jsonl  # noqa: E402

from chord.text import sentence_spans, words  # noqa: E402
from chord.utils.hashing import sha256_text  # noqa: E402
from chord.utils.records import PassageRecord  # noqa: E402


def rewrite_parents(rows: List[Dict], spec: Dict) -> tuple:
    """Pick the relation-rewrite parents; return (manifest records, leftover rows)."""
    picked: List[Dict] = []
    rest: List[Dict] = []
    per_domain: Counter = Counter()
    for row in rows:
        text, domain = row["text"], row["domain"]
        n_words, n_sents = len(words(text)), len(sentence_spans(text))
        if (
            per_domain[domain] >= spec["per_domain"]
            or n_sents < spec["min_sentences"]
            or not spec["min_words"] <= n_words <= spec["max_words"]
        ):
            rest.append(row)
            continue
        per_domain[domain] += 1
        pid = f"relgen-{domain}-{per_domain[domain]:04d}"
        record = PassageRecord(
            sample_id=pid,
            source_document_id=pid,
            split="test",
            role="candidate",
            text_hash=sha256_text(text),
            clean_text=text,
            corpus_version="relation-rewrites-v1",
            generator="corpus",
            domain=domain,
            word_count=n_words,
            sentence_count=n_sents,
        )
        picked.append(record.to_dict())
    return picked, rest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    rng = random.Random(int(cfg["seed"]))
    texts = ROOT / cfg["set_dir"] / "texts"

    # ---- parents and the probe reserve ------------------------------------
    parent_text = {r["parent_id"]: r["text"] for r in read_jsonl(texts / "clean_candidates.jsonl")}
    pids = sorted(parent_text)
    reserved = set(rng.sample(pids, max(1, round(cfg["probe_reserve_frac"] * len(pids)))))
    trained = [p for p in pids if p not in reserved]

    # ---- edits of trained parents, by family ------------------------------
    harmful = defaultdict(list)
    benign: Dict[str, List[str]] = defaultdict(list)
    reserved_benign: List[str] = []
    for path in sorted((texts / "conditions").glob("*.jsonl")):
        family = path.stem.split("__")[0]
        is_benign = family == "benign_paraphrase"
        if not is_benign and family not in cfg["harmful_families"]:
            continue
        for r in read_jsonl(path):
            pid = r.get("parent_id")
            if pid not in parent_text or norm(r["text"]) == norm(parent_text[pid]):
                continue  # not a training parent, or a no-op edit
            if is_benign and pid in reserved:
                reserved_benign.append(r["text"])
            elif is_benign:
                benign[pid].append(r["text"])
            elif pid not in reserved:
                harmful[family].append(r["text"])

    # ---- the layer ----------------------------------------------------------
    layer: List[Dict] = []
    seen: set = set()

    def add(text: str, domain: str, kind: str) -> None:
        k = key(text)
        if k not in seen:
            seen.add(k)
            layer.append({"text": text, "domain": domain, "kind": kind})

    for pid in trained:
        add(parent_text[pid], pid.split("-")[1], "clean_parent")
    for family in sorted(harmful):
        pool = harmful[family]
        rng.shuffle(pool)
        for text in pool[: cfg["per_family"]]:
            add(text, "perturbed", f"pert_{family}")
    for pid in sorted(benign):
        for text in benign[pid]:
            add(text, "perturbed", "benign_paraphrase")
    rng.shuffle(layer)

    # ---- relation-rewrite parents and the unseen stream -------------------
    unused = [
        {"text": r["text"], "domain": r["domain"]}
        for r in read_jsonl(ROOT / cfg["unused_pool"])
        if key(r["text"]) not in seen
    ]
    rng.shuffle(unused)
    parents, unseen = rewrite_parents(unused, cfg["rewrite_parents"])

    # ---- membership probe ---------------------------------------------------
    half = cfg["probe_per_side"] // 2
    trained_benign = [t for pid in sorted(benign) for t in benign[pid]]
    reserved_parents = [parent_text[p] for p in sorted(reserved)]
    probe_seen = rng.sample([parent_text[p] for p in trained], half) + rng.sample(
        trained_benign, half
    )
    probe_reserved = rng.sample(reserved_parents, min(half, len(reserved_parents))) + rng.sample(
        reserved_benign, min(half, len(reserved_benign))
    )

    out = cfg["outputs"]
    write_jsonl(ROOT / out["layer"], layer)
    write_jsonl(ROOT / out["rewrite_parents"], parents)
    write_jsonl(ROOT / out["probe_seen"], ({"text": t} for t in probe_seen))
    write_jsonl(ROOT / out["probe_reserved"], ({"text": t} for t in probe_reserved))
    write_jsonl(ROOT / out["unseen_stream"], unseen)  # last: its presence marks the step done
    print(f"parents: {len(trained)} trained, {len(reserved)} probe-reserved")
    print(f"layer: {len(layer)} rows {dict(Counter(r['kind'] for r in layer))}")
    print(f"relation-rewrite parents: {len(parents)} {dict(Counter(r['domain'] for r in parents))}")
    print(f"unseen stream: {len(unseen)}")
    print(f"probe: seen {len(probe_seen)}, reserved {len(probe_reserved)}")


if __name__ == "__main__":
    main()
