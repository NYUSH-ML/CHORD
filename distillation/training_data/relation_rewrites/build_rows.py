#!/usr/bin/env python3
"""Turn the editor's raw output into corpus rows, with the evaluation set's own gates.

A rewrite is kept only if ``chord.data.perturbations.validation.validate_record``
passes and, for the anchored contradiction family, the quoted anchor sentence
survives unchanged in the rewritten passage (``build_corpus.anchor_survives``,
the gate that rejects local fact flips). Writes
``data/distill/training_data/relation_rewrites/rewrite_rows.jsonl`` in the corpus row schema
``{text, domain, kind, source, parent}``: rewrites as ``pert_contradiction`` /
``pert_causal_reverse`` (``domain: perturbed``, ``source: doseK``) and each
parent once as ``clean_parent`` (its own text domain, ``source: relgen``).

    python distillation/training_data/relation_rewrites/build_rows.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chord.data.counterfactual import anchor_survives  # noqa: E402
from chord.data.perturbations.validation import validate_record  # noqa: E402
from chord.utils.records import PassageRecord  # noqa: E402

DATA_DIR = ROOT / "data/distill/training_data/relation_rewrites"
RAW_PERTURBATIONS = ROOT / "data/distill/training_data/relation_rewrites/llm_edits.jsonl"
KIND = {"internal_contradiction": "pert_contradiction", "causal_reverse": "pert_causal_reverse"}


def normalize(text: str) -> str:
    return " ".join(text.split())


def dose_label(record: PassageRecord) -> str:
    if record.requested_rate:
        return f"dose{int(record.requested_rate)}"
    dose = record.sample_id.split(":")[-1] if ":" in record.sample_id else record.severity
    return f"dose{dose}"


def main() -> None:
    parents: Dict[str, Dict] = {
        row["sample_id"]: row
        for row in map(json.loads, open(DATA_DIR / "parents_manifest.jsonl", encoding="utf-8"))
    }
    raw = [json.loads(line) for line in open(RAW_PERTURBATIONS, encoding="utf-8")]

    kept: List[Dict] = []
    dropped: collections.Counter = collections.Counter()
    seen: set = set()
    for payload in raw:
        record = PassageRecord.from_dict(payload)
        family = record.perturbation
        if family not in KIND:
            dropped["other_family"] += 1
            continue
        if not validate_record(record, require_operation_evidence=True).get("passed"):
            dropped[f"{family}:validate"] += 1
            continue
        if family == "internal_contradiction" and not anchor_survives(
            str(record.validation.get("operation_evidence", "")), record.text, record.clean_text
        ):
            dropped["contradiction:anchor"] += 1
            continue
        text = normalize(record.text)
        if text in seen or text == normalize(record.clean_text):
            dropped[f"{family}:dup"] += 1
            continue
        seen.add(text)
        kept.append(
            {
                "text": text,
                "domain": "perturbed",
                "kind": KIND[family],
                "source": dose_label(record),
                "parent": record.parent_sample_id,
            }
        )
    for parent_id, parent in parents.items():
        kept.append(
            {
                "text": normalize(parent["clean_text"]),
                "domain": parent["domain"],
                "kind": "clean_parent",
                "source": "relgen",
                "parent": parent_id,
            }
        )

    out = DATA_DIR / "rewrite_rows.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("kept:", collections.Counter((row["kind"], row["source"]) for row in kept))
    print("dropped:", dict(dropped))
    print(f"wrote {len(kept)} rows -> {out}")


if __name__ == "__main__":
    main()
