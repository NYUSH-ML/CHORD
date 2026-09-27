"""Text utilities, perturbation rules and file I/O used to build corpora (CPU)."""

import random

from chord.data.perturbations.rules import RULES, apply_changed_spans, generate_rule_perturbation
from chord.text import sentence_spans, sentences
from chord.utils.hashing import sha256_text
from chord.utils.io import load_texts, read_jsonl, write_jsonl
from chord.utils.records import PassageRecord

TEXT = (
    "The bridge opened in 1932. It carried trains until the war. "
    "After 1945 it was rebuilt for cars. Today it is a footpath. "
    "Tourists cross it every summer."
)


def _record() -> PassageRecord:
    return PassageRecord(
        sample_id="p0",
        source_document_id="d0",
        split="test",
        role="parent",
        text_hash=sha256_text(TEXT),
        clean_text=TEXT,
    )


def test_sentence_split_and_spans_agree():
    spans = sentence_spans(TEXT)
    assert len(spans) == len(sentences(TEXT)) == 5
    assert [TEXT[a:b] for a, b in spans] == sentences(TEXT)


def test_sentence_permute_keeps_sentences_changes_order():
    out, spans, realized = RULES["sentence_permute"](TEXT, 1.0, random.Random(0))
    assert out != TEXT
    assert sorted(sentences(out)) == sorted(sentences(TEXT))
    assert apply_changed_spans(TEXT, spans) == out
    assert realized == 1.0


def test_rule_perturbation_is_deterministic():
    a = generate_rule_perturbation(_record(), "local_shuffle", "high", 0.5, global_seed=7)
    b = generate_rule_perturbation(_record(), "local_shuffle", "high", 0.5, global_seed=7)
    assert a.perturbed_text == b.perturbed_text != TEXT
    assert a.parent_sample_id == "p0" and a.text_hash == sha256_text(a.perturbed_text)


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    rows = [{"text": " a  b "}, {"text": "c"}]
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows
    assert load_texts(path, normalization="whitespace") == ["a b", "c"]
