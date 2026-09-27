"""DLM-splice perturbation family — realistic "locally fluent, globally incoherent"
failures.

Motivation: LLM-authored logic edits are "too well-formed" (a careful human inserting
one clean error). Real low-quality generation is locally okay-fluent but globally
incoherent, with no discourse structure. We synthesize that by replacing k sentences of
a real passage with sentences drawn from a sentence pool:

  - pool = real DLM output  -> HARMFUL   (genuine generative incoherence)
  - pool = other real human passages -> matched CONTROL (topic break, but human
    fluency) -> isolates "DLM-specific incoherence" from "mere topic discontinuity"

Properties: deterministic / reproducible, NO LLM API, position-controlled by
construction (so no end-append confound), and span-reconstructable. Severity = fraction
of sentences replaced. Position = which third the replaced sentences are drawn from.

Inserted sentences are normalized (capitalized, terminal punctuation) so the splice is a
clean sentence-boundary swap; the incoherence comes from content, not broken markup.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...text import looks_like_english, sentence_spans, sentences
from .rules import _choose_count, _span, apply_changed_spans

POSITION_THIRDS = {
    "early": (0.0, 1.0 / 3),
    "middle": (1.0 / 3, 2.0 / 3),
    "late": (2.0 / 3, 1.01),
}


def _normalize_inserted(sentence: str) -> str:
    """Make an inserted sentence a drop-in replacement: collapse whitespace, capitalize,
    ensure terminal punctuation. Keeps the splice clean at sentence boundaries."""
    s = " ".join(sentence.split())
    if not s:
        return s
    if s[0].islower():
        s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s = s + "."
    return s


def build_sentence_pool(
    texts: Sequence[str],
    min_chars: int = 20,
    max_chars: int = 300,
    exclude: Optional[set] = None,
) -> List[str]:
    """Sentence-level pool from raw texts, filtered + deduped + sorted for
    deterministic sampling. `exclude` removes specific normalized sentences (e.g. to
    avoid drawing from the passage being perturbed)."""
    exclude = exclude or set()
    pool = set()
    for text in texts:
        for sent in sentences(text):
            sent = sent.strip()
            if not (min_chars <= len(sent) <= max_chars):
                continue
            if not looks_like_english(sent):
                continue
            norm = _normalize_inserted(sent)
            if norm and norm not in exclude:
                pool.add(norm)
    return sorted(pool)


def _eligible_sentence_indices(
    spans: Sequence[Tuple[int, int]], text_length: int, position: Optional[str]
) -> List[int]:
    # `None` and the explicit `"uniform"` keep every sentence eligible; the caller
    # draws replacement positions uniformly at random, so splice edits land
    # uniformly over the passage. The thirds stay available as a position sweep.
    if position is None or position == "uniform":
        return list(range(len(spans)))
    if position not in POSITION_THIRDS:
        raise ValueError(f"unknown position: {position}")
    low, high = POSITION_THIRDS[position]
    out = []
    for i, (start, end) in enumerate(spans):
        midpoint = ((start + end) / 2) / max(text_length, 1)
        if low <= midpoint < high:
            out.append(i)
    return out


def splice_sentences(
    text: str,
    sentence_pool: Sequence[str],
    rate: float,
    rng: random.Random,
    position: Optional[str] = None,
    min_sentences: int = 2,
) -> Tuple[str, List[Dict[str, Any]], float]:
    """Replace a `rate` fraction of sentences (within the target `position` third) with
    sentences sampled from `sentence_pool`. Returns (output, changed_spans, realized_rate).

    Returns the text UNCHANGED (empty spans, rate 0.0) when there is nothing valid to do
    — too few sentences, an empty target third, or an empty sentence pool. Callers should
    skip such no-op records rather than emit them (avoids the dataset-polluting empties
    that the rule path historically produced)."""
    all_spans = sentence_spans(text)
    if len(all_spans) < min_sentences or not sentence_pool:
        return text, [], 0.0
    eligible = _eligible_sentence_indices(all_spans, len(text), position)
    if not eligible:
        return text, [], 0.0
    count = _choose_count(len(eligible), rate)
    if count <= 0:
        return text, [], 0.0
    chosen = sorted(rng.sample(eligible, count))
    changed: List[Dict[str, Any]] = []
    for idx in chosen:
        start, end = all_spans[idx]
        inserted = sentence_pool[rng.randrange(len(sentence_pool))]
        if inserted == text[start:end]:  # degenerate self-swap; skip this sentence
            continue
        changed.append(_span(start, end, text[start:end], inserted))
    if not changed:
        return text, [], 0.0
    output = apply_changed_spans(text, changed)
    realized = len(changed) / max(len(all_spans), 1)
    return output, changed, realized
