from __future__ import annotations

import math
import random
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...utils.hashing import sha256_text, stable_seed
from ...utils.records import PassageRecord
from ...text import sentence_spans, words

ARTICLES_RE = re.compile(
    r"\b(?:a|an|the|this|that|these|those|each|every|some|any)\b(?:\s+)?",
    flags=re.IGNORECASE,
)
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
FUNCTION_WORD_RE = re.compile(
    r"\b(?:a|an|the|this|that|these|those|and|or|but|if|because|"
    r"of|to|in|on|at|for|from|with|by)\b",
    flags=re.IGNORECASE,
)
AGREEMENT_RE = re.compile(
    r"\b(?P<subject>he|she|it|this|that|they|we|these|those)\s+"
    r"(?P<verb>is|are|has|have|does|do)\b",
    flags=re.IGNORECASE,
)

IRREGULAR_INFLECTIONS = {
    "is": "are",
    "are": "is",
    "was": "were",
    "were": "was",
    "has": "have",
    "have": "has",
    "does": "do",
    "do": "does",
    "goes": "go",
    "go": "goes",
    "says": "say",
    "say": "says",
    "children": "child",
    "child": "children",
    "people": "person",
    "person": "people",
    "men": "man",
    "man": "men",
    "women": "woman",
    "woman": "women",
}

AGREEMENT_REPLACEMENTS = {
    "is": "are",
    "are": "is",
    "has": "have",
    "have": "has",
    "does": "do",
    "do": "does",
}


def _preserve_case(before: str, after: str) -> str:
    if before.isupper():
        return after.upper()
    if before[:1].isupper():
        return after.capitalize()
    return after


def _inflect_badly(token: str) -> Optional[str]:
    lower = token.lower()
    if lower in IRREGULAR_INFLECTIONS:
        return _preserve_case(token, IRREGULAR_INFLECTIONS[lower])
    if lower.endswith("ies") and len(lower) > 4:
        return _preserve_case(token, lower[:-3] + "y")
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        return _preserve_case(token, lower[:-1])
    if lower.endswith("ed") and len(lower) > 4:
        return _preserve_case(token, lower[:-2])
    if lower.endswith("ing") and len(lower) > 5:
        return _preserve_case(token, lower[:-3] + "s")
    if len(lower) > 3:
        return _preserve_case(token, lower + "s")
    return None


def _span(start: int, end: int, before: str, after: str) -> Dict[str, Any]:
    return {"start": start, "end": end, "before": before, "after": after}


def apply_changed_spans(text: str, changed_spans: Sequence[Dict[str, Any]]) -> str:
    result = text
    for span in sorted(changed_spans, key=lambda item: (item["start"], item["end"]), reverse=True):
        start = int(span["start"])
        end = int(span["end"])
        before = str(span["before"])
        if text[start:end] != before:
            raise ValueError(f"changed span does not match source at {start}:{end}")
        result = result[:start] + str(span["after"]) + result[end:]
    return result


def _choose_count(total: int, rate: float) -> int:
    if total == 0 or rate <= 0:
        return 0
    return min(total, max(1, int(math.ceil(total * rate))))


def _position_filter(
    spans: Sequence[Tuple[int, int]], text_length: int, position: Optional[str]
) -> List[Tuple[int, int]]:
    # `None` (unspecified) and the explicit `"uniform"` both keep every eligible
    # span; the caller then draws edit locations uniformly at random via
    # rng.sample, so aggregated over the corpus the edit position is uniform over
    # the passage (no fixed / end-append placement). The early/middle/late thirds
    # remain available as a controlled position sweep for diagnostics.
    if position is None or position == "uniform":
        return list(spans)
    thirds = {"early": (0.0, 1.0 / 3), "middle": (1.0 / 3, 2.0 / 3), "late": (2.0 / 3, 1.01)}
    if position not in thirds:
        raise ValueError(f"unknown position: {position}")
    low, high = thirds[position]
    selected = []
    for start, end in spans:
        midpoint = ((start + end) / 2) / max(text_length, 1)
        if low <= midpoint < high:
            selected.append((start, end))
    return selected


def article_delete(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    matches = list(ARTICLES_RE.finditer(text))
    eligible = _position_filter(
        [(match.start(), match.end()) for match in matches], len(text), position
    )
    selected = rng.sample(eligible, _choose_count(len(eligible), rate))
    spans = [_span(start, end, text[start:end], "") for start, end in selected]
    return apply_changed_spans(text, spans), spans, len(selected) / max(len(eligible), 1)


def inflection_error(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    eligible: List[Tuple[int, int, str]] = []
    allowed_positions = _position_filter(
        [(match.start(), match.end()) for match in WORD_RE.finditer(text)], len(text), position
    )
    allowed = set(allowed_positions)
    for match in WORD_RE.finditer(text):
        replacement = _inflect_badly(match.group())
        if replacement and (match.start(), match.end()) in allowed:
            eligible.append((match.start(), match.end(), replacement))
    selected = rng.sample(eligible, _choose_count(len(eligible), rate))
    spans = [_span(start, end, text[start:end], after) for start, end, after in selected]
    return apply_changed_spans(text, spans), spans, len(selected) / max(len(eligible), 1)


def agreement_error(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    matches = list(AGREEMENT_RE.finditer(text))
    eligible_positions = set(
        _position_filter(
            [(match.start("verb"), match.end("verb")) for match in matches],
            len(text),
            position,
        )
    )
    eligible = [
        match for match in matches if (match.start("verb"), match.end("verb")) in eligible_positions
    ]
    selected = rng.sample(eligible, _choose_count(len(eligible), rate))
    spans = []
    for match in selected:
        before = match.group("verb")
        after = _preserve_case(before, AGREEMENT_REPLACEMENTS[before.lower()])
        spans.append(_span(match.start("verb"), match.end("verb"), before, after))
    return apply_changed_spans(text, spans), spans, len(selected) / max(len(eligible), 1)


def function_word_repeat(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    matches = list(FUNCTION_WORD_RE.finditer(text))
    eligible_positions = set(
        _position_filter([(match.start(), match.end()) for match in matches], len(text), position)
    )
    eligible = [match for match in matches if (match.start(), match.end()) in eligible_positions]
    selected = rng.sample(eligible, _choose_count(len(eligible), rate))
    spans = [_span(match.end(), match.end(), "", " " + match.group()) for match in selected]
    return apply_changed_spans(text, spans), spans, len(selected) / max(len(eligible), 1)


def case_lower(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    del rate, rng, position
    output = text.lower()
    if output == text:
        return text, [], 0.0
    return output, [_span(0, len(text), text, output)], 1.0


def local_shuffle(
    text: str,
    rate: float,
    rng: random.Random,
    position: Optional[str] = None,
    window_size: int = 4,
) -> Tuple[str, List[Dict[str, Any]], float]:
    all_sentence_spans = sentence_spans(text)
    eligible_sentences = _position_filter(all_sentence_spans, len(text), position)
    candidates: List[List[re.Match]] = []
    for start, end in eligible_sentences:
        matches = list(WORD_RE.finditer(text, start, end))
        if len(matches) >= 2:
            candidates.append(matches)
    selected = rng.sample(candidates, _choose_count(len(candidates), rate))
    changed: List[Dict[str, Any]] = []
    for matches in selected:
        width = min(window_size, len(matches))
        window_start = rng.randrange(0, len(matches) - width + 1)
        window = matches[window_start : window_start + width]
        original = [match.group() for match in window]
        shuffled = original[:]
        for _ in range(10):
            rng.shuffle(shuffled)
            if shuffled != original:
                break
        if shuffled == original:
            shuffled = original[1:] + original[:1]
        for match, replacement in zip(window, shuffled):
            if match.group() != replacement:
                changed.append(_span(match.start(), match.end(), match.group(), replacement))
    return apply_changed_spans(text, changed), changed, len(selected) / max(len(candidates), 1)


def phrase_repeat(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    eligible = _position_filter(sentence_spans(text), len(text), position)
    eligible = [
        span for span in eligible if len(list(WORD_RE.finditer(text, span[0], span[1]))) >= 3
    ]
    selected = rng.sample(eligible, _choose_count(len(eligible), rate))
    changed: List[Dict[str, Any]] = []
    repeated_words = 0
    for start, end in selected:
        matches = list(WORD_RE.finditer(text, start, end))
        width = min(5, max(2, len(matches) // 3))
        phrase_start_index = rng.randrange(0, len(matches) - width + 1)
        phrase_matches = matches[phrase_start_index : phrase_start_index + width]
        phrase_start = phrase_matches[0].start()
        phrase_end = phrase_matches[-1].end()
        phrase = text[phrase_start:phrase_end]
        changed.append(_span(phrase_end, phrase_end, "", " " + phrase))
        repeated_words += width
    output = apply_changed_spans(text, changed)
    realized = repeated_words / max(len(words(output)), 1)
    return output, changed, realized


def sentence_repeat(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    all_spans = sentence_spans(text)
    eligible = _position_filter(all_spans, len(text), position)
    selected = rng.sample(eligible, _choose_count(len(eligible), rate))
    changed = [_span(end, end, "", " " + text[start:end]) for start, end in selected]
    return apply_changed_spans(text, changed), changed, len(selected) / max(len(all_spans), 1)


def sentence_permute(
    text: str, rate: float, rng: random.Random, position: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    # position-aware: reorder only sentences whose span falls in the target third
    # (was previously `del position` -> ignored position entirely). In-place span
    # edits (not a normalize+rejoin) keep inter-sentence whitespace faithful.
    spans = sentence_spans(text)
    if len(spans) < 2:
        return text, [], 0.0
    if position is None or position == "uniform":
        eligible = list(range(len(spans)))
    else:
        in_third = set(_position_filter(spans, len(text), position))
        eligible = [i for i, span in enumerate(spans) if span in in_third]
    if len(eligible) < 2:
        return text, [], 0.0
    count = max(2, _choose_count(len(eligible), rate))
    chosen = sorted(rng.sample(eligible, min(count, len(eligible))))
    originals = [text[spans[i][0] : spans[i][1]] for i in chosen]
    shuffled = originals[:]
    for _ in range(10):
        rng.shuffle(shuffled)
        if shuffled != originals:
            break
    if shuffled == originals:
        shuffled = shuffled[1:] + shuffled[:1]
    changed = [
        _span(spans[i][0], spans[i][1], original, replacement)
        for i, original, replacement in zip(chosen, originals, shuffled)
        if original != replacement
    ]
    return apply_changed_spans(text, changed), changed, len(chosen) / len(spans)


RULES: Dict[str, Callable[..., Tuple[str, List[Dict[str, Any]], float]]] = {
    "article_delete": article_delete,
    "inflection_error": inflection_error,
    "agreement_error": agreement_error,
    "function_word_repeat": function_word_repeat,
    "case_lower": case_lower,
    "local_shuffle": local_shuffle,
    "phrase_repeat": phrase_repeat,
    "sentence_repeat": sentence_repeat,
    "sentence_permute": sentence_permute,
}


def generate_rule_perturbation(
    record: PassageRecord,
    perturbation: str,
    severity: str,
    rate: float,
    global_seed: int,
    position: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> PassageRecord:
    if perturbation not in RULES:
        raise ValueError(f"unknown rule perturbation: {perturbation}")
    seed = stable_seed(global_seed, record.sample_id, perturbation, severity, position)
    rng = random.Random(seed)
    kwargs = dict(options or {})
    output, spans, realized = RULES[perturbation](
        record.clean_text, rate, rng, position=position, **kwargs
    )
    suffix = f":{position}" if position else ""
    return replace(
        record,
        parent_sample_id=record.sample_id,
        text_hash=sha256_text(output),
        perturbed_text=output,
        perturbation=perturbation,
        severity=severity,
        requested_rate=rate,
        realized_rate=realized,
        generator="rule",
        generator_revision="rules-v1",
        seed=seed,
        validation={},
        changed_spans=spans,
        position=position,
        sample_id=f"{record.sample_id}:{perturbation}:{severity}{suffix}",
    )
