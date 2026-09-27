from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Sequence

from ...utils.records import PassageRecord
from ...text import looks_like_english, tokens, words
from .rules import apply_changed_spans

INSTRUCTION_MARKERS = (
    "system prompt",
    "as an ai language model",
    "<|im_start|>",
    "<|assistant|>",
)
PARAPHRASE_PERTURBATIONS = {
    "para_lexical",
    "para_syntactic",
    "para_discourse",
}
LLM_ERROR_PERTURBATIONS = {
    "broken_transition",
    "topic_drift",
    "internal_contradiction",
    "contradiction",
    "causal_reverse",
    "key_omission",
    "agreement_error_llm",
    "tense_inconsistency",
    "pronoun_reference_error",
}
NO_ERROR_MARKERS = {
    "n/a",
    "na",
    "no error",
    "no error introduced",
    "none",
    "none.",
    "not applicable",
}


def _introduced_error_is_documented(record: PassageRecord) -> bool:
    value = str(record.validation.get("introduced_error", "")).strip().lower()
    if bool(value) and value not in NO_ERROR_MARKERS:
        return True
    # current editor prompt documents the edit under operation_evidence instead
    return bool(str(record.validation.get("operation_evidence", "")).strip())


def token_edit_ratio(before: str, after: str) -> float:
    before_tokens = tokens(before)
    after_tokens = tokens(after)
    matcher = SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)
    changed = sum(
        max(before_end - before_start, after_end - after_start)
        for operation, before_start, before_end, after_start, after_end in matcher.get_opcodes()
        if operation != "equal"
    )
    return changed / max(len(before_tokens), 1)


def validate_record(
    record: PassageRecord,
    min_length_ratio: float = 0.5,
    max_length_ratio: float = 1.75,
    edit_ratio_bounds: Optional[Sequence[float]] = None,
    enforce_edit_ratio_bounds: bool = False,
    require_operation_evidence: bool = False,
) -> Dict[str, Any]:
    text = record.text
    checks: Dict[str, Any] = {
        "non_empty": bool(text.strip()),
        "english": looks_like_english(text),
        "no_instruction_leakage": not any(marker in text.lower() for marker in INSTRUCTION_MARKERS),
    }
    clean_length = max(len(words(record.clean_text)), 1)
    ratio = len(words(text)) / clean_length
    checks["length_ratio"] = ratio
    checks["bounded_length_ratio"] = min_length_ratio <= ratio <= max_length_ratio
    if record.perturbation != "clean":
        checks["text_changed"] = record.text != record.clean_text
        checks["has_documented_change"] = bool(record.changed_spans)
        checks["token_edit_ratio"] = token_edit_ratio(record.clean_text, record.text)
        if edit_ratio_bounds is not None:
            low, high = (float(value) for value in edit_ratio_bounds)
            checks["bounded_token_edit_ratio"] = low <= checks["token_edit_ratio"] <= high
    if record.perturbation != "clean":
        try:
            reconstructed = apply_changed_spans(record.clean_text, record.changed_spans)
            checks["span_reconstruction"] = reconstructed == text
        except (KeyError, TypeError, ValueError):
            checks["span_reconstruction"] = False
    if record.generator == "rule":
        checks["has_documented_change"] = checks["has_documented_change"] and (
            record.requested_rate == 0 or bool(record.changed_spans) or record.realized_rate == 0
        )
    elif record.perturbation != "clean":
        if require_operation_evidence:
            checks["operation_evidence_documented"] = bool(
                str(record.validation.get("operation_evidence", "")).strip()
            )
        if record.perturbation in PARAPHRASE_PERTURBATIONS:
            checks["preserved_facts_documented"] = bool(
                record.validation.get("preserved_facts")
            ) or bool(str(record.validation.get("operation_evidence", "")).strip())
        if record.perturbation in LLM_ERROR_PERTURBATIONS:
            checks["introduced_error_documented"] = _introduced_error_is_documented(record)
    checks["passed"] = all(
        value is True
        for key, value in checks.items()
        if key
        not in {
            "length_ratio",
            "token_edit_ratio",
            "passed",
            *(set() if enforce_edit_ratio_bounds else {"bounded_token_edit_ratio"}),
        }
    )
    return checks
