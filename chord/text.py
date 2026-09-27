from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
TOKEN_RE = re.compile(r"\b[\w'-]+\b|[^\w\s]", flags=re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def sentences(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [part.strip() for part in SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    cursor = 0
    for sentence in sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            continue
        end = start + len(sentence)
        result.append((start, end))
        cursor = end
    return result


def looks_like_english(text: str, threshold: float = 0.85) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin = sum("LATIN" in unicodedata.name(char, "") for char in letters)
    return latin / len(letters) >= threshold
