from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PassageRecord:
    sample_id: str
    source_document_id: str
    split: str
    role: str
    text_hash: str
    clean_text: str
    parent_sample_id: Optional[str] = None
    corpus_version: str = "unknown"
    normalization_version: str = "whitespace-v1"
    perturbed_text: Optional[str] = None
    perturbation: str = "clean"
    severity: str = "none"
    requested_rate: float = 0.0
    realized_rate: float = 0.0
    generator: str = "corpus"
    generator_revision: str = "whitespace-v1"
    prompt_version: Optional[str] = None
    seed: int = 0
    validation: Dict[str, Any] = field(default_factory=dict)
    changed_spans: List[Dict[str, Any]] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    topic: Optional[str] = None
    domain: Optional[str] = None
    near_duplicate_group: Optional[str] = None
    position: Optional[str] = None

    @property
    def text(self) -> str:
        return self.perturbed_text if self.perturbed_text is not None else self.clean_text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PassageRecord":
        known = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in known if key in value})
