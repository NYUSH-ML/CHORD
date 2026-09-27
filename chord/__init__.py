"""CHORD: a coherence-aware corpus-level distance for open-ended text generation.

Public entry points:

    from chord import ChordScorer, score_chord

    result = score_chord(generated_texts, reference_texts, model_key="qwen3.5-27b")
"""

from .api import MODEL_PRESETS, ChordScore, ChordScorer, score_chord

__all__ = ["ChordScore", "ChordScorer", "MODEL_PRESETS", "score_chord"]
