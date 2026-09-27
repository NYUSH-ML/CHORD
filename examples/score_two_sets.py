"""Score one set of texts against another with CHORD.

    python examples/score_two_sets.py                            # distilled Qwen3.5-0.8B
    python examples/score_two_sets.py --model-key qwen3.5-27b    # the paper's encoder
    python examples/score_two_sets.py --generated gen.jsonl --reference human.jsonl \
        --null-reference human_pool.jsonl --null-draws 200

The bundled files are 12 short passages (texts/human.jsonl) and the same passages
with their sentences shuffled (texts/shuffled.jsonl). They only show the calls:
a real comparison uses a few hundred texts per side (the paper uses 500) and, for
a z-score, a larger human pool disjoint from the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chord import MODEL_PRESETS, ChordScorer
from chord.api import read_jsonl_texts

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generated", default=str(HERE / "texts/shuffled.jsonl"))
    ap.add_argument("--reference", default=str(HERE / "texts/human.jsonl"))
    ap.add_argument("--null-reference", default=None, help="larger human pool for the null")
    ap.add_argument("--null-draws", type=int, default=0)
    ap.add_argument("--model-key", choices=sorted(MODEL_PRESETS), default="qwen3.5-0.8b-student")
    ap.add_argument("--model", default=None, help="local checkpoint directory or Hub id")
    args = ap.parse_args()

    scorer = ChordScorer(args.model_key, model=args.model)
    generated = read_jsonl_texts(args.generated)
    reference = read_jsonl_texts(args.reference)
    null_reference = read_jsonl_texts(args.null_reference) if args.null_reference else None

    # Embed once, then score: the same embeddings can be reused for other comparisons.
    gen, ref = scorer.embed(generated), scorer.embed(reference)
    null = scorer.embed(null_reference) if null_reference else None
    result = scorer.score_embeddings(
        gen, ref, null_reference_embeddings=null, null_draws=args.null_draws
    )
    print(f"encoder      {scorer.model}")
    print(f"texts        {result.n_generated} generated vs {result.n_reference} reference")
    print(f"CHORD (MMD²) {result.raw_mmd:.4f}   bandwidth {result.bandwidth:.3f}")
    if result.z_score is not None:
        print(f"z vs null    {result.z_score:.2f}")


if __name__ == "__main__":
    main()
