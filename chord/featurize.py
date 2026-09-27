"""Cache reference and real-DLM features for every evaluation representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .embeddings import protocol_payload
from .feature_encoding import embed_batched
from .utils.io import load_runs, load_texts, texts_hash


def _cache_one(
    texts: List[str],
    protocol: Dict,
    output: Path,
    *,
    batch_size: int,
    overwrite: bool,
) -> None:
    meta_path = output.with_suffix(".json")
    expected = {
        "text_hash": texts_hash(texts),
        "count": len(texts),
        "protocol": protocol_payload(protocol),
    }
    if output.is_file() and meta_path.is_file() and not overwrite:
        if json.loads(meta_path.read_text(encoding="utf-8")) == expected:
            print(f"[real-dlm feat] cached: {output}", flush=True)
            return
    matrix = embed_batched(texts, protocol, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(matrix, dtype=np.float32))
    meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"[real-dlm feat] {output}: {matrix.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-encoder")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-runs", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # a config without `runs_manifest` featurizes only its reference texts
    runs = (
        load_runs(
            (config_path.parent / cfg["runs_manifest"]).resolve(),
            require_samples=not args.allow_missing_runs,
        )
        if cfg.get("runs_manifest")
        else []
    )
    reference_path = (config_path.parent / cfg["reference"]["path"]).resolve()
    normalization = str(cfg.get("text_normalization", "none"))
    reference = load_texts(
        reference_path,
        cfg["reference"].get("text_key", "text"),
        normalization,
    )
    limit = int(cfg["reference"].get("limit", len(reference)))
    reference = reference[:limit]
    output_dir = (config_path.parent / cfg["output_dir"]).resolve()

    for encoder in cfg["encoders"]:
        name = encoder["name"]
        if args.only_encoder and args.only_encoder != name:
            continue
        protocol = dict(encoder["protocol"])
        batch_size = int(encoder.get("batch_size", cfg.get("batch_size", 32)))
        _cache_one(
            reference,
            protocol,
            output_dir / "features" / name / "reference.npy",
            batch_size=batch_size,
            overwrite=args.overwrite,
        )
        for run in runs:
            texts = load_texts(run.samples_path, run.text_key, normalization)
            _cache_one(
                texts,
                protocol,
                output_dir / "features" / name / f"{run.run_id}.npy",
                batch_size=batch_size,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
