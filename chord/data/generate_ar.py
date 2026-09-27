"""Unconditional autoregressive (GPT-2) sampler for the OWT contrast lane.

GPT-2 is the fair AR baseline for an OpenWebText benchmark: OpenWebText is the
open reproduction of GPT-2's WebText training set, and the diffusion-LM papers
themselves benchmark against GPT-2. Use a model DISTINCT from the gen-PPL scorer
(gpt2-large) so AR samples are not graded by their own model.

Writes `{"text", "sample_id"}` JSONL compatible with `io.load_texts`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..utils.hashing import sha256_text


def generate(config_path: Path) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_name = str(cfg["model"])
    n_samples = int(cfg.get("n_samples", 48))
    max_new = int(cfg.get("max_new_tokens", 512))
    min_new = int(cfg.get("min_new_tokens", max_new))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 0.95))
    repetition_penalty = float(cfg.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(cfg.get("no_repeat_ngram_size", 0))
    batch_size = int(cfg.get("batch_size", 8))
    seed = int(cfg.get("seed", 20260621))
    cache_dir = cfg.get("cache_dir")
    output_path = (config_path.parent / cfg["output_path"]).resolve()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()

    texts: list[str] = []
    done = 0
    while done < n_samples:
        take = min(batch_size, n_samples - done)
        torch.manual_seed(seed + done)
        start = torch.full((take, 1), tok.bos_token_id, dtype=torch.long, device=device)
        attn = torch.ones_like(start)
        with torch.no_grad():
            out = model.generate(
                start,
                attention_mask=attn,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                pad_token_id=tok.eos_token_id,
            )
        for row in out:
            text = tok.decode(row[1:], skip_special_tokens=True).strip()
            if text:
                texts.append(text)
        done += take
        print(f"generated {len(texts)}/{n_samples}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            sample_id = sha256_text(f"{model_name}:{seed}:{index}")[:16]
            handle.write(
                json.dumps({"sample_id": sample_id, "text": text}, ensure_ascii=False) + "\n"
            )
    print(f"wrote {len(texts)} AR samples ({model_name}, {device}) to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate(Path(args.config).resolve())


if __name__ == "__main__":
    main()
