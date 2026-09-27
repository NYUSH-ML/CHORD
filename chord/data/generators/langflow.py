"""Generate normalized samples with the official LangFlow implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import torch
    from langflow import LangFlow, LangFlowConfig
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = LangFlowConfig.from_pretrained(os.path.join(repo, "langflow"))
    model = LangFlow(config)
    model.load_state_dict(load_file(args.checkpoint, device=str(device)))
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8") as handle, torch.no_grad():
        while written < args.num_samples:
            batch = min(args.batch_size, args.num_samples - written)
            token_ids = model.generate_samples(
                num_samples=batch,
                seq_length=args.length,
                num_steps=args.steps,
                device=device,
            )
            texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            for text in texts:
                text = " ".join(text.strip().split())
                if not text:
                    continue
                row = {
                    "sample_id": f"{args.run_id}:{written:06d}",
                    "text": text,
                    "run_id": args.run_id,
                    "model": "langflow-owt",
                    "seed": args.seed,
                    "steps": args.steps,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} samples to {destination}")


if __name__ == "__main__":
    main()
