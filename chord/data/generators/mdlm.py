"""Generate normalized samples from the official HuggingFace MDLM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=False)  # interface symmetry
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--predictor", default="ddpm_cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.predictor != "ddpm_cache":
        raise ValueError("the standalone MDLM adapter currently supports ddpm_cache")

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from .flash_compat import install_if_missing

    install_if_missing()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    # accept either a local snapshot directory or a Hugging Face repo id
    checkpoint = args.checkpoint
    if Path(checkpoint).exists():
        checkpoint = str(Path(checkpoint).resolve())
    model = (
        AutoModelForMaskedLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    mask_index = int(model.config.vocab_size) - 1

    @torch.no_grad()
    def sample(batch: int):
        def forward_logits(tokens):
            output = model(
                input_ids=tokens,
                timesteps=torch.zeros(batch, device=device),
            )
            if hasattr(output, "logits"):
                return output.logits
            if isinstance(output, tuple):
                return output[0]
            return output

        eps = 1e-5
        x = torch.full((batch, args.length), mask_index, dtype=torch.long, device=device)
        timesteps = torch.linspace(1, eps, args.steps + 1, device=device)
        dt = (1 - eps) / args.steps
        cached = None
        for index in range(args.steps):
            t = timesteps[index]
            s = t - dt
            if cached is None:
                logits = forward_logits(x)
                logits[:, :, mask_index] = -1_000_000.0
                log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                unmasked = x != mask_index
                log_probs[unmasked] = -1_000_000.0
                log_probs[unmasked, x[unmasked]] = 0.0
                cached = log_probs.exp()
            probabilities = cached * (t - s)
            probabilities[:, :, mask_index] = s
            proposed = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), 1
            ).reshape_as(x)
            next_x = torch.where(x != mask_index, x, proposed)
            if not torch.equal(next_x, x):
                cached = None
            x = next_x
        logits = forward_logits(x)
        logits[:, :, mask_index] = -1_000_000.0
        return torch.where(x == mask_index, logits.argmax(-1), x)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        while written < args.num_samples:
            current = min(args.batch_size, args.num_samples - written)
            token_ids = sample(current)
            texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            for text in texts[:current]:
                text = " ".join(text.strip().split())
                if not text:
                    continue
                row = {
                    "sample_id": f"{args.run_id}:{written:06d}",
                    "text": text,
                    "run_id": args.run_id,
                    "model": "mdlm-owt",
                    "seed": args.seed,
                    "steps": args.steps,
                    "predictor": args.predictor,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} samples to {destination}")


if __name__ == "__main__":
    main()
