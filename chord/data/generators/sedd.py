"""Generate normalized unconditional samples with the official SEDD code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from .flash_compat import install_if_missing

    install_if_missing()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import sampling
    import torch
    from load_model import load_model
    from transformers import GPT2TokenizerFast

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model_path = Path(args.model)
    legacy_weights = model_path / "pytorch_model.bin"
    if (
        model_path.is_dir()
        and legacy_weights.is_file()
        and not (model_path / "model.safetensors").is_file()
    ):
        # huggingface_hub>=0.36's PyTorchModelHubMixin expects safetensors,
        # while the official SEDD repository still publishes pytorch_model.bin.
        import graph_lib
        import noise_lib
        from model import SEDD
        from omegaconf import OmegaConf

        config = OmegaConf.load(model_path / "config.json")
        model = SEDD(config).to(device)
        model.load_state_dict(torch.load(legacy_weights, map_location=device, weights_only=True))
        graph = graph_lib.get_graph(config, device)
        noise = noise_lib.get_noise(config).to(device)
    else:
        model, graph, noise = load_model(args.model, device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        while written < args.num_samples:
            batch = min(args.batch_size, args.num_samples - written)
            sampler = sampling.get_pc_sampler(
                graph,
                noise,
                (batch, args.length),
                "analytic",
                args.steps,
                device=device,
            )
            token_ids = sampler(model)
            texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            for text in texts:
                text = " ".join(text.strip().split())
                if not text:
                    continue
                row = {
                    "sample_id": f"{args.run_id}:{written:06d}",
                    "text": text,
                    "run_id": args.run_id,
                    "model": "sedd-small",
                    "seed": args.seed,
                    "steps": args.steps,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} samples to {destination}")


if __name__ == "__main__":
    main()
