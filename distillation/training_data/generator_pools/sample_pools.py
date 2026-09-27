#!/usr/bin/env python3
"""Sample the training-side generator pools listed in ``pools.yaml``.

One generator per call, because the generators live in different environments:
SEDD / MDLM / LangFlow in ``chord-generators`` (their released code targets
transformers<5), GPT-2 in ``chord-gpu``, ELF-L in its own ``elf`` environment
through the official ELF repository (``ELF_ROOT``). Every pool is written to
``<out_dir>/<name>.jsonl`` through a ``.tmp`` file; finished pools are skipped,
so a requeued job resumes.

    python distillation/training_data/generator_pools/sample_pools.py --generator sedd
    python distillation/training_data/generator_pools/sample_pools.py --generator elf \\
        --elf-root third_party/ELF --elf-env elf

(run from the ``chord-gpu`` env for gpt2 / elf, ``chord-generators`` for the rest;
the ELF branch activates ``--elf-env`` through ``$CONDA_ACTIVATE``)
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent / "pools.yaml"
BATCH = 32


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})


def _snapshot(repo: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision)


def sample(pool: dict, models: dict, out: Path, args: argparse.Namespace) -> None:
    gen, n, seed = pool["generator"], str(pool["n"]), str(pool["seed"])
    common = [
        "--run-id",
        pool["name"],
        "--seed",
        seed,
        "--num-samples",
        n,
        "--batch-size",
        str(BATCH),
        "--length",
        "512",
        "--output",
        str(out),
    ]
    if gen == "sedd":
        m = models["sedd"]
        _run(
            [
                sys.executable,
                "-m",
                "chord.data.generators.sedd",
                "--repo",
                m["repo"],
                "--model",
                _snapshot(m["checkpoint"], m.get("revision")),
                "--steps",
                str(pool["steps"]),
                *common,
            ]
        )
    elif gen == "mdlm":
        _run(
            [
                sys.executable,
                "-m",
                "chord.data.generators.mdlm",
                "--checkpoint",
                models["mdlm"]["checkpoint"],
                "--steps",
                str(pool["steps"]),
                *common,
            ]
        )
    elif gen == "langflow":
        m = models["langflow"]
        _run(
            [
                sys.executable,
                "-m",
                "chord.data.generators.langflow",
                "--repo",
                m["repo"],
                "--checkpoint",
                m["checkpoint"],
                "--steps",
                str(pool["steps"]),
                *common,
            ]
        )
    elif gen == "gpt2":
        # the Table-2 AR sampler and operating point (nucleus p=0.95, T=1.0, 480-512 tokens)
        cfg = {
            "model": pool["model"],
            "n_samples": pool["n"],
            "max_new_tokens": 512,
            "min_new_tokens": 480,
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "batch_size": BATCH,
            "seed": pool["seed"],
            "output_path": str(out),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(cfg, fh)
        _run([sys.executable, "-m", "chord.data.generate_ar", "--config", fh.name])
    elif gen == "elf":
        m = models["elf"]
        elf_root = Path(args.elf_root).resolve()
        raw = elf_root / "outputs" / "chord_training_pools" / pool["name"]
        overrides = [
            "use_bf16=true",
            "use_compile=true",
            "use_wandb=false",
            "global_batch_size=none",
            f"batch_size={BATCH}",
            f"num_samples={n}",
            f"output_dir={raw}",
            f"sampling_configs_path={ROOT / m['sampling_config']}",
        ]
        elf_cfg = "src/configs/training_configs/train_owt_ELF-L.yml"
        launch = [
            "bash",
            "scripts/launch.sh",
            "eval",
            elf_cfg,
            "--checkpoint_path",
            m["checkpoint"],
            "--seed",
            seed,
        ]
        for o in overrides:
            launch += ["--config_override", o]
        # the official sampler runs in its own environment
        activate = os.environ.get("CONDA_ACTIVATE", "activate")
        _run(
            [
                "bash",
                "-c",
                f"source {shlex.quote(activate)} {shlex.quote(args.elf_env)} && "
                f"NGPU=1 {shlex.join(launch)}",
            ],
            cwd=elf_root,
        )
        generated = sorted(raw.glob("*/all_generated_*.jsonl"))[0]
        _run(
            [
                sys.executable,
                "-m",
                "chord.data.generators.normalize_samples",
                "--format",
                "jsonl",
                "--input-key",
                "generated",
                "--model",
                "ELF-L-OWT",
                "--run-id",
                pool["name"],
                "--seed",
                seed,
                "--input",
                str(generated),
                "--output",
                str(out),
            ]
        )
    else:
        raise SystemExit(f"unknown generator {gen!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    generators = ["sedd", "mdlm", "langflow", "gpt2", "elf"]
    ap.add_argument("--generator", required=True, choices=generators)
    ap.add_argument("--elf-root", default=os.environ.get("ELF_ROOT", "third_party/ELF"))
    ap.add_argument("--elf-env", default="elf", help="conda env of the official ELF code")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for pool in cfg["pools"]:
        if pool["generator"] != args.generator:
            continue
        final = out_dir / f"{pool['name']}.jsonl"
        if final.is_file() and len(final.read_text().splitlines()) >= pool["n"]:
            print(f"skip {final.name} (done)", flush=True)
            continue
        tmp = final.with_suffix(".jsonl.tmp")
        sample(pool, cfg["models"], tmp, args)
        tmp.rename(final)
        print(f"wrote {final.name}", flush=True)


if __name__ == "__main__":
    main()
