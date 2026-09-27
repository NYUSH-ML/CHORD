"""Distill the CHORD teacher readout into a small student (LoRA + linear readout).

One objective. For every text the student's last-token hidden state under the
coherence prompt is mapped by a trained linear readout ``P_S`` into the
teacher's top-``rank`` PCA coordinates and matched to the teacher's embedding
of the same text there, with a per-sample mean squared error normalized by the
target variance (it reads as ``1 - R^2``). Relational, angular or
statistic-level terms are deliberately absent: the RBF-MMD statistic the
scorer computes depends on absolute distances, and a positional match fixes
scale, direction and pairwise structure at once.

    python distillation/training/train.py \
        --config distillation/configs/student_qwen3.5-2b/train.yaml

Config blocks (see distillation/configs/student_qwen3.5-2b/train.yaml for the released
values): ``train`` / ``val`` (texts + teacher features), ``bridge`` (rank,
readout init and learning rate), ``lora``, ``batch_block`` (same-kind blocks
inside a batch), ``sampler`` (per-epoch row repeats by kind/source), ``fresh``
(clean rows appended to every batch), ``fingerprint`` (membership checks),
``holdout_align`` and ``decomp_monitor`` (read-only monitors), ``resume``,
``wandb``. Checkpoints are plain merged AutoModel directories plus
``projector.pt``; ``chord.embeddings.HuggingFaceEncoder`` loads the projection
automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chord.embeddings import PROMPTEOL_TEMPLATES, template_input_ids  # noqa: E402
from chord.metrics.distribution import kernel_sum, median_bandwidth, rbf_kernel  # noqa: E402
from chord.utils.io import load_texts as _load_texts_io  # noqa: E402


def _resolve_template(name: str) -> str:
    return PROMPTEOL_TEMPLATES.get(name, name)


def load_texts(path: Path) -> List[str]:
    return _load_texts_io(path)  # same defaults as the corpus builders


def _block_order(rng, groups: dict, block: int, n: int) -> "np.ndarray":
    """Epoch order made of same-(kind, domain) runs of length ``block``.

    Every row index appears as often as it does in ``groups``; runs are
    shuffled globally so consecutive blocks come from different groups.
    """
    runs = []
    for idx in groups.values():
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[rng.permutation(len(idx))]
        for i in range(0, len(idx), block):
            runs.append(idx[i : i + block])
    order = np.concatenate([runs[i] for i in rng.permutation(len(runs))])
    assert len(order) == n, (len(order), n)
    return order


def _fingerprint_metrics(seen_emb: np.ndarray, res_emb: np.ndarray, seed: int = 0):
    """Membership diagnostics on training-side ("seen") vs reserved texts.

    fp_ratio  RBF-MMD(seen, res_half2) / RBF-MMD(res_half1, res_half2)
              (1.0 = seen texts are indistinguishable from unseen ones)
    probe_acc held-out accuracy of a logistic probe classifying seen-vs-
              reserved on the embeddings (0.5 = chance)
    """
    rng = np.random.default_rng(seed)
    s = seen_emb.astype(np.float64)
    r = res_emb.astype(np.float64)
    ri = rng.permutation(len(r))
    r1, r2 = r[ri[: len(r) // 2]], r[ri[len(r) // 2 :]]
    sigma = median_bandwidth(np.concatenate([r1, r2], 0), max_samples=2000, seed=seed)
    k = rbf_kernel(sigma)

    def _mmd(a, b):
        return (
            kernel_sum(a, a, k) / (len(a) ** 2)
            + kernel_sum(b, b, k) / (len(b) ** 2)
            - 2 * kernel_sum(a, b, k) / (len(a) * len(b))
        )

    null = _mmd(r1, r2)
    fp_ratio = _mmd(s, r2) / null if null > 0 else float("nan")
    x = np.concatenate([s, r], 0)
    y = np.concatenate([np.ones(len(s)), np.zeros(len(r))])
    perm = rng.permutation(len(x))
    cut = len(x) // 2
    tr, te = perm[:cut], perm[cut:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    xt = (x[tr] - mu) / sd
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(xt @ w + b)))
        g = p - y[tr]
        w -= 0.1 * (xt.T @ g / len(tr) + 1e-3 * w)
        b -= 0.1 * g.mean()
    pe = 1 / (1 + np.exp(-(((x[te] - mu) / sd) @ w + b)))
    acc = float(((pe > 0.5) == (y[te] > 0.5)).mean())
    return float(fp_ratio), acc


def _holdout_metrics(emb: np.ndarray, target: np.ndarray) -> dict:
    """Per-sample cosine to the teacher target (mean, p10) plus pairwise-cosine
    agreement, on a corpus the student never trains on."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    t = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-9)
    cos = (e * t).sum(1)
    ps = (e @ e.T)[np.triu_indices(len(e), 1)]
    pt = (t @ t.T)[np.triu_indices(len(t), 1)]
    return {
        "cos": float(cos.mean()),
        "cos_p10": float(np.percentile(cos, 10)),
        "pair_student": float(ps.mean()),
        "pair_teacher": float(pt.mean()),
    }


def _decomp(anchor: np.ndarray, pools: dict) -> "tuple[float, dict]":
    """kbb / kab of each pool against a human anchor at the anchor's median
    bandwidth -- the two terms the Table-2 statistic turns on."""
    sigma = median_bandwidth(anchor, max_samples=2000, seed=0)
    k = rbf_kernel(sigma)
    out = {}
    for name, x in pools.items():
        kbb = kernel_sum(x, x, k) / (len(x) ** 2)
        kab = kernel_sum(anchor, x, k) / (len(anchor) * len(x))
        out[name] = (float(kbb), float(kab))
    return float(sigma), out


def _bridge_fit(x: np.ndarray, y: np.ndarray, lam: float):
    """Closed-form ridge regression x -> y with intercept: (W, mean_x, mean_y)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    w = np.linalg.solve(xc.T @ xc + lam * len(x) * np.eye(x.shape[1]), xc.T @ yc)
    return w, mx, my


def _bridge_loss(student_proj, teacher, dim_weight, target_var):
    """Per-sample MSE in the teacher subspace, normalized by the mean target
    variance per dimension (so 1.0 = predicting the mean; ~ 1 - R^2)."""
    d = student_proj - teacher
    if dim_weight is not None:
        d = d * dim_weight.sqrt()
    return (d * d).mean() / target_var


def _wandb_init(wb_cfg: dict, cfg: dict, config_path, out_dir, extra: dict):
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping tracking", flush=True)
        return None
    try:
        return wandb.init(
            project=str(wb_cfg.get("project", "chord-distill")),
            name=wb_cfg.get("name"),
            group=wb_cfg.get("group"),
            tags=list(wb_cfg.get("tags") or []),
            notes=wb_cfg.get("notes"),
            config={**cfg, **extra, "config_path": str(config_path), "output_dir": str(out_dir)},
            dir=str(out_dir),
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] init failed: {exc!r}; continuing without tracking", flush=True)
        return None


def _wb_log(run, data: dict, step: int) -> None:
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed: {exc!r}", flush=True)


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint dir to continue from (an epochN dir); overrides resume.from",
    )
    parser.add_argument(
        "--resume-epochs-done",
        type=int,
        default=None,
        help="epochs that checkpoint completed (overrides resume.epochs_done)",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = _resolve_template(str(cfg.get("prompteol_template", "coherence")))
    batch_block = int(cfg.get("batch_block", 0))
    max_length = int(cfg.get("max_length", 532))
    batch_size = int(cfg.get("batch_size", 16))
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    warmup = int(cfg.get("warmup_steps", 50))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    seed = int(cfg.get("seed", 20260714))
    eval_every = int(cfg.get("eval_every", 120))
    # stop after this many optimizer steps (0 = full schedule); the final val /
    # save still run, so a short run exercises every code path
    max_steps = int(cfg.get("max_steps", 0))
    out_dir = (config_path.parent / cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_cfg = cfg.get("bridge")
    assert bridge_cfg and int(bridge_cfg.get("rank", 0)) > 0, (
        "a `bridge:` block with `rank` is required (the student's readout targets)"
    )
    bridge_rank = int(bridge_cfg["rank"])

    # Resume. EXACT when `train_state.pt` exists next to the checkpoint (LoRA
    # weights, readout, optimizer, schedule, step, RNG and the sampling walks
    # are restored); WARM otherwise (the merged checkpoint becomes the base,
    # fresh adapters and optimizer go on top, `projector.pt` seeds the readout,
    # training restarts at epoch `epochs_done`).
    resume_cfg = dict(cfg.get("resume") or {})
    if args.resume:
        resume_cfg["from"] = args.resume
    if args.resume_epochs_done is not None:
        resume_cfg["epochs_done"] = args.resume_epochs_done
    resume_dir = None
    resume_state = None
    start_epoch = 0
    if resume_cfg.get("from"):
        resume_dir = Path(resume_cfg["from"])
        if not resume_dir.is_absolute():
            resume_dir = (config_path.parent / resume_dir).resolve()
        assert (resume_dir / "config.json").exists(), f"resume: no checkpoint at {resume_dir}"
        state_path = resume_dir / "train_state.pt"
        if state_path.exists():
            resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
            start_epoch = int(resume_state["epochs_done"])
            print(
                f"resume EXACT from {resume_dir}: {start_epoch} epochs done, "
                f"step {resume_state['step']}",
                flush=True,
            )
        else:
            assert "epochs_done" in resume_cfg, (
                "resume: no train_state.pt in the checkpoint; set resume.epochs_done"
            )
            start_epoch = int(resume_cfg["epochs_done"])
            print(
                f"resume WARM from {resume_dir}: merged weights as base, fresh LoRA + "
                f"optimizer, continuing at epoch {start_epoch + 1}/{epochs}",
                flush=True,
            )
        assert 0 < start_epoch < epochs, f"resume: epochs_done={start_epoch} vs epochs={epochs}"

    def _texts(rel: str) -> List[str]:
        return load_texts((config_path.parent / rel).resolve())

    def _feats(rel: str) -> np.ndarray:
        """Teacher targets, truncated to the top-`rank` PCA components."""
        x = np.load((config_path.parent / rel).resolve()).astype(np.float32)
        assert x.shape[1] >= bridge_rank, f"{rel}: width {x.shape[1]} < rank {bridge_rank}"
        return np.ascontiguousarray(x[:, :bridge_rank])

    def _rows(rel: str) -> list:
        rows = [json.loads(line) for line in open((config_path.parent / rel).resolve())]
        assert len(rows) == len(train_texts), "corpus/label row mismatch"
        return rows

    train_texts = _texts(cfg["train"]["texts"])
    train_teacher = _feats(cfg["train"]["teacher_features"])
    assert len(train_texts) == len(train_teacher), "train texts/features misaligned"

    # Same-(kind, domain) block sampling: each batch is made of `batch_block`
    # consecutive rows from one group, so the per-sample loss sees coherent
    # runs of one perturbation type or one generator at a time.
    train_groups = None
    if batch_block > 1:
        train_groups = {}
        for i, row in enumerate(_rows(cfg["train"]["texts"])):
            train_groups.setdefault((row.get("kind", ""), row.get("domain", "")), []).append(i)
        print(
            f"block sampling ON: block={batch_block}, {len(train_groups)} (kind, domain) groups",
            flush=True,
        )

    # Row repeats (`sampler:`): rows whose `kind` / `source` is listed are
    # visited that many times per epoch; `schedule:` overrides the repeats for
    # a range of epochs. Every row still appears at least once per epoch.
    sampler_cfg = cfg.get("sampler")
    sampler_rows = _rows(cfg["train"]["texts"]) if sampler_cfg else None

    def _row_repeat(epoch: int) -> "np.ndarray":
        rep_ = np.ones(len(train_texts), dtype=np.int64)
        if not sampler_cfg:
            return rep_
        by_kind = dict(sampler_cfg.get("repeat_kinds") or {})
        by_src = dict(sampler_cfg.get("repeat_sources") or {})
        for sched in sampler_cfg.get("schedule") or []:
            lo, hi = sched["epochs"]
            if lo <= epoch <= hi:
                by_kind.update(sched.get("repeat_kinds") or {})
                by_src.update(sched.get("repeat_sources") or {})
        for i, r in enumerate(sampler_rows):
            k = max(int(by_kind.get(r.get("kind"), 1)), int(by_src.get(r.get("source"), 1)))
            rep_[i] = max(1, k)
        return rep_

    def _epoch_indices(epoch: int) -> "np.ndarray":
        return np.repeat(np.arange(len(train_texts)), _row_repeat(epoch))

    if sampler_cfg:
        r0 = _row_repeat(0)
        touched = sorted({sampler_rows[i]["kind"] for i in np.flatnonzero(r0 > 1)})
        share_before = (
            float(np.isin([r["kind"] for r in sampler_rows], touched).mean()) if touched else 0.0
        )
        share_after = (
            float(
                sum(r0[i] for i in range(len(r0)) if sampler_rows[i]["kind"] in touched) / r0.sum()
            )
            if touched
            else 0.0
        )
        print(
            f"sampler ON: +{int(r0.sum() - len(r0))} rows/epoch at epoch 0 "
            f"({len(r0)} -> {int(r0.sum())}), kinds {touched} share "
            f"{share_before:.1%} -> {share_after:.1%}"
            + (f", schedule {sampler_cfg.get('schedule')}" if sampler_cfg.get("schedule") else ""),
            flush=True,
        )

    val_sets = []
    for v in cfg.get("val", []):
        vt, vf = _texts(v["texts"]), _feats(v["teacher_features"])
        assert len(vt) == len(vf), f"val {v['name']} misaligned"
        val_sets.append((v["name"], vt, vf))

    # Clean stream: rows appended to every batch from a large clean pool that
    # is walked without replacement, an anchor on unperturbed text.
    fresh_cfg = cfg.get("fresh")
    fresh_texts: List[str] = []
    fresh_teacher = None
    fresh_per_batch = 0
    if fresh_cfg:
        fresh_texts = _texts(fresh_cfg["texts"])
        fresh_teacher = _feats(fresh_cfg["teacher_features"])
        assert len(fresh_texts) == len(fresh_teacher), "fresh texts/features misaligned"
        fresh_per_batch = int(fresh_cfg.get("per_batch", 8))
        print(
            f"fresh stream ON: {len(fresh_texts)} texts, {fresh_per_batch}/batch "
            f"(no-replacement walk)",
            flush=True,
        )

    # Membership checks at every validation: training-side vs reserved texts
    # from the same pools that were never trained on.
    fp_cfg = cfg.get("fingerprint")
    fp_seen_texts: List[str] = []
    fp_res_texts: List[str] = []
    fp_tol = 1.05
    if fp_cfg:
        fp_seen_texts = _texts(fp_cfg["seen_texts"])
        fp_res_texts = _texts(fp_cfg["reserved_texts"])
        fp_tol = float(fp_cfg.get("val_tolerance", 1.05))
        print(
            f"fingerprint probe ON: seen={len(fp_seen_texts)} "
            f"reserved={len(fp_res_texts)} val_tolerance={fp_tol}",
            flush=True,
        )

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        str(cfg.get("model_dtype", "float32"))
    ]
    base_path = (
        str(resume_dir)
        if (resume_dir is not None and resume_state is None)
        else cfg["student_model"]
    )
    model = AutoModel.from_pretrained(base_path, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    lora_cfg = cfg.get("lora")
    if lora_cfg:
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()  # gradient checkpointing + frozen base
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                target_modules=list(
                    lora_cfg.get(
                        "target_modules",
                        [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    )
                ),
            ),
        )
        model.print_trainable_parameters()
    if resume_state is not None and resume_state.get("trainable"):
        _, unexpected = model.load_state_dict(resume_state["trainable"], strict=False)
        assert not unexpected, f"resume: unexpected keys {unexpected[:5]}"
        print(f"resume: restored {len(resume_state['trainable'])} trainable tensors", flush=True)
    model.train()

    # Readout projection P_S: closed-form ridge from the UNTRAINED student's
    # last-token states (`init_features`, row-aligned with train.texts) onto
    # the teacher targets, then trained with the adapters.
    d_student = int(
        model.config.text_config.hidden_size
        if hasattr(model.config, "text_config")
        else model.config.hidden_size
    )
    init = np.load((config_path.parent / bridge_cfg["init_features"]).resolve())
    assert init.shape == (len(train_texts), d_student), (
        f"bridge init_features {init.shape} != ({len(train_texts)}, {d_student})"
    )
    w0, mx0, my0 = _bridge_fit(init, train_teacher, float(bridge_cfg.get("ridge", 1e-3)))
    pred0 = (init.astype(np.float64) - mx0) @ w0 + my0
    r2_0 = (
        1.0
        - ((pred0 - train_teacher) ** 2).sum()
        / ((train_teacher - train_teacher.mean(0)) ** 2).sum()
    )
    bridge_proj = torch.nn.Linear(d_student, bridge_rank).to(device).float()
    with torch.no_grad():
        bridge_proj.weight.copy_(torch.from_numpy(w0.T).float())
        bridge_proj.bias.copy_(torch.from_numpy(my0 - mx0 @ w0).float())
    del init, pred0
    if resume_dir is not None:
        src = resume_state.get("bridge_proj") if resume_state is not None else None
        if src is None and (resume_dir / "projector.pt").exists():
            src = torch.load(resume_dir / "projector.pt", map_location="cpu")
        if src is not None:
            bridge_proj.load_state_dict(src)
            print(f"resume: readout restored from {resume_dir.name}", flush=True)
    bridge_freeze = bool(bridge_cfg.get("freeze", False))
    for p_ in bridge_proj.parameters():
        p_.requires_grad_(not bridge_freeze)
    var = train_teacher.astype(np.float64).var(0)
    bridge_var = float(var.mean())
    bridge_w = None
    if str(bridge_cfg.get("dim_weight", "none")) == "whiten":
        wgt = 1.0 / np.maximum(var, 1e-12)
        bridge_w = torch.from_numpy(wgt / wgt.mean()).to(device).float()
    print(
        f"readout P_S {d_student}->{bridge_rank}: init ridge R2={r2_0:.3f} on "
        f"{len(train_texts)} rows, "
        f"{'frozen' if bridge_freeze else 'trained, lr=' + str(bridge_cfg.get('lr', lr))}, "
        f"dim_weight={bridge_cfg.get('dim_weight', 'none')}, target var/dim={bridge_var:.2f}",
        flush=True,
    )

    def save_student(path) -> None:
        """Plain AutoModel checkpoint (LoRA merged on a deep copy) + tokenizer
        + projector.pt, so the scoring pipeline loads it without peft."""
        if lora_cfg:
            import copy

            merged = copy.deepcopy(model).merge_and_unload()
            merged.save_pretrained(path)
            del merged
        else:
            model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save(bridge_proj.state_dict(), Path(path) / "projector.pt")
        (Path(path) / "bridge.json").write_text(
            json.dumps(
                {
                    "rank": bridge_rank,
                    "in_features": bridge_proj.in_features,
                    "teacher_features": cfg["train"]["teacher_features"],
                }
            )
            + "\n"
        )

    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    if not bridge_freeze:
        param_groups.append(
            {"params": bridge_proj.parameters(), "lr": float(bridge_cfg.get("lr", lr))}
        )
    optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    def embed(texts: List[str], train_mode: bool):
        """Deployed embedding: last-token state of the prompted text through P_S.
        The prompt suffix is reserved so it survives truncation, exactly as in
        the scorer."""
        prompt_ids = [template_input_ids(tokenizer, template, text, max_length) for text in texts]
        batch = tokenizer.pad({"input_ids": prompt_ids}, padding=True, return_tensors="pt").to(
            device
        )
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            hidden = model(**batch).last_hidden_state
            return bridge_proj(hidden[:, -1, :].float())  # left padding: -1 is the last real token

    def val_loss() -> dict:
        model.eval()
        out = {}
        for name, texts, feats in val_sets:
            losses = []
            for i in range(0, len(texts), batch_size):
                sb = embed(texts[i : i + batch_size], False)
                tb = torch.from_numpy(feats[i : i + batch_size]).to(device).float()
                losses.append(float(_bridge_loss(sb, tb, bridge_w, bridge_var)))
            out[name] = float(np.mean(losses)) if losses else float("nan")
        model.train()
        return out

    def embed_all_eval(texts: List[str]) -> np.ndarray:
        model.eval()
        outs = [
            embed(texts[i : i + batch_size], False).cpu().numpy()
            for i in range(0, len(texts), batch_size)
        ]
        model.train()
        return np.concatenate(outs, 0)

    n = len(train_texts)
    total_steps = sum(len(_epoch_indices(e)) // batch_size for e in range(epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_fp = float("inf")
    step = (
        sum(len(_epoch_indices(e)) // batch_size for e in range(start_epoch)) if start_epoch else 0
    )
    metrics_log = out_dir / "train_metrics.jsonl"

    # Hold-out alignment monitor: generators the student never trains on,
    # scored against their pre-projected teacher targets. Read-only.
    ho_cfg = cfg.get("holdout_align")
    ho_sets = []
    ho_every = 0
    if ho_cfg:
        ho_every = int(ho_cfg.get("every", 480))
        for hs in ho_cfg["sets"]:
            tgt = np.ascontiguousarray(
                np.load((config_path.parent / hs["targets"]).resolve()).astype(np.float64)[
                    :, :bridge_rank
                ]
            )
            txt = _load_texts_io(
                (config_path.parent / hs["texts"]).resolve(), "text", "whitespace"
            )[: len(tgt)]
            assert len(txt) == len(tgt), (
                f"holdout {hs['name']}: {len(txt)} texts vs {len(tgt)} targets"
            )
            ho_sets.append((str(hs["name"]), txt, tgt))
        print(
            f"holdout align monitor ON: every {ho_every} steps, sets "
            f"{[(nm, len(t)) for nm, t, _ in ho_sets]}",
            flush=True,
        )
    holdout_log = out_dir / "holdout_metrics.jsonl"

    def holdout_eval(at_step: int) -> dict:
        out = {}
        for nm, txt, tgt in ho_sets:
            m = _holdout_metrics(embed_all_eval(txt).astype(np.float64), tgt)
            out.update({f"holdout/{nm}_{k}": v for k, v in m.items()})
        print(
            f"[holdout] step {at_step} "
            + " | ".join(
                f"{nm} cos {out[f'holdout/{nm}_cos']:.3f} p10 {out[f'holdout/{nm}_cos_p10']:.3f} "
                f"pair {out[f'holdout/{nm}_pair_student']:.3f}"
                f"/{out[f'holdout/{nm}_pair_teacher']:.3f}"
                for nm, _, _ in ho_sets
            ),
            flush=True,
        )
        with open(holdout_log, "a") as fh:
            fh.write(json.dumps({"step": at_step, **out}) + "\n")
        return out

    # kbb / kab monitor on fixed training-side pools (plus the hold-out sets)
    # against a human anchor, in the student's own space at its own median
    # bandwidth; the teacher's values on the same rows are the target. Never
    # supervised, and never on a generator the Table-2 ladder reports.
    dm_cfg = cfg.get("decomp_monitor")
    dm_pools: list = []
    dm_anchor_txt: List[str] = []
    dm_every = 0
    dm_teacher: dict = {}
    if dm_cfg:
        dm_every = int(dm_cfg.get("every", 480))
        dm_n = int(dm_cfg.get("n", 150))
        dm_rng = np.random.default_rng(seed + 7)
        dm_rows = _rows(cfg["train"]["texts"])
        ladder_sources = {"gpt2m_t1.0", "gpt2m_t0.8", "gpt2l_t1.0", "gpt2l_t0.8"}

        def _select(spec: dict) -> np.ndarray:
            keys = {k: spec[k] for k in ("kind", "domain", "source") if k in spec}
            assert keys.get("source") not in ladder_sources, (
                f"decomp pool {spec}: {keys['source']} is a Table-2 ladder generator"
            )
            idx = [i for i, r in enumerate(dm_rows) if all(r.get(k) == v for k, v in keys.items())]
            assert len(idx) >= dm_n, f"decomp pool {spec}: {len(idx)} rows < n {dm_n}"
            return dm_rng.choice(np.asarray(idx), dm_n, replace=False)

        a_idx = _select(dm_cfg.get("anchor", {"kind": "human_packed"}))
        dm_anchor_txt = [train_texts[i] for i in a_idx]
        t_pools = {}
        for pool in dm_cfg["pools"]:
            idx = _select(pool)
            dm_pools.append(
                (
                    str(pool["name"]),
                    str(pool.get("group", "")),
                    [train_texts[i] for i in idx],
                    train_teacher[idx],
                )
            )
            t_pools[str(pool["name"])] = train_teacher[idx]
        if bool(dm_cfg.get("include_holdout", True)):
            for nm, txt, tgt in ho_sets:
                k = min(dm_n, len(txt))
                dm_pools.append((f"holdout_{nm}", "ar", list(txt[:k]), tgt[:k]))
                t_pools[f"holdout_{nm}"] = tgt[:k]
        sig_t, dm_teacher = _decomp(train_teacher[a_idx], t_pools)
        print(
            f"decomp monitor ON: every {dm_every} steps, n={dm_n}, anchor human_packed, "
            f"pools {[nm for nm, *_ in dm_pools]}; teacher sigma {sig_t:.2f}",
            flush=True,
        )
        print(
            "  teacher   "
            + " | ".join(
                f"{nm} kbb {dm_teacher[nm][0]:.3f} kab {dm_teacher[nm][1]:.3f}"
                for nm, *_ in dm_pools
            ),
            flush=True,
        )
    decomp_log = out_dir / "decomp_metrics.jsonl"

    def decomp_eval(at_step: int) -> dict:
        anchor = embed_all_eval(dm_anchor_txt).astype(np.float64)
        pools = {nm: embed_all_eval(txt).astype(np.float64) for nm, _, txt, _ in dm_pools}
        sigma, res = _decomp(anchor, pools)
        out = {"decomp/sigma": sigma}
        groups: dict = {}
        for nm, grp, _, _ in dm_pools:
            kbb, kab = res[nm]
            out[f"decomp/{nm}_kbb"] = kbb
            out[f"decomp/{nm}_kab"] = kab
            if grp:
                groups.setdefault(grp, []).append((kbb, kab))
        for grp, vals in groups.items():
            out[f"decomp/{grp}_kbb"] = float(np.mean([v[0] for v in vals]))
            out[f"decomp/{grp}_kab"] = float(np.mean([v[1] for v in vals]))
        print(
            f"[decomp] step {at_step} sigma {sigma:.2f} "
            + " | ".join(
                f"{nm} kbb {res[nm][0]:.3f}/{dm_teacher[nm][0]:.3f} "
                f"kab {res[nm][1]:.3f}/{dm_teacher[nm][1]:.3f}"
                for nm, *_ in dm_pools
            ),
            flush=True,
        )
        if "ar" in groups and "dlm" in groups:
            print(
                f"[decomp] step {at_step} group means  AR kbb {out['decomp/ar_kbb']:.3f} "
                f"kab {out['decomp/ar_kab']:.3f} | DLM kbb {out['decomp/dlm_kbb']:.3f} "
                f"kab {out['decomp/dlm_kab']:.3f}  (student; teacher on the same rows "
                f"printed at start)",
                flush=True,
            )
        with open(decomp_log, "a") as fh:
            fh.write(json.dumps({"step": at_step, **out}) + "\n")
        return out

    wb_cfg = cfg.get("wandb")
    wb_run = (
        _wandb_init(
            wb_cfg, cfg, config_path, out_dir, {"n_train_rows": n, "total_steps": total_steps}
        )
        if wb_cfg
        else None
    )
    if wb_run is not None and dm_teacher:
        for nm, (kbb, kab) in dm_teacher.items():
            wb_run.summary[f"decomp_teacher/{nm}_kbb"] = kbb
            wb_run.summary[f"decomp_teacher/{nm}_kab"] = kab
    if ho_sets:
        _wb_log(wb_run, holdout_eval(0), 0)
    if dm_pools:
        _wb_log(wb_run, decomp_eval(0), 0)

    fresh_order = rng.permutation(len(fresh_texts)) if fresh_texts else None
    fresh_ptr = 0
    fresh_cycles = 0

    if resume_state is not None:
        optim.load_state_dict(resume_state["optim"])
        sched.load_state_dict(resume_state["sched"])
        rng.bit_generator.state = resume_state["rng"]
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        step = int(resume_state["step"])
        best = float(resume_state.get("best", float("inf")))
        best_fp = float(resume_state.get("best_fp", float("inf")))
        if resume_state.get("fresh") is not None and fresh_texts:
            fresh_order, fresh_ptr, fresh_cycles = resume_state["fresh"]
        print(
            f"resume: optimizer, schedule, RNG and walks restored; continuing at "
            f"step {step}, epoch {start_epoch + 1}/{epochs}",
            flush=True,
        )

    def save_state(path, epochs_done: int) -> None:
        """Everything an EXACT resume needs, next to the merged checkpoint."""
        trainable = (
            {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
            if lora_cfg
            else {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )
        torch.save(
            {
                "trainable": trainable,
                "bridge_proj": bridge_proj.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "epochs_done": epochs_done,
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "best": best,
                "best_fp": best_fp,
                "fresh": ((fresh_order, fresh_ptr, fresh_cycles) if fresh_texts else None),
            },
            Path(path) / "train_state.pt",
        )

    def next_walk(order, ptr, cycles, k, label):
        if ptr + k > len(order):
            order = rng.permutation(len(order))
            ptr = 0
            cycles += 1
            print(f"[walk] {label} pool exhausted -> reshuffle (cycle {cycles})", flush=True)
        return order, ptr + k, cycles, order[ptr : ptr + k]

    for epoch in range(start_epoch, epochs):
        if max_steps and step >= max_steps:
            break
        ep_idx = _epoch_indices(epoch)
        n_ep = len(ep_idx)
        if train_groups:
            groups_ep = train_groups
            if sampler_cfg:
                rep_e = _row_repeat(epoch)
                groups_ep = {
                    g: np.repeat(np.asarray(v, dtype=np.int64), rep_e[np.asarray(v)])
                    for g, v in train_groups.items()
                }
            order = _block_order(rng, groups_ep, batch_block, n_ep)
        else:
            order = rng.permutation(ep_idx)
        for i in range(0, n_ep - batch_size + 1, batch_size):
            if max_steps and step >= max_steps:
                print(f"[max_steps] stopping at step {step}", flush=True)
                break
            idx = order[i : i + batch_size]
            texts_b = [train_texts[j] for j in idx]
            teach_b = [train_teacher[idx]]
            if fresh_texts:
                fresh_order, fresh_ptr, fresh_cycles, fidx = next_walk(
                    fresh_order, fresh_ptr, fresh_cycles, fresh_per_batch, "fresh"
                )
                texts_b += [fresh_texts[j] for j in fidx]
                teach_b.append(fresh_teacher[fidx])
            sb = embed(texts_b, True)
            tb = torch.from_numpy(np.concatenate(teach_b, 0)).to(device).float()
            loss = _bridge_loss(sb, tb, bridge_w, bridge_var)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {float(loss):.4f}", flush=True)
                _wb_log(
                    wb_run,
                    {
                        "train/loss": float(loss),
                        "train/lr": float(sched.get_last_lr()[0]),
                        "epoch": epoch,
                    },
                    step,
                )
            if ho_sets and ho_every > 0 and step % ho_every == 0:
                _wb_log(wb_run, holdout_eval(step), step)
            if dm_pools and dm_every > 0 and step % dm_every == 0:
                _wb_log(wb_run, decomp_eval(step), step)
            if step % eval_every == 0:
                vl = val_loss()
                mean_val = float(np.mean(list(vl.values())))
                fp_ratio = probe_acc = float("nan")
                if fp_cfg:
                    fp_ratio, probe_acc = _fingerprint_metrics(
                        embed_all_eval(fp_seen_texts), embed_all_eval(fp_res_texts), seed=seed
                    )
                print(
                    f"[val] step {step} {vl} mean {mean_val:.5f} "
                    f"fp_ratio {fp_ratio:.2f} probe_acc {probe_acc:.3f}",
                    flush=True,
                )
                with open(metrics_log, "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "step": step,
                                "mean_val": mean_val,
                                "val": vl,
                                "fp_ratio": fp_ratio,
                                "probe_acc": probe_acc,
                            }
                        )
                        + "\n"
                    )
                _wb_log(
                    wb_run,
                    {
                        **{f"val/{k}": v for k, v in vl.items()},
                        "val/mean": mean_val,
                        "fingerprint/fp_ratio": fp_ratio,
                        "fingerprint/probe_acc": probe_acc,
                    },
                    step,
                )
                if mean_val < best:
                    best = mean_val
                    save_student(out_dir / "best")
                    print(f"[val] new best {best:.5f} -> saved", flush=True)
                # among checkpoints within tolerance of the best fit, keep the
                # least-membership one as `best_clean`
                if (
                    fp_cfg
                    and np.isfinite(fp_ratio)
                    and mean_val <= fp_tol * best
                    and fp_ratio < best_fp
                ):
                    best_fp = fp_ratio
                    save_student(out_dir / "best_clean")
                    print(
                        f"[val] new best_clean fp_ratio {best_fp:.2f} "
                        f"(val {mean_val:.5f} <= {fp_tol}x best) -> saved",
                        flush=True,
                    )
        if bool(cfg.get("save_epoch_checkpoints", False)):
            save_student(out_dir / f"epoch{epoch + 1}")
            save_state(out_dir / f"epoch{epoch + 1}", epoch + 1)
            print(f"[epoch] saved checkpoint epoch{epoch + 1} (+ train_state.pt)", flush=True)

    vl = val_loss()
    print(f"[final val] {vl}", flush=True)
    if dm_pools:
        _wb_log(wb_run, decomp_eval(step), step)
    save_student(out_dir / "final")
    if best == float("inf"):
        save_student(out_dir / "best")
    print(
        f"saved student to {out_dir} (best mean val {best:.5f}, best_clean fp_ratio {best_fp:.2f})",
        flush=True,
    )
    if ho_sets and ho_every > 0 and step % ho_every != 0:
        _wb_log(wb_run, holdout_eval(step), step)
    _wb_log(wb_run, {f"final_val/{k}": v for k, v in vl.items()}, step)
    if wb_run is not None:
        try:
            wb_run.summary.update({"best_mean_val": best, "best_clean_fp_ratio": best_fp})
            wb_run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] finish failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
