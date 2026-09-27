from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .text import words


def _looks_like_repo_id(name: str) -> bool:
    """True for a Hub id like 'Qwen/Qwen3.5-27B' (exactly one '/', no path marks)."""
    return name.count("/") == 1 and not name.startswith((".", "/", "~")) and "\\" not in name


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def pool_hidden_states(hidden: Any, attention_mask: Any, pooling: str) -> Any:
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling != "masked_mean":
        raise ValueError(f"unknown pooling strategy: {pooling}")
    if isinstance(hidden, np.ndarray):
        mask = np.expand_dims(np.asarray(attention_mask, dtype=hidden.dtype), -1)
        return (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


# Decoder-only (LLM) poolings:
#   last             final-token hidden state of the raw text
#   mean             attention-masked mean over all tokens
#   prompteol        final-token state after a PromptEOL template (CHORD's readout;
#                    the template is chosen by `prompteol_template`)
#   prompteol_multi  MetaEOL-style multi-view readout: the prompteol final-token
#                    state of each template in `prompteol_templates`, each
#                    L2-normalized, then concatenated or averaged (`multiview_combine`)
LLM_POOLINGS = {"last", "mean", "prompteol", "prompteol_multi"}

PROMPTEOL_TEMPLATE = 'This sentence : "{text}" means in one word:'
# Structure/order-inducing prompteol variants (the "single prompt does both" route).
# Each reads the final-token state after the trailing colon (position-robust like
# prompteol), but the cue steers what the gist integrates toward — order/coherence
# rather than topic. Frozen here; never tuned on the evaluation set.
PROMPTEOL_TEMPLATES = {
    "meaning": 'This sentence : "{text}" means in one word:',  # = PromptEOL baseline
    "order": (
        'This passage : "{text}" , read in its given order from start to end, means in one word:'
    ),
    "coherence": (
        'This passage : "{text}" , in terms of its logical coherence and the order of '
        "its ideas, means in one word:"
    ),
    "flow": (
        'This passage : "{text}" , and the logical flow connecting its sentences in '
        "sequence, means in one word:"
    ),
    "consistency": (
        'This passage : "{text}" , judged for whether its statements are logically '
        "consistent and correctly ordered, means in one word:"
    ),
}


def template_input_ids(tokenizer: Any, template: str, text: str, max_length: int) -> List[int]:
    """Tokenize a ``{text}`` template without ever dropping its suffix.

    Hugging Face tokenizers truncate on the right by default.  For PromptEOL
    this is unsafe: a long passage can consume the whole context window and
    silently remove the coherence cue and trailing readout token.  Preserve the
    exact legacy tokenization when the rendered prompt fits.  On overflow,
    truncate only the prefix-plus-passage portion and append the complete fixed
    suffix.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if template.count("{text}") != 1:
        raise ValueError("prompt template must contain exactly one {text} slot")

    rendered = template.format(text=text)
    full_ids = list(tokenizer(rendered, add_special_tokens=True, truncation=False)["input_ids"])
    if len(full_ids) <= max_length:
        return full_ids

    prefix, suffix = template.split("{text}", 1)
    suffix_ids = list(tokenizer(suffix, add_special_tokens=False, truncation=False)["input_ids"])
    if not suffix_ids:
        return full_ids[:max_length]
    if len(suffix_ids) >= max_length:
        raise ValueError("max_length is too small to retain the complete prompt suffix")

    # Tokenizing the complete rendered prompt is preferred because it preserves
    # boundary-sensitive BPE merges.  The separately tokenized suffix normally
    # matches its tail exactly; retain a defensive fallback for tokenizers whose
    # boundary behavior differs.
    if full_ids[-len(suffix_ids) :] == suffix_ids:
        body_ids = full_ids[: -len(suffix_ids)]
    else:
        body_ids = list(
            tokenizer(
                prefix + text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
    budget = max_length - len(suffix_ids)
    return body_ids[:budget] + suffix_ids


FEATURE_IMPLEMENTATION_VERSION = 2  # v2 reserves PromptEOL suffix budget


# Exact token-space (bag-of-words) featurizer for the hidden-vs-token appendix.
# Per-doc L1-normalized count vector over a FIXED vocabulary of the top-K words in
# the reference pool (lowercased, via the same `words()` tokenizer the rest of the
# suite uses), plus one trailing OOV bucket. This is an exact unigram-distribution
# feature: it depends ONLY on the token multiset, so any order-only perturbation
# (e.g. sentence shuffle) leaves each row exactly invariant -> RBF-MMD z -> 0. No
# downloads, fully deterministic.
_UNIGRAM_VOCAB_CACHE: Dict[Tuple[str, int], Dict[str, int]] = {}


def _load_vocab_source_texts(vocab_source: str) -> List[str]:
    """Read newline-delimited JSON records with a ``text`` field (the reference
    pool jsonl); fall back to one-text-per-line for a plain text file."""
    texts: List[str] = []
    with open(vocab_source, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, TypeError, KeyError):
                texts.append(line)
    return texts


def build_unigram_vocab(vocab_source: str, vocab_size: int) -> Dict[str, int]:
    """Deterministic top-K word -> column index over the reference pool. Ties are
    broken lexicographically so the vocabulary is reproducible across runs."""
    key = (str(vocab_source), int(vocab_size))
    cached = _UNIGRAM_VOCAB_CACHE.get(key)
    if cached is not None:
        return cached
    counter: Counter = Counter()
    for text in _load_vocab_source_texts(vocab_source):
        counter.update(words(text.lower()))
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    vocab = {token: index for index, (token, _count) in enumerate(ranked[:vocab_size])}
    _UNIGRAM_VOCAB_CACHE[key] = vocab
    return vocab


def unigram_count_embeddings(
    texts: Sequence[str], vocab_source: str, vocab_size: int = 8192
) -> np.ndarray:
    """L1-normalized unigram count vector over the fixed vocabulary (+1 OOV bucket).
    Dimension = vocab_size + 1."""
    vocab = build_unigram_vocab(vocab_source, vocab_size)
    oov_index = vocab_size
    matrix = np.zeros((len(texts), vocab_size + 1), dtype=np.float32)
    for row_index, text in enumerate(texts):
        for token in words(text.lower()):
            matrix[row_index, vocab.get(token, oov_index)] += 1.0
        total = matrix[row_index].sum()
        if total > 0:
            matrix[row_index] /= total
    return matrix


class HuggingFaceEncoder:
    def __init__(self, protocol: Dict[str, Any]) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Hugging Face encoding requires the 'hf' extra") from exc
        transformers_major = int(transformers.__version__.split(".", 1)[0])
        if transformers_major >= 5 and not hasattr(torch, "float8_e8m0fnu"):
            raise RuntimeError(
                "This environment has transformers>=5 with a torch build that lacks "
                "torch.float8_e8m0fnu. Install a compatible stack, e.g. "
                "`pip install 'transformers>=4.41,<5'` in this conda environment."
            )
        self.torch = torch
        model_name = protocol["model"]
        # `model` goes straight to from_pretrained, which treats a string as a
        # Hub repo id unless it is an existing local directory. A RELATIVE local
        # path therefore fails with an opaque "Repo id must be in the form
        # 'namespace/repo_name'" — a real trap when pointing a config at a
        # locally trained checkpoint, since every OTHER path in these configs is
        # resolved relative to the config file. Catch it here and say so.
        if ("/" in model_name or "\\" in model_name) and not _looks_like_repo_id(model_name):
            if not os.path.isdir(model_name):
                raise FileNotFoundError(
                    f"encoder protocol `model: {model_name}` looks like a filesystem "
                    f"path but no such directory exists (resolved from "
                    f"{os.getcwd()!r}). Use an ABSOLUTE path for a local checkpoint, "
                    "or a Hugging Face repo id. Unlike the other paths in these "
                    "configs, `model` is not resolved relative to the config file."
                )
        revision = protocol.get("revision")
        tokenizer_revision = protocol.get("tokenizer_revision") or revision
        cache_dir = protocol.get("cache_dir")
        trust_remote_code = bool(protocol.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=tokenizer_revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        dtype_name = protocol.get("model_dtype", "float32")
        dtype = getattr(torch, dtype_name)
        load_kwargs = {}
        # Gemma-2 attention soft-capping is only exact under eager attention;
        # SDPA silently skips the cap and distorts the hidden states we read.
        attn_impl = protocol.get("attn_implementation")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        device = protocol.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = int(protocol.get("max_length", 256))
        self.pooling = protocol.get("pooling", "masked_mean")
        # Read-position depth. "last" = the final hidden state (post-final-norm,
        # i.e. output.last_hidden_state); an int K reads output.hidden_states[K]
        # (0 = embeddings, num_layers = final block). The layer-selection ablation
        # (outputs/paper/e1/fig6_layer_sweep) shows the diagnostic logic signal builds
        # monotonically up the stack and peaks ~2 blocks before the end — the final
        # block + RMSNorm slightly regress it — so an upper-but-not-final layer can be
        # a stronger read. Default stays "last" (frozen recipe unchanged).
        # A LIST of ints reads several depths in ONE forward pass and returns
        # them concatenated along the feature axis, in the given order
        # ([N, len(layers)*d]). Used to featurize a teacher for layer-wise
        # (multi-bridge) distillation without one 27B pass per layer; split the
        # columns back per layer (multi-layer targets; not used by the released recipe).
        layer = protocol.get("layer", "last")
        if isinstance(layer, (list, tuple)):
            self.layer = [int(k) for k in layer]
        else:
            self.layer = "last" if layer in ("last", None) else int(layer)
        self._needs_hidden_states = self.layer != "last"
        # decoder-only LLM-embedder support: many causal LMs ship without a pad
        # token; reuse EOS so right/left padding works for batched forwards.
        self.is_llm_pooling = self.pooling in LLM_POOLINGS
        if self.is_llm_pooling and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # evaluation-induced prompted readout: frozen prompt appended after the text.
        # prompteol: allow swapping the shaping template (order/structure-inducing
        # variants) without new pooling code. Accept a named key or a raw template.
        prompteol = protocol.get("prompteol_template", PROMPTEOL_TEMPLATE)
        self.prompteol_template = PROMPTEOL_TEMPLATES.get(prompteol, prompteol)
        # prompteol_multi (MetaEOL): the set of templates to read + how to combine.
        templates = protocol.get("prompteol_templates", ["{text}", PROMPTEOL_TEMPLATE])
        self.prompteol_templates = [PROMPTEOL_TEMPLATES.get(t, t) for t in templates]
        self.multiview_combine = protocol.get("multiview_combine", "concat")
        # Optional linear head on the pooled vector. A bridge-distilled student
        # (distillation/training/train.py `bridge:`) is trained so that P_S·h, not h, matches
        # the teacher, and it writes P_S into each checkpoint as projector.pt.
        # Auto-detected from a local checkpoint dir or a Hub repo that ships one
        # (the released 2B student); `projector: none` disables, an explicit
        # path overrides.
        self.projector = self._load_projector(protocol, model_name)

    @staticmethod
    def _find_projector(model_name: str, revision=None):
        """projector.pt next to the weights: local checkpoint dir or Hub repo."""
        local = Path(model_name) / "projector.pt"
        if local.is_file():
            return local
        if Path(model_name).exists():
            return None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(model_name, "projector.pt", revision=revision))
        except EntryNotFoundError:
            return None

    def _load_projector(self, protocol, model_name):
        torch = self.torch
        spec = protocol.get("projector", "auto")
        if spec in (None, "none", False):
            return None
        if spec == "auto":
            path = self._find_projector(model_name, protocol.get("revision"))
            if path is None:
                return None
        else:
            path = Path(str(spec))
        state = torch.load(path, map_location="cpu")
        head = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
        head.load_state_dict(state)
        head.to(self.device).float().eval()
        print(
            f"[encoder] projector {tuple(head.weight.shape[::-1])} loaded from {path}",
            file=sys.stderr,
            flush=True,
        )
        return head

    def _project(self, pooled):
        if self.projector is None:
            return pooled
        with self.torch.inference_mode():
            return self.projector(pooled.float())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.is_llm_pooling:
            return self._encode_llm(texts)
        torch = self.torch
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model(**batch, output_hidden_states=self._needs_hidden_states)
            if self.pooling == "pooler_output":
                pooled = output.pooler_output
                if pooled is None:
                    raise ValueError(
                        f"{self.model.config.name_or_path} does not expose pooler_output"
                    )
            else:
                pooled = pool_hidden_states(
                    self._select_layer(output),
                    batch["attention_mask"],
                    self.pooling,
                )
        return self._project(pooled).float().cpu().numpy()

    def _select_layer(self, output):
        """Hidden state at the configured read depth ("last" or hidden_states[K]);
        for a list of depths, their concatenation along the feature axis."""
        if self.layer == "last":
            return output.last_hidden_state
        if isinstance(self.layer, list):
            return self.torch.cat([output.hidden_states[k] for k in self.layer], dim=-1)
        return output.hidden_states[self.layer]

    def _forward_hidden(self, input_texts, padding_side):
        """Tokenize + forward a causal LM, returning (last_hidden, attn_mask)."""
        torch = self.torch
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            batch = self.tokenizer(
                list(input_texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = prev_side
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model(**batch, output_hidden_states=self._needs_hidden_states)
        return self._select_layer(output), batch["attention_mask"]

    def _forward_ids(self, ids_list, padding_side):
        """Forward pre-tokenized id lists (padded to the batch max), returning
        (hidden at the read depth, attention mask)."""
        torch = self.torch
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            batch = self.tokenizer.pad({"input_ids": ids_list}, padding=True, return_tensors="pt")
        finally:
            self.tokenizer.padding_side = prev_side
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model(**batch, output_hidden_states=self._needs_hidden_states)
        return self._select_layer(output), batch["attention_mask"]

    def _forward_template(self, texts, template, padding_side="left"):
        """Forward a text template while reserving its fixed suffix budget."""
        ids = [
            template_input_ids(self.tokenizer, template, text, self.max_length) for text in texts
        ]
        return self._forward_ids(ids, padding_side=padding_side)

    def _encode_llm(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        pooling = self.pooling
        if pooling in {"last", "prompteol"}:
            # left-pad so the genuine final token is always at position -1.
            if pooling == "prompteol":
                hidden, _ = self._forward_template(
                    texts, self.prompteol_template, padding_side="left"
                )
            else:
                hidden, _ = self._forward_hidden(list(texts), padding_side="left")
            pooled = hidden[:, -1, :]
        elif pooling == "prompteol_multi":
            # MetaEOL-style multi-view: last-token state of each template, L2-norm
            # each view, then concat (preserve both subspaces) or mean (MetaEOL
            # default). A "{text}" template yields raw-last (no shaping).
            views = []
            for tmpl in self.prompteol_templates:
                hidden, _ = self._forward_template(texts, tmpl, padding_side="left")
                v = hidden[:, -1, :].float()
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                views.append(v)
            if self.multiview_combine == "concat":
                pooled = torch.cat(views, dim=-1)
            elif self.multiview_combine == "mean":
                pooled = torch.stack(views, dim=0).mean(dim=0)
            else:
                raise ValueError(f"unknown multiview_combine: {self.multiview_combine}")
        elif pooling == "mean":
            hidden, mask = self._forward_hidden(list(texts), padding_side="right")
            pooled = pool_hidden_states(hidden, mask, "masked_mean")
        else:
            raise ValueError(f"unknown LLM pooling strategy: {pooling}")
        return self._project(pooled).float().cpu().numpy()


def encode_texts(texts: Sequence[str], protocol: Dict[str, Any]) -> np.ndarray:
    backend = protocol.get("backend", "huggingface")
    if backend == "unigram-count":
        matrix = unigram_count_embeddings(
            texts,
            protocol["vocab_source"],
            int(protocol.get("vocab_size", 8192)),
        )
    elif backend == "huggingface":
        matrix = HuggingFaceEncoder(protocol).encode(texts)
    else:
        raise ValueError(f"unknown embedding backend: {backend}")
    if protocol.get("normalization", "none") == "l2":
        matrix = _l2_normalize(matrix)
    return matrix.astype(protocol.get("cache_dtype", "float32"), copy=False)


def protocol_payload(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Protocol fields hashed into every feature sidecar. The key list is frozen
    (including fields no current readout uses, e.g. `readout_prompt`) so that
    features cached by earlier runs keep matching their configs."""
    keys = (
        "name",
        "backend",
        "model",
        "revision",
        "tokenizer_revision",
        "cache_dir",
        "trust_remote_code",
        "max_length",
        "pooling",
        "prompteol_template",
        "prompteol_templates",
        "multiview_combine",
        "readout_prompt",
        "layer",
        "normalizations",
        "model_dtype",
        "cache_dtype",
        "dimension",
        "reuse_index_resolved_revision",
    )
    payload = {key: protocol.get(key) for key in keys}
    payload["implementation_version"] = FEATURE_IMPLEMENTATION_VERSION
    return payload
