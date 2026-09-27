from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...utils.hashing import stable_seed
from ...text import sentence_spans

DEFAULT_TEMPLATES = [
    (
        "Rewrite the passage for the requested controlled transformation. "
        "Change only the named criterion and preserve unrelated facts."
    ),
    (
        "Apply exactly one targeted text-quality operation. Keep topic, entities, "
        "and approximate length fixed unless the operation says otherwise."
    ),
    (
        "Produce a minimally edited passage satisfying the transformation rubric. "
        "Return auditable JSON and do not add commentary."
    ),
]


def select_uniform_sentences(text: str, n_edits: int, rng: random.Random) -> List[Tuple[int, str]]:
    """Pick ``n_edits`` distinct sentence targets uniformly at random over the
    passage (returns ``(sentence_index, sentence_text)`` pairs, index-sorted).

    This is the position + dose primitive for the LLM counterfactual families:
    drawing targets uniformly spreads the edits over the whole passage (no
    end-append confound), and ``n_edits`` is a monotone severity knob — one edit
    is a single localized defect, more edits degrade coherence more. Deterministic
    given ``rng``; unit-testable without any LLM server."""
    spans = sentence_spans(text)
    if not spans:
        return []
    k = max(1, min(int(n_edits), len(spans)))
    chosen = sorted(rng.sample(range(len(spans)), k))
    return [(i, text[spans[i][0] : spans[i][1]].strip()) for i in chosen]


def derive_changed_spans(before: str, after: str) -> List[Dict[str, Any]]:
    spans = []
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    for operation, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        spans.append(
            {
                "start": before_start,
                "end": before_end,
                "before": before[before_start:before_end],
                "after": after[after_start:after_end],
            }
        )
    return spans


def build_prompt(
    sample_id: str,
    text: str,
    perturbation: str,
    severity: str,
    rubric: str,
    template: str,
    target_sentences: Optional[Sequence[Tuple[int, str]]] = None,
) -> str:
    schema = {
        "sample_id": sample_id,
        "perturbation": perturbation,
        "severity": severity,
        "perturbed_text": "string",
        "changed_spans": [{"before": "string", "after": "string"}],
        "preserved_facts": ["string"],
        "introduced_error": "string",
        "operation_evidence": "brief description of the exact changed relation or wording",
        "changed_sentence_indices": [0],
    }
    target_clause = ""
    if target_sentences:
        listing = "\n".join(f"  [{idx}] {sent}" for idx, sent in target_sentences)
        target_clause = (
            "\nApply the transformation at EACH of the following target sentences "
            "(0-indexed within the passage) and leave all other sentences "
            "unchanged. Make one localized change per target sentence:\n"
            f"{listing}\n"
        )
    return (
        f"{template}\n\nTransformation: {perturbation}\nSeverity: {severity}\n"
        f"Rubric: {rubric}\n{target_clause}\nPassage:\n{text}\n\n"
        f"Return only JSON matching this schema:\n{json.dumps(schema)}"
    )


def build_rewrite_prompt(
    text: str,
    perturbation: str,
    severity: str,
    rubric: str,
    template: str,
    target_sentences: Sequence[Tuple[int, str]],
) -> str:
    """Prompt for the TARGETED sentence-rewrite path (harmful dose families).

    Instead of regenerating the whole passage (which a non-reasoning model tends to
    echo back unchanged, and which is slow), we ask ONLY for rewritten versions of
    the k target sentences. We splice them back deterministically, which guarantees
    a real, localized edit and keeps outputs short."""
    listing = "\n".join(f"  [{idx}] {sent}" for idx, sent in target_sentences)
    schema = {
        "rewrites": [{"index": 0, "sentence": "the full rewritten sentence"}],
        "operation_evidence": "brief description of the defect you introduced",
    }
    return (
        f"{template}\n\nTransformation: {perturbation}\nSeverity: {severity}\n"
        f"Rubric: {rubric}\n\n"
        "Rewrite ONLY the numbered target sentences below according to the "
        "transformation rubric. Return, for each target, its index and the FULL "
        "rewritten sentence. The rewritten sentence MUST differ from the original, "
        "stay close in length, and remain locally fluent; do NOT return the original "
        "sentence unchanged, and do NOT edit any other sentence.\n"
        f"Target sentences (0-indexed within the passage):\n{listing}\n\n"
        f"Full passage (context only, do not return it):\n{text}\n\n"
        f"Return only JSON matching this schema:\n{json.dumps(schema)}"
    )


def apply_sentence_rewrites(
    text: str,
    target_sentences: Sequence[Tuple[int, str]],
    rewrites: Sequence[Dict[str, Any]],
) -> Tuple[str, List[int]]:
    """Splice model-provided sentence rewrites back into ``text`` at the target
    sentence indices. Returns (new_text, indices_actually_changed). Only targets
    with a non-empty rewrite that differs from the original are applied; edits are
    applied right-to-left so earlier character offsets stay valid."""
    spans = sentence_spans(text)
    target_idx = {idx for idx, _ in target_sentences}
    by_idx: Dict[int, str] = {}
    for r in rewrites or []:
        try:
            i = int(r["index"])
        except (KeyError, TypeError, ValueError):
            continue
        sent = r.get("sentence")
        if i in target_idx and isinstance(sent, str) and sent.strip():
            by_idx[i] = " ".join(sent.split())
    changed: List[Tuple[int, int, str, int]] = []
    for i, new in by_idx.items():
        if i >= len(spans):
            continue
        s, e = spans[i]
        if new == text[s:e].strip():
            continue
        changed.append((s, e, new, i))
    if not changed:
        return text, []
    out = text
    for s, e, new, _ in sorted(changed, key=lambda c: -c[0]):
        out = out[:s] + new + out[e:]
    return out, sorted(c[3] for c in changed)


def call_openai_compatible(
    base_url: str,
    model: str,
    prompt: str,
    seed: int,
    temperature: float | None,
    timeout_seconds: int,
    api_key: str | None = None,
    max_retries: int = 5,
    retry_base_seconds: float = 2.0,
    max_completion_tokens: int | None = None,
    required_string_field: str | None = "perturbed_text",
    reasoning_effort: str | None = None,
    chat_template_kwargs: Dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    # Optional fixed system message: some vendor chat templates (e.g.
    # Mistral-Small-24B) otherwise inject a default system prompt containing the
    # CURRENT DATE, making renders date-dependent; an explicit system message
    # replaces it entirely.
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "response_format": {"type": "json_object"},
    }
    # vLLM extension: forward chat-template controls (e.g. {"enable_thinking": false}
    # to turn OFF Qwen3 reasoning tokens, which otherwise blow the token budget and
    # truncate the JSON body). Harmless on servers that ignore unknown fields.
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    # Some hosted reasoning models (e.g. gpt-5-mini) only accept the default
    # temperature; pass None to omit it. vLLM generators still send 0.0.
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            # Strip Qwen3/reasoning-model thinking tokens when present in content.
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            parsed = json.loads(content)
            if required_string_field is not None and not isinstance(
                parsed.get(required_string_field), str
            ):
                raise TypeError(f"LLM response is missing a string {required_string_field}")
            return parsed
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= max_retries:
                raise RuntimeError(
                    f"LLM request failed with HTTP {exc.code}: {endpoint}\n{body}"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                reason = getattr(exc, "reason", str(exc))
                raise RuntimeError(f"LLM request failed: {endpoint}\nReason: {reason}") from exc
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM returned invalid JSON after {attempt + 1} attempts: {endpoint}"
                ) from exc
        delay = retry_base_seconds * (2**attempt)
        delay += random.Random(seed + attempt).uniform(0.0, retry_base_seconds)
        time.sleep(delay)
    raise AssertionError("unreachable")


def generate(
    sample_id: str,
    text: str,
    perturbation: str,
    severity: str,
    rubric: str,
    config: Dict[str, Any],
    global_seed: int,
    position: str = "",
    n_edits: int = 1,
    dose_label: str = "",
) -> Dict[str, Any]:
    templates: List[str] = config.get("templates") or DEFAULT_TEMPLATES
    # dose_label (e.g. "dose2") keys the seed for the meta-eval dose sweep so each
    # dose level is a distinct, reproducible draw; falls back to severity so the
    # legacy mild/moderate/severe path is byte-for-byte unchanged.
    seed = stable_seed(global_seed, sample_id, perturbation, dose_label or severity, position)
    target_sentences: Optional[List[Tuple[int, str]]] = None
    if position == "uniform" or n_edits > 1:
        # Uniform target placement + n_edits dose (meta-eval path). Draw targets
        # from a seed-derived rng so the same (sample, dose, position) is stable.
        target_sentences = select_uniform_sentences(text, n_edits, random.Random(seed))
    elif position:
        rubric = f"{rubric} Place the targeted change in the {position} third."
    anchor_sentence: Optional[Tuple[int, str]] = None
    if target_sentences is not None and "{ANCHOR}" in rubric:
        # Programmatic anchoring (anchored-contradiction family): WE choose the
        # anchor — seeded, uniform over NON-target sentences (>=6 words when
        # available) — and inject it into the rubric. Only target rewrites are
        # spliced back, so the anchor cannot be edited and "both incompatible
        # claims coexist" holds by construction; build_corpus verifies the
        # recorded anchor survived verbatim (deterministic gate). This replaced
        # letting the editor pick its own anchor, which a no-think model
        # resolved by quoting the target's original wording ~half the time.
        spans = sentence_spans(text)
        tgt = {idx for idx, _ in target_sentences}
        elig = [(i, text[a:b].strip()) for i, (a, b) in enumerate(spans) if i not in tgt]
        preferred = [p for p in elig if len(p[1].split()) >= 6]
        pool = preferred or elig
        if pool:
            arng = random.Random(stable_seed(seed, "anchor"))
            anchor_sentence = pool[arng.randrange(len(pool))]
            rubric = rubric.replace(
                "{ANCHOR}", f'sentence [{anchor_sentence[0]}]: "{anchor_sentence[1]}"'
            )
        else:
            rubric = rubric.replace("{ANCHOR}", "another sentence of the passage")
    template = templates[seed % len(templates)]
    # Targeted-rewrite path (harmful dose families): ask for only the k rewritten
    # target sentences and splice them back. Whole-passage regeneration is kept for
    # the legacy path (benign paraphrase), which is a genuine whole-passage rewrite.
    rewrite_mode = target_sentences is not None
    if rewrite_mode:
        prompt = build_rewrite_prompt(
            text, perturbation, severity, rubric, template, target_sentences
        )
        required_field = None
    else:
        prompt = build_prompt(
            sample_id, text, perturbation, severity, rubric, template, target_sentences
        )
        required_field = "perturbed_text"
    result = call_openai_compatible(
        base_url=config.get("base_url", "http://127.0.0.1:8000/v1"),
        model=config["model"],
        prompt=prompt,
        seed=seed,
        temperature=float(config.get("temperature", 0.0)),
        timeout_seconds=int(config.get("timeout_seconds", 180)),
        api_key=os.environ.get(config.get("api_key_env", "OPENAI_API_KEY")),
        max_retries=int(config.get("max_retries", 5)),
        retry_base_seconds=float(config.get("retry_base_seconds", 2.0)),
        max_completion_tokens=(
            int(config["max_completion_tokens"])
            if config.get("max_completion_tokens") is not None
            else None
        ),
        chat_template_kwargs=config.get("chat_template_kwargs"),
        system_prompt=config.get("system_prompt"),
        required_string_field=required_field,
    )
    if rewrite_mode:
        spliced, changed_idx = apply_sentence_rewrites(
            text, target_sentences, result.get("rewrites") or []
        )
        result["perturbed_text"] = spliced
        result["_changed_indices"] = changed_idx
    result["_seed"] = seed
    result["_n_edits"] = n_edits
    result["_target_indices"] = [idx for idx, _ in target_sentences] if target_sentences else []
    if anchor_sentence is not None:
        result["_anchor_index"] = anchor_sentence[0]
        result["_anchor_sentence"] = anchor_sentence[1]
    prompt_revision = str(config.get("prompt_revision", "v1"))
    result["_prompt_version"] = f"template-{seed % len(templates)}-{prompt_revision}"
    return result
