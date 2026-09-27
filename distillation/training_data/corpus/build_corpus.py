"""Assemble a distillation training corpus from a YAML recipe.

The student's training corpus was grown in layers: a disjoint clean base,
then real-generator coverage, then packed 512-token windows, then ELF
native-length coverage. Each layer was historically a separate hand-edited
script; this module expresses the same construction declaratively so a variant
is a config edit rather than a new script.

    python distillation/training_data/corpus/build_corpus.py \\
        --config distillation/training_data/corpus/train_corpus.yaml

Run it from the repo root: paths inside a recipe are relative to the working
directory (override with ``--root``), so the recipe can live anywhere.

Why the layering matters: **data coverage, not the loss, has been the dominant
lever on the case-study ranking lane.** Successive students went from a fully
inverted generator ladder, to correct-except-ELF once 512-token packed windows
were added, to broadly correct once ELF's native ~929-token format was covered.
If a variant misplaces a generator, first ask what format that generator emits
and whether the student has ever seen it.

Recipe format
-------------

```yaml
seed: 20260714
output: data/distill/training_data/train_corpus.jsonl
report: data/distill/training_data/train_corpus.report.json

# Texts that must never enter training. Every evaluation-side corpus the student
# will be scored on belongs here: leakage inflates the student's apparent quality
# on exactly the lanes you use to judge it.
ban:
  - data/distill/eval_ban/reference_packed512.jsonl
  - data/distill/eval_ban/human_packed_held.jsonl

# Optional starting corpus, taken verbatim in file order (this is how the base
# corpus was built: previous corpus + one new layer).
base: data/distill/training_data/counterfactual/layer_counterfactual.jsonl

# Layers appended after the base, in order.
add:
  - kind: elf_native            # written into every row as "kind"
    domain: elf                 # written into every row as "domain"
    mode: documents             # whole documents, in file order
    source: data/distill/training_data/generator_pools/elf-l-s64c4-seed20260714.jsonl

  - kind: gen_packed_native
    domain: elf
    mode: packed                # EOS-joined stream cut into fixed windows
    window: 929                 # tokens per window
    limit: 400                  # stop after this many accepted windows
    tokenizer: gpt2             # HF tokenizer id used to cut the stream
    source:                     # several pools are concatenated, then shuffled
      - data/distill/training_data/generator_pools/elf-l-s64c4-seed20260620.jsonl
      - data/distill/training_data/generator_pools/elf-l-s64c4-seed20260714.jsonl

# Optional ablation knobs (omit for a faithful rebuild).
drop_kinds: []                  # remove these kinds entirely
cap_kinds: {}                   # {kind: n} -> random subsample down to n

shuffle: true                   # shuffle the assembled corpus (recommended)
```

`documents` layers also accept `limit` (take at most n accepted rows),
`text_key` (default `text`) and `min_words` (skip shorter texts). Every layer
accepts globs in `source` and `where: {field: value}` to keep only matching
source rows (e.g. the OWT rows of a three-domain pool).

`documents` layers can also keep fields of the source rows (`keep_fields:
[kind, domain, source]` overrides the layer's `kind` / `domain` per row) and
record the source file's name as the row's `source` (`tag_from_file: true`).
`where` values may be lists (the field must be one of them).

`exclude_eval_texts: true` adds every released evaluation text (`../excluded_eval_texts.yaml`) to
the ban list; `ban:` lists further files whose texts may not enter the corpus.

A `splice_mix` layer builds cross-model splices: each parent (an existing
corpus row of `parents_kind`) gets `frac` of its sentences replaced by
sentences drawn from 2-3 different generators' pools, at most `per_parent` times,
until `limit` rows are accepted:

```yaml
  - kind: dlm_mix_splice
    mode: splice_mix
    parents_kind: clean_parent
    limit: 1000
    per_parent: 2
    frac: [0.3, 0.5]
    n_models: [2, 3]
    splice_sources:             # generator -> pools (sentences of >= 5 words)
      sedd: data/distill/training_data/generator_pools/sedd-small-s*-seed20260620.jsonl
      ar:   data/distill/training_data/generator_pools/gpt2-*-seed20260620.jsonl
```

The row's `domain` is the '+'-joined generators.

Every layer is deduplicated against the ban list and against everything already
in the corpus, on whitespace-normalized lowercased sha256 — the same key the
historical builders used.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _key(text: str) -> str:
    """Dedup/ban key: whitespace-collapsed, lowercased sha256."""
    return hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()


def _matches(row: Dict, where: Dict | None) -> bool:
    """`where: {field: value}` keeps rows whose field equals value (or is in a list)."""
    for k, v in (where or {}).items():
        if row.get(k) not in (v if isinstance(v, list) else [v]):
            return False
    return True


def _read_rows(path: Path, where: Dict | None = None) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if _matches(row, where):
                    yield row


def _read_texts(path: Path, text_key: str = "text", where: Dict | None = None) -> Iterator[str]:
    for row in _read_rows(path, where):
        yield row[text_key]


def _sources(spec) -> List[str]:
    src = spec.get("source")
    if src is None:
        raise SystemExit(f"layer {spec.get('kind')!r}: missing `source`")
    return [src] if isinstance(src, str) else list(src)


def _expand(root: Path, patterns: List[str]) -> List[Path]:
    """Paths (relative to `root` unless absolute) with globs expanded, sorted."""
    out: List[Path] = []
    for pat in patterns:
        full = pat if Path(pat).is_absolute() else str(root / pat)
        hits = sorted(glob.glob(full)) if any(c in pat for c in "*?[") else [full]
        if not hits:
            raise SystemExit(f"no files match {pat!r}")
        out.extend(Path(h) for h in hits)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble a distillation corpus from a recipe.")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--root",
        default=None,
        help="base directory for the recipe's relative paths "
        "(default: the current working directory — run from the repo root)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report the composition without writing the corpus"
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # Relative paths in a recipe are resolved against the working directory, not
    # against the config's own location: a recipe can then live anywhere without
    # its paths silently changing meaning. Absolute paths are used as given.
    root = Path(args.root).resolve() if args.root else Path.cwd()

    def _p(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (root / p)

    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)

    ban = set()
    for rel in cfg.get("ban", []) or []:
        path = _p(rel)
        if not path.is_file():
            raise SystemExit(f"ban file not found: {path}")
        for text in _read_texts(path):
            ban.add(_key(text))
    if cfg.get("exclude_eval_texts"):
        from common import eval_text_keys

        ban |= eval_text_keys()
    print(f"[corpus] ban list: {len(ban)} texts")

    corpus: List[Dict[str, str]] = []
    seen: set = set()
    added: Counter = Counter()
    dropped: Counter = Counter()

    base = cfg.get("base")
    if base:
        base_path = _p(base)
        if not base_path.is_file():
            raise SystemExit(f"base corpus not found: {base_path}")
        with open(base_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                seen.add(_key(row["text"]))
                corpus.append(row)
        print(f"[corpus] base {base_path.name}: {len(corpus)} rows verbatim")

    for spec in cfg.get("add", []) or []:
        kind = str(spec["kind"])
        domain = str(spec.get("domain", ""))
        mode = str(spec.get("mode", "documents"))
        limit = spec.get("limit")
        text_key = str(spec.get("text_key", "text"))
        where = spec.get("where")
        min_words = int(spec.get("min_words", 0))
        keep_fields = list(spec.get("keep_fields") or [])
        tag_from_file = bool(spec.get("tag_from_file", False))
        paths = [] if mode == "splice_mix" else _expand(root, _sources(spec))
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"layer {kind!r}: source not found: {path}")

        layer_added = [0]  # rows accepted by this layer (limits are per layer)

        def _accept(text: str, domain: str = domain, extra: Dict | None = None) -> bool:
            row = {"text": text, "domain": domain, "kind": kind, **(extra or {})}
            k = row["kind"]
            if not text.strip() or len(text.split()) < min_words:
                dropped[f"{k}:short"] += 1
                return False
            h = _key(text)
            if h in ban:
                dropped[f"{k}:banned"] += 1
                return False
            if h in seen:
                dropped[f"{k}:dup"] += 1
                return False
            seen.add(h)
            corpus.append(row)
            added[k] += 1
            layer_added[0] += 1
            return True

        if mode == "documents":
            for path in paths:
                for row in _read_rows(path, where):
                    if limit is not None and layer_added[0] >= int(limit):
                        break
                    extra = {f: row[f] for f in keep_fields if f in row}
                    if tag_from_file:
                        extra["source"] = path.stem
                    _accept(row[text_key], extra=extra)
            print(f"[corpus] +{layer_added[0]} {kind} documents from {len(paths)} pool(s)")

        elif mode == "packed":
            from transformers import AutoTokenizer

            window = int(spec.get("window", 512))
            tok_id = str(spec.get("tokenizer", "gpt2"))
            tok = AutoTokenizer.from_pretrained(tok_id)
            pool: List[str] = []
            for path in paths:
                pool.extend(_read_texts(path, text_key, where))
            rng.shuffle(pool)
            stream: List[int] = []
            for text in pool:
                stream.extend(tok(text)["input_ids"] + [tok.eos_token_id])
            for i in range(0, len(stream) - window, window):
                if limit is not None and layer_added[0] >= int(limit):
                    break
                _accept(tok.decode(stream[i : i + window], skip_special_tokens=True).strip())
            print(
                f"[corpus] stream {len(stream)} toks -> "
                f"+{layer_added[0]} {kind} windows of {window}"
            )

        elif mode == "splice_mix":
            from chord.text import sentences

            splice_sources: Dict[str, List[str]] = {}
            for model, pats in spec["splice_sources"].items():
                pats = [pats] if isinstance(pats, str) else list(pats)
                splice_sources[model] = [
                    s.strip()
                    for path in _expand(root, pats)
                    for text in _read_texts(path, text_key)
                    for s in sentences(text)
                    if len(s.split()) >= 5
                ]
            models = sorted(splice_sources)
            lo, hi = (float(x) for x in spec.get("frac", [0.3, 0.5]))
            n_models = [int(x) for x in spec.get("n_models", [2, 3])]
            parents = [r["text"] for r in corpus if r.get("kind") == spec["parents_kind"]]
            for parent in parents:
                if limit is not None and layer_added[0] >= int(limit):
                    break
                for _ in range(int(spec.get("per_parent", 2))):
                    sents = [s for s in sentences(parent) if s.strip()]
                    if len(sents) < 4 or (limit is not None and layer_added[0] >= int(limit)):
                        break
                    k = min(max(2, round(len(sents) * rng.uniform(lo, hi))), len(sents) - 1)
                    chosen = rng.sample(models, min(len(models), rng.choice(n_models)))
                    for i, pos in enumerate(rng.sample(range(len(sents)), k)):
                        pool = splice_sources[chosen[i % len(chosen)]]
                        sents[pos] = pool[rng.randrange(len(pool))]
                    _accept(" ".join(sents), "+".join(chosen))
            print(f"[corpus] +{layer_added[0]} {kind} splices from {len(parents)} parents")

        else:
            raise SystemExit(
                f"layer {kind!r}: unknown mode {mode!r} "
                "(expected 'documents', 'packed' or 'splice_mix')"
            )

    drop_kinds = set(cfg.get("drop_kinds") or [])
    if drop_kinds:
        before = len(corpus)
        corpus = [r for r in corpus if r.get("kind") not in drop_kinds]
        print(f"[corpus] drop_kinds {sorted(drop_kinds)}: {before} -> {len(corpus)}")

    cap_kinds = cfg.get("cap_kinds") or {}
    if cap_kinds:
        # Subsample with a DERIVED rng so the main stream (and therefore a
        # faithful rebuild without cap_kinds) is unaffected.
        cap_rng = random.Random(seed ^ 0x5EED)
        by_kind: Dict[str, List[int]] = {}
        for i, r in enumerate(corpus):
            by_kind.setdefault(r.get("kind", ""), []).append(i)
        keep = set(range(len(corpus)))
        for kind, n in cap_kinds.items():
            idx = by_kind.get(str(kind), [])
            if len(idx) <= int(n):
                continue
            cap_rng.shuffle(idx)
            for i in idx[int(n) :]:
                keep.discard(i)
            print(f"[corpus] cap {kind}: {len(idx)} -> {int(n)}")
        corpus = [r for i, r in enumerate(corpus) if i in keep]

    if bool(cfg.get("shuffle", True)):
        rng.shuffle(corpus)

    kinds = dict(Counter(r.get("kind", "") for r in corpus))
    report = {
        "seed": seed,
        "total": len(corpus),
        "kinds": kinds,
        "added": dict(added),
        "dropped": dict(dropped),
        "recipe": str(cfg_path.name),
    }
    print(f"[corpus] total {len(corpus)} | {kinds}")

    if args.dry_run:
        print("[corpus] --dry-run: nothing written")
        return

    out = _p(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in corpus:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {out}")

    rep = cfg.get("report")
    if rep:
        rep_path = _p(rep)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[corpus] wrote {rep_path}")


if __name__ == "__main__":
    main()
