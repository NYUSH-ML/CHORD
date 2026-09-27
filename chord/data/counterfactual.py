"""Build the COMPREHENSIVE meta-evaluation set (one corpus, one manifest).

This is the single entry point for the paper's meta-evaluation (methodology
§"Calibrated Counterfactual Meta-Evaluation" + §"Counterfactual Families"). It
assembles all criterion-isolated families into the standard counterfactual layout
(``texts/reference.jsonl``, ``texts/clean_candidates.jsonl``,
``texts/conditions/<name>.jsonl``, ``conditions_manifest.json``) that
``featurize.py`` and ``selectivity.py`` consume unchanged.

Every family is built with the SAME two properties the paper claims:

  * **Uniform edit position** — edits are drawn uniformly over the passage
    (``position="uniform"``), not appended at a fixed spot, so a metric cannot
    win by exploiting recency / position artifacts.
  * **A monotone, tunable dose** — each family is emitted at several increasing
    severity levels so ``dose_response.py`` can show the metric grows
    monotonically with the strength of the coherence disruption (evidence it
    tracks coherence rather than a surface proxy).

Three groups, matching the methodology and each independently skippable when its
input is absent (a skip is always logged — no silent coverage gaps):

  1. COUNTERFACTUAL PERTURBATIONS
       a. deterministic order / repetition rules (dose = replacement rate),
          generated inline (CPU, no model);
       b. LLM semantic / discourse edits — contradiction, causal reversal,
          broken transition, topic drift, omission (dose = ``n_edits``),
          consumed from a pre-generated ``generate_perturbations`` file (needs
          the editor LLM server; see distillation/training_data/counterfactual/perturbations.yaml).
  2. DLM MIX (``dlm_splice`` harmful vs matched ``human_splice`` control; dose =
     replacement rate) — realistic "locally fluent, globally incoherent" failure.
  3. CORPUS MIX (graded splice: replacement-rate sweep + source-count sweep).

Benign control: the meaning-preserving paraphrase (from the LLM file) when
available, else the unspliced 0-dose parents. Selectivity contrasts every family
against this single benign; ``dlm_splice`` vs ``human_splice`` (same reference /
null) isolates DLM-specific incoherence beyond mere topic discontinuity.

    python -m chord.data.counterfactual \
        --config distillation/training_data/counterfactual/build.yaml
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..utils.hashing import stable_seed
from ..utils.io import load_records
from .perturbations.rules import RULES
from .perturbations.splice import build_sentence_pool, splice_sentences
from ..text import sentence_spans


def _read_texts(path: Path) -> List[str]:
    return [json.loads(line).get("text", "") for line in open(path) if line.strip()]


def _read_rows(path: Path) -> List[Dict]:
    """Read (parent_id, text, domain) rows; parent_id falls back to a positional
    id so legacy text-only files still work."""
    rows = []
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(
            {
                "parent_id": r.get("parent_id", f"par-{i}"),
                "text": r.get("text", ""),
                "domain": r.get("domain"),
            }
        )
    return rows


def _read_texts_many(root: Path, spec: Optional[str]) -> List[str]:
    """Read texts from a single file or a glob (e.g. DLM sample shards)."""
    if not spec:
        return []
    pattern = spec if Path(spec).is_absolute() else str(root / spec)
    texts: List[str] = []
    for fp in sorted(glob.glob(pattern)):
        texts.extend(_read_texts(Path(fp)))
    return texts


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")


def _resolve(root: Path, p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else root / path


def build(config_path: str) -> Dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    root = Path(cfg.get("repo_root", "."))
    out = _resolve(root, cfg["output_dir"])
    corpus_name = cfg.get("corpus_name", "meta_eval")
    seed = int(cfg.get("seed", 20260701))
    n_parents = int(cfg.get("n_parents", 300))
    n_reference = int(cfg.get("n_reference", 3000))
    min_sentences = int(cfg.get("min_sentences", 4))
    position = cfg.get("position", "uniform")

    src = cfg["sources"]
    cand_path = _resolve(root, src["clean_candidates"])
    ref_path = _resolve(root, src["reference"])
    human_pool_path = _resolve(root, src.get("human_splice_pool"))
    (out / "texts" / "conditions").mkdir(parents=True, exist_ok=True)

    # ---- parents (shared across families) + reference pool -------------------
    # parent_id is taken from clean_candidates so it MATCHES the LLM families'
    # parent_sample_id (both come from prepare_meta_eval_corpus) -> every family
    # shares one matched parent set.
    rng0 = random.Random(seed)
    cand = [r for r in _read_rows(cand_path) if len(sentence_spans(r["text"])) >= min_sentences]
    rng0.shuffle(cand)
    parent_rows = cand[:n_parents]
    parents = [r["text"] for r in parent_rows]
    parent_of = {i: parent_rows[i]["parent_id"] for i in range(len(parent_rows))}
    # human passages whose sentences are spliced in (human-splice control,
    # corpus mix): a DISJOINT pool when provided (no self-splice leakage), else
    # fall back to the candidate pool (legacy).
    human_src = human_pool_path if (human_pool_path and human_pool_path.exists()) else cand_path
    human_docs = [t for t in _read_texts(human_src) if len(sentence_spans(t)) >= 2]

    ref = [t for t in _read_texts(ref_path) if t]
    rng0.shuffle(ref)
    _write_jsonl(
        out / "texts" / "reference.jsonl",
        [{"parent_id": f"ref-{i}", "text": t} for i, t in enumerate(ref[:n_reference])],
    )

    clean_rows = [{"parent_id": parent_of[i], "text": t} for i, t in enumerate(parents)]
    _write_jsonl(out / "texts" / "clean_candidates.jsonl", clean_rows)
    _write_jsonl(out / "texts" / "conditions" / "clean_candidates.jsonl", clean_rows)
    print(
        f"[meta] {len(parents)} parents (>= {min_sentences} sents), "
        f"{min(len(ref), n_reference)} reference, {len(human_docs)} human splice passages",
        flush=True,
    )

    conditions: List[Dict] = []
    summary: Dict[str, List] = {"included": [], "skipped": []}

    def emit(name, rows, *, family, group, kind, dose, dose_kind, role="harmful"):
        if not rows:
            summary["skipped"].append(f"{name} (0 rows)")
            return
        _write_jsonl(out / "texts" / "conditions" / f"{name}.jsonl", rows)
        conditions.append(
            {
                "name": name,
                "family": family,
                "group": group,
                "experiment": group,
                "kind": kind,
                "role": role,
                "dose": dose,
                "dose_kind": dose_kind,
                "criterion": family,
            }
        )
        summary["included"].append(f"{name} ({len(rows)} rows, dose={dose})")

    # ---- GROUP 1a: deterministic order / repetition rules --------------------
    rule_doses = [float(d) for d in cfg.get("rule_doses", [0.10, 0.25, 0.50, 0.75, 1.00])]
    for spec in cfg.get("rule_families", []):
        alias, rule = spec["alias"], spec["rule"]
        group = spec.get("group", "order")
        options = spec.get("options", {})
        for dose in rule_doses:
            rows = []
            for i, p in enumerate(parents):
                rng = random.Random(stable_seed(seed, parent_of[i], rule, f"r{dose}", position))
                text, spans, realized = RULES[rule](p, dose, rng, position=position, **options)
                if not spans or text == p:
                    continue
                rows.append(
                    {"parent_id": parent_of[i], "text": text, "realized_rate": round(realized, 3)}
                )
            emit(
                f"{alias}__r{int(round(dose * 100)):03d}",
                rows,
                family=alias,
                group=group,
                kind="harmful",
                dose=dose,
                dose_kind="rate",
            )

    # ---- GROUP 3: corpus mix (graded splice: rate sweep + source-count sweep)
    cm = cfg.get("corpus_mix")
    if cm:
        rate_sweep = cm.get("rate_sweep", {})
        for rate in [float(r) for r in rate_sweep.get("rates", [])]:
            ndon = int(rate_sweep.get("n_sources", 8))
            rows = _splice_rows(parents, parent_of, human_docs, rate, ndon, seed, "corpusmix")
            emit(
                f"corpus_mix__r{int(round(rate * 100)):03d}_d{ndon}",
                rows,
                family="corpus_mix_rate",
                group="corpus_mix",
                kind="harmful",
                dose=rate,
                dose_kind="rate",
            )
        source_sweep = cm.get("source_sweep", {})
        if source_sweep:
            rate = float(source_sweep.get("rate", 0.5))
            for ndon in [int(n) for n in source_sweep.get("n_sources_list", [])]:
                rows = _splice_rows(
                    parents, parent_of, human_docs, rate, ndon, seed, "corpusmix_src"
                )
                emit(
                    f"corpus_mix__src_d{ndon}",
                    rows,
                    family="corpus_mix_source",
                    group="corpus_mix",
                    kind="harmful",
                    dose=ndon,
                    dose_kind="n_donors",  # field name of the released manifests
                )
    else:
        summary["skipped"].append("corpus_mix (no config)")

    # ---- GROUP 2: DLM mix (dlm_splice harmful vs matched human_splice control)-
    dm = cfg.get("dlm_splice")
    dlm_texts = _read_texts_many(root, src.get("dlm_splice_pool")) if dm else []
    if dm and dlm_texts:
        dlm_pool = build_sentence_pool(dlm_texts)
        human_pool = build_sentence_pool(human_docs)
        for rate in [float(r) for r in dm.get("doses", [0.10, 0.25, 0.50])]:
            dlm_rows, human_rows = [], []
            for i, p in enumerate(parents):
                # shared seed -> dlm & human replace the SAME sentence positions,
                # differing only in the source of the inserted sentences (matched control).
                s = stable_seed(seed, parent_of[i], "dlm_splice", f"r{rate}", position)
                d_out, d_sp, d_r = splice_sentences(
                    p, dlm_pool, rate, random.Random(s), position=position
                )
                h_out, h_sp, h_r = splice_sentences(
                    p, human_pool, rate, random.Random(s), position=position
                )
                if d_sp and d_out != p:
                    dlm_rows.append(
                        {"parent_id": parent_of[i], "text": d_out, "realized_rate": round(d_r, 3)}
                    )
                if h_sp and h_out != p:
                    human_rows.append(
                        {"parent_id": parent_of[i], "text": h_out, "realized_rate": round(h_r, 3)}
                    )
            emit(
                f"dlm_splice__r{int(round(rate * 100)):03d}",
                dlm_rows,
                family="dlm_splice",
                group="dlm_mix",
                kind="harmful",
                dose=rate,
                dose_kind="rate",
                role="harmful",
            )
            emit(
                f"human_splice__r{int(round(rate * 100)):03d}",
                human_rows,
                family="human_splice",
                group="dlm_mix",
                kind="harmful",
                dose=rate,
                dose_kind="rate",
                role="control",
            )
    else:
        summary["skipped"].append("dlm_mix (no dlm_splice_pool source)")

    # ---- GROUP 1b + benign: LLM semantic / discourse edits + paraphrase ------
    llm_src = _resolve(root, src.get("llm_perturbations"))
    benign_name = None
    if llm_src and llm_src.exists():
        parent_text_by_id = {parent_of[i]: parents[i] for i in range(len(parents))}
        benign_name = _emit_llm_families(
            llm_src, cfg, parents, out, emit, conditions, summary, parent_text_by_id
        )
    else:
        summary["skipped"].append("llm_families + benign_paraphrase (no llm_perturbations file)")

    # ---- benign control resolution -------------------------------------------
    if benign_name is None:
        # fall back to the unspliced 0-dose parents as the benign anchor.
        conditions.append(
            {
                "name": "clean_candidates",
                "family": "clean_candidates",
                "group": "control",
                "experiment": "control",
                "kind": "benign",
                "role": "benign",
                "dose": 0.0,
                "dose_kind": "none",
                "criterion": "clean",
            }
        )
        summary["included"].append("clean_candidates (benign anchor, 0 dose)")
    else:
        # paraphrase is the benign; keep clean_candidates as a labeled 0-dose control.
        conditions.append(
            {
                "name": "clean_candidates",
                "family": "clean_candidates",
                "group": "control",
                "experiment": "control",
                "kind": "clean",
                "role": "control",
                "dose": 0.0,
                "dose_kind": "none",
                "criterion": "clean",
            }
        )

    manifest = {"corpus": corpus_name, "position": position, "seed": seed, "conditions": conditions}
    (out / "conditions_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "meta_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[meta] included {len(summary['included'])} conditions:")
    for line in summary["included"]:
        print(f"         + {line}")
    if summary["skipped"]:
        print("[meta] skipped:")
        for line in summary["skipped"]:
            print(f"         - {line}")
    print(f"[meta] WROTE {out}", flush=True)
    return manifest


def _splice_rows(parents, parent_of, human_docs, rate, ndon, seed, tag) -> List[Dict]:
    rows = []
    for i, p in enumerate(parents):
        rng = random.Random(stable_seed(seed, parent_of[i], tag, f"r{rate}", f"d{ndon}"))
        docs = rng.sample(human_docs, min(ndon, len(human_docs)))
        pool = build_sentence_pool(docs)
        out, spans, r = splice_sentences(p, pool, rate, rng, position="uniform")
        if not spans or r <= 0.0:
            continue
        rows.append(
            {
                "parent_id": parent_of[i],
                "text": out,
                "realized_rate": round(r, 3),
                "rate": rate,
                "ndonors": ndon,  # number of source passages (released field name)
            }
        )
    return rows


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


_ANCHOR_RE = re.compile(
    r'ANCHOR\s*:\s*["“]?(.+?)["”]?\s*(?:\|\||NEW\s*:|$)', re.IGNORECASE | re.DOTALL
)


def anchor_survives(evidence: str, perturbed_text: str, clean_text: str = "") -> bool:
    """ANCHORED-contradiction gate. The rubric makes the editor quote the
    contradicted counterpart sentence as `ANCHOR: "..."`; a row is valid only
    if that anchor matches an UNCHANGED sentence of the perturbed passage —
    i.e. both incompatible propositions coexist, and the anchor is not the
    rewritten sentence itself. Matching the anchor against the whole perturbed
    text (or fuzzy against every sentence) is NOT safe: a small fact-flip like
    "31.5 inch"->"34 inch" leaves the edited sentence fuzzy-similar (~0.93) to
    the quoted original, which would wave the exact failure mode this gate
    exists to kill straight through. So the candidate set is restricted to
    perturbed sentences whose normalized form also occurs verbatim in the
    clean parent (= untouched sentences)."""
    if not evidence or not clean_text:
        return False
    m = _ANCHOR_RE.search(evidence)
    if not m:
        return False
    anchor = _norm_ws(m.group(1))
    if len(anchor) < 15:  # too short to be a real sentence-level claim
        return False
    clean_set = {_norm_ws(s) for s in re.split(r"(?<=[.!?])\s+", clean_text) if s}
    unchanged = [
        _norm_ws(s)
        for s in re.split(r"(?<=[.!?])\s+", perturbed_text)
        if s and _norm_ws(s) in clean_set
    ]
    return any(
        anchor == s or anchor in s or difflib.SequenceMatcher(a=anchor, b=s).ratio() >= 0.85
        for s in unchanged
    )


def contradiction_row_valid(validation: Dict, perturbed_text: str, clean_text: str = "") -> bool:
    """Validity of one anchored-contradiction row.

    PROGRAMMATIC path (preferred): the generator recorded the exact anchor
    sentence it injected (validation.anchor_sentence); the row is valid iff
    that anchor survives verbatim (whitespace/case-normalized substring) in
    the perturbed passage. The anchor was chosen by US from non-target
    sentences and only target rewrites are spliced back, so this is a
    deterministic no-loophole check. LEGACY path: editor-chosen anchor quoted
    in operation_evidence -> the fuzzy unchanged-sentence gate."""
    validation = validation or {}
    anchor = _norm_ws(str(validation.get("anchor_sentence", "") or ""))
    if anchor:
        return len(anchor) >= 15 and anchor in _norm_ws(perturbed_text)
    return anchor_survives(validation.get("operation_evidence", ""), perturbed_text, clean_text)


def _emit_llm_families(
    llm_src, cfg, parents, out, emit, conditions, summary, parent_text_by_id=None
) -> Optional[str]:
    """Regroup a pre-generated dose-graded LLM perturbation file into conditions.

    Rows are PassageRecord dicts (perturbation, severity=``dose{n}``, position,
    perturbed_text, parent_sample_id). NO-OP edits (perturbed_text identical to the
    parent's clean text) are DROPPED — a harmful edit that changed nothing is not a
    valid harmful example (and an unchanged benign is just the clean text). Families
    with ``require_anchor: true`` (anchored contradiction) additionally drop rows
    whose quoted ANCHOR sentence does not survive in the perturbed passage. Returns
    the benign paraphrase condition name (or None if no paraphrase rows present)."""
    rows = list(load_records(llm_src))
    parent_text_by_id = parent_text_by_id or {}
    # index rows by (perturbation, severity)
    by_key: Dict[tuple, List[Dict]] = {}
    for r in rows:
        pert = r.get("perturbation")
        sev = r.get("severity", "")
        by_key.setdefault((pert, sev), []).append(r)

    def _dose_of(sev: str) -> float:
        return float(sev[4:]) if isinstance(sev, str) and sev.startswith("dose") else 1.0

    def _rows_for(members: List[str], require_anchor: bool = False):
        """collapse the raw records of the requested perturbation names, keyed by
        severity; returns (rows_by_severity, n_dropped_by_anchor_gate)."""
        out_by_sev: Dict[str, List[Dict]] = {}
        dropped_anchor = 0
        for (pert, sev), recs in by_key.items():
            if pert not in members:
                continue
            for rec in recs:
                text = rec.get("perturbed_text")
                pid = rec.get("parent_sample_id") or rec.get("sample_id")
                if not text:
                    continue
                clean = parent_text_by_id.get(pid, "")
                if text == clean:
                    continue  # drop no-op edits (perturbed == clean parent)
                if require_anchor:
                    if not contradiction_row_valid(rec.get("validation"), text, clean):
                        dropped_anchor += 1
                        continue
                out_by_sev.setdefault(sev, []).append({"parent_id": pid, "text": text})
        return out_by_sev, dropped_anchor

    for spec in cfg.get("llm_families", []):
        alias = spec["alias"]
        members = spec.get("members", [spec.get("perturbation", alias)])
        group = spec.get("group", "relation")
        require_anchor = bool(spec.get("require_anchor", False))
        per_sev, dropped = _rows_for(members, require_anchor=require_anchor)
        if not per_sev:
            summary["skipped"].append(f"{alias} (no rows in llm file)")
            continue
        if require_anchor:
            kept = sum(len(v) for v in per_sev.values())
            summary["included"].append(f"{alias}: anchor gate kept {kept}, dropped {dropped}")
        for sev, cond_rows in sorted(per_sev.items()):
            emit(
                f"{alias}__{sev}",
                cond_rows,
                family=alias,
                group=group,
                kind="harmful",
                dose=_dose_of(sev),
                dose_kind="n_edits",
            )

    # benign paraphrase (meaning-preserving surface change) = the primary control,
    # now DOSE-GRADED: mild/moderate/severe -> dose1/2/3 (equal counts, both
    # para_lexical+para_syntactic pooled per level). A good metric should stay FLAT
    # (low z, S~0) across the benign strength ladder while rising on the harmful
    # families. The strongest paraphrase present (most surface change = strictest
    # control) is the kind=benign selectivity ANCHOR; the others are kind=benign_dose
    # so selectivity still computes their z + S (vs the anchor) and dose_response
    # draws the benign curve.
    benign_members = cfg.get("benign_paraphrase", ["para_lexical", "para_syntactic"])
    per_sev, _ = _rows_for(benign_members)

    def _bdose(sev: str) -> float:
        # benign now uses the n_edits dose path (severity="dose1/2/3"); still accept
        # the legacy mild/moderate/severe rubric labels.
        if isinstance(sev, str) and sev.startswith("dose"):
            try:
                return float(sev[4:])
            except ValueError:
                return 99.0
        return {"mild": 1.0, "moderate": 2.0, "severe": 3.0}.get(sev, 99.0)

    emitted = []  # (name, dose, n_rows)
    for sev, rows in sorted(per_sev.items(), key=lambda kv: _bdose(kv[0])):
        if not rows:
            continue
        dose = _bdose(sev)
        name = f"benign_paraphrase__dose{int(dose)}"
        _write_jsonl(out / "texts" / "conditions" / f"{name}.jsonl", rows)
        emitted.append((name, dose, len(rows)))
    if not emitted:
        return None
    anchor_name = max(emitted, key=lambda e: e[1])[0]  # highest dose = strictest control
    for name, dose, nrows in emitted:
        is_anchor = name == anchor_name
        conditions.append(
            {
                "name": name,
                "family": "benign_paraphrase",
                "group": "benign",
                "experiment": "benign",
                "kind": "benign" if is_anchor else "benign_dose",
                "role": "benign",
                "dose": dose,
                "dose_kind": "paraphrase_strength",
                "criterion": "paraphrase",
            }
        )
        tag = "benign ANCHOR" if is_anchor else "benign dose"
        summary["included"].append(f"{name} ({nrows} rows, {tag})")
    return anchor_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
