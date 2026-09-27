from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Dict, Iterator, List

from ..utils.config import config_hash, load_config, resolve_path
from ..utils.hashing import sha256_text
from ..utils.io import load_records
from .perturbations.llm import derive_changed_spans
from .perturbations.llm import generate as generate_llm
from .perturbations.rules import generate_rule_perturbation
from ..utils.records import PassageRecord


def _rubric_with_constraints(spec: Dict, severity: str, rubric: str) -> str:
    clauses = [rubric]
    edit_targets = spec.get("edit_ratio_targets", {})
    if severity in edit_targets:
        low, high = edit_targets[severity]
        clauses.append(
            "Keep the token edit ratio approximately between "
            f"{float(low):.0%} and {float(high):.0%}; prefer the smallest edit "
            "that satisfies the requested severity."
        )
    length_targets = spec.get("length_ratio_targets", {})
    if severity in length_targets:
        low, high = length_targets[severity]
        clauses.append(
            "Keep output length between "
            f"{float(low):.0%} and {float(high):.0%} of the source length."
        )
    if spec.get("require_operation_evidence", False):
        clauses.append(
            "In operation_evidence, identify the exact sentence, claim, relation, "
            "or wording changed. Return zero-based changed_sentence_indices."
        )
    if spec.get("preserve_sentence_order", False):
        clauses.append("Preserve sentence order.")
    return " ".join(clauses)


def _rule_records(
    clean_records: List[PassageRecord], config: Dict, completed: set[str]
) -> Iterator[PassageRecord]:
    rates = config["severity_rates"]
    global_seed = int(config["seed"])
    position_tests = set(config.get("position_tests", []))
    for spec in config.get("rule_perturbations", []):
        name = spec["name"] if isinstance(spec, dict) else spec
        options = spec.get("options", {}) if isinstance(spec, dict) else {}
        positions = (
            config.get("positions", ["early", "middle", "late"])
            if name in position_tests
            else [None]
        )
        for record in clean_records:
            for severity, rate in rates.items():
                for position in positions:
                    suffix = f":{position}" if position else ""
                    sample_id = f"{record.sample_id}:{name}:{severity}{suffix}"
                    if sample_id in completed:
                        continue
                    rule_record = generate_rule_perturbation(
                        record,
                        perturbation=name,
                        severity=severity,
                        rate=float(rate),
                        global_seed=global_seed,
                        position=position,
                        options=options,
                    )
                    # Skip no-op records (empty position third, or an already-satisfied
                    # transform like case_lower on lowercase text): emitting them
                    # pollutes the dataset with text_changed=False rows.
                    if rule_record.perturbed_text == record.clean_text:
                        continue
                    yield rule_record


def _llm_records(
    clean_records: List[PassageRecord], config: Dict, completed: set[str]
) -> Iterator[PassageRecord]:
    llm_config = config.get("llm", {})
    if not llm_config.get("enabled", False):
        return
    global_seed = int(config["seed"])
    total_requests = _count_llm_requests(clean_records, config)
    tasks = []
    for spec in config.get("llm_perturbations", []):
        tasks.extend(_llm_tasks_for_spec(spec, clean_records, config, completed))

    def generate_task(task) -> PassageRecord:
        (
            record,
            name,
            severity,
            rubric,
            position,
            sample_id,
            n_edits,
            dose_label,
        ) = task
        result = generate_llm(
            sample_id=record.sample_id,
            text=record.clean_text,
            perturbation=name,
            severity=severity,
            rubric=rubric,
            config=llm_config,
            global_seed=global_seed,
            position=position or "",
            n_edits=n_edits,
            dose_label=dose_label,
        )
        output = result["perturbed_text"]
        return replace(
            record,
            sample_id=sample_id,
            parent_sample_id=record.sample_id,
            text_hash=sha256_text(output),
            perturbed_text=output,
            perturbation=name,
            severity=severity,
            # For the dose path requested_rate carries n_edits (the monotone dose
            # knob); it stays 0.0 on the legacy severity-rubric path.
            requested_rate=float(n_edits) if dose_label else 0.0,
            realized_rate=0.0,
            generator=llm_config.get("provider", "openai-compatible"),
            generator_revision=llm_config.get("revision", "unpinned"),
            prompt_version=result["_prompt_version"],
            seed=result["_seed"],
            changed_spans=derive_changed_spans(record.clean_text, output),
            position=position,
            validation={
                "preserved_facts": result.get("preserved_facts", []),
                "introduced_error": result.get("introduced_error", ""),
                "operation_evidence": result.get("operation_evidence", ""),
                "changed_sentence_indices": result.get("changed_sentence_indices", []),
                "n_edits": result.get("_n_edits", n_edits),
                "target_indices": result.get("_target_indices", []),
                # programmatic-anchor families ({ANCHOR} rubric): the exact
                # anchor sentence we injected, for the deterministic build gate.
                "anchor_sentence": result.get("_anchor_sentence", ""),
                "anchor_index": result.get("_anchor_index", -1),
            },
        )

    remaining = len(tasks)
    if not remaining:
        return
    max_workers = max(1, int(llm_config.get("max_workers", 1)))
    print(
        f"starting {remaining}/{total_requests} remaining llm requests with {max_workers} workers",
        file=sys.stderr,
        flush=True,
    )
    max_in_flight = max_workers * max(1, int(llm_config.get("in_flight_per_worker", 2)))
    task_iter = iter(tasks)
    pending: Dict[Future, tuple] = {}
    completed_count = 0
    success_count = 0
    failures: List[str] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def fill_queue() -> None:
        while len(pending) < max_in_flight:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending[executor.submit(generate_task, task)] = task

    try:
        fill_queue()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            ready = []
            for future in done:
                task = pending.pop(future)
                record, name, severity, _, position, sample_id, _, _ = task
                completed_count += 1
                try:
                    result_record = future.result()
                except Exception as exc:
                    failures.append(sample_id)
                    print(
                        f"llm failed {completed_count}/{remaining}: "
                        f"sample={record.sample_id} perturbation={name} "
                        f"severity={severity} position={position or 'none'} "
                        f"error={type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                success_count += 1
                print(
                    f"llm done {completed_count}/{remaining}: "
                    f"successful={success_count} failed={len(failures)} "
                    f"sample={record.sample_id} perturbation={name} "
                    f"severity={severity} position={position or 'none'}",
                    file=sys.stderr,
                    flush=True,
                )
                ready.append(result_record)
            fill_queue()
            yield from ready
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        preview = ", ".join(failures[:5])
        print(
            f"WARNING: {len(failures)} LLM requests failed and were skipped "
            f"(resume will retry them). First failed IDs: {preview}",
            file=sys.stderr,
            flush=True,
        )


def _first_rubric(spec: Dict) -> str:
    """Base rubric for the dose sweep: prefer an explicit ``base_rubric``, else the
    ``moderate`` rubric, else the first rubric listed."""
    if spec.get("base_rubric"):
        return str(spec["base_rubric"])
    rubrics = spec.get("rubrics", {})
    if "moderate" in rubrics:
        return str(rubrics["moderate"])
    if rubrics:
        return str(next(iter(rubrics.values())))
    raise ValueError(f"llm_perturbation {spec.get('name')!r} needs base_rubric or rubrics")


def _spec_positions(spec: Dict, config: Dict) -> List:
    """Legacy severity-path positions for one spec (thirds only for position_tests)."""
    position_tests = set(config.get("position_tests", []))
    if spec["name"] in position_tests:
        return list(config.get("positions", ["early", "middle", "late"]))
    return [None]


def _llm_tasks_for_spec(
    spec: Dict, clean_records: List[PassageRecord], config: Dict, completed: set[str]
) -> List[tuple]:
    """Build LLM edit tasks for one perturbation spec.

    Two mutually exclusive modes:
      * ``dose_levels: [1, 2, 3]`` -> the meta-eval dose sweep. Each level applies
        ``n_edits = level`` localized edits at sentences drawn uniformly over the
        passage (``position = "uniform"`` by default), giving a monotone severity
        gradient with uniform placement. severity is tagged ``dose{level}``.
      * legacy ``rubrics: {mild, moderate, severe}`` -> unchanged behavior.
    """
    name = spec["name"]
    dose_levels = spec.get("dose_levels")
    tasks: List[tuple] = []
    if dose_levels:
        base_rubric = _first_rubric(spec)
        position = spec.get("position", config.get("llm_position", "uniform"))
        for record in clean_records:
            for level in dose_levels:
                n_edits = int(level)
                dose_label = f"dose{n_edits}"
                sample_id = f"{record.sample_id}:{name}:{dose_label}"
                if sample_id in completed:
                    continue
                rubric = _rubric_with_constraints(spec, dose_label, base_rubric)
                tasks.append(
                    (record, name, dose_label, rubric, position, sample_id, n_edits, dose_label)
                )
        return tasks
    rubrics = spec["rubrics"]
    positions = _spec_positions(spec, config)
    for record in clean_records:
        for severity, rubric in rubrics.items():
            for position in positions:
                suffix = f":{position}" if position else ""
                sample_id = f"{record.sample_id}:{name}:{severity}{suffix}"
                if sample_id in completed:
                    continue
                constrained = _rubric_with_constraints(spec, severity, rubric)
                tasks.append((record, name, severity, constrained, position, sample_id, 1, ""))
    return tasks


def _count_llm_requests(clean_records: List[PassageRecord], config: Dict) -> int:
    llm_config = config.get("llm", {})
    if not llm_config.get("enabled", False):
        return 0
    total = 0
    for spec in config.get("llm_perturbations", []):
        if spec.get("dose_levels"):
            total += len(clean_records) * len(spec["dose_levels"])
        else:
            total += len(clean_records) * len(spec["rubrics"]) * len(_spec_positions(spec, config))
    return total


def run(config_path: str) -> None:
    config = load_config(config_path)
    llm_config = config.get("llm", {})
    if llm_config.get("enabled") and llm_config.get("require_pinned_revision"):
        revision = str(llm_config.get("revision", "")).strip()
        if not revision or revision.lower() in {
            "main",
            "api",
            "unpinned",
            "set_me",
        }:
            raise ValueError(
                "llm.require_pinned_revision is enabled, but llm.revision is "
                f"not pinned: {revision!r}"
            )
    manifest = resolve_path(config, config["input"]["manifest"])
    output = resolve_path(config, config["output"]["perturbations"])
    rows = load_records(manifest)
    splits = set(config.get("splits", ["dev", "cal", "test"]))
    clean_records = [
        PassageRecord.from_dict(row)
        for row in rows
        if row["role"] == "candidate" and row["split"] in splits
    ]
    max_candidates_per_split = config.get("max_candidates_per_split")
    if max_candidates_per_split is not None:
        selected = []
        limit = int(max_candidates_per_split)
        for split in config.get("splits", ["dev", "cal", "test"]):
            split_records = [record for record in clean_records if record.split == split]
            selected.extend(split_records[:limit])
        clean_records = selected
    else:
        max_candidates = config.get("max_candidates")
        if max_candidates is not None:
            clean_records = clean_records[: int(max_candidates)]

    resume = bool(config.get("resume", False))
    completed: set[str] = set()
    mode = "w"
    completed_source = config.get("input", {}).get("completed")
    if completed_source:
        completed_path = resolve_path(config, completed_source)
        if completed_path.exists():
            completed.update(row["sample_id"] for row in load_records(completed_path))
    if resume and output.exists():
        completed.update(row["sample_id"] for row in load_records(output))
        mode = "a"
    if completed:
        print(
            f"resuming with {len(completed)} completed perturbations",
            file=sys.stderr,
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open(mode, encoding="utf-8") as handle:
        for record in _rule_records(clean_records, config, completed):
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed.add(record.sample_id)
            written += 1
        for record in _llm_records(clean_records, config, completed):
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed.add(record.sample_id)
            written += 1
    meta = {
        "config_hash": config_hash(config),
        "config_path": config["_config_path"],
        "clean_candidate_count": len(clean_records),
        "llm_request_count": _count_llm_requests(clean_records, config),
        "new_record_count": written,
        "total_record_count": len(completed),
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {written} new perturbations for {len(clean_records)} clean candidates "
        f"to {output}; {len(completed)} total"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
