"""Fail-closed, deterministic evidence status stratification for B0 K=3 hard rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.io_utils import (
    load_json,
    load_jsonl,
    verify_sha_sidecar,
)
from . import (
    CLASSIFICATION_ALGORITHM_VERSION,
    EXPECTED_HARD_COUNT,
    EXPECTED_HARD_BY_SPLIT,
    EXPECTED_SPLIT_COUNTS,
    STRATIFICATION_VERSION,
)


REQUIRED_SOURCE_FILES = (
    "difficulty_manifest.jsonl",
    "train_hard_manifest.jsonl",
    "dev_hard_manifest.jsonl",
    "test_hard_manifest.jsonl",
    "program_gate_results.jsonl",
    "root_split_manifest.jsonl",
)
SPLITS = ("train", "dev", "test")
STATUSES = ("EVIDENCE_CLEAN", "EVIDENCE_RISK", "EVIDENCE_INVALID", "EVIDENCE_UNCLASSIFIABLE")
MECHANISMS = ("position_only", "sampling_only", "mixed", "none", "unknown")

ROOT_REF_RE = re.compile(r"^(cc-train-episode-[^:]+):P[0-9]+:T[0-9]+$")
CHILD_REF_RE = re.compile(r"^R[0-9]+:S[0-9]+:T[0-9]+$")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _require_source_sidecars(source_run: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in REQUIRED_SOURCE_FILES:
        path = source_run / name
        if not path.is_file():
            raise RuntimeError(f"source_file_missing:{path}")
        verify_sha_sidecar(path)
        hashes[name] = _sha(path)
    return hashes


def _id_rows(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"{label}_row_not_object:{index}")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise RuntimeError(f"{label}_candidate_id_missing:{index}")
        if candidate_id in result:
            raise RuntimeError(f"{label}_candidate_id_duplicate:{candidate_id}")
        result[candidate_id] = dict(row)
    return result


def _root_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"root_split_row_not_object:{index}")
        root_id = str(row.get("root_id") or "")
        if not root_id:
            raise RuntimeError(f"root_split_root_id_missing:{index}")
        if root_id in result:
            raise RuntimeError(f"root_split_root_id_duplicate:{root_id}")
        result[root_id] = dict(row)
    return result


def _validate_source_inputs(source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    hashes = _require_source_sidecars(source_run)
    difficulty_rows = load_jsonl(source_run / "difficulty_manifest.jsonl")
    difficulty = _id_rows(difficulty_rows, label="difficulty_manifest")
    if len(difficulty) != 3286:
        raise RuntimeError(f"difficulty_candidate_count_mismatch:{len(difficulty)}:3286")
    hard = {
        candidate_id: row for candidate_id, row in difficulty.items()
        if row.get("difficulty_bucket") == "hard_candidate"
    }
    if len(hard) != EXPECTED_HARD_COUNT:
        raise RuntimeError(f"hard_count_mismatch:{len(hard)}:{EXPECTED_HARD_COUNT}")
    hard_by_split = Counter(str(row.get("split") or "") for row in hard.values())
    if dict(hard_by_split) != EXPECTED_HARD_BY_SPLIT:
        raise RuntimeError(f"hard_split_count_mismatch:{dict(hard_by_split)}:{EXPECTED_HARD_BY_SPLIT}")

    split_files: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        rows = load_jsonl(source_run / f"{split}_hard_manifest.jsonl")
        observed = _id_rows(rows, label=f"{split}_hard_manifest")
        expected = {candidate_id: row for candidate_id, row in hard.items() if row.get("split") == split}
        if set(observed) != set(expected):
            raise RuntimeError(f"{split}_hard_candidate_ids_mismatch")
        for candidate_id, row in observed.items():
            source = expected[candidate_id]
            for key in ("split", "candidate_pool_path", "candidate_pool_sha256", "dialogue_path", "dialogue_sha256"):
                if row.get(key) != source.get(key):
                    raise RuntimeError(f"{split}_hard_source_field_mismatch:{candidate_id}:{key}")
        split_files[split] = observed

    gate = _id_rows(load_jsonl(source_run / "program_gate_results.jsonl"), label="program_gate")
    roots = _root_rows(load_jsonl(source_run / "root_split_manifest.jsonl"))
    if not gate or not roots:
        raise RuntimeError("source_gate_or_root_split_empty")
    root_splits = {root_id: str(row.get("split") or "") for root_id, row in roots.items()}
    root_ids = {str(row.get("root_id") or "") for row in hard.values()}
    missing_roots = sorted(root_ids - set(root_splits))
    if missing_roots:
        raise RuntimeError(f"hard_root_split_rows_missing:{missing_roots}")
    for candidate_id, row in hard.items():
        root_id = str(row.get("root_id") or "")
        if str(row.get("split") or "") != root_splits[root_id]:
            raise RuntimeError(f"hard_split_root_mismatch:{candidate_id}")
        gate_row = gate.get(candidate_id)
        if gate_row is None:
            raise RuntimeError(f"hard_program_gate_row_missing:{candidate_id}")
        for key in ("candidate_pool_path", "candidate_pool_sha256", "dialogue_path", "dialogue_sha256"):
            if row.get(key) != gate_row.get(key):
                raise RuntimeError(f"hard_gate_source_field_mismatch:{candidate_id}:{key}")
        if gate_row.get("gate_status") != "PROGRAM_PASS":
            raise RuntimeError(f"hard_program_gate_not_pass:{candidate_id}")
        root_row = roots[root_id]
        if row.get("candidate_pool_sha256") != root_row.get("candidate_pool_sha256"):
            raise RuntimeError(f"hard_root_pool_sha_mismatch:{candidate_id}")
        if row.get("dialogue_sha256") != root_row.get("dialogue_sha256"):
            raise RuntimeError(f"hard_root_dialogue_sha_mismatch:{candidate_id}")
    return {
        "source_hashes": hashes,
        "difficulty": difficulty,
        "hard": hard,
        "split_files": split_files,
        "gate": gate,
        "roots": roots,
    }, {
        "difficulty_count": len(difficulty),
        "hard_count": len(hard),
        "hard_by_split": dict(sorted(hard_by_split.items())),
        "source_hashes": hashes,
    }


def _load_lineage(
    hard: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    pool_cache: dict[str, dict[str, Any]] = {}
    dialogue_refs: dict[str, set[str]] = {}
    dialogue_cache: dict[str, dict[str, Any]] = {}
    for candidate_id, row in hard.items():
        root_id = str(row.get("root_id") or "")
        pool_path = Path(str(row.get("candidate_pool_path") or ""))
        dialogue_path = Path(str(row.get("dialogue_path") or ""))
        if not pool_path.is_file() or not dialogue_path.is_file():
            raise RuntimeError(f"lineage_file_missing:{candidate_id}")
        pool_key = str(pool_path.resolve())
        if pool_key not in pool_cache:
            if _sha(pool_path) != str(row.get("candidate_pool_sha256") or ""):
                raise RuntimeError(f"candidate_pool_sha_mismatch:{candidate_id}")
            payload = load_json(pool_path)
            if not isinstance(payload, list):
                raise RuntimeError(f"candidate_pool_not_list:{root_id}")
            mapped = _id_rows(payload, label=f"candidate_pool:{root_id}")
            pool_cache[pool_key] = mapped
        dialogue_key = str(dialogue_path.resolve())
        if dialogue_key not in dialogue_cache:
            if _sha(dialogue_path) != str(row.get("dialogue_sha256") or ""):
                raise RuntimeError(f"dialogue_sha_mismatch:{candidate_id}")
            payload = load_json(dialogue_path)
            if not isinstance(payload, Mapping) or str(payload.get("root_id") or "") != root_id:
                raise RuntimeError(f"dialogue_root_identity_mismatch:{candidate_id}")
            turns = payload.get("turns")
            if not isinstance(turns, list) or not turns:
                raise RuntimeError(f"dialogue_turns_missing:{root_id}")
            refs = [str(turn.get("turn_ref") or "") for turn in turns if isinstance(turn, Mapping)]
            if len(refs) != len(set(refs)) or any(not ref for ref in refs):
                raise RuntimeError(f"dialogue_turn_refs_invalid:{root_id}")
            dialogue_cache[dialogue_key] = dict(payload)
            dialogue_refs[dialogue_key] = set(refs)
        candidate = pool_cache[pool_key].get(candidate_id)
        if candidate is None:
            raise RuntimeError(f"candidate_missing_from_pool:{candidate_id}")
        if str(candidate.get("root_id") or "") != root_id:
            raise RuntimeError(f"candidate_pool_root_mismatch:{candidate_id}")
        if canonical_json(candidate) != canonical_json(row.get("candidate")):
            raise RuntimeError(f"difficulty_candidate_lineage_drift:{candidate_id}")
    return pool_cache, dialogue_refs, dialogue_cache


def _append_reason(bucket: list[str], reason: str) -> None:
    if reason not in bucket:
        bucket.append(reason)


def _ref_error(ref: Any, *, root_id: str, valid_refs: set[str]) -> str | None:
    if not isinstance(ref, str) or not ref:
        return "evidence_ref_format_invalid"
    root_match = ROOT_REF_RE.fullmatch(ref)
    if root_match is None and CHILD_REF_RE.fullmatch(ref) is None:
        return "evidence_ref_format_invalid"
    if root_match is not None and root_match.group(1) != root_id:
        return "evidence_ref_wrong_root"
    if ref not in valid_refs:
        return "evidence_ref_not_in_frozen_snapshot"
    return None


def _alignment_status(candidate: Mapping[str, Any], generation_source: str) -> tuple[str, list[str], list[str]]:
    risk: list[str] = []
    unclassifiable: list[str] = []
    if generation_source == "free_exploration":
        return "NOT_APPLICABLE", risk, unclassifiable
    if generation_source != "contract_guided":
        _append_reason(unclassifiable, "generation_source_unknown")
        return "UNKNOWN", risk, unclassifiable
    alignment = candidate.get("contract_group_alignment")
    if not isinstance(alignment, Mapping) or not alignment:
        _append_reason(unclassifiable, "contract_group_alignment_missing")
        return "UNKNOWN", risk, unclassifiable
    group_names = set(str(name) for name in (candidate.get("evidence_groups") or {}) if name)
    required = set(str(name) for name in (candidate.get("required_reasoning") or []) if name)
    if required and not required.issubset(set(str(name) for name in alignment)):
        _append_reason(risk, "contract_group_alignment_incomplete")
    if group_names and not group_names.issubset(set(str(name) for name in alignment)):
        _append_reason(risk, "contract_group_alignment_incomplete")
    values_ok = True
    for value in alignment.values():
        if value is True:
            continue
        if isinstance(value, str) and value.strip().casefold() in {"pass", "full", "true"}:
            continue
        values_ok = False
    if not values_ok:
        _append_reason(risk, "contract_group_alignment_not_full")
    return "FULL" if values_ok and not risk else "PARTIAL_OR_FALSE", risk, unclassifiable


def classify_evidence(
    row: Mapping[str, Any],
    *,
    valid_refs: set[str],
    difficulty_manifest_path: Path,
    difficulty_manifest_sha256: str,
) -> dict[str, Any]:
    candidate = row.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError(f"candidate_payload_missing:{row.get('candidate_id')}")
    root_id = str(row.get("root_id") or "")
    flat = candidate.get("evidence_refs")
    groups = candidate.get("evidence_groups")
    invalid: list[str] = []
    risk: list[str] = []
    unclassifiable: list[str] = []

    if flat is None:
        _append_reason(unclassifiable, "flat_evidence_refs_missing")
        flat_refs: list[Any] = []
    elif not isinstance(flat, list):
        _append_reason(unclassifiable, "flat_evidence_refs_not_list")
        flat_refs = []
    else:
        flat_refs = list(flat)
        if not flat_refs:
            _append_reason(unclassifiable, "flat_evidence_refs_empty")
    for ref in flat_refs:
        error = _ref_error(ref, root_id=root_id, valid_refs=valid_refs)
        if error is not None:
            _append_reason(invalid, error)
    string_flat = [ref for ref in flat_refs if isinstance(ref, str)]
    if len(string_flat) != len(set(string_flat)):
        _append_reason(risk, "flat_evidence_duplicate_ref")

    if groups is None:
        _append_reason(unclassifiable, "evidence_groups_missing")
        group_union: list[Any] = []
    elif not isinstance(groups, Mapping):
        _append_reason(unclassifiable, "evidence_groups_not_object")
        group_union = []
    elif not groups:
        _append_reason(unclassifiable, "evidence_groups_empty")
        group_union = []
    else:
        group_union = []
        for name, refs in groups.items():
            if not isinstance(name, str) or not name:
                _append_reason(unclassifiable, "evidence_group_name_invalid")
            if not isinstance(refs, list):
                _append_reason(unclassifiable, "evidence_group_refs_not_list")
                continue
            if not refs:
                _append_reason(risk, "evidence_group_empty")
            group_union.extend(refs)
            for ref in refs:
                error = _ref_error(ref, root_id=root_id, valid_refs=valid_refs)
                if error is not None:
                    _append_reason(invalid, error)
        string_union = [ref for ref in group_union if isinstance(ref, str)]
        if len(string_union) != len(set(string_union)):
            _append_reason(risk, "evidence_group_duplicate_ref")
        if isinstance(flat, list) and set(string_flat) != set(string_union):
            _append_reason(risk, "flat_group_union_mismatch")

    alignment_status, alignment_risk, alignment_unclassifiable = _alignment_status(
        candidate, str(row.get("generation_source") or ""),
    )
    for reason in alignment_risk:
        _append_reason(risk, reason)
    for reason in alignment_unclassifiable:
        _append_reason(unclassifiable, reason)
    if row.get("program_gate_status") != "PROGRAM_PASS":
        _append_reason(risk, "program_gate_not_pass")
    if row.get("program_errors"):
        _append_reason(risk, "program_or_static_blocker")
    if candidate.get("schema_valid") is not True:
        _append_reason(risk, "candidate_schema_invalid")
    if candidate.get("schema_errors"):
        _append_reason(risk, "candidate_schema_errors")
    required_reasoning = candidate.get("required_reasoning")
    reasoning_used = candidate.get("reasoning_groups_used")
    if not isinstance(required_reasoning, list) or not isinstance(reasoning_used, list):
        _append_reason(unclassifiable, "reasoning_group_metadata_missing")

    if invalid:
        status = "EVIDENCE_INVALID"
    elif unclassifiable:
        status = "EVIDENCE_UNCLASSIFIABLE"
    elif risk:
        status = "EVIDENCE_RISK"
    else:
        status = "EVIDENCE_CLEAN"
    split = str(row.get("split") or "")
    strict = split == "train" and status == "EVIDENCE_CLEAN"
    broad_only = split == "train" and status == "EVIDENCE_RISK"
    classification_core = {
        "candidate_id": str(row.get("candidate_id") or ""),
        "root_id": root_id,
        "split": split,
        "evidence_status": status,
        "evidence_risk_reasons": sorted(risk),
        "evidence_invalid_reasons": sorted(invalid),
        "evidence_unclassifiable_reasons": sorted(unclassifiable),
        "strict_eligible": strict,
        "broad_only_eligible": broad_only,
        "contract_alignment_status": alignment_status,
    }
    classification_hash = hashlib.sha256(canonical_json(classification_core).encode("utf-8")).hexdigest()
    return {
        **classification_core,
        "question_type": str(row.get("question_type") or ""),
        "generation_source": str(row.get("generation_source") or ""),
        "difficulty_bucket": str(row.get("difficulty_bucket") or ""),
        "position_mechanism": str(row.get("position_mechanism") or "unknown"),
        "candidate_pool_path": str(row.get("candidate_pool_path") or ""),
        "candidate_pool_sha256": str(row.get("candidate_pool_sha256") or ""),
        "dialogue_path": str(row.get("dialogue_path") or ""),
        "dialogue_sha256": str(row.get("dialogue_sha256") or ""),
        "difficulty_manifest_path": str(difficulty_manifest_path),
        "difficulty_manifest_sha256": difficulty_manifest_sha256,
        "classification_algorithm_version": CLASSIFICATION_ALGORITHM_VERSION,
        "stratification_version": STRATIFICATION_VERSION,
        "classification_hash": classification_hash,
    }


def _counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("evidence_status") or "") for row in rows)
    return {status: counts[status] for status in STATUSES}


def _cross_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    values = sorted({str(row.get(field) or "unknown") for row in rows})
    for value in values:
        result[value] = _counts(row for row in rows if str(row.get(field) or "unknown") == value)
    return result


def _root_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["root_id"])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for root_id in sorted(grouped):
        values = grouped[root_id]
        output[root_id] = {
            "split": str(values[0]["split"]),
            "candidate_count": len(values),
            "evidence_status_counts": _counts(values),
            "position_mechanism_counts": dict(sorted(Counter(str(row.get("position_mechanism") or "unknown") for row in values).items())),
            "risk_candidate_count": sum(row["evidence_status"] == "EVIDENCE_RISK" for row in values),
        }
    return output


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"output_artifact_exists:{path}")
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), _sha(path) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"output_artifact_exists:{path}")
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_text(path, text)
    atomic_text(path.with_name(path.name + ".sha256"), _sha(path) + "\n")


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"output_artifact_exists:{path}")
    atomic_text(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), _sha(path) + "\n")


def _preflight_report(source_run: Path, details: Mapping[str, Any], counts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preflight_report_version": f"{STRATIFICATION_VERSION}_preflight_v1",
        "stratification_version": STRATIFICATION_VERSION,
        "classification_algorithm_version": CLASSIFICATION_ALGORITHM_VERSION,
        "status": "PASS",
        "source_run": str(source_run),
        "source_run_read_only": True,
        "source_files": details["source_hashes"],
        "difficulty_count": counts["difficulty_count"],
        "hard_count": counts["hard_count"],
        "hard_by_split": counts["hard_by_split"],
        "expected_hard_count": EXPECTED_HARD_COUNT,
        "expected_hard_by_split": EXPECTED_HARD_BY_SPLIT,
        "lineage_pool_count": len(details["roots"]),
        "lineage_dialogue_count": len({str(row.get("dialogue_path")) for row in details["hard"].values()}),
        "candidate_ids_and_splits_verified": True,
        "source_hashes_verified": True,
        "candidate_pool_and_dialogue_lineage_verified": True,
        "solver_calls": 0,
        "external_api_calls": 0,
        "semantic_auditor_calls": 0,
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# B0 K=3 Hard Evidence Stratification",
        "",
        f"- Status: **{report['status']}**",
        f"- Hard candidates: `{report['hard_count']}`",
        f"- Train clean/risk/invalid/unclassifiable: `{report['train_evidence_status_counts']}`",
        f"- Dev clean/risk/invalid/unclassifiable: `{report['dev_evidence_status_counts']}`",
        f"- Test clean/risk/invalid/unclassifiable: `{report['test_evidence_status_counts']}`",
        f"- Train strict eligible: `{report['strict_eligible_count']}`",
        f"- Train broad-only risk: `{report['broad_only_risk_count']}`",
        f"- Train risk roots: `{report['train_risk_root_count']}`",
        "",
        "This is an offline evidence-packaging classification only. It made zero Solver, external API, and semantic-auditor calls.",
        "EVIDENCE_RISK does not assert that the gold answer is wrong or semantically invalid.",
        "No matched Strict/Broad training manifest, model training, or gold/evidence rewrite was performed.",
        "",
        "## Main Outputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["major_result_paths"])
    return "\n".join(lines) + "\n"


def run(*, source_run: Path, run_root: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"output_run_already_exists:{run_root}")
    if not source_run.is_dir():
        raise RuntimeError(f"source_run_missing:{source_run}")
    details, counts = _validate_source_inputs(source_run)
    _load_lineage(details["hard"])
    run_root.mkdir(parents=True, exist_ok=False)
    difficulty_path = source_run / "difficulty_manifest.jsonl"
    difficulty_sha = details["source_hashes"]["difficulty_manifest.jsonl"]
    classified: list[dict[str, Any]] = []
    lineage_by_dialogue: dict[str, set[str]] = {}
    for row in details["hard"].values():
        dialogue_path = str(Path(str(row["dialogue_path"])).resolve())
        if dialogue_path not in lineage_by_dialogue:
            payload = load_json(Path(dialogue_path))
            lineage_by_dialogue[dialogue_path] = {
                str(turn.get("turn_ref")) for turn in payload.get("turns") or []
            }
        classified.append(classify_evidence(
            row,
            valid_refs=lineage_by_dialogue[dialogue_path],
            difficulty_manifest_path=difficulty_path,
            difficulty_manifest_sha256=difficulty_sha,
        ))
    classified.sort(key=lambda row: (row["root_id"], row["candidate_id"]))
    if len(classified) != EXPECTED_HARD_COUNT or len({row["candidate_id"] for row in classified}) != EXPECTED_HARD_COUNT:
        raise RuntimeError("classified_hard_identity_count_mismatch")

    status_counts = _counts(classified)
    by_split = {split: _counts(row for row in classified if row["split"] == split) for split in SPLITS}
    strict_rows = [row for row in classified if row["strict_eligible"]]
    broad_rows = [row for row in classified if row["broad_only_eligible"]]
    invalid_rows = [row for row in classified if row["split"] == "train" and row["evidence_status"] == "EVIDENCE_INVALID"]
    unclass_rows = [row for row in classified if row["split"] == "train" and row["evidence_status"] == "EVIDENCE_UNCLASSIFIABLE"]
    risk_reason_counts = Counter(
        reason for row in classified for reason in row["evidence_risk_reasons"]
        if row["evidence_status"] == "EVIDENCE_RISK"
    )
    invalid_reason_counts = Counter(reason for row in classified for reason in row["evidence_invalid_reasons"])
    unclass_reason_counts = Counter(reason for row in classified for reason in row["evidence_unclassifiable_reasons"])
    mechanism_by_status = {
        status: dict(sorted(Counter(str(row["position_mechanism"]) for row in classified if row["evidence_status"] == status).items()))
        for status in STATUSES
    }
    mechanism_by_split_status = {
        split: {
            status: dict(sorted(Counter(str(row["position_mechanism"]) for row in classified if row["split"] == split and row["evidence_status"] == status).items()))
            for status in STATUSES
        }
        for split in SPLITS
    }
    root_status = _root_summary(classified)
    train_risk_roots = sorted({row["root_id"] for row in classified if row["split"] == "train" and row["evidence_status"] == "EVIDENCE_RISK"})
    final_status = "COMPLETE"
    major_paths = [
        str(run_root / name) for name in (
            "preflight_report.json", "hard_evidence_status_manifest.jsonl", "hard_evidence_summary.json",
            "train_strict_eligible_pool.jsonl", "train_broad_only_risk_pool.jsonl", "train_evidence_invalid_pool.jsonl",
            "train_evidence_unclassifiable_pool.jsonl", "dev_hard_evidence_slices.jsonl", "test_hard_evidence_slices.jsonl",
            "hard_mechanism_by_evidence.json", "evidence_risk_reason_counts.json", "evidence_status_by_root.json",
            "final_stratification_report.json", "final_stratification_report.md",
        )
    ]
    preflight = _preflight_report(source_run, details, counts)
    summary = {
        "hard_evidence_summary_version": f"{STRATIFICATION_VERSION}_summary_v1",
        "stratification_version": STRATIFICATION_VERSION,
        "classification_algorithm_version": CLASSIFICATION_ALGORITHM_VERSION,
        "status": final_status,
        "source_run": str(source_run),
        "source_file_sha256": details["source_hashes"],
        "hard_count": len(classified),
        "evidence_status_counts": status_counts,
        "evidence_status_by_split": by_split,
        "strict_eligible_count": len(strict_rows),
        "broad_only_risk_count": len(broad_rows),
        "train_evidence_invalid_count": len(invalid_rows),
        "train_evidence_unclassifiable_count": len(unclass_rows),
        "train_position_mechanism_counts": dict(sorted(Counter(row["position_mechanism"] for row in classified if row["split"] == "train").items())),
        "dev_position_mechanism_counts": dict(sorted(Counter(row["position_mechanism"] for row in classified if row["split"] == "dev").items())),
        "test_position_mechanism_counts": dict(sorted(Counter(row["position_mechanism"] for row in classified if row["split"] == "test").items())),
        "risk_reason_counts": dict(sorted(risk_reason_counts.items())),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "unclassifiable_reason_counts": dict(sorted(unclass_reason_counts.items())),
        "train_risk_root_count": len(train_risk_roots),
        "train_risk_root_ids": train_risk_roots,
        "solver_calls": 0,
        "external_api_calls": 0,
        "semantic_auditor_calls": 0,
    }
    mechanism = {
        "hard_mechanism_by_evidence_version": f"{STRATIFICATION_VERSION}_mechanism_v1",
        "status": final_status,
        "by_evidence_status": mechanism_by_status,
        "by_split_and_evidence_status": mechanism_by_split_status,
        "train_hard_position_mechanism_counts": summary["train_position_mechanism_counts"],
    }
    risk_reasons = {
        "evidence_risk_reason_counts_version": f"{STRATIFICATION_VERSION}_risk_reasons_v1",
        "status": final_status,
        "risk_reason_counts": dict(sorted(risk_reason_counts.items())),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "unclassifiable_reason_counts": dict(sorted(unclass_reason_counts.items())),
    }
    root_report = {
        "evidence_status_by_root_version": f"{STRATIFICATION_VERSION}_root_v1",
        "status": final_status,
        "roots": root_status,
    }
    final = {
        "final_stratification_report_version": f"{STRATIFICATION_VERSION}_final_v1",
        "stratification_version": STRATIFICATION_VERSION,
        "classification_algorithm_version": CLASSIFICATION_ALGORITHM_VERSION,
        "status": final_status,
        "source_run": str(source_run),
        "source_file_sha256": details["source_hashes"],
        "hard_count": len(classified),
        "expected_hard_count": EXPECTED_HARD_COUNT,
        "train_hard_count": by_split["train"]["EVIDENCE_CLEAN"] + by_split["train"]["EVIDENCE_RISK"] + by_split["train"]["EVIDENCE_INVALID"] + by_split["train"]["EVIDENCE_UNCLASSIFIABLE"],
        "dev_hard_count": sum(by_split["dev"].values()),
        "test_hard_count": sum(by_split["test"].values()),
        "train_evidence_status_counts": by_split["train"],
        "dev_evidence_status_counts": by_split["dev"],
        "test_evidence_status_counts": by_split["test"],
        "strict_eligible_count": len(strict_rows),
        "broad_only_risk_count": len(broad_rows),
        "train_invalid_count": len(invalid_rows),
        "train_unclassifiable_count": len(unclass_rows),
        "risk_reason_counts": dict(sorted(risk_reason_counts.items())),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "unclassifiable_reason_counts": dict(sorted(unclass_reason_counts.items())),
        "question_type_by_evidence": _cross_counts(classified, "question_type"),
        "generation_source_by_evidence": _cross_counts(classified, "generation_source"),
        "position_mechanism_by_evidence": mechanism_by_status,
        "train_position_mechanism_counts": summary["train_position_mechanism_counts"],
        "train_risk_root_count": len(train_risk_roots),
        "train_risk_root_ids": train_risk_roots,
        "strict_vs_broad_ablation_quantity": {
            "strict_eligible_count": len(strict_rows),
            "broad_only_risk_count": len(broad_rows),
            "quantity_sufficient_for_planning": bool(strict_rows and broad_rows),
            "decision": "SUFFICIENT_FOR_PLANNING_ONLY" if strict_rows and broad_rows else "INSUFFICIENT",
            "training_started": False,
            "matched_manifest_created": False,
            "semantic_validation_required_before_training": True,
        },
        "major_result_paths": major_paths,
        "solver_calls": 0,
        "external_api_calls": 0,
        "semantic_auditor_calls": 0,
        "gold_modified": False,
        "evidence_modified": False,
        "source_run_modified": False,
    }

    _write_json(run_root / "preflight_report.json", preflight)
    _write_jsonl(run_root / "hard_evidence_status_manifest.jsonl", classified)
    _write_json(run_root / "hard_evidence_summary.json", summary)
    _write_jsonl(run_root / "train_strict_eligible_pool.jsonl", strict_rows)
    _write_jsonl(run_root / "train_broad_only_risk_pool.jsonl", broad_rows)
    _write_jsonl(run_root / "train_evidence_invalid_pool.jsonl", invalid_rows)
    _write_jsonl(run_root / "train_evidence_unclassifiable_pool.jsonl", unclass_rows)
    _write_jsonl(run_root / "dev_hard_evidence_slices.jsonl", [row for row in classified if row["split"] == "dev"])
    _write_jsonl(run_root / "test_hard_evidence_slices.jsonl", [row for row in classified if row["split"] == "test"])
    _write_json(run_root / "hard_mechanism_by_evidence.json", mechanism)
    _write_json(run_root / "evidence_risk_reason_counts.json", risk_reasons)
    _write_json(run_root / "evidence_status_by_root.json", root_report)
    _write_json(run_root / "final_stratification_report.json", final)
    _write_text(run_root / "final_stratification_report.md", _markdown_report(final))
    return final


__all__ = [
    "CLASSIFICATION_ALGORITHM_VERSION",
    "REQUIRED_SOURCE_FILES",
    "classify_evidence",
    "run",
]
