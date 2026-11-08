#!/usr/bin/env python3
"""Build hash-bound projections containing only the already admitted R02 turns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WEEK_ROOT = PROJECT_ROOT / "research-protocols" / "selfplay_week_20261019"
ACCEPTANCE_ROOT = WEEK_ROOT / "base_qa_v2_2_1_acceptance_20261102"
ADMISSION_ROOT = WEEK_ROOT / "base_qa_v2_2_1_real_20261102" / "admission"
OUTPUT_ROOT = ACCEPTANCE_ROOT / "source_view_refresh_20261102"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def encoded_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_immutable(path: Path, value: dict) -> bytes:
    raw = encoded_json(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite different artifact: {path}")
        return raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build_root(root_id: str, caller_authorization: dict, caller_authorization_sha: str) -> dict:
    source_dir = ADMISSION_ROOT / root_id
    source_path = source_dir / "FORMAL_SOURCE_VIEW.json"
    source_scope_path = source_dir / "SOURCE_SCOPE.json"
    admission_path = source_dir / "MATERIAL_ADMISSION.json"
    validation_path = source_dir / "ADMISSION_VALIDATION.json"
    source, source_raw = read_json(source_path)
    source_scope, source_scope_raw = read_json(source_scope_path)
    admission, admission_raw = read_json(admission_path)
    admission_validation, _ = read_json(validation_path)
    source_sha = sha256_bytes(source_raw)
    source_scope_sha = sha256_bytes(source_scope_raw)
    admission_sha = sha256_bytes(admission_raw)

    authorization_rows = {row["root_id"]: row for row in caller_authorization["roots"]}
    authorization = authorization_rows[root_id]
    checks = {
        "root_matches": source["root_id"] == root_id == source_scope["root_id"] == admission["root_id"],
        "caller_source_hash_matches": authorization["formal_source_view_sha256"] == source_sha,
        "caller_scope_hash_matches": authorization["source_scope_sha256"] == source_scope_sha,
        "caller_admission_hash_matches": authorization["admission_receipt_sha256"] == admission_sha,
        "scope_source_binding_matches": source_scope["source_binding"]["sha256"] == source_sha,
        "scope_admission_binding_matches": source_scope["admission_sha256"] == admission_sha,
        "admission_source_hash_matches": admission["source_sha256"] == source_sha,
        "admission_ready": admission["decision"] == "READY",
        "admission_validation_complete": admission_validation["admission_complete"] is True,
        "allowed_turn_gate_passes": admission_validation["formal_evidence_gate_for_all_allowed_R02_turns"]["eligible"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"upstream binding/gate failure for {root_id}: {checks}")

    allowed_refs = source_scope["formal_qa_eligible_turn_refs"]
    if len(allowed_refs) != 82 or len(set(allowed_refs)) != 82:
        raise RuntimeError(f"expected exactly 82 unique authorized refs for {root_id}")
    if authorization["allowed_turn_count"] != 82:
        raise RuntimeError(f"caller authorization does not bind 82 refs for {root_id}")
    explicitly_excluded = set(authorization["excluded_turn_refs"])
    if explicitly_excluded.intersection(allowed_refs):
        raise RuntimeError(f"excluded refs found in allowed set for {root_id}")

    source_turns = source["turns"]
    source_by_ref = {row["turn_ref"]: row for row in source_turns}
    scope_rows = {row["turn_id"]: row for row in source_scope["turns"]}
    admission_rows = {row["turn_id"]: row for row in admission["turns"]}
    if len(source_by_ref) != len(source_turns):
        raise RuntimeError(f"duplicate source turn refs in {root_id}")
    if len(scope_rows) != len(source_scope["turns"]):
        raise RuntimeError(f"duplicate scope turn refs in {root_id}")

    ordered_refs = [row["turn_ref"] for row in source_turns if row["turn_ref"] in set(allowed_refs)]
    if len(ordered_refs) != 82 or set(ordered_refs) != set(allowed_refs):
        raise RuntimeError(f"authorized refs do not map exactly to source turns for {root_id}")

    projected_turns = []
    projected_scope_rows = []
    for turn_ref in ordered_refs:
        source_turn = source_by_ref[turn_ref]
        scope_row = scope_rows.get(turn_ref)
        admission_row = admission_rows.get(turn_ref)
        if scope_row is None or admission_row is None:
            raise RuntimeError(f"missing per-turn admission records for {root_id} {turn_ref}")
        if scope_row["material_review_status"] != "PASS" or scope_row["formal_qa_allowed"] is not True:
            raise RuntimeError(f"selected turn not PASS/allowed: {root_id} {turn_ref}")
        if scope_row["eligibility_errors"] or admission_row["review_status"] != "PASS":
            raise RuntimeError(f"selected turn has admission errors or non-PASS receipt: {root_id} {turn_ref}")
        if admission_row["formal_qa_allowed"] is not True:
            raise RuntimeError(f"admission receipt disallows selected turn: {root_id} {turn_ref}")
        if not turn_ref.startswith("R02:"):
            raise RuntimeError(f"non-R02 turn found in allowed set: {root_id} {turn_ref}")

        review_path = Path(scope_row["material_review_record"])
        review_raw = review_path.read_bytes()
        review_sha = sha256_bytes(review_raw)
        review_binding = admission_row["review_record_binding"]
        if review_binding["sha256"] != review_sha or review_binding["bytes"] != len(review_raw):
            raise RuntimeError(f"review record binding mismatch: {root_id} {turn_ref}")
        if relative(review_path) != relative(Path(admission_row["review_record"])):
            raise RuntimeError(f"review record path mismatch: {root_id} {turn_ref}")

        projected_turns.append(source_turn)
        projected_scope_rows.append(
            {
                "turn_ref": turn_ref,
                "source_id": scope_row["source_id"],
                "source_provenance_label": scope_row["source_provenance_label"],
                "provenance_type": scope_row["provenance_type"],
                "source_version": scope_row["source_version"],
                "material_review_status": scope_row["material_review_status"],
                "formal_qa_allowed": scope_row["formal_qa_allowed"],
                "eligibility_errors": scope_row["eligibility_errors"],
                "review_record_binding": {
                    "path": relative(review_path),
                    "sha256": review_sha,
                    "bytes": len(review_raw),
                },
            }
        )

    evidence_source = {
        "schema_version": "base_qa_exact_evidence_view_v1",
        "root_id": root_id,
        "source_id": source["turns"][0]["source_id"],
        "canonical_source_sha256": authorization["canonical_source_sha256"],
        "upstream_formal_source_sha256": source_sha,
        "upstream_scope_sha256": source_scope_sha,
        "material_admission_sha256": admission_sha,
        "caller_authorization_sha256": caller_authorization_sha,
        "turn_count": len(projected_turns),
        "turns": projected_turns,
    }
    root_out = OUTPUT_ROOT / root_id
    evidence_path = root_out / "EVIDENCE_SOURCE.json"
    evidence_raw = write_immutable(evidence_path, evidence_source)
    evidence_sha = sha256_bytes(evidence_raw)
    scope_projection = {
        "schema_version": "base_qa_exact_evidence_scope_projection_v1",
        "root_id": root_id,
        "source_id": evidence_source["source_id"],
        "evidence_source_binding": {
            "path": relative(evidence_path),
            "sha256": evidence_sha,
            "bytes": len(evidence_raw),
        },
        "upstream_bindings": {
            "canonical_source_sha256": authorization["canonical_source_sha256"],
            "formal_source_view": {"path": relative(source_path), "sha256": source_sha},
            "source_scope": {"path": relative(source_scope_path), "sha256": source_scope_sha},
            "material_admission": {"path": relative(admission_path), "sha256": admission_sha},
            "caller_authorization": {
                "path": relative(ACCEPTANCE_ROOT / "USER_SCOPE_CONFIRMATION.json"),
                "sha256": caller_authorization_sha,
            },
        },
        "admission_decision": admission["decision"],
        "admission_reviewer_mode": admission["reviewer_mode"],
        "reviewer_external_certification": admission["reviewer_external_certification"],
        "projection_is_new_admission": False,
        "selected_turns_formally_allowed_by_bound_receipt": True,
        "formal_qa_use_authorized_by_caller_scope_record": True,
        "all_source_view_turns_formally_allowed": False,
        "original_source_turn_count": len(source_turns),
        "included_turn_count": len(projected_turns),
        "excluded_context_turn_count": len(source_turns) - len(projected_turns),
        "excluded_turn_refs": sorted(explicitly_excluded),
        "included_turn_refs": ordered_refs,
        "turn_admission_records": projected_scope_rows,
        "checks": checks,
        "limitations": [
            "This is a deterministic projection of existing PASS/allowed turns, not a new independent review or admission receipt.",
            "The source review is same-model Codex review and has no human or external certification.",
            "The projection contains only approved evidence turns; it does not preserve the full conversation context.",
        ],
    }
    scope_path = root_out / "EVIDENCE_SCOPE.json"
    scope_raw = write_immutable(scope_path, scope_projection)

    validation = {
        "schema_version": "base_qa_exact_evidence_view_validation_v1",
        "root_id": root_id,
        "evidence_source_sha256": evidence_sha,
        "evidence_scope_sha256": sha256_bytes(scope_raw),
        "source_turn_count": len(projected_turns),
        "scope_turn_count": len(projected_scope_rows),
        "all_refs_unique": len({row["turn_ref"] for row in projected_turns}) == 82,
        "all_refs_exactly_match_authorized_set": set(ordered_refs) == set(allowed_refs),
        "all_refs_are_R02": all(ref.startswith("R02:") for ref in ordered_refs),
        "all_turns_PASS_and_allowed": all(
            row["material_review_status"] == "PASS" and row["formal_qa_allowed"] is True
            for row in projected_scope_rows
        ),
        "excluded_refs_absent": not explicitly_excluded.intersection(ordered_refs),
        "turn_contents_match_upstream_exactly": all(
            projected_turns[index] == source_by_ref[turn_ref]
            for index, turn_ref in enumerate(ordered_refs)
        ),
        "hash_bound_review_records_match": True,
        "formal_qa_use_authorized_by_caller_scope_record": True,
        "new_independent_admission_created": False,
    }
    integrity_checks = (
        "all_refs_unique",
        "all_refs_exactly_match_authorized_set",
        "all_refs_are_R02",
        "all_turns_PASS_and_allowed",
        "excluded_refs_absent",
        "turn_contents_match_upstream_exactly",
        "hash_bound_review_records_match",
        "formal_qa_use_authorized_by_caller_scope_record",
    )
    validation["pass"] = all(validation[key] is True for key in integrity_checks)
    validation["validation_note"] = (
        "PASS covers projection integrity and inherited per-turn authorization only. "
        "new_independent_admission_created=false is an explicit non-claim, not a failed check."
    )
    write_immutable(root_out / "VALIDATION_RECHECK.json", validation)
    return {
        "root_id": root_id,
        "source_path": relative(evidence_path),
        "source_sha256": evidence_sha,
        "scope_path": relative(scope_path),
        "scope_sha256": sha256_bytes(scope_raw),
        "turn_count": len(projected_turns),
        "excluded_context_turn_count": len(source_turns) - len(projected_turns),
        "validation_pass": validation["pass"],
    }


def main() -> None:
    caller_path = ACCEPTANCE_ROOT / "USER_SCOPE_CONFIRMATION.json"
    caller_authorization, caller_raw = read_json(caller_path)
    caller_sha = sha256_bytes(caller_raw)
    summaries = [
        build_root(root_id, caller_authorization, caller_sha)
        for root_id in ("cc-train-episode-103866", "cc-train-episode-104814")
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
