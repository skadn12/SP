"""Create a reviewer control envelope after the immutable public packet is frozen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "current_method"))
sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "research-protocols"
        / "selfplay_week_20261019"
        / "base_qa_v2_2_1_acceptance_20261102"
        / "implementation"
    ),
)

import base_qa_option_freeze_v2_2_1 as freeze  # noqa: E402


CONTROL_FILE = "BLIND_REQUEST_CONTROL.json"
CONTROL_FREEZE_FILE = "BLIND_REQUEST_CONTROL_FREEZE.json"
REQUEST_FILE = "BLIND_REVIEW_REQUEST.json"
REQUEST_FREEZE_FILE = "BLIND_REVIEW_REQUEST_FREEZE.json"


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8", errors="strict")), raw


def _verify_evidence_projection(bundle: dict, source_path: Path, scope_path: Path) -> tuple[dict, dict, str, str]:
    source_view, source_raw = _read_json(source_path)
    scope_view, scope_raw = _read_json(scope_path)
    original_source = bundle["source"]
    original_scope = bundle["source_scope"]
    allowed_refs = original_scope.get("formal_qa_eligible_turn_refs")
    if type(allowed_refs) is not list or len(allowed_refs) != 82 or len(set(allowed_refs)) != 82:
        raise ValueError("SOURCE_SCOPE_ALLOWED_R02_SET_INVALID")
    if source_view.get("root_id") != bundle["candidate"].get("root_id"):
        raise ValueError("EVIDENCE_VIEW_ROOT_MISMATCH")
    if scope_view.get("root_id") != source_view.get("root_id"):
        raise ValueError("EVIDENCE_SCOPE_ROOT_MISMATCH")
    if (scope_view.get("admission_decision") != "READY"
            or scope_view.get("selected_turns_formally_allowed_by_bound_receipt") is not True
            or scope_view.get("formal_qa_use_authorized_by_caller_scope_record") is not True
            or scope_view.get("projection_is_new_admission") is not False):
        raise ValueError("EVIDENCE_SCOPE_AUTHORIZATION_CLAIMS_INVALID")
    source_binding = scope_view.get("evidence_source_binding")
    if type(source_binding) is not dict:
        raise ValueError("EVIDENCE_SOURCE_BINDING_MISSING")
    if (source_binding.get("sha256") != _sha256(source_raw)
            or source_binding.get("bytes") != len(source_raw)):
        raise ValueError("EVIDENCE_SOURCE_BINDING_MISMATCH")
    bound_source_path = Path(source_binding.get("path", ""))
    if not bound_source_path.is_absolute():
        bound_source_path = PROJECT_ROOT / bound_source_path
    if bound_source_path.resolve() != source_path.resolve():
        raise ValueError("EVIDENCE_SOURCE_PATH_MISMATCH")

    original_by_ref = {row["turn_ref"]: row for row in original_source.get("turns", [])}
    source_turns = source_view.get("turns")
    scope_rows = scope_view.get("turn_admission_records")
    if type(source_turns) is not list or type(scope_rows) is not list:
        raise ValueError("EVIDENCE_VIEW_TURNS_INVALID")
    view_refs = [row.get("turn_ref") for row in source_turns if type(row) is dict]
    if (len(source_turns) != 82 or view_refs != allowed_refs
            or len(set(view_refs)) != len(view_refs)):
        raise ValueError("EVIDENCE_VIEW_DOES_NOT_MATCH_AUTHORIZED_ALLOWLIST")
    if len(scope_rows) != 82 or [row.get("turn_ref") for row in scope_rows] != allowed_refs:
        raise ValueError("EVIDENCE_SCOPE_DOES_NOT_MATCH_AUTHORIZED_ALLOWLIST")
    for turn, admission in zip(source_turns, scope_rows):
        ref = turn["turn_ref"]
        if not ref.startswith("R02:") or original_by_ref.get(ref) != turn:
            raise ValueError("EVIDENCE_TURN_NOT_EXACT_UPSTREAM_TURN:" + str(ref))
        if (admission.get("turn_ref") != ref
                or admission.get("material_review_status") != "PASS"
                or admission.get("formal_qa_allowed") is not True
                or admission.get("eligibility_errors") != []):
            raise ValueError("EVIDENCE_TURN_NOT_ADMITTED:" + str(ref))
    return source_view, scope_view, _sha256(source_raw), _sha256(scope_raw)


def _reviewer_scope_projection(bundle: dict, source_view: dict, scope_view: dict,
                               source_sha: str, scope_sha: str) -> dict:
    original_turn_refs = [row["turn_ref"] for row in bundle["source"].get("turns", [])]
    allowed_refs = bundle["source_scope"]["formal_qa_eligible_turn_refs"]
    return {
        "schema_version": "base_qa_reviewer_scope_projection_v1",
        "root_id": source_view["root_id"],
        "source_id": source_view["source_id"],
        "canonical_source_sha256": source_view["canonical_source_sha256"],
        "canonical_source_scope_sha256": bundle["packet"]["source_scope_sha256"],
        "reviewer_evidence_source_sha256": source_sha,
        "reviewer_evidence_scope_artifact_sha256": scope_sha,
        "admission_decision": scope_view["admission_decision"],
        "admission_reviewer_mode": scope_view["admission_reviewer_mode"],
        "reviewer_external_certification": scope_view["reviewer_external_certification"],
        "selected_turns_formally_allowed_by_bound_receipt": True,
        "formal_qa_use_authorized_by_caller_scope_record": True,
        "projection_is_new_admission": False,
        "original_source_turn_count": len(original_turn_refs),
        "included_turn_count": len(allowed_refs),
        "excluded_context_turn_count": len(original_turn_refs) - len(allowed_refs),
        "included_turn_refs": allowed_refs,
        "excluded_context_turn_refs": [ref for ref in original_turn_refs if ref not in set(allowed_refs)],
        "turn_admission_records": scope_view["turn_admission_records"],
        "limitations": scope_view["limitations"],
    }


def build_control_envelope(run_dir: Path, evidence_source_path: Path, evidence_scope_path: Path) -> dict:
    bundle = freeze.load_bundle(run_dir)
    candidate = bundle["candidate"]
    packet = bundle["packet"]
    receipt = bundle["orchestration_receipt"]
    projection = freeze.reviewer_receipt_projection(run_dir)
    source_view, scope_view, evidence_source_sha256, evidence_scope_sha256 = _verify_evidence_projection(
        bundle, Path(evidence_source_path), Path(evidence_scope_path))
    reviewer_scope = _reviewer_scope_projection(
        bundle, source_view, scope_view, evidence_source_sha256, evidence_scope_sha256)
    return {
        "schema_version": "base_qa_blind_request_control_v1",
        "candidate_id": candidate["candidate_id"],
        "public_packet_sha256": packet["packet_sha256"],
        "source_scope_sha256": packet["source_scope_sha256"],
        "freeze_envelope_sha256": _sha256((Path(run_dir) / freeze.ENVELOPE).read_bytes()),
        "orchestration_receipt_sha256": freeze.fingerprint(receipt),
        "reviewer_id": receipt["reviewer_id"],
        "reviewer_id_semantics": "logical_review_assignment_id",
        "reviewer_mode": "same_model_blind",
        "model_id": "unavailable_exact_model_identifier",
        "isolation_receipt": projection,
        "reviewer_evidence_source_sha256": evidence_source_sha256,
        "reviewer_evidence_scope_artifact_sha256": evidence_scope_sha256,
        "reviewer_scope_payload_sha256": freeze.fingerprint(reviewer_scope),
        "author_material_visible": False,
        "reviewer_instructions": [
            "Treat reviewer_id and isolation_receipt as coordinator-supplied control metadata, not question content.",
            "Copy reviewer_id and isolation_receipt exactly into the response requested by the public packet.",
            "The reviewer_id names the logical review assignment; it is not the runtime agent path.",
            "Do not add author Gold, repair a response, or retry after a schema failure.",
        ],
        "author_gold_included": False,
        "author_certificate_included": False,
    }


def build_reviewer_request(run_dir: Path, evidence_source_path: Path, evidence_scope_path: Path) -> dict:
    bundle = freeze.load_bundle(run_dir)
    source_view, scope_view, source_sha, scope_sha = _verify_evidence_projection(
        bundle, Path(evidence_source_path), Path(evidence_scope_path))
    control = build_control_envelope(run_dir, evidence_source_path, evidence_scope_path)
    packet = bundle["packet"]["packet"]
    public_question = {
        key: packet[key]
        for key in ("stage", "instruction", "candidate_id", "reviewer_mode", "question_type", "question", "options")
    }
    reviewer_scope = _reviewer_scope_projection(bundle, source_view, scope_view, source_sha, scope_sha)
    return {
        "schema_version": "base_qa_blind_review_request_v1",
        "control": control,
        "canonical_public_packet_sha256": bundle["packet"]["packet_sha256"],
        "canonical_source_scope_sha256": bundle["packet"]["source_scope_sha256"],
        "reviewer_evidence_source_sha256": source_sha,
        "reviewer_evidence_scope_artifact_sha256": scope_sha,
        "reviewer_scope_payload_sha256": freeze.fingerprint(reviewer_scope),
        "question_packet": public_question,
        "source": source_view,
        "source_scope": reviewer_scope,
        "author_gold_included": False,
        "author_certificate_included": False,
        "author_audit_included": False,
    }


def _freeze_json_artifact(path: Path, freeze_path: Path, value: dict, schema_version: str) -> dict:
    raw = _json_bytes(value)
    if path.exists() or freeze_path.exists():
        if not path.exists() or not freeze_path.exists():
            raise RuntimeError("partial reviewer artifact freeze exists")
        existing_raw = path.read_bytes()
        existing_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        expected_freeze = {"schema_version": schema_version + ":freeze", "bytes": len(raw), "sha256": _sha256(raw)}
        if existing_raw != raw or existing_freeze != expected_freeze:
            raise RuntimeError("refusing to overwrite changed reviewer artifact")
        return value
    path.write_bytes(raw)
    freeze_path.write_bytes(_json_bytes({
        "schema_version": schema_version + ":freeze", "bytes": len(raw), "sha256": _sha256(raw)}))
    return value


def freeze_control_envelope(run_dir: Path, evidence_source_path: Path, evidence_scope_path: Path) -> dict:
    run_dir = Path(run_dir)
    control = build_control_envelope(run_dir, evidence_source_path, evidence_scope_path)
    _freeze_json_artifact(run_dir / CONTROL_FILE, run_dir / CONTROL_FREEZE_FILE, control,
                          "base_qa_blind_request_control_v1")
    request = build_reviewer_request(run_dir, evidence_source_path, evidence_scope_path)
    return _freeze_json_artifact(run_dir / REQUEST_FILE, run_dir / REQUEST_FREEZE_FILE, request,
                                 "base_qa_blind_review_request_v1")


def verify_control_envelope(run_dir: Path, evidence_source_path: Path, evidence_scope_path: Path) -> dict:
    run_dir = Path(run_dir)
    raw = (run_dir / CONTROL_FILE).read_bytes()
    stored = json.loads(raw)
    freeze_record = json.loads((run_dir / CONTROL_FREEZE_FILE).read_text(encoding="utf-8"))
    expected = build_control_envelope(run_dir, evidence_source_path, evidence_scope_path)
    if stored != expected:
        raise ValueError("BLIND_REQUEST_CONTROL_BINDING_MISMATCH")
    if freeze_record != {
        "schema_version": "base_qa_blind_request_control_v1:freeze",
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }:
        raise ValueError("BLIND_REQUEST_CONTROL_FREEZE_MISMATCH")
    request_raw = (run_dir / REQUEST_FILE).read_bytes()
    request = json.loads(request_raw)
    expected_request = build_reviewer_request(run_dir, evidence_source_path, evidence_scope_path)
    request_freeze = json.loads((run_dir / REQUEST_FREEZE_FILE).read_text(encoding="utf-8"))
    if request != expected_request:
        raise ValueError("BLIND_REVIEW_REQUEST_BINDING_MISMATCH")
    if request_freeze != {
        "schema_version": "base_qa_blind_review_request_v1:freeze",
        "bytes": len(request_raw),
        "sha256": _sha256(request_raw),
    }:
        raise ValueError("BLIND_REVIEW_REQUEST_FREEZE_MISMATCH")
    return stored
