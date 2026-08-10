from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .minimal_admission import (SPAN_IDS_ONLY_SCHEMA_PROFILE,
                                SPELL_STYLE_MINIMAL_MODE,
                                materialize_spell_style_minimal)
from .minimal_replay import verify_bad_ground_permutation_source
from .validation import materialize_questioner_item


SEMANTIC_LEAKAGE_REPAIR_VERSION = "semantic_leakage_repair_v1"
SEMANTIC_LEAKAGE_REPAIR_REASON = "bad_ground_semantic_leakage"


def semantic_repair_eligibility(*, original_final_decision: str,
                                diagnostic_decision: str,
                                repair_attempt_count: int,
                                source_authoritative: bool,
                                source_provenance_mixed: bool) -> dict[str, Any]:
    reasons = []
    if original_final_decision != "bad_ground":
        reasons.append("semantic_repair_requires_bad_ground")
    if diagnostic_decision != SEMANTIC_LEAKAGE_REPAIR_REASON:
        reasons.append("semantic_repair_requires_semantic_leakage")
    if repair_attempt_count != 0:
        reasons.append("semantic_repair_attempt_exhausted")
    if not source_authoritative or source_provenance_mixed:
        reasons.append("semantic_repair_source_not_authoritative")
    return {"eligible":not reasons, "reasons":reasons,
        "repair_attempt_count":repair_attempt_count,
        "maximum_repair_attempts":1}


def _verify_rollout_ledgers(root: Path, rows: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for row in rows:
        request_id = row.get("request_id")
        ledger_path = root / f"ledger/requests/{request_id}.json"
        if not isinstance(request_id, str) or not ledger_path.is_file():
            raise RuntimeError("semantic_repair_source_ledger_integrity_failed")
        ledger = read_json(ledger_path)
        raw_path = Path(str(ledger.get("raw_response_path") or ""))
        if (ledger.get("status") != "completed" or ledger.get("request_id") != request_id or
                not raw_path.is_file() or sha256_file(raw_path) != ledger.get("raw_response_sha256") or
                raw_path.read_text(encoding="utf-8") != row.get("raw_response")):
            raise RuntimeError("semantic_repair_source_ledger_integrity_failed")
        hashes[request_id] = sha256_file(ledger_path)
    aggregate = sha256_bytes(canonical_json(hashes).encode())
    return aggregate, hashes


def verify_semantic_repair_sources(*, solver_root: Path, diagnostic_root: Path,
                                   qa_id: str) -> dict[str, Any]:
    solver = verify_bad_ground_permutation_source(solver_root, qa_id)
    diagnostic_root = diagnostic_root.resolve()
    manifest_path = diagnostic_root / "config/run_manifest.json"
    provenance_path = diagnostic_root / "reports/post_run_provenance.json"
    report_path = diagnostic_root / "reports/no_context_permutation_diagnostic.json"
    if any(not path.is_file() for path in (manifest_path, provenance_path, report_path)):
        raise RuntimeError("semantic_repair_diagnostic_source_missing")
    manifest, provenance, report = (read_json(manifest_path), read_json(provenance_path),
                                    read_json(report_path))
    manifest_sha = sha256_file(manifest_path)
    result = report.get("result") or {}
    diagnostic = result.get("diagnostic") or {}
    if (provenance.get("run_manifest_sha256") != manifest_sha or
            provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False or
            report.get("qa_id") != qa_id or result.get("qa_id") != qa_id or
            result.get("authoritative_grounding_decision") != "bad_ground" or
            result.get("authoritative_grounding_decision_changed") is not False or
            result.get("original_no_context_reused") is not True or
            diagnostic.get("decision") != SEMANTIC_LEAKAGE_REPAIR_REASON):
        raise RuntimeError("semantic_repair_diagnostic_source_not_authoritative")
    if (manifest.get("source_bad_ground_solver_root") != str(Path(solver["source_root"])) or
            manifest.get("source_bad_ground_manifest_sha256") != solver["source_manifest_sha256"] or
            manifest.get("source_bad_ground_report_sha256") != solver["source_report_sha256"] or
            manifest.get("source_bad_ground_qa_wrapper_sha256") != solver[
                "source_qa_wrapper_sha256"]):
        raise RuntimeError("semantic_repair_source_binding_mismatch")
    new_rows = [row for order in result.get("permutation_orders") or []
                for row in (order.get("rollout") or {}).get("rollouts") or []]
    diagnostic_ledger_sha, diagnostic_ledger_hashes = _verify_rollout_ledgers(
        diagnostic_root, new_rows)
    original_rows = list(solver["original_no_context"].get("rollouts") or [])
    solver_ledger_sha, solver_ledger_hashes = _verify_rollout_ledgers(
        Path(solver["source_root"]), original_rows)
    original_raw = solver["frozen_qa"].get("source_generated_qa")
    admission = solver["frozen_qa"].get("schema_acceptance")
    target_letter = (admission or {}).get("target_letter")
    if (not isinstance(original_raw, dict) or original_raw.get("qa_id") != qa_id or
            not isinstance(target_letter, str) or target_letter not in "ABCD"):
        raise RuntimeError("semantic_repair_source_qa_integrity_failed")
    eligibility = semantic_repair_eligibility(original_final_decision="bad_ground",
        diagnostic_decision=diagnostic["decision"], repair_attempt_count=0,
        source_authoritative=True, source_provenance_mixed=False)
    if not eligibility["eligible"]:
        raise RuntimeError(f"semantic repair not eligible: {eligibility['reasons']}")
    return {**solver, "diagnostic_root":str(diagnostic_root),
        "diagnostic_manifest_sha256":manifest_sha,
        "diagnostic_provenance_sha256":sha256_file(provenance_path),
        "diagnostic_report_sha256":sha256_file(report_path),
        "solver_ledger_aggregate_sha256":solver_ledger_sha,
        "solver_ledger_sha256":solver_ledger_hashes,
        "diagnostic_ledger_aggregate_sha256":diagnostic_ledger_sha,
        "diagnostic_ledger_sha256":diagnostic_ledger_hashes,
        "original_raw_item":original_raw, "target_letter":target_letter,
        "permutation_diagnostic":diagnostic, "eligibility":eligibility}


def admit_semantic_repair_item(raw: Any, *, expected_qa_id: str, target_letter: str,
                               span_catalog: dict[str, Any],
                               reasoning_contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"minimal_decision":"reject", "minimal_reasons":["bad_format"],
            "strict_v7_decision":"reject", "strict_v7_reasons":["bad_format"],
            "materialized_qa":None}
    minimal_qa, minimal_errors, diagnostics = materialize_spell_style_minimal(
        raw, target_letter, span_catalog=span_catalog, reasoning_contract=reasoning_contract,
        schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
    if raw.get("qa_id") != expected_qa_id:
        minimal_errors = list(dict.fromkeys([*minimal_errors, "repair_qa_id_mismatch"]))
        minimal_qa = None
    strict_qa, strict_errors = materialize_questioner_item(raw, target_letter,
        span_catalog=span_catalog, reasoning_contract=reasoning_contract)
    return {"minimal_decision":"accept" if not minimal_errors and minimal_qa else "reject",
        "minimal_reasons":minimal_errors,
        "strict_v7_decision":"accept" if not strict_errors and strict_qa else "reject",
        "strict_v7_reasons":strict_errors, "minimal_diagnostics":diagnostics,
        "materialized_qa":minimal_qa, "strict_materialized_qa":strict_qa}


def build_semantic_repair_feedback_summary(results: list[dict[str, Any]],
                                           source_item_count: int = 3) -> dict[str, Any]:
    decisions = [str(item.get("final_decision") or "") for item in results]
    return {"source_failure_type":SEMANTIC_LEAKAGE_REPAIR_REASON,
        "source_item_count":source_item_count, "repair_attempted":len(results),
        "repair_json_valid":sum(bool(item.get("repair_json_valid")) for item in results),
        "repair_minimal_accepted":sum(bool(item.get("repair_minimal_accepted")) for item in results),
        "repair_passed_grounding":sum(bool(item.get("repair_passed_grounding")) for item in results),
        "repair_reject_easy":sum(value == "reject_easy_after_semantic_repair" for value in decisions),
        "repair_reject_all_wrong":sum(value == "reject_all_wrong_after_semantic_repair" for value in decisions),
        "repair_retain_frontier":sum(value == "retain_frontier" for value in decisions)}
