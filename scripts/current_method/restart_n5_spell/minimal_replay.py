from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import canonical_json, sha256_bytes, sha256_file
from .input_adapter import read_json
from .questioner_contract import reasoning_contract_sha256, span_catalog_sha256


MINIMAL_REPLAY_VERSION = "spell_style_minimal_replay_v1"


def verify_solver_feedback_source(source_root: Path, *, expected_qa_id: str,
                                  expected_decision: str) -> dict[str, Any]:
    """Attest a completed local Solver result before exposing it as prompt feedback."""
    source_root = source_root.resolve()
    manifest_path = source_root / "config/run_manifest.json"
    provenance_path = source_root / "reports/post_run_provenance.json"
    report_path = source_root / "reports/minimal_solver_full_audit.json"
    if any(not path.is_file() for path in (manifest_path, provenance_path, report_path)):
        raise RuntimeError("solver_feedback_source_integrity_failed")
    manifest, provenance, report = (read_json(manifest_path), read_json(provenance_path),
                                    read_json(report_path))
    manifest_sha = sha256_file(manifest_path)
    if (provenance.get("run_manifest_sha256") != manifest_sha or
            provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("solver_feedback_source_not_authoritative")
    result = report.get("result") or {}
    if (report.get("qa_id") != expected_qa_id or result.get("qa_id") != expected_qa_id or
            result.get("final_decision") != expected_decision):
        raise RuntimeError("solver_feedback_result_mismatch")
    rollout_rows: list[dict[str, Any]] = []
    for key in ("initial_no_context", "adaptive_extension", "full_context",
                "evidence_only", "evidence_masked"):
        stage = result.get(key)
        if isinstance(stage, dict):
            rollout_rows.extend(stage.get("rollouts") or [])
    for order in result.get("option_orders") or []:
        rollout_rows.extend((order.get("rollout") or {}).get("rollouts") or [])
    for row in rollout_rows:
        request_id = row.get("request_id")
        ledger_path = source_root / f"ledger/requests/{request_id}.json"
        if not isinstance(request_id, str) or not ledger_path.is_file():
            raise RuntimeError("solver_feedback_ledger_integrity_failed")
        ledger = read_json(ledger_path)
        raw_path = Path(str(ledger.get("raw_response_path") or ""))
        if (ledger.get("status") != "completed" or ledger.get("request_id") != request_id or
                ledger.get("backend_mode") != "local" or not raw_path.is_file() or
                sha256_file(raw_path) != ledger.get("raw_response_sha256") or
                raw_path.read_text(encoding="utf-8") != row.get("raw_response")):
            raise RuntimeError("solver_feedback_ledger_integrity_failed")
    if len(rollout_rows) != report.get("vllm_generation_count"):
        raise RuntimeError("solver_feedback_generation_count_mismatch")
    diagnosis = ("answer was guessable without dialogue" if expected_decision == "bad_ground"
                 else "answer was a direct lookup with full dialogue")
    feedback = {"qa_id":expected_qa_id, "final_decision":expected_decision,
        "no_context_rewards":list((result.get("no_context") or {}).get("verified_rewards") or []),
        "diagnosis":diagnosis}
    if result.get("full_context"):
        feedback["full_context_rewards"] = list(
            result["full_context"].get("verified_rewards") or [])
    return {"source_root":str(source_root), "source_manifest_sha256":manifest_sha,
        "source_provenance_sha256":sha256_file(provenance_path),
        "source_report_sha256":sha256_file(report_path),
        "source_materialized_qa_sha256":manifest.get("materialized_qa_sha256"),
        "rollout_count":len(rollout_rows), "feedback":feedback}


def verify_bad_ground_permutation_source(source_root: Path, qa_id: str) -> dict[str, Any]:
    """Attest an immutable bad-ground K3 and its materialized QA for permutation replay."""
    attestation = verify_solver_feedback_source(source_root,
        expected_qa_id=qa_id, expected_decision="bad_ground")
    source_root = source_root.resolve()
    manifest_path = source_root / "config/run_manifest.json"
    report_path = source_root / "reports/minimal_solver_full_audit.json"
    qa_path = source_root / "frozen/solver_qa.json"
    context_path = source_root / "frozen/questioner_smoke_context.json"
    catalog_path = source_root / "frozen/questioner_span_catalog.json"
    contracts_path = source_root / "frozen/questioner_reasoning_contracts.json"
    required = (manifest_path, report_path, qa_path, context_path, catalog_path,
                contracts_path)
    if any(not path.is_file() for path in required):
        raise RuntimeError("bad_ground_permutation_source_integrity_failed")
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    frozen_qa = read_json(qa_path)
    qa = frozen_qa.get("qa")
    result = report.get("result") or {}
    original = result.get("initial_no_context")
    if (sha256_file(qa_path) != manifest.get("frozen_solver_qa_sha256") or
            not isinstance(qa, dict) or qa.get("qa_id") != qa_id or
            sha256_bytes(canonical_json(qa).encode()) != frozen_qa.get("original_qa_sha256") or
            qa.get("admission_mode") != "spell_style_minimal_v1" or
            not isinstance(original, dict) or
            original.get("requested_valid_rollouts") != 3 or
            original.get("obtained_valid_rollouts") != 3 or
            original.get("verified_rewards") != [1, 1, 1] or
            result.get("adaptive_extension") is not None or result.get("full_context") is not None):
        raise RuntimeError("bad_ground_permutation_source_result_mismatch")
    rows = original.get("rollouts") or []
    if (len(rows) != 3 or any(row.get("evaluation_mode") != "no_context" or
            row.get("order_id") != "original" or row.get("local_reward") != 1
            for row in rows)):
        raise RuntimeError("bad_ground_permutation_source_result_mismatch")
    context = read_json(context_path)
    catalog = read_json(catalog_path)
    contracts = read_json(contracts_path)
    if (sha256_file(context_path) != manifest.get("frozen_dialogue_sha256") or
            span_catalog_sha256(catalog) != manifest.get("span_catalog_sha256") or
            reasoning_contract_sha256(contracts) != manifest.get("reasoning_contract_sha256")):
        raise RuntimeError("bad_ground_permutation_source_integrity_failed")
    return {**attestation, "source_qa_wrapper_sha256":sha256_file(qa_path),
        "source_context_sha256":sha256_file(context_path),
        "source_span_catalog_sha256":span_catalog_sha256(catalog),
        "source_reasoning_contract_sha256":reasoning_contract_sha256(contracts),
        "qa":qa, "frozen_qa":frozen_qa, "original_no_context":original,
        "context":context, "span_catalog":catalog, "reasoning_contracts":contracts,
        "source_qa_path":str(qa_path), "source_context_path":str(context_path),
        "source_catalog_path":str(catalog_path),
        "source_contracts_path":str(contracts_path)}


def verify_questioner_replay_source(source_root: Path) -> dict[str, Any]:
    """Attest a completed immutable Questioner run before offline replay."""
    source_root = source_root.resolve()
    fixed_paths = {
        "manifest":source_root / "config/run_manifest.json",
        "provenance":source_root / "reports/post_run_provenance.json",
        "report":source_root / "reports/live_questioner_smoke.json",
        "raw_envelope":source_root / "raw/questioner_batch_00.json",
        "context":source_root / "frozen/questioner_smoke_context.json",
        "span_catalog":source_root / "frozen/questioner_span_catalog.json",
        "reasoning_contracts":source_root / "frozen/questioner_reasoning_contracts.json",
    }
    if any(not path.is_file() for path in fixed_paths.values()):
        raise RuntimeError("source_run_integrity_failed: Questioner replay artifact missing")
    manifest = read_json(fixed_paths["manifest"])
    provenance = read_json(fixed_paths["provenance"])
    report = read_json(fixed_paths["report"])
    request_id = str(report.get("final_request_id") or report.get("request_id") or
                     "questioner-b00")
    ledger_path = source_root / f"ledger/requests/{request_id}.json"
    if not ledger_path.is_file():
        raise RuntimeError("source_run_integrity_failed: final Questioner ledger missing")
    ledger = read_json(ledger_path)
    raw_path = Path(str(ledger.get("raw_response_path") or ""))
    paths = {**fixed_paths, "ledger":ledger_path, "raw":raw_path}
    envelope = read_json(paths["raw_envelope"])
    context = read_json(paths["context"])
    catalog = read_json(paths["span_catalog"])
    contracts = read_json(paths["reasoning_contracts"])
    manifest_sha = sha256_file(paths["manifest"])
    if (provenance.get("run_manifest_sha256") != manifest_sha or
            provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("source_run_not_authoritative")
    raw_bytes = paths["raw"].read_bytes()
    raw_sha = sha256_bytes(raw_bytes)
    if (ledger.get("status") != "completed" or ledger.get("intent") != "questioner_batch" or
            ledger.get("raw_response_sha256") != raw_sha or
            Path(str(ledger.get("raw_response_path") or "")).resolve() != paths["raw"].resolve() or
            envelope.get("request_id") != request_id or
            envelope.get("raw_response") != raw_bytes.decode("utf-8") or
            report.get("parse_status") != "json_parsed"):
        raise RuntimeError("source_run_integrity_failed: raw response binding mismatch")
    try:
        raw_batch = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("source_run_integrity_failed: raw response is not JSON") from error
    if not isinstance(raw_batch, list) or len(raw_batch) != 3:
        raise RuntimeError("source_run_integrity_failed: expected one three-item batch")
    if (span_catalog_sha256(catalog) != manifest.get("span_catalog_sha256") or
            sha256_file(paths["span_catalog"]) != manifest.get("span_catalog_file_sha256") or
            reasoning_contract_sha256(contracts) != manifest.get("reasoning_contract_sha256") or
            sha256_file(paths["reasoning_contracts"]) != manifest.get("reasoning_contract_file_sha256") or
            sha256_file(paths["context"]) != manifest.get("frozen_dialogue_sha256")):
        raise RuntimeError("source_run_integrity_failed: frozen Questioner input drift")
    letters = manifest.get("target_gold_letters")
    if (not isinstance(letters, list) or len(letters) != 3 or
            any(letter not in "ABCD" for letter in letters)):
        raise RuntimeError("source_run_integrity_failed: target letters missing")
    attempt_attestations = []
    for attempt in report.get("attempts") or []:
        attempt_id = attempt.get("request_id")
        attempt_ledger_path = source_root / f"ledger/requests/{attempt_id}.json"
        if not isinstance(attempt_id, str) or not attempt_ledger_path.is_file():
            raise RuntimeError("source_run_integrity_failed: retry ledger missing")
        state = read_json(attempt_ledger_path)
        attempt_raw_path = Path(str(state.get("raw_response_path") or ""))
        if (state.get("status") != "completed" or not attempt_raw_path.is_file() or
                sha256_file(attempt_raw_path) != state.get("raw_response_sha256") or
                state.get("final_decision") != attempt.get("decision")):
            raise RuntimeError("source_run_integrity_failed: retry ledger drift")
        attempt_attestations.append({"request_id":attempt_id,
            "ledger_sha256":sha256_file(attempt_ledger_path),
            "raw_response_sha256":sha256_file(attempt_raw_path),
            "decision":state.get("final_decision")})
    return {"source_root":str(source_root), "source_run_id":manifest.get("run_id"),
        "source_manifest_sha256":manifest_sha,
        "source_provenance_sha256":sha256_file(paths["provenance"]),
        "source_ledger_sha256":sha256_file(paths["ledger"]),
        "source_raw_sha256":raw_sha,
        "source_raw_envelope_sha256":sha256_file(paths["raw_envelope"]),
        "source_frozen_dialogue_sha256":sha256_file(paths["context"]),
        "source_span_catalog_file_sha256":sha256_file(paths["span_catalog"]),
        "source_span_catalog_sha256":span_catalog_sha256(catalog),
        "source_reasoning_contract_file_sha256":sha256_file(paths["reasoning_contracts"]),
        "source_reasoning_contract_sha256":reasoning_contract_sha256(contracts),
        "source_prompt_sha256":ledger.get("prompt_sha256"),
        "source_questioner_schema_version":manifest.get("questioner_schema_version"),
        "source_questioner_output_profile":manifest.get("questioner_output_profile"),
        "source_final_request_id":request_id,
        "source_attempt_attestations":attempt_attestations,
        "target_gold_letters":letters, "raw_batch":raw_batch, "context":context,
        "span_catalog":catalog, "reasoning_contracts":contracts,
        "paths":{key:str(path) for key,path in paths.items()}}


def render_minimal_replay_markdown(report: dict[str, Any]) -> str:
    lines = ["# SPELL-style minimal replay", "",
        "This is an offline dual-admission replay. No model generation was performed.", "",
        f"- Raw items: {report['raw_count']}",
        f"- Strict v7 accepted: {report['strict_v7_accepted_count']}",
        f"- Minimal accepted: {report['minimal_accepted_count']}", "",
        "| QA | strict v7 | SPELL-style minimal | Solver eligible |", "|---|---|---|---|"]
    for item in report.get("items") or []:
        strict = item["strict_v7"]
        minimal = item["spell_style_minimal_v1"]
        lines.append(f"| {item['qa_id']} | {strict['decision']}: "
                     f"{', '.join(strict['reasons']) or 'none'} | {minimal['decision']}: "
                     f"{', '.join(minimal['reasons']) or 'none'} | "
                     f"{str(item['solver_eligible']).lower()} |")
    lines.extend(["", f"Replay SHA material: `{sha256_bytes(canonical_json(report).encode())}`", ""])
    return "\n".join(lines)


def verify_minimal_replay_source(source_root: Path, qa_id: str) -> dict[str, Any]:
    """Attest one minimal-accepted materialized QA before local Solver loading."""
    source_root = source_root.resolve()
    manifest_path = source_root / "config/run_manifest.json"
    provenance_path = source_root / "reports/post_run_provenance.json"
    report_path = source_root / "reports/spell_style_minimal_replay.json"
    raw_path = source_root / "frozen/source_questioner_raw.txt"
    context_path = source_root / "frozen/questioner_smoke_context.json"
    catalog_path = source_root / "frozen/questioner_span_catalog.json"
    contracts_path = source_root / "frozen/questioner_reasoning_contracts.json"
    required = (manifest_path, provenance_path, report_path, raw_path, context_path,
                catalog_path, contracts_path)
    if any(not path.is_file() for path in required):
        raise RuntimeError("source_run_integrity_failed: minimal replay artifact missing")
    manifest, provenance, report = (read_json(manifest_path), read_json(provenance_path),
                                    read_json(report_path))
    manifest_sha = sha256_file(manifest_path)
    if (manifest.get("admission_mode") != "spell_style_minimal_v1" or
            provenance.get("run_manifest_sha256") != manifest_sha or
            provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False):
        raise RuntimeError("source_run_not_authoritative")
    if sha256_file(report_path) != manifest.get("dual_admission_replay_report_sha256"):
        raise RuntimeError("source_run_integrity_failed: replay report drift")
    matches = [item for item in report.get("items") or [] if item.get("qa_id") == qa_id]
    if len(matches) != 1 or matches[0].get("spell_style_minimal_v1", {}).get("decision") != "accept" or \
            matches[0].get("solver_eligible") is not True:
        raise RuntimeError("solver_qa_not_admitted")
    item = matches[0]
    relative = item.get("materialized_qa_path")
    if not isinstance(relative, str):
        raise RuntimeError("solver_qa_not_materialized")
    materialized_path = (source_root / relative).resolve()
    if source_root not in materialized_path.parents or not materialized_path.is_file() or \
            sha256_file(materialized_path) != item.get("materialized_qa_sha256"):
        raise RuntimeError("solver_qa_integrity_mismatch")
    catalog = read_json(catalog_path)
    contracts = read_json(contracts_path)
    context = read_json(context_path)
    raw_text = raw_path.read_text(encoding="utf-8")
    try:
        raw_batch = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("source_run_integrity_failed: replay raw invalid") from error
    if (sha256_file(raw_path) != manifest.get("source_questioner_raw_sha256") or
            sha256_file(context_path) != manifest.get("frozen_dialogue_sha256") or
            span_catalog_sha256(catalog) != manifest.get("span_catalog_sha256") or
            reasoning_contract_sha256(contracts) != manifest.get("reasoning_contract_sha256")):
        raise RuntimeError("source_run_integrity_failed: replay frozen input drift")
    return {"source_root":str(source_root), "source_manifest_sha256":manifest_sha,
        "source_provenance_sha256":sha256_file(provenance_path),
        "source_replay_report_sha256":sha256_file(report_path),
        "source_raw_sha256":sha256_file(raw_path), "raw_batch":raw_batch,
        "context":context, "span_catalog":catalog, "reasoning_contracts":contracts,
        "materialized_qa":read_json(materialized_path),
        "materialized_qa_path":str(materialized_path),
        "materialized_qa_sha256":sha256_file(materialized_path), "admission_item":item}
