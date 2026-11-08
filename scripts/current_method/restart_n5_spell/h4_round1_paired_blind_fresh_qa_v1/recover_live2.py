"""Recover the single ambiguous live_2 Feedback repair in an isolated run."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import _accepted_candidates, _prefix_candidate
from ..h4_qa_multiroot_k3_t08_probe_v1.permutation import audit_gold_rebinding, position_variants
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_batch
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SOURCE_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2"
OUTPUT_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_3"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
ROOT_ID = "cc-train-episode-254852"
OLD_REQUEST_ID = f"{ROOT_ID}-contract_guided-repair-I001"
NEW_REQUEST_ID = f"{OLD_REQUEST_ID}-attempt-02"
TEMPERATURE = 0.1
MAX_TOKENS = 8192
GENERATOR_VERSION = "h4_qa_multiroot_k3_t08_fresh_generation_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def scope_candidate(row: Mapping[str, Any], branch: str) -> dict[str, Any]:
    item = dict(row)
    original = str(item.get("candidate_id") or "")
    source = str(item.get("generation_source") or "")
    qtype = str(item.get("question_type") or "")
    ordinal = int(item.get("source_row_index") or 0)
    item["source_candidate_id"] = original
    item["candidate_id"] = f"{ROOT_ID}:R01-R02-{branch}:blind:{source}:{qtype}:Q{ordinal:02d}"
    item.update({"root_id": ROOT_ID, "branch": branch, "round": "R01-R02", "qa_mode": "blind"})
    return item


def preflight() -> dict[str, Any]:
    old_state_path = SOURCE_RUN / "feedback" / "ledger" / "requests" / f"{OLD_REQUEST_ID}.json"
    prompt_path = SOURCE_RUN / "feedback" / "raw" / "prompts" / f"{OLD_REQUEST_ID}.txt"
    state = load_json(old_state_path)
    prompt = prompt_path.read_text(encoding="utf-8")
    feedback_states = [load_json(path) for path in (SOURCE_RUN / "feedback" / "ledger" / "requests").glob("*.json")]
    control_states = [load_json(path) for path in (SOURCE_RUN / "control" / "ledger" / "requests").glob("*.json")]
    checks = {
        "source_final_failed": load_json(SOURCE_RUN / "final_paired_qa_smoke_report.json").get("status") == "ROUND1_PAIRED_BLIND_QA_SMOKE_FAIL",
        "exactly_one_feedback_ambiguous": sum(row.get("status") != "completed" for row in feedback_states) == 1,
        "target_is_reserved": state.get("status") == "reserved",
        "target_attempt_is_one": state.get("attempt") == 1,
        "prompt_hash_matches": sha256_bytes(prompt.encode()) == state.get("prompt_sha256"),
        "target_temperature_is_point_one": (state.get("backend_identity") or {}).get("temperature") == TEMPERATURE,
        "control_all_completed": len(control_states) == 8 and all(row.get("status") == "completed" for row in control_states),
        "feedback_other_requests_completed": sum(row.get("status") == "completed" for row in feedback_states) == 7,
        "source_model_matches": (state.get("backend_identity") or {}).get("model") == "Qwen/Qwen3-14B",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_run": str(SOURCE_RUN), "output_run": str(OUTPUT_RUN),
        "source_final_report_sha256": sha256_file(SOURCE_RUN / "final_paired_qa_smoke_report.json"),
        "old_request_state_path": str(old_state_path), "old_request_state_sha256": sha256_file(old_state_path),
        "frozen_repair_prompt_path": str(prompt_path), "frozen_repair_prompt_sha256": sha256_file(prompt_path),
        "old_request_id": OLD_REQUEST_ID, "new_request_id": NEW_REQUEST_ID,
        "attempt": 2, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
    }


def run() -> dict[str, Any]:
    if OUTPUT_RUN.exists():
        final_path = OUTPUT_RUN / "final_paired_qa_smoke_report.json"
        state_path = OUTPUT_RUN / "feedback" / "ledger" / "requests" / f"{NEW_REQUEST_ID}.json"
        if not final_path.is_file() or not state_path.is_file():
            raise RuntimeError(f"output_run_incomplete_or_unrecognized:{OUTPUT_RUN}")
        # Reuse the immutable completed attempt-2 response and rerun only the
        # deterministic aggregation below.
        reuse_existing = True
    else:
        reuse_existing = False
    report = preflight()
    if not reuse_existing:
        OUTPUT_RUN.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT_RUN / "preflight_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("recovery_preflight_failed")
    prompt_path = Path(report["frozen_repair_prompt_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    atomic_text(OUTPUT_RUN / "feedback" / "raw_prompts" / f"{NEW_REQUEST_ID}.txt", prompt)
    contract_payload = {
        "recovery_version": "h4_round1_paired_blind_fresh_qa_live2_single_retry_v1",
        "source_run": str(SOURCE_RUN), "source_run_read_only": True,
        "root_id": ROOT_ID, "branch": "feedback", "qa_mode": "blind",
        "generator_version": GENERATOR_VERSION, "model": "Qwen/Qwen3-14B",
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "old_request_id": OLD_REQUEST_ID, "new_request_id": NEW_REQUEST_ID,
        "old_attempt_status": "reserved", "new_attempt": 2,
        "completed_source_requests_replayed": 0,
        "solver_calls": 0, "memorax_add_calls": 0, "semantic_auditor_calls": 0,
    }
    manifest_sha = sha256_bytes(canonical_json(contract_payload).encode())
    write_json(OUTPUT_RUN / "recovery_contract.json", {**contract_payload, "manifest_sha256": manifest_sha})
    config = load_challenger_config(API_CONFIG)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=TEMPERATURE, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))
    ledger = RequestLedger(OUTPUT_RUN / "feedback", manifest_sha, "live")
    response = ledger.invoke(
        request_id=NEW_REQUEST_ID, intent="h4_k3_candidate_schema_repair",
        attempt=2, prompt_sha256=sha256_bytes(prompt.encode()),
        call=lambda: client.complete(prompt=prompt, max_tokens=MAX_TOKENS, json_object=True),
        budget_class="fresh_qa_generation_recovery", budget_limit=1, total_budget_limit=1,
        request_metadata={"probe_version": GENERATOR_VERSION, "recovery_of": OLD_REQUEST_ID,
                          "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        backend_identity={"transport": "openai_compatible_challenger", "model": client.config.model,
                          "base_url": client.config.base_url, "temperature": TEMPERATURE},
    )
    raw = response.raw_text
    atomic_text(OUTPUT_RUN / "feedback" / "raw_responses" / f"{NEW_REQUEST_ID}.json.txt", raw)
    parsed = load_json_object_strict(raw)
    dialogue = load_json(SOURCE_RUN / "feedback" / "dialogue_source.json")
    contracts = load_json(SOURCE_RUN / "feedback" / "reasoning_contracts.json")
    contract = next(row for row in contracts if row.get("intent_id") == "I001")
    normalized = normalize_batch(parsed, intent_id="I001", source="contract_guided",
                                 contract=contract, expanded=dialogue)
    accepted, acceptance = _accepted_candidates(normalized, "contract_guided")
    normalized["minimum_acceptance"] = acceptance
    write_json(OUTPUT_RUN / "feedback" / "attempt_02_normalized.json", normalized)

    source_candidates = load_json(SOURCE_RUN / "feedback" / "candidate_pool.json")
    unscoped_source = []
    for row in source_candidates:
        item = dict(row)
        source_id = str(item.get("source_candidate_id") or item.get("candidate_id") or "")
        if source_id.startswith(ROOT_ID + ":"):
            source_id = source_id[len(ROOT_ID) + 1:]
        item["candidate_id"] = f"{ROOT_ID}:{source_id}"
        item.pop("source_candidate_id", None)
        item.pop("branch", None); item.pop("round", None); item.pop("qa_mode", None)
        unscoped_source.append(item)
    recovered = [_prefix_candidate(row, root_id=ROOT_ID, generation_source="contract_guided") for row in accepted]
    combined = unscoped_source + recovered if acceptance.get("accepted") else unscoped_source
    valid_refs = {str(row.get("turn_ref")) for row in dialogue.get("turns") or []}
    validity = validate_batch(combined, valid_turn_refs=valid_refs)
    scoped_all = [scope_candidate(row, "feedback") for row in combined]
    scoped_valid = [scope_candidate(candidate, "feedback") for candidate, audit in zip(combined, validity["rows"])
                    if audit.get("decision") == "pass"]
    write_json(OUTPUT_RUN / "feedback" / "candidate_pool.json", scoped_all)
    write_json(OUTPUT_RUN / "feedback" / "candidates_valid.json", scoped_valid)
    write_json(OUTPUT_RUN / "feedback" / "validity_audit.json", validity)
    permutation_rows = []
    permutation_errors = []
    for candidate in combined:
        audit = next((row for row in validity["rows"] if row.get("candidate_id") == candidate.get("candidate_id")), {})
        if audit.get("decision") != "pass":
            continue
        try:
            variants = position_variants(candidate)
            rebound = audit_gold_rebinding(candidate, variants)
            permutation_rows.append({"candidate_id": scope_candidate(candidate, "feedback")["candidate_id"],
                                     "question_type": candidate.get("question_type"),
                                     "variants": variants, "gold_rebinding_audit": rebound})
            if not rebound.get("pass"):
                permutation_errors.extend(rebound.get("errors") or [])
        except Exception as exc:
            permutation_errors.append(f"{candidate.get('candidate_id')}:{type(exc).__name__}:{exc}")
    write_json(OUTPUT_RUN / "feedback" / "permutations.json", {
        "candidate_count": len(permutation_rows), "gold_rebinding_errors": len(permutation_errors),
        "errors": permutation_errors, "rows": permutation_rows,
    })
    coverage = {(str(row.get("generation_source")), str(row.get("question_type"))) for row in scoped_valid}
    expected = {(source, qtype) for source in ("contract_guided", "free_exploration")
                for qtype in ("single_choice", "multi_select", "ordering")}
    state = ledger.read(NEW_REQUEST_ID) or {}
    # The formal generator admits structurally valid candidates across all
    # intent/source batches; one under-quota batch is observable but does not
    # invalidate the root when aggregate source x question-type coverage is
    # complete.  This mirrors generate_root_qa's root-level ready rule.
    final_pass = (
        state.get("status") == "completed"
        and (state.get("response_metadata") or {}).get("finish_reason") != "length"
        and validity.get("valid") is True
        and not permutation_errors
        and expected <= coverage
    )
    final = {
        "status": "ROUND1_PAIRED_BLIND_QA_SMOKE_PASS" if final_pass else "ROUND1_PAIRED_BLIND_QA_SMOKE_FAIL",
        "root_id": ROOT_ID, "source_run": str(SOURCE_RUN),
        "control_reused_by_reference": True, "control_new_requests": 0,
        "feedback_completed_source_requests_reused_by_reference": 7,
        "feedback_recovery_requests": 1, "feedback_recovery_request_status": state.get("status"),
        "feedback_recovery_finish_reason": (state.get("response_metadata") or {}).get("finish_reason"),
        "temperature": (state.get("backend_identity") or {}).get("temperature"),
        "repair_minimum_acceptance": acceptance,
        "repair_batch_nonblocking_gap": not bool(acceptance.get("accepted")),
        "feedback_candidate_count": len(scoped_all), "feedback_program_pass_count": len(scoped_valid),
        "feedback_program_reject_count": len(scoped_all) - len(scoped_valid),
        "gold_rebinding_errors": len(permutation_errors),
        "coverage": sorted([list(x) for x in coverage]), "required_coverage_met": expected <= coverage,
        "solver_calls": 0, "memorax_add_calls": 0, "semantic_auditor_calls": 0,
        "dev_test_accessed": False,
        "ready_for_8_root_paired_blind_qa_generation": final_pass,
    }
    write_json(OUTPUT_RUN / "final_paired_qa_smoke_report.json", final)
    atomic_text(OUTPUT_RUN / "final_paired_qa_smoke_report.md",
                "# Round-1 Paired Blind Fresh QA Recovery\n\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        raise RuntimeError("recovery_requires_explicit_live")
    print(json.dumps(run(), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
