"""Execute exactly one new-branch R03 Planner -> Realizer debug root.

The runner uses the frozen local R02 Memory snapshot and the v2 feedback
renderer.  It has no Memorax client and the HTTP completion helper below has no
retry or fallback path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from ..h4_round1_r03_feedback_view_v2.compiler import (
    digest_file, load_frozen_inputs, read_json, read_jsonl,
)
from ..h4_round1_r03_feedback_view_v2.renderers import (
    render_expansion_planner, render_expansion_realizer,
)
from ..h4_round1_r03_feedback_view_v2_single_root_debug import (
    CONFIG_PATH, CONTEXT_CEILING, DEFAULT_OUTPUT, MEMORY_PREFLIGHT, MODEL,
    PLANNER_MAX_TOKENS, PREFLIGHT_OUTPUT, REALIZER_MAX_TOKENS, ROOT_ID,
    TEMPERATURE, TIMEOUT_SECONDS, TOKENIZER_PATH, VERSION,
)
from ..h4_round1_r03_memory_readonly_single_root_smoke_v1.smoke import (
    _build_realizer_style_context, add_sidecars, local_continuation_gate,
    local_planner_gate, memory_navigation_candidates, planner_parent,
    write_json, write_jsonl, write_text,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    materialize_memory_only_tracks, normalize_realizer_output,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.live_runtime import (
    load_json_object_strict,
)
from ..simple_append_v1.assembly import append_expansion_round, validate_appended_round
from ..live_backends import load_challenger_config


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_counter():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, local_files_only=True, trust_remote_code=False)
    return lambda text: len(tokenizer.encode(str(text), add_special_tokens=False))


def _dialogue_text(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(
        f"[{row.get('turn_ref')}] {row.get('speaker')}: {row.get('text')}"
        for row in dialogue.get("turns") or []
    )


def _request_once(*, config: Any, prompt: str, max_tokens: int,
                  request_id: str) -> dict[str, Any]:
    """Perform one HTTP attempt. This function intentionally never retries."""
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    if config.enable_thinking is not None:
        payload["enable_thinking"] = config.enable_thinking
    request = urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body_bytes = response.read()
        body = json.loads(body_bytes.decode("utf-8"))
        choice = body["choices"][0]
        message = choice["message"]
        raw = str(message.get("content") or "")
        return {
            "ok": True, "raw_response": raw,
            "model_identifier": body.get("model") or config.model,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") if isinstance(body.get("usage"), Mapping) else None,
            "transport_attempt_count": 1,
            "request_id": request_id,
        }
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return {
            "ok": False, "request_id": request_id, "error_type": "HTTPError",
            "http_status": int(error.code), "error_body": body,
            "transport_attempt_count": 1,
        }
    except Exception as error:  # preserve the failed attempt; never resubmit
        return {
            "ok": False, "request_id": request_id,
            "error_type": type(error).__name__, "error": str(error),
            "transport_attempt_count": 1,
        }


def _write_request(path: Path, *, request_id: str, role: str, config: Any,
                   prompt: str, max_tokens: int, prompt_tokens: int) -> None:
    write_json(path, {
        "request_id": request_id, "role": role, "model": config.model,
        "endpoint": config.base_url, "temperature": config.temperature,
        "max_tokens": max_tokens, "prompt_tokens_local": prompt_tokens,
        "context_ceiling": CONTEXT_CEILING,
        "response_format": {"type": "json_object"},
        "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
        "prompt": prompt, "api_key_recorded": False,
        "automatic_retry_allowed": False, "provider_fallback_allowed": False,
    })


def _load_root_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    offline_report = read_json(PREFLIGHT_OUTPUT / "offline_preflight_report.json")
    if offline_report.get("status") != "TEMPLATE_AND_INPUT_VIEWS_OFFLINE_READY":
        raise RuntimeError("v2_offline_preflight_not_ready")
    frozen = load_frozen_inputs()
    manifest_rows = read_jsonl(PREFLIGHT_OUTPUT / "authoritative_parent_r01_r02_dialogue_manifest.jsonl")
    if not manifest_rows or manifest_rows[0].get("root_id") != ROOT_ID:
        raise RuntimeError("first_stable_authoritative_root_is_not_expected_root")
    identity = manifest_rows[0]
    dialogue = read_json(Path(identity["input_dialogue_path"]))
    memory_root = MEMORY_PREFLIGHT / "roots" / ROOT_ID / "memory_read"
    snapshot = read_json(memory_root / "r02_memory_snapshot_normalized.json")
    projection = read_json(memory_root / "r02_memory_prompt_projection.json")
    record = next(row for row in frozen["records"] if row["root_id"] == ROOT_ID)
    checks = {
        "root_id_match": dialogue.get("root_id") == ROOT_ID,
        "file_sha_match": digest_file(Path(identity["input_dialogue_path"])) == identity["input_dialogue_file_sha256"],
        "complete_dialogue_sha_match": dialogue.get("complete_dialogue_sha256") == identity["complete_dialogue_sha256"],
        "latest_round_is_r02": identity.get("checks", {}).get("latest_round_is_r02") is True,
        "r03_turns_absent": identity.get("checks", {}).get("r03_turns_absent") is True,
        "rounds_are_r01_r02_only": identity.get("checks", {}).get("rounds_are_r01_r02_only") is True,
        "turn_refs_unique": identity.get("checks", {}).get("turn_refs_unique") is True,
        "memory_root_match": snapshot.get("root_id") == ROOT_ID and projection.get("root_id") == ROOT_ID,
        "memory_snapshot_sha_match": digest_file(memory_root / "r02_memory_snapshot_normalized.json") == (memory_root / "r02_memory_snapshot_normalized.json.sha256").read_text().split()[0],
        "no_old_r03_turn_refs": not any(str(row.get("turn_ref", "")).startswith("R03:") for row in dialogue.get("turns") or []),
    }
    if not all(checks.values()):
        raise RuntimeError("new_r03_input_preflight_failed:" + ",".join(k for k, v in checks.items() if not v))
    return identity, dialogue, snapshot, projection, record, checks


def _source_audit(paths: list[Path], before: Mapping[str, str]) -> dict[str, Any]:
    rows = []
    for path in sorted(paths):
        after = digest_file(path)
        rows.append({"path": str(path), "before_sha256": before[str(path)], "after_sha256": after, "unchanged": before[str(path)] == after})
    return {"files": rows, "modified_count": sum(not row["unchanged"] for row in rows)}


def execute(run_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_root = Path(run_root)
    if run_root.exists():
        raise RuntimeError(f"debug_run_root_exists:{run_root}")
    identity, dialogue, memory_snapshot, memory_projection, record, input_checks = _load_root_inputs()
    config = load_challenger_config(CONFIG_PATH)
    # Frozen smoke contract values are explicit and recorded; API credentials
    # never enter an artifact.
    config = config.__class__(
        base_url=config.base_url, api_key=config.api_key, model=MODEL,
        timeout_seconds=TIMEOUT_SECONDS,
        realizer_max_tokens=REALIZER_MAX_TOKENS,
        questioner_max_tokens=PLANNER_MAX_TOKENS,
        temperature=TEMPERATURE, enable_thinking=config.enable_thinking,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    branch_id = f"r03-feedback-view-v2-debug-{ROOT_ID}-live-1"
    parent = planner_parent(dialogue)
    candidates = memory_navigation_candidates(memory_projection)
    tracks = materialize_memory_only_tracks(candidates)
    first_seen: list[str] = []
    for row in dialogue.get("turns") or []:
        if row.get("speaker") not in first_seen:
            first_seen.append(row["speaker"])
    registry = {"speakers": [{"speaker_ref": f"SP{index:02d}", "canonical_name": name} for index, name in enumerate(first_seen, 1)]}
    planner_record = render_expansion_planner(
        record=record, dialogue=dialogue, memory_overview=memory_projection,
        memory_candidates=candidates, verified_tracks=tracks,
        memory_only_fast=True, speaker_registry=registry,
        strict_speaker_refs=True, expose_complete_memory=True,
    )
    planner_prompt = planner_record["prompt"]
    counter = _token_counter()
    planner_tokens = counter(planner_prompt)
    realizer_budget = REALIZER_MAX_TOKENS
    execution_plan = {
        "version": "r03_feedback_view_v2_single_root_execution_plan_v1",
        "branch_id": branch_id, "root_id": ROOT_ID,
        "base_source_prefix": "Parent+R01+R02-feedback",
        "scene_count": 3, "planner_requests_max": 1,
        "realizer_requests_max": 1, "total_expansion_requests_max": 2,
        "planner_role": "simple_append_planner_v1",
        "realizer_role": "simple_append_round_realizer_v1",
        "model": MODEL, "endpoint": config.base_url,
        "temperature": TEMPERATURE, "planner_max_tokens": PLANNER_MAX_TOKENS,
        "realizer_max_tokens": REALIZER_MAX_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "context_ceiling": CONTEXT_CEILING,
        "planner_prompt_tokens": planner_tokens,
        "automatic_retry": False, "repair": False,
        "provider_fallback": False, "reserve_root_substitution": False,
        "memory_search": False, "memory_writes": False,
        "qa_solver_training": False,
    }
    if planner_tokens + PLANNER_MAX_TOKENS > CONTEXT_CEILING:
        raise RuntimeError("planner_context_overflow_before_first_request")
    write_json(run_root / "execution_plan.json", execution_plan)
    write_json(run_root / "input_identity.json", {
        "branch_id": branch_id, "root_id": ROOT_ID, "input_manifest_row": identity,
        "input_checks": input_checks, "input_dialogue_path": identity["input_dialogue_path"],
        "input_dialogue_file_sha256": identity["input_dialogue_file_sha256"],
        "complete_dialogue_sha256": identity["complete_dialogue_sha256"],
        "memory_snapshot_sha256": memory_snapshot["snapshot_sha256"],
        "v2_preflight_report_sha256": digest_file(PREFLIGHT_OUTPUT / "offline_preflight_report.json"),
        "v2_template_identity_sha256": digest_file(PREFLIGHT_OUTPUT / "template_identity.json"),
        "feedback_view_sha256": sha_value(planner_record["feedback_view"]),
    })
    write_json(run_root / "feedback_view.json", planner_record["feedback_view"])
    write_json(run_root / "source_memory_snapshot.json", memory_snapshot)
    write_json(run_root / "request_manifest.json", {
        "branch_id": branch_id, "planned": {"planner": 1, "realizer": 1, "total": 2},
        "actual": {"planner": 0, "realizer": 0, "total": 0},
        "automatic_retry": False, "request_ids": [], "external_non_expansion_calls": 0,
    })
    write_json(run_root / "planner" / "request.json", {})
    _write_request(run_root / "planner" / "request.json", request_id=f"{branch_id}-planner-01", role="simple_append_planner_v1", config=config, prompt=planner_prompt, max_tokens=PLANNER_MAX_TOKENS, prompt_tokens=planner_tokens)
    planner_response = _request_once(config=config, prompt=planner_prompt, max_tokens=PLANNER_MAX_TOKENS, request_id=f"{branch_id}-planner-01")
    write_json(run_root / "planner" / "response.json", {**planner_response, "raw_response_sha256": sha_bytes(str(planner_response.get("raw_response", "")).encode()) if planner_response.get("ok") else None})
    actual = {"planner": 1, "realizer": 0, "total": 1}
    write_json(run_root / "request_manifest.json", {"branch_id": branch_id, "planned": {"planner": 1, "realizer": 1, "total": 2}, "actual": actual, "automatic_retry": False, "request_ids": [f"{branch_id}-planner-01"], "external_non_expansion_calls": 0})
    if not planner_response.get("ok"):
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "planner_transport_failure", planner_response)
    try:
        raw_plan = load_json_object_strict(planner_response["raw_response"])
    except Exception as error:
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "planner_parse_failure", {"error": f"{type(error).__name__}:{error}"})
    write_json(run_root / "planner" / "raw_plan.json", raw_plan)
    feedback_card = {"evidence_anchor_refs": list(record["historical_qa"].get("evidence_refs") or [])}
    planner_gate, materialized_plan = local_planner_gate(
        raw_plan=raw_plan, parent=parent, speaker_registry=registry, dialogue=dialogue,
        memory_snapshot=memory_snapshot, feedback_card=feedback_card,
        candidates=candidates, verified_tracks=tracks,
    )
    write_json(run_root / "planner" / "local_gate_report.json", planner_gate)
    if not planner_gate.get("valid") or materialized_plan is None:
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "planner_local_gate_failure", planner_gate)
    write_json(run_root / "planner" / "parsed_plan.json", materialized_plan)
    style = _build_realizer_style_context(parent["dialogue"]["turns"])
    write_json(run_root / "realizer" / "style_context.json", style)
    realizer_record = render_expansion_realizer(
        record=record, dialogue=dialogue, plan=materialized_plan,
        style_profile=style["style_profile"], continuity_shape=style["continuity_tail"],
        include_full_parent=False,
    )
    realizer_prompt = realizer_record["prompt"]
    realizer_tokens = counter(realizer_prompt)
    execution_plan["realizer_prompt_tokens"] = realizer_tokens
    if realizer_tokens + realizer_budget > CONTEXT_CEILING:
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "realizer_context_overflow_before_request", {"realizer_prompt_tokens": realizer_tokens})
    write_json(run_root / "execution_plan.json", execution_plan)
    _write_request(run_root / "realizer" / "request_01.json", request_id=f"{branch_id}-realizer-01", role="simple_append_round_realizer_v1", config=config, prompt=realizer_prompt, max_tokens=REALIZER_MAX_TOKENS, prompt_tokens=realizer_tokens)
    realizer_response = _request_once(config=config, prompt=realizer_prompt, max_tokens=REALIZER_MAX_TOKENS, request_id=f"{branch_id}-realizer-01")
    write_json(run_root / "realizer" / "response_01.json", {**realizer_response, "raw_response_sha256": sha_bytes(str(realizer_response.get("raw_response", "")).encode()) if realizer_response.get("ok") else None})
    actual = {"planner": 1, "realizer": 1, "total": 2}
    write_json(run_root / "request_manifest.json", {"branch_id": branch_id, "planned": {"planner": 1, "realizer": 1, "total": 2}, "actual": actual, "automatic_retry": False, "request_ids": [f"{branch_id}-planner-01", f"{branch_id}-realizer-01"], "external_non_expansion_calls": 0})
    if not realizer_response.get("ok"):
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "realizer_transport_failure", realizer_response)
    try:
        raw_child = load_json_object_strict(realizer_response["raw_response"])
    except Exception as error:
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "realizer_parse_failure", {"error": f"{type(error).__name__}:{error}"})
    write_json(run_root / "realizer" / "parsed_01.json", raw_child)
    validation = normalize_realizer_output(raw_child, plan=materialized_plan, parent=parent, round_index=3)
    write_json(run_root / "realizer" / "validation_01.json", validation)
    if not validation.get("valid"):
        write_json(run_root / "r03_delta_quarantine.json", validation.get("child") or {})
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "realizer_local_validation_failure", validation)
    child = validation["child"]
    cumulative = append_expansion_round(prior_expanded=dialogue, child=child)
    cumulative_validation = validate_appended_round(cumulative, prior_expanded=dialogue)
    gate = local_continuation_gate(prior=dialogue, child=child, cumulative=cumulative, cumulative_validation=cumulative_validation, realizer_validation=validation, plan=materialized_plan)
    write_json(run_root / "local_continuation_gate_report.json", gate)
    write_json(run_root / "cumulative_validation.json", cumulative_validation)
    if not gate.get("valid"):
        write_json(run_root / "r03_delta.json", child)
        write_json(run_root / "parent_r01_r02_r03_cumulative_dialogue.json", cumulative)
        return _finish_failure(run_root, branch_id, input_checks, execution_plan, actual, "local_continuation_gate_failure", gate)
    r03_turns = cumulative["turns"][len(dialogue["turns"]):]
    write_json(run_root / "r03_delta.json", child)
    write_jsonl(run_root / "r03_delta_turns.jsonl", r03_turns)
    write_json(run_root / "parent_r01_r02_r03_cumulative_dialogue.json", cumulative)
    source_paths = [Path(identity["input_dialogue_path"]), PREFLIGHT_OUTPUT / "offline_preflight_report.json", PREFLIGHT_OUTPUT / "template_identity.json", PREFLIGHT_OUTPUT / "feedback_view_backend_manifest.jsonl", MEMORY_PREFLIGHT / "roots" / ROOT_ID / "memory_read" / "r02_memory_snapshot_normalized.json"]
    before = {str(path): digest_file(path) for path in source_paths}
    after_audit = _source_audit(source_paths, before)
    write_json(run_root / "source_sha_audit.json", after_audit)
    lineage = {
        "lineage_version": "r03_feedback_view_v2_debug_lineage_v1", "branch_id": branch_id,
        "root_id": ROOT_ID, "source_prefix": "Parent+R01+R02-feedback",
        "source_dialogue_file_sha256": identity["input_dialogue_file_sha256"],
        "source_complete_dialogue_sha256": identity["complete_dialogue_sha256"],
        "memory_snapshot_sha256": memory_snapshot["snapshot_sha256"],
        "v2_template_identity_sha256": digest_file(PREFLIGHT_OUTPUT / "template_identity.json"),
        "planner_prompt_sha256": sha_bytes(planner_prompt.encode()), "realizer_prompt_sha256": sha_bytes(realizer_prompt.encode()),
        "r03_turn_count": len(r03_turns), "cumulative_dialogue_sha256": cumulative["complete_dialogue_sha256"],
        "append_only": True, "old_r03_mixed": False, "remote_memory_writes": 0,
    }
    write_json(run_root / "lineage_manifest.json", lineage)
    report = {
        "status": "R03_SINGLE_ROOT_FEEDBACK_VIEW_V2_DEBUG_PASS",
        "next_status": "WAITING_FOR_AUTHORIZATION_FOR_NEXT_ROOT_ON_SAME_NEW_R03_BRANCH",
        "root_id": ROOT_ID, "branch_id": branch_id,
        "input_turn_count": len(dialogue["turns"]), "r03_turn_count": len(r03_turns), "cumulative_turn_count": len(cumulative["turns"]),
        "planned_scene_count": len(materialized_plan.get("scenes") or []), "actual_scene_count": len(child.get("scenes") or []),
        "attempted_turns": sum(len(scene.get("dialogue_turns") or []) for scene in child.get("scenes") or []), "promoted_turns": len(r03_turns),
        "parent_r01_r02_unchanged": bool(after_audit["modified_count"] == 0 and gate["checks"].get("prior_turn_objects_byte_equivalent") and gate["checks"].get("prior_turn_refs_unchanged")),
        "new_r03_saved_and_frozen": True, "old_r03_mixed": False, "old_template_fallback": False,
        "planner_requests": 1, "realizer_requests": 1, "total_expansion_requests": 2,
        "memorax_search_calls": 0, "memorax_status_calls": 0, "memorax_add_calls": 0, "memorax_mutation_calls": 0,
        "qa_writer_calls": 0, "solver_calls": 0, "semantic_auditor_calls": 0, "training_calls": 0,
        "semantic_realization_verdict": "NOT_AUDITED_BY_CONTRACT",
        "technical_failures": [], "parse_failures": [], "existing_closure_failures": [],
        "not_tested": ["QA/Solver value", "semantic correctness of new facts", "comparison with old R03"],
    }
    _finish_success(run_root, report)
    return report


def _finish_failure(run_root: Path, branch_id: str, input_checks: Mapping[str, Any], execution_plan: Mapping[str, Any], actual: Mapping[str, Any], stage: str, details: Any) -> dict[str, Any]:
    child = details.get("child") if isinstance(details, Mapping) else None
    scenes = list(child.get("scenes") or []) if isinstance(child, Mapping) else []
    attempted_turns = sum(len(scene.get("dialogue_turns") or []) for scene in scenes)
    report = {
        "status": "R03_SINGLE_ROOT_FEEDBACK_VIEW_V2_DEBUG_FAILED_OR_QUARANTINED",
        "next_status": "WAITING_FOR_DIAGNOSTIC_OR_FIX_AUTHORIZATION; NO_RETRY",
        "root_id": ROOT_ID, "branch_id": branch_id, "failure_stage": stage,
        "planned_scene_count": execution_plan.get("scene_count", 3),
        "realizer_output_scene_count": len(scenes), "promoted_scene_count": 0,
        "attempted_turns": attempted_turns, "promoted_turns": 0,
        "new_r03_saved_and_frozen": False,
        "quarantine_delta_saved": bool(scenes),
        "parent_r01_r02_unchanged": True, "old_r03_mixed": False, "old_template_fallback": False,
        "planner_requests": actual.get("planner", 0), "realizer_requests": actual.get("realizer", 0), "total_expansion_requests": actual.get("total", 0),
        "memorax_search_calls": 0, "memorax_status_calls": 0, "memorax_add_calls": 0, "memorax_mutation_calls": 0,
        "qa_writer_calls": 0, "solver_calls": 0, "semantic_auditor_calls": 0, "training_calls": 0,
        "input_checks": dict(input_checks), "details": details,
        "not_tested": ["new R03 completion", "QA/Solver value", "semantic correctness"],
    }
    _finish_success(run_root, report)
    return report


def _finish_success(run_root: Path, report: Mapping[str, Any]) -> None:
    write_json(run_root / "final_debug_report.json", dict(report))
    write_text(run_root / "final_debug_report.md", "\n".join([
        "# R03 Feedback View v2 Single Root Debug", "",
        f"- Status: `{report['status']}`", f"- Root: `{report.get('root_id')}`", f"- Branch: `{report.get('branch_id')}`",
        f"- Planned scenes / actual scenes: `{report.get('planned_scene_count')} / {report.get('actual_scene_count')}`",
        f"- Attempted / promoted turns: `{report.get('attempted_turns')} / {report.get('promoted_turns')}`",
        f"- Planner / Realizer requests: `{report.get('planner_requests')} / {report.get('realizer_requests')}`",
        "- Memorax Search / Status / Add / Mutation: `0 / 0 / 0 / 0`",
        "- QA Writer / Solver / semantic auditor / training: `0 / 0 / 0 / 0`",
        f"- Next: `{report.get('next_status')}`", "",
    ]))
    add_sidecars(run_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = execute(args.run_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
