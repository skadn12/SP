"""Regression tests for the R03 read-only Memory preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx

from . import CONTEXT_CEILING, QUARANTINE_ROOTS, TRIAL_ROOTS
from .memory_readonly_dialogue_adapter import (
    MemoraxEndpointNotAllowedError,
    MemoraxWriteDisabledError,
    ReadOnlyMemoraxClient,
    ReadOnlyMemoraxConfig,
    assert_allowed_endpoint,
    build_local_r03_turn_refs,
    memory_prompt_projection,
    normalize_search_response,
    render_planner_dry_run,
)


OUT = Path(os.environ.get(
    "R03_READONLY_PREFLIGHT_OUT",
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_r03_memory_readonly_trial_preflight_v2_live_2",
))


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _jsonl(name: str):
    return [json.loads(line) for line in (OUT / name).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def test_01_roots_match_live13():
    audit = _json("r03_trial_root_scope_audit.json")
    assert audit["trial_set_matches_live13"]
    assert tuple(audit["trial_root_ids"]) == TRIAL_ROOTS


def test_02_quarantine_disjoint():
    assert set(TRIAL_ROOTS).isdisjoint(QUARANTINE_ROOTS)


def test_03_root_96462_retained():
    assert "cc-train-episode-96462" in TRIAL_ROOTS


def test_04_primary_reserve_frozen():
    rows = _jsonl("r03_feedback_artifact_identity_manifest.jsonl")
    assert len(rows) == 8
    assert all(row["reserve_count"] == 2 for row in rows)
    assert all(row["primary_reserve_disjoint"] for row in rows)


def test_05_dialogue_sha_no_drift():
    assert _json("r03_input_dialogue_identity_audit.json")["pass_count"] == 8


def test_06_control_namespace_rejected():
    try:
        assert_allowed_endpoint("POST", "/v1/control/create")
    except MemoraxEndpointNotAllowedError:
        return
    raise AssertionError("control_endpoint_was_not_rejected")


def test_07_root_scoped_user_ids():
    rows = _jsonl("r03_source_memory_identity_manifest.jsonl")
    assert all(row["source_memory_user_id"].endswith(row["root_id"])
               for row in rows)
    assert len({row["source_memory_user_id"] for row in rows}) == 8


def test_08_search_endpoint_callable_with_mock():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/memory-buckets/search"
        return httpx.Response(200, json={"success": True, "data": {
            "buckets": {name: {} for name in
                        ("person", "location", "event", "item", "dreaming")}}})
    client = ReadOnlyMemoraxClient(
        ReadOnlyMemoraxConfig("https://memorax.invalid", "test-secret"),
        transport=httpx.MockTransport(handler))
    value = client.search_buckets({"user_id": "root"}, request_id="test-search")
    assert value["success"] and len(client.calls) == 1


def _client() -> ReadOnlyMemoraxClient:
    return ReadOnlyMemoraxClient(
        ReadOnlyMemoraxConfig("https://memorax.invalid", "test-secret"),
        transport=httpx.MockTransport(lambda _: httpx.Response(500)))


def test_09_add_fails_closed():
    try:
        _client().add({}, request_id="x")
    except MemoraxWriteDisabledError:
        return
    raise AssertionError("add_not_blocked")


def test_10_add_status_fails_closed():
    try:
        _client().add_status("example-task", request_id="x")
    except MemoraxWriteDisabledError:
        return
    raise AssertionError("add_status_not_blocked")


def test_11_other_endpoints_fail_closed():
    for method, path in (("DELETE", "/x"), ("GET", "/x"), ("POST", "/x")):
        try:
            assert_allowed_endpoint(method, path)
        except MemoraxEndpointNotAllowedError:
            continue
        raise AssertionError((method, path))


def test_12_no_child_add_artifacts():
    audit = _json("memorax_forbidden_artifact_audit.json")
    assert audit["child_add_intent_files"] == 0
    assert audit["child_add_completion_files"] == 0


def test_13_no_r03_memory_session():
    assert _json("memorax_forbidden_artifact_audit.json")[
        "r03_memory_session_count"] == 0


def test_14_write_visibility_not_applicable():
    contract = _json("r03_existing_memory_readonly_trial_contract.json")
    assert contract["memory_write_verified"] == "NOT_APPLICABLE"
    assert contract["memory_visibility_verified"] == "NOT_APPLICABLE"


def test_15_snapshot_freeze_replay():
    for root in TRIAL_ROOTS:
        base = OUT / "roots" / root / "memory_read"
        response = json.loads((base / "search_response.json").read_text())
        frozen = json.loads((base / "r02_memory_snapshot_normalized.json").read_text())
        replay = normalize_search_response(
            response, root_id=root, user_id=frozen["user_id"])
        assert replay == frozen


def test_16_source_turn_refs_not_inferred():
    for root in TRIAL_ROOTS:
        frozen = json.loads((OUT / "roots" / root / "memory_read" /
                             "r02_memory_snapshot_normalized.json").read_text())
        assert all("source_turn_refs" not in row for row in frozen["buckets"])


def test_17_whole_runtime_not_hard_gate():
    audit = _json("r03_core_generation_contract_compatibility.json")
    assert audit["historical_whole_runtime_match"] is False
    assert audit["valid"] is True


def test_18_core_component_drift_blocks_contract():
    audit = _json("r03_core_generation_contract_compatibility.json")
    assert all(row["contract_match"] for row in audit["component_rows"])
    assert audit["core_generation_contract_match"]


def test_19_expansion_temperature_is_point25():
    token = _json("r03_readonly_dialogue_token_preflight.json")
    assert token["effective_api_temperature"] == 0.25
    assert token["generation_planning_temperature"] == 0.45


def test_20_context_is_not_solver_16k():
    assert _json("r03_readonly_dialogue_token_preflight.json")[
        "context_ceiling"] == CONTEXT_CEILING == 40960


def test_21_append_only_output_contract():
    contract = _json("r03_readonly_dialogue_output_contract.json")
    assert len(contract["append_only_checks"]) == 5
    assert contract["r03_local_session_only"]


def test_22_local_turn_refs_unique():
    refs = build_local_r03_turn_refs(
        root_id=TRIAL_ROOTS[0], scene_turn_counts=(4, 5, 6))
    assert len({row["turn_ref"] for row in refs}) == 15
    assert all(row["turn_ref"].startswith("R03:S") for row in refs)
    assert all(":r03:s" in row["local_session_ref"] for row in refs)


def test_23_smoke_root_deterministic():
    plan = _json("r03_single_root_readonly_memory_smoke_plan.json")
    assert plan["root_id"] == TRIAL_ROOTS[0]
    assert plan["selection_order"] == list(TRIAL_ROOTS)


def test_24_source_files_unchanged():
    audit = _json("source_sha_audit.json")
    assert audit["source_manifests_modified"] == 0
    assert audit["all_sources_unchanged"]


def test_25_expansion_api_zero():
    assert _json("final_r03_memory_readonly_trial_preflight_report.json")[
        "expansion_api_calls"] == 0


def test_26_sidecars_valid():
    for path in OUT.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        sidecar = path.with_name(path.name + ".sha256")
        assert sidecar.is_file()
        assert sidecar.read_text().split()[0] == hashlib.sha256(
            path.read_bytes()).hexdigest()


def test_27_prompt_render_is_deterministic():
    root = TRIAL_ROOTS[0]
    base = OUT / "roots" / root
    response = json.loads((base / "memory_read" / "search_response.json").read_text())
    frozen = json.loads((base / "memory_read" /
                         "r02_memory_snapshot_normalized.json").read_text())
    projection = memory_prompt_projection(normalize_search_response(
        response, root_id=root, user_id=frozen["user_id"]))
    input_path = next(row["input_dialogue_path"] for row in _jsonl(
        "r03_input_dialogue_identity_manifest.jsonl") if row["root_id"] == root)
    card_path = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
                     "h4_round1_question_bank_feedback_compiler_v1_live_13/"
                     "r03_expansion_feedback_cards_prompt_view.jsonl")
    card = next(row for row in [json.loads(line) for line in card_path.read_text(
        encoding="utf-8").splitlines()] if row["root_id"] == root)
    kwargs = {
        "dialogue": json.loads(Path(input_path).read_text()),
        "memory_projection": projection,
        "feedback_card": card,
    }
    assert render_planner_dry_run(**kwargs) == render_planner_dry_run(**kwargs)


def test_28_nonempty_run_refuses_rerun():
    from .preflight import run_preflight
    try:
        run_preflight(OUT)
    except RuntimeError as error:
        assert str(error).startswith("output_root_not_empty:")
        return
    raise AssertionError("nonempty_run_was_not_rejected")
