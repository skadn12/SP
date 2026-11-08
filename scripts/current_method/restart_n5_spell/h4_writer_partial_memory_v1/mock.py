"""Exercise the frozen Writer entry with synthetic replies and network disabled."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from ..backends import BackendResponse
from ..core import atomic_json, sha256_file
from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import batch_fixture
from ..h4_writer_model_comparison_v1.transport import build_payload
from .sources import MEMORY_ONLY, evidence_catalog, memory_navigation_block


def read(path):
    return json.loads(Path(path).read_text())


def synthetic_candidates(expanded, slot_id, *, stage="primary"):
    """Schema/flow fixtures only; no claim of evidence semantics or model quality."""
    refs = list(evidence_catalog(expanded))[:3]
    if not refs:
        raise ValueError("MOCK_REQUIRES_AT_LEAST_ONE_EVIDENCE_REFERENCE")
    result = batch_fixture("SYNTHETIC " + slot_id + " " + stage)
    for row in result["candidates"]:
        row["evidence_refs"] = refs
        row["evidence_groups"] = {"synthetic_support": refs}
        if row["question_type"] == "ordering":
            items = row.pop("items")
            row.pop("gold")
            row.pop("evidence_refs")
            row["ordering_contract_version"] = "ordering_temporal_contract_v1"
            row["event_nodes"] = [
                {"item_id": item, "item_text": text, "actor": "Synthetic fixture actor",
                 "event_state": "completed", "evidence_refs": [refs[index % len(refs)]]}
                for index, (item, text) in enumerate(items.items())]
            row["temporal_relations"] = [
                {"before": f"I{i}", "after": f"I{i+1}", "evidence_refs": refs,
                 "temporal_basis": "Synthetic structural fixture only; not an evidence claim."}
                for i in range(1, 4)]
    return result


def exercise(run):
    from . import runtime
    from .prepare import verify

    run = Path(run)
    verify(run)
    schedule = read(run / "schedule.json")
    if isinstance(schedule, dict):
        schedule = schedule["slots"]
    outcomes = []
    scenarios = ("success", "bad_ordering_repair", "timeout_stop", "unknown_finish_stop",
                 "length_stop", "memory_abstention")
    with forbid_network():
        for scenario in scenarios:
            with tempfile.TemporaryDirectory(prefix="h4_partial_memory_mock_") as temporary:
                target = Path(temporary) / "run"
                shutil.copytree(run, target, ignore=shutil.ignore_patterns("executions", "__pycache__"))
                calls = []
                http_callbacks = []

                class Client:
                    def __init__(self, row, profile, callback):
                        self.row, self.profile, self.callback = row, profile, callback
                        self.config = SimpleNamespace(model=profile["model"], base_url=profile["base_url"], temperature=0.1)

                    def complete(self, *, prompt, **kwargs):
                        stage = "repair" if "REPAIR THE FOLLOWING BATCH" in prompt else "primary"
                        calls.append({**self.row, "stage": stage, "prompt": prompt})
                        self.callback(build_payload(self.profile, prompt))
                        http_callbacks.append({"order": self.row["order"], "stage": stage})
                        if scenario == "timeout_stop":
                            raise TimeoutError("offline synthetic timeout after request attempt")
                        mode = self.row["source_mode"]
                        expanded = read(target / "inputs" / mode / self.row["root_id"] / "expanded.json")
                        raw = synthetic_candidates(expanded, self.row["slot_id"], stage=stage)
                        if scenario == "bad_ordering_repair":
                            raw["candidates"][-1]["temporal_relations"] = []
                        if scenario == "memory_abstention" and mode == MEMORY_ONLY:
                            raw = {"candidates": [], "abstentions": [
                                {"question_type": kind, "status": "INSUFFICIENT_EVIDENCE",
                                 "reason": "Synthetic fixture: evidence shortage is an allowed result."}
                                for kind in ("single_choice", "multi_select", "ordering")]}
                        return BackendResponse(json.dumps(raw, ensure_ascii=False), "OFFLINE_SYNTHETIC", {
                            "finish_reason": (None if scenario == "unknown_finish_stop" else
                                              "length" if scenario == "length_stop" else "stop"),
                            "actual_model": "OFFLINE_SYNTHETIC", "returned_model": "OFFLINE_SYNTHETIC",
                            "completion_status": "RESPONSE_RECEIVED", "usage": None,
                            "http_attempt_count": 1, "transport_attempt_count": 1,
                            "transport_retry_count": 0, "input_tokens": None,
                            "output_tokens": None, "reasoning_tokens": None})

                result = {"scopes": {}}
                for scope in ("memory_assist", "memory_only"):
                    try:
                        result["scopes"][scope] = runtime.execute(
                            target, scope, authorized=True, allow_unknown_service=True,
                            client_factory=lambda row, profile, callback: Client(row, profile, callback))
                    except runtime.ExecutionStopped:
                        if scenario not in ("timeout_stop", "unknown_finish_stop", "length_stop"):
                            raise
                        result["scopes"][scope] = read(target / "executions" / scope / "reports/summary.json")
                        break
                expected = (1 if scenario in ("timeout_stop", "unknown_finish_stop", "length_stop")
                            else len(schedule) * (2 if scenario == "bad_ordering_repair" else 1))
                if len(calls) != expected or len(http_callbacks) != expected:
                    raise AssertionError(f"MOCK_CALL_COUNT_MISMATCH:{scenario}:{len(calls)}:{expected}")
                for call in calls:
                    mode = call["source_mode"]
                    if call["stage"] == "primary":
                        original = run / "policy_snapshot/primary_requests" / mode / call["root_id"] / call["slot_id"] / "prompt.txt"
                        if call["prompt"].encode() != original.read_bytes():
                            raise AssertionError("MOCK_FROZEN_PRIMARY_PROMPT_MISMATCH")
                    if mode != "TRANSCRIPT_ONLY":
                        expanded = read(run / "inputs" / mode / call["root_id"] / "expanded.json")
                        if call["prompt"].count(memory_navigation_block(expanded)) != 1:
                            raise AssertionError("MOCK_FROZEN_MEMORY_BLOCK_DRIFT")
                if scenario == "memory_abstention" and any(
                        call["source_mode"] == MEMORY_ONLY and call["stage"] == "repair" for call in calls):
                    raise AssertionError("MOCK_ABSTENTION_TRIGGERED_REPAIR")
                accepted = sum(scope["program_accepted"] for scope in result["scopes"].values())
                expected_accepted = {"success": 56, "bad_ordering_repair": 42,
                    "timeout_stop": 0, "unknown_finish_stop": 0, "length_stop": 0,
                    "memory_abstention": 48}[scenario]
                if accepted != expected_accepted:
                    raise AssertionError(f"MOCK_PROGRAM_ADMISSION_MISMATCH:{scenario}:{accepted}:{expected_accepted}")
                outcomes.append({"scenario": scenario, "status": "PASS", "mock_calls": len(calls),
                                 "synthetic_http_attempts": len(http_callbacks), "result": result,
                                 "external_calls": 0, "service_capacity_verified": False})
    verify(run)
    result = {"status": "PASS", "scenarios": outcomes, "external_calls": 0,
              "code": read(run / "audit/frozen_manifest.json").get("code", []),
              "profile_sha256": sha256_file(run / "profile.json"),
              "all_fourteen_real_primary_slots_exercised": len(schedule) == 14,
              "qualification": "Synthetic responses validate the frozen input, admission, and stop paths; model quality and service capacity remain untested."}
    atomic_json(run / "audit/mock_report.json", result)
    return result
