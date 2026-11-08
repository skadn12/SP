"""No-network budget/recovery tests; fixtures never represent experiment QA."""
from __future__ import annotations

import json
from pathlib import Path
import re
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from ..backends import BackendResponse
from ..core import atomic_json, atomic_text, sha256_file, sha256_bytes, canonical_json
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import batch_fixture, expanded_fixture
from . import runtime
from .constants import GENERATION, arm_policy
from .runtime import repair_spec


def prompt_spec(expanded, contract, arm_id, run=None):
    return build_prompt(expanded, contract, intent_guidance=True,
                        ordering_policy=arm_policy(arm_id))


def seed_run(temp, source_run=None):
    """Create a wholly temporary diagnostic identity, never historical QA."""
    run = Path(temp)
    run.mkdir(parents=True, exist_ok=True)
    source = Path(source_run) if source_run else None
    for root in runtime.ROOTS:
        expanded = expanded_fixture()
        expanded["root_id"] = root
        contracts = [{"intent_id": f"intent_{i}", "intent_type": "state_transition",
                      "discriminative_variable": f"synthetic reservation {i}",
                      "required_evidence_groups": {"before_state": ["D001"], "trigger": ["D002"],
                                                   "after_state": ["D003"]}} for i in (1, 2, 3)]
        if source:
            expanded = runtime.read(source / "inputs/shared" / root / "expanded_dialogue_clean.json")
            contract_path = source / "inputs/shared" / root / "contracts.json"
            if not contract_path.exists():
                contract_path = (source / "arms/V3_OFF/qa/shards/shard-00/roots" / root / "reasoning_contracts.json")
            contracts = runtime.read(contract_path)
        folder = run / "inputs/shared" / root
        atomic_json(folder / "expanded_dialogue_clean.json", expanded)
        atomic_json(folder / "contracts.json", contracts)
    schedule = runtime.expected_schedule()
    atomic_json(run / "schedule.json", schedule)
    manifest = []
    for row in schedule:
        folder = run / "inputs/shared" / row["root_id"]
        prompt = prompt_spec(runtime.read(folder / "expanded_dialogue_clean.json"),
                              runtime.read(folder / "contracts.json")[row["raw_index"]],
                              row["arm_id"], run)["prompt"]
        target = run / "policy_snapshot/primary_requests" / row["arm_id"] / row["root_id"] / row["slot_id"]
        atomic_text(target / "prompt.txt", prompt)
        manifest.append({**row, "source": "contract_guided", "intent_guidance": True,
                         "slot_suffix_sha256": sha256_bytes(b""),
                         "ordering_policy": arm_policy(row["arm_id"]),
                         "hooks": None, "prompt_sha256": sha256_bytes(prompt.encode())})
    atomic_text(run / "slot_call_manifest.jsonl", "".join(canonical_json(row) + "\n" for row in manifest))
    atomic_json(run / "audit/frozen_manifest.json", {"fixture": "OFFLINE_TEST_ONLY"})
    atomic_json(run / "audit/preflight_report.json", {"status": "PASS", "fixture": True})
    atomic_json(run / "execution_contract.json", {"generation": GENERATION,
        "planned_primary_calls": 12, "maximum_repair_calls": 12, "maximum_logical_calls": 24,
        "roots": list(runtime.ROOTS), "arms": list(runtime.ARMS),
        "per_arm": {"primary": 6, "repair": 6, "total": 12},
        "per_root_arm": {"primary": 3, "repair": 3, "total": 6},
        "per_cell": {"primary": 1, "repair": 1}, "context_reserve_tokens": 512})
    return run


def synthetic_candidates(prompt, counts=(2, 1, 1)):
    """Structurally valid responses only; no semantic QA claim."""
    match = re.search(r'"intent_reference":\s*"([^"]+)"', prompt)
    slot = match.group(1) if match else "unknown-slot"
    result = batch_fixture("SYNTHETIC " + slot, counts)
    dialogue = prompt.split("DIALOGUE:\n", 1)[1]
    refs = [line.split(" | ", 1)[0] for line in dialogue.splitlines() if " | " in line][:3]
    for row in result["candidates"]:
        row["evidence_refs"] = refs
        row["evidence_groups"] = dict(zip(("before_state", "trigger", "after_state"), ([x] for x in refs)))
        if row["question_type"] == "ordering" and "ORDERING EVENT CONTRACT" in prompt:
            items = row.pop("items")
            row.pop("gold")
            row.pop("evidence_refs")
            row["ordering_contract_version"] = "ordering_temporal_contract_v1"
            row["event_nodes"] = [{"item_id": item, "item_text": text, "actor": "Synthetic actor",
                "event_state": "completed", "evidence_refs": [refs[index % len(refs)]]}
                for index, (item, text) in enumerate(items.items())]
            row["temporal_relations"] = [{"before": f"I{i}", "after": f"I{i+1}",
                "evidence_refs": refs, "temporal_basis": "Synthetic test relation; no content claim."}
                for i in range(1, 4)]
    return result


class FakeTokenPreflight:
    def __init__(self, *args, block_repair=False):
        self.calls, self.usages = [], []
        self.block_repair = block_repair

    def check(self, prompt, max_output_tokens, request_id):
        self.calls.append((prompt, max_output_tokens, request_id))
        if self.block_repair and "::repair::" in request_id:
            raise RuntimeError("TOKEN_CONTEXT_EXCEEDED:synthetic")
        return {"status": "PASS"}

    def verify_usage(self, request_id, usage):
        self.usages.append((request_id, usage))


class FakeClient:
    def __init__(self, calls, reply=None):
        self.calls, self.reply = calls, reply
        self.config = SimpleNamespace(model="Qwen/Qwen3-14B", base_url="https://api.siliconflow.cn/v1")

    def complete(self, **request):
        self.calls.append(request)
        raw = self.reply(request) if callable(self.reply) else self.reply
        if isinstance(raw, BaseException):
            raise raw
        if raw is None:
            raw = synthetic_candidates(request["prompt"])
        return BackendResponse(raw if isinstance(raw, str) else json.dumps(raw), self.config.model,
                               {"usage": None, "transport_attempt_count": 0})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = seed_run(self.temporary.name)
        self.calls, self.verifications = [], []
        self.tokens = FakeTokenPreflight()
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("external network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        for target in ("urllib.request.urlopen", "scripts.current_method.restart_n5_spell.live_backends.OpenAICompatibleChallengerClient"):
            guard = patch(target, side_effect=AssertionError("offline tests must not construct or invoke a service client"))
            guard.start()
            self.addCleanup(guard.stop)

    def execute(self, reply=None, verifier=None):
        return runtime.execute_schedule(self.run,
            client_factory=lambda row: FakeClient(self.calls, reply),
            verifier=verifier or (lambda: self.verifications.append(True)), token_preflight=self.tokens)

    def test_all_12_fixed_primaries_isolated_and_resume_without_sends(self):
        outcome = self.execute()
        self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 0))
        self.assertEqual(len(self.calls), 12)
        self.assertEqual(len(self.tokens.calls), 12)
        self.assertTrue(all(call[1] == 8192 for call in self.tokens.calls))
        self.assertEqual(len({call[2] for call in self.tokens.calls}), 12)
        self.assertEqual(len(runtime.rows(self.run / "reports/runtime_stage_records.jsonl")), 12)
        final = runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")
        self.assertEqual(len(final), 48)
        self.assertEqual(len({tuple(row["analysis_identity"]) for row in final}), 48)
        self.assertEqual([s["valid_candidate_count"] for s in outcome["root_arm_outcomes"]], [12] * 4)
        replay = self.execute()
        self.assertEqual(replay, outcome)
        self.assertEqual(len(self.calls), 12)

    def test_all_24_allowed_once_repair_replaces_primary_and_preserves_rows(self):
        def reply(request):
            repair = "REPAIR THE FOLLOWING BATCH" in request["prompt"]
            return synthetic_candidates(request["prompt"], (2, 1, 1) if repair else (2, 1, 0))
        outcome = self.execute(reply)
        self.assertEqual((outcome["primary_calls"], outcome["repair_calls"], len(self.calls)), (12, 12, 24))
        events = [e for e in runtime.rows(self.run / "audit/call_events.jsonl") if e["event"] == "new_logical_call"]
        self.assertEqual([e["stage"] for e in events], ["primary", "repair"] * 12)
        self.assertEqual(len({e["global_request_id"] for e in events}), 24)
        stages = runtime.rows(self.run / "reports/runtime_stage_records.jsonl")
        self.assertEqual(len(stages), 24)
        self.assertTrue(all(s["replaced_by_repair"] for s in stages if s["stage"] == "primary"))
        self.assertTrue(all(s["stage"] == "repair" for s in runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")))
        self.execute(reply)
        self.assertEqual(len(self.calls), 24)

    def test_manifest_actual_shared_args_prompts_and_sidecars_keep_both_intents(self):
        with patch.object(generation, "execute_writer_batch", wraps=generation.execute_writer_batch) as shared, \
                patch.object(generation, "generate_root_intents", side_effect=AssertionError("Miner forbidden")), \
                patch.object(generation, "build_free_prompt", side_effect=AssertionError("Free forbidden")):
            self.execute()
        self.assertEqual(shared.call_count, 12)
        schedule = runtime.frozen_schedule(self.run)
        for row, call, sent in zip(schedule, shared.call_args_list, self.calls):
            self.assertIs(call.kwargs["intent_guidance"], True)
            self.assertEqual(call.kwargs["source"], "contract_guided")
            self.assertIsNone(call.kwargs["hooks"])
            self.assertEqual(call.kwargs["slot_suffix"], "")
            self.assertEqual(call.kwargs["ordering_policy"], arm_policy(row["arm_id"]))
            self.assertIn("CONTRACT GUIDANCE:", sent["prompt"])
            self.assertIn(call.kwargs["contract"]["discriminative_variable"], sent["prompt"])
            self.assertNotIn("BEGIN_SCMS_QUALITY_GUIDANCE_V1", sent["prompt"])
            self.assertEqual("ORDERING EVENT CONTRACT" in sent["prompt"], row["arm_id"] == runtime.ARMS[1])
            manifest = runtime.rows(self.run / "slot_call_manifest.jsonl")[row["order"] - 1]
            self.assertEqual(manifest["prompt_sha256"], sha256_bytes(sent["prompt"].encode()))
        for filename in ("runtime_stage_records.jsonl", "runtime_final_candidates.jsonl"):
            for record in runtime.rows(self.run / "reports" / filename):
                self.assertIs(record["intent_guidance"], True)
                self.assertEqual(record["contract_alignment_analysis"], "AVAILABLE_FOR_OFFLINE_CONTENT_REVIEW")
                self.assertEqual(record["source"], "contract_guided")
        for state in runtime.audit_reservations(self.run).values():
            self.assertIs(state["request_metadata"]["intent_guidance"], True)

    def test_manifest_cannot_silently_disable_second_arm_intent(self):
        manifest = runtime.rows(self.run / "slot_call_manifest.jsonl")
        manifest[1]["intent_guidance"] = False
        atomic_text(self.run / "slot_call_manifest.jsonl", "".join(canonical_json(r) + "\n" for r in manifest))
        with self.assertRaisesRegex(runtime.GuardrailStop, "stage_arguments_drift"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_both_repair_prompts_preserve_builder_with_same_ordering_policy(self):
        def reply(request):
            return synthetic_candidates(request["prompt"],
                (2, 1, 1) if "REPAIR THE FOLLOWING BATCH" in request["prompt"] else (2, 1, 0))
        self.execute(reply)
        records = runtime.rows(self.run / "reports/runtime_stage_records.jsonl")
        for row in runtime.frozen_schedule(self.run):
            stages = [r for r in records if all(r[key] == row[key]
                      for key in ("arm_id", "root_id", "slot_id"))]
            primary, repaired = stages
            folder = self.run / "inputs/shared" / row["root_id"]
            expanded = runtime.read(folder / "expanded_dialogue_clean.json")
            contract = runtime.read(folder / "contracts.json")[row["raw_index"]]
            # The archived JSON sidecar canonicalizes mapping order; reconstruct
            # this synthetic response in its original transport key order.
            original_raw = synthetic_candidates(primary["prompt"]["prompt"], (2, 1, 0))
            expected = repair_spec(expanded, contract, original_raw, row["arm_id"], self.run)
            self.assertEqual(repaired["prompt"], expected)
            self.assertTrue(all("BEGIN_SCMS_QUALITY_GUIDANCE_V1" not in r["prompt"]["prompt"] for r in stages))

    def test_invalid_json_uses_original_raw_none_repair_rule_in_both_arms(self):
        def reply(request):
            return synthetic_candidates(request["prompt"]) if "REPAIR THE FOLLOWING BATCH" in request["prompt"] else "{bad-json"
        outcome = self.execute(reply)
        self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 12))
        records = runtime.rows(self.run / "reports/runtime_stage_records.jsonl")
        for record in records:
            if record["stage"] == "primary":
                self.assertIsNone(record["raw_parsed"])
                self.assertIsNotNone(record["parse_error"])
            else:
                prompt = record["prompt"]["prompt"]
                self.assertTrue(prompt.endswith("\n{}"))
                self.assertNotIn("{bad-json", prompt)

    def test_extra_legal_rows_are_kept_without_repair(self):
        outcome = self.execute(lambda request: synthetic_candidates(request["prompt"], (3, 1, 1)))
        self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 0))
        final = runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")
        self.assertEqual(len(final), 60)
        self.assertEqual([row["valid_candidate_count"] for row in outcome["root_arm_outcomes"]], [15] * 4)

    def test_sc_ms_deficits_and_structural_failure_keep_original_repair_trigger(self):
        for fault in ("missing_sc", "missing_ms", "invalid_sc_evidence"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                run, calls = seed_run(temporary), []
                def reply(request):
                    result = synthetic_candidates(request["prompt"])
                    if len(calls) == 1:
                        if fault == "invalid_sc_evidence":
                            result["candidates"][0].pop("evidence_refs")
                        else:
                            counts = (1, 1, 1) if fault == "missing_sc" else (2, 0, 1)
                            result = synthetic_candidates(request["prompt"], counts)
                    return result
                outcome = runtime.execute_schedule(run, client_factory=lambda row: FakeClient(calls, reply),
                    verifier=lambda: None, token_preflight=FakeTokenPreflight())
                self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 1))
                first = runtime.rows(run / "reports/runtime_stage_records.jsonl")[0]
                self.assertTrue(first["replaced_by_repair"])
                self.assertFalse(first["validation"]["minimum_acceptance"]["accepted"])
                if fault == "invalid_sc_evidence":
                    self.assertNotIn("evidence_refs", first["raw_parsed"]["candidates"][0])
                self.assertEqual(len(runtime.rows(run / "reports/runtime_final_candidates.jsonl")), 48)

    def test_insufficient_repair_replaces_without_salvage_in_both_arms(self):
        def reply(request):
            return synthetic_candidates(request["prompt"],
                (0, 0, 1) if "REPAIR THE FOLLOWING BATCH" in request["prompt"] else (2, 1, 0))
        outcome = self.execute(reply)
        self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 12))
        self.assertEqual(runtime.rows(self.run / "reports/runtime_final_candidates.jsonl"), [])
        self.assertEqual(len(runtime.rows(self.run / "reports/runtime_stage_records.jsonl")), 24)

    def test_final_validator_rejects_single_gold_multi_select_without_repair_or_fix(self):
        def reply(request):
            result = synthetic_candidates(request["prompt"])
            result["candidates"][2]["gold"] = ["A"]
            return result
        outcome = self.execute(reply)
        self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 0))
        final = runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")
        self.assertEqual(len(final), 36)
        self.assertTrue(all(row["question_type"] != "multi_select" for row in final))
        for record in runtime.rows(self.run / "reports/runtime_stage_records.jsonl"):
            self.assertEqual(record["raw_parsed"]["candidates"][2]["gold"], ["A"])
        for arm in runtime.ARMS:
            for root in runtime.ROOTS:
                audit = runtime.read(runtime.root_path(self.run, arm, root) / "validity_audit.json")
                rejected = [r for r in audit["rows"] if "multi_gold_invalid" in r["blocking_errors"]]
                self.assertEqual(len(rejected), 3)

    def test_temporal_arm_slots_have_distinct_ledger_and_candidate_identity(self):
        # Independence is identified by arm/root/slot/stage, never by prompt.
        self.execute()
        b_states = [s for s in runtime.audit_reservations(self.run).values()
                    if s["request_metadata"]["arm_id"] == runtime.ARMS[1]]
        self.assertEqual(len(b_states), 6)
        self.assertEqual(len({s["request_metadata"]["global_request_id"] for s in b_states}), 6)
        self.assertTrue(all("intent-miner" not in s["request_id"] for s in b_states))

    def test_unknown_completion_stops_before_repair_and_resume(self):
        with self.assertRaises(runtime.GuardrailStop):
            self.execute(lambda request: TimeoutError("synthetic unknown completion"))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(list(self.run.glob("**/ledger/requests/*.json"))), 1)
        with self.assertRaises(runtime.GuardrailStop):
            self.execute()
        self.assertEqual(len(self.calls), 1)

    def test_http_5xx_unknown_versus_known_429_budgeted(self):
        for code in (500, 429):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                run, calls = seed_run(temporary), []
                def reply(request):
                    if len(calls) == 1:
                        return urllib.error.HTTPError("https://invalid.test", code, "synthetic", {}, None)
                    return synthetic_candidates(request["prompt"])
                invoke = lambda: runtime.execute_schedule(run, client_factory=lambda row: FakeClient(calls, reply),
                    verifier=lambda: None, token_preflight=FakeTokenPreflight())
                if code == 500:
                    with self.assertRaises(runtime.GuardrailStop):
                        invoke()
                    self.assertEqual(len(calls), 1)
                else:
                    outcome = invoke()
                    self.assertEqual((outcome["primary_calls"], outcome["repair_calls"]), (12, 1))
                    self.assertEqual(len(calls), 13)
                    invoke()
                    self.assertEqual(len(calls), 13)

    def test_swallowed_freeze_exception_sticky_after_call_no_repair(self):
        def verifier():
            self.verifications.append(True)
            if len(self.verifications) > 1:
                raise RuntimeError("synthetic freeze drift")
        with self.assertRaises(runtime.GuardrailStop):
            self.execute(verifier=verifier)
        self.assertEqual(self.calls, [])
        self.assertTrue(runtime.rows(self.run / "audit/guardrail_events.jsonl"))

    def test_dynamic_repair_capacity_stop_without_send(self):
        self.tokens = FakeTokenPreflight(block_repair=True)
        with self.assertRaises(runtime.GuardrailStop):
            self.execute(lambda request: synthetic_candidates(request["prompt"], (2, 1, 0)))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(list(self.run.glob("**/ledger/requests/*.json"))), 1)
        self.assertEqual(len(self.tokens.calls), 2)

    def test_primary_prompt_drift_stops_without_send(self):
        first = runtime.frozen_schedule(self.run)[0]
        path = self.run / "policy_snapshot/primary_requests" / first["arm_id"] / first["root_id"] / first["slot_id"] / "prompt.txt"
        path.write_text("changed frozen fixture", encoding="utf-8")
        with self.assertRaises(runtime.GuardrailStop):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_missing_or_moved_state_cannot_restore_budget(self):
        self.execute()
        path = next(self.run.glob("**/ledger/requests/*.json"))
        path.unlink()
        with self.assertRaises(runtime.GuardrailStop):
            self.execute()
        self.assertEqual(len(self.calls), 12)

    def test_cell_checkpoint_drift_not_silently_regenerated(self):
        self.execute()
        first = runtime.frozen_schedule(self.run)[0]
        path = runtime.cell_path(self.run, first)
        value = runtime.read(path)
        value["result"]["candidates"] = []
        atomic_json(path, value)
        with self.assertRaises(runtime.GuardrailStop):
            self.execute()
        self.assertEqual(len(self.calls), 12)

    def test_incomplete_stage_completed_cache_reuses_without_send(self):
        # Interrupt after real stage output but before cell checkpoint; restore
        # re-enters the same ledger identity and the original cache verifies it.
        original = generation.execute_writer_batch
        def interrupt(**kwargs):
            original(**kwargs)
            raise KeyboardInterrupt("synthetic before checkpoint")
        with patch.object(generation, "execute_writer_batch", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        self.assertEqual(len(self.calls), 1)
        self.execute()
        self.assertEqual(len(self.calls), 12)
        self.assertTrue(any(e["event"] == "completed_cache_reused_no_send"
                            for e in runtime.rows(self.run / "audit/call_events.jsonl")))

    def test_explicit_scope_and_budget_reject_extra_request(self):
        self.execute()
        row = runtime.frozen_schedule(self.run)[0]
        ledger = runtime.BoundedLedger(self.run, row, manifest_sha256=sha256_file(self.run / "audit/frozen_manifest.json"),
            verifier=lambda: None, token_preflight=self.tokens)
        with self.assertRaises(runtime.GuardrailStop):
            ledger.invoke(request_id="outside-experiment", attempt=1, intent="h4_k3_intent_miner",
                          request_metadata={}, backend_identity={})
        self.assertEqual(len(self.calls), 12)

    def test_schedule_cannot_reorder_or_add_identity_fields(self):
        schedule = runtime.read(self.run / "schedule.json")
        schedule[0]["request_id"] += "-other"
        atomic_json(self.run / "schedule.json", schedule)
        with self.assertRaisesRegex(runtime.GuardrailStop, "schedule_drift"):
            self.execute()
        self.assertFalse(self.calls)

    def test_temporal_graph_failure_repair_retains_raw_and_drops_whole_batch(self):
        def reply(request):
            result = synthetic_candidates(request["prompt"])
            for row in result["candidates"]:
                if row.get("ordering_contract_version"):
                    row["temporal_relations"] = []
            return result
        result = self.execute(reply)
        self.assertEqual((result["primary_calls"], result["repair_calls"]), (12, 6))
        finals = runtime.rows(self.run / "reports/runtime_final_candidates.jsonl")
        self.assertEqual(len(finals), 24)
        self.assertTrue(all(row["arm_id"] == runtime.ARMS[0] for row in finals))
        stages = [row for row in runtime.rows(self.run / "reports/runtime_stage_records.jsonl")
                  if row["arm_id"] == runtime.ARMS[1]]
        self.assertEqual(len(stages), 12)
        for row in stages:
            ordering = row["validation"]["candidates"][-1]
            self.assertFalse(ordering["schema_valid"])
            self.assertIn("ORDER_NOT_DETERMINED_BY_CONSTRAINTS", ordering["schema_errors"])
            self.assertFalse(row["validation"]["minimum_acceptance"]["accepted"])
            self.assertNotIn("gold", row["raw_parsed"]["candidates"][-1])

    def test_budget_contract_cannot_be_raised(self):
        contract = runtime.read(self.run / "execution_contract.json")
        contract["maximum_logical_calls"] = 25
        atomic_json(self.run / "execution_contract.json", contract)
        with self.assertRaises(runtime.GuardrailStop):
            self.execute()
        self.assertEqual(len(self.calls), 0)


if __name__ == "__main__":
    unittest.main()
