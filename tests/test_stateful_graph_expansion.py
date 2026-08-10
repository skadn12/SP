import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.core import atomic_json
from scripts.current_method.restart_n5_spell.stateful_graph_expansion import (
    apply_authoritative_scene_updates,
    build_stateful_graph_realizer_prompt,
    compile_stateful_scene_contracts,
    initialize_stateful_ledger,
    cross_arc_stage_conflict_gate,
    legacy_round_reclassification,
    stateful_completion_report,
    validate_and_apply_stateful_scene,
    validate_candidate_handoff_updates,
    validate_candidate_stage_updates,
    validate_stateful_outline,
)
from scripts.current_method.restart_n5_spell.stateful_graph_orchestrator import (
    StatefulGraphRoundOrchestrator,
)


def _outline():
    arcs = []
    for arc_index in range(1, 5):
        stages = []
        for stage_index in range(1, 5):
            object_id = f"arc{arc_index}_object{stage_index}"
            state_key = "review_status"
            stages.append({
                "description": f"Resolve review {arc_index}.{stage_index}.",
                "state_instance_id": f"{object_id}:{state_key}:v1",
                "state_key": state_key,
                "object_id": object_id,
                "event_identity_key": f"event_{arc_index}_{stage_index}",
                "stage_role": "base_fact_completion",
                "predecessor_stage_ref": None,
                "version": 1,
                "pre_state": "awaiting_review",
                "target_post_state": "approved",
                "allowed_post_states": ["approved", "rejected"],
                "terminal_post_states": ["approved", "rejected"],
                "completion_event_type": "review_decision",
                "mandatory_resolution": True,
                "authorized_result_speakers": ["John"],
                "completion_status": "achieved",
                "completion_temporality": "current_or_past",
                "completion_conditionality": "unconditional",
                "completion_evidence_phrases": ["I approve this version"],
                "predecessor_state_instance_id": None,
                "outcome_handoffs": [],
            })
        arcs.append({"event_id": f"E{arc_index}", "summary": f"Arc {arc_index}",
                     "stages": stages})
    return {
        "schema_version": "stateful_graph_outline_v4",
        "completion_contract_version": "structured_canonical_v1",
        "planned_event_arcs": arcs,
        "scene_plan": [{
            "scene_id": f"S{index}", "purpose": f"Scene {index}",
            "participants": ["John"], "required_parent_anchor_ids": ["P1"],
            "session_location": "office",
        } for index in range(1, 11)],
    }


def _payload(contract, *, text="I have reviewed it, and I approve this version.",
             status="completed", result_state="approved"):
    return {
        "dialogue_turns": [{"speaker": "John", "text": text}],
        "used_parent_anchor_ids": ["P1"],
        "advanced_event_arc_ids": list(contract["assigned_event_arc_ids"]),
        "candidate_stage_updates": [{
            "stage_id": stage_id, "status": status,
            "result_state": result_state,
            "evidence_turn_index": 0 if status != "carry_forward" else None,
            "result_speaker": "John" if status != "carry_forward" else None,
        } for stage_id in contract["assigned_event_stage_ids"]],
        "candidate_handoff_updates": [],
        "durable_assertion_candidates": [],
        "local_scene_facts": [],
    }


class StatefulGraphExpansionTests(unittest.TestCase):
    def setUp(self):
        self.outline = _outline()
        self.contracts = compile_stateful_scene_contracts(self.outline)
        self.ledger = initialize_stateful_ledger(self.contracts)

    def test_valid_outline_and_contract_shape(self):
        self.assertEqual([], validate_stateful_outline(self.outline))
        self.assertEqual(10, len(self.contracts))
        self.assertEqual(16, sum(len(row["assigned_event_stage_ids"])
                                 for row in self.contracts))
        self.assertTrue(all(
            row["completion_contract_version"] == "structured_canonical_v1"
            for row in self.contracts))

    def test_legacy_v3_outline_is_not_silently_interpreted_as_v4(self):
        legacy = deepcopy(self.outline)
        legacy["schema_version"] = "stateful_graph_outline_v3"
        errors = validate_stateful_outline(legacy)
        self.assertIn("stateful_outline_schema_version_invalid", errors)

    def test_graph_rejects_structured_completion_contradiction(self):
        outline = deepcopy(self.outline)
        stage = outline["planned_event_arcs"][0]["stages"][0]
        stage["completion_temporality"] = "future"
        errors = validate_stateful_outline(outline)
        self.assertIn("completion_contract_inconsistent", errors)
        self.assertIn(
            "mandatory_terminal_completion_not_current_or_past", errors)

    def test_graph_rejects_duplicate_cross_arc_base_completion(self):
        outline = deepcopy(self.outline)
        first = outline["planned_event_arcs"][0]["stages"][0]
        conflict = outline["planned_event_arcs"][1]["stages"][0]
        conflict["event_identity_key"] = first["event_identity_key"]
        gate = cross_arc_stage_conflict_gate(
            outline["planned_event_arcs"])
        self.assertIn("duplicate_completion_obligation", gate["errors"])
        self.assertIn(
            "cross_arc:duplicate_completion_obligation",
            validate_stateful_outline(outline))

    def test_assignment_is_not_completion(self):
        report = stateful_completion_report(self.contracts, self.ledger)
        self.assertEqual(16, report["counts"]["assigned"])
        self.assertEqual(0, report["counts"]["completed"])
        self.assertFalse(report["layers"]["semantic_transition_complete"])

    def test_explicit_authorized_terminal_evidence_is_accepted(self):
        contract = self.contracts[0]
        result = validate_and_apply_stateful_scene(
            _payload(contract), contract, self.ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])
        stage_id = contract["assigned_event_stage_ids"][0]
        self.assertEqual("completed",
                         result["ledger"]["stage_states"][stage_id]["status"])
        self.assertEqual("approved", result["ledger"]["state_instances"][
            contract["assigned_event_stages"][0]["state_instance_id"]][
                "current_state"])

    def test_typographic_apostrophe_matches_ascii_terminal_marker(self):
        outline = deepcopy(self.outline)
        stage = outline["planned_event_arcs"][0]["stages"][0]
        stage["completion_evidence_phrases"] = [
            "I've finalized the financial plan."]
        contracts = compile_stateful_scene_contracts(outline)
        ledger = initialize_stateful_ledger(contracts)
        payload = _payload(
            contracts[0], text="I’ve finalized the financial plan.")
        result = validate_and_apply_stateful_scene(
            payload, contracts[0], ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])

    def test_apostrophe_normalization_does_not_bypass_future_gate(self):
        outline = deepcopy(self.outline)
        stage = outline["planned_event_arcs"][0]["stages"][0]
        stage["completion_evidence_phrases"] = [
            "I've finalized the financial plan."]
        contracts = compile_stateful_scene_contracts(outline)
        ledger = initialize_stateful_ledger(contracts)
        payload = _payload(
            contracts[0],
            text="Once I’ve finalized the financial plan. we can submit it.")
        result = validate_and_apply_stateful_scene(
            payload, contracts[0], ledger)
        self.assertEqual("reject_stateful_scene", result["decision"])
        self.assertIn("terminal_evidence_is_future_or_conditional",
                      result["errors"])

    def test_future_or_conditional_evidence_cannot_complete(self):
        contract = self.contracts[0]
        payload = _payload(
            contract, text="Once I approve this version, we can submit it.")
        result = validate_and_apply_stateful_scene(payload, contract, self.ledger)
        self.assertEqual("reject_stateful_scene", result["decision"])
        self.assertIn("terminal_evidence_is_future_or_conditional",
                      result["errors"])

    def test_nonterminal_activity_cannot_be_claimed_as_completed(self):
        contract = self.contracts[0]
        for text in (
                "I started the review.", "I was assigned the review.",
                "I planned the review.", "I agreed to review it."):
            result = validate_and_apply_stateful_scene(
                _payload(contract, text=text), contract, self.ledger)
            self.assertEqual("reject_stateful_scene", result["decision"], text)
            self.assertIn("terminal_evidence_marker_missing",
                          result["errors"], text)

    def test_nonterminal_activity_is_valid_as_in_progress(self):
        contract = self.contracts[0]
        for text in (
                "I started the review.", "I was assigned the review.",
                "I planned the review.", "I agreed to review it."):
            result = validate_and_apply_stateful_scene(
                _payload(contract, text=text, status="in_progress",
                         result_state=None), contract, self.ledger)
            self.assertEqual("accept_stateful_scene", result["decision"], text)

    def test_completed_plan_wording_is_not_future_by_itself(self):
        contract = self.contracts[0]
        payload = _payload(
            contract, text="The plan is set, and I approve this version.")
        result = validate_and_apply_stateful_scene(payload, contract, self.ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])

    def test_later_future_sentence_does_not_negate_terminal_marker(self):
        contract = self.contracts[0]
        payload = _payload(contract, text=(
            "I have reviewed it, and I approve this version. "
            "I am ready to submit it tomorrow."))
        result = validate_and_apply_stateful_scene(payload, contract, self.ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])

    def test_realizer_prompt_explicitly_requires_dialogue_length(self):
        prompt = build_stateful_graph_realizer_prompt(
            contract=self.contracts[0], recent_sessions=[],
            active_handoffs=[])
        self.assertIn("dialogue_turns text itself must meet", prompt)
        self.assertIn("hard ranges", prompt)

    def test_realizer_prompt_contains_exact_handoff_whitelist(self):
        prompt = build_stateful_graph_realizer_prompt(
            contract=self.contracts[0], recent_sessions=[],
            active_handoffs=[])
        self.assertIn('"allowed_handoff_ids": []', prompt)
        self.assertIn(
            "candidate_handoff_updates must be an empty array", prompt)
        self.assertIn("Never invent, rename, normalize", prompt)

        prompt_with_handoff = build_stateful_graph_realizer_prompt(
            contract=self.contracts[0], recent_sessions=[],
            active_handoffs=[{"handoff_id": "H-EXACT"}])
        self.assertIn('"allowed_handoff_ids": [\n    "H-EXACT"',
                      prompt_with_handoff)

    def test_handoff_ids_require_exact_authoritative_membership(self):
        ledger = deepcopy(self.ledger)
        ledger["handoffs"]["H-EXACT"] = {
            "handoff_id": "H-EXACT", "status": "pending",
            "resolution_evidence_phrases": ["The task is resolved."],
            "authorized_result_speakers": ["John"],
        }
        exact = _payload(self.contracts[0])
        exact["candidate_handoff_updates"] = [{
            "handoff_id": "H-EXACT", "status": "carry_forward",
            "evidence_turn_index": None, "result_speaker": None,
        }]
        self.assertEqual([], validate_candidate_handoff_updates(
            exact, ledger)["errors"])

        unknown = deepcopy(exact)
        unknown["candidate_handoff_updates"][0]["handoff_id"] = "h-exact"
        before = deepcopy(ledger)
        result = validate_candidate_handoff_updates(unknown, ledger)
        self.assertEqual(["unknown_handoff_id"], result["errors"])
        self.assertEqual(before, ledger)

    def test_in_progress_does_not_change_state(self):
        contract = self.contracts[0]
        payload = _payload(contract, text="I am reviewing the current version.",
                           status="in_progress", result_state=None)
        result = validate_and_apply_stateful_scene(payload, contract, self.ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])
        instance_id = contract["assigned_event_stages"][0]["state_instance_id"]
        self.assertEqual("awaiting_review",
                         result["ledger"]["state_instances"][instance_id][
                             "current_state"])

    def test_pre_state_must_match_authoritative_ledger(self):
        contract = self.contracts[0]
        broken = deepcopy(self.ledger)
        instance_id = contract["assigned_event_stages"][0]["state_instance_id"]
        broken["state_instances"][instance_id]["current_state"] = "rejected"
        result = validate_candidate_stage_updates(
            _payload(contract), contract, broken)
        self.assertIn("pre_state_ledger_mismatch", result["errors"])

    def test_terminal_state_instance_cannot_be_reopened(self):
        broken = _outline()
        first = broken["planned_event_arcs"][0]["stages"][0]
        second = deepcopy(first)
        second["description"] = "Try to reopen the same completed object."
        second["pre_state"] = "approved"
        second["target_post_state"] = "rejected"
        second["completion_evidence_phrases"] = ["I reject this version"]
        broken["planned_event_arcs"][0]["stages"][1] = second
        self.assertIn("completed_state_instance_reopened",
                      validate_stateful_outline(broken))

    def test_stage_and_handoff_ledgers_are_independent(self):
        outline = _outline()
        stage = outline["planned_event_arcs"][0]["stages"][0]
        stage["outcome_handoffs"] = [{
            "handoff_key": "submit_approved_version",
            "description": "Submit the approved version.",
            "owner": "John", "created_on_post_state": "approved",
            "resolution_evidence_phrases": ["I submitted the approved version"],
            "authorized_result_speakers": ["John"],
        }]
        contracts = compile_stateful_scene_contracts(outline)
        ledger = initialize_stateful_ledger(contracts)
        contract = contracts[0]
        stage_result = validate_candidate_stage_updates(
            _payload(contract), contract, ledger)
        next_ledger = apply_authoritative_scene_updates(
            ledger, contract, stage_result["accepted_stage_updates"], [],
            stage_result["prospective_handoffs"])
        stage_id = contract["assigned_event_stage_ids"][0]
        self.assertEqual("completed", next_ledger["stage_states"][stage_id][
            "status"])
        self.assertEqual(1, len(next_ledger["handoffs"]))
        self.assertEqual("pending", next(iter(next_ledger["handoffs"].values()))[
            "status"])

    def test_negative_result_can_resolve_a_mandatory_stage(self):
        outline = _outline()
        stage = outline["planned_event_arcs"][0]["stages"][0]
        stage["target_post_state"] = "rejected"
        stage["completion_evidence_phrases"] = ["I reject this version"]
        contracts = compile_stateful_scene_contracts(outline)
        ledger = initialize_stateful_ledger(contracts)
        contract = contracts[0]
        payload = _payload(
            contract, text="I reviewed it, and I reject this version.",
            result_state="rejected")
        result = validate_and_apply_stateful_scene(payload, contract, ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])

    def test_four_layer_completion_report(self):
        ledger = deepcopy(self.ledger)
        for contract in self.contracts:
            ledger = validate_and_apply_stateful_scene(
                _payload(contract), contract, ledger)["ledger"]
        report = stateful_completion_report(self.contracts, ledger)
        self.assertEqual({
            "scene_realization_complete": True,
            "stage_coverage_complete": True,
            "semantic_transition_complete": True,
            "handoff_accounting_complete": True,
        }, report["layers"])
        self.assertTrue(report["complete"])

    def test_legacy_migration_does_not_trust_completed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frozen").mkdir()
            (root / "accepted").mkdir()
            (root / "frozen/expansion_outline.json").write_text(json.dumps({
                "schema_version": "stage3_expansion_outline_v1"}))
            contracts = [{"assigned_event_stage_ids": ["E1:stage_01"]}]
            (root / "frozen/graph_scene_contracts.json").write_text(json.dumps({
                "contracts": contracts}))
            (root / "accepted/session_01.json").write_text(json.dumps({
                "completed_event_stage_ids": ["E1:stage_01"]}))
            (root / "unresolved_hooks.json").write_text(json.dumps([{
                "hook_id": "H0001", "description": "Review remains pending."}]))
            report = legacy_round_reclassification(root)
        self.assertEqual(0, report["v2_authoritative_completed_stage_count"])
        self.assertEqual("unknown_legacy",
                         report["stage_states"][0]["reclassified_status"])
        self.assertEqual("pending", report["handoffs"][0]["status"])
        self.assertEqual([], report["legacy_hook_tracking"][0][
            "resolved_in_sessions"])
        self.assertFalse(report["layers"]["semantic_transition_complete"])

    def test_stateful_orchestrator_executes_without_v1_auto_completion(self):
        contracts = self.contracts

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                lines = 0 if not text else text.count("\n") + 1
                return list(range(lines * 50))

        class Realizer:
            model_identifier = "mock-stateful-realizer"

            def generate_session(self, request):
                contract = contracts[request.session_index - 1]
                turns = [{"speaker": "John", "text": (
                    "I have reviewed the item, and I approve this version. ")}
                    for _ in range(4)]
                payload = {
                    "dialogue_turns": turns,
                    "used_parent_anchor_ids": ["P1"],
                    "advanced_event_arc_ids": contract[
                        "assigned_event_arc_ids"],
                    "candidate_stage_updates": [{
                        "stage_id": stage_id, "status": "completed",
                        "result_state": "approved", "evidence_turn_index": 0,
                        "result_speaker": "John",
                    } for stage_id in contract["assigned_event_stage_ids"]],
                    "candidate_handoff_updates": [],
                    "durable_assertion_candidates": [],
                    "local_scene_facts": [],
                }
                return BackendResponse(json.dumps(payload), self.model_identifier)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "config/run_manifest.json", {
                "run_id": "stateful-mock"})
            runner = StatefulGraphRoundOrchestrator(
                run_root=root, manifest={"run_id": "stateful-mock"},
                contracts=contracts, tokenizer=Tokenizer(), realizer=Realizer(),
                backend_mode="mock")
            result = runner.run()
            self.assertTrue(result["completion"]["complete"])
            self.assertEqual(10, len(result["sessions"]))
            self.assertEqual(16, result["completion"]["counts"]["completed"])
            self.assertFalse(any(
                "completed_event_stage_ids" in session
                for session in result["sessions"]))


if __name__ == "__main__":
    unittest.main()
