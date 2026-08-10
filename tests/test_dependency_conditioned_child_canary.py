import copy
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.dependency_conditioned_child_canary import (
    POSITIONAL_DEPENDENCY_BATCH_SCHEMA,
    apply_anti_leak_gate,
    assign_dependency_ids_by_position,
    build_dependency_planner_prompt,
    build_positional_dependency_planner_prompt,
    normalize_dependency_batch,
    project_authoritative_parent,
    validate_child_graph,
    validate_dependency_batch_schema,
    validate_positional_dependency_batch_schema,
    validate_qa_object,
)
from scripts.current_method.restart_n5_spell.input_adapter import load_lineage
from scripts.current_method.restart_n5_spell.run_dependency_conditioned_child_canary import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_LEGACY_ROOT,
    DEPENDENCY_AUDIT_VERSION,
    _usage,
    derive_route_conclusion,
    select_dependencies_for_stage1,
    select_realization_ids_for_qa,
    tree_manifest_sha256,
    validate_dependency_audit,
)
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file
from scripts.current_method.restart_n5_spell.lineage_three_stage_canary import ApiBudget


class DependencyConditionedChildCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded = load_lineage(DEFAULT_INPUT_ROOT, "conv-41", DEFAULT_LEGACY_ROOT)
        cls.parent = project_authoritative_parent(loaded["parent"])
        cls.events = {event["event_id"]: event for event in cls.parent["parent_graph"]["events"]}
        cls.aliases = {turn["turn_id"]: turn["alias"] for turn in cls.parent["parent_dialogue"]}
        cls.event = next(event for event in cls.parent["parent_graph"]["events"]
                         if event["evidence_turn_ids"] and
                         event["evidence_turn_ids"][0] in cls.aliases)
        cls.alias = cls.aliases[cls.event["evidence_turn_ids"][0]]

    def dependency(self, dependency_id="D1"):
        event = self.event
        return {
            "dependency_id": dependency_id,
            "status": "valid",
            "operator": "commitment_outcome",
            "parent_event_ids": [event["event_id"]],
            "parent_turn_aliases": [self.alias],
            "parent_operand": {
                "subject": str(event["entity"]), "predicate": str(event["slot"]),
                "object": "", "value": str(event["value"]),
                "time": str(event["time_index"] if event["time_index"] is not None else ""),
            },
            "planned_child_event": {
                "subject": str(event["entity"]), "predicate": "completed",
                "object": "Project X", "value": "completed", "time": "next month",
            },
            "joint_result": {"answer_type": "state", "value": "fulfilled",
                             "meaning": "The earlier commitment was fulfilled."},
            "unknown_from_parent_only": "Whether Project X was completed.",
            "unknown_from_planned_child_only": "Who had made the earlier commitment.",
            "forbidden_child_restatements": ["promised to organize Project X"],
            "natural_relation_explanation": "A later completion determines the outcome of an earlier commitment.",
        }

    def batch(self):
        rows = [self.dependency("D1")]
        for index in range(2, 5):
            rows.append({"dependency_id": f"D{index}",
                         "status": "no_valid_dependency", "reason": "No valid distinct dependency."})
        return {"dependencies": rows}

    def child_graph(self, dependency=None):
        dependency = dependency or self.dependency()
        planned = dependency["planned_child_event"]
        return {"dependency_id": dependency["dependency_id"], "child_graph_id": "child-1",
            "entity_bindings": [{"name": planned["subject"]}],
            "child_events": [{"event_id": "NEW:E1", **planned, "status": "completed"}],
            "time_order_constraints": ["NEW:E1 occurs after the parent history"],
            "realizes_planned_child_event_id": "NEW:E1"}

    def realizer(self, dependency=None):
        dependency = dependency or self.dependency()
        return {"dependency_id": dependency["dependency_id"], "session_id": "child-session-1",
            "dialogue_turns": [
                {"speaker": "Maria", "text": "How did Project X go this month?"},
                {"speaker": dependency["planned_child_event"]["subject"],
                 "text": "Project X is completed now."},
                {"speaker": "Maria", "text": "The final report was filed yesterday."},
                {"speaker": dependency["planned_child_event"]["subject"],
                 "text": "Good, we can close this update."},
            ], "verbalized_child_event_ids": ["NEW:E1"],
            "event_realization_turn_indices": {"NEW:E1": [0, 1]}}

    def test_parent_only_extraction_excludes_all_child_turns(self):
        self.assertEqual(self.parent["parent_count"], 237)
        self.assertEqual(self.parent["child_count_in_generation_input"], 0)
        self.assertTrue(all(turn["source_partition"] == "parent"
                            for turn in self.parent["parent_dialogue"]))
        self.assertTrue(all(turn["alias"].startswith("P")
                            for turn in self.parent["parent_dialogue"]))

    def test_dependency_schema_and_alias_only_output(self):
        payload = self.batch()
        self.assertEqual(validate_dependency_batch_schema(payload)["decision"],
                         "dependency_batch_accept")
        payload["dependencies"][0]["parent_quotes"] = ["forbidden"]
        self.assertEqual(validate_dependency_batch_schema(payload)["decision"],
                         "dependency_schema_invalid")

    def test_positional_dependency_schema_assigns_ids_locally(self):
        payload = self.batch()
        model_payload = {"dependencies": []}
        for row in payload["dependencies"]:
            copy_row = copy.deepcopy(row)
            del copy_row["dependency_id"]
            model_payload["dependencies"].append(copy_row)
        validation = validate_positional_dependency_batch_schema(model_payload)
        self.assertEqual(validation["decision"], "dependency_batch_accept")
        assigned = assign_dependency_ids_by_position(model_payload)
        self.assertEqual([row["dependency_id"] for row in assigned["dependencies"]],
                         ["D1", "D2", "D3", "D4"])
        self.assertEqual(assigned, payload)
        self.assertEqual(normalize_dependency_batch(
            assigned, self.parent)["dependencies"][0]["local_validation"], "accept")

    def test_positional_schema_rejects_model_owned_id(self):
        payload = self.batch()
        self.assertEqual(validate_positional_dependency_batch_schema(payload)["decision"],
                         "dependency_schema_invalid")
        self.assertFalse("dependency_id" in POSITIONAL_DEPENDENCY_BATCH_SCHEMA[
            "properties"]["dependencies"]["items"]["oneOf"][0]["properties"])

    def test_positional_prompt_changes_only_id_ownership_protocol(self):
        prompt = build_positional_dependency_planner_prompt(self.parent, "contract")
        self.assertIn("AUTHORITATIVE 237-TURN PARENT DIALOGUE", prompt)
        self.assertIn("Do not output dependency_id", prompt)
        self.assertIn("local program assigns D1-D4", prompt)
        self.assertIn("[P237]", prompt)
        self.assertNotIn("[C001]", prompt)

    def test_subject_mismatch_rejection(self):
        payload = self.batch()
        payload["dependencies"][0]["parent_operand"]["subject"] = "Wrong person"
        row = normalize_dependency_batch(payload, self.parent)["dependencies"][0]
        self.assertIn("subject_mismatch", row["local_validation_errors"])

    def test_unsupported_parent_operand_rejection(self):
        payload = self.batch()
        payload["dependencies"][0]["parent_operand"]["predicate"] = "invented_predicate"
        row = normalize_dependency_batch(payload, self.parent)["dependencies"][0]
        self.assertIn("unsupported_parent_operand", row["local_validation_errors"])

    def test_fabricated_parent_value_rejection(self):
        payload = self.batch()
        payload["dependencies"][0]["parent_operand"]["value"] = "three invented years"
        row = normalize_dependency_batch(payload, self.parent)["dependencies"][0]
        self.assertIn("fabricated_value", row["local_validation_errors"])

    def test_duplicate_dependency_support_rejection(self):
        payload = self.batch()
        payload["dependencies"][1] = self.dependency("D2")
        normalized = normalize_dependency_batch(payload, self.parent)
        self.assertEqual(normalized["dependencies"][0]["local_validation"], "accept")
        self.assertIn("duplicate_dependency_support",
                      normalized["dependencies"][1]["local_validation_errors"])

    def test_vague_joint_result_rejection(self):
        payload = self.batch()
        payload["dependencies"][0]["joint_result"] = {
            "answer_type": "state", "value": "alignment",
            "meaning": "The child aligns with the parent."}
        row = normalize_dependency_batch(payload, self.parent)["dependencies"][0]
        self.assertIn("vague_joint_result", row["local_validation_errors"])

    def test_combined_not_supported_audit_and_zero_pass_stop(self):
        normalized = normalize_dependency_batch(self.batch(), self.parent)
        source_sha = "a" * 64
        audit = {"audit_version": DEPENDENCY_AUDIT_VERSION,
            "source_dependency_contracts_sha256": source_sha, "items": [{
                "dependency_id": "D1", "parent_only_sufficient": False,
                "planned_child_only_sufficient": False, "combined_sufficient": False,
                "natural_relation": True, "minimal_parent_evidence": True, "notes": ""}]}
        validated = validate_dependency_audit(audit, normalized, source_sha)
        self.assertEqual(validated["items"][0]["final_dependency_judgment"],
                         "combined_not_supported")
        self.assertEqual(select_dependencies_for_stage1(normalized, validated), [])

    def test_child_graph_must_exactly_realize_planned_event(self):
        dep = self.dependency(); graph = self.child_graph(dep)
        parent_ids = set(self.events)
        self.assertEqual(validate_child_graph(graph, dep, parent_ids)["decision"],
                         "child_graph_accept")
        graph["child_events"][0]["subject"] = "Wrong person"
        self.assertIn("planned_child_event_mismatch:subject",
                      validate_child_graph(graph, dep, parent_ids)["reasons"])

    def test_parent_restatement_detection(self):
        dep = self.dependency(); raw = self.realizer(dep)
        raw["dialogue_turns"][2]["text"] = "You promised to organize Project X."
        gate = apply_anti_leak_gate(raw, dep, self.child_graph(dep))
        self.assertIn("parent_fact_restatement", gate["reasons"])

    def test_joint_answer_leakage_detection(self):
        dep = self.dependency(); raw = self.realizer(dep)
        raw["dialogue_turns"][2]["text"] = "The earlier commitment was fulfilled."
        gate = apply_anti_leak_gate(raw, dep, self.child_graph(dep))
        self.assertIn("joint_answer_leakage", gate["reasons"])

    def test_single_turn_shortcut_detection(self):
        dep = self.dependency()
        dep["parent_operand"]["object"] = "Project X"
        dep["parent_operand"]["value"] = "organize"
        raw = self.realizer(dep)
        raw["dialogue_turns"][1]["text"] = (
            "The plan to organize Project X is completed, so the commitment was fulfilled.")
        gate = apply_anti_leak_gate(raw, dep, self.child_graph(dep))
        self.assertIn("single_turn_shortcut", gate["reasons"])

    def test_qa_writer_cannot_change_aliases_or_joint_result(self):
        dep = self.dependency(); parent_aliases = [self.alias]; child_aliases = ["C001"]
        qa = {"dependency_id": "D1", "fixed_parent_turn_aliases": parent_aliases,
              "fixed_child_turn_aliases": child_aliases, "question": "Was it fulfilled?",
              "canonical_answer": "fulfilled", "answer_type": "state",
              "relation_preserved": True}
        self.assertEqual(validate_qa_object(qa, dep, parent_aliases, child_aliases)["decision"],
                         "qa_accept")
        drift = copy.deepcopy(qa); drift["fixed_parent_turn_aliases"] = ["P999"]
        self.assertIn("qa_alias_drift", validate_qa_object(
            drift, dep, parent_aliases, child_aliases)["reasons"])
        drift = copy.deepcopy(qa); drift["canonical_answer"] = "not fulfilled"
        self.assertIn("qa_joint_result_drift", validate_qa_object(
            drift, dep, parent_aliases, child_aliases)["reasons"])

    def test_stage1_failure_stops_qa_writer_selection(self):
        audit = {"items": [{"dependency_id": "D1",
                            "final_realization_judgment": "joint_answer_leakage"}]}
        self.assertEqual(select_realization_ids_for_qa(audit), [])

    def test_prompt_contains_parent_only_input(self):
        prompt = build_dependency_planner_prompt(self.parent, "contract")
        self.assertIn("AUTHORITATIVE 237-TURN PARENT DIALOGUE", prompt)
        self.assertIn("[P237]", prompt)
        self.assertNotIn("[C001]", prompt)

    def test_zero_request_accounting_and_tree_hash_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "config/run_manifest.json", {"run_id": "test"})
            ledger = RequestLedger(root, sha256_file(root / "config/run_manifest.json"), "live")
            usage = _usage(ledger)
            self.assertEqual(usage["total_paid_api_requests"], 0)
            self.assertEqual(usage["realizer_generations"], 0)
            before = tree_manifest_sha256(root)
            after = tree_manifest_sha256(root)
            self.assertEqual(before, after)

    def test_shared_api_budget_supports_fixed_custom_stages(self):
        budget = ApiBudget(normal_caps={"dependency_planner": 1},
                           format_retry_cap=1, physical_cap=2)
        budget.reserve("dependency_planner", is_retry=False)
        budget.reserve("dependency_planner", is_retry=True)
        self.assertEqual(budget.normal_counts["dependency_planner"], 1)
        self.assertEqual(budget.retry_counts["dependency_planner"], 1)

    def test_schema_exhaustion_is_not_reported_as_semantic_bottleneck(self):
        self.assertEqual(derive_route_conclusion(stage0_schema_accepted=False,
            dependency_pass=0, realization_pass=0, joint_qa=0),
            "dependency_planner_schema_invalid_exhausted")


if __name__ == "__main__":
    unittest.main()
