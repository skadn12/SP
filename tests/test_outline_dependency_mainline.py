import copy
import unittest

from scripts.current_method.restart_n5_spell.outline_dependency_mainline import (
    QA_SCHEMA, apply_anti_leak_gate, assign_outline_dependency_ids,
    build_planner_prompt, normalize_outline_dependencies, validate_child_graph,
    validate_planner_schema, validate_qa, validate_realizer,
)
from scripts.current_method.restart_n5_spell.run_outline_dependency_mainline import _source


class OutlineDependencyMainlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _source()

    def row(self):
        return {"status": "valid", "operator": "commitment_outcome",
            "parent_anchor_ids": ["P1"],
            "parent_condition": {"subject": "John", "predicate": "intends",
                "object": "enter local politics", "value": "help the community",
                "time": "before the child continuation"},
            "planned_child_stage": "opportunity",
            "planned_child_event": {"subject": "John", "predicate": "learns_about",
                "object": "a city advisory opening", "value": "the opening is available",
                "time": "during the continuation"},
            "joint_result": {"answer_type": "state", "value": "The opening advances John's earlier political goal.",
                "meaning": "The later opportunity is a concrete route toward the earlier stated goal."},
            "unknown_from_parent_only": "Whether a concrete opening appears.",
            "unknown_from_planned_child_only": "Whether John had an earlier political goal.",
            "forbidden_child_restatements": ["hoping to get into local politics"],
            "natural_relation_explanation": "An opening supplies the later opportunity for an earlier goal."}

    def batch(self):
        return {"dependencies": [self.row(),
            {"status": "no_valid_dependency", "reason": "No valid E2 dependency."},
            {"status": "no_valid_dependency", "reason": "No valid E3 dependency."},
            {"status": "no_valid_dependency", "reason": "No valid E4 dependency."}]}

    def dependency(self):
        return normalize_outline_dependencies(self.batch(), self.source["outline"],
            self.source["parent_dialogue"])["dependencies"][0]

    def child_graph(self):
        dep = self.dependency(); planned = dep["planned_child_event"]
        return {"dependency_id": "D1", "arc_id": "E1", "stage": "opportunity",
            "child_graph_id": "outline-child-D1", "child_events": [{
                "event_id": "D1:NEW:1", **planned, "status": "completed"}],
            "realizes_planned_child_event_id": "D1:NEW:1"}

    def realizer(self):
        return {"dependency_id": "D1", "session_id": "outline-D1",
            "dialogue_turns": [
                {"speaker": "Maria", "text": "I saw a new notice at city hall."},
                {"speaker": "John", "text": "There is a city advisory opening available."},
                {"speaker": "Maria", "text": "The application details are on the notice."},
                {"speaker": "John", "text": "I'll read them carefully tonight."}],
            "verbalized_child_event_ids": ["D1:NEW:1"],
            "event_realization_turn_indices": {"D1:NEW:1": [0, 1]}}

    def test_source_is_outline_not_flat_semantic_graph(self):
        self.assertEqual(self.source["outline"]["outline_role"],
                         "dialogue_expansion_plan_only")
        self.assertEqual(len(self.source["outline"]["planned_event_arcs"]), 4)
        self.assertEqual(len(self.source["outline"]["parent_anchors"]), 8)
        self.assertEqual(len(self.source["parent_dialogue"]), 237)

    def test_planner_schema_and_positional_ids(self):
        batch = self.batch()
        self.assertEqual(validate_planner_schema(batch)["decision"],
                         "outline_dependency_batch_accept")
        assigned = assign_outline_dependency_ids(batch)
        self.assertEqual([(row["dependency_id"], row["arc_id"])
                          for row in assigned["dependencies"]],
                         [("D1", "E1"), ("D2", "E2"), ("D3", "E3"), ("D4", "E4")])
        invalid = copy.deepcopy(batch); invalid["dependencies"][0]["dependency_id"] = "D1"
        self.assertEqual(validate_planner_schema(invalid)["decision"],
                         "outline_dependency_schema_invalid")

    def test_planner_schema_enforces_outline_anchor_namespace(self):
        invalid = self.batch(); invalid["dependencies"][0]["parent_anchor_ids"] = ["D1:6"]
        self.assertEqual(validate_planner_schema(invalid)["decision"],
                         "outline_dependency_schema_invalid")

    def test_planner_schema_allows_all_four_declared_arc_anchors(self):
        payload = self.batch()
        row = copy.deepcopy(self.row())
        row["parent_anchor_ids"] = ["P1", "P3", "P5", "P8"]
        row["planned_child_stage"] = "presentation"
        payload["dependencies"][3] = row
        self.assertEqual(validate_planner_schema(payload)["decision"],
                         "outline_dependency_batch_accept")

    def test_prompt_hides_original_turn_ids_and_flat_graph(self):
        prompt = build_planner_prompt(self.source["outline"], "contract")
        self.assertIn("FROZEN EXPANSION OUTLINE", prompt)
        self.assertIn("E1", prompt)
        self.assertNotIn("original_turn_id", prompt)
        self.assertNotIn('"turn_id"', prompt)
        self.assertNotIn("stage2_rebuild:", prompt)

    def test_local_normalization_resolves_frozen_parent_text(self):
        normalized = normalize_outline_dependencies(self.batch(), self.source["outline"],
                                                     self.source["parent_dialogue"])
        row = normalized["dependencies"][0]
        self.assertEqual(row["local_validation"], "accept")
        self.assertEqual(row["resolved_parent_turns"][0]["outline_anchor_id"], "P1")
        self.assertEqual(row["resolved_parent_turns"][0]["alias"], "P006")

    def test_parent_anchor_outside_arc_rejected(self):
        invalid = self.batch(); invalid["dependencies"][0]["parent_anchor_ids"] = ["P4"]
        row = normalize_outline_dependencies(invalid, self.source["outline"],
                                             self.source["parent_dialogue"])["dependencies"][0]
        self.assertIn("parent_anchor_outside_arc", row["local_validation_errors"])

    def test_child_graph_exactly_preserves_planned_event(self):
        dep = self.dependency(); graph = self.child_graph()
        self.assertEqual(validate_child_graph(graph, dep)["decision"],
                         "outline_child_graph_accept")
        graph["child_events"][0]["subject"] = "Maria"
        self.assertIn("planned_child_event_mismatch:subject",
                      validate_child_graph(graph, dep)["reasons"])

    def test_realizer_and_anti_leak(self):
        dep = self.dependency(); graph = self.child_graph(); raw = self.realizer()
        self.assertEqual(validate_realizer(raw, dep, graph)["decision"],
                         "outline_realizer_accept")
        gate = apply_anti_leak_gate(raw, dep, graph)
        self.assertEqual(gate["decision"], "outline_realization_local_gate_pass")
        self.assertEqual([row["alias"] for row in gate["planned_event_evidence_turns"]],
                         ["C001", "C002"])
        raw["dialogue_turns"][2]["text"] = "You were hoping to get into local politics."
        self.assertIn("parent_fact_restatement",
                      apply_anti_leak_gate(raw, dep, graph)["reasons"])

    def test_qa_writer_cannot_select_or_change_evidence(self):
        self.assertNotIn("fixed_parent_turn_aliases", QA_SCHEMA["properties"])
        self.assertNotIn("fixed_child_turn_aliases", QA_SCHEMA["properties"])
        dep = self.dependency()
        qa = {"dependency_id": "D1", "question": "What did the opening advance?",
            "canonical_answer": dep["joint_result"]["value"], "answer_type": "state",
            "relation_preserved": True}
        self.assertEqual(validate_qa(qa, dep)["decision"], "outline_qa_accept")
        qa["canonical_answer"] = "A different answer"
        self.assertIn("qa_joint_result_drift", validate_qa(qa, dep)["reasons"])


if __name__ == "__main__":
    unittest.main()
