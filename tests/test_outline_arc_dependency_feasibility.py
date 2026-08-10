import copy, unittest

from scripts.current_method.restart_n5_spell.outline_arc_dependency_feasibility import build_audit
from scripts.current_method.restart_n5_spell.single_arc_typed_dependency import (build_child_expander_prompt,
    build_realizer_prompt, materialize, validate, validate_child_graph_with_semantic_split)


class OutlineFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.anchors=[{"anchor_id":f"P{i}","turn_id":f"D:{i}","speaker":"S","text":f"t{i}","continuity_role":"r"} for i in range(1,9)]
        self.outline={"planned_event_arcs":[{"event_id":f"E{i}"} for i in range(1,5)],"parent_anchors":copy.deepcopy(self.anchors)}
        self.snapshot={"anchors":copy.deepcopy(self.anchors)}
        self.turns=[{"turn_id":f"D:{i}","speaker":"S","text":f"t{i}","alias":f"P{i:03d}","session_id":"D"} for i in range(1,238)]

    def test_audit_is_zero_api_and_has_three_feasible_arcs(self):
        result=build_audit(self.outline,self.snapshot,self.turns)
        self.assertEqual(result["feasible_arc_count"],3)
        self.assertTrue(result["zero_api"])
        self.assertFalse(result["semantic_graph_used"])

    def test_parent_anchor_text_must_be_verbalized(self):
        self.turns[0]["text"]="drift"
        with self.assertRaisesRegex(ValueError,"parent_anchor_not_verbalized"):
            build_audit(self.outline,self.snapshot,self.turns)

    def test_typed_schema_no_valid_is_semantic_accept(self):
        candidate=build_audit(self.outline,self.snapshot,self.turns)["items"][0]
        self.assertEqual(validate({"status":"no_valid_dependency"},candidate)["decision"],"typed_dependency_no_valid_accept")

    def test_typed_key_mismatch_rejected(self):
        candidate=build_audit(self.outline,self.snapshot,self.turns)["items"][0]
        raw={"status":"valid","parent_anchor_ids":["P5"],"operator":"qualification_requirement","planned_child_event":{"requirement_key":"wrong","required_value":"completed_degree","time_relation":"later"}}
        self.assertEqual(validate(raw,candidate)["decision"],"typed_dependency_schema_invalid")

    def test_program_computes_joint_result(self):
        candidate=build_audit(self.outline,self.snapshot,self.turns)["items"][0]
        raw={"status":"valid","parent_anchor_ids":["P5"],"operator":"qualification_requirement","planned_child_event":{"requirement_key":"completed_degree","required_value":"completed_degree","time_relation":"later"}}
        dep=materialize(raw,candidate,self.anchors[4],self.turns[4])
        self.assertEqual(dep["joint_result"]["value"],"meets_requirement")
        self.assertTrue(all(dep["typed_gate"].values()))

    def test_downstream_prompt_boundaries(self):
        candidate=build_audit(self.outline,self.snapshot,self.turns)["items"][0]
        raw={"status":"valid","parent_anchor_ids":["P5"],"operator":"qualification_requirement","planned_child_event":{"requirement_key":"completed_degree","required_value":"completed_degree","time_relation":"later"}}
        dep=materialize(raw,candidate,self.anchors[4],self.turns[4])
        arc={"event_id":"E1","summary":"s","stages":["eligibility check"]}
        child=build_child_expander_prompt(arc,dep,"contract")
        self.assertNotIn("parent_condition",child)
        self.assertNotIn("joint_result",child)
        graph={"dependency_id":"D1","arc_id":"E1","stage":"eligibility check","child_graph_id":"g","child_events":[],"realizes_planned_child_event_id":"e"}
        real=build_realizer_prompt(graph,"contract")
        self.assertNotIn(self.anchors[4]["text"],real)
        self.assertNotIn("meets_requirement",real)

    def test_child_value_drift_is_not_format_retryable(self):
        candidate=build_audit(self.outline,self.snapshot,self.turns)["items"][1]
        raw={"status":"valid","parent_anchor_ids":["P8"],"operator":"commitment_outcome","planned_child_event":{"action_key_ref":"maria_brainstorms_community_ideas_with_john","outcome_state":"completed","time_relation":"later"}}
        dep=materialize(raw,candidate,self.anchors[7],self.turns[7])
        graph={"dependency_id":"D2","arc_id":"E2","stage":"collection plan","child_graph_id":"g","child_events":[{"event_id":"e","subject":"Maria","predicate":"brainstorms","object":"community ideas with John","value":"completed","time":"later","status":"completed"}],"realizes_planned_child_event_id":"e"}
        self.assertEqual(validate_child_graph_with_semantic_split(graph,dep)["decision"],"outline_child_graph_semantic_mismatch")

if __name__ == "__main__": unittest.main()
