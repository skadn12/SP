from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path

from scripts.current_method import stage3_parent_continuity as continuity


class Stage3ParentContinuityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parents = [
            {
                "event_id": "P-plan",
                "entity": "Joanna",
                "slot": "studio_flooring_plan",
                "value": "replace the dance studio floor",
                "status": "active",
                "global_time_index": 142,
                "memory_text": "Joanna plans to replace the dance studio floor.",
            },
            {
                "event_id": "P-room",
                "entity": "Nate",
                "slot": "gaming_room_lighting",
                "value": "red and purple lighting",
                "status": "active",
                "global_time_index": 155,
            },
        ]

    def test_explicit_parent_reference_and_topic_pass(self) -> None:
        child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-plan"],
                    "parent_topic_anchor_event_ids": ["P-plan"],
                    "continuity_relations": [
                        {
                            "parent_event_id": "P-plan",
                            "child_event_id": "E1",
                            "relation": "updates",
                        }
                    ],
                }
            },
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Joanna",
                    "slot": "studio_flooring_plan",
                    "value": "schedule the dance studio floor installation",
                    "status": "planned",
                    "global_time_index": 156,
                }
            ],
        }
        audit = continuity.parent_child_continuity_audit(
            child,
            self.parents,
            selected_parent_event_ids=["P-plan"],
        )
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["depends_on_parent_event_ids"], ["P-plan"])
        self.assertIn("same_entity_slot_extension", audit["continuity_reasons"])
        self.assertIn("shared_parent_resource_or_topic", audit["continuity_reasons"])

    def test_entity_only_overlap_is_rejected(self) -> None:
        child = {
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Joanna",
                    "slot": "department_access",
                    "value": "finance department",
                }
            ]
        }
        audit = continuity.parent_child_continuity_audit(child, self.parents)
        self.assertFalse(audit["pass"])
        self.assertIn(
            "parent_continuity_reference_missing",
            {issue["code"] for issue in audit["issues"]},
        )

    def test_reference_must_come_from_retrieved_view(self) -> None:
        child = {
            "metadata": {"depends_on_parent_event_ids": ["P-room"]},
            "events": [{"event_id": "E1", "entity": "Nate", "slot": "room", "value": "gaming room"}],
        }
        audit = continuity.parent_child_continuity_audit(
            child,
            self.parents,
            selected_parent_event_ids=["P-plan"],
        )
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["unselected_parent_event_ids"], ["P-room"])

    def test_declared_reference_cannot_manufacture_semantic_continuity(self) -> None:
        child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-plan"],
                    "parent_topic_anchor_event_ids": ["P-plan"],
                    "continuity_relations": [
                        {
                            "parent_event_id": "P-plan",
                            "child_event_id": "E1",
                            "relation": "continues",
                        }
                    ],
                }
            },
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Archive Desk",
                    "slot": "department_access",
                    "value": "finance department",
                    "global_time_index": 156,
                }
            ],
        }
        audit = continuity.parent_child_continuity_audit(child, self.parents)
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["continuity_score"], 0.0)
        codes = {issue["code"] for issue in audit["issues"]}
        self.assertIn(
            "parent_continuity_reference_semantically_unsupported", codes
        )
        self.assertIn(
            "parent_continuity_relation_semantically_unsupported", codes
        )

    def test_entity_overlap_alone_fails_even_with_valid_declared_id(self) -> None:
        child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-plan"],
                    "parent_topic_anchor_event_ids": ["P-plan"],
                }
            },
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Joanna",
                    "slot": "department_access",
                    "value": "finance department",
                    "global_time_index": 156,
                }
            ],
        }
        audit = continuity.parent_child_continuity_audit(child, self.parents)
        self.assertFalse(audit["pass"])
        codes = {issue["code"] for issue in audit["issues"]}
        self.assertIn("parent_continuity_entity_only", codes)
        self.assertEqual(audit["verified_parent_event_ids"], [])

    def test_metadata_reference_mutation_to_unrelated_known_parent_fails(self) -> None:
        child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-room"],
                    "parent_topic_anchor_event_ids": ["P-room"],
                    "continuity_relations": [
                        {
                            "parent_event_id": "P-room",
                            "child_event_id": "E1",
                            "relation": "continues",
                        }
                    ],
                }
            },
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Joanna",
                    "slot": "studio_flooring_plan",
                    "value": "schedule the dance studio floor installation",
                    "global_time_index": 156,
                }
            ],
        }
        audit = continuity.parent_child_continuity_audit(child, self.parents)
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["unsupported_declared_parent_event_ids"], ["P-room"])
        self.assertEqual(audit["depends_on_parent_event_ids"], [])

    def test_relation_child_id_and_relation_semantics_are_independently_checked(self) -> None:
        base = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-plan"],
                    "parent_topic_anchor_event_ids": ["P-plan"],
                    "continuity_relations": [
                        {
                            "parent_event_id": "P-plan",
                            "child_event_id": "E404",
                            "relation": "updates",
                        }
                    ],
                }
            },
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Joanna",
                    "slot": "studio_flooring_plan",
                    "value": "schedule the dance studio floor installation",
                    "global_time_index": 156,
                }
            ],
        }
        wrong_child = continuity.parent_child_continuity_audit(base, self.parents)
        self.assertFalse(wrong_child["pass"])
        self.assertIn(
            "parent_continuity_relation_semantically_unsupported",
            {issue["code"] for issue in wrong_child["issues"]},
        )

        base["metadata"]["parent_continuity"]["continuity_relations"][0][
            "child_event_id"
        ] = "E1"
        base["metadata"]["parent_continuity"]["continuity_relations"][0][
            "relation"
        ] = "causes"
        false_cause = continuity.parent_child_continuity_audit(base, self.parents)
        self.assertFalse(false_cause["pass"])
        self.assertEqual(false_cause["explicit_relation_markers"], [])

    def test_slot_family_and_causal_links_are_derived_from_child_events(self) -> None:
        location_parent = {
            "event_id": "P-location",
            "entity": "Joanna",
            "slot": "travel_destination",
            "value": "Cedar Hall",
            "status": "planned",
            "global_time_index": 140,
        }
        location_child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-location"],
                    "parent_topic_anchor_event_ids": ["P-location"],
                    "continuity_relations": [
                        {
                            "parent_event_id": "P-location",
                            "child_event_id": "E-location",
                            "relation": "updates",
                        }
                    ],
                }
            },
            "events": [
                {
                    "event_id": "E-location",
                    "entity": "Joanna",
                    "slot": "location",
                    "value": "Riverside Studio",
                    "status": "active",
                    "global_time_index": 156,
                }
            ],
        }
        family_audit = continuity.parent_child_continuity_audit(
            location_child, [location_parent]
        )
        self.assertTrue(family_audit["pass"])
        link = family_audit["independently_verified_links"][0]
        self.assertIn("location", link["shared_slot_families"])
        self.assertIn("same_entity_slot_family", link["reasons"])

        causal_child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-plan"],
                    "continuity_relations": [
                        {
                            "parent_event_id": "P-plan",
                            "child_event_id": "E-effect",
                            "relation": "caused_by",
                        }
                    ],
                }
            },
            "events": [
                {
                    "event_id": "E-effect",
                    "entity": "Installation Crew",
                    "slot": "work_delay",
                    "value": "the reopening moved to Friday",
                    "caused_by_event_id": "P-plan",
                    "global_time_index": 156,
                }
            ],
        }
        causal_audit = continuity.parent_child_continuity_audit(
            causal_child, self.parents
        )
        self.assertTrue(causal_audit["pass"])
        self.assertIn("verified_causal_relation", causal_audit["continuity_reasons"])

    def test_shared_access_class_or_action_is_not_a_shared_resource(self) -> None:
        visibility_parent = {
            "event_id": "P-visible",
            "event_type": "visibility_permission",
            "owner": "Nate",
            "observer": "Joanna",
            "resource": "Nate's gaming room",
            "action": "view",
            "access_class": "Private",
            "status": "active",
            "global_time_index": 140,
        }
        child = {
            "metadata": {
                "parent_continuity": {
                    "depends_on_parent_event_ids": ["P-visible"],
                    "parent_topic_anchor_event_ids": ["P-visible"],
                }
            },
            "events": [
                {
                    "event_id": "E-visible",
                    "event_type": "department_access",
                    "owner": "Ava",
                    "observer": "Maya",
                    "resource": "finance department",
                    "action": "view",
                    "access_class": "Private",
                    "status": "active",
                    "global_time_index": 156,
                }
            ],
        }
        audit = continuity.parent_child_continuity_audit(
            child, [visibility_parent]
        )
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["verified_parent_event_ids"], [])
        self.assertEqual(audit["continuity_score"], 0.0)

    def test_real_v4_temporal_materialization_trace_is_independently_verified(self) -> None:
        from scripts.current_method import native_api_graph_dialogue_expansion as native
        from scripts.current_method import stage3_truth_constructors as truth

        parent_file = (
            Path(__file__).resolve().parents[1]
            / "stage3-data-fix"
            / "stage3_parent_v4_memory_backfilled.jsonl"
        )
        parent = json.loads(parent_file.read_text(encoding="utf-8").splitlines()[1])
        target = {
            "qa_operation": "temporal_state_update",
            "preferred_answer_type": "LocationOrScope",
            "reasoning_subtype": "status_gated_canceled_update",
            "reasoning_template": "reject_canceled_value_then_resolve_update",
            "hop_count": 4,
            "hard_negative_strategy": "same_slot_canceled_value",
            "forbid_max_time_shortcut": True,
            "target_basis": "continuity_integration_test",
        }
        prepared = native.prepare_truth_first_graph_contract(
            parent,
            {"stage64_target": target},
            random.Random(17),
            sample_index=0,
            round_id="continuity_real_v4_test",
        )
        payload = truth.example_content_bindings(prepared["skeleton"])
        anchor_terms = prepared["skeleton"]["parent_anchor_contract"]["topic_terms"]
        payload["discourse_frame"] = {
            "topic": f"a follow-up about {anchor_terms[0]}",
            "setting": "the participants' next conversation",
        }
        graph = truth.materialize_truth_skeleton(
            prepared["skeleton"], payload
        )["graph"]
        audit = continuity.parent_child_continuity_audit(
            graph,
            prepared["selected_parent_events"],
            selected_parent_event_ids=prepared["parent_event_retrieval"][
                "selected_parent_event_ids"
            ],
        )
        self.assertTrue(audit["pass"], audit["issues"])
        trace = graph["metadata"]["parent_continuity"][
            "continuity_relations"
        ][0]
        self.assertEqual(trace["relation"], "updates")
        self.assertEqual(
            audit["verified_parent_event_ids"], [trace["parent_event_id"]]
        )
        verified_pairs = {
            (link["parent_event_id"], link["child_event_id"])
            for link in audit["independently_verified_links"]
        }
        self.assertIn(
            (trace["parent_event_id"], trace["child_event_id"]),
            verified_pairs,
        )

        false_trace = copy.deepcopy(graph)
        false_trace["metadata"]["parent_continuity"][
            "continuity_relations"
        ][0]["child_event_id"] = "E8"
        mutated = continuity.parent_child_continuity_audit(
            false_trace,
            prepared["selected_parent_events"],
            selected_parent_event_ids=prepared["parent_event_retrieval"][
                "selected_parent_event_ids"
            ],
        )
        self.assertFalse(mutated["pass"])
        self.assertIn(
            "parent_continuity_relation_semantically_unsupported",
            {issue["code"] for issue in mutated["issues"]},
        )

    def test_child_global_time_starts_after_parent_and_keeps_local_time(self) -> None:
        plan = {
            "events": [
                {"event_id": "E1", "time_index": 1},
                {"event_id": "E2", "time_index": 1},
                {"event_id": "E3", "time_index": 3},
            ]
        }
        materialized = continuity.materialize_child_global_time(plan, self.parents)
        events = materialized["events"]
        self.assertEqual([event["local_time_index"] for event in events], [1, 1, 3])
        self.assertEqual([event["time_index"] for event in events], [1, 1, 3])
        self.assertEqual([event["global_time_index"] for event in events], [156, 156, 157])
        self.assertTrue(
            materialized["metadata"]["child_global_time_strictly_after_parent"]
        )
        self.assertNotIn("global_time_index", plan["events"][0])


if __name__ == "__main__":
    unittest.main()
