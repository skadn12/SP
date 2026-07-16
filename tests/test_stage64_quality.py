from __future__ import annotations

import copy
import unittest

from scripts.current_method import stage64_quality


def valid_sample() -> dict:
    events = [
        {
            "event_instance_id": f"event-{index}",
            "entity": entity,
            "slot": slot,
            "value": value,
            "status": status,
        }
        for index, (entity, slot, value, status) in enumerate(
            (
                ("Maya", "item_beside_lamp", "Red notebook", "active"),
                ("Maya", "item_beside_lamp", "Blue binder", "outdated"),
                ("Noah", "item_beside_lamp", "Ceramic mug", "active"),
                ("Maya", "desk_drawer_item", "Brass key", "active"),
            ),
            start=1,
        )
    ]
    return {
        "graph_update": {"events": events},
        "qa": {
            "question_type": "single_choice",
            "question": "What item did Maya leave beside the lamp?",
            "qa_operation": "relation_lookup",
            "qa_subtype": "object_location_lookup",
            "answer_type": "ObjectOrCommonNoun",
            "gold": ["A"],
            "options": [
                {"id": label, "text": text, "source_event_instance_id": f"event-{index}"}
                for index, (label, text) in enumerate(
                    zip("ABCD", ("Red notebook", "Blue binder", "Ceramic mug", "Brass key")),
                    start=1,
                )
            ],
            "query_plan": {
                "reasoning_anchor_path": ["event-1", "event-2", "event-3"]
            },
        },
    }


def valid_comparison_sample() -> dict:
    events = [
        {
            "event_id": "e1",
            "event_instance_id": "e1",
            "entity": "Maya",
            "slot": "review_points",
            "value": 14,
            "status": "active",
            "time_index": 7,
        },
        {
            "event_id": "e2",
            "event_instance_id": "e2",
            "entity": "Noah",
            "slot": "review_points",
            "value": 9,
            "status": "active",
            "time_index": 7,
        },
        {
            "event_id": "e3",
            "event_instance_id": "e3",
            "entity": "Maya",
            "slot": "review_points",
            "value": 12,
            "status": "outdated",
            "time_index": 6,
        },
        {
            "event_id": "e4",
            "event_instance_id": "e4",
            "entity": "Noah",
            "slot": "review_note",
            "value": "complete",
            "status": "active",
            "time_index": 7,
        },
    ]
    pair_ids = ["e1", "e2"]
    return {
        "graph_update": {"graph_snapshot_id": "stage3:comparison", "events": events},
        "qa": {
            "question": "What is the current difference between Maya and Noah in review points?",
            "qa_operation": "comparison",
            "qa_subtype": "numeric_difference",
            "answer_type": "Number",
            "gold": ["C"],
            "gold_answer_text": "5",
            "options": [
                {"id": "A", "text": "3", "source_event_instance_id": "e1"},
                {"id": "B", "text": "4", "source_event_instance_id": "e2"},
                {"id": "C", "text": "5", "source_event_instance_id": "e1"},
                {"id": "D", "text": "6", "source_event_instance_id": "e2"},
            ],
            "query_plan": {
                "reasoning_anchor_path": pair_ids,
                "resolved_reasoning_event_instance_ids": pair_ids,
                "resolved_required_event_instance_ids": pair_ids,
                "resolved_supporting_event_instance_ids": pair_ids,
                "resolved_gold_event_instance_ids": pair_ids,
            },
        },
    }


def valid_bridge_sample() -> dict:
    events = [
        {
            "event_id": "e1",
            "event_instance_id": "e1",
            "entity": "Maya",
            "slot": "route_constraint",
            "value": "north corridor",
            "status": "active",
            "time_index": 1,
            "depends_on": [],
            "turn_ids": ["t1"],
        },
        {
            "event_id": "e2",
            "event_instance_id": "e2",
            "entity": "Maya",
            "slot": "permit_constraint",
            "value": "harbor permit",
            "status": "active",
            "time_index": 2,
            "depends_on": [],
            "turn_ids": ["t2"],
        },
        {
            "event_id": "e3",
            "event_instance_id": "e3",
            "entity": "Maya",
            "slot": "approved_route",
            "value": "Harbor route",
            "status": "active",
            "time_index": 3,
            "depends_on": ["e1", "e2"],
            "turn_ids": ["t3"],
        },
        {
            "event_id": "e4",
            "event_instance_id": "e4",
            "entity": "Maya",
            "slot": "approved_route",
            "value": "Ridge route",
            "status": "outdated",
            "time_index": 4,
            "depends_on": [],
            "turn_ids": ["t4"],
        },
    ]
    return {
        "stage64_target": {
            "qa_operation": "bridge",
            "preferred_answer_type": "LocationOrScope",
        },
        "graph_update": {"graph_snapshot_id": "stage3:bridge", "events": events},
        "dialogue": {
            "turns": [
                {"turn_id": "t1", "text": "Maya must use the north corridor."},
                {"turn_id": "t2", "text": "Her permit is valid only at the harbor."},
                {"turn_id": "t3", "text": "The routing record was updated after both checks."},
                {"turn_id": "t4", "text": "The ridge alternative was withdrawn."},
            ]
        },
        "qa": {
            "question": "Which route satisfies both Maya's corridor and permit constraints?",
            "qa_operation": "bridge",
            "qa_subtype": "multi_hop_dependency",
            "answer_type": "LocationOrScope",
            "gold": ["A"],
            "gold_answer_text": "Harbor route",
            "options": [
                {"id": "A", "text": "Harbor route", "source_event_instance_id": "e3"},
                {"id": "B", "text": "North corridor", "source_event_instance_id": "e1"},
                {"id": "C", "text": "Harbor permit", "source_event_instance_id": "e2"},
                {"id": "D", "text": "Ridge route", "source_event_instance_id": "e4"},
            ],
            "query_plan": {
                "reasoning_anchor_path": ["e1", "e2", "e3"],
                "resolved_reasoning_event_instance_ids": ["e1", "e2", "e3"],
            },
        },
    }


def valid_cause_sample(*, direction: str = "cause") -> dict:
    events = [
        {
            "event_id": "e1",
            "event_instance_id": "e1",
            "entity": "Atlas launch",
            "slot": "cause",
            "value": "permit denial",
            "status": "active",
            "time_index": 1,
            "depends_on": [],
        },
        {
            "event_id": "e2",
            "event_instance_id": "e2",
            "entity": "Atlas launch",
            "slot": "effect",
            "value": "launch delay",
            "status": "active",
            "time_index": 2,
            "depends_on": ["e1"],
        },
        {
            "event_id": "e3",
            "event_instance_id": "e3",
            "entity": "Atlas launch",
            "slot": "cause",
            "value": "weather review",
            "status": "report",
            "time_index": 2,
            "depends_on": [],
        },
        {
            "event_id": "e4",
            "event_instance_id": "e4",
            "entity": "Atlas launch",
            "slot": "effect",
            "value": "staff meeting",
            "status": "outdated",
            "time_index": 3,
            "depends_on": [],
        },
    ]
    asks_cause = direction == "cause"
    return {
        "stage64_target": {
            "qa_operation": "cause_consequence",
            "preferred_answer_type": "Event",
        },
        "graph_update": {"graph_snapshot_id": "stage3:cause", "events": events},
        "qa": {
            "question": (
                "What caused the Atlas launch to be delayed?"
                if asks_cause
                else "What result followed from the Atlas permit denial?"
            ),
            "qa_operation": "cause_consequence",
            "qa_subtype": "causal_effect",
            "answer_type": "Event",
            "gold": ["A"],
            "options": [
                {
                    "id": "A",
                    "text": "Permit denial" if asks_cause else "Launch delay",
                    "source_event_instance_id": "e1" if asks_cause else "e2",
                },
                {"id": "B", "text": "Weather review", "source_event_instance_id": "e3"},
                {"id": "C", "text": "Staff meeting", "source_event_instance_id": "e4"},
                {
                    "id": "D",
                    "text": "Launch delay" if asks_cause else "Permit denial",
                    "source_event_instance_id": "e2" if asks_cause else "e1",
                },
            ],
            "query_plan": {
                "reasoning_anchor_path": ["e1", "e2"],
                "resolved_reasoning_event_instance_ids": ["e1", "e2"],
            },
        },
    }


def valid_visibility_sample() -> dict:
    rows = (
        ("e1", "Caroline", "active", "Analyst"),
        ("e2", "Melanie", "scope_limited", "Reviewer"),
        ("e3", "Reviewer", "expired", "Auditor"),
        ("e4", "Analyst", "canceled", "Manager"),
    )
    events = [
        {
            "event_id": event_id,
            "event_instance_id": event_id,
            "entity": entity,
            "slot": "access_scope",
            "value": "Internal Report",
            "status": status,
            "time_index": index,
            "scope": {"observer": observer, "permission": "read"},
        }
        for index, (event_id, entity, status, observer) in enumerate(rows, start=1)
    ]
    return {
        "stage64_target": {
            "qa_operation": "visibility",
            "preferred_answer_type": "Person",
        },
        "graph_update": {"graph_snapshot_id": "stage3:visibility", "events": events},
        "qa": {
            "question": "Who has valid read access to the Internal Report?",
            "qa_operation": "visibility",
            "qa_subtype": "permission_scope",
            "answer_type": "Person",
            "gold": ["A"],
            "options": [
                {"id": label, "text": event["entity"], "source_event_instance_id": event["event_id"]}
                for label, event in zip("ABCD", events)
            ],
            "query_plan": {
                "reasoning_anchor_path": ["e1", "e2"],
                "resolved_reasoning_event_instance_ids": ["e1", "e2"],
            },
        },
    }


def valid_temporal_reasoning_sample() -> dict:
    values = ("Paris", "Lyon", "Berlin", "Rome")
    statuses = ("outdated", "active", "expired", "canceled")
    events = []
    for index, (value, status) in enumerate(zip(values, statuses), start=1):
        events.append(
            {
                "event_id": f"e{index}",
                "event_instance_id": f"e{index}",
                "entity": "Maya",
                "slot": "meeting_city",
                "value": value,
                "status": status,
                "time_index": index,
                "previous_value": values[index - 2] if index > 1 else None,
                "depends_on": [f"e{index - 1}"] if index > 1 else [],
                "turn_ids": [f"t{index}"],
            }
        )
    return {
        "stage64_target": {
            "qa_operation": "temporal_state_update",
            "preferred_answer_type": "LocationOrScope",
            "hop_count": 4,
        },
        "graph_update": {"graph_snapshot_id": "stage3:temporal", "events": events},
        "dialogue": {
            "turns": [
                {"turn_id": "t1", "text": "An early note proposed Paris."},
                {"turn_id": "t2", "text": "A planning note placed Maya's meeting in Lyon."},
                {"turn_id": "t3", "text": "The Berlin proposal later expired."},
                {"turn_id": "t4", "text": "The Rome proposal was canceled."},
            ]
        },
        "qa": {
            "question": "After accounting for the canceled proposal, where is Maya's meeting?",
            "qa_operation": "temporal_state_update",
            "qa_subtype": "latest_state_update",
            "answer_type": "LocationOrScope",
            "hop_count": 4,
            "forbid_max_time_shortcut": True,
            "hard_negative_strategy": "same_slot_canceled_value",
            "gold": ["B"],
            "gold_answer_text": "Lyon",
            "options": [
                {"id": label, "text": value, "source_event_instance_id": f"e{index}"}
                for index, (label, value) in enumerate(zip("ABCD", values), start=1)
            ],
            "query_plan": {
                "reasoning_anchor_path": ["e1", "e2", "e3", "e4"],
                "resolved_reasoning_event_instance_ids": ["e1", "e2", "e3", "e4"],
                "resolved_distractor_event_instance_ids": ["e1", "e3", "e4"],
            },
        },
    }


class Stage64QualityTest(unittest.TestCase):
    @staticmethod
    def _compiled_operation_sample(operation: str, events: list[dict]) -> dict:
        return {
            "graph_update": {"graph_snapshot_id": "stage3:compiled", "events": events},
            "qa": {
                "question": "What result follows from the earlier cause?",
                "qa_operation": operation,
                "qa_subtype": {
                    "bridge": "multi_hop_dependency",
                    "visibility": "permission_scope",
                    "cause_consequence": "causal_effect",
                }[operation],
                "answer_type": "Event",
                "gold": ["A"],
                "options": [],
                "query_plan": {"reasoning_anchor_path": ["e1", "e2"]},
            },
        }

    def test_compiled_graph_gate_rechecks_bridge_visibility_and_cause_structure(self) -> None:
        common = [
            {"event_id": f"e{i}", "event_instance_id": f"e{i}", "entity": "Atlas", "slot": "step", "value": str(i), "status": "active", "time_index": i}
            for i in range(1, 5)
        ]
        cases = {
            "bridge": "bridge_missing_two_hop_dependency",
            "visibility": "visibility_missing_scope",
            "cause_consequence": "cause_missing_ordered_dependency",
        }
        for operation, expected_reason in cases.items():
            with self.subTest(operation=operation):
                row = self._compiled_operation_sample(operation, copy.deepcopy(common))
                reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}
                self.assertIn(expected_reason, reasons)

    def test_compiled_graph_gate_accepts_preserved_bridge_visibility_and_cause_structure(self) -> None:
        base_events = [
            {"event_id": f"e{i}", "event_instance_id": f"e{i}", "entity": "Atlas", "slot": "step", "value": str(i), "status": "active", "time_index": i}
            for i in range(1, 5)
        ]
        valid_by_operation = {
            "bridge": [{**event, "depends_on": (["e1", "e2"] if event["event_id"] == "e3" else [])} for event in base_events],
            "visibility": [{**event, **({"scope": {"observer": "Maya"}} if event["event_id"] == "e1" else {})} for event in base_events],
            "cause_consequence": [{**event, "depends_on": (["e1"] if event["event_id"] == "e2" else [])} for event in base_events],
        }
        for operation, events in valid_by_operation.items():
            with self.subTest(operation=operation):
                row = self._compiled_operation_sample(operation, events)
                reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}
                expected_reason = {
                    "bridge": "bridge_missing_two_hop_dependency",
                    "visibility": "visibility_missing_scope",
                    "cause_consequence": "cause_missing_ordered_dependency",
                }[operation]
                self.assertNotIn(expected_reason, reasons)

    def test_valid_count_is_derived_from_resolved_compiled_members(self) -> None:
        events = [
            {
                "event_id": f"e{index}",
                "event_instance_id": f"e{index}",
                "entity": "Atlas project",
                "slot": "active_member",
                "value": name,
                "status": "active",
                "time_index": index,
                "set_membership": {
                    "set_id": "atlas_active_members",
                    "included": index <= 3,
                    "condition": "completed orientation",
                },
            }
            for index, name in enumerate(("Maya", "Noah", "Li", "Omar", "Zoe", "Iris"), start=1)
        ]
        row = {
            "graph_update": {"graph_snapshot_id": "stage3:count", "events": events},
            "qa": {
                "question": "How many Atlas project atlas active members completed orientation?",
                "qa_operation": "count",
                "qa_subtype": "filtered_count",
                "answer_type": "Number",
                "gold": ["C"],
                "gold_answer_text": "3",
                "options": [
                    {"id": label, "text": text, "source_event_instance_id": f"e{index}"}
                    for index, (label, text) in enumerate(zip("ABCD", ("1", "2", "3", "4")), start=1)
                ],
                "query_plan": {
                    "filter_predicate": "completed orientation",
                    "reasoning_anchor_path": [f"marker-{index}" for index in range(1, 7)],
                    "resolved_reasoning_event_instance_ids": [f"e{index}" for index in range(1, 7)],
                    "resolved_required_event_instance_ids": [f"e{index}" for index in range(1, 7)],
                    "resolved_supporting_event_instance_ids": [f"e{index}" for index in range(1, 7)],
                    "resolved_gold_event_instance_ids": ["e1", "e2", "e3"],
                    "resolved_distractor_event_instance_ids": ["e4", "e5", "e6"],
                },
            },
        }

        self.assertEqual(stage64_quality.semantic_quality_issues(row), [])

        wrong_gold = copy.deepcopy(row)
        wrong_gold["qa"]["gold"] = ["B"]
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(wrong_gold)}
        self.assertIn("count_gold_not_exact_included_count", reasons)

        wrong_declared_gold = copy.deepcopy(row)
        wrong_declared_gold["qa"]["gold_answer_text"] = "2"
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(wrong_declared_gold)}
        self.assertIn("count_gold_answer_text_not_exact_included_count", reasons)

        unbracketed = copy.deepcopy(row)
        for option, text in zip(unbracketed["qa"]["options"], ("3", "4", "5", "6")):
            option["text"] = text
        unbracketed["qa"]["gold"] = ["A"]
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(unbracketed)}
        self.assertIn("count_options_not_nearby_bracketing_values", reasons)

        outside_ref = copy.deepcopy(row)
        outside_ref["qa"]["query_plan"]["resolved_distractor_event_instance_ids"] = ["outside"]
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(outside_ref)}
        self.assertIn("count_reference_group_outside_candidate_set", reasons)

        unnamed_dimension = copy.deepcopy(row)
        unnamed_dimension["qa"]["question"] = "How many are included?"
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(unnamed_dimension)}
        self.assertIn("count_question_missing_candidate_set_dimensions", reasons)

        partial_required = copy.deepcopy(row)
        partial_required["qa"]["query_plan"]["resolved_required_event_instance_ids"] = ["e1", "e2", "e3"]
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(partial_required)}
        self.assertIn("count_reference_group_does_not_cover_candidate_set", reasons)

        mixed = copy.deepcopy(row)
        mixed["graph_update"]["events"][5]["slot"] = "former_member"
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(mixed)}
        self.assertIn("count_missing_candidate_set", reasons)
        self.assertIn("count_reasoning_events_not_one_candidate_set", reasons)

        contaminated = copy.deepcopy(row)
        contaminated["graph_update"]["events"].append(
            {
                "event_id": "e7",
                "event_instance_id": "e7",
                "entity": "Beacon project",
                "slot": "former_member",
                "value": "Zoe",
                "status": "outdated",
                "time_index": 7,
                "set_membership": {"set_id": "atlas_active_members", "included": False},
            }
        )
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(contaminated)}
        self.assertIn("count_missing_candidate_set", reasons)
        self.assertIn("count_set_id_has_mixed_semantic_dimensions", reasons)

        missing_condition = copy.deepcopy(row)
        del missing_condition["graph_update"]["events"][0]["set_membership"]["condition"]
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(missing_condition)
        }
        self.assertIn("count_membership_condition_missing", reasons)

        missing_filter = copy.deepcopy(row)
        del missing_filter["qa"]["query_plan"]["filter_predicate"]
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(missing_filter)
        }
        self.assertIn("count_filter_predicate_missing", reasons)

        mismatched_filter = copy.deepcopy(row)
        mismatched_filter["qa"]["query_plan"]["filter_predicate"] = "verified member"
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(mismatched_filter)
        }
        self.assertIn("count_filter_predicate_mismatch", reasons)

        unnamed_condition = copy.deepcopy(row)
        unnamed_condition["qa"]["question"] = (
            "How many Atlas project atlas active members are there?"
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(unnamed_condition)
        }
        self.assertIn("count_question_missing_filter_condition", reasons)

    def test_valid_comparison_is_derived_from_exact_numeric_pair(self) -> None:
        self.assertEqual(stage64_quality.semantic_quality_issues(valid_comparison_sample()), [])

    def test_comparison_rejects_wrong_derived_gold(self) -> None:
        row = valid_comparison_sample()
        row["qa"]["gold"] = ["B"]

        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}

        self.assertIn("comparison_gold_not_derived_difference", reasons)

    def test_comparison_rejects_partial_resolved_reference_group(self) -> None:
        row = valid_comparison_sample()
        row["qa"]["query_plan"]["resolved_supporting_event_instance_ids"] = ["e1"]

        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}

        self.assertIn("comparison_reference_group_not_exact_pair", reasons)

    def test_comparison_rejects_qualitative_string_values(self) -> None:
        row = valid_comparison_sample()
        row["graph_update"]["events"][0]["value"] = "high"
        row["graph_update"]["events"][1]["value"] = "low"

        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}

        self.assertIn("comparison_missing_unique_numeric_pair", reasons)
        self.assertIn("comparison_unique_numeric_pair_missing", reasons)

    def test_comparison_question_must_name_both_entities(self) -> None:
        row = valid_comparison_sample()
        row["qa"]["question"] = "What is the difference in Maya's review points?"

        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}

        self.assertIn("comparison_question_missing_pair_entities", reasons)

    def test_comparison_question_must_name_exact_pair_slot(self) -> None:
        missing_slot = valid_comparison_sample()
        missing_slot["qa"]["question"] = (
            "What is the current absolute difference between Maya and Noah?"
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(missing_slot)
        }
        self.assertIn("comparison_question_missing_pair_slot", reasons)

        wrong_slot = valid_comparison_sample()
        wrong_slot["qa"]["question"] = (
            "What is the current difference between Maya and Noah in review scores?"
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(wrong_slot)
        }
        self.assertIn("comparison_question_missing_pair_slot", reasons)

        correct_slot = valid_comparison_sample()
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(correct_slot)
        }
        self.assertNotIn("comparison_question_missing_pair_slot", reasons)

    def test_comparison_multi_time_slot_requires_natural_time_qualifier(self) -> None:
        row = valid_comparison_sample()
        row["qa"]["question"] = (
            "What is the difference between Maya and Noah in review points?"
        )

        reasons = {
            issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)
        }

        self.assertIn("comparison_question_missing_pair_time_qualifier", reasons)

        row["qa"]["question"] = (
            "What is the difference at time_index 7 between Maya and Noah in review points?"
        )
        reasons = {
            issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)
        }
        self.assertIn("comparison_question_exposes_internal_time_index", reasons)
        self.assertIn("comparison_question_missing_pair_time_qualifier", reasons)

    def test_run112_comparison_slot_surface_remains_valid(self) -> None:
        row = valid_comparison_sample()
        row["graph_update"]["events"][0].update(
            entity="John",
            slot="hours_of_weekly_exercise",
            value=5,
            time_index=1,
        )
        row["graph_update"]["events"][1].update(
            entity="Maria",
            slot="hours_of_weekly_exercise",
            value=8,
            time_index=1,
        )
        row["graph_update"]["events"][2].update(
            entity="John",
            slot="political_ambition_level",
            value=7,
            time_index=2,
        )
        row["qa"]["question"] = (
            "What is the absolute difference between John's and Maria's weekly exercise hours?"
        )
        row["qa"]["gold"] = ["A"]
        row["qa"]["gold_answer_text"] = "3"
        for option, text in zip(row["qa"]["options"], ("3", "2", "4", "5")):
            option["text"] = text

        self.assertEqual(stage64_quality.semantic_quality_issues(row), [])

    def test_declarative_predicate_fragments_are_valid_graph_values(self) -> None:
        plan = copy.deepcopy(valid_sample()["graph_update"])
        values = (
            "is seeking help for bonding with Toby",
            "will try to improve the bond",
            "was moved to Lyon",
            "bond is a work in progress",
        )
        for event, value in zip(plan["events"], values):
            event["value"] = value

        reasons = {
            issue["reason"]
            for issue in stage64_quality.graph_plan_quality_issues(plan, "relation_lookup")
        }
        self.assertNotIn("question_used_as_graph_value", reasons)

    def test_valid_semantic_relation_item_passes(self) -> None:
        self.assertEqual(stage64_quality.semantic_quality_issues(valid_sample()), [])

    def test_bridge_requires_complete_dependency_closure_and_no_direct_copy(self) -> None:
        self.assertEqual(stage64_quality.semantic_quality_issues(valid_bridge_sample()), [])

        partial = valid_bridge_sample()
        partial["qa"]["query_plan"]["reasoning_anchor_path"] = ["e1", "e3"]
        partial["qa"]["query_plan"]["resolved_reasoning_event_instance_ids"] = [
            "e1",
            "e3",
        ]
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(partial)
        }
        self.assertIn("bridge_reasoning_missing_dependency_closure", reasons)

        direct = valid_bridge_sample()
        direct["dialogue"]["turns"][2]["text"] = (
            "The approved route is Harbor route after both checks."
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(direct)
        }
        self.assertIn("bridge_gold_answer_copied_from_single_endpoint", reasons)

    def test_cause_gold_endpoint_follows_question_direction(self) -> None:
        self.assertEqual(stage64_quality.semantic_quality_issues(valid_cause_sample()), [])
        self.assertEqual(
            stage64_quality.semantic_quality_issues(
                valid_cause_sample(direction="effect")
            ),
            [],
        )

        wrong_endpoint = valid_cause_sample()
        wrong_endpoint["qa"]["options"][0].update(
            text="Launch delay",
            source_event_instance_id="e2",
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(wrong_endpoint)
        }
        self.assertIn("cause_gold_endpoint_direction_mismatch", reasons)

        no_edge = valid_cause_sample()
        no_edge["graph_update"]["events"][1]["depends_on"] = []
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(no_edge)
        }
        self.assertIn("cause_reasoning_path_missing_directed_edge", reasons)

    def test_visibility_requires_one_projection_unique_gold_and_fact_scope_refs(self) -> None:
        self.assertEqual(stage64_quality.semantic_quality_issues(valid_visibility_sample()), [])

        mixed_projection = valid_visibility_sample()
        mixed_projection["qa"]["options"][2]["text"] = "Auditor"
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(mixed_projection)
        }
        self.assertIn("visibility_option_projection_inconsistent", reasons)

        nonunique = valid_visibility_sample()
        nonunique["graph_update"]["events"][1]["status"] = "active"
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(nonunique)
        }
        self.assertIn("visibility_gold_not_unique_in_valid_scope", reasons)

        one_reference = valid_visibility_sample()
        one_reference["qa"]["query_plan"]["reasoning_anchor_path"] = ["e1"]
        one_reference["qa"]["query_plan"][
            "resolved_reasoning_event_instance_ids"
        ] = ["e1"]
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(one_reference)
        }
        self.assertIn("visibility_reasoning_missing_fact_scope_refs", reasons)

    def test_relation_requires_dual_constraint_path_and_wrong_entity(self) -> None:
        no_wrong_entity = valid_sample()
        no_wrong_entity["qa"]["query_plan"]["reasoning_anchor_path"] = [
            "event-1",
            "event-2",
        ]
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(no_wrong_entity)
        }
        self.assertIn("relation_missing_wrong_entity_hard_negative", reasons)

        no_same_entity_competitor = valid_sample()
        no_same_entity_competitor["qa"]["query_plan"]["reasoning_anchor_path"] = [
            "event-1",
            "event-3",
        ]
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(
                no_same_entity_competitor
            )
        }
        self.assertIn("relation_reasoning_missing_dual_constraint_path", reasons)

        unnamed_relation = valid_sample()
        unnamed_relation["qa"]["question"] = "What item did Maya leave?"
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(unnamed_relation)
        }
        self.assertIn("relation_question_missing_dual_constraints", reasons)

    def test_temporal_requires_dependency_chain_and_no_direct_gold_status_hint(self) -> None:
        self.assertEqual(
            stage64_quality.semantic_quality_issues(valid_temporal_reasoning_sample()),
            [],
        )

        disconnected = valid_temporal_reasoning_sample()
        disconnected["graph_update"]["events"][2]["depends_on"] = []
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(disconnected)
        }
        self.assertIn("temporal_reasoning_path_not_dependency_chain", reasons)

        direct_hint = valid_temporal_reasoning_sample()
        direct_hint["dialogue"]["turns"][1]["text"] = (
            "Lyon is now the confirmed active meeting city."
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(direct_hint)
        }
        self.assertIn("temporal_gold_evidence_direct_status_hint", reasons)

    def test_non_evidence_filler_cannot_restate_graph_fact(self) -> None:
        row = valid_temporal_reasoning_sample()
        row["dialogue"]["turns"].append(
            {
                "turn_id": "D27:18",
                "text": (
                    "Maya's initial meeting plan mentioned Paris, but the latest "
                    "version clearly states Lyon."
                ),
            }
        )

        issues = stage64_quality.semantic_quality_issues(row)
        issue = next(
            item
            for item in issues
            if item["reason"] == "graph_fact_repeated_outside_evidence_turn"
        )
        self.assertEqual(issue["turn_id"], "D27:18")
        self.assertEqual(issue["entity"], "Maya")
        self.assertEqual(set(issue["matched_event_ids"]), {"e1", "e2"})

    def test_graph_fact_repetition_gate_has_narrow_compatibility_exemptions(self) -> None:
        neutral_filler = valid_temporal_reasoning_sample()
        neutral_filler["dialogue"]["turns"].append(
            {
                "turn_id": "t5",
                "text": "Maya asked the planning team to review the agenda.",
            }
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(neutral_filler)
        }
        self.assertNotIn("graph_fact_repeated_outside_evidence_turn", reasons)

        evidence_turn = valid_temporal_reasoning_sample()
        evidence_turn["dialogue"]["turns"][0]["text"] = (
            "Maya compared the early Paris proposal with Lyon."
        )
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(evidence_turn)
        }
        self.assertNotIn("graph_fact_repeated_outside_evidence_turn", reasons)

        no_evidence_ids = valid_sample()
        no_evidence_ids["dialogue"] = {
            "turns": [
                {
                    "turn_id": "t1",
                    "text": "Maya left the Red notebook beside the lamp.",
                }
            ]
        }
        reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(no_evidence_ids)
        }
        self.assertNotIn("graph_fact_repeated_outside_evidence_turn", reasons)

    def test_stage64_target_preferred_answer_type_is_binding(self) -> None:
        row = valid_sample()
        row["stage64_target"] = {
            "qa_operation": "relation_lookup",
            "preferred_answer_type": "Person",
        }

        reasons = {
            issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)
        }

        self.assertIn("stage64_preferred_answer_type_mismatch", reasons)

    def test_relation_lookup_requires_disambiguating_second_anchor(self) -> None:
        row = valid_sample()
        row["qa"]["query_plan"]["reasoning_anchor_path"] = ["event-1"]

        issues = stage64_quality.semantic_quality_issues(row)

        self.assertIn("insufficient_reasoning_anchors", {issue["reason"] for issue in issues})
        anchor_issue = next(issue for issue in issues if issue["reason"] == "insufficient_reasoning_anchors")
        self.assertEqual(anchor_issue["minimum_reasoning_anchor_count"], 2)

    def test_bridge_yesno_is_rejected_as_shortcut(self) -> None:
        row = valid_sample()
        row["qa"].update(
            {
                "question": "Did Maya's notebook location satisfy both recorded conditions?",
                "qa_operation": "bridge",
                "qa_subtype": "multi_hop_dependency",
                "answer_type": "YesNo",
            }
        )
        row["qa"]["options"] = [
            {"id": "A", "text": "Yes", "source_event_instance_id": "event-1"},
            {"id": "B", "text": "No", "source_event_instance_id": "event-2"},
        ]
        row["qa"]["gold"] = ["A"]

        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}

        self.assertIn("bridge_yesno_shortcut", reasons)

    def test_count_requires_four_member_reasoning_anchors(self) -> None:
        row = valid_sample()
        row["qa"].update(
            {
                "question": "How many notebooks belong in Maya's desk set?",
                "qa_operation": "count",
                "qa_subtype": "filtered_count",
                "answer_type": "Number",
                "query_plan": {"reasoning_anchor_path": ["event-1", "event-2", "event-3"]},
            }
        )
        for index, option in enumerate(row["qa"]["options"], start=1):
            option["text"] = str(index)

        issues = stage64_quality.semantic_quality_issues(row)

        anchor_issue = next(issue for issue in issues if issue["reason"] == "insufficient_reasoning_anchors")
        self.assertEqual(anchor_issue["minimum_reasoning_anchor_count"], 4)

    def test_number_options_cannot_be_questions(self) -> None:
        row = valid_sample()
        row["qa"].update(
            {
                "question": "How many people joined the project?",
                "qa_operation": "count",
                "qa_subtype": "explicit_count_lookup",
                "answer_type": "Number",
                "query_plan": {"reasoning_anchor_path": ["event-1", "event-2"]},
            }
        )
        for option in row["qa"]["options"]:
            option["text"] = "How many people joined?"
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}
        self.assertIn("option_restates_question", reasons)
        self.assertIn("number_answer_surface_mismatch", reasons)

    def test_operation_label_must_match_question(self) -> None:
        row = valid_sample()
        row["qa"]["qa_operation"] = "cause_consequence"
        row["qa"]["qa_subtype"] = "final_state"
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}
        self.assertIn("qa_operation_question_mismatch", reasons)
        self.assertIn("qa_operation_subtype_mismatch", reasons)

    def test_raw_qa_slots_are_rejected(self) -> None:
        row = copy.deepcopy(valid_sample())
        row["graph_update"]["events"][0]["slot"] = "raw_qa_temporal"
        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}
        self.assertIn("generic_raw_qa_slot", reasons)

    def test_option_normalization_ignores_case_punctuation_and_whitespace(self) -> None:
        self.assertEqual(
            stage64_quality.normalized_option_text(" Red_notebook! "),
            stage64_quality.normalized_option_text("red notebook"),
        )

    def test_graph_gate_rejects_generic_slots_and_question_values(self) -> None:
        plan = copy.deepcopy(valid_sample()["graph_update"])
        plan["events"][0]["slot"] = "raw_qa_temporal"
        plan["events"][1]["value"] = "When did Maya leave?"

        reasons = {
            issue["reason"]
            for issue in stage64_quality.graph_plan_quality_issues(plan, "relation_lookup")
        }

        self.assertIn("generic_graph_slot", reasons)
        self.assertIn("question_used_as_graph_value", reasons)

    def test_temporal_graph_gate_requires_same_slot_update_chain(self) -> None:
        plan = copy.deepcopy(valid_sample()["graph_update"])
        for index, event in enumerate(plan["events"]):
            event["slot"] = f"unrelated_slot_{index}"

        reasons = {
            issue["reason"]
            for issue in stage64_quality.graph_plan_quality_issues(plan, "temporal_state_update")
        }

        self.assertIn("temporal_graph_missing_same_slot_update_chain", reasons)

    def test_temporal_graph_gate_accepts_current_and_superseded_chain(self) -> None:
        plan = copy.deepcopy(valid_sample()["graph_update"])
        for index, event in enumerate(plan["events"], start=1):
            event["entity"] = "Maya"
            event["slot"] = "meeting_city"
            event["value"] = ("Paris", "Lyon", "Berlin", "Rome")[index - 1]
            event["time_index"] = index
            event["status"] = "active" if index == 4 else "outdated"

        self.assertEqual(
            stage64_quality.graph_plan_quality_issues(plan, "temporal_state_update"),
            [],
        )

    def test_temporal_graph_gate_accepts_targeted_later_invalid_after_current(self) -> None:
        plan = copy.deepcopy(valid_sample()["graph_update"])
        statuses = ("outdated", "active", "expired", "canceled")
        for index, event in enumerate(plan["events"], start=1):
            event["entity"] = "Maya"
            event["slot"] = "meeting_city"
            event["value"] = ("Paris", "Lyon", "Berlin", "Rome")[index - 1]
            event["time_index"] = index
            event["status"] = statuses[index - 1]
        target = {
            "forbid_max_time_shortcut": True,
            "hard_negative_strategy": "same_slot_canceled_value",
        }

        self.assertEqual(
            stage64_quality.graph_plan_quality_issues(
                plan,
                "temporal_state_update",
                target,
            ),
            [],
        )
        legacy_reasons = {
            issue["reason"]
            for issue in stage64_quality.graph_plan_quality_issues(
                plan,
                "temporal_state_update",
            )
        }
        self.assertIn("temporal_graph_current_state_not_latest", legacy_reasons)

    def test_temporal_graph_gate_rejects_duplicate_values_and_multiple_current_events(self) -> None:
        plan = {
            "events": [
                {"event_id": "e1", "entity": "John", "slot": "vacation_destination", "value": "Europe", "status": "active", "time_index": 1},
                {"event_id": "e2", "entity": "John", "slot": "vacation_destination", "value": "Asia", "status": "canceled", "time_index": 2},
                {"event_id": "e3", "entity": "John", "slot": "vacation_destination", "value": "Europe", "status": "restored", "time_index": 3},
                {"event_id": "e4", "entity": "Tim", "slot": "recommendation", "value": "castles", "status": "active", "time_index": 4},
            ]
        }

        reasons = {
            issue["reason"]
            for issue in stage64_quality.graph_plan_quality_issues(plan, "temporal_state_update")
        }

        self.assertIn("temporal_graph_duplicate_update_value", reasons)
        self.assertIn("temporal_graph_current_state_not_unique", reasons)

    def test_temporal_qa_rejects_status_annotated_duplicate_value_and_nonfinal_gold(self) -> None:
        events = [
            {"event_instance_id": "e1", "entity": "John", "slot": "vacation_destination", "value": "Europe", "status": "active", "time_index": 1},
            {"event_instance_id": "e2", "entity": "John", "slot": "vacation_destination", "value": "Asia", "status": "canceled", "time_index": 2},
            {"event_instance_id": "e3", "entity": "John", "slot": "vacation_destination", "value": "Europe", "status": "restored", "time_index": 3},
            {"event_instance_id": "e4", "entity": "John", "slot": "vacation_duration", "value": "long", "status": "expired", "time_index": 4},
        ]
        row = {
            "graph_update": {"events": events},
            "qa": {
                "question": "After the later correction, where is John currently going on vacation?",
                "qa_operation": "temporal_state_update",
                "qa_subtype": "latest_state_update",
                "answer_type": "LocationOrScope",
                "gold": ["A"],
                "options": [
                    {"id": "A", "text": "Europe", "source_event_instance_id": "e1"},
                    {"id": "B", "text": "Asia", "source_event_instance_id": "e2"},
                    {"id": "C", "text": "Europe (restored after cancellation)", "source_event_instance_id": "e3"},
                    {"id": "D", "text": "Long (expired assumption)", "source_event_instance_id": "e4"},
                ],
                "query_plan": {"reasoning_anchor_path": ["e1", "e2", "e3"]},
            },
        }

        reasons = {issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)}

        self.assertIn("temporal_duplicate_graph_answer_values", reasons)
        self.assertIn("temporal_duplicate_option_value_surface", reasons)
        self.assertIn("temporal_option_status_annotation", reasons)
        self.assertIn("temporal_gold_not_final_update", reasons)

    def test_temporal_anti_shortcut_qa_requires_later_invalid_in_reasoning_and_distractors(self) -> None:
        events = [
            {"event_id": "e1", "event_instance_id": "e1", "entity": "Maya", "slot": "meeting_city", "value": "Paris", "status": "outdated", "time_index": 1},
            {"event_id": "e2", "event_instance_id": "e2", "entity": "Maya", "slot": "meeting_city", "value": "Lyon", "status": "active", "time_index": 2, "previous_value": "Paris"},
            {"event_id": "e3", "event_instance_id": "e3", "entity": "Maya", "slot": "meeting_city", "value": "Berlin", "status": "expired", "time_index": 3, "previous_value": "Lyon"},
            {"event_id": "e4", "event_instance_id": "e4", "entity": "Maya", "slot": "meeting_city", "value": "Rome", "status": "canceled", "time_index": 4, "previous_value": "Berlin"},
            {"event_id": "e5", "event_instance_id": "e5", "entity": "Noah", "slot": "hotel_name", "value": "Orchid", "status": "active", "time_index": 5},
            {"event_id": "e6", "event_instance_id": "e6", "entity": "Li", "slot": "train_number", "value": "T18", "status": "active", "time_index": 6},
        ]
        reasoning_ids = ["e1", "e2", "e3", "e4"]
        row = {
            "graph_update": {"graph_snapshot_id": "stage3:temporal", "events": events},
            "qa": {
                "question": "After disregarding the canceled proposal, where is Maya's current meeting city?",
                "qa_operation": "temporal_state_update",
                "qa_subtype": "latest_state_update",
                "answer_type": "LocationOrScope",
                "forbid_max_time_shortcut": True,
                "hard_negative_strategy": "same_slot_canceled_value",
                "gold": ["B"],
                "gold_answer_text": "Lyon",
                "options": [
                    {"id": "A", "text": "Paris", "source_event_instance_id": "e1"},
                    {"id": "B", "text": "Lyon", "source_event_instance_id": "e2"},
                    {"id": "C", "text": "Berlin", "source_event_instance_id": "e3"},
                    {"id": "D", "text": "Rome", "source_event_instance_id": "e4"},
                ],
                "query_plan": {
                    "reasoning_anchor_path": reasoning_ids,
                    "resolved_reasoning_event_instance_ids": reasoning_ids,
                    "resolved_distractor_event_instance_ids": ["e1", "e3", "e4"],
                },
            },
        }

        reasons = {
            issue["reason"] for issue in stage64_quality.semantic_quality_issues(row)
        }
        self.assertNotIn("temporal_anti_shortcut_gold_not_earlier_current", reasons)
        self.assertNotIn("temporal_anti_shortcut_missing_later_strategy_update", reasons)
        self.assertNotIn("temporal_anti_shortcut_later_update_not_referenced", reasons)

        missing_reference = copy.deepcopy(row)
        missing_reference["qa"]["query_plan"]["reasoning_anchor_path"] = [
            "e1", "e2", "e3"
        ]
        missing_reference["qa"]["query_plan"][
            "resolved_reasoning_event_instance_ids"
        ] = ["e1", "e2", "e3"]
        missing_reference["qa"]["query_plan"][
            "resolved_distractor_event_instance_ids"
        ] = ["e1", "e3"]
        missing_reasons = {
            issue["reason"]
            for issue in stage64_quality.semantic_quality_issues(missing_reference)
        }
        self.assertIn(
            "temporal_anti_shortcut_later_update_not_referenced",
            missing_reasons,
        )

    def test_visibility_graph_gate_accepts_explicit_scope_object(self) -> None:
        plan = copy.deepcopy(valid_sample()["graph_update"])
        plan["events"][0]["scope"] = {
            "observer": "Tim",
            "permission": "confirmed",
        }

        self.assertEqual(
            stage64_quality.graph_plan_quality_issues(plan, "visibility"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
