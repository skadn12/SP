from __future__ import annotations

import copy
import unittest

from scripts.current_method import semantic_graph_v2


def event(
    event_id: str,
    entity: str,
    slot: str,
    value: object,
    *,
    status: str = "active",
    time_index: int = 1,
    **optional: object,
) -> dict:
    row = {
        "event_id": event_id,
        "entity": entity,
        "slot": slot,
        "value": value,
        "status": status,
        "time_index": time_index,
    }
    row.update(optional)
    return row


def delta(events: list[dict], *, operation: str | None = None) -> dict:
    row = {
        "schema_version": semantic_graph_v2.SCHEMA_VERSION,
        "graph_id": "stage3:child:001",
        "parent_graph_id": "stage2:parent:001",
        "events": events,
    }
    if operation is not None:
        row["operation"] = operation
    return row


def issue_codes(issues: list[dict]) -> set[str]:
    return {issue["code"] for issue in issues}


class SemanticGraphV2ContractTest(unittest.TestCase):
    def test_declarative_predicate_fragments_are_not_questions(self) -> None:
        for value in (
            "is seeking help for bonding with Toby",
            "will try to improve the bond",
            "was moved to Lyon",
        ):
            issues = semantic_graph_v2.validate_event(
                event("e1", "Andrew", "bond_status", value)
            )
            self.assertNotIn("question_or_qa_wrapper_value", issue_codes(issues))

    def test_valid_delta_preserves_optional_fields_and_input(self) -> None:
        row = delta(
            [
                event(
                    "e1",
                    "Li",
                    "meeting_month",
                    "May",
                    previous_value="April",
                    depends_on=[],
                    scope={"observers": ["Li"]},
                    set_membership={"set_id": "meetings", "included": True},
                    evidence_turn_ids=["D3:4"],
                    provenance={"source_sample_id": "stage2-17"},
                )
            ]
        )
        before = copy.deepcopy(row)

        self.assertEqual(semantic_graph_v2.validate_graph_delta(row), [])
        self.assertTrue(semantic_graph_v2.is_valid_graph_delta(row))
        self.assertEqual(row, before)
        self.assertEqual(row["events"][0]["value"], "May")

    def test_short_scalar_values_are_valid(self) -> None:
        row = delta(
            [
                event("e1", "Li", "headcount", 3, time_index=1),
                event("e2", "Li", "approved", "No", time_index=2),
                event("e3", "Li", "activity", "yoga", time_index=3),
            ]
        )
        self.assertEqual(semantic_graph_v2.validate_graph_delta(row), [])

    def test_required_graph_and_event_fields_are_reported_with_paths(self) -> None:
        issues = semantic_graph_v2.validate_graph_delta({"events": [{}]})
        codes = issue_codes(issues)

        self.assertIn("missing_graph_field", codes)
        self.assertIn("unsupported_schema_version", codes)
        self.assertIn("missing_event_field", codes)
        self.assertTrue(all({"code", "path", "message"} <= set(issue) for issue in issues))
        self.assertIn("$.events[0].event_id", {issue["path"] for issue in issues})

    def test_raw_generic_slots_and_question_values_are_forbidden(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "raw_qa_temporal", "July"),
                event("e2", "Maya", "generic_fact", "Q: When did Maya leave? A: Friday", time_index=2),
                event("e3", "Maya", "destination", "Where did Maya leave", time_index=3),
                event("e4", "Maya", "RAW-QA.Single_Hop", "Friday", time_index=4),
            ]
        )
        issues = semantic_graph_v2.validate_graph_delta(row)
        codes = issue_codes(issues)

        self.assertIn("generic_or_raw_qa_slot", codes)
        self.assertIn("question_or_qa_wrapper_value", codes)
        self.assertEqual(sum(issue["code"] == "generic_or_raw_qa_slot" for issue in issues), 3)

    def test_unique_ids_and_existing_dependencies_are_enforced(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "plan", "Paris", depends_on=["e1"]),
                event("e1", "Maya", "plan", "Lyon", time_index=2, depends_on=["missing"]),
            ]
        )
        codes = issue_codes(semantic_graph_v2.validate_graph_delta(row))

        self.assertIn("duplicate_event_id", codes)
        self.assertIn("self_dependency", codes)
        self.assertIn("dependency_not_found", codes)

    def test_optional_field_types_are_validated_without_repair(self) -> None:
        row = delta(
            [
                event(
                    "e1",
                    "Maya",
                    "plan",
                    "Paris",
                    previous_value={"silently": "coerce me"},
                    depends_on="e0",
                    scope="Maya only",
                    set_membership={"set_id": "", "included": "yes"},
                    evidence_turn_ids=["D1:1", "D1:1"],
                    provenance="stage2",
                )
            ]
        )
        before = copy.deepcopy(row)
        codes = issue_codes(semantic_graph_v2.validate_graph_delta(row))

        self.assertTrue(
            {
                "invalid_previous_value",
                "invalid_depends_on",
                "invalid_scope",
                "invalid_set_membership_set_id",
                "invalid_set_membership_included",
                "duplicate_evidence_turn_ids_item",
                "invalid_provenance",
            }
            <= codes
        )
        self.assertEqual(row, before)

    def test_generation_status_vocabulary_matches_downstream_quality_gate(self) -> None:
        for status in ("inactive", "pending", "superseded"):
            codes = issue_codes(
                semantic_graph_v2.validate_event(
                    event("e1", "Maya", "meeting_state", "June 12", status=status)
                )
            )
            self.assertIn("unsupported_status", codes)

    def test_malformed_fields_return_issues_even_with_operation_validation(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "score", {"not": "a scalar"}, depends_on=[{"bad": "id"}]),
                event("e2", "Noah", "score", {"another": "object"}, depends_on=[[]]),
            ]
        )

        comparison_codes = issue_codes(
            semantic_graph_v2.validate_graph_delta(row, operation="comparison")
        )
        bridge_codes = issue_codes(semantic_graph_v2.validate_graph_delta(row, operation="bridge"))
        cause_codes = issue_codes(semantic_graph_v2.validate_graph_delta(row, operation="cause"))

        self.assertIn("invalid_value", comparison_codes)
        self.assertIn("invalid_depends_on_item", bridge_codes)
        self.assertIn("cause_missing_ordered_dependency", cause_codes)


class SemanticGraphV2OperationTest(unittest.TestCase):
    def test_temporal_requires_an_explicit_same_slot_update_chain(self) -> None:
        valid = delta(
            [
                event("e1", "Maya", "meeting_date", "June 3", status="outdated", time_index=1),
                event(
                    "e2",
                    "Maya",
                    "meeting_date",
                    "June 8",
                    status="canceled",
                    time_index=2,
                    previous_value="June 3",
                ),
                event(
                    "e3",
                    "Maya",
                    "meeting_date",
                    "June 12",
                    time_index=3,
                    previous_value="June 8",
                ),
                event("e4", "Noah", "travel_city", "Lyon", time_index=4),
                event("e5", "Noah", "hotel_name", "Orchid", time_index=5),
                event("e6", "Li", "train_number", "T18", time_index=6),
            ]
        )
        invalid = delta([event("e1", "Maya", "meeting_date", "June 12")])

        self.assertEqual(semantic_graph_v2.validate_graph_delta(valid, operation="temporal"), [])
        self.assertIn(
            "temporal_missing_update_chain",
            issue_codes(semantic_graph_v2.validate_graph_delta(invalid, operation="temporal")),
        )

    def test_temporal_rejects_repeated_answer_values_and_multiple_current_events(self) -> None:
        row = delta(
            [
                event("e1", "John", "vacation_destination", "Europe", status="active", time_index=1),
                event(
                    "e2",
                    "John",
                    "vacation_destination",
                    "Asia",
                    status="canceled",
                    time_index=2,
                    previous_value="Europe",
                ),
                event(
                    "e3",
                    "John",
                    "vacation_destination",
                    "Europe",
                    status="restored",
                    time_index=3,
                    previous_value="Asia",
                ),
                event("e4", "Tim", "recommendation", "castles", time_index=4),
            ]
        )

        codes = issue_codes(semantic_graph_v2.validate_graph_delta(row, operation="temporal"))

        self.assertIn("temporal_duplicate_update_value", codes)
        self.assertIn("temporal_current_state_not_unique", codes)

    def test_temporal_requires_the_unique_current_event_to_be_latest(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "meeting_city", "Paris", status="outdated", time_index=1),
                event(
                    "e2",
                    "Maya",
                    "meeting_city",
                    "Lyon",
                    status="active",
                    time_index=2,
                    previous_value="Paris",
                ),
                event(
                    "e3",
                    "Maya",
                    "meeting_city",
                    "Berlin",
                    status="canceled",
                    time_index=3,
                    previous_value="Lyon",
                ),
                event("e4", "Noah", "meeting_city", "Rome", time_index=4),
            ]
        )

        self.assertIn(
            "temporal_current_state_not_latest",
            issue_codes(semantic_graph_v2.validate_graph_delta(row, operation="temporal")),
        )

    def test_temporal_anti_shortcut_accepts_earlier_current_with_later_target_invalid(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "meeting_city", "Paris", status="outdated", time_index=1),
                event(
                    "e2",
                    "Maya",
                    "meeting_city",
                    "Lyon",
                    status="active",
                    time_index=2,
                    previous_value="Paris",
                    depends_on=["e1"],
                ),
                event(
                    "e3",
                    "Maya",
                    "meeting_city",
                    "Berlin",
                    status="expired",
                    time_index=3,
                    previous_value="Lyon",
                    depends_on=["e2"],
                ),
                event(
                    "e4",
                    "Maya",
                    "meeting_city",
                    "Rome",
                    status="canceled",
                    time_index=4,
                    previous_value="Berlin",
                    depends_on=["e3"],
                ),
                event("e5", "Noah", "hotel_name", "Orchid", time_index=5),
                event("e6", "Li", "train_number", "T18", time_index=6),
            ]
        )
        target = {
            "forbid_max_time_shortcut": True,
            "hard_negative_strategy": "same_slot_canceled_value",
        }

        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                row,
                operation="temporal",
                reasoning_target=target,
            ),
            [],
        )
        self.assertIn(
            "temporal_current_state_not_latest",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(row, operation="temporal")
            ),
        )

        wrong_strategy = copy.deepcopy(row)
        wrong_strategy["events"][3]["status"] = "outdated"
        self.assertIn(
            "temporal_anti_shortcut_missing_later_strategy_update",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    wrong_strategy,
                    operation="temporal",
                    reasoning_target=target,
                )
            ),
        )

    def test_bridge_requires_two_hops_or_two_dependencies(self) -> None:
        valid = delta(
            [
                event("e1", "Maya", "owns", "blue key", time_index=1),
                event("e2", "blue key", "opens", "studio", time_index=2, depends_on=["e1"]),
                event("e3", "studio", "contains", "violin", time_index=3, depends_on=["e2"]),
                event("e4", "Noah", "owns", "red key", time_index=4),
                event("e5", "Maya", "prefers", "tea", time_index=5),
                event("e6", "Li", "prefers", "coffee", time_index=6),
            ]
        )
        self.assertEqual(semantic_graph_v2.validate_graph_delta(valid, operation="bridge"), [])
        self.assertIn(
            "bridge_missing_two_hop_dependency",
            issue_codes(semantic_graph_v2.validate_graph_delta(delta(valid["events"][:1]), operation="bridge")),
        )

    def test_comparison_requires_same_attribute_and_time_frame(self) -> None:
        valid = delta(
            [
                event("e1", "Maya", "project_score", 8, time_index=4),
                event("e2", "Noah", "project_score", 6, time_index=4),
                event("e3", "Maya", "project_name", "Atlas", time_index=3),
                event("e4", "Noah", "project_owner", "Nova", time_index=3),
                event("e5", "Li", "office_city", "Rome", time_index=5),
                event("e6", "Omar", "office_floor", 4, time_index=6),
            ]
        )
        self.assertEqual(semantic_graph_v2.validate_graph_delta(valid, operation="comparison"), [])
        valid["events"][1]["time_index"] = 5
        self.assertIn(
            "comparison_missing_unique_numeric_pair",
            issue_codes(semantic_graph_v2.validate_graph_delta(valid, operation="comparison")),
        )

    def test_visibility_requires_explicit_scope(self) -> None:
        valid = delta(
            [
                event(
                    "e1",
                    "Maya",
                    "document_access",
                    "budget.xlsx",
                    status="scope_limited",
                    scope={"observers": ["Maya", "Noah"], "visibility": "private"},
                ),
                event("e2", "Maya", "document_owner", "finance", time_index=2),
                event("e3", "Noah", "document_owner", "legal", time_index=3),
                event("e4", "Li", "document_owner", "operations", time_index=4),
                event("e5", "Omar", "office_city", "Paris", time_index=5),
                event("e6", "Zoe", "office_city", "Lyon", time_index=6),
            ]
        )
        self.assertEqual(semantic_graph_v2.validate_graph_delta(valid, operation="visibility"), [])
        del valid["events"][0]["scope"]
        self.assertIn(
            "visibility_missing_scope",
            issue_codes(semantic_graph_v2.validate_graph_delta(valid, operation="visibility")),
        )

    def test_count_requires_six_balanced_explicit_set_memberships(self) -> None:
        events = [
            event(
                f"e{index}",
                "Atlas project",
                "active_member",
                name,
                time_index=index,
                set_membership={"set_id": "active_atlas_members", "included": index <= 3},
            )
            for index, name in enumerate(("Maya", "Noah", "Li", "Omar", "Zoe", "Iris"), start=1)
        ]
        self.assertEqual(semantic_graph_v2.validate_graph_delta(delta(events), operation="count"), [])
        self.assertIn(
            "count_missing_candidate_set",
            issue_codes(semantic_graph_v2.validate_graph_delta(delta(events[:5]), operation="count")),
        )
        with_unrelated_filler = events + [
            event("e7", "Maya", "office_city", "Paris", time_index=7)
        ]
        self.assertIn(
            "count_missing_candidate_set",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    delta(with_unrelated_filler), operation="count"
                )
            ),
        )

    def test_count_rejects_mixed_entities_slots_and_duplicate_members(self) -> None:
        events = [
            event(
                f"e{index}",
                "Atlas project",
                "active_member",
                name,
                time_index=index,
                set_membership={"set_id": "active_atlas_members", "included": True},
            )
            for index, name in enumerate(("Maya", "Noah", "Li", "Omar"), start=1)
        ]
        mixed_entity = copy.deepcopy(events)
        mixed_entity[3]["entity"] = "Beacon project"
        mixed_slot = copy.deepcopy(events)
        mixed_slot[3]["slot"] = "former_member"
        duplicate_member = copy.deepcopy(events)
        duplicate_member[3]["value"] = "Maya"
        contaminated_set = copy.deepcopy(events)
        contaminated_set.append(
            event(
                "e5",
                "Beacon project",
                "former_member",
                "Zoe",
                time_index=5,
                set_membership={"set_id": "active_atlas_members", "included": False},
            )
        )

        for invalid in (mixed_entity, mixed_slot, duplicate_member, contaminated_set):
            self.assertIn(
                "count_missing_candidate_set",
                issue_codes(semantic_graph_v2.validate_graph_delta(delta(invalid), operation="count")),
            )

    def test_cause_requires_an_ordered_dependency(self) -> None:
        valid = delta(
            [
                event("e1", "storm", "effect", "power outage", time_index=2),
                event("e2", "Maya", "meeting_state", "canceled", time_index=3, depends_on=["e1"]),
                event("e3", "Noah", "meeting_state", "delayed", time_index=4),
                event("e4", "Li", "meeting_state", "confirmed", time_index=5),
                event("e5", "Omar", "route", "north", time_index=6),
                event("e6", "Zoe", "route", "south", time_index=7),
            ]
        )
        self.assertEqual(semantic_graph_v2.validate_graph_delta(valid, operation="cause"), [])
        valid["events"][1]["depends_on"] = []
        self.assertIn(
            "cause_missing_ordered_dependency",
            issue_codes(semantic_graph_v2.validate_graph_delta(valid, operation="cause")),
        )

    def test_relation_requires_both_hard_negative_shapes(self) -> None:
        valid = delta(
            [
                event("e1", "Maya", "office_city", "Paris", status="outdated", time_index=1),
                event("e2", "Maya", "office_city", "Lyon", time_index=2),
                event("e3", "Noah", "office_city", "Paris", time_index=2),
                event("e4", "Li", "favorite_city", "Rome", time_index=3),
                event("e5", "Omar", "office_floor", 3, time_index=4),
                event("e6", "Zoe", "favorite_color", "green", time_index=5),
            ]
        )
        self.assertEqual(semantic_graph_v2.validate_graph_delta(valid, operation="relation"), [])
        self.assertIn(
            "relation_missing_hard_negative_structure",
            issue_codes(semantic_graph_v2.validate_graph_delta(delta(valid["events"][:1]), operation="relation")),
        )

    def test_embedded_operation_must_match_requested_operation(self) -> None:
        row = delta([event("e1", "Maya", "document_access", "budget.xlsx")], operation="visibility")
        codes = issue_codes(semantic_graph_v2.validate_graph_delta(row, operation="count"))
        self.assertIn("operation_mismatch", codes)

    def test_every_stage3_operation_requires_six_to_eight_events(self) -> None:
        for operation in sorted(set(semantic_graph_v2.OPERATION_ALIASES.values())):
            too_few = delta(
                [
                    event(f"e{index}", f"entity-{index}", f"slot-{index}", f"value-{index}")
                    for index in range(1, 6)
                ]
            )
            self.assertIn(
                "operation_too_few_events",
                issue_codes(semantic_graph_v2.validate_graph_delta(too_few, operation=operation)),
            )
            too_many = delta(
                [
                    event(f"e{index}", f"entity-{index}", f"slot-{index}", f"value-{index}")
                    for index in range(1, 10)
                ]
            )
            self.assertIn(
                "operation_too_many_events",
                issue_codes(semantic_graph_v2.validate_graph_delta(too_many, operation=operation)),
            )

    def test_formal_operation_rejects_optional_or_invalid_required_flags(self) -> None:
        events = [
            event(
                f"e{index}",
                f"entity-{index}",
                f"slot-{index}",
                f"value-{index}",
                required_for_dialogue=True,
            )
            for index in range(1, 7)
        ]
        events[-1]["required_for_dialogue"] = False
        codes = issue_codes(
            semantic_graph_v2.validate_graph_delta(delta(events), operation="bridge")
        )
        self.assertIn("operation_too_few_required_events", codes)

        events[-1]["required_for_dialogue"] = "true"
        codes = issue_codes(
            semantic_graph_v2.validate_graph_delta(delta(events), operation="bridge")
        )
        self.assertIn("invalid_required_for_dialogue", codes)

    def test_temporal_requires_every_adjacent_link_and_at_most_four_chain_events(self) -> None:
        broken = delta(
            [
                event("e1", "Maya", "city", "Paris", status="outdated", time_index=1),
                event("e2", "Maya", "city", "Lyon", status="canceled", time_index=2, previous_value="Paris"),
                event("e3", "Maya", "city", "Berlin", time_index=3, previous_value="Paris"),
                event("e4", "Noah", "hotel", "Orchid", time_index=4),
                event("e5", "Li", "route", "north", time_index=5),
                event("e6", "Omar", "room", "B12", time_index=6),
            ]
        )
        self.assertIn(
            "temporal_broken_previous_value_chain",
            issue_codes(semantic_graph_v2.validate_graph_delta(broken, operation="temporal")),
        )

        five_update_chain = delta(
            [
                event(
                    f"e{index}",
                    "Maya",
                    "city",
                    city,
                    status="active" if index == 5 else "outdated",
                    time_index=index,
                    **({"previous_value": previous} if previous else {}),
                )
                for index, (city, previous) in enumerate(
                    (("Paris", None), ("Lyon", "Paris"), ("Berlin", "Lyon"), ("Rome", "Berlin"), ("Oslo", "Rome")),
                    start=1,
                )
            ]
            + [event("e6", "Noah", "hotel", "Orchid", time_index=6)]
        )
        self.assertIn(
            "temporal_chain_length_out_of_range",
            issue_codes(semantic_graph_v2.validate_graph_delta(five_update_chain, operation="temporal")),
        )

    def test_relation_does_not_count_surface_only_value_variants_as_hard_negatives(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "office_city", "Paris", time_index=1),
                event("e2", "Maya", "office_city", " PARIS! ", time_index=2),
                event("e3", "Noah", "office_city", "Lyon", time_index=2),
                event("e4", "Li", "favorite_city", "Rome", time_index=3),
            ]
        )

        self.assertIn(
            "relation_missing_hard_negative_structure",
            issue_codes(semantic_graph_v2.validate_graph_delta(row, operation="relation")),
        )


class SemanticGraphV2TargetAwareOperationTest(unittest.TestCase):
    def test_temporal_target_requires_immediate_dependency_chain(self) -> None:
        row = delta(
            [
                event("e1", "Maya", "meeting_city", "Paris", status="outdated", time_index=1),
                event(
                    "e2",
                    "Maya",
                    "meeting_city",
                    "Lyon",
                    status="canceled",
                    time_index=2,
                    previous_value="Paris",
                    depends_on=["e1"],
                ),
                event(
                    "e3",
                    "Maya",
                    "meeting_city",
                    "Berlin",
                    time_index=3,
                    previous_value="Lyon",
                    depends_on=["e2"],
                ),
                event("e4", "Noah", "hotel_name", "Orchid", time_index=4),
                event("e5", "Li", "train_number", "T18", time_index=5),
                event("e6", "Omar", "office_floor", 4, time_index=6),
            ]
        )
        target = {"reasoning_subtype": "status_gated_outdated_update"}

        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                row,
                operation="temporal",
                reasoning_target=target,
            ),
            [],
        )

        missing_dependency = copy.deepcopy(row)
        del missing_dependency["events"][2]["depends_on"]
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                missing_dependency,
                operation="temporal",
            ),
            [],
        )
        self.assertIn(
            "temporal_target_broken_dependency_chain",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    missing_dependency,
                    operation="temporal",
                    reasoning_target=target,
                )
            ),
        )

    def test_bridge_target_requires_one_identifiable_dependency_endpoint(self) -> None:
        target = {"reasoning_subtype": "dependency_chain_bridge", "hop_count": 2}
        valid = delta(
            [
                event("e1", "Maya", "owns", "blue key", time_index=1),
                event("e2", "blue key", "opens", "studio", time_index=2, depends_on=["e1"]),
                event("e3", "studio", "contains", "violin", time_index=3, depends_on=["e2"]),
                event("e4", "Noah", "owns", "red key", time_index=4),
                event("e5", "Li", "prefers", "tea", time_index=5),
                event("e6", "Omar", "prefers", "coffee", time_index=6),
            ]
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                valid,
                operation="bridge",
                reasoning_target=target,
            ),
            [],
        )

        parallel_endpoints = delta(
            [
                event("e1", "Maya", "membership", "Signal Watch", time_index=1),
                event("e2", "Maya", "permission", "read", time_index=2),
                event("e3", "Maya", "eligible_report", "Internal Report", time_index=3, depends_on=["e1", "e2"]),
                event("e4", "Joanna", "membership", "Signal Watch", time_index=4),
                event("e5", "Joanna", "permission", "read", time_index=5),
                event("e6", "Joanna", "eligible_report", "Internal Report", time_index=6, depends_on=["e4", "e5"]),
            ]
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(parallel_endpoints, operation="bridge"),
            [],
        )
        self.assertIn(
            "bridge_target_missing_unique_dependency_endpoint",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    parallel_endpoints,
                    operation="bridge",
                    reasoning_target=target,
                )
            ),
        )

    def test_comparison_target_requires_wrong_time_pair_and_strategy_difference(self) -> None:
        target = {
            "reasoning_subtype": "same_slot_absolute_difference",
            "hard_negative_strategy": "near_numeric_off_by_one",
        }
        valid = delta(
            [
                event("e1", "Maya", "project_score", 10, time_index=2),
                event("e2", "Noah", "project_score", 15, time_index=2),
                event("e3", "Maya", "project_score", 11, status="outdated", time_index=1),
                event("e4", "Noah", "project_score", 15, status="outdated", time_index=1),
                event("e5", "Li", "office_city", "Rome", time_index=5),
                event("e6", "Omar", "office_floor", 4, time_index=6),
            ]
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                valid,
                operation="comparison",
                reasoning_target=target,
            ),
            [],
        )

        direct_only = copy.deepcopy(valid)
        direct_only["events"][2].update(
            {"entity": "Li", "slot": "office_city", "value": "Paris"}
        )
        direct_only["events"][3].update(
            {"entity": "Omar", "slot": "office_floor", "value": 3}
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(direct_only, operation="comparison"),
            [],
        )
        self.assertIn(
            "comparison_target_missing_unique_wrong_time_pair",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    direct_only,
                    operation="comparison",
                    reasoning_target=target,
                )
            ),
        )

        wrong_difference = copy.deepcopy(valid)
        wrong_difference["events"][2]["value"] = 12
        self.assertIn(
            "comparison_target_near_numeric_strategy_mismatch",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    wrong_difference,
                    operation="comparison",
                    reasoning_target=target,
                )
            ),
        )

    def test_visibility_target_requires_structured_unambiguous_scope(self) -> None:
        target = {"reasoning_subtype": "observer_permission_scope"}
        valid = delta(
            [
                event(
                    "e1",
                    "Caroline",
                    "report_access",
                    "Internal Report",
                    scope={"observer": "Jules", "permission": "read"},
                ),
                event(
                    "e2",
                    "Caroline",
                    "report_access",
                    "Internal Report",
                    status="scope_limited",
                    time_index=2,
                    scope={"observer": "Nina", "permission": "read"},
                ),
                event(
                    "e3",
                    "Caroline",
                    "report_access",
                    "Internal Report",
                    status="outdated",
                    time_index=3,
                    scope={"observer": "Jules", "permission": "write"},
                ),
                event("e4", "Noah", "office_city", "Rome", time_index=4),
                event("e5", "Li", "office_city", "Paris", time_index=5),
                event("e6", "Omar", "office_floor", 4, time_index=6),
            ]
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                valid,
                operation="visibility",
                reasoning_target=target,
            ),
            [],
        )

        ambiguous = copy.deepcopy(valid)
        ambiguous["events"][1].update(
            {
                "entity": "Melanie",
                "status": "active",
                "scope": {"observer": "Jules", "permission": "read"},
            }
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(ambiguous, operation="visibility"),
            [],
        )
        self.assertIn(
            "visibility_target_ambiguous_active_scope_subject",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    ambiguous,
                    operation="visibility",
                    reasoning_target=target,
                )
            ),
        )

        unstructured = copy.deepcopy(valid)
        unstructured["events"][0]["scope"] = {
            "observers": ["Jules"],
            "visibility": "private",
        }
        self.assertIn(
            "visibility_target_missing_structured_scope",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    unstructured,
                    operation="visibility",
                    reasoning_target=target,
                )
            ),
        )

    def test_count_target_requires_one_explicit_membership_condition(self) -> None:
        target = {"reasoning_subtype": "filtered_membership_count"}
        events = [
            event(
                f"e{index}",
                "Atlas project",
                "eligible_member",
                name,
                time_index=index,
                set_membership={
                    "set_id": "eligible_atlas_members",
                    "included": index <= 3,
                    "condition": "membership is active or restored",
                },
            )
            for index, name in enumerate(
                ("Maya", "Noah", "Li", "Omar", "Zoe", "Iris"),
                start=1,
            )
        ]
        row = delta(events)
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                row,
                operation="count",
                reasoning_target=target,
            ),
            [],
        )

        missing_condition = copy.deepcopy(row)
        for item in missing_condition["events"]:
            del item["set_membership"]["condition"]
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(missing_condition, operation="count"),
            [],
        )
        self.assertIn(
            "count_target_missing_shared_condition",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    missing_condition,
                    operation="count",
                    reasoning_target=target,
                )
            ),
        )

        mixed_condition = copy.deepcopy(row)
        mixed_condition["events"][-1]["set_membership"]["condition"] = (
            "membership is active"
        )
        self.assertIn(
            "count_target_missing_shared_condition",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    mixed_condition,
                    operation="count",
                    reasoning_target=target,
                )
            ),
        )

    def test_direct_cause_target_requires_unique_edge_and_adjacent_non_cause(self) -> None:
        target = {
            "reasoning_subtype": "direct_causal_dependency",
            "reasoning_template": "trace_cause_to_effect",
        }
        valid = delta(
            [
                event("e1", "storm", "weather_event", "hail", time_index=2),
                event("e2", "Maya", "meeting_state", "canceled", time_index=4, depends_on=["e1"]),
                event("e3", "Noah", "route_state", "delayed", time_index=3),
                event("e4", "Li", "meeting_state", "confirmed", time_index=6),
                event("e5", "Omar", "route", "north", time_index=7),
                event("e6", "Zoe", "route", "south", time_index=8),
            ]
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                valid,
                operation="cause",
                reasoning_target=target,
            ),
            [],
        )

        multiple_edges = copy.deepcopy(valid)
        multiple_edges["events"][3]["depends_on"] = ["e3"]
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(multiple_edges, operation="cause"),
            [],
        )
        self.assertIn(
            "cause_target_not_unique_direct_pair",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    multiple_edges,
                    operation="cause",
                    reasoning_target=target,
                )
            ),
        )

        missing_adjacent = copy.deepcopy(valid)
        missing_adjacent["events"][2]["time_index"] = 6
        missing_adjacent["events"][3]["time_index"] = 7
        missing_adjacent["events"][4]["time_index"] = 8
        missing_adjacent["events"][5]["time_index"] = 9
        self.assertIn(
            "cause_target_missing_adjacent_noncausal_negative",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    missing_adjacent,
                    operation="cause",
                    reasoning_target=target,
                )
            ),
        )

    def test_relation_target_requires_unique_binder_dependency(self) -> None:
        target = {"reasoning_subtype": "dual_constraint_relation"}
        valid = delta(
            [
                event("e1", "Maya", "team_code", "team-17", time_index=1),
                event("e2", "Maya", "appointment_date", "June 12", time_index=2, depends_on=["e1"]),
                event("e3", "Maya", "appointment_date", "June 18", status="outdated", time_index=3),
                event("e4", "Noah", "appointment_date", "June 12", time_index=2),
                event("e5", "Li", "office_city", "Rome", time_index=4),
                event("e6", "Omar", "office_floor", 4, time_index=5),
            ]
        )
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(
                valid,
                operation="relation",
                reasoning_target=target,
            ),
            [],
        )

        no_binder = copy.deepcopy(valid)
        del no_binder["events"][1]["depends_on"]
        self.assertEqual(
            semantic_graph_v2.validate_graph_delta(no_binder, operation="relation"),
            [],
        )
        self.assertIn(
            "relation_target_missing_unique_binder_dependency",
            issue_codes(
                semantic_graph_v2.validate_graph_delta(
                    no_binder,
                    operation="relation",
                    reasoning_target=target,
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
