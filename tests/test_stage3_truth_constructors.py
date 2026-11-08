from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import stage3_truth_constructors as truth  # noqa: E402


def immutable_target(operation: str) -> dict[str, object]:
    specs = {
        "temporal_state_update": {
            "preferred_answer_type": "LocationOrScope",
            "reasoning_subtype": "status_gated_canceled_update",
            "reasoning_template": "reject_canceled_value_then_resolve_update",
            "hop_count": 4,
            "hard_negative_strategy": "same_slot_canceled_value",
            "forbid_max_time_shortcut": True,
        },
        "bridge": {
            "preferred_answer_type": "Person",
            "reasoning_subtype": "conjunctive_two_support_bridge",
            "reasoning_template": "join_two_support_facts",
            "hop_count": 2,
            "hard_negative_strategy": "one_hop_preserving_alternative",
        },
        "comparison": {
            "preferred_answer_type": "Number",
            "reasoning_subtype": "same_slot_absolute_difference",
            "reasoning_template": "bind_pair_then_compute_difference",
            "hop_count": 2,
            "hard_negative_strategy": "near_numeric_off_by_one",
        },
        "visibility": {
            "preferred_answer_type": "Person",
            "reasoning_subtype": "observer_permission_scope",
            "reasoning_template": "resolve_observer_permission_and_scope",
            "hop_count": 2,
            "hard_negative_strategy": "correct_fact_wrong_observer",
        },
        "count": {
            "preferred_answer_type": "Number",
            "reasoning_subtype": "filtered_membership_count",
            "reasoning_template": "filter_members_then_count",
            "hop_count": 2,
            "hard_negative_strategy": "include_exclude_boundary_count",
        },
        "cause_consequence": {
            "preferred_answer_type": "Person",
            "reasoning_subtype": "direct_causal_dependency",
            "reasoning_template": "trace_cause_to_effect",
            "hop_count": 2,
            "hard_negative_strategy": "temporally_adjacent_non_cause",
        },
        "relation_lookup": {
            "preferred_answer_type": "Person",
            "reasoning_subtype": "entity_constrained_relation",
            "reasoning_template": "bind_entity_then_relation",
            "hop_count": 2,
            "hard_negative_strategy": "same_relation_wrong_entity",
        },
    }
    return {
        "qa_operation": operation,
        **copy.deepcopy(specs[operation]),
        "target_basis": "truth_constructor_unit_test",
    }


def skeleton(operation: str) -> dict[str, object]:
    return truth.construct_truth_skeleton(
        immutable_target(operation),
        graph_id=f"truth-test-{operation}",
        parent_graph_id="parent-test-001",
        selected_parent_event_ids=["P17", "P29"],
    )


def valid_payload(operation: str) -> dict[str, object]:
    return truth.example_content_bindings(skeleton(operation))


def skeleton_with_answer_type(operation: str, answer_type: str) -> dict[str, object]:
    target = immutable_target(operation)
    target["preferred_answer_type"] = answer_type
    return truth.construct_truth_skeleton(
        target,
        graph_id=f"truth-test-{operation}-{answer_type}",
        parent_graph_id="parent-test-001",
        selected_parent_event_ids=["P17", "P29"],
    )


def value_binding_ids_for_slot(
    local_skeleton: dict[str, object], slot: str
) -> list[str]:
    result: list[str] = []
    for event in local_skeleton["event_templates"]:
        value = event.get("value")
        if event.get("slot") == slot and isinstance(value, dict):
            binding_id = value.get("$binding")
            if isinstance(binding_id, str) and binding_id not in result:
                result.append(binding_id)
    return result


def by_event_id(materialized: dict[str, object]) -> dict[str, dict[str, object]]:
    graph = materialized["graph"]
    return {event["event_id"]: event for event in graph["events"]}


def derive_gold_from_materialized(materialized: dict[str, object]) -> object:
    manifest = materialized["truth_manifest"]
    proof = manifest["proof_contract"]
    events = by_event_id(materialized)
    projection = proof["gold_projection"]
    if projection["kind"] == "event_value":
        return events[projection["event_id"]]["value"]
    if projection["kind"] == "event_field":
        return events[projection["event_id"]][projection["field"]]
    if projection["kind"] == "absolute_difference":
        left_id, right_id = projection["event_ids"]
        return abs(events[left_id]["value"] - events[right_id]["value"])
    if projection["kind"] == "count_membership":
        return sum(
            event["set_membership"]["included"] is projection["included"]
            for event in events.values()
            if isinstance(event.get("set_membership"), dict)
        )
    raise AssertionError(f"unknown projection: {projection}")


class TruthConstructorHappyPathTest(unittest.TestCase):
    def test_all_seven_operations_construct_and_materialize_locally(self) -> None:
        self.assertEqual(set(truth.OPERATIONS), set(immutable_target(op)["qa_operation"] for op in truth.OPERATIONS))
        for operation in truth.OPERATIONS:
            with self.subTest(operation=operation):
                local_skeleton = skeleton(operation)
                payload = truth.example_content_bindings(local_skeleton)
                validation = truth.validate_content_bindings(local_skeleton, payload)
                self.assertTrue(validation["pass"], validation["issues"])

                materialized = truth.materialize_truth_skeleton(local_skeleton, payload)
                graph = materialized["graph"]
                manifest = materialized["truth_manifest"]
                proof = manifest["proof_contract"]

                self.assertEqual(graph["operation"], operation)
                self.assertEqual(len(graph["events"]), 8)
                self.assertEqual([event["event_id"] for event in graph["events"]], [f"E{i}" for i in range(1, 9)])
                self.assertTrue(all(event["source"] == "local_truth_constructor" for event in graph["events"]))
                self.assertEqual(manifest["api_defined_truth_fields"], [])
                self.assertEqual(proof["compiled_gold"], derive_gold_from_materialized(materialized))
                self.assertGreaterEqual(len(proof["reasoning_event_ids"]), 2)
                self.assertTrue(set(proof["required_event_ids"]).issubset(by_event_id(materialized)))
                self.assertNotIn("parent_continuity", graph["metadata"])
                self.assertTrue(
                    all(
                        "depends_on_parent_event_ids" not in event
                        and "parent_topic_anchor_event_ids" not in event
                        for event in graph["events"]
                    ),
                    "IDs without parent facts must never fabricate continuity",
                )

                discourse = materialized["discourse_plan"]
                node_claim_ids = {
                    claim["subject_event_id"]
                    for claim in discourse["claims"]
                    if claim["claim_type"] in {"node", "membership", "visibility"}
                }
                self.assertEqual(node_claim_ids, {f"E{i}" for i in range(1, 9)})
                for claim in discourse["claims"]:
                    semantic_claim = claim["canonical_support_claim"]
                    self.assertIsInstance(semantic_claim, dict)
                    self.assertTrue(
                        {
                            "subject",
                            "predicate",
                            "objects",
                            "polarity",
                            "status",
                            "time",
                            "condition",
                            "scope",
                        }.issubset(semantic_claim)
                    )
                    self.assertTrue(claim["support_realization_hint"])
                    self.assertIs(claim["surface_realization_api_writable"], True)
                    self.assertNotIn("exact_support_text", claim)
                self.assertTrue(discourse["api_may_change_claim_text"])
                self.assertFalse(discourse["api_may_change_claim_semantics"])
                self.assertTrue(discourse["isolated_fact_turns_forbidden"])
                self.assertTrue(discourse["unrelated_filler_forbidden"])

    def test_api_contract_exposes_only_typed_content_bindings(self) -> None:
        forbidden_truth_names = {
            "events",
            "slots",
            "status",
            "depends_on",
            "set_membership",
            "scope",
            "gold",
            "proof",
            "reasoning_path",
            "answer_role",
        }
        for operation in truth.OPERATIONS:
            with self.subTest(operation=operation):
                local_skeleton = skeleton(operation)
                contract = truth.api_binding_contract(local_skeleton)
                self.assertEqual(
                    set(contract["output_schema"]),
                    {"bindings", "discourse_frame"},
                )
                self.assertEqual(
                    set(contract["bindings"]),
                    {spec["binding_id"] for spec in local_skeleton["binding_specs"]},
                )
                self.assertTrue(forbidden_truth_names.issubset(contract["forbidden_output_keys"]))
                self.assertTrue(all(path.startswith("$.bindings.") for path in local_skeleton["api_writable_paths"]))

    def test_omitted_local_envelope_is_restored_without_touching_api_content(self) -> None:
        local_skeleton = skeleton("temporal_state_update")
        complete = truth.example_content_bindings(local_skeleton)
        raw = {
            "bindings": copy.deepcopy(complete["bindings"]),
            "discourse_frame": copy.deepcopy(complete["discourse_frame"]),
        }
        normalized, audit = truth.normalize_content_binding_envelope(
            local_skeleton, raw
        )
        self.assertEqual(
            audit["injected_fields"], ["schema_version", "skeleton_sha256"]
        )
        self.assertEqual(normalized["bindings"], raw["bindings"])
        self.assertEqual(normalized["discourse_frame"], raw["discourse_frame"])
        self.assertTrue(truth.validate_content_bindings(local_skeleton, normalized)["pass"])

    def test_explicit_wrong_envelope_is_preserved_and_rejected(self) -> None:
        local_skeleton = skeleton("bridge")
        payload = truth.example_content_bindings(local_skeleton)
        payload["schema_version"] = "wrong-schema"
        payload["skeleton_sha256"] = "wrong-hash"
        normalized, audit = truth.normalize_content_binding_envelope(
            local_skeleton, payload
        )
        self.assertEqual(audit["injected_fields"], [])
        self.assertEqual(normalized["schema_version"], "wrong-schema")
        self.assertEqual(normalized["skeleton_sha256"], "wrong-hash")
        issues = {
            issue["code"]
            for issue in truth.validate_content_bindings(local_skeleton, normalized)[
                "issues"
            ]
        }
        self.assertEqual(
            issues, {"binding_schema_mismatch", "binding_skeleton_hash_mismatch"}
        )

    def test_partial_or_null_envelope_is_never_overwritten(self) -> None:
        local_skeleton = skeleton("bridge")
        complete = truth.example_content_bindings(local_skeleton)

        partial = copy.deepcopy(complete)
        del partial["skeleton_sha256"]
        normalized, audit = truth.normalize_content_binding_envelope(
            local_skeleton, partial
        )
        self.assertEqual(audit["injected_fields"], ["skeleton_sha256"])
        self.assertEqual(normalized["schema_version"], complete["schema_version"])

        explicit_null = copy.deepcopy(complete)
        explicit_null["schema_version"] = None
        normalized_null, null_audit = truth.normalize_content_binding_envelope(
            local_skeleton, explicit_null
        )
        self.assertEqual(null_audit["injected_fields"], [])
        self.assertIsNone(normalized_null["schema_version"])
        self.assertIn(
            "binding_schema_mismatch",
            {
                issue["code"]
                for issue in truth.validate_content_bindings(
                    local_skeleton, normalized_null
                )["issues"]
            },
        )

    def test_envelope_normalizer_does_not_delete_forbidden_truth_fields(self) -> None:
        local_skeleton = skeleton("bridge")
        complete = truth.example_content_bindings(local_skeleton)
        raw = {
            "bindings": copy.deepcopy(complete["bindings"]),
            "discourse_frame": copy.deepcopy(complete["discourse_frame"]),
            "events": [{"slot": "discovery_date", "value": "Lila"}],
        }
        normalized, _audit = truth.normalize_content_binding_envelope(
            local_skeleton, raw
        )
        self.assertEqual(normalized["events"], raw["events"])
        self.assertIn(
            "forbidden_binding_payload_keys",
            {
                issue["code"]
                for issue in truth.validate_content_bindings(
                    local_skeleton, normalized
                )["issues"]
            },
        )

    def test_event_truth_fields_equal_hashed_local_templates(self) -> None:
        truth_fields = (
            "event_id",
            "constructor_role",
            "slot",
            "status",
            "time_index",
            "depends_on",
            "required_for_dialogue",
        )
        for operation in truth.OPERATIONS:
            with self.subTest(operation=operation):
                local_skeleton = skeleton(operation)
                materialized = truth.materialize_truth_skeleton(
                    local_skeleton,
                    truth.example_content_bindings(local_skeleton),
                )
                for template, event in zip(local_skeleton["event_templates"], materialized["graph"]["events"]):
                    for field in truth_fields:
                        self.assertEqual(event.get(field), template.get(field), (operation, event["event_id"], field))

    def test_tampered_skeleton_is_rejected_before_binding_validation(self) -> None:
        local_skeleton = skeleton("bridge")
        local_skeleton["event_templates"][1]["slot"] = "discovery_date"
        with self.assertRaisesRegex(truth.TruthConstructionError, "hash mismatch"):
            truth.validate_content_bindings(local_skeleton, truth.example_content_bindings(skeleton("bridge")))

    def test_child_global_timeline_starts_after_parent_and_keeps_local_time(self) -> None:
        local_skeleton = truth.construct_truth_skeleton(
            immutable_target("relation_lookup"),
            graph_id="truth-test-timeline",
            parent_graph_id="parent-test-timeline",
            selected_parent_event_ids=["P17"],
            parent_max_global_time_index=148,
            parent_max_session_index=6,
        )
        materialized = truth.materialize_truth_skeleton(
            local_skeleton, truth.example_content_bindings(local_skeleton)
        )
        events = materialized["graph"]["events"]
        self.assertEqual([event["local_time_index"] for event in events], list(range(1, 9)))
        self.assertEqual([event["global_time_index"] for event in events], list(range(149, 157)))
        self.assertTrue(all(event["session_index"] == 7 for event in events))
        self.assertTrue(all(event["global_time_index"] > 148 for event in events))
        self.assertTrue(
            truth.validate_materialized_truth(local_skeleton, materialized)["pass"]
        )

    def test_parent_continuity_requires_topic_anchor_beyond_entity_overlap(self) -> None:
        unrelated_parent_event = {
            "event_id": "P-flooring",
            "entity": "Maria",
            "slot": "studio_flooring",
            "value": "sprung maple floor",
            "status": "active",
            "time_index": 12,
            "global_time_index": 144,
            "session_index": 6,
            "evidence_turn_ids": ["D6:4"],
            "node_type": "atomic_memory_event",
            "provenance": {"source": "unit_test"},
        }
        with self.assertRaisesRegex(
            truth.TruthConstructionError,
            "no type-compatible semantic continuity anchor",
        ):
            truth.construct_truth_skeleton(
                immutable_target("bridge"),
                graph_id="truth-test-continuity-unrelated",
                parent_graph_id="parent-test-continuity",
                selected_parent_event_ids=["P-flooring"],
                selected_parent_events=[unrelated_parent_event],
                parent_max_global_time_index=144,
                parent_max_session_index=6,
            )

        parent_event = {
            **unrelated_parent_event,
            "event_id": "P-roster",
            "entity": "Workshop Roster",
            "slot": "selected_person",
            "value": "Maria",
        }
        local_skeleton = truth.construct_truth_skeleton(
            immutable_target("bridge"),
            graph_id="truth-test-continuity",
            parent_graph_id="parent-test-continuity",
            selected_parent_event_ids=["P-roster"],
            selected_parent_events=[parent_event],
            parent_max_global_time_index=144,
            parent_max_session_index=6,
        )
        unrelated = truth.example_content_bindings(local_skeleton)
        unrelated["discourse_frame"] = {
            "topic": "department membership access",
            "setting": "an office meeting",
        }
        report = truth.validate_content_bindings(local_skeleton, unrelated)
        self.assertFalse(report["pass"])
        self.assertIn(
            "parent_topic_anchor_not_realized",
            {issue["code"] for issue in report["issues"]},
        )

        related = truth.example_content_bindings(local_skeleton)
        related["discourse_frame"] = {
            "topic": "the Workshop Roster follow-up",
            "setting": "Maria's selection conversation",
        }
        materialized = truth.materialize_truth_skeleton(local_skeleton, related)
        first = materialized["graph"]["events"][0]
        self.assertEqual(first["depends_on_parent_event_ids"], ["P-roster"])
        self.assertEqual(
            first["parent_relation"], "modifies"
        )
        continuity = materialized["graph"]["metadata"]["parent_continuity"]
        self.assertFalse(continuity["entity_overlap_only"])
        self.assertEqual(continuity["continuity_score"], 1.0)
        self.assertTrue(continuity["semantic_link_compiled_locally"])
        self.assertEqual(
            continuity["continuity_relations"][0]["matched_features"],
            ["same_identity", "exact_slot", "child_after_parent_timeline"],
        )

        tampered = copy.deepcopy(related)
        anchor_binding = local_skeleton["parent_anchor_contract"][
            "continuity_match"
        ]["child_binding_id"]
        tampered["bindings"][anchor_binding] = "Archive Desk"
        tampered_report = truth.validate_content_bindings(
            local_skeleton, tampered
        )
        self.assertFalse(tampered_report["pass"])
        self.assertIn(
            "binding_enum_mismatch",
            {issue["code"] for issue in tampered_report["issues"]},
        )


class TruthConstructorMutationTest(unittest.TestCase):
    @staticmethod
    def invalid_value(value_type: str) -> object:
        if value_type == "Number":
            return "seven"
        if value_type == "YesNo":
            return "yes"
        if value_type == "Person":
            return 17
        if value_type == "DateOrTime":
            return "sometime eventually"
        return ["not", "an", "atom"]

    def test_every_binding_type_is_fail_closed_under_wrong_type_mutation(self) -> None:
        for operation in truth.OPERATIONS:
            local_skeleton = skeleton(operation)
            base_payload = truth.example_content_bindings(local_skeleton)
            for spec in local_skeleton["binding_specs"]:
                with self.subTest(operation=operation, binding_id=spec["binding_id"], value_type=spec["value_type"]):
                    payload = copy.deepcopy(base_payload)
                    payload["bindings"][spec["binding_id"]] = self.invalid_value(spec["value_type"])
                    result = truth.validate_content_bindings(local_skeleton, payload)
                    self.assertFalse(result["pass"])
                    self.assertIn(
                        ("binding_type_mismatch", spec["binding_id"]),
                        {(issue["code"], issue.get("binding_id")) for issue in result["issues"]},
                    )

    def test_location_or_scope_rejects_action_and_state_clauses(self) -> None:
        local_skeleton = skeleton("temporal_state_update")
        for invalid in (
            "amazed by the architecture and history",
            "secure another win",
            "able to jog without pain",
            "get more insights about countries",
            "visited the new gym",
            "moved into Cedar Hall",
            "works at Cedar Hall",
            "the gym was closed",
            "Cedar Hall is closed",
            "members can enter",
            "staff approved the request",
            "new gym opened yesterday",
        ):
            with self.subTest(invalid=invalid):
                payload = truth.example_content_bindings(local_skeleton)
                payload["bindings"]["temporal_candidate_1_location"] = invalid
                result = truth.validate_content_bindings(local_skeleton, payload)
                self.assertIn(
                    ("binding_type_mismatch", "temporal_candidate_1_location"),
                    {
                        (issue["code"], issue.get("binding_id"))
                        for issue in result["issues"]
                    },
                )

    def test_destination_accepts_locations_but_rejects_audience_scopes(self) -> None:
        local_skeleton = skeleton("temporal_state_update")
        payload = truth.example_content_bindings(local_skeleton)
        payload["bindings"].update(
            {
                "temporal_candidate_1_location": "Woodhaven",
                "temporal_candidate_2_location": "European coastline",
                "temporal_candidate_3_location": "new gym",
                "temporal_candidate_4_location": "castle in the UK",
            }
        )
        self.assertTrue(
            truth.validate_content_bindings(local_skeleton, payload)["pass"]
        )

        for valid in (
            "Midwest",
            "Cedar Hall",
            "downtown",
            "outdoors",
            "online",
            "abroad",
            "northern Europe",
            "western Midwest",
            "gorgeous forest",
            "studio near the park",
        ):
            with self.subTest(valid=valid):
                candidate = truth.example_content_bindings(local_skeleton)
                candidate["bindings"]["temporal_candidate_1_location"] = valid
                self.assertTrue(
                    truth.validate_content_bindings(local_skeleton, candidate)["pass"]
                )

        for invalid_scope in (
            "members only",
            "authorized users only",
            "registered guests",
            "public",
            "Private",
            "Restricted",
            "Internal",
            "Confidential",
        ):
            with self.subTest(invalid_scope=invalid_scope):
                candidate = truth.example_content_bindings(local_skeleton)
                binding_id = "temporal_candidate_1_location"
                candidate["bindings"][binding_id] = invalid_scope
                result = truth.validate_content_bindings(local_skeleton, candidate)
                self.assertIn(
                    (
                        "binding_semantic_type_mismatch",
                        binding_id,
                        "Location",
                    ),
                    {
                        (
                            issue["code"],
                            issue.get("binding_id"),
                            issue.get("required_semantic_value_type"),
                        )
                        for issue in result["issues"]
                    },
                )
                issue = next(
                    issue
                    for issue in result["issues"]
                    if issue.get("binding_id") == binding_id
                )
                self.assertIn(
                    "AudienceScope", issue["detected_semantic_value_types"]
                )

    def test_access_class_atoms_never_become_locations_by_capitalization(self) -> None:
        variants = [
            skeleton_with_answer_type(operation, "LocationOrScope")
            for operation in (
                "temporal_state_update",
                "bridge",
                "cause_consequence",
                "relation_lookup",
            )
        ]
        access_classes = (
            "Private",
            "PUBLIC",
            "Restricted",
            "internal",
            "Confidential",
            "Protected",
            "Classified",
            "Secret",
        )
        tested = 0
        for local_skeleton in variants:
            for binding_id in value_binding_ids_for_slot(
                local_skeleton, "destination"
            ):
                for access_class in access_classes:
                    tested += 1
                    with self.subTest(
                        operation=local_skeleton["operation"],
                        binding_id=binding_id,
                        access_class=access_class,
                    ):
                        payload = truth.example_content_bindings(local_skeleton)
                        payload["bindings"][binding_id] = access_class
                        result = truth.validate_content_bindings(
                            local_skeleton, payload
                        )
                        issue = next(
                            issue
                            for issue in result["issues"]
                            if issue.get("binding_id") == binding_id
                            and issue["code"]
                            == "binding_semantic_type_mismatch"
                        )
                        self.assertEqual(
                            issue["required_semantic_value_type"], "Location"
                        )
                        self.assertEqual(
                            issue["detected_semantic_value_types"],
                            ["AudienceScope"],
                        )
        self.assertGreater(tested, 0)

        local_skeleton = skeleton("temporal_state_update")
        binding_id = "temporal_candidate_1_location"
        for named_location in (
            "Woodhaven",
            "London",
            "Midwest",
            "UK",
            "Lake Geneva",
            "São Paulo",
            "Cedar Hall",
        ):
            with self.subTest(named_location=named_location):
                payload = truth.example_content_bindings(local_skeleton)
                payload["bindings"][binding_id] = named_location
                self.assertTrue(
                    truth.validate_content_bindings(local_skeleton, payload)[
                        "pass"
                    ]
                )

    def test_arbitrary_capitalized_named_entities_are_not_locations(self) -> None:
        variants = [
            skeleton_with_answer_type(operation, "LocationOrScope")
            for operation in (
                "temporal_state_update",
                "bridge",
                "cause_consequence",
                "relation_lookup",
            )
        ]
        invalid_named_entities = ("John", "Nike", "Dune", "Project Aurora")
        tested = 0
        for local_skeleton in variants:
            for binding_id in value_binding_ids_for_slot(
                local_skeleton, "destination"
            ):
                for invalid in invalid_named_entities:
                    tested += 1
                    with self.subTest(
                        operation=local_skeleton["operation"],
                        binding_id=binding_id,
                        invalid=invalid,
                    ):
                        payload = truth.example_content_bindings(local_skeleton)
                        payload["bindings"][binding_id] = invalid
                        result = truth.validate_content_bindings(
                            local_skeleton, payload
                        )
                        self.assertFalse(result["pass"])
                        self.assertTrue(
                            any(
                                issue["code"]
                                in {
                                    "binding_type_mismatch",
                                    "binding_semantic_type_mismatch",
                                }
                                and issue.get("binding_id") == binding_id
                                and issue.get("required_semantic_value_type")
                                == "Location"
                                for issue in result["issues"]
                            ),
                            result["issues"],
                        )
        self.assertGreater(tested, 0)

    def test_slot_refinements_are_derived_and_exposed_for_all_applicable_constructors(self) -> None:
        constructor_variants = [skeleton(operation) for operation in truth.OPERATIONS]
        for operation in (
            "temporal_state_update",
            "bridge",
            "cause_consequence",
            "relation_lookup",
        ):
            constructor_variants.extend(
                [
                    skeleton_with_answer_type(operation, "LocationOrScope"),
                    skeleton_with_answer_type(operation, "Event"),
                ]
            )

        seen = {"Location": 0, "EventName": 0}
        for local_skeleton in constructor_variants:
            specs = {
                spec["binding_id"]: spec
                for spec in local_skeleton["binding_specs"]
            }
            contract = truth.api_binding_contract(local_skeleton)
            for slot, semantic_value_type in (
                ("destination", "Location"),
                ("scheduled_event", "EventName"),
            ):
                for binding_id in value_binding_ids_for_slot(local_skeleton, slot):
                    with self.subTest(
                        operation=local_skeleton["operation"],
                        slot=slot,
                        binding_id=binding_id,
                    ):
                        self.assertEqual(
                            specs[binding_id]["semantic_value_type"],
                            semantic_value_type,
                        )
                        self.assertEqual(
                            contract["bindings"][binding_id]["semantic_value_type"],
                            semantic_value_type,
                        )
                        self.assertEqual(
                            contract["bindings"][binding_id]["semantic_value_rule"],
                            truth.SEMANTIC_VALUE_RULES[semantic_value_type],
                        )
                        self.assertEqual(
                            contract["bindings"][binding_id]["value_type"],
                            semantic_value_type,
                        )
                        self.assertEqual(
                            contract["bindings"][binding_id]["base_value_type"],
                            specs[binding_id]["value_type"],
                        )
                        self.assertEqual(
                            contract["bindings"][binding_id]["semantic_value_examples"],
                            truth.SEMANTIC_VALUE_EXAMPLES[semantic_value_type],
                        )
                        self.assertIn(
                            f"SLOT-SPECIFIC TYPE {semantic_value_type}",
                            contract["output_schema"]["bindings"][binding_id],
                        )
                        seen[semantic_value_type] += 1
        self.assertGreater(seen["Location"], 0)
        self.assertGreater(seen["EventName"], 0)

    def test_every_destination_binding_rejects_scope_mutations(self) -> None:
        variants = [
            skeleton_with_answer_type(operation, "LocationOrScope")
            for operation in (
                "temporal_state_update",
                "bridge",
                "cause_consequence",
                "relation_lookup",
            )
        ]
        for local_skeleton in variants:
            binding_ids = value_binding_ids_for_slot(local_skeleton, "destination")
            self.assertTrue(binding_ids)
            for binding_id in binding_ids:
                for invalid_scope in (
                    "members only",
                    "authorized users only",
                    "registered guests",
                ):
                    with self.subTest(
                        operation=local_skeleton["operation"],
                        binding_id=binding_id,
                        invalid_scope=invalid_scope,
                    ):
                        payload = truth.example_content_bindings(local_skeleton)
                        payload["bindings"][binding_id] = invalid_scope
                        result = truth.validate_content_bindings(
                            local_skeleton, payload
                        )
                        self.assertIn(
                            (
                                "binding_semantic_type_mismatch",
                                binding_id,
                                "Location",
                            ),
                            {
                                (
                                    issue["code"],
                                    issue.get("binding_id"),
                                    issue.get("required_semantic_value_type"),
                                )
                                for issue in result["issues"]
                            },
                        )
                        with self.assertRaises(truth.TruthConstructionError):
                            truth.materialize_truth_skeleton(
                                local_skeleton, payload
                            )

                valid = truth.example_content_bindings(local_skeleton)
                valid["bindings"][binding_id] = "castle in the UK"
                self.assertTrue(
                    truth.validate_content_bindings(local_skeleton, valid)["pass"]
                )

    def test_every_scheduled_event_binding_rejects_outcome_and_status_mutations(self) -> None:
        variants = [skeleton(operation) for operation in truth.OPERATIONS]
        variants.extend(
            skeleton_with_answer_type(operation, "Event")
            for operation in (
                "temporal_state_update",
                "bridge",
                "cause_consequence",
                "relation_lookup",
            )
        )
        tested = 0
        for local_skeleton in variants:
            for binding_id in value_binding_ids_for_slot(
                local_skeleton, "scheduled_event"
            ):
                tested += 1
                for invalid, detected_type in (
                    ("faced a setback", "EventOutcome"),
                    ("schedule conflict", "EventOutcome"),
                    ("was postponed", "StatusPhrase"),
                    ("currently canceled", "StatusPhrase"),
                ):
                    with self.subTest(
                        operation=local_skeleton["operation"],
                        binding_id=binding_id,
                        invalid=invalid,
                    ):
                        payload = truth.example_content_bindings(local_skeleton)
                        payload["bindings"][binding_id] = invalid
                        result = truth.validate_content_bindings(
                            local_skeleton, payload
                        )
                        issue = next(
                            issue
                            for issue in result["issues"]
                            if issue.get("binding_id") == binding_id
                            and issue["code"]
                            == "binding_semantic_type_mismatch"
                        )
                        self.assertEqual(
                            issue["required_semantic_value_type"], "EventName"
                        )
                        self.assertIn(
                            detected_type,
                            issue["detected_semantic_value_types"],
                        )
                        with self.assertRaises(truth.TruthConstructionError):
                            truth.materialize_truth_skeleton(
                                local_skeleton, payload
                            )

                for valid_event_name in (
                    "Autumn Symposium",
                    "gaming party",
                    "charity concert",
                ):
                    valid = truth.example_content_bindings(local_skeleton)
                    valid["bindings"][binding_id] = valid_event_name
                    self.assertTrue(
                        truth.validate_content_bindings(local_skeleton, valid)["pass"],
                        (binding_id, valid_event_name),
                    )
        self.assertGreater(tested, 0)

    def test_every_event_binding_is_fail_closed_as_an_event_name(self) -> None:
        variants = [skeleton(operation) for operation in truth.OPERATIONS]
        variants.extend(
            skeleton_with_answer_type(operation, "Event")
            for operation in (
                "temporal_state_update",
                "bridge",
                "cause_consequence",
                "relation_lookup",
            )
        )
        tested = 0
        for local_skeleton in variants:
            for spec in local_skeleton["binding_specs"]:
                if spec["value_type"] != "Event":
                    continue
                tested += 1
                binding_id = spec["binding_id"]
                self.assertEqual(spec["semantic_value_type"], "EventName")
                for invalid in (
                    "faced a setback",
                    "currently canceled",
                    "members only",
                    "John",
                    "Nike",
                    "Dune",
                    "Project Aurora",
                ):
                    with self.subTest(
                        operation=local_skeleton["operation"],
                        binding_id=binding_id,
                        invalid=invalid,
                    ):
                        payload = truth.example_content_bindings(local_skeleton)
                        payload["bindings"][binding_id] = invalid
                        result = truth.validate_content_bindings(
                            local_skeleton, payload
                        )
                        self.assertIn(
                            (
                                "binding_semantic_type_mismatch",
                                binding_id,
                                "EventName",
                            ),
                            {
                                (
                                    issue["code"],
                                    issue.get("binding_id"),
                                    issue.get("required_semantic_value_type"),
                                )
                                for issue in result["issues"]
                            },
                        )
        self.assertGreater(tested, 0)

    def test_every_distinctness_group_rejects_duplicate_typed_values(self) -> None:
        for operation in truth.OPERATIONS:
            local_skeleton = skeleton(operation)
            base_payload = truth.example_content_bindings(local_skeleton)
            grouped: dict[str, list[str]] = {}
            for spec in local_skeleton["binding_specs"]:
                if spec.get("distinct_group"):
                    grouped.setdefault(spec["distinct_group"], []).append(spec["binding_id"])
            for group, binding_ids in grouped.items():
                if len(binding_ids) < 2:
                    continue
                with self.subTest(operation=operation, distinct_group=group):
                    payload = copy.deepcopy(base_payload)
                    payload["bindings"][binding_ids[1]] = payload["bindings"][binding_ids[0]]
                    result = truth.validate_content_bindings(local_skeleton, payload)
                    self.assertFalse(result["pass"])
                    self.assertIn(
                        group,
                        {
                            issue.get("distinct_group")
                            for issue in result["issues"]
                            if issue["code"] == "binding_distinctness_violation"
                        },
                    )

    def test_forbidden_or_unknown_payload_fields_are_rejected_not_ignored(self) -> None:
        local_skeleton = skeleton("bridge")
        payload = truth.example_content_bindings(local_skeleton)
        payload["events"] = [{"slot": "discovery_date", "value": "Lila"}]
        payload["answer_slot"] = "discovery_date"
        payload["gold"] = "Lila"
        payload["proof"] = {"confirmation": ["Mira", "Eli", "Rhea"]}
        payload["bindings"]["confirmation"] = "Mira"
        result = truth.validate_content_bindings(local_skeleton, payload)
        self.assertFalse(result["pass"])
        issues = {issue["code"]: issue for issue in result["issues"]}
        self.assertEqual(
            set(issues["forbidden_binding_payload_keys"]["keys"]),
            {"answer_slot", "events", "gold", "proof"},
        )
        self.assertIn("confirmation", issues["unknown_bindings"]["binding_ids"])

    def test_missing_comparison_bindings_returns_failure_report_not_exception(self) -> None:
        local_skeleton = skeleton("comparison")
        payload = truth.example_content_bindings(local_skeleton)
        del payload["bindings"]["left_value"]
        result = truth.validate_content_bindings(local_skeleton, payload)
        self.assertFalse(result["pass"])
        self.assertIn("missing_bindings", {issue["code"] for issue in result["issues"]})

    def test_random_binding_deletions_never_materialize_partial_truth(self) -> None:
        rng = random.Random(20260715)
        for operation in truth.OPERATIONS:
            local_skeleton = skeleton(operation)
            binding_ids = [spec["binding_id"] for spec in local_skeleton["binding_specs"]]
            for _ in range(5):
                payload = truth.example_content_bindings(local_skeleton)
                removed = rng.choice(binding_ids)
                del payload["bindings"][removed]
                with self.subTest(operation=operation, removed=removed):
                    result = truth.validate_content_bindings(local_skeleton, payload)
                    self.assertFalse(result["pass"])
                    with self.assertRaises(truth.TruthConstructionError):
                        truth.materialize_truth_skeleton(local_skeleton, payload)


class TruthConstructorOperationInvariantTest(unittest.TestCase):
    def materialized(self, operation: str) -> dict[str, object]:
        local_skeleton = skeleton(operation)
        return truth.materialize_truth_skeleton(
            local_skeleton,
            truth.example_content_bindings(local_skeleton),
        )

    def test_temporal_gold_is_status_gated_not_max_time(self) -> None:
        materialized = self.materialized("temporal_state_update")
        events = by_event_id(materialized)
        proof = materialized["truth_manifest"]["proof_contract"]
        self.assertEqual(proof["gold_projection"]["event_id"], "E2")
        self.assertEqual(events["E2"]["status"], "active")
        self.assertEqual(events["E4"]["status"], "canceled")
        self.assertGreater(events["E4"]["time_index"], events["E2"]["time_index"])
        self.assertEqual(proof["compiled_gold"], events["E2"]["value"])
        self.assertTrue(proof["forbid_max_time_shortcut"])

    def test_bridge_is_a_true_two_hop_join_and_question_cannot_name_answer_node(self) -> None:
        materialized = self.materialized("bridge")
        events = by_event_id(materialized)
        proof = materialized["truth_manifest"]["proof_contract"]
        question_contract = proof["question_contract"]
        self.assertEqual(events["E1"]["value"], events["E2"]["entity"])
        self.assertEqual(proof["reasoning_event_ids"], ["E1", "E2"])
        self.assertEqual(proof["gold_source_event_id"], "E2")
        self.assertEqual(proof["compiled_gold"], events["E2"]["value"])
        self.assertEqual(events["E3"]["depends_on"], ["E1", "E2"])
        self.assertEqual(question_contract["name_event_ids"], ["E1"])
        self.assertIn("E2", question_contract["forbidden_surface_event_ids"])
        self.assertEqual(
            set(question_contract["must_not_name_binding_ids"]),
            {"bound_person", "bridge_answer"},
        )
        self.assertNotIn("E2", question_contract["name_event_ids"])

    def test_comparison_gold_is_locally_computed_from_fixed_operands(self) -> None:
        materialized = self.materialized("comparison")
        events = by_event_id(materialized)
        proof = materialized["truth_manifest"]["proof_contract"]
        primary = abs(events["E1"]["value"] - events["E2"]["value"])
        historical = abs(events["E3"]["value"] - events["E4"]["value"])
        self.assertEqual(proof["compiled_gold"], primary)
        self.assertEqual(abs(primary - historical), 1)
        self.assertEqual(events["E1"]["slot"], events["E2"]["slot"])
        self.assertEqual(events["E1"]["time_index"], events["E2"]["time_index"])

    def test_visibility_preserves_typed_roles_and_unique_signatures(self) -> None:
        materialized = self.materialized("visibility")
        events = by_event_id(materialized)
        permissions = [events[f"E{i}"] for i in range(2, 9)]
        signatures = {
            (
                event["observer"],
                event["action"],
                event["resource"],
                event["condition"],
                event["status"],
            )
            for event in permissions
        }
        self.assertEqual(len(signatures), len(permissions))
        self.assertTrue(all(event["event_type"] == "visibility_permission" for event in permissions))
        self.assertTrue(all(event["scope"]["observer"] == event["observer"] for event in permissions))
        self.assertTrue(all(event["scope"]["permission"] == event["action"] for event in permissions))
        proof = materialized["truth_manifest"]["proof_contract"]
        self.assertEqual(proof["compiled_gold"], events["E2"]["observer"])

    def test_count_uses_explicit_membership_and_is_order_independent(self) -> None:
        materialized = self.materialized("count")
        events = materialized["graph"]["events"]
        original_flags = [event["set_membership"]["included"] for event in events]
        self.assertEqual(original_flags, [True, False, True, True, False, True, False, False])
        self.assertNotEqual(original_flags, sorted(original_flags, reverse=True))
        expected = sum(event["set_membership"]["included"] is True for event in events)
        shuffled = copy.deepcopy(events)
        random.Random(17).shuffle(shuffled)
        shuffled_gold = sum(event["set_membership"]["included"] is True for event in shuffled)
        proof = materialized["truth_manifest"]["proof_contract"]
        self.assertEqual(expected, 4)
        self.assertEqual(shuffled_gold, expected)
        self.assertEqual(proof["compiled_gold"], expected)
        self.assertEqual(proof["membership_source"], "events[*].set_membership.included")
        self.assertTrue(proof["array_position_is_truth_forbidden"])

    def test_cause_has_one_directed_cause_to_effect_edge_and_text_evidence_claim(self) -> None:
        materialized = self.materialized("cause_consequence")
        events = by_event_id(materialized)
        manifest = materialized["truth_manifest"]
        required = manifest["proof_contract"]["required_relation"]
        self.assertEqual(
            required,
            {"predicate": "caused", "source_event_id": "E1", "target_event_id": "E2"},
        )
        self.assertEqual(events["E2"]["depends_on"], ["E1"])
        self.assertNotIn("E2", events["E1"]["depends_on"])
        cause_claims = [claim for claim in manifest["relation_claims"] if claim["predicate"] == "caused"]
        self.assertEqual(len(cause_claims), 1)
        self.assertEqual(cause_claims[0]["subject_event_id"], "E2")
        self.assertEqual(cause_claims[0]["object_event_ids"], ["E1"])
        canonical = cause_claims[0]["canonical_support_claim"]
        self.assertEqual(canonical["predicate"], "caused")
        self.assertEqual(
            canonical["direction"],
            {"source_event_ids": ["E1"], "target_event_id": "E2"},
        )
        self.assertIn("Because ", cause_claims[0]["support_realization_hint"])
        self.assertIs(cause_claims[0]["surface_realization_api_writable"], True)

    def test_relation_lookup_binds_subject_before_reading_fixed_relation(self) -> None:
        materialized = self.materialized("relation_lookup")
        events = by_event_id(materialized)
        proof = materialized["truth_manifest"]["proof_contract"]
        self.assertEqual(events["E1"]["value"], events["E2"]["entity"])
        self.assertEqual(events["E2"]["depends_on"], ["E1"])
        self.assertEqual(proof["reasoning_event_ids"], ["E1", "E2"])
        self.assertEqual(proof["compiled_gold"], events["E2"]["value"])

    def test_run229_failure_shapes_are_not_expressible_as_api_truth(self) -> None:
        local_skeleton = skeleton("bridge")
        fixed_slots = {event["slot"] for event in local_skeleton["event_templates"]}
        self.assertNotIn("discovery_date", fixed_slots)
        self.assertNotIn("confirmation", fixed_slots)

        payload = truth.example_content_bindings(local_skeleton)
        # Lila is a valid Person binding when used in the locally assigned role.
        payload["bindings"]["bridge_answer"] = "Lila"
        self.assertTrue(truth.validate_content_bindings(local_skeleton, payload)["pass"])
        materialized = truth.materialize_truth_skeleton(local_skeleton, payload)
        events = by_event_id(materialized)
        self.assertEqual(events["E2"]["value"], "Lila")
        self.assertNotEqual(events["E2"]["slot"], "discovery_date")

        forged = copy.deepcopy(payload)
        forged["answer_slot"] = "discovery_date"
        forged["events"] = [
            {"event_id": "E2", "slot": "discovery_date", "value": "Lila"},
            {"event_id": "E4", "slot": "confirmation", "value": "Mira"},
        ]
        forged["bindings"]["confirmation"] = "Eli"
        result = truth.validate_content_bindings(local_skeleton, forged)
        self.assertFalse(result["pass"])
        self.assertEqual(
            {issue["code"] for issue in result["issues"]},
            {"forbidden_binding_payload_keys", "unknown_bindings"},
        )
        with self.assertRaises(truth.TruthConstructionError):
            truth.materialize_truth_skeleton(local_skeleton, forged)


if __name__ == "__main__":
    unittest.main()
