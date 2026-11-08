from __future__ import annotations

import copy
import unittest

from scripts.current_method import stage3_claim_ir as subject


CANONICAL_CLAIM_FIELDS = {
    "claim_id",
    "claim_type",
    "subject_event_id",
    "predicate",
    "object_event_ids",
    "edge_role",
    "qa_eligible",
    "polarity",
    "status",
    "time_index",
    "global_time_index",
    "condition",
    "scope",
    "source_event_ids",
    "canonical_support_claim",
    "truth_signature",
}


def event(
    event_id: str,
    *,
    entity: str = "Atlas",
    slot: str = "access_scope",
    value="Internal",
    status: str = "active",
    time_index: int = 1,
    **extra,
) -> dict:
    return {
        "event_instance_id": event_id,
        "event_id": event_id,
        "entity": entity,
        "slot": slot,
        "value": value,
        "status": status,
        "time_index": time_index,
        "required_for_dialogue": True,
        "dialogue_support_claim": f"{entity}'s {slot} is {value}",
        **extra,
    }


def temporal_plan() -> dict:
    return {
        "schema_version": "semantic_graph_v2",
        "graph_id": "graph-1",
        "graph_snapshot_hash": "existing-graph-hash",
        "operation": "temporal_state_update",
        "metadata": {"frozen_graph": True, "keep_me": {"nested": [1, 2]}},
        "events": [
            event(
                "E1",
                value="Internal",
                status="outdated",
                time_index=1,
                global_time_index=41,
                unknown_source_field={"must": "survive"},
            ),
            event(
                "E2",
                value="Public",
                status="active",
                time_index=2,
                global_time_index=42,
                previous_value="Internal",
                depends_on=["E1"],
            ),
        ],
    }


class Stage3ClaimIRTests(unittest.TestCase):
    def test_compile_is_non_mutating_preserves_plan_and_emits_canonical_schema(self) -> None:
        plan = temporal_plan()
        original = copy.deepcopy(plan)

        bundle = subject.compile_claim_ir(plan)

        self.assertEqual(plan, original)
        self.assertEqual(bundle["current_event_plan"], original)
        self.assertIsNot(bundle["current_event_plan"], plan)
        self.assertEqual(
            bundle["current_event_plan"]["events"][0]["unknown_source_field"],
            {"must": "survive"},
        )
        self.assertEqual(bundle["schema_version"], subject.SCHEMA_VERSION)
        self.assertEqual(bundle["source_event_count"], 2)
        self.assertEqual(bundle["graph_snapshot_hash"], "existing-graph-hash")
        self.assertEqual(bundle["claim_bundle_hash"], bundle["frozen_claims_hash"])
        self.assertEqual(bundle["claim_bundle_hash"], bundle["frozen_graph_hash"])
        self.assertEqual(bundle["claim_bundle_hash"], bundle["frozen_hash"])

        for claim in bundle["claims"]:
            self.assertTrue(CANONICAL_CLAIM_FIELDS.issubset(claim), claim)
            self.assertRegex(claim["claim_id"], r"^clm_[0-9a-f]{32}$")
            self.assertEqual(
                claim["truth_signature_hash"],
                subject.canonical_json_sha256(claim["truth_signature"]),
            )
            self.assertIn(claim["edge_role"], subject.EDGE_ROLES)
            if claim["edge_role"] == "control":
                self.assertFalse(claim["qa_eligible"])

        nodes = [claim for claim in bundle["claims"] if claim["claim_type"] == "node"]
        relations = [
            claim for claim in bundle["claims"] if claim["claim_kind"] == "relation"
        ]
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(not claim["qa_eligible"] for claim in nodes))
        self.assertTrue(all(not claim["qa_eligible"] for claim in relations))
        self.assertTrue(all(not claim["answer_bearing"] for claim in relations))
        self.assertEqual(nodes[0]["entity"], "Atlas")
        self.assertEqual(nodes[0]["slot"], "access_scope")
        self.assertEqual(nodes[0]["value"], "Internal")
        self.assertEqual(nodes[0]["source_event"]["unknown_source_field"], {"must": "survive"})

    def test_current_alignment_binds_exact_node_ledger_and_controls_default_eligibility(self) -> None:
        plan = temporal_plan()
        alignment = {
            "E1": {
                "turn_ids": ["D2:4"],
                "support_quotes": ["Atlas used to have Internal access."],
            },
            "E2": {
                "turn_ids": ["D2:9"],
                "support_quotes": ["Atlas now has Public access."],
            },
        }
        dialogue = {
            "turns": [
                {"turn_id": "D2:4", "text": "For context, Atlas used to have Internal access."},
                {"turn_id": "D2:9", "text": "Confirmed: Atlas now has Public access."},
            ]
        }
        alignment_before = copy.deepcopy(alignment)
        dialogue_before = copy.deepcopy(dialogue)

        bundle = subject.compile_claim_ir(
            plan,
            current_alignment=alignment,
            dialogue_turns=dialogue,
        )
        pre_dialogue = subject.compile_claim_ir(plan)

        self.assertEqual(alignment, alignment_before)
        self.assertEqual(dialogue, dialogue_before)
        self.assertEqual(bundle["claim_bundle_hash"], pre_dialogue["claim_bundle_hash"])
        self.assertEqual(
            {claim["claim_id"] for claim in bundle["claims"]},
            {claim["claim_id"] for claim in pre_dialogue["claims"]},
        )
        self.assertNotEqual(
            bundle["support_ledger_hash"], pre_dialogue["support_ledger_hash"]
        )
        self.assertNotEqual(
            bundle["validated_bundle_hash"], pre_dialogue["validated_bundle_hash"]
        )
        nodes = [claim for claim in bundle["claims"] if claim["claim_type"] == "node"]
        self.assertTrue(all(claim["qa_eligible"] for claim in nodes))
        ledger = bundle["support_ledger"]
        for claim in nodes:
            entry = ledger["by_claim_id"][claim["claim_id"]]
            source_id = claim["subject_event_id"]
            self.assertEqual(entry["turn_id"], alignment[source_id]["turn_ids"][0])
            self.assertEqual(
                entry["exact_support_quote"], alignment[source_id]["support_quotes"][0]
            )
            self.assertEqual(entry["verification_status"], "exact_quote_bound")

        # A byte-for-byte mismatch is not accepted as an exact ledger binding.
        broken = copy.deepcopy(dialogue)
        broken["turns"][1]["text"] = "Atlas now has private access."
        broken_bundle = subject.compile_claim_ir(
            plan,
            current_alignment=alignment,
            dialogue_turns=broken,
        )
        broken_e2 = next(
            claim
            for claim in broken_bundle["claims"]
            if claim["claim_type"] == "node" and claim["subject_event_id"] == "E2"
        )
        self.assertFalse(broken_e2["qa_eligible"])
        self.assertIsNone(
            broken_bundle["support_ledger"]["by_claim_id"][broken_e2["claim_id"]][
                "exact_support_quote"
            ]
        )

    def test_dependency_is_control_for_temporal_but_semantic_for_bridge(self) -> None:
        plan = temporal_plan()
        temporal_claims = subject.extract_claims(plan)
        dependency = next(
            claim for claim in temporal_claims if claim["claim_type"] == "dependency"
        )
        transition = next(
            claim
            for claim in temporal_claims
            if claim["claim_type"] == "status_transition"
        )
        self.assertEqual(dependency["subject_event_id"], "E2")
        self.assertEqual(dependency["object_event_ids"], ["E1"])
        self.assertEqual(dependency["predicate"], "depends_on")
        self.assertEqual(dependency["edge_role"], "control")
        self.assertFalse(dependency["qa_eligible"])
        self.assertEqual(transition["subject_event_id"], "E1")
        self.assertEqual(transition["object_event_ids"], ["E2"])
        self.assertEqual(transition["edge_role"], "semantic")
        self.assertTrue(transition["qa_candidate"])
        self.assertFalse(transition["qa_eligible"])
        self.assertFalse(transition["answer_bearing"])
        self.assertEqual(transition["predicate"], "transitions_to")
        self.assertEqual(
            transition["canonical_support_claim"]["predicate"],
            transition["predicate"],
        )

        bridge = copy.deepcopy(plan)
        bridge["operation"] = "bridge"
        bridge_dependency = next(
            claim
            for claim in subject.extract_claims(bridge)
            if claim["claim_type"] == "dependency"
        )
        self.assertEqual(bridge_dependency["edge_role"], "semantic")
        self.assertTrue(bridge_dependency["qa_candidate"])
        self.assertFalse(bridge_dependency["qa_eligible"])
        self.assertFalse(bridge_dependency["answer_bearing"])

    def test_control_role_always_forces_qa_ineligible(self) -> None:
        plan = {
            "operation": "bridge",
            "events": [
                event("E1"),
                event(
                    "E2",
                    value="External",
                    time_index=2,
                    depends_on=[
                        {"event_id": "E1", "edge_role": "control", "qa_eligible": True}
                    ],
                ),
            ],
        }

        dependency = next(
            claim
            for claim in subject.extract_claims(plan)
            if claim["claim_type"] == "dependency"
        )
        self.assertEqual(dependency["edge_role"], "control")
        self.assertFalse(dependency["qa_eligible"])

    def test_membership_and_visibility_relations_are_typed_and_directional(self) -> None:
        plan = {
            "operation": "visibility",
            "events": [
                event(
                    "E1",
                    entity="archive",
                    slot="access_class",
                    value="private",
                    set_membership={
                        "set_id": "approved_resources",
                        "included": True,
                        "condition": "after review",
                    },
                    scope={
                        "observer": "Maya",
                        "permission": "view",
                        "condition": "after 6 PM",
                    },
                    owner="Nate",
                    resource="gaming archive",
                    observer="Maya",
                    action="view",
                    access_class="private",
                ),
                event(
                    "E2",
                    entity="archive",
                    slot="access_class",
                    value="draft cabinet",
                    time_index=2,
                    set_membership={
                        "set_id": "approved_resources",
                        "included": False,
                        "condition": "after review",
                    },
                ),
                event(
                    "E3",
                    entity="archive",
                    slot="retention_status",
                    value="retained",
                    time_index=3,
                    condition="if the owner signs",
                ),
            ],
        }

        claims = subject.extract_claims(plan)
        nodes = {
            claim["subject_event_id"]: claim
            for claim in claims
            if claim["claim_type"] == "node"
        }
        memberships = [claim for claim in claims if claim["claim_type"] == "membership"]
        visibility = next(claim for claim in claims if claim["claim_type"] == "visibility")
        self.assertEqual(len(memberships), 2)
        included = next(claim for claim in memberships if claim["included"])
        excluded = next(claim for claim in memberships if not claim["included"])
        self.assertEqual(included["predicate"], "member_of")
        self.assertEqual(included["polarity"], "positive")
        self.assertEqual(excluded["predicate"], "not_member_of")
        self.assertEqual(excluded["polarity"], "negative")
        self.assertTrue(all(claim["edge_role"] == "semantic" for claim in memberships))
        for claim in memberships:
            self.assertTrue(claim["qa_candidate"])
            self.assertFalse(claim["qa_eligible"])
            self.assertFalse(claim["answer_bearing"])
            self.assertEqual(len(claim["object_event_ids"]), 1)
            self.assertNotEqual(
                claim["subject_event_id"], claim["object_event_ids"][0]
            )
            self.assertEqual(
                len(claim["object_event_ids"]),
                len(set(claim["object_event_ids"])),
            )
            self.assertTrue(
                set([claim["subject_event_id"], *claim["object_event_ids"]])
                <= set(claim["source_event_ids"])
            )
            self.assertEqual(claim["condition"], "after review")
        # A membership filter qualifies only the membership relation; it must
        # not contaminate the ordinary entity/slot/value node claim.
        self.assertIsNone(nodes["E1"]["condition"])
        self.assertIsNone(nodes["E1"]["canonical_support_claim"]["condition"])
        self.assertIsNone(nodes["E2"]["condition"])
        self.assertIsNone(nodes["E2"]["canonical_support_claim"]["condition"])
        # Direct node conditions remain intact.
        self.assertEqual(nodes["E3"]["condition"], "if the owner signs")
        self.assertEqual(
            nodes["E3"]["canonical_support_claim"]["condition"],
            "if the owner signs",
        )
        self.assertEqual(visibility["observer"], "Maya")
        self.assertEqual(visibility["predicate"], "view")
        self.assertEqual(visibility["resource"], "gaming archive")
        self.assertEqual(visibility["canonical_support_claim"]["subject"], "Maya")
        self.assertEqual(visibility["canonical_support_claim"]["objects"], ["gaming archive"])
        self.assertEqual(visibility["condition"], "after 6 PM")
        self.assertEqual(
            nodes["E1"]["scope"],
            {
                "observer": "Maya",
                "permission": "view",
                "condition": "after 6 PM",
            },
        )
        self.assertEqual(nodes["E1"]["canonical_support_claim"]["scope"], nodes["E1"]["scope"])
        self.assertEqual(visibility["scope"], nodes["E1"]["scope"])
        self.assertEqual(len(visibility["object_event_ids"]), 1)
        self.assertNotEqual(
            visibility["subject_event_id"], visibility["object_event_ids"][0]
        )
        self.assertFalse(visibility["qa_eligible"])

    def test_cause_relation_reverses_dependency_into_cause_to_effect_direction(self) -> None:
        plan = {
            "operation": "cause_consequence",
            "events": [
                event("CAUSE", entity="storm", slot="severity", value="high", time_index=3),
                event(
                    "EFFECT",
                    entity="ferry",
                    slot="service_status",
                    value="canceled",
                    time_index=4,
                    depends_on=["CAUSE"],
                ),
            ],
        }

        claims = subject.extract_claims(plan)
        dependency = next(claim for claim in claims if claim["claim_type"] == "dependency")
        cause = next(claim for claim in claims if claim["claim_type"] == "cause")
        self.assertEqual(dependency["subject_event_id"], "EFFECT")
        self.assertEqual(dependency["object_event_ids"], ["CAUSE"])
        self.assertEqual(dependency["edge_role"], "control")
        self.assertEqual(cause["subject_event_id"], "CAUSE")
        self.assertEqual(cause["object_event_ids"], ["EFFECT"])
        self.assertEqual(cause["predicate"], "causes")
        self.assertEqual(cause["edge_role"], "semantic")
        self.assertTrue(cause["qa_candidate"])
        self.assertFalse(cause["qa_eligible"])
        self.assertFalse(cause["answer_bearing"])

    def test_event_alignment_binds_membership_and_cause_relations_only(self) -> None:
        membership_plan = {
            "operation": "count",
            "events": [
                event(
                    "E1",
                    entity="Writers Guild",
                    slot="candidate_member",
                    value="Maya",
                    set_membership={
                        "set_id": "weekend roster",
                        "included": True,
                        "condition": "submitted a complete draft",
                    },
                )
            ],
        }
        membership_quote = (
            "Writers Guild candidate member Maya is a member of weekend roster "
            "under the condition submitted a complete draft."
        )
        membership_bundle = subject.compile_claim_ir(
            membership_plan,
            current_alignment={
                "E1": {
                    "turn_ids": ["D1:1"],
                    "support_quotes": [membership_quote],
                }
            },
            dialogue_turns={
                "turns": [{"turn_id": "D1:1", "text": membership_quote}]
            },
        )
        membership_claim = next(
            claim
            for claim in membership_bundle["claims"]
            if claim["claim_type"] == "membership"
        )
        self.assertTrue(membership_claim["qa_eligible"])
        self.assertEqual(
            membership_claim["support_binding_source"],
            "event_alignment:E1",
        )

        cause_plan = {
            "operation": "cause_consequence",
            "events": [
                event(
                    "CAUSE",
                    entity="Storm alert",
                    slot="trigger",
                    value="severe weather",
                    time_index=1,
                ),
                event(
                    "EFFECT",
                    entity="Workshop",
                    slot="replacement_host",
                    value="Maya",
                    time_index=2,
                    depends_on=["CAUSE"],
                ),
            ],
        }
        cause_quote = (
            "Storm alert trigger severe weather causes Workshop replacement "
            "host to be Maya."
        )
        cause_bundle = subject.compile_claim_ir(
            cause_plan,
            current_alignment={
                "EFFECT": {
                    "turn_ids": ["D1:2"],
                    "support_quotes": [cause_quote],
                }
            },
            dialogue_turns={
                "turns": [{"turn_id": "D1:2", "text": cause_quote}]
            },
        )
        cause_claim = next(
            claim
            for claim in cause_bundle["claims"]
            if claim["claim_type"] == "cause"
        )
        self.assertTrue(cause_claim["qa_eligible"])
        self.assertEqual(
            cause_claim["support_binding_source"],
            "event_alignment:EFFECT",
        )
        dependency = next(
            claim
            for claim in cause_bundle["claims"]
            if claim["claim_type"] == "dependency"
        )
        self.assertFalse(dependency["qa_eligible"])
        self.assertIsNone(dependency["support_binding_source"])

    def test_comparison_is_locally_derived_from_unique_numeric_pair(self) -> None:
        plan = {
            "operation": "comparison",
            "events": [
                event("B", entity="Beta", slot="score", value=7, time_index=9),
                event("A", entity="Alpha", slot="score", value=11, time_index=9),
                event("D", entity="Distractor", slot="other_score", value=100, time_index=9),
            ],
        }

        comparison = next(
            claim
            for claim in subject.extract_claims(plan)
            if claim["claim_type"] == "comparison"
        )
        self.assertEqual(comparison["subject_event_id"], "A")
        self.assertEqual(comparison["object_event_ids"], ["B"])
        self.assertEqual(comparison["predicate"], "absolute_difference")
        self.assertEqual(comparison["derived_value"], 4)
        self.assertTrue(comparison["qa_candidate"])
        self.assertFalse(comparison["qa_eligible"])
        self.assertFalse(comparison["answer_bearing"])
        self.assertEqual(
            comparison["canonical_support_claim"]["absolute_difference"], 4
        )

    def test_claim_ids_and_bundle_hash_are_content_addressed_and_order_independent(self) -> None:
        plan = temporal_plan()
        first = subject.compile_claim_ir(plan)
        second_plan = copy.deepcopy(plan)
        second_plan["events"].reverse()
        second = subject.compile_claim_ir(second_plan)

        self.assertEqual(
            [claim["claim_id"] for claim in first["claims"]],
            [claim["claim_id"] for claim in second["claims"]],
        )
        self.assertEqual(first["claim_bundle_hash"], second["claim_bundle_hash"])
        self.assertNotEqual(first["source_plan_hash"], second["source_plan_hash"])

        surface_only = copy.deepcopy(plan)
        surface_only["events"][0]["dialogue_support_claim"] = "A nicer surface form."
        surface_bundle = subject.compile_claim_ir(surface_only)
        self.assertEqual(first["claim_bundle_hash"], surface_bundle["claim_bundle_hash"])
        self.assertNotEqual(first["source_plan_hash"], surface_bundle["source_plan_hash"])

        truth_change = copy.deepcopy(plan)
        truth_change["events"][0]["status"] = "false"
        truth_bundle = subject.compile_claim_ir(truth_change)
        self.assertNotEqual(first["claim_bundle_hash"], truth_bundle["claim_bundle_hash"])

    def test_relation_requires_claim_addressed_quote_not_implicit_node_quotes(self) -> None:
        plan = temporal_plan()
        node_alignment = {
            "E1": {"turn_ids": ["D1:1"], "support_quotes": ["Old was Internal."]},
            "E2": {"turn_ids": ["D1:2"], "support_quotes": ["Now it is Public."]},
        }
        first = subject.compile_claim_ir(plan, current_alignment=node_alignment)
        transition = next(
            claim
            for claim in first["claims"]
            if claim["claim_type"] == "status_transition"
        )
        transition_entry = first["support_ledger"]["by_claim_id"][transition["claim_id"]]
        self.assertIsNone(transition_entry["exact_support_quote"])
        self.assertTrue(transition["qa_candidate"])
        self.assertFalse(transition["qa_eligible"])
        self.assertFalse(transition["answer_bearing"])

        relation_alignment = copy.deepcopy(node_alignment)
        relation_alignment[transition["claim_id"]] = {
            "turn_id": "D1:3",
            "exact_support_quote": "Atlas changed from Internal to Public.",
        }
        second = subject.compile_claim_ir(plan, current_alignment=relation_alignment)
        second_transition = next(
            claim
            for claim in second["claims"]
            if claim["claim_type"] == "status_transition"
        )
        entry = second["support_ledger"]["by_claim_id"][second_transition["claim_id"]]
        self.assertEqual(entry["turn_id"], "D1:3")
        self.assertEqual(
            entry["exact_support_quote"], "Atlas changed from Internal to Public."
        )
        self.assertEqual(second_transition["claim_id"], transition["claim_id"])
        self.assertTrue(second_transition["qa_candidate"])
        self.assertTrue(second_transition["qa_eligible"])
        self.assertTrue(second_transition["answer_bearing"])
        self.assertEqual(first["claim_bundle_hash"], second["claim_bundle_hash"])
        self.assertNotEqual(
            first["validated_bundle_hash"], second["validated_bundle_hash"]
        )

    def test_bind_support_evidence_returns_copy_and_preserves_exact_quote(self) -> None:
        bundle = subject.compile_claim_ir(temporal_plan())
        ledger = bundle["support_ledger"]
        original = copy.deepcopy(ledger)
        claim_id = bundle["claims"][0]["claim_id"]
        exact_quote = "  Keep this spacing exactly.  "

        bound = subject.bind_support_evidence(
            ledger,
            claim_id=claim_id,
            turn_id="D8:2",
            exact_support_quote=exact_quote,
        )

        self.assertEqual(ledger, original)
        self.assertEqual(bound["by_claim_id"][claim_id]["turn_id"], "D8:2")
        self.assertEqual(
            bound["by_claim_id"][claim_id]["exact_support_quote"], exact_quote
        )
        self.assertEqual(
            bound["by_claim_id"][claim_id]["verification_status"],
            "pending_semantic_validation",
        )
        self.assertNotEqual(
            bound["support_ledger_hash"], ledger["support_ledger_hash"]
        )

    def test_attach_claim_ir_preserves_top_level_fields_without_mutation(self) -> None:
        plan = temporal_plan()
        original = copy.deepcopy(plan)

        attached = subject.attach_claim_ir(plan)

        self.assertEqual(plan, original)
        for key, value in original.items():
            self.assertEqual(attached[key], value)
        self.assertIn("claim_ir", attached)
        self.assertNotIn("claim_ir", plan)

    def test_rejects_duplicate_ids_invalid_roles_and_non_json_data(self) -> None:
        duplicate = {"events": [event("E1"), event("E1", time_index=2)]}
        with self.assertRaisesRegex(subject.ClaimIRError, "duplicate event id"):
            subject.extract_claims(duplicate)

        invalid_role = {
            "events": [
                event("E1"),
                event(
                    "E2",
                    time_index=2,
                    depends_on=[{"event_id": "E1", "edge_role": "maybe"}],
                ),
            ]
        }
        with self.assertRaisesRegex(subject.ClaimIRError, "edge_role"):
            subject.extract_claims(invalid_role)

        non_json = temporal_plan()
        non_json["metadata"]["bad"] = {1, 2}
        with self.assertRaisesRegex(subject.ClaimIRError, "finite JSON"):
            subject.compile_claim_ir(non_json)


if __name__ == "__main__":
    unittest.main()
