from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import stage3_claim_proof_compiler as compiler  # noqa: E402
from scripts.current_method import stage3_claim_ir  # noqa: E402
from scripts.current_method import stage3_claim_semantic_validator  # noqa: E402
from scripts.current_method import stage64_quality  # noqa: E402


def node(
    claim_id: str,
    entity: str,
    slot: str,
    value: object,
    *,
    status: str = "active",
    time_index: int = 1,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_type": "node_claim",
        "subject_event_id": claim_id.replace("C", "E"),
        "entity": entity,
        "slot": slot,
        "value": value,
        "predicate": slot,
        "object_event_ids": [],
        "edge_role": "semantic",
        "qa_eligible": True,
        "polarity": "positive",
        "status": status,
        "global_time_index": time_index,
        "condition": None,
        "scope": None,
        "source_event_ids": [claim_id.replace("C", "E")],
        "canonical_support_claim": f"{entity}'s {slot} is {value} ({status})",
        "truth_signature": f"{entity}|{slot}|{value}|{status}",
    }


def bundle() -> dict[str, object]:
    claims = [
        node("C01", "Alice", "hobby", "chess", time_index=1),
        node("C02", "Alice", "job", "doctor", time_index=2),
        node("C03", "Alice", "city", "Paris", time_index=3),
        node("C04", "Bob", "hobby", "cycling", time_index=4),
        node("C05", "Bob", "job", "teacher", time_index=5),
        node("C06", "Bob", "city", "Rome", time_index=6),
        node("C07", "Alice", "service_hours", 25, time_index=9),
        node("C08", "Bob", "service_hours", 18, time_index=9),
        node("C09", "Alice", "service_hours", 12, status="expired", time_index=8),
        node("C10", "Bob", "service_hours", 14, status="canceled", time_index=8),
    ]
    ledger = [
        {
            "claim_id": item["claim_id"],
            "turn_id": f"D1:{index}",
            "exact_support_quote": item["canonical_support_claim"],
        }
        for index, item in enumerate(claims, start=1)
    ]
    return {
        "schema_version": "stage3_claim_bundle_v1",
        "graph_snapshot_hash": "graph_hash_123",
        "claim_bundle_hash": "claim_hash_456",
        "source_graph_operation": "bridge",
        "claims": claims,
        "support_ledger": ledger,
    }


def temporal_target_bundle() -> dict[str, object]:
    statuses = ("belief", "active", "outdated", "canceled")
    values = ("Paris", "Lyon", "Berlin", "Rome")
    claims = [
        node(
            f"C{index:02d}",
            "Maya",
            "meeting_city",
            value,
            status=status,
            time_index=index,
        )
        for index, (status, value) in enumerate(zip(statuses, values), start=1)
    ]
    events = [
        {
            "event_id": f"E{index:02d}",
            "event_instance_id": f"E{index:02d}",
            "entity": "Maya",
            "slot": "meeting_city",
            "value": value,
            "status": status,
            "time_index": index,
            "depends_on": [f"E{index - 1:02d}"] if index > 1 else [],
        }
        for index, (status, value) in enumerate(zip(statuses, values), start=1)
    ]
    return {
        "schema_version": "stage3_claim_bundle_v1",
        "graph_snapshot_hash": "temporal_graph_hash",
        "claim_bundle_hash": "temporal_claim_hash",
        "source_graph_operation": "temporal_state_update",
        "claims": claims,
        "support_ledger": [
            {
                "claim_id": item["claim_id"],
                "turn_id": f"D1:{index}",
                "exact_support_quote": item["canonical_support_claim"],
            }
            for index, item in enumerate(claims, start=1)
        ],
        "current_event_plan": {
            "operation": "temporal_state_update",
            "events": events,
        },
    }


def canceled_temporal_target() -> dict[str, object]:
    return {
        "qa_operation": "temporal_state_update",
        "preferred_answer_type": "LocationOrScope",
        "reasoning_subtype": "status_gated_canceled_update",
        "reasoning_template": "reject_canceled_value_then_resolve_update",
        "hop_count": 4,
        "hard_negative_strategy": "same_slot_canceled_value",
        "forbid_max_time_shortcut": True,
        "target_basis": "unit_test",
    }



def immutable_target(operation: str) -> dict[str, object]:
    specs = {
        "bridge": (
            "Person",
            "conjunctive_two_support_bridge",
            "join_two_support_facts",
            "one_hop_preserving_alternative",
        ),
        "comparison": (
            "Number",
            "same_slot_absolute_difference",
            "bind_pair_then_compute_difference",
            "near_numeric_off_by_one",
        ),
        "visibility": (
            "Person",
            "observer_permission_scope",
            "resolve_observer_permission_and_scope",
            "correct_fact_wrong_observer",
        ),
        "count": (
            "Number",
            "filtered_membership_count",
            "filter_members_then_count",
            "include_exclude_boundary_count",
        ),
        "cause_consequence": (
            "Person",
            "direct_causal_dependency",
            "trace_cause_to_effect",
            "temporally_adjacent_non_cause",
        ),
        "relation_lookup": (
            "Person",
            "entity_constrained_relation",
            "bind_entity_then_relation",
            "same_relation_wrong_entity",
        ),
    }
    answer_type, subtype, template, negative = specs[operation]
    return {
        "qa_operation": operation,
        "preferred_answer_type": answer_type,
        "reasoning_subtype": subtype,
        "reasoning_template": template,
        "hop_count": 2,
        "hard_negative_strategy": negative,
        "target_basis": "unit_test",
    }


def event(
    index: int,
    entity: str,
    slot: str,
    value: object,
    *,
    status: str = "active",
    time_index: int | None = None,
    depends_on: list[str] | None = None,
    **extra: object,
) -> dict[str, object]:
    event_id = f"E{index:02d}"
    return {
        "event_id": event_id,
        "event_instance_id": event_id,
        "entity": entity,
        "slot": slot,
        "value": value,
        "status": status,
        "time_index": index if time_index is None else time_index,
        "global_time_index": 100 + index,
        "depends_on": copy.deepcopy(depends_on or []),
        "required_for_dialogue": True,
        **copy.deepcopy(extra),
    }


def operation_plan(operation: str) -> dict[str, object]:
    if operation == "bridge":
        events = [
            event(1, "Workshop", "selected_person", "Maya"),
            event(2, "Maya", "festival_partner", "Noah"),
            event(3, "Bridge closure", "linked_record", "complete", depends_on=["E01", "E02"]),
            event(4, "Archive", "linked_record", "Ava"),
            event(5, "Studio", "festival_partner", "Liam"),
            event(6, "Panel", "festival_partner", "Zoe"),
        ]
    elif operation == "comparison":
        events = [
            event(1, "Maya", "service_hours", 12, time_index=9),
            event(2, "Noah", "service_hours", 7, time_index=9),
            event(3, "Maya", "service_hours", 10, status="outdated", time_index=4),
            event(4, "Noah", "service_hours", 6, status="outdated", time_index=4),
            event(5, "Ava", "score", 3, time_index=5),
            event(6, "Liam", "age", 31, time_index=6),
            event(7, "Zoe", "count", 2, time_index=7),
            event(8, "Omar", "duration_minutes", 45, time_index=8),
        ]
    elif operation == "count":
        values = ["Maya", "Noah", "Ava", "Liam", "Zoe", "Omar", "Iris", "Eli"]
        events = [
            event(
                index,
                "Writers Guild",
                "candidate_member",
                value,
                set_membership={
                    "set_id": "weekend workshop roster",
                    "condition": "submitted a complete draft",
                    "included": index <= 4,
                },
            )
            for index, value in enumerate(values, start=1)
        ]
    elif operation == "cause_consequence":
        events = [
            event(1, "Storm alert", "trigger", "severe weather", time_index=1),
            event(2, "Workshop", "replacement_host", "Maya", time_index=2, depends_on=["E01"]),
            event(3, "Panel", "replacement_host", "Ava", time_index=3),
            event(4, "Venue change", "contact", "Noah", time_index=4),
            event(5, "Panel", "speaker", "Liam", time_index=5),
            event(6, "Archive", "curator", "Zoe", time_index=6),
        ]
    elif operation == "relation_lookup":
        events = [
            event(1, "Roster", "selected_person", "Maya"),
            event(2, "Maya", "mentor", "Noah", depends_on=["E01"]),
            event(3, "Maya", "mentor", "Ava"),
            event(4, "Liam", "mentor", "Zoe"),
            event(5, "Omar", "mentor", "Iris"),
            event(6, "Studio", "location", "Paris"),
            event(7, "Workshop", "date", "Friday"),
            event(8, "Guild", "host", "Eli"),
        ]
    elif operation == "visibility":
        scopes = [
            {"observer": "Maya", "permission": "view", "condition": "after six PM"},
            {"observer": "Noah", "permission": "view", "condition": "after six PM"},
            {"observer": "Maya", "permission": "edit", "condition": "after six PM"},
            {"observer": "Maya", "permission": "view", "condition": "on weekends"},
            {"observer": "Ava", "permission": "view", "condition": "on weekends"},
            {"observer": "Liam", "permission": "edit", "condition": "after six PM"},
            {"observer": "Zoe", "permission": "view", "condition": "before noon"},
        ]
        events = [event(1, "Editing suite", "resource", "editing suite")]
        events.extend(
            event(
                index,
                "editing suite",
                "access_level",
                "Private",
                status="active" if index == 2 else "scope_limited",
                depends_on=["E01"] if index == 2 else [],
                scope=scope,
                observer=scope["observer"],
                action=scope["permission"],
                resource="editing suite",
                access_class="Private",
            )
            for index, scope in enumerate(scopes, start=2)
        )
    else:
        raise AssertionError(operation)
    return {
        "operation": operation,
        "graph_snapshot_id": f"fixture:{operation}",
        "graph_snapshot_hash": f"hash:{operation}",
        "events": events,
    }


class Stage3ClaimProofCompilerTest(unittest.TestCase):
    def test_compiles_three_to_six_frozen_qas(self) -> None:
        result = compiler.compile_claim_first_qas(
            bundle(), seed_sample_id="seed-1", min_qas=3, max_qas=6
        )
        self.assertGreaterEqual(result["qa_count"], 3)
        self.assertLessEqual(result["qa_count"], 6)
        self.assertEqual(result["api_calls_made"], 0)
        self.assertTrue(result["semantic_contract_frozen"])
        for qa in result["qas"]:
            self.assertTrue(qa["semantic_contract_frozen"])
            self.assertEqual(
                qa["operation_specific_target"]["qa_operation"], qa["qa_operation"]
            )
            self.assertEqual(
                qa["operation_specific_target"]["source_graph_operation"], "bridge"
            )
            self.assertEqual(
                qa["frozen_graph_snapshot_hash"], "graph_hash_123"
            )
            self.assertEqual(qa["frozen_claim_bundle_hash"], "claim_hash_456")
            proof_view = qa["operation_proof_view"]
            self.assertEqual(
                proof_view["source_graph_snapshot_hash"], "graph_hash_123"
            )
            self.assertEqual(proof_view["qa_operation"], qa["qa_operation"])
            self.assertTrue(proof_view["proof_sha256"])
            self.assertEqual(
                set(qa["evidence_claim_ids"]),
                {item["claim_id"] for item in qa["evidence_ledger"]},
            )

    def test_comparison_separates_operands_from_dependency_closure(self) -> None:
        result = compiler.compile_claim_first_qas(
            bundle(), seed_sample_id="seed-1", min_qas=3, max_qas=6
        )
        comparison = next(qa for qa in result["qas"] if qa["qa_operation"] == "comparison")
        proof = comparison["proof"]
        self.assertEqual(set(proof["gold_operand_claim_ids"]), {"C07", "C08"})
        self.assertEqual(
            set(proof["dependency_closure_claim_ids"]),
            {"C07", "C08", "C09", "C10"},
        )
        self.assertEqual(proof["gold_answer_text"], "7")
        self.assertEqual(proof["derivation"]["rule"], "absolute_difference")
        self.assertEqual(proof["derivation"]["answer_claim_ids"], [])
        true_option = next(
            item for item in comparison["options"] if item["truth_value"] is True
        )
        self.assertEqual(true_option["anchor_kind"], "derived_claim")
        self.assertEqual(true_option["source_claim_id"], proof["proof_id"])
        self.assertEqual(
            set(true_option["support_event_instance_ids"]), {"E07", "E08"}
        )

    def test_family_split_is_shared_by_every_qa_and_stable(self) -> None:
        first = compiler.compile_claim_first_qas(
            bundle(), seed_sample_id="seed-1", lineage_group_id="lineage-a"
        )
        second = compiler.compile_claim_first_qas(
            bundle(), seed_sample_id="seed-1", lineage_group_id="lineage-a"
        )
        self.assertEqual(first["dialogue_family_id"], second["dialogue_family_id"])
        self.assertEqual(first["dialogue_family_split"], second["dialogue_family_split"])
        self.assertEqual(
            {qa["dialogue_family_split"] for qa in first["qas"]},
            {first["dialogue_family_split"]},
        )

    def test_missing_support_is_not_used_as_answer_truth(self) -> None:
        value = bundle()
        value["support_ledger"] = [
            row for row in value["support_ledger"] if row["claim_id"] != "C02"
        ]
        result = compiler.compile_claim_first_qas(
            value, seed_sample_id="seed-1", min_qas=3, max_qas=6
        )
        used = {
            claim_id
            for qa in result["qas"]
            for claim_id in qa["proof"]["required_claim_ids"]
        }
        self.assertNotIn("C02", used)

    def test_pre_dialogue_truth_freezes_then_support_binds_per_qa(self) -> None:
        pre_bundle = bundle()
        pre_bundle["support_ledger"] = []
        frozen = compiler.compile_claim_first_qas(
            pre_bundle,
            seed_sample_id="seed-1",
            min_qas=3,
            max_qas=6,
            require_support=False,
        )
        self.assertEqual(frozen["support_binding_status"], "pending")
        self.assertTrue(all(not qa["evidence_ledger"] for qa in frozen["qas"]))

        post_bundle = bundle()
        # Make exactly the QAs depending on C02 fail support binding; sibling
        # QAs and the dialogue must remain available.
        post_bundle["support_ledger"] = [
            row for row in post_bundle["support_ledger"] if row["claim_id"] != "C02"
        ]
        bound = compiler.bind_compiled_qas_to_support(frozen, post_bundle)
        self.assertGreater(bound["qa_support_reject_count"], 0)
        self.assertGreater(bound["qa_count"], 0)
        self.assertLess(bound["qa_count"], bound["pre_dialogue_qa_count"])
        self.assertTrue(bound["dialogue_retained"])
        self.assertTrue(
            all(
                "C02" not in qa["proof"]["required_claim_ids"]
                for qa in bound["qas"]
            )
        )

    def test_ambiguous_or_too_small_bundle_rejects_before_api(self) -> None:
        value = bundle()
        value["claims"] = copy.deepcopy(value["claims"][:2])
        value["support_ledger"] = copy.deepcopy(value["support_ledger"][:2])
        with self.assertRaisesRegex(compiler.ClaimProofCompileError, "need at least"):
            compiler.compile_claim_first_qas(value, seed_sample_id="seed-1")

    def test_option_gold_is_unique(self) -> None:
        result = compiler.compile_claim_first_qas(bundle(), seed_sample_id="seed-1")
        for qa in result["qas"]:
            true_options = [item for item in qa["options"] if item["truth_value"] is True]
            self.assertEqual(len(true_options), 1)
            self.assertEqual([true_options[0]["id"]], qa["gold"])
            self.assertEqual(
                len({str(item["text"]).casefold() for item in qa["options"]}),
                len(qa["options"]),
            )

    def test_immutable_temporal_target_compiles_active_gold_and_canceled_negative(self) -> None:
        source = temporal_target_bundle()
        target = canceled_temporal_target()
        result = compiler.compile_claim_first_qas(
            source,
            seed_sample_id="target-seed",
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target,
        )
        primary_id = result["target_fulfillment_qa_id"]
        primary = next(qa for qa in result["qas"] if qa["qa_id"] == primary_id)
        self.assertEqual(
            sum(qa.get("fulfills_immutable_stage64_target") is True for qa in result["qas"]),
            1,
        )
        self.assertEqual(primary["gold_answer_text"], "Lyon")
        gold_option = next(option for option in primary["options"] if option["id"] in primary["gold"])
        canceled_option = next(option for option in primary["options"] if option["text"] == "Rome")
        self.assertEqual(gold_option["source_event_instance_id"], "E02")
        self.assertTrue(gold_option["truth_value"])
        self.assertEqual(canceled_option["source_event_instance_id"], "E04")
        self.assertFalse(canceled_option["truth_value"])
        for field in (
            "reasoning_subtype",
            "reasoning_template",
            "hop_count",
            "hard_negative_strategy",
        ):
            self.assertEqual(primary[field], target[field])
            self.assertEqual(primary["query_plan"][field], target[field])
        self.assertEqual(primary["answer_type"], target["preferred_answer_type"])
        self.assertEqual(
            primary["query_plan"]["reasoning_anchor_path"],
            ["E01", "E02", "E03", "E04"],
        )
        self.assertEqual(
            primary["query_plan"]["resolved_distractor_event_instance_ids"],
            ["E01", "E03", "E04"],
        )
        self.assertEqual(
            primary["proof"]["claim_id_to_subject_event_id"],
            {"C01": "E01", "C02": "E02", "C03": "E03", "C04": "E04"},
        )
        self.assertTrue(
            all(
                option["anchor_ref"] == option["source_event_instance_id"]
                for option in primary["options"]
            )
        )

        legacy_qa = copy.deepcopy(primary)
        legacy_qa["question"] = legacy_qa["question_text_draft"]
        sample = {
            "stage64_target": target,
            "graph_update": {
                "graph_snapshot_id": "stage3:target-test",
                "events": copy.deepcopy(source["current_event_plan"]["events"]),
            },
            "dialogue": {"turns": []},
            "qa": legacy_qa,
        }
        self.assertEqual(stage64_quality.semantic_quality_issues(sample), [])

    def test_target_qa_is_reserved_inside_max_qas(self) -> None:
        result = compiler.compile_claim_first_qas(
            temporal_target_bundle(),
            min_qas=3,
            max_qas=3,
            immutable_stage64_target=canceled_temporal_target(),
        )
        self.assertEqual(result["qa_count"], 3)
        self.assertIn(
            result["target_fulfillment_qa_id"],
            {qa["qa_id"] for qa in result["qas"]},
        )

    def test_claim_event_mapping_fails_closed_on_missing_or_collision(self) -> None:
        missing = temporal_target_bundle()
        missing["claims"][0].pop("subject_event_id")
        with self.assertRaisesRegex(compiler.ClaimProofCompileError, "lacks subject_event_id"):
            compiler.compile_claim_first_qas(
                missing,
                immutable_stage64_target=canceled_temporal_target(),
            )

        collision = temporal_target_bundle()
        collision["claims"][1]["subject_event_id"] = "E01"
        with self.assertRaisesRegex(compiler.ClaimProofCompileError, "not one-to-one"):
            compiler.compile_claim_first_qas(
                collision,
                immutable_stage64_target=canceled_temporal_target(),
            )

    def test_target_mutations_do_not_forge_fulfillment(self) -> None:
        cases = []
        no_hard_negative = temporal_target_bundle()
        no_hard_negative["claims"][3]["status"] = "expired"
        cases.append(no_hard_negative)
        two_active = temporal_target_bundle()
        two_active["claims"][2]["status"] = "active"
        cases.append(two_active)
        broken_chain = temporal_target_bundle()
        broken_chain["current_event_plan"]["events"][3]["depends_on"] = []
        cases.append(broken_chain)
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    compiler.ClaimProofCompileError,
                    "does not resolve to exactly one",
                ):
                    compiler.compile_claim_first_qas(
                        source,
                        immutable_stage64_target=canceled_temporal_target(),
                    )


    def test_six_non_temporal_immutable_targets_compile_and_pass_quality(self) -> None:
        for operation in (
            "bridge",
            "comparison",
            "visibility",
            "count",
            "cause_consequence",
            "relation_lookup",
        ):
            with self.subTest(operation=operation):
                plan = operation_plan(operation)
                claim_ir = stage3_claim_ir.compile_claim_ir(plan)
                target = immutable_target(operation)
                result = compiler.compile_claim_first_qas(
                    claim_ir,
                    seed_sample_id=f"seed:{operation}",
                    min_qas=1,
                    max_qas=1,
                    require_support=False,
                    immutable_stage64_target=target,
                )
                self.assertEqual(result["qa_count"], 1)
                qa = result["qas"][0]
                self.assertEqual(qa["qa_id"], result["target_fulfillment_qa_id"])
                self.assertTrue(qa["fulfills_immutable_stage64_target"])
                self.assertEqual(qa["answer_type"], target["preferred_answer_type"])
                for field in (
                    "qa_operation",
                    "reasoning_subtype",
                    "reasoning_template",
                    "hop_count",
                    "hard_negative_strategy",
                ):
                    self.assertEqual(qa[field], target[field])
                    self.assertEqual(qa["query_plan"][field], target[field])
                forward = qa["query_plan"]["claim_id_to_event_instance_id"]
                reverse = qa["query_plan"]["event_instance_id_to_claim_id"]
                self.assertEqual(
                    {event_id: claim_id for claim_id, event_id in forward.items()},
                    reverse,
                )
                semantic = stage3_claim_semantic_validator.validate_qa_semantics(
                    claim_ir, qa
                )
                self.assertTrue(semantic["pass"], semantic["issues"])
                legacy_qa = copy.deepcopy(qa)
                legacy_qa["question"] = qa["question_text_draft"]
                sample = {
                    "stage64_target": target,
                    "graph_update": {
                        "graph_snapshot_id": plan["graph_snapshot_id"],
                        "events": copy.deepcopy(plan["events"]),
                    },
                    "dialogue": {"turns": []},
                    "qa": legacy_qa,
                }
                self.assertEqual(
                    stage64_quality.semantic_quality_issues(sample), []
                )

    def test_count_truth_is_order_independent_and_membership_grounded(self) -> None:
        plan = operation_plan("count")
        reordered = copy.deepcopy(plan)
        reordered["events"] = [
            reordered["events"][index]
            for index in (4, 0, 5, 1, 6, 2, 7, 3)
        ]
        result = compiler.compile_claim_first_qas(
            stage3_claim_ir.compile_claim_ir(reordered),
            min_qas=1,
            max_qas=1,
            require_support=False,
            immutable_stage64_target=immutable_target("count"),
        )
        qa = result["qas"][0]
        self.assertEqual(qa["gold_answer_text"], "4")
        included_claim_ids = qa["proof"]["derivation"]["included_claim_ids"]
        included_event_ids = {
            qa["query_plan"]["claim_id_to_event_instance_id"][claim_id]
            for claim_id in included_claim_ids
        }
        self.assertEqual(included_event_ids, {"E01", "E02", "E03", "E04"})
        mutated_qa = copy.deepcopy(qa)
        mutated_qa["proof"]["derivation"]["computed_value"] = "99"
        semantic = stage3_claim_semantic_validator.validate_qa_semantics(
            stage3_claim_ir.compile_claim_ir(reordered),
            mutated_qa,
        )
        self.assertFalse(semantic["pass"])
        self.assertIn(
            "derivation_computed_value_mismatch",
            semantic["issue_counts_by_code"],
        )
        claims_by_id = {
            claim["claim_id"]: claim
            for claim in stage3_claim_ir.compile_claim_ir(reordered)["claims"]
        }
        self.assertTrue(
            all(
                claims_by_id[claim_id]["claim_type"] == "membership"
                for claim_id in qa["proof"]["gold_operand_claim_ids"]
            )
        )

        changed = copy.deepcopy(reordered)
        changed_event = next(
            event for event in changed["events"] if event["event_id"] == "E04"
        )
        changed_event["set_membership"]["included"] = False
        changed_result = compiler.compile_claim_first_qas(
            stage3_claim_ir.compile_claim_ir(changed),
            min_qas=1,
            max_qas=1,
            require_support=False,
            immutable_stage64_target=immutable_target("count"),
        )
        self.assertEqual(changed_result["qas"][0]["gold_answer_text"], "3")

    def test_bridge_rejects_join_closure_and_terminal_copy_mutants(self) -> None:
        mutations = []
        missing_edge = operation_plan("bridge")
        missing_edge["events"][2]["depends_on"] = ["E01"]
        mutations.append(missing_edge)
        broken_join = operation_plan("bridge")
        broken_join["events"][0]["value"] = "Ava"
        mutations.append(broken_join)
        copied_answer = operation_plan("bridge")
        copied_answer["events"][2]["value"] = copied_answer["events"][1]["value"]
        mutations.append(copied_answer)
        proposition_answer = operation_plan("bridge")
        proposition_answer["events"][1]["value"] = (
            "Joanna watched the trilogy based on Nate's recommendation"
        )
        mutations.append(proposition_answer)
        extra_edge = operation_plan("bridge")
        extra_edge["events"][3]["depends_on"] = ["E01"]
        mutations.append(extra_edge)
        for plan in mutations:
            with self.subTest(plan=plan):
                with self.assertRaises(compiler.ClaimProofCompileError):
                    compiler.compile_claim_first_qas(
                        stage3_claim_ir.compile_claim_ir(plan),
                        min_qas=1,
                        max_qas=1,
                        require_support=False,
                        immutable_stage64_target=immutable_target("bridge"),
                    )

    def test_bridge_options_preserve_person_answer_domain(self) -> None:
        valid = operation_plan("bridge")
        valid["events"][1]["value"] = "Mary Jane Watson"
        result = compiler.compile_claim_first_qas(
            stage3_claim_ir.compile_claim_ir(valid),
            min_qas=1,
            max_qas=1,
            require_support=False,
            immutable_stage64_target=immutable_target("bridge"),
        )
        self.assertTrue(
            all(
                compiler._person_value_is_atomic(option["text"])
                for option in result["qas"][0]["options"]
            )
        )

        mixed = operation_plan("bridge")
        mixed["events"][3]["value"] = "pending review"
        with self.assertRaisesRegex(
            compiler.ClaimProofCompileError,
            "fewer than three unique distractors",
        ):
            compiler.compile_claim_first_qas(
                stage3_claim_ir.compile_claim_ir(mixed),
                min_qas=1,
                max_qas=1,
                require_support=False,
                immutable_stage64_target=immutable_target("bridge"),
            )

    def test_cause_requires_typed_directed_relation(self) -> None:
        plan = operation_plan("cause_consequence")
        result = compiler.compile_claim_first_qas(
            stage3_claim_ir.compile_claim_ir(plan),
            min_qas=1,
            max_qas=1,
            require_support=False,
            immutable_stage64_target=immutable_target("cause_consequence"),
        )
        qa = result["qas"][0]
        cause_claim_id = qa["proof"]["derivation"]["cause_claim_ids"][0]
        effect_claim_id = qa["proof"]["derivation"]["target_claim_ids"][0]
        self.assertEqual(
            qa["query_plan"]["claim_id_to_event_instance_id"][cause_claim_id],
            "E01",
        )
        self.assertEqual(
            qa["query_plan"]["claim_id_to_event_instance_id"][effect_claim_id],
            "E02",
        )
        broken = operation_plan("cause_consequence")
        broken["events"][1]["depends_on"] = []
        with self.assertRaises(compiler.ClaimProofCompileError):
            compiler.compile_claim_first_qas(
                stage3_claim_ir.compile_claim_ir(broken),
                min_qas=1,
                max_qas=1,
                require_support=False,
                immutable_stage64_target=immutable_target("cause_consequence"),
            )

    def test_visibility_rejects_role_and_status_divergence(self) -> None:
        action_mismatch = operation_plan("visibility")
        action_mismatch["events"][1]["action"] = "edit"
        nonactive_gold = operation_plan("visibility")
        nonactive_gold["events"][1]["status"] = "scope_limited"
        duplicate_active = operation_plan("visibility")
        duplicate_active["events"][2]["status"] = "active"
        for plan in (action_mismatch, nonactive_gold, duplicate_active):
            with self.subTest(plan=plan):
                with self.assertRaises(compiler.ClaimProofCompileError):
                    compiler.compile_claim_first_qas(
                        stage3_claim_ir.compile_claim_ir(plan),
                        min_qas=1,
                        max_qas=1,
                        require_support=False,
                        immutable_stage64_target=immutable_target("visibility"),
                    )

    def test_relation_requires_explicit_binder_edge_identity(self) -> None:
        missing = operation_plan("relation_lookup")
        missing["events"][1]["depends_on"] = []
        wrong = operation_plan("relation_lookup")
        wrong["events"][1]["depends_on"] = ["E05"]
        for plan in (missing, wrong):
            with self.subTest(plan=plan):
                with self.assertRaises(compiler.ClaimProofCompileError):
                    compiler.compile_claim_first_qas(
                        stage3_claim_ir.compile_claim_ir(plan),
                        min_qas=1,
                        max_qas=1,
                        require_support=False,
                        immutable_stage64_target=immutable_target("relation_lookup"),
                    )

    def test_comparison_preserves_both_operand_provenances(self) -> None:
        plan = operation_plan("comparison")
        result = compiler.compile_claim_first_qas(
            stage3_claim_ir.compile_claim_ir(plan),
            min_qas=1,
            max_qas=1,
            require_support=False,
            immutable_stage64_target=immutable_target("comparison"),
        )
        qa = result["qas"][0]
        self.assertEqual(
            set(qa["proof"]["gold_operand_event_instance_ids"]),
            {"E01", "E02"},
        )
        self.assertEqual(
            set(qa["operation_proof_view"]["gold_event_instance_ids"]),
            {"E01", "E02"},
        )
        self.assertEqual(
            set(qa["query_plan"]["resolved_gold_event_instance_ids"]),
            {"E01", "E02"},
        )
        gold_option = next(
            option for option in qa["options"] if option["truth_value"] is True
        )
        self.assertEqual(
            set(gold_option["support_event_instance_ids"]),
            {"E01", "E02"},
        )

        ambiguous = operation_plan("comparison")
        ambiguous["events"][4].update(
            {
                "entity": "Iris",
                "slot": "bonus_hours",
                "value": 8,
                "status": "active",
                "time_index": 11,
            }
        )
        ambiguous["events"][5].update(
            {
                "entity": "Eli",
                "slot": "bonus_hours",
                "value": 3,
                "status": "active",
                "time_index": 11,
            }
        )
        with self.assertRaises(compiler.ClaimProofCompileError):
            compiler.compile_claim_first_qas(
                stage3_claim_ir.compile_claim_ir(ambiguous),
                min_qas=1,
                max_qas=1,
                require_support=False,
                immutable_stage64_target=immutable_target("comparison"),
            )

    def test_noncanonical_target_signature_fails_before_compilation(self) -> None:
        plan = operation_plan("count")
        target = immutable_target("count")
        target["reasoning_template"] = "read_answer_directly"
        with self.assertRaisesRegex(
            compiler.ClaimProofCompileError, "not a canonical reasoning signature"
        ):
            compiler.compile_claim_first_qas(
                stage3_claim_ir.compile_claim_ir(plan),
                min_qas=1,
                max_qas=1,
                require_support=False,
                immutable_stage64_target=target,
            )


if __name__ == "__main__":
    unittest.main()
