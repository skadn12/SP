from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import stage3_claim_ir as claim_ir  # noqa: E402
from scripts.current_method import stage3_truth_constructors as constructors  # noqa: E402
from scripts.current_method import stage3_truth_qa_compiler as compiler  # noqa: E402


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
        "target_basis": "truth_qa_compiler_unit_test",
    }


def fixture(operation: str) -> tuple[dict, dict, dict]:
    target = immutable_target(operation)
    skeleton = constructors.construct_truth_skeleton(
        target,
        graph_id=f"truth-qa-test-{operation}",
        parent_graph_id="parent-truth-qa-test",
        selected_parent_event_ids=["P17", "P29"],
    )
    materialized = constructors.materialize_truth_skeleton(
        skeleton,
        constructors.example_content_bindings(skeleton),
    )
    bundle = claim_ir.compile_claim_ir(materialized["graph"])
    return materialized, bundle, target


class TruthQACompilerHappyPathTest(unittest.TestCase):
    def test_all_seven_operations_compile_four_frozen_qas_without_api(self) -> None:
        for operation in constructors.OPERATIONS:
            with self.subTest(operation=operation):
                materialized, bundle, target = fixture(operation)
                result = compiler.compile_truth_first_qas(
                    materialized,
                    bundle,
                    immutable_stage64_target=target,
                    qa_count=4,
                    seed_sample_id="seed-001",
                    lineage_group_id="lineage-001",
                )
                self.assertEqual(result["qa_count"], 4)
                self.assertEqual(result["api_calls_made"], 0)
                self.assertEqual(len(result["qas"]), 4)
                self.assertEqual(result["qas"][0]["qa_role"], "primary")
                self.assertTrue(result["qas"][0]["fulfills_immutable_stage64_target"])
                self.assertEqual(result["target_fulfillment_qa_id"], result["qas"][0]["qa_id"])
                self.assertEqual(len({row["qa_id"] for row in result["qas"]}), 4)
                self.assertEqual(
                    len({row["question_text_draft"] for row in result["qas"]}),
                    4,
                )
                self.assertTrue(all(row["semantic_contract_frozen"] for row in result["qas"]))
                self.assertTrue(all(row["provenance"]["api_calls_made"] == 0 for row in result["qas"]))
                self.assertTrue(all(report["pass"] for report in result["semantic_validation"]))

    def test_primary_exactly_fulfills_target_and_manifest_proof(self) -> None:
        for operation in constructors.OPERATIONS:
            materialized, bundle, target = fixture(operation)
            primary = compiler.compile_truth_first_qas(
                materialized,
                bundle,
                immutable_stage64_target=target,
            )["qas"][0]
            self.assertEqual(primary["immutable_stage64_target"], target)
            self.assertEqual(primary["qa_operation"], target["qa_operation"])
            self.assertEqual(primary["answer_type"], target["preferred_answer_type"])
            self.assertEqual(primary["reasoning_subtype"], target["reasoning_subtype"])
            self.assertEqual(primary["reasoning_template"], target["reasoning_template"])
            self.assertEqual(primary["hop_count"], target["hop_count"])
            self.assertEqual(primary["hard_negative_strategy"], target["hard_negative_strategy"])
            self.assertEqual(
                primary["truth_manifest_proof_contract"],
                materialized["truth_manifest"]["proof_contract"],
            )
            self.assertEqual(
                primary["proof"]["gold_projection"],
                materialized["truth_manifest"]["proof_contract"]["gold_projection"],
            )
            self.assertEqual(
                primary["gold_answer"],
                materialized["truth_manifest"]["proof_contract"]["compiled_gold"],
            )

    def test_claim_to_event_to_query_plan_mapping_is_lossless(self) -> None:
        materialized, bundle, target = fixture("cause_consequence")
        primary = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        proof = primary["proof"]
        rows = proof["claim_event_mapping"]
        self.assertEqual(
            [row["claim_id"] for row in rows],
            proof["required_claim_ids"],
        )
        self.assertEqual(set(proof["claim_id_to_event_id"]), set(proof["required_claim_ids"]))
        # E1 is the subject of both its node fact and the directed cause claim.
        # The reverse mapping must preserve both instead of collapsing them.
        self.assertGreaterEqual(len(proof["event_id_to_claim_ids"]["E1"]), 2)
        query_claim_ids = [
            step["claim_id"]
            for step in primary["query_plan"]["steps"]
            if "claim_id" in step
        ]
        self.assertEqual(query_claim_ids, proof["required_claim_ids"])

    def test_output_is_stable_for_identical_frozen_inputs(self) -> None:
        materialized, bundle, target = fixture("comparison")
        first = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
            lineage_group_id="same-lineage",
        )
        second = compiler.compile_truth_first_qas(
            copy.deepcopy(materialized),
            copy.deepcopy(bundle),
            immutable_stage64_target=copy.deepcopy(target),
            lineage_group_id="same-lineage",
        )
        self.assertEqual(first, second)

    def test_three_and_four_are_valid_family_sizes(self) -> None:
        for operation in constructors.OPERATIONS:
            materialized, bundle, target = fixture(operation)
            for qa_count in (3, 4):
                with self.subTest(operation=operation, qa_count=qa_count):
                    result = compiler.compile_truth_first_qas(
                        materialized,
                        bundle,
                        immutable_stage64_target=target,
                        qa_count=qa_count,
                    )
                    self.assertGreaterEqual(result["qa_count"], compiler.MIN_QAS)
                    self.assertLessEqual(result["qa_count"], compiler.MAX_QAS)


class OperationTruthTest(unittest.TestCase):
    def test_bridge_primary_is_real_two_hop_and_excludes_control_event(self) -> None:
        materialized, bundle, target = fixture("bridge")
        graph_events = {row["event_id"]: row for row in materialized["graph"]["events"]}
        primary = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        question = primary["question_text_draft"].casefold()
        self.assertIn(graph_events["E1"]["entity"].casefold(), question)
        self.assertIn(graph_events["E1"]["slot"].replace("_", " "), question)
        self.assertIn(graph_events["E2"]["slot"].replace("_", " "), question)
        self.assertNotIn(graph_events["E1"]["value"].casefold(), question)
        self.assertNotIn(graph_events["E2"]["value"].casefold(), question)
        self.assertEqual(primary["proof"]["reasoning_event_ids"], ["E1", "E2"])
        self.assertNotIn("E3", primary["proof"]["reasoning_event_ids"])
        self.assertEqual(primary["gold_answer"], graph_events["E2"]["value"])

    def test_count_uses_explicit_membership_and_survives_event_shuffle(self) -> None:
        materialized, bundle, target = fixture("count")
        baseline = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )
        shuffled = copy.deepcopy(materialized)
        shuffled["graph"]["events"] = [
            shuffled["graph"]["events"][index]
            for index in (5, 1, 7, 0, 4, 2, 6, 3)
        ]
        shuffled["truth_manifest"]["graph_sha256"] = compiler.canonical_sha256(
            shuffled["graph"]
        )
        shuffled_bundle = claim_ir.compile_claim_ir(shuffled["graph"])
        replay = compiler.compile_truth_first_qas(
            shuffled,
            shuffled_bundle,
            immutable_stage64_target=target,
        )
        self.assertEqual(baseline["qas"][0]["gold_answer"], 4)
        self.assertEqual(replay["qas"][0]["gold_answer"], 4)
        self.assertEqual(
            {row["claim_type"] for row in replay["qas"][0]["proof"]["claim_event_mapping"]},
            {"membership"},
        )
        self.assertEqual(
            replay["qas"][0]["proof"]["gold_projection"]["kind"],
            "count_membership",
        )

    def test_comparison_gold_is_computed_locally(self) -> None:
        materialized, bundle, target = fixture("comparison")
        events = {row["event_id"]: row for row in materialized["graph"]["events"]}
        primary = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        self.assertEqual(primary["gold_answer"], abs(events["E1"]["value"] - events["E2"]["value"]))
        self.assertEqual(primary["query_plan"]["final_operator"], "absolute_difference")

    def test_cause_primary_has_directed_e1_to_e2_claim(self) -> None:
        materialized, bundle, target = fixture("cause_consequence")
        primary = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        cause_rows = [
            row
            for row in primary["proof"]["claim_event_mapping"]
            if row["claim_type"] == "cause"
        ]
        self.assertEqual(len(cause_rows), 1)
        self.assertEqual(cause_rows[0]["subject_event_id"], "E1")
        self.assertEqual(cause_rows[0]["object_event_ids"], ["E2"])

    def test_visibility_primary_preserves_typed_scope(self) -> None:
        materialized, bundle, target = fixture("visibility")
        event = next(row for row in materialized["graph"]["events"] if row["event_id"] == "E2")
        self.assertTrue(
            {"observer", "action", "resource", "condition", "access_class", "status", "scope"}.issubset(event)
        )
        primary = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        self.assertEqual(primary["gold_answer"], event["observer"])
        self.assertIn("visibility", {row["claim_type"] for row in primary["proof"]["claim_event_mapping"]})


class TruthQAMutationTest(unittest.TestCase):
    def test_graph_mutation_is_rejected_before_qa_compilation(self) -> None:
        materialized, bundle, target = fixture("relation_lookup")
        materialized["graph"]["events"][1]["value"] = "Tampered Person"
        with self.assertRaisesRegex(compiler.TruthQACompileError, "graph hash mismatch"):
            compiler.compile_truth_first_qas(
                materialized,
                bundle,
                immutable_stage64_target=target,
            )

    def test_claim_row_mutation_is_rejected_even_if_declared_hash_is_unchanged(self) -> None:
        materialized, bundle, target = fixture("bridge")
        bundle["claims"][0]["value"] = "Tampered"
        with self.assertRaisesRegex(compiler.TruthQACompileError, "claim rows differ"):
            compiler.compile_truth_first_qas(
                materialized,
                bundle,
                immutable_stage64_target=target,
            )

    def test_wrong_gold_fails_after_attacker_recomputes_qa_hash(self) -> None:
        materialized, bundle, target = fixture("comparison")
        qa = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        qa["gold_answer"] += 1
        qa["gold_answer_text"] = str(qa["gold_answer"])
        qa["qa_semantics_sha256"] = compiler.qa_semantics_sha256(qa)
        report = compiler.validate_compiled_qa(qa, materialized, bundle)
        self.assertFalse(report["pass"])
        self.assertIn("qa_gold_does_not_match_projection", {row["code"] for row in report["issues"]})

    def test_fake_bridge_that_names_intermediate_is_rejected(self) -> None:
        materialized, bundle, target = fixture("bridge")
        qa = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        events = {row["event_id"]: row for row in materialized["graph"]["events"]}
        qa["question_text_draft"] = (
            f"What is {events['E1']['value']}'s {events['E2']['slot'].replace('_', ' ')}?"
        )
        qa["qa_semantics_sha256"] = compiler.qa_semantics_sha256(qa)
        report = compiler.validate_compiled_qa(qa, materialized, bundle)
        self.assertFalse(report["pass"])
        codes = {row["code"] for row in report["issues"]}
        self.assertIn("bridge_question_names_intermediate_or_gold", codes)

    def test_fake_bridge_with_direct_e2_path_is_rejected(self) -> None:
        materialized, bundle, target = fixture("bridge")
        qa = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )["qas"][0]
        qa["proof"]["reasoning_event_ids"] = ["E2"]
        qa["qa_semantics_sha256"] = compiler.qa_semantics_sha256(qa)
        report = compiler.validate_compiled_qa(qa, materialized, bundle)
        self.assertFalse(report["pass"])
        codes = {row["code"] for row in report["issues"]}
        self.assertIn("bridge_reasoning_path_is_not_true_two_hop", codes)

    def test_one_bad_sibling_is_removed_without_deleting_dialogue_or_other_qas(self) -> None:
        materialized, bundle, target = fixture("count")
        compiled = compiler.compile_truth_first_qas(
            materialized,
            bundle,
            immutable_stage64_target=target,
        )
        bad_id = compiled["qas"][2]["qa_id"]
        compiled["qas"][2]["gold_answer"] = not compiled["qas"][2]["gold_answer"]
        compiled["qas"][2]["gold_answer_text"] = str(compiled["qas"][2]["gold_answer"])
        compiled["qas"][2]["qa_semantics_sha256"] = compiler.qa_semantics_sha256(compiled["qas"][2])
        filtered = compiler.validate_and_filter_qas(compiled, materialized, bundle)
        self.assertTrue(filtered["dialogue_retained"])
        self.assertEqual(filtered["semantic_clean_qa_count"], 3)
        self.assertEqual(filtered["semantic_reject_qa_count"], 1)
        self.assertEqual(filtered["semantic_rejects"][0]["qa_id"], bad_id)
        self.assertEqual(filtered["api_calls_made"], 0)

    def test_invalid_requested_family_size_fails_closed(self) -> None:
        materialized, bundle, target = fixture("count")
        for qa_count in (2, 7):
            with self.subTest(qa_count=qa_count):
                with self.assertRaises(ValueError):
                    compiler.compile_truth_first_qas(
                        materialized,
                        bundle,
                        immutable_stage64_target=target,
                        qa_count=qa_count,
                    )


if __name__ == "__main__":
    unittest.main()
