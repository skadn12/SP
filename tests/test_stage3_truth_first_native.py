from __future__ import annotations

import argparse
import copy
import json
import random
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import native_api_graph_dialogue_expansion as native
from scripts.current_method import stage3_truth_constructors as truth
from scripts.current_method import stage3_truth_first_discourse as discourse
from scripts.current_method import stage3_multiqa_rows
from scripts.current_method import recalibrate_stage64_shuffled_frontier
from scripts.current_method import shuffle_stage64_options
from scripts.current_method import stage3_solver_feedback_mining
from scripts.current_method import stage64_quality
from tests.stage3_truth_test_helpers import context_sentence, flat_context_body


ROOT = Path(__file__).resolve().parents[1]
PARENT_FILE = ROOT / "stage3-data-fix" / "stage3_parent_v4_memory_backfilled.jsonl"
PARENT_ROWS = [
    json.loads(line)
    for line in PARENT_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
PARENT_INDEX_BY_OPERATION = {
    "temporal_state_update": 1,
    "count": 3,
}


def target(operation: str = "bridge") -> dict[str, object]:
    specs = {
        "temporal_state_update": (
            "LocationOrScope",
            "status_gated_canceled_update",
            "reject_canceled_value_then_resolve_update",
            4,
            "same_slot_canceled_value",
        ),
        "bridge": (
            "Person",
            "conjunctive_two_support_bridge",
            "join_two_support_facts",
            2,
            "one_hop_preserving_alternative",
        ),
        "comparison": (
            "Number",
            "same_slot_absolute_difference",
            "bind_pair_then_compute_difference",
            2,
            "near_numeric_off_by_one",
        ),
        "visibility": (
            "Person",
            "observer_permission_scope",
            "resolve_observer_permission_and_scope",
            2,
            "correct_fact_wrong_observer",
        ),
        "count": (
            "Number",
            "filtered_membership_count",
            "filter_members_then_count",
            2,
            "include_exclude_boundary_count",
        ),
        "cause_consequence": (
            "Person",
            "direct_causal_dependency",
            "trace_cause_to_effect",
            2,
            "temporally_adjacent_non_cause",
        ),
        "relation_lookup": (
            "Person",
            "entity_constrained_relation",
            "bind_entity_then_relation",
            2,
            "same_relation_wrong_entity",
        ),
    }
    answer_type, subtype, template, hops, negative = specs[operation]
    return {
        "qa_operation": operation,
        "preferred_answer_type": answer_type,
        "reasoning_subtype": subtype,
        "reasoning_template": template,
        "hop_count": hops,
        "hard_negative_strategy": negative,
        "forbid_max_time_shortcut": operation == "temporal_state_update",
        "target_basis": "truth_first_native_unit_test",
    }


def first_parent() -> dict[str, object]:
    return copy.deepcopy(PARENT_ROWS[0])


def parent_for(operation: str) -> dict[str, object]:
    """Return a deterministic fixture with a real operation-compatible anchor."""

    return copy.deepcopy(
        PARENT_ROWS[PARENT_INDEX_BY_OPERATION.get(operation, 0)]
    )


def prepared(operation: str = "bridge") -> dict[str, object]:
    return native.prepare_truth_first_graph_contract(
        parent_for(operation),
        {"stage64_target": target(operation)},
        random.Random(17),
        sample_index=0,
        round_id="truth_native_test",
    )


def valid_payload(value: dict[str, object]) -> dict[str, object]:
    payload = truth.example_content_bindings(value["skeleton"])
    anchor_terms = value["skeleton"]["parent_anchor_contract"]["topic_terms"]
    payload["discourse_frame"] = {
        "topic": f"a follow-up about {anchor_terms[0]}",
        "setting": "the established participants' next conversation",
    }
    return payload


def api_args() -> argparse.Namespace:
    return argparse.Namespace(
        truth_first_generate_by_construction=True,
        graph_semantic_retries=2,
        graph_expansion_max_tokens=2048,
        challenger_temperature=0.2,
        challenger_max_tokens=8192,
    )


def _semantic_quote(canonical: dict[str, object]) -> str:
    subject = str(canonical["subject"])
    predicate = str(canonical["predicate"]).replace("_", " ")
    values = canonical.get("objects", [""])
    if not isinstance(values, list):
        values = [values]
    value = " and ".join(str(item) for item in values)
    time = canonical.get("time") if isinstance(canonical.get("time"), dict) else {}
    time_index = time.get("time_index", 1)
    status = str(canonical.get("status") or "active")
    possessive = f"{subject}'" if subject.endswith("s") else f"{subject}'s"
    status_clause = {
        "report": "this is an unconfirmed report",
        "belief": "the speaker thinks this might be true but remains unsure",
        "outdated": "this is outdated and no longer applies",
        "canceled": "this was canceled before taking effect",
        "scope_limited": "this applies only within the stated scope",
    }.get(status, f"the recorded status is {status}")
    condition = canonical.get("condition")
    condition_clause = f" provided that {condition}" if condition else ""
    scope = canonical.get("scope")
    scope_clause = ""
    if isinstance(scope, dict) and scope:
        scope_clause = " within the scope " + ", ".join(
            f"{key.replace('_', ' ')} {scope[key]}" for key in sorted(scope)
        )
    return (
        f"At time {time_index}, {possessive} {predicate} is {value}; "
        f"{status_clause}{condition_clause}{scope_clause}."
    )


def _response_from_discourse_prompt(prompt: dict[str, object]) -> dict[str, object]:
    contract = prompt["api_contract"]
    brief = contract["read_only_brief"]
    topic = str(brief["topic_spine"]["topic"])
    turns = []
    for slot in brief["turn_slots"]:
        context = []
        sentence_cursor = int(slot["order_index"]) * 32
        minimum_words = int(
            slot.get("minimum_context_words")
            or slot.get("target_words")
            or 24
        )
        minimum_sentences = max(
            2, int(slot.get("target_context_sentence_count") or 0)
        )
        while (
            len(context) < minimum_sentences
            or sum(len(sentence.split()) for sentence in context) < minimum_words
        ):
            context.append(
                context_sentence(sentence_cursor + len(context), topic)
            )
        turns.append(
            {
                "turn_slot_id": slot["turn_slot_id"],
                "context_sentences": context,
            }
        )
    return {"turns": turns}


class TruthFirstNativeGraphTest(unittest.TestCase):
    def test_append_only_freeze_preserves_bijective_planned_turn_id_remap(
        self,
    ) -> None:
        source_turns = [
            {
                "turn_id": "truth-turn-1",
                "speaker": "A",
                "text": "The permit is active.",
            },
            {
                "turn_id": "truth-turn-2",
                "speaker": "B",
                "text": "I recorded that update.",
            },
        ]
        clean_turns = native.strip_alignment_anchors_from_dialogue(source_turns)
        self.assertEqual(
            [turn["turn_id"] for turn in clean_turns],
            ["truth-turn-1", "truth-turn-2"],
        )
        frozen = native.freeze_append_only_dialogue(
            [
                {
                    "turn_id": "D1:1",
                    "speaker": "A",
                    "text": "Earlier context.",
                }
            ],
            clean_turns,
            sample_index=9,
        )
        self.assertTrue(frozen["appended_turn_id_map_validation"]["pass"])
        self.assertEqual(
            frozen["appended_turn_id_map"],
            {"truth-turn-1": "D2:1", "truth-turn-2": "D2:2"},
        )
        self.assertEqual(frozen["appended_turn_count"], 2)

        source_alignment = {
            "claim-1": {
                "turn_id": "truth-turn-1",
                "exact_support_quote": "permit is active",
            },
            "event-1": {
                "turn_id": "truth-turn-1",
                "exact_support_quote": "permit is active",
            },
        }
        remapped = (
            native.remap_truth_first_claim_alignment_to_frozen_dialogue(
                source_alignment,
                frozen,
            )
        )
        self.assertEqual(remapped["claim-1"]["turn_id"], "D2:1")
        self.assertEqual(remapped["event-1"]["turn_id"], "D2:1")
        self.assertEqual(source_alignment["claim-1"]["turn_id"], "truth-turn-1")

    def test_truth_first_turn_id_remap_rejects_non_bijective_freeze(self) -> None:
        frozen = native.freeze_append_only_dialogue(
            [],
            [
                {"turn_id": "duplicate", "speaker": "A", "text": "First."},
                {"turn_id": "duplicate", "speaker": "B", "text": "Second."},
            ],
            sample_index=1,
        )
        self.assertFalse(frozen["appended_turn_id_map_validation"]["pass"])
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "not complete and bijective",
        ):
            native.remap_truth_first_claim_alignment_to_frozen_dialogue(
                {
                    "claim-1": {
                        "turn_id": "duplicate",
                        "exact_support_quote": "First.",
                    }
                },
                frozen,
            )

    def test_truth_first_turn_id_remap_rejects_missing_turn_or_quote(self) -> None:
        frozen = native.freeze_append_only_dialogue(
            [],
            [
                {
                    "turn_id": "truth-turn-1",
                    "speaker": "A",
                    "text": "The permit is active.",
                }
            ],
            sample_index=1,
        )
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "unmapped planned turn",
        ):
            native.remap_truth_first_claim_alignment_to_frozen_dialogue(
                {
                    "claim-1": {
                        "turn_id": "truth-turn-missing",
                        "exact_support_quote": "permit is active",
                    }
                },
                frozen,
            )
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "not an exact substring",
        ):
            native.remap_truth_first_claim_alignment_to_frozen_dialogue(
                {
                    "claim-1": {
                        "turn_id": "truth-turn-1",
                        "exact_support_quote": "permit was canceled",
                    }
                },
                frozen,
            )
        outside_target = copy.deepcopy(frozen)
        outside_target["appended_turn_id_map"]["truth-turn-1"] = "D999:1"
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "points outside the frozen dialogue",
        ):
            native.remap_truth_first_claim_alignment_to_frozen_dialogue(
                {
                    "claim-1": {
                        "turn_id": "truth-turn-1",
                        "exact_support_quote": "permit is active",
                    }
                },
                outside_target,
            )

    def test_frozen_turn_id_remap_keeps_truth_first_support_bound(self) -> None:
        value = prepared("bridge")
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_frozen_turn_id_remap",
        )
        pre_claim_ir, compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target("bridge"),
        )
        source_alignment: dict[str, dict[str, object]] = {}
        appended_turns = []
        for index, claim in enumerate(pre_claim_ir["claims"], start=1):
            quote = native.stage3_truth_support.compile_typed_frozen_support(
                claim
            )["quote"]
            source_turn_id = f"planned-support-{index}"
            alignment = {
                "turn_id": source_turn_id,
                "exact_support_quote": quote,
            }
            source_alignment[claim["claim_id"]] = copy.deepcopy(alignment)
            if claim["claim_type"] == "node":
                source_alignment[claim["subject_event_id"]] = copy.deepcopy(
                    alignment
                )
            appended_turns.append(
                {
                    "turn_id": source_turn_id,
                    "speaker": "A",
                    "text": quote,
                }
            )
        frozen = native.freeze_append_only_dialogue(
            native.dialogue_turns_from_row(first_parent()),
            appended_turns,
            sample_index=0,
        )
        remapped = native.remap_truth_first_claim_alignment_to_frozen_dialogue(
            source_alignment,
            frozen,
        )
        bound = native.bind_validate_truth_first_multiqa(
            current_event_plan=plan,
            claim_alignment=remapped,
            final_dialogue=frozen,
            pre_claim_ir=pre_claim_ir,
            precompiled=compiled,
            discourse_validation={"pass": True},
            min_qas=3,
        )
        self.assertTrue(bound["proof_ready"], bound["qa_rejects"])
        self.assertEqual(bound["qa_count"], 4)
        planned_ids = set(frozen["appended_turn_id_map"])
        for qa in bound["qas"]:
            self.assertTrue(qa["evidence_ledger"])
            self.assertTrue(
                all(
                    entry["turn_id"] not in planned_ids
                    for entry in qa["evidence_ledger"]
                )
            )

    def test_prepare_uses_v4_retrieval_and_exposes_only_binding_schema(self) -> None:
        value = prepared()
        ledger = value["parent_event_retrieval"]
        self.assertEqual(ledger["full_parent_event_count"], 143)
        self.assertGreaterEqual(ledger["selected_parent_event_count"], 15)
        self.assertLessEqual(ledger["selected_parent_event_count"], 40)
        contract = value["prompt_payload"]["binding_contract"]
        self.assertEqual(
            set(contract["output_schema"]),
            {"bindings", "discourse_frame"},
        )
        continuity_contract = value["prompt_payload"][
            "parent_continuity_contract"
        ]
        continuity_binding_id = continuity_contract["continuity_binding_id"]
        self.assertTrue(continuity_binding_id)
        self.assertEqual(
            contract["bindings"][continuity_binding_id]["enum"],
            [continuity_contract["required_binding_value"]],
        )
        self.assertIn(
            repr([continuity_contract["required_binding_value"]]),
            contract["output_schema"]["bindings"][continuity_binding_id],
        )
        prompt_text = json.dumps(value["prompt_payload"], ensure_ascii=False)
        self.assertNotIn('"events": [{"event_id": "E1"', prompt_text)

    def test_incompatible_parent_is_rejected_before_any_api_contract(self) -> None:
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "parent continuity construction failed before API",
        ):
            native.prepare_truth_first_graph_contract(
                PARENT_ROWS[0],
                {"stage64_target": target("count")},
                random.Random(17),
                sample_index=0,
                round_id="truth_native_incompatible_parent_test",
            )

    def test_easy_v2_single_choice_survives_local_materialization(self) -> None:
        policy = {
            "stage64_target": target("temporal_state_update"),
            "native_curriculum_mode": native.EASY_V2_MODE,
            "planned_cell": {
                "question_type": "single_choice",
                "attack_skill": "conditional_plan",
            },
        }
        value = native.prepare_truth_first_graph_contract(
            parent_for("temporal_state_update"),
            policy,
            random.Random(17),
            sample_index=0,
            round_id="truth_native_easy_v2_test",
        )
        self.assertEqual(value["question_type"], "single_choice")
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_easy_v2_test",
        )
        self.assertEqual(plan["question_type"], "single_choice")
        self.assertEqual(plan["attack_skill"], "conditional_plan")

    def test_local_materialization_preserves_timeline_continuity_and_truth_gate(self) -> None:
        value = prepared()
        plan, meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_test",
        )
        self.assertTrue(meta["materialization_admission"]["pass"])
        self.assertTrue(meta["parent_child_continuity_audit"]["pass"])
        self.assertTrue(plan["metadata"]["truth_first_generate_by_construction"])
        self.assertEqual(len(plan["events"]), 8)
        parent_max = value["skeleton"]["timeline_contract"][
            "parent_max_global_time_index"
        ]
        self.assertTrue(
            all(event["global_time_index"] > parent_max for event in plan["events"])
        )
        continuity = plan["metadata"]["parent_continuity"]
        relation = continuity["continuity_relations"][0]
        linked_child = next(
            event
            for event in plan["events"]
            if event["event_id"] == relation["child_event_id"]
        )
        self.assertEqual(
            linked_child["depends_on_parent_event_ids"],
            [relation["parent_event_id"]],
        )
        self.assertEqual(
            continuity["depends_on_parent_event_ids"],
            [value["skeleton"]["parent_anchor_contract"]["primary_parent_event_id"]],
        )
        self.assertTrue(plan["graph_snapshot_hash"])
        self.assertEqual(meta["binding_validation"]["issue_count"], 0)

    def test_api_graph_fields_are_rejected_before_materialization(self) -> None:
        value = prepared()
        payload = valid_payload(value)
        payload["events"] = [{"slot": "discovery_date", "value": "Lila"}]
        with self.assertRaisesRegex(native.NativeGraphDialogueReject, "typed content binding"):
            native.materialize_truth_first_graph_contract(
                value,
                payload,
                sample_index=0,
                round_id="truth_native_test",
            )

    def test_api_binding_retry_never_accepts_forbidden_truth_fields(self) -> None:
        value = prepared()
        bad = valid_payload(value)
        bad["events"] = [{"slot": "confirmation", "value": "Mira"}]
        good = valid_payload(value)
        responses = [
            (bad, {"used": True, "attempts": 1, "model": "mock", "base_url": "mock"}),
            (good, {"used": True, "attempts": 1, "model": "mock", "base_url": "mock"}),
        ]
        with mock.patch.object(
            native, "prepare_truth_first_graph_contract", return_value=value
        ), mock.patch.object(native, "call_raw_json_api", side_effect=responses) as api:
            plan, meta, raw = native.call_api_truth_first_graph_expansion(
                first_parent(),
                {"stage64_target": target()},
                api_args(),
                random.Random(17),
                sample_index=0,
                round_id="truth_native_test",
            )
        self.assertEqual(api.call_count, 2)
        self.assertTrue(meta["used"])
        self.assertEqual(meta["semantic_attempts"], 2)
        self.assertFalse(meta["api_returned_graph_events"])
        self.assertEqual(meta["api_defined_truth_fields"], [])
        self.assertNotIn("events", raw)
        self.assertEqual(len(plan["events"]), 8)
        first_reject = meta["attempt_reports"][0]
        self.assertEqual(first_reject["stage"], "typed_binding_validation")
        self.assertIn(
            "forbidden_binding_payload_keys",
            {issue["code"] for issue in first_reject["issues"]},
        )
        second_prompt = api.call_args_list[1].args[1]
        feedback = second_prompt["prior_binding_rejection"]
        self.assertEqual(feedback["stage"], "typed_binding_validation")
        self.assertNotIn("api_meta", feedback)
        self.assertNotIn("raw_response", json.dumps(feedback))

    def test_semantic_binding_retry_names_rejected_value_and_narrow_type(self) -> None:
        value = prepared("temporal_state_update")
        bad = valid_payload(value)
        bad["bindings"]["context_event_name_2"] = "faced a setback"
        bad["bindings"]["temporal_candidate_1_location"] = (
            "local writers group"
        )
        good = valid_payload(value)
        good["bindings"]["context_event_name_2"] = "gaming party"
        responses = [
            (bad, {"used": True, "attempts": 1, "model": "mock", "base_url": "mock"}),
            (good, {"used": True, "attempts": 1, "model": "mock", "base_url": "mock"}),
        ]
        with mock.patch.object(
            native, "prepare_truth_first_graph_contract", return_value=value
        ), mock.patch.object(
            native, "call_raw_json_api", side_effect=responses
        ) as api:
            plan, meta, _raw = native.call_api_truth_first_graph_expansion(
                first_parent(),
                {"stage64_target": target("temporal_state_update")},
                api_args(),
                random.Random(17),
                sample_index=0,
                round_id="truth_native_semantic_retry_test",
            )

        self.assertEqual(api.call_count, 2)
        self.assertTrue(meta["used"])
        self.assertEqual(len(plan["events"]), 8)
        first_prompt = api.call_args_list[0].args[1]
        event_contract = first_prompt["binding_contract"]["bindings"][
            "context_event_name_2"
        ]
        self.assertEqual(event_contract["value_type"], "EventName")
        self.assertEqual(event_contract["base_value_type"], "Event")
        feedback = api.call_args_list[1].args[1]["prior_binding_rejection"]
        issue = next(
            row
            for row in feedback["issues"]
            if row.get("binding_id") == "context_event_name_2"
        )
        self.assertEqual(issue["required_semantic_value_type"], "EventName")
        self.assertEqual(issue["detected_semantic_value_types"], ["EventOutcome"])
        self.assertEqual(issue["rejected_value"], "faced a setback")
        self.assertTrue(issue["must_change_rejected_value"])
        self.assertIn(
            "gaming party", issue["required_semantic_value_examples"]["valid"]
        )
        location_issue = next(
            row
            for row in feedback["issues"]
            if row.get("binding_id") == "temporal_candidate_1_location"
        )
        self.assertEqual(
            location_issue["required_semantic_value_type"], "Location"
        )
        self.assertEqual(
            location_issue["rejected_value"], "local writers group"
        )
        self.assertIn(
            "Cedar Hall",
            location_issue["required_semantic_value_examples"]["valid"],
        )

    def test_continuity_binding_retry_exposes_exact_allowed_parent_identity(self) -> None:
        value = prepared("temporal_state_update")
        payload = valid_payload(value)
        continuity = value["skeleton"]["parent_anchor_contract"][
            "continuity_match"
        ]
        binding_id = continuity["child_binding_id"]
        required_value = continuity["parent_identity"]
        payload["bindings"][binding_id] = "Project Aurora"
        report = truth.validate_content_bindings(value["skeleton"], payload)
        self.assertFalse(report["pass"])
        feedback = native._compact_truth_binding_retry_feedback(report)
        issue = next(
            row for row in feedback["issues"] if row.get("binding_id") == binding_id
        )
        self.assertEqual(issue["allowed_values"], [required_value])
        self.assertEqual(issue["rejected_value"], "Project Aurora")
        self.assertTrue(issue["must_change_rejected_value"])

    def test_api_binding_accepts_missing_locally_owned_envelope(self) -> None:
        value = prepared("temporal_state_update")
        complete = valid_payload(value)
        raw = {
            "bindings": copy.deepcopy(complete["bindings"]),
            "discourse_frame": copy.deepcopy(complete["discourse_frame"]),
        }
        with mock.patch.object(
            native, "prepare_truth_first_graph_contract", return_value=value
        ), mock.patch.object(
            native,
            "call_raw_json_api",
            return_value=(
                raw,
                {"used": True, "attempts": 1, "model": "mock", "base_url": "mock"},
            ),
        ) as api:
            plan, meta, returned_raw = native.call_api_truth_first_graph_expansion(
                first_parent(),
                {"stage64_target": target("temporal_state_update")},
                api_args(),
                random.Random(17),
                sample_index=0,
                round_id="truth_native_envelope_test",
            )
        self.assertEqual(returned_raw, raw)
        self.assertTrue(meta["used"])
        self.assertEqual(
            meta["binding_envelope_normalization"]["injected_fields"],
            ["schema_version", "skeleton_sha256"],
        )
        self.assertEqual(
            meta["api_binding_raw_body_sha256"], truth.canonical_sha256(raw)
        )
        normalized = copy.deepcopy(raw)
        normalized.update(
            {
                "schema_version": truth.BINDING_SCHEMA_VERSION,
                "skeleton_sha256": value["skeleton"]["skeleton_sha256"],
            }
        )
        self.assertEqual(
            meta["normalized_binding_payload_sha256"],
            truth.canonical_sha256(normalized),
        )
        self.assertEqual(
            meta["binding_envelope_correlation_source"],
            "live_request_scoped_truth_skeleton",
        )
        self.assertEqual(len(plan["events"]), 8)
        self.assertEqual(
            api.call_args.kwargs["required_keys"],
            ("bindings", "discourse_frame"),
        )

    def test_api_binding_does_not_repair_explicit_wrong_hash(self) -> None:
        value = prepared("bridge")
        wrong = valid_payload(value)
        wrong["skeleton_sha256"] = "wrong-hash"
        responses = [
            (wrong, {"used": True, "attempts": 1, "model": "mock", "base_url": "mock"})
        ] * 3
        with mock.patch.object(
            native, "prepare_truth_first_graph_contract", return_value=value
        ), mock.patch.object(native, "call_raw_json_api", side_effect=responses):
            plan, meta, returned_raw = native.call_api_truth_first_graph_expansion(
                first_parent(),
                {"stage64_target": target("bridge")},
                api_args(),
                random.Random(17),
                sample_index=0,
                round_id="truth_native_wrong_hash_test",
            )
        self.assertEqual(plan, {})
        self.assertFalse(meta["used"])
        self.assertEqual(returned_raw["skeleton_sha256"], "wrong-hash")
        self.assertTrue(
            all(
                "binding_skeleton_hash_mismatch"
                in {issue["code"] for issue in report["issues"]}
                for report in meta["attempt_reports"]
            )
        )

    def test_failure_taxonomy_does_not_confuse_api_meta_with_meta_leak(self) -> None:
        exhausted = native.NativeGraphDialogueReject(
            "api graph expansion failed: {'api_meta': {}, "
            "'reason': 'typed_content_binding_exhausted'}"
        )
        self.assertEqual(
            native._failure_reason_from_exception(exhausted),
            "truth_content_binding",
        )
        api_graph = native.NativeGraphDialogueReject(
            "api graph expansion failed: {'api_meta': {}, 'reason': 'server_error'}"
        )
        self.assertEqual(
            native._failure_reason_from_exception(api_graph), "api_graph"
        )
        real_leak = native.NativeGraphDialogueReject(
            "clean dialogue meta leakage: internal/project terms"
        )
        self.assertEqual(
            native._failure_reason_from_exception(real_leak), "meta_leak"
        )

    def test_main_graph_entry_dispatches_to_truth_first_branch(self) -> None:
        expected = ({"events": []}, {"used": True}, {"bindings": {}})
        with mock.patch.object(
            native, "call_api_truth_first_graph_expansion", return_value=expected
        ) as branch:
            actual = native.call_api_graph_expansion(
                first_parent(),
                {},
                {"stage64_target": target()},
                api_args(),
                random.Random(17),
                sample_index=0,
                round_id="truth_native_test",
            )
        self.assertEqual(actual, expected)
        branch.assert_called_once()

    def test_native_precompile_uses_truth_qa_compiler_not_legacy_bridge(self) -> None:
        value = prepared()
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_test",
        )
        plan = native.assign_current_alignment_anchors(plan, random.Random(19))
        claim_ir, compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target(),
        )
        self.assertEqual(compiled["qa_count"], 4)
        self.assertTrue(compiled["truth_first_generate_by_construction"])
        primary = next(
            qa
            for qa in compiled["qas"]
            if qa["qa_id"] == compiled["target_fulfillment_qa_id"]
        )
        self.assertEqual(primary["proof"]["reasoning_event_ids"], ["E1", "E2"])
        materialized_events = plan["_truth_first_materialized"]["graph"]["events"]
        intermediate = materialized_events[0]["value"]
        self.assertNotIn(intermediate.casefold(), primary["question"].casefold())
        self.assertEqual(primary["gold_answer"], materialized_events[1]["value"])
        self.assertEqual(claim_ir["current_event_plan"], plan["_truth_first_materialized"]["graph"])

    def test_all_seven_primary_qas_pass_existing_stage64_quality_offline(self) -> None:
        operations = (
            "temporal_state_update",
            "bridge",
            "comparison",
            "visibility",
            "count",
            "cause_consequence",
            "relation_lookup",
        )
        for index, operation in enumerate(operations):
            with self.subTest(operation=operation):
                value = prepared(operation)
                plan, _meta = native.materialize_truth_first_graph_contract(
                    value,
                    valid_payload(value),
                    sample_index=index,
                    round_id="truth_native_all_ops",
                )
                claim_ir, compiled = native.precompile_truth_first_multiqa(
                    plan,
                    first_parent(),
                    min_qas=3,
                    max_qas=6,
                    immutable_stage64_target=target(operation),
                )
                primary = native.select_claim_first_primary_qa(
                    compiled, target(operation)
                )
                qa = native.render_claim_first_primary_qa(
                    primary,
                    claim_ir,
                    {
                        "graph_snapshot_hash": plan["graph_snapshot_hash"],
                        "dialogue_hash": "offline-dialogue",
                        "alignment_hash": "offline-alignment",
                    },
                )
                issues = stage64_quality.semantic_quality_issues(
                    {
                        "stage64_target": target(operation),
                        "graph_update": {
                            "graph_snapshot_id": plan["graph_snapshot_id"],
                            "events": copy.deepcopy(plan["events"]),
                            "metadata": {
                                "truth_first_generate_by_construction": True
                            },
                        },
                        "dialogue": {"turns": []},
                        "qa": qa,
                        "truth_first_generate_by_construction": True,
                    }
                )
                self.assertEqual(issues, [])
                self.assertEqual(compiled["qa_count"], 4)

    def test_all_twenty_eight_rows_pass_independent_stage64_quality_offline(self) -> None:
        operations = (
            "temporal_state_update",
            "bridge",
            "comparison",
            "visibility",
            "count",
            "cause_consequence",
            "relation_lookup",
        )
        accepted = 0
        for index, operation in enumerate(operations):
            with self.subTest(operation=operation):
                value = prepared(operation)
                plan, _meta = native.materialize_truth_first_graph_contract(
                    value,
                    valid_payload(value),
                    sample_index=index,
                    round_id="truth_native_all_siblings",
                )
                claim_ir, compiled = native.precompile_truth_first_multiqa(
                    plan,
                    first_parent(),
                    min_qas=3,
                    max_qas=6,
                    immutable_stage64_target=target(operation),
                )
                for raw_qa in compiled["qas"]:
                    qa = stage3_multiqa_rows.render_sibling_qa(raw_qa)
                    row_target = {
                        field: copy.deepcopy(qa[field])
                        for field in stage3_multiqa_rows.TARGET_FIELDS
                    }
                    row_target.update(
                        {
                            "preferred_answer_type": qa["answer_type"],
                            "forbid_max_time_shortcut": qa[
                                "forbid_max_time_shortcut"
                            ],
                            "target_basis": "truth_first_local_sibling_qa",
                            "source_qa_id": qa["qa_id"],
                        }
                    )
                    row_bundle = copy.deepcopy(compiled)
                    row_bundle["claim_ir"] = copy.deepcopy(claim_ir)
                    row_bundle["qas"] = [copy.deepcopy(qa)]
                    issues = stage64_quality.semantic_quality_issues(
                        {
                            "stage64_target": row_target,
                            "graph_update": {
                                "graph_snapshot_id": plan["graph_snapshot_id"],
                                "events": copy.deepcopy(plan["events"]),
                                "metadata": {
                                    "truth_first_generate_by_construction": True
                                },
                            },
                            "dialogue": {"turns": []},
                            "qa": qa,
                            "claim_first_multiqa_bundle": row_bundle,
                            "truth_first_generate_by_construction": True,
                        }
                    )
                    self.assertEqual(issues, [], raw_qa["qa_id"])
                    accepted += 1
        self.assertEqual(accepted, 28)

    def test_all_seven_offline_discourses_bind_and_expand_to_twenty_eight_clean_rows(self) -> None:
        operations = (
            "temporal_state_update",
            "bridge",
            "comparison",
            "visibility",
            "count",
            "cause_consequence",
            "relation_lookup",
        )
        total_rows = 0
        all_rows = []
        all_marked_rows = []
        for index, operation in enumerate(operations):
            with self.subTest(operation=operation):
                value = prepared(operation)
                plan, _meta = native.materialize_truth_first_graph_contract(
                    value,
                    valid_payload(value),
                    sample_index=index,
                    round_id="truth_native_offline_pipeline",
                )
                plan = native.assign_current_alignment_anchors(
                    plan, random.Random(100 + index)
                )
                claim_ir, compiled = native.precompile_truth_first_multiqa(
                    plan,
                    first_parent(),
                    min_qas=3,
                    max_qas=6,
                    immutable_stage64_target=target(operation),
                )
                parent_turns = native.dialogue_turns_from_row(first_parent())
                parent_context = native._truth_first_discourse_parent_context(
                    plan,
                    {"selected_relevant_parent_turns": parent_turns[-8:]},
                    parent_turns,
                )
                discourse_plan = discourse.build_discourse_plan(
                    plan["_truth_first_materialized"]["graph"],
                    claim_ir,
                    parent_context,
                )
                response = _response_from_discourse_prompt(
                    {"api_contract": discourse.build_api_contract(discourse_plan)}
                )
                response, normalization = discourse.normalize_api_response_envelope(
                    discourse_plan, response
                )
                self.assertTrue(normalization["pass"], normalization["issues"])
                discourse_validation = discourse.validate_api_response(
                    discourse_plan,
                    response,
                    frozen_graph=plan["_truth_first_materialized"]["graph"],
                    frozen_claim_bundle=claim_ir,
                    parent_context=parent_context,
                )
                self.assertTrue(
                    discourse_validation["pass"],
                    discourse_validation["issues"],
                )
                appended, claim_alignment = (
                    native._flatten_truth_first_discourse_response(
                        response, discourse_plan, claim_ir, plan
                    )
                )
                appended_by_turn = {
                    turn["turn_id"]: turn for turn in appended
                }
                event_to_anchor = plan["current_event_instance_to_anchor"]
                for claim in claim_ir["claims"]:
                    if (
                        claim["claim_type"] == "node"
                        or claim["claim_id"] not in claim_alignment
                    ):
                        continue
                    alignment = claim_alignment[claim["claim_id"]]
                    endpoint_anchors = {
                        event_to_anchor[event_id]
                        for event_id in [
                            claim["subject_event_id"],
                            *claim.get("object_event_ids", []),
                        ]
                        if event_id in event_to_anchor
                    }
                    self.assertTrue(
                        endpoint_anchors.issubset(
                            set(
                                appended_by_turn[alignment["turn_id"]][
                                    "alignment_anchors"
                                ]
                            )
                        ),
                        claim["claim_id"],
                    )
                final_dialogue = {
                    "turns": copy.deepcopy(parent_turns) + appended,
                    "dialogue_hash": f"offline-dialogue-{operation}",
                }
                bound = native.bind_validate_truth_first_multiqa(
                    current_event_plan=plan,
                    claim_alignment=claim_alignment,
                    final_dialogue=final_dialogue,
                    pre_claim_ir=claim_ir,
                    precompiled=compiled,
                    discourse_validation=discourse_validation,
                    min_qas=3,
                )
                self.assertTrue(bound["proof_ready"], bound["qa_rejects"])
                self.assertEqual(bound["qa_count"], 4)
                graph_hash = plan["graph_snapshot_hash"]
                alignment_hash = f"offline-alignment-{operation}"
                clean = {
                    "sample_id": f"offline-{operation}",
                    "source_id": "parent",
                    "parent_sample_id": "parent",
                    "lineage_group": "offline-root",
                    "dialogue_family_id": bound["dialogue_family_id"],
                    "dialogue": final_dialogue,
                    "graph_update": {
                        "graph_snapshot_id": plan["graph_snapshot_id"],
                        "graph_snapshot_hash": graph_hash,
                        "events": copy.deepcopy(plan["events"]),
                        "metadata": {
                            "truth_first_generate_by_construction": True
                        },
                    },
                    "metadata": {
                        "graph_snapshot_hash": graph_hash,
                        "dialogue_hash": final_dialogue["dialogue_hash"],
                        "alignment_hash": alignment_hash,
                        "lineage_group": "offline-root",
                    },
                    "native_contract": {"valid": True},
                    "validation": {
                        "graph_dialogue_consistent": True,
                        "gold_derivable": True,
                        "option_valid": True,
                        "anti_shortcut_pass": True,
                        "graph_validation_pass": True,
                    },
                    "claim_bundle_hash": bound["claim_bundle_hash"],
                    "claim_first_multiqa_bundle": bound,
                }
                envelope = {
                    "clean_sample": clean,
                    "hashes": {
                        "graph_snapshot_hash": graph_hash,
                        "dialogue_hash": final_dialogue["dialogue_hash"],
                        "alignment_hash": alignment_hash,
                    },
                    "claim_bundle_hash": bound["claim_bundle_hash"],
                    "claim_first_multiqa_bundle": bound,
                    "current_event_plan": plan,
                    "clean_dialogue": final_dialogue,
                    "current_alignment": {},
                }
                expanded = native.expand_truth_first_multiqa_dataset(
                    [clean], [envelope], min_qas=3, max_qas=6
                )
                self.assertEqual(expanded["qa_row_count"], 4)
                for row in expanded["qa_rows"]:
                    self.assertEqual(
                        stage64_quality.semantic_quality_issues(row),
                        [],
                        row["qa_id"],
                    )
                total_rows += expanded["qa_row_count"]
                all_rows.extend(copy.deepcopy(expanded["qa_rows"]))
                all_marked_rows.extend(
                    copy.deepcopy(expanded["marked_qa_rows"])
                )
        self.assertEqual(total_rows, 28)
        accepted, rejected, quality = (
            recalibrate_stage64_shuffled_frontier.quality_filter_samples(
                all_rows,
                near_duplicate_threshold=0.92,
                reject_near_duplicates=False,
                enforce_semantic_quality=True,
            )
        )
        self.assertEqual(len(accepted), 28, rejected)
        self.assertEqual(quality["accepted_count"], 28)
        shuffled, _shuffle_summary = shuffle_stage64_options.balance_options(
            accepted, seed=20260715
        )
        marked_by_id = {
            row["clean_sample"]["sample_id"]: row for row in all_marked_rows
        }
        for row in shuffled:
            stage3_solver_feedback_mining.stage64_target_freeze_payload(row)
            stage3_solver_feedback_mining.validate_marked_bundle(
                row, marked_by_id[row["sample_id"]]
            )

    def test_independent_sibling_quality_rejects_gold_graph_and_status_mutations(self) -> None:
        value = prepared("bridge")
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_mutation",
        )
        claim_ir, compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target("bridge"),
        )
        raw_qa = next(
            qa
            for qa in compiled["qas"]
            if qa["qa_subtype"] == "status_qualified_outdated_attribute_lookup"
        )
        qa = stage3_multiqa_rows.render_sibling_qa(raw_qa)
        bundle = copy.deepcopy(compiled)
        bundle["claim_ir"] = copy.deepcopy(claim_ir)
        bundle["qas"] = [copy.deepcopy(qa)]
        sample = {
            "graph_update": {
                "graph_snapshot_id": plan["graph_snapshot_id"],
                "events": copy.deepcopy(plan["events"]),
            },
            "dialogue": {"turns": []},
            "qa": qa,
            "claim_first_multiqa_bundle": bundle,
        }
        self.assertEqual(stage64_quality.semantic_quality_issues(sample), [])

        wrong_gold = copy.deepcopy(sample)
        wrong_gold["qa"]["gold_answer"] = "Zoe"
        self.assertIn(
            "truth_first_sibling_gold_projection_mismatch",
            {
                issue["reason"]
                for issue in stage64_quality.semantic_quality_issues(wrong_gold)
            },
        )
        graph_drift = copy.deepcopy(sample)
        graph_drift["graph_update"]["events"][3]["value"] = "Rhea"
        self.assertIn(
            "truth_first_sibling_compiled_truth_drift",
            {
                issue["reason"]
                for issue in stage64_quality.semantic_quality_issues(graph_drift)
            },
        )
        missing_status = copy.deepcopy(sample)
        missing_status["qa"]["question"] = "What is Trail Record's assigned partner?"
        missing_status["qa"]["question_text"] = missing_status["qa"]["question"]
        self.assertIn(
            "truth_first_sibling_status_qualifier_missing",
            {
                issue["reason"]
                for issue in stage64_quality.semantic_quality_issues(
                    missing_status
                )
            },
        )

    def test_segmented_discourse_and_post_bind_keep_four_semantic_qas(self) -> None:
        value = prepared()
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_test",
        )
        plan = native.assign_current_alignment_anchors(plan, random.Random(23))
        pre_claim_ir, compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target(),
        )
        parent_turns = native.dialogue_turns_from_row(first_parent())
        parent_context = {
            "selected_relevant_parent_turns": copy.deepcopy(parent_turns[-8:]),
            "parent_dialogue_digest": {"source": "unit_test"},
        }

        def fake_api(_args, prompt, **_kwargs):
            bare_turns = _response_from_discourse_prompt(prompt)["turns"]
            response, diagnostics = (
                native.normalize_truth_discourse_segment_bare_list_payload(
                    bare_turns
                )
            )
            if prompt["segment_index"] == 1:
                response["turns"].append(
                    copy.deepcopy(response["turns"][0])
                )
            return response, {
                "used": True,
                "attempts": 1,
                "model": "mock",
                "base_url": "mock",
                "normalizer_diagnostics": diagnostics,
            }

        with mock.patch.object(native, "call_raw_json_api", side_effect=fake_api) as api:
            proposal, dialogue_meta = native.call_api_truth_first_discourse_generation(
                plan,
                pre_claim_ir,
                api_args(),
                parent_context=parent_context,
                parent_dialogue_turns=parent_turns,
                min_added_turns=6,
                min_added_tokens=512,
            )
        self.assertEqual(api.call_count, 4)
        live_prompts = [call.args[1] for call in api.call_args_list]
        self.assertEqual(
            [prompt["segment_index"] for prompt in live_prompts], [0, 1, 2, 3]
        )
        self.assertEqual(
            [prompt["segment_count"] for prompt in live_prompts], [4, 4, 4, 4]
        )
        for live_prompt in live_prompts:
            self.assertNotIn("parent_dialogue_tail", live_prompt)
            self.assertEqual(
                live_prompt["parent_context_exposure"],
                "redacted_topic_speaker_roster_and_validated_api_neutral_prefix_only",
            )
            prompt_text = json.dumps(live_prompt, ensure_ascii=False).casefold()
            for claim in pre_claim_ir["claims"]:
                canonical = claim.get("canonical_support_claim") or {}
                for field in ("object", "value", "status", "condition"):
                    surface = str(canonical.get(field) or "").strip().casefold()
                    if len(surface) >= 4:
                        self.assertNotIn(surface, prompt_text)
        first_prefix = live_prompts[0]["api_contract"]["read_only_brief"][
            "accepted_api_neutral_prefix_tail"
        ]
        self.assertEqual(first_prefix, [])
        self.assertTrue(
            all(
                prompt["api_contract"]["read_only_brief"][
                    "accepted_api_neutral_prefix_tail"
                ]
                for prompt in live_prompts[1:]
            )
        )
        self.assertTrue(
            all(
                call.kwargs["required_keys"] == ("turns",)
                for call in api.call_args_list
            )
        )
        self.assertTrue(dialogue_meta["used"], dialogue_meta)
        self.assertEqual(dialogue_meta["segment_count"], 4)
        self.assertEqual(dialogue_meta["segment_api_call_count"], 4)
        self.assertEqual(dialogue_meta["overflow_suffix_normalized_count"], 1)
        self.assertEqual(dialogue_meta["overflow_suffix_row_count"], 1)
        self.assertTrue(
            dialogue_meta["segment_attempt_reports"][1][
                "overflow_normalization"
            ]["retained_prefix_content_preserved"]
        )
        self.assertTrue(
            all(
                report["api_meta"]["normalizer_diagnostics"][
                    "truth_discourse_segment_bare_list_normalizer_used"
                ]
                for report in dialogue_meta["segment_attempt_reports"]
            )
        )
        self.assertTrue(
            dialogue_meta["response_envelope_normalization"]["pass"]
        )
        self.assertEqual(dialogue_meta["fact_only_chunk_count"], 0)
        self.assertEqual(dialogue_meta["continuation_call_count"], 0)
        appended = proposal["appended_dialogue_turns"]
        final_dialogue = {
            "turns": copy.deepcopy(parent_turns) + copy.deepcopy(appended),
            "dialogue_hash": "unit-test-dialogue-hash",
        }
        bound = native.bind_validate_truth_first_multiqa(
            current_event_plan=plan,
            claim_alignment=proposal["truth_first_claim_alignment"],
            final_dialogue=final_dialogue,
            pre_claim_ir=pre_claim_ir,
            precompiled=compiled,
            discourse_validation=proposal["truth_first_discourse_validation"],
            min_qas=3,
        )
        self.assertTrue(bound["proof_ready"], bound.get("qa_rejects"))
        self.assertTrue(bound["target_fulfillment_pass"])
        self.assertEqual(bound["qa_count"], 4)
        self.assertEqual(bound["semantic_validator_pass_qa_count"], 4)
        self.assertTrue(all(qa["evidence_ledger"] for qa in bound["qas"]))
        primary = native.select_claim_first_primary_qa(bound, target())
        rendered = native.render_claim_first_primary_qa(
            primary,
            bound["claim_ir"],
            {
                "graph_snapshot_hash": plan["graph_snapshot_hash"],
                "dialogue_hash": final_dialogue["dialogue_hash"],
                "alignment_hash": "unit-test-alignment-hash",
            },
        )
        self.assertEqual(rendered["qa_id"], bound["target_fulfillment_qa_id"])
        self.assertEqual(rendered["gold_answer_text"], plan["events"][1]["value"])
        quality_issues = stage64_quality.semantic_quality_issues(
            {
                "stage64_target": target(),
                "graph_update": {
                    "graph_snapshot_id": plan["graph_snapshot_id"],
                    "events": copy.deepcopy(plan["events"]),
                    "metadata": {"truth_first_generate_by_construction": True},
                },
                "dialogue": final_dialogue,
                "qa": rendered,
                "truth_first_generate_by_construction": True,
            }
        )
        self.assertEqual(quality_issues, [])

    def test_failed_second_segment_retries_without_regenerating_prefix(self) -> None:
        value = prepared()
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_targeted_act_retry",
        )
        plan = native.assign_current_alignment_anchors(plan, random.Random(73))
        pre_claim_ir, _compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target(),
        )
        parent_turns = native.dialogue_turns_from_row(first_parent())
        parent_context = {
            "selected_relevant_parent_turns": copy.deepcopy(parent_turns[-8:]),
            "parent_dialogue_digest": {"source": "unit_test"},
        }
        calls = []

        def fake_api(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            body = _response_from_discourse_prompt(prompt)
            if prompt["segment_index"] == 1 and len(calls) == 2:
                anchor = prompt["api_contract"]["read_only_brief"][
                    "topic_spine"
                ]["anchor_terms"][0]
                for row in body["turns"]:
                    row["context_sentences"] = [
                        f"The {anchor} discussion stays focused for everyone involved today.",
                        f"Each participant considers the {anchor} topic with appropriate care.",
                    ]
            return body, {
                "used": True,
                "attempts": 1,
                "model": "mock",
                "base_url": "mock",
            }

        with mock.patch.object(native, "call_raw_json_api", side_effect=fake_api) as api:
            proposal, meta = native.call_api_truth_first_discourse_generation(
                plan,
                pre_claim_ir,
                api_args(),
                parent_context=parent_context,
                parent_dialogue_turns=parent_turns,
                min_added_turns=6,
                min_added_tokens=512,
            )

        self.assertTrue(meta["used"], meta)
        self.assertIsInstance(proposal, dict)
        self.assertEqual(api.call_count, 5)
        self.assertEqual(
            [row["segment_index"] for row in calls], [0, 1, 1, 2, 3]
        )
        first_segment_calls = [
            row for row in calls if row["segment_index"] == 0
        ]
        self.assertEqual(len(first_segment_calls), 1)
        retry_feedback = calls[2]["api_contract"]["read_only_brief"][
            "retry_feedback"
        ]
        self.assertEqual(
            retry_feedback["previous_failure_stage"],
            "local_segment_validation",
        )
        self.assertGreater(retry_feedback["context_word_deficit"], 0)
        self.assertGreater(retry_feedback["minimum_context_words"], 0)
        self.assertGreater(retry_feedback["previous_context_words"], 0)
        self.assertIn("entire segment from zero", retry_feedback["instruction"])
        self.assertIn("full minimum_context_words", retry_feedback["instruction"])
        first_second_prefix = calls[1]["api_contract"]["read_only_brief"][
            "accepted_api_neutral_prefix_tail"
        ]
        retried_second_prefix = calls[2]["api_contract"]["read_only_brief"][
            "accepted_api_neutral_prefix_tail"
        ]
        self.assertEqual(first_second_prefix, retried_second_prefix)

    def test_api_usage_accounting_accumulates_every_recorded_attempt(self) -> None:
        summary = {}
        native._record_api_meta_retry_diagnostics(  # noqa: SLF001
            summary,
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                }
            },
        )
        native._record_api_meta_retry_diagnostics(  # noqa: SLF001
            summary,
            {
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                }
            },
        )
        self.assertEqual(summary["api_usage_prompt_tokens"], 180)
        self.assertEqual(summary["api_usage_completion_tokens"], 60)
        self.assertEqual(summary["api_usage_total_tokens"], 240)

    def test_truth_discourse_bare_list_adapter_is_strict_and_lossless(self) -> None:
        rows = [
            {
                "turn_slot_id": "TS001",
                "context_sentences": [
                    "The workshop discussion remains practical and focused today."
                ],
            },
            {
                "turn_slot_id": "TS002",
                "context_sentences": [
                    "Everyone continues that workshop thread without changing its facts."
                ],
            },
        ]
        before = copy.deepcopy(rows)
        wrapped, diagnostics = (
            native.normalize_truth_discourse_segment_bare_list_payload(rows)
        )
        self.assertEqual(rows, before)
        self.assertEqual(wrapped, {"turns": before})
        self.assertTrue(
            diagnostics["truth_discourse_segment_bare_list_normalizer_used"]
        )
        self.assertTrue(
            diagnostics["truth_discourse_segment_bare_list_content_preserved"]
        )
        self.assertEqual(
            diagnostics["truth_discourse_segment_bare_list_original_sha256"],
            diagnostics["truth_discourse_segment_bare_list_wrapped_sha256"],
        )

        malformed = copy.deepcopy(rows)
        malformed[0]["speaker"] = "Maya"
        rejected, rejected_diagnostics = (
            native.normalize_truth_discourse_segment_bare_list_payload(malformed)
        )
        self.assertIs(rejected, malformed)
        self.assertFalse(
            rejected_diagnostics[
                "truth_discourse_segment_bare_list_normalizer_used"
            ]
        )
        self.assertTrue(
            rejected_diagnostics[
                "truth_discourse_segment_bare_list_normalizer_failure"
            ]
        )

    def test_raw_json_api_accepts_truth_discourse_bare_list(self) -> None:
        rows = [
            {
                "turn_slot_id": "TS001",
                "context_sentences": [
                    "The workshop discussion remains practical and focused today."
                ],
            }
        ]
        captured_requests = []

        def respond(_url, request_payload, _api_key, _timeout):
            captured_requests.append(copy.deepcopy(request_payload))
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps(rows)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }

        args = api_args()
        args.challenger_api_key = "test-key"
        args.challenger_base_url = "https://example.invalid/v1"
        args.challenger_model = "test-model"
        args.llm_timeout = 1
        args.api_yield_normalizers = False
        args.max_llm_retries = 0
        with mock.patch.object(
            native.old_gate,
            "post_json_with_curl",
            side_effect=respond,
        ):
            payload, meta = native.call_raw_json_api(
                args,
                {"task": "truth discourse bare list"},
                system_prompt="Return JSON only.",
                required_keys=("turns",),
                call_kind="api_truth_discourse_segment_generation",
                max_retries_override=0,
            )

        self.assertEqual(payload, {"turns": rows})
        self.assertTrue(meta["used"])
        self.assertEqual(
            captured_requests[0]["response_format"], {"type": "json_object"}
        )
        self.assertTrue(
            meta["normalizer_diagnostics"][
                "truth_discourse_segment_bare_list_normalizer_used"
            ]
        )
        self.assertTrue(
            meta["normalizer_diagnostics"][
                "truth_discourse_segment_bare_list_content_preserved"
            ]
        )

    def test_raw_json_api_rejects_single_truth_discourse_turn_object(self) -> None:
        singleton = {
            "turn_slot_id": "TS001",
            "context_sentences": [
                "The workshop discussion remains practical for everyone today.",
                "Participants continue that topic with careful attention to context.",
            ],
        }

        def respond(_url, _request_payload, _api_key, _timeout):
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps(singleton)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }

        args = api_args()
        args.challenger_api_key = "test-key"
        args.challenger_base_url = "https://example.invalid/v1"
        args.challenger_model = "test-model"
        args.llm_timeout = 1
        args.api_yield_normalizers = False
        args.max_llm_retries = 0
        with mock.patch.object(
            native.old_gate,
            "post_json_with_curl",
            side_effect=respond,
        ):
            payload, meta = native.call_raw_json_api(
                args,
                {"task": "truth discourse singleton"},
                system_prompt="Return JSON only.",
                required_keys=("turns",),
                call_kind="api_truth_discourse_segment_generation",
                max_retries_override=0,
            )

        self.assertIsNone(payload)
        self.assertFalse(meta["used"])
        self.assertEqual(meta["error_type"], "missing_required_keys")
        self.assertEqual(meta["usage"]["total_tokens"], 30)
        self.assertEqual(len(meta["errors"]), 1)
        self.assertEqual(meta["errors"][0]["raw_response"], json.dumps(singleton))

    def test_raw_json_api_accounts_for_parse_failed_usage(self) -> None:
        responses = [
            {
                "choices": [
                    {
                        "message": {"content": "not-json"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
            {
                "choices": [
                    {
                        "message": {"content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 21,
                    "total_tokens": 32,
                },
            },
        ]
        args = api_args()
        args.challenger_api_key = "test-key"
        args.challenger_base_url = "https://example.invalid/v1"
        args.challenger_model = "test-model"
        args.llm_timeout = 1
        args.api_yield_normalizers = False
        args.max_llm_retries = 1
        with mock.patch.object(
            native.old_gate,
            "post_json_with_curl",
            side_effect=responses,
        ), mock.patch.object(native.time, "sleep"):
            payload, meta = native.call_raw_json_api(
                args,
                {"task": "usage accounting"},
                system_prompt="Return JSON only.",
                required_keys=("ok",),
                call_kind="api_graph_expansion",
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(
            meta["usage"],
            {
                "prompt_tokens": 21,
                "completion_tokens": 41,
                "total_tokens": 62,
            },
        )
        self.assertEqual(meta["last_attempt_usage"], responses[1]["usage"])
        self.assertEqual(len(meta["previous_errors"]), 1)

    def test_truth_first_post_bind_uses_only_independent_typed_support_validator(
        self,
    ) -> None:
        value = prepared("bridge")
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_typed_post_bind",
        )
        pre_claim_ir, compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target("bridge"),
        )
        claim_alignment: dict[str, dict[str, object]] = {}
        turns = []
        for index, claim in enumerate(pre_claim_ir["claims"]):
            quote = native.stage3_truth_support.compile_typed_frozen_support(
                claim
            )["quote"]
            turn_id = f"typed-post-bind:{index + 1}"
            alignment = {
                "turn_id": turn_id,
                "exact_support_quote": quote,
            }
            claim_alignment[claim["claim_id"]] = copy.deepcopy(alignment)
            if claim["claim_type"] == "node":
                claim_alignment[claim["subject_event_id"]] = copy.deepcopy(
                    alignment
                )
            turns.append(
                {"turn_id": turn_id, "speaker": "A", "text": quote}
            )

        typed_pass = {
            "schema_version": (
                native.stage3_truth_support.VALIDATION_SCHEMA_VERSION
            ),
            "pass": True,
            "issues": [],
        }
        with mock.patch.object(
            native.stage3_claim_validator,
            "_validate_quote_semantics",
            side_effect=AssertionError(
                "legacy quote validator must not run in Truth-first post-bind"
            ),
        ) as legacy_validator, mock.patch.object(
            native.stage3_truth_support,
            "validate_typed_frozen_support",
            return_value=typed_pass,
        ) as typed_validator:
            bound = native.bind_validate_truth_first_multiqa(
                current_event_plan=plan,
                claim_alignment=claim_alignment,
                final_dialogue={"turns": turns},
                pre_claim_ir=pre_claim_ir,
                precompiled=compiled,
                discourse_validation={"pass": True},
                min_qas=3,
            )

        expected_calls = sum(
            len(qa["proof"]["required_claim_ids"])
            for qa in compiled["qas"]
        )
        # Each required claim is admitted once while its frozen record is
        # compiled and once again against the bound dialogue quote.
        self.assertEqual(typed_validator.call_count, expected_calls * 2)
        self.assertTrue(
            all(
                isinstance(call.args[1], dict)
                and call.args[1].get("schema_version")
                == native.stage3_truth_support.SCHEMA_VERSION
                for call in typed_validator.call_args_list
            )
        )
        self.assertEqual(legacy_validator.call_count, 0)
        self.assertEqual(bound["qa_count"], 4)
        self.assertTrue(bound["proof_ready"])
        for report in bound[
            "independent_semantic_validation_reports"
        ].values():
            self.assertEqual(
                report["support_validator"],
                "independent_typed_truth_support",
            )
            self.assertEqual(
                report["support_validator_schema_version"],
                native.stage3_truth_support.VALIDATION_SCHEMA_VERSION,
            )

    def test_live_discourse_rejects_nested_only_api_body(self) -> None:
        operation = "comparison"
        value = prepared(operation)
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_nested_reject",
        )
        plan = native.assign_current_alignment_anchors(plan, random.Random(31))
        pre_claim_ir, _compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target(operation),
        )
        parent_turns = native.dialogue_turns_from_row(first_parent())
        parent_context = {
            "selected_relevant_parent_turns": copy.deepcopy(parent_turns[-8:]),
            "parent_dialogue_digest": {"source": "unit_test"},
        }

        def fake_nested_api(_args, _prompt, **_kwargs):
            return {"acts": []}, {
                "used": True,
                "attempts": 1,
                "model": "mock",
                "base_url": "mock",
            }

        with mock.patch.object(
            native, "call_raw_json_api", side_effect=fake_nested_api
        ) as api:
            response, dialogue_meta = (
                native.call_api_truth_first_discourse_generation(
                    plan,
                    pre_claim_ir,
                    api_args(),
                    parent_context=parent_context,
                    parent_dialogue_turns=parent_turns,
                    min_added_turns=6,
                    min_added_tokens=512,
                )
            )

        self.assertEqual(api.call_count, 2)
        self.assertEqual(response["failed_segment_id"], "A1B1")
        self.assertEqual(response["accepted_prefix_turns"], [])
        self.assertFalse(dialogue_meta["used"])
        self.assertEqual(
            dialogue_meta["reason"],
            "truth_first_discourse_segment_validation_failed",
        )
        self.assertEqual(dialogue_meta["segment_api_call_count"], 2)
        validations = [
            row["segment_validation"]
            for row in dialogue_meta["segment_attempt_reports"]
        ]
        self.assertTrue(all(not report["pass"] for report in validations))
        self.assertTrue(
            all(
                "segment_turns_not_array"
                in {issue["code"] for issue in report["issues"]}
                for report in validations
            )
        )
        self.assertTrue(
            all(
                row["response_payload"] == {"acts": []}
                for row in dialogue_meta["segment_attempt_reports"]
            )
        )

    def test_live_discourse_classifies_missing_turns_as_response_contract(self) -> None:
        operation = "comparison"
        value = prepared(operation)
        plan, _meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_response_contract",
        )
        plan = native.assign_current_alignment_anchors(plan, random.Random(41))
        pre_claim_ir, _compiled = native.precompile_truth_first_multiqa(
            plan,
            first_parent(),
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=target(operation),
        )
        parent_turns = native.dialogue_turns_from_row(first_parent())
        parent_context = {
            "selected_relevant_parent_turns": copy.deepcopy(parent_turns[-8:]),
            "parent_dialogue_digest": {"source": "unit_test"},
        }
        prompts = []

        def fake_missing_turns(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            return None, {
                "used": False,
                "call_kind": "api_truth_discourse_segment_generation",
                "attempts": 1,
                "error_type": "missing_required_keys",
                "errors": [{"missing_keys": ["turns"]}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

        with mock.patch.object(
            native,
            "call_raw_json_api",
            side_effect=fake_missing_turns,
        ) as api:
            response, dialogue_meta = (
                native.call_api_truth_first_discourse_generation(
                    plan,
                    pre_claim_ir,
                    api_args(),
                    parent_context=parent_context,
                    parent_dialogue_turns=parent_turns,
                    min_added_turns=6,
                    min_added_tokens=512,
                )
            )

        self.assertEqual(api.call_count, 2)
        self.assertEqual(response["failed_segment_id"], "A1B1")
        self.assertEqual(
            dialogue_meta["reason"],
            "truth_first_discourse_segment_response_contract_failed",
        )
        self.assertTrue(
            all(
                row["failure_stage"] == "api_response_contract"
                for row in dialogue_meta["segment_attempt_reports"]
            )
        )
        retry_feedback = prompts[1]["api_contract"]["read_only_brief"][
            "retry_feedback"
        ]
        self.assertEqual(retry_feedback["required_top_level_key"], "turns")
        self.assertGreater(retry_feedback["expected_turn_count"], 1)
        self.assertEqual(
            len(retry_feedback["required_turn_slot_ids"]),
            retry_feedback["expected_turn_count"],
        )

    def test_run_local_artifact_reload_revalidates_constructor_manifest(self) -> None:
        value = prepared()
        plan, local_meta = native.materialize_truth_first_graph_contract(
            value,
            valid_payload(value),
            sample_index=0,
            round_id="truth_native_artifact",
        )
        plan = native.assign_current_alignment_anchors(plan, random.Random(29))
        source_meta = {
            **local_meta,
            "truth_first_generate_by_construction": True,
        }
        artifact = native.build_stage3_graph_artifact(
            seed_row=first_parent(),
            current_event_plan=plan,
            child_graph_delta=copy.deepcopy(
                plan["_truth_first_materialized"]["graph"]
            ),
            stage64_target=target(),
            stage64_target_identity="f" * 64,
            source_graph_meta=source_meta,
        )
        self.assertEqual(
            artifact["generation_contract"],
            native.TRUTH_FIRST_GENERATION_CONTRACT,
        )
        issues = native.fixed_stage3_graph_artifact_issues(
            artifact,
            seed_row=first_parent(),
            stage64_target_identity="f" * 64,
            stage64_target=target(),
            truth_first_mode=True,
        )
        self.assertEqual(issues, [])
        reuse_meta = native.fixed_stage3_graph_artifact_reuse_graph_meta(
            artifact,
            expected_operation="bridge",
            artifact_source="materialized_reuse",
            artifact_path="graph_artifacts/f.json",
            truth_first_mode=True,
        )
        self.assertTrue(reuse_meta["truth_first_generate_by_construction"])
        self.assertTrue(reuse_meta["materialization_admission"]["pass"])

        tampered = copy.deepcopy(artifact)
        tampered["current_event_plan"]["events"][0]["slot"] = "discovery_date"
        tamper_codes = {
            issue["code"]
            for issue in native.fixed_stage3_graph_artifact_issues(
                tampered,
                seed_row=first_parent(),
                stage64_target_identity="f" * 64,
                stage64_target=target(),
                truth_first_mode=True,
            )
        }
        self.assertIn("truth_first_artifact_event_truth_mismatch", tamper_codes)
        self.assertIn("fixed_graph_declared_plan_hash_invalid", tamper_codes)


if __name__ == "__main__":
    unittest.main()
