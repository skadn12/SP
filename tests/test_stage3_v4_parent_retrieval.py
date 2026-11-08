from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.current_method import native_api_graph_dialogue_expansion as subject


ROOT = Path(__file__).resolve().parents[1]
V4_PARENTS = ROOT / "stage3-data-fix" / "stage3_parent_v4_memory_backfilled.jsonl"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def turn_text_by_id(row: dict) -> dict[str, str]:
    return {
        str(turn.get("turn_id") or turn.get("id") or turn.get("dia_id")): str(
            turn.get("text") or turn.get("content") or turn.get("utterance") or ""
        ).strip()
        for turn in row["dialogue"]["turns"]
    }


def stage3_policy(operation: str) -> dict:
    return {
        "native_curriculum_mode": subject.EASY_V2_MODE,
        "planned_cell": {
            "question_type": "single_choice",
            "attack_skill": "conditional_plan",
        },
        "stage64_target": {
            "qa_operation": operation,
            "preferred_answer_type": "ObjectOrCommonNoun",
        },
    }


def relation_payload(prompt: dict) -> dict:
    selected_parent_ids = set(
        prompt["seed"]["parent_event_retrieval"]["selected_parent_event_ids"]
    )
    selected_parent_events = [
        event
        for event in prompt["seed"]["semantic_parent_graph"]["events"]
        if event.get("event_id") in selected_parent_ids
    ]
    parent_anchor = next(
        (
            event
            for event in selected_parent_events
            if "relationship" in str(event.get("slot") or "").casefold()
        ),
        selected_parent_events[0],
    )
    selected_parent_id = parent_anchor["event_id"]
    parent_entity = str(parent_anchor["entity"])
    parent_slot = str(parent_anchor.get("slot") or "relationship")
    return {
        "schema_version": "semantic_graph_v2",
        "graph_id": "stage3:S0000:semantic_graph_v2",
        "parent_graph_id": prompt["seed"]["semantic_parent_graph"]["graph_id"],
        "operation": "relation_lookup",
        "question_type": "single_choice",
        "attack_skill": "conditional_plan",
        "events": [
            {
                "event_id": "E1",
                "entity": parent_entity,
                "slot": parent_slot,
                "value": parent_entity,
                "status": "active",
                "time_index": 1,
            },
            {
                "event_id": "E2",
                "entity": parent_entity,
                "slot": "desk_item",
                "value": "red notebook",
                "status": "active",
                "time_index": 2,
                "depends_on": ["E1"],
            },
            {
                "event_id": "E3",
                "entity": parent_entity,
                "slot": "desk_item",
                "value": "blue notebook",
                "status": "outdated",
                "time_index": 3,
            },
            {
                "event_id": "E4",
                "entity": "Noah",
                "slot": "desk_item",
                "value": "red folder",
                "status": "active",
                "time_index": 4,
            },
            {
                "event_id": "E5",
                "entity": parent_entity,
                "slot": "shelf_item",
                "value": "brass key",
                "status": "active",
                "time_index": 5,
            },
            {
                "event_id": "E6",
                "entity": "Li",
                "slot": "locker_item",
                "value": "silver badge",
                "status": "active",
                "time_index": 6,
            },
            {
                "event_id": "E7",
                "entity": "Omar",
                "slot": "train_route",
                "value": "West",
                "status": "active",
                "time_index": 7,
            },
            {
                "event_id": "E8",
                "entity": "Pia",
                "slot": "meeting_room",
                "value": "Cedar",
                "status": "active",
                "time_index": 8,
            },
        ],
        "false_candidates": [],
        "metadata": {
            "parent_continuity": {
                "depends_on_parent_event_ids": [selected_parent_id],
                "parent_topic_anchor_event_ids": [selected_parent_id],
                "continuity_relations": [
                    {
                        "parent_event_id": selected_parent_id,
                        "child_event_id": "E1",
                        "relation": "continues",
                    }
                ],
            }
        },
    }


class Stage3V4ParentRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in V4_PARENTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_full_backfill_file_counts_evidence_ids_duplicates_and_leaks(self) -> None:
        total_events = 0
        added_events = 0
        for row_index, row in enumerate(self.rows):
            turns = turn_text_by_id(row)
            events = row["semantic_parent_graph"]["events"]
            event_ids = [str(event.get("event_id") or "") for event in events]
            with self.subTest(row_index=row_index):
                self.assertEqual(len(event_ids), len(set(event_ids)))
                self.assertTrue(all(event_ids))
            total_events += len(events)
            for event_index, event in enumerate(events):
                if event.get("node_type") in {
                    "atomic_memory_event",
                    "session_event_summary",
                }:
                    added_events += 1
                with self.subTest(row_index=row_index, event_index=event_index):
                    self.assertTrue(
                        set(event.get("evidence_turn_ids") or []).issubset(turns)
                    )
                    self.assertFalse(subject._v4_parent_atom_leak_issues(event))
                    serialized = json.dumps(event, ensure_ascii=False)
                    self.assertIsNone(
                        re.search(r"raw[_\s-]*qa|(?:^|\s)[QA]\s*:", serialized, re.I)
                    )
        self.assertEqual(total_events, 9105)
        self.assertEqual(added_events, 8720)

    def test_all_77_rows_have_grounded_operation_conditioned_views(self) -> None:
        self.assertEqual(len(self.rows), 77)
        total_ineligible = 0
        for row_index, row in enumerate(self.rows):
            with self.subTest(row_index=row_index):
                self.assertEqual(row["schema_version"], "stage3_parent_v4_memory_backfilled")
                before = canonical_sha256(row)
                source_by_id = {
                    event["event_id"]: event
                    for event in row["semantic_parent_graph"]["events"]
                }
                selection_sets: set[tuple[str, ...]] = set()
                for operation in subject.STAGE64_QA_OPERATIONS:
                    graph, ledger = subject.build_v4_parent_prompt_view(
                        row,
                        operation=operation,
                    )
                    full_ledger_out: dict = {}
                    prompt_seed = subject.compact_seed_for_prompt(
                        row,
                        semantic_graph_v2=True,
                        parent_operation=operation,
                        full_parent_event_retrieval_out=full_ledger_out,
                    )
                    self.assertEqual(full_ledger_out, ledger)
                    self.assertEqual(prompt_seed["semantic_parent_graph"], graph)
                    compact_ledger = prompt_seed["parent_event_retrieval"]
                    self.assertEqual(
                        set(compact_ledger),
                        set(subject.V4_PARENT_PROMPT_LEDGER_FIELDS),
                    )
                    self.assertNotIn("retrieval_ledger", graph)
                    selected_ids = ledger["selected_parent_event_ids"]
                    selection_sets.add(tuple(selected_ids))
                    self.assertEqual(ledger["retrieval_operation"], operation)
                    self.assertTrue(ledger["retrieval_reason"])
                    self.assertEqual(ledger["full_parent_event_count"], len(source_by_id))
                    self.assertEqual(
                        ledger["eligible_parent_event_count"],
                        len(ledger["eligible_parent_event_ids"]),
                    )
                    self.assertEqual(
                        ledger["selected_parent_event_count"], len(selected_ids)
                    )
                    self.assertGreaterEqual(
                        len(selected_ids), subject.V4_PARENT_RETRIEVAL_MIN_EVENTS
                    )
                    self.assertLessEqual(
                        len(selected_ids), subject.V4_PARENT_RETRIEVAL_MAX_EVENTS
                    )
                    self.assertTrue(
                        set(selected_ids).issubset(
                            set(ledger["eligible_parent_event_ids"])
                        )
                    )
                    self.assertEqual(
                        [event["event_id"] for event in graph["events"]],
                        selected_ids,
                    )
                    for prompt_event in graph["events"]:
                        source_event = source_by_id[prompt_event["event_id"]]
                        self.assertTrue(
                            set(prompt_event).issubset(
                                set(subject.V4_PARENT_PROMPT_EVENT_FIELDS)
                            )
                        )
                        self.assertEqual(
                            prompt_event["node_type"],
                            source_event.get("node_type", "canonical_event"),
                        )
                        for heavy_field in (
                            "memory_text",
                            "supporting_span",
                            "supporting_spans",
                            "provenance",
                        ):
                            self.assertNotIn(heavy_field, prompt_event)
                        self.assertIn("evidence_turn_ids", prompt_event)
                        self.assertTrue(
                            "time_index" in prompt_event
                            or "global_time_index" in prompt_event
                        )
                        compatibility_event = subject._v4_parent_compatibility_event(
                            source_event
                        )
                        for key, value in source_event.items():
                            self.assertIn(key, compatibility_event)
                            self.assertEqual(compatibility_event[key], value)
                    for factor in (
                        "session_recency",
                        "evidence_position",
                        "topic_continuity",
                    ):
                        self.assertGreater(
                            ledger["scoring_factor_hit_counts"][factor], 0
                        )
                    selected_reason_codes = [
                        reason
                        for record in ledger["selection_reasons"]
                        for reason in record["reason_codes"]
                    ]
                    self.assertTrue(
                        any(
                            reason.startswith("session_recency:")
                            for reason in selected_reason_codes
                        )
                    )
                    self.assertTrue(
                        any(
                            reason.startswith("evidence_position:")
                            for reason in selected_reason_codes
                        )
                    )
                    self.assertTrue(
                        any(
                            reason.startswith("topic_continuity:")
                            for reason in selected_reason_codes
                        )
                    )
                    self.assertTrue(
                        all(
                            not subject.v4_parent_event_eligibility_issues(
                                source_by_id[event["event_id"]],
                                turn_text_by_id(row),
                            )
                            for event in graph["events"]
                        )
                    )
                    self.assertFalse(
                        any(
                            subject._v4_parent_atom_leak_issues(event)
                            for event in graph["events"]
                        )
                    )
                    if operation == subject.STAGE64_QA_OPERATIONS[-1]:
                        total_ineligible += ledger["ineligible_event_count"]
                self.assertGreaterEqual(len(selection_sets), 2)
                self.assertEqual(canonical_sha256(row), before)
        # These are real ID-present-but-span-mismatched v4 events. They remain
        # auditable in full_parent_event_ids and never enter a selected view.
        self.assertEqual(total_ineligible, 830)

    def test_selected_atom_prompt_audit_is_path_scoped(self) -> None:
        row = self.rows[0]
        seed = subject.compact_seed_for_prompt(
            row,
            semantic_graph_v2=True,
            parent_operation="relation_lookup",
        )
        audit = subject.audit_graph_parent_prompt_verbatim_exposure(
            {"seed": seed}, row
        )
        self.assertTrue(audit["pass"])
        self.assertTrue(audit["v4_selected_parent_fact_view"])
        self.assertEqual(audit["v4_selected_parent_fact_count"], 24)

        rogue_prompt = {
            "seed": seed,
            "rogue_parent_text": row["dialogue"]["turns"][0]["text"],
        }
        rogue_audit = subject.audit_graph_parent_prompt_verbatim_exposure(
            rogue_prompt, row
        )
        self.assertFalse(rogue_audit["pass"])
        self.assertIn("$.rogue_parent_text", rogue_audit["exposed_prompt_paths"])

    def test_qa_wrapper_and_mismatched_support_are_ineligible(self) -> None:
        row = self.rows[0]
        event = copy.deepcopy(row["semantic_parent_graph"]["events"][5])
        event["value"] = "Q: hidden prompt A: leaked answer"
        issues = subject.v4_parent_event_eligibility_issues(
            event, turn_text_by_id(row)
        )
        self.assertIn("v4_parent_qa_text_leak", {issue["code"] for issue in issues})

        event = copy.deepcopy(row["semantic_parent_graph"]["events"][5])
        event["supporting_spans"] = ["This span does not occur in its cited turn."]
        issues = subject.v4_parent_event_eligibility_issues(
            event, turn_text_by_id(row)
        )
        self.assertIn(
            "v4_parent_supporting_span_mismatch",
            {issue["code"] for issue in issues},
        )

    def test_requested_retrieval_size_is_clamped_to_15_40(self) -> None:
        row = self.rows[0]
        _small_view, small = subject.build_v4_parent_prompt_view(
            row, operation="bridge", target_event_count=1
        )
        _large_view, large = subject.build_v4_parent_prompt_view(
            row, operation="bridge", target_event_count=100
        )
        self.assertEqual(small["selected_parent_event_count"], 15)
        self.assertEqual(large["selected_parent_event_count"], 40)

    def test_n1_temporal_prompt_is_lean_and_timeout_does_not_semantic_retry(self) -> None:
        row = self.rows[0]
        source_before = canonical_sha256(row)
        policy = stage3_policy("temporal_state_update")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        captured: list[dict] = []

        def exhausted_timeout(_args, prompt, **_kwargs):
            captured.append(copy.deepcopy(prompt))
            return None, {
                "used": False,
                "call_kind": "api_graph_expansion",
                "attempts": 2,
                "error_type": "timeout",
                "errors": [
                    {"attempt": 1, "error_type": "timeout"},
                    {"attempt": 2, "error_type": "timeout"},
                ],
            }

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            side_effect=exhausted_timeout,
        ) as semantic_api:
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "exhausted transport timeout retries",
            ) as caught:
                subject.call_api_graph_expansion(
                    row,
                    {},
                    policy,
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=3,
                    ),
                    random.Random(20260713),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(semantic_api.call_count, 1)
        self.assertEqual(len(captured), 1)
        prompt = captured[0]
        prompt_chars = len(
            json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        )
        # The n=1 pre-fix prompt was 140,303 chars because it sent two full
        # 46K ledgers plus evidence/provenance text. Keep a fail-closed ceiling.
        self.assertLess(prompt_chars, 40_000)
        self.assertEqual(
            set(prompt["seed"]["parent_event_retrieval"]),
            set(subject.V4_PARENT_PROMPT_LEDGER_FIELDS),
        )
        self.assertNotIn(
            "retrieval_ledger", prompt["seed"]["semantic_parent_graph"]
        )
        self.assertEqual(
            json.dumps(prompt, ensure_ascii=False).count(
                '"parent_event_retrieval"'
            ),
            1,
        )
        role_schemas = prompt["output_schema"]["events"]
        self.assertEqual(len(role_schemas), 8)
        self.assertEqual(
            [event["event_id"] for event in role_schemas],
            [f"E{index}" for index in range(1, 9)],
        )
        self.assertEqual(
            [event["time_index"] for event in role_schemas],
            list(range(1, 9)),
        )
        self.assertEqual(
            [event["depends_on"] for event in role_schemas],
            [[], ["E1"], ["E2"], ["E3"], [], [], [], []],
        )
        self.assertNotIn("previous_value", role_schemas[0])
        self.assertEqual(
            [role_schemas[index]["previous_value"] for index in range(1, 4)],
            [
                "copy E1.value byte-for-byte",
                "copy E2.value byte-for-byte",
                "copy E3.value byte-for-byte",
            ],
        )
        for event in prompt["seed"]["semantic_parent_graph"]["events"]:
            self.assertFalse(
                {"provenance", "supporting_span", "supporting_spans", "memory_text"}
                & set(event)
            )
        self.assertEqual(
            caught.exception.api_graph_meta["semantic_retry_suppressed"],
            "transport_timeout_exhausted",
        )
        self.assertIn(
            "selection_reasons",
            caught.exception.api_graph_meta["parent_event_retrieval"],
        )
        self.assertEqual(canonical_sha256(row), source_before)

    def test_api_graph_meta_and_plan_keep_retrieval_and_continuity_audits(self) -> None:
        row = self.rows[0]
        source_before = canonical_sha256(row)
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return relation_payload(prompt), {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            plan, meta, raw = subject.call_api_graph_expansion(
                row,
                {},
                stage3_policy("relation_lookup"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=0,
                ),
                random.Random(17),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(len(captured), 1)
        prompt_seed = captured[0]["seed"]
        self.assertNotIn("retrieval_ledger", prompt_seed["semantic_parent_graph"])
        self.assertEqual(
            set(prompt_seed["parent_event_retrieval"]),
            set(subject.V4_PARENT_PROMPT_LEDGER_FIELDS),
        )
        prompt_json = json.dumps(captured[0], ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(prompt_json.count('"parent_event_retrieval":'), 1)
        for event in prompt_seed["semantic_parent_graph"]["events"]:
            self.assertFalse(
                {"provenance", "supporting_span", "supporting_spans", "memory_text"}
                & set(event)
            )
        self.assertTrue(meta["graph_parent_prompt_audit"]["pass"])
        self.assertTrue(meta["parent_child_continuity_audit"]["pass"])
        self.assertEqual(
            meta["parent_event_retrieval"]["retrieval_operation"],
            "relation_lookup",
        )
        self.assertIn("eligible_parent_event_ids", meta["parent_event_retrieval"])
        self.assertIn("ineligible_events", meta["parent_event_retrieval"])
        self.assertIn("selection_reasons", meta["parent_event_retrieval"])
        self.assertEqual(
            plan["metadata"]["parent_event_retrieval"],
            meta["parent_event_retrieval"],
        )
        self.assertEqual(
            plan["metadata"]["parent_child_continuity_audit"],
            meta["parent_child_continuity_audit"],
        )
        continuity = plan["metadata"]["parent_continuity"]
        self.assertEqual(
            continuity["depends_on_parent_event_ids"],
            meta["parent_child_continuity_audit"]["depends_on_parent_event_ids"],
        )
        self.assertEqual(
            continuity["parent_topic_anchor_event_ids"],
            meta["parent_child_continuity_audit"][
                "parent_topic_anchor_event_ids"
            ],
        )
        self.assertEqual(
            continuity["continuity_score"],
            meta["parent_child_continuity_audit"]["continuity_score"],
        )
        self.assertEqual(
            continuity["continuity_reasons"],
            meta["parent_child_continuity_audit"]["continuity_reasons"],
        )

        parent_max = subject.stage3_parent_continuity.parent_max_global_time_index(
            row["semantic_parent_graph"]["events"]
        )
        for event in plan["events"]:
            self.assertEqual(event["local_time_index"], event["time_index"])
            self.assertGreater(event["global_time_index"], parent_max)
        self.assertTrue(
            plan["metadata"]["child_global_time_strictly_after_parent"]
        )
        self.assertEqual(canonical_sha256(row), source_before)

        alignment = {
            event["event_instance_id"]: {"turn_ids": [f"D999:{index}"]}
            for index, event in enumerate(plan["events"], start=1)
        }
        hashes = {
            "graph_snapshot_hash": plan["graph_snapshot_hash"],
            "dialogue_hash": "dialogue-hash",
            "alignment_hash": "alignment-hash",
        }
        graph_update = subject.compile_graph_update_from_current_event_plan(
            plan, alignment, hashes
        )
        self.assertEqual(graph_update["parent_continuity"], continuity)
        self.assertEqual(
            graph_update["parent_event_retrieval"],
            plan["metadata"]["parent_event_retrieval"],
        )
        for plan_event, compiled_event in zip(plan["events"], graph_update["events"]):
            self.assertEqual(
                compiled_event["local_time_index"], plan_event["local_time_index"]
            )
            self.assertEqual(
                compiled_event["global_time_index"], plan_event["global_time_index"]
            )

        # The immutable artifact binds the complete parent, not only graph_id.
        anchors = {
            event["event_instance_id"]: f"<EV_{index:02d}>"
            for index, event in enumerate(plan["events"], start=1)
        }
        plan["current_event_instance_to_anchor"] = copy.deepcopy(anchors)
        plan["current_anchor_to_event_instance_id"] = {
            marker: event_id for event_id, marker in anchors.items()
        }
        target = stage3_policy("relation_lookup")["stage64_target"]
        artifact = subject.build_stage3_graph_artifact(
            seed_row=row,
            current_event_plan=plan,
            child_graph_delta=raw,
            stage64_target=target,
            stage64_target_identity="a" * 64,
            source_graph_meta=meta,
        )
        self.assertEqual(
            artifact["combined_lineage_graph"]["schema_version"],
            "stage3_lineage_graph_v2",
        )
        self.assertEqual(
            artifact["combined_lineage_graph"]["event_count"],
            len(row["semantic_parent_graph"]["events"]) + len(plan["events"]),
        )
        self.assertEqual(
            artifact["lineage_hashes"]["parent_graph_sha256"],
            canonical_sha256(row["semantic_parent_graph"]),
        )
        self.assertFalse(
            subject.fixed_stage3_graph_artifact_issues(
                artifact,
                seed_row=row,
                stage64_target_identity="a" * 64,
                stage64_target=target,
            )
        )
        changed_parent = copy.deepcopy(row)
        changed_parent["semantic_parent_graph"]["events"][0]["value"] = (
            "same graph id, different full parent content"
        )
        mismatch_codes = {
            issue["code"]
            for issue in subject.fixed_stage3_graph_artifact_issues(
                artifact,
                seed_row=changed_parent,
                stage64_target_identity="a" * 64,
                stage64_target=target,
            )
        }
        self.assertIn("fixed_graph_parent_full_hash_mismatch", mismatch_codes)

        with mock.patch.object(
            subject.base,
            "validate_sample",
            return_value={
                "graph_dialogue_consistent": True,
                "gold_derivable": True,
                "option_valid": True,
                "anti_shortcut_pass": True,
                "warnings": [],
            },
        ):
            child_turns = [
                {
                    "turn_id": f"D999:{index}",
                    "speaker": "Maya",
                    "text": (
                        f"The {event['slot']} for {event['entity']} is "
                        f"{event['value']}."
                    ),
                }
                for index, event in enumerate(plan["events"], start=1)
            ]
            clean = subject.build_clean_sample(
                sample_index=0,
                seed_row=row,
                current_event_plan=plan,
                final_dialogue={
                    "turns": copy.deepcopy(row["dialogue"]["turns"]) + child_turns
                },
                graph_update=graph_update,
                qa={
                    "question_type": "single_choice",
                    "question": "Which item is current?",
                    "options": ["red notebook", "blue notebook"],
                    "gold": "A",
                    "graph_validation_pass": True,
                },
                hashes=hashes,
                length_metadata={},
            )
        self.assertEqual(len(clean["graph_update"]["events"]), len(plan["events"]))
        inherited = clean["semantic_inherited_graph"]
        self.assertEqual(inherited, clean["combined_lineage_graph"])
        self.assertEqual(
            inherited["parent_event_count"],
            len(row["semantic_parent_graph"]["events"]),
        )
        self.assertEqual(inherited["event_count"], len(inherited["events"]))
        self.assertEqual(inherited["parent_continuity"], continuity)
        self.assertTrue(subject._is_v4_memory_parent(clean))
        # Round-trip through the explicit inherited reader: full v4 memory is
        # still counted while child-only graph_update remains untouched.
        roundtrip_full_ledger: dict = {}
        roundtrip = subject.compact_seed_for_prompt(
            clean,
            semantic_graph_v2=True,
            parent_operation="relation_lookup",
            full_parent_event_retrieval_out=roundtrip_full_ledger,
        )
        self.assertEqual(
            roundtrip_full_ledger["full_parent_event_count"],
            len(row["semantic_parent_graph"]["events"]) + len(plan["events"]),
        )
        compiled_child_ids = {
            event["event_id"] for event in clean["graph_update"]["events"]
        }
        self.assertTrue(
            compiled_child_ids.issubset(
                roundtrip_full_ledger["eligible_parent_event_ids"]
            )
        )
        self.assertTrue(
            subject.audit_graph_parent_prompt_verbatim_exposure(
                {"seed": roundtrip}, clean
            )["pass"]
        )


if __name__ == "__main__":
    unittest.main()
