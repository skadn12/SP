from __future__ import annotations

import copy
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.current_method import native_api_graph_dialogue_expansion as native


def _plan(count: int = 3) -> list[dict]:
    return native._build_stage64_canonical_targets(
        count,
        {
            "temporal_state_update": 1,
            "comparison": 1,
            "count": 1,
        },
        None,
    )


def _sample(record: dict, generation_slot_id: int) -> dict:
    target = record["stage64_target"]
    operation = target["qa_operation"]
    reasoning_identity = {
        field: target[field] for field in native.STAGE64_REASONING_TARGET_FIELDS
    }
    return {
        "sample_id": f"sample-{generation_slot_id}",
        "generation_slot_id": generation_slot_id,
        "target_index": record["target_index"],
        "stage64_target": target.copy(),
        "stage64_target_identity": record["target_identity"],
        "stage64_target_plan_sha256": record["target_plan_sha256"],
        "qa_operation": operation,
        **reasoning_identity,
        "answer_type": target.get("preferred_answer_type"),
        "qa": {
            "qa_operation": operation,
            **reasoning_identity,
            "answer_type": target.get("preferred_answer_type"),
            "query_plan": reasoning_identity.copy(),
        },
    }


class Stage64TargetLedgerTest(unittest.TestCase):
    def test_canonical_targets_have_stable_unique_identity(self) -> None:
        first = _plan()
        second = _plan()
        self.assertEqual(first, second)
        self.assertEqual([row["target_index"] for row in first], [0, 1, 2])
        self.assertEqual(
            [row["stage64_target"]["qa_operation"] for row in first],
            ["temporal_state_update", "comparison", "count"],
        )
        self.assertEqual(len({row["target_identity"] for row in first}), 3)
        self.assertEqual(
            {row["target_id"] for row in first},
            {row["target_identity"] for row in first},
        )
        self.assertEqual(len({row["target_plan_sha256"] for row in first}), 1)
        for row in first:
            target = row["stage64_target"]
            self.assertTrue(
                all(field in target for field in native.STAGE64_REASONING_TARGET_FIELDS)
            )

    def test_fallback_answer_types_are_compatible_with_each_operation(self) -> None:
        plan = native._build_stage64_canonical_targets(25, None, None)
        for record in plan:
            target = record["stage64_target"]
            self.assertIn(
                target["preferred_answer_type"],
                native.STAGE64_OPERATION_ANSWER_TYPE_CANDIDATES[
                    target["qa_operation"]
                ],
            )

    def test_operation_quotas_are_exact_and_interleaved(self) -> None:
        targets = native._build_stage64_canonical_targets(
            10,
            {"bridge": 0.6, "comparison": 0.4},
            None,
        )
        operations = [row["stage64_target"]["qa_operation"] for row in targets]
        self.assertEqual(operations.count("bridge"), 6)
        self.assertEqual(operations.count("comparison"), 4)
        self.assertEqual(
            operations,
            [
                "bridge",
                "comparison",
                "bridge",
                "comparison",
                "bridge",
                "comparison",
                "bridge",
                "comparison",
                "bridge",
                "bridge",
            ],
        )

    def test_seven_row_smoke_covers_every_enabled_operation_once(self) -> None:
        targets = native._build_stage64_canonical_targets(7, None, None)
        operations = [row["stage64_target"]["qa_operation"] for row in targets]
        self.assertEqual(operations, list(native.STAGE64_QA_OPERATIONS))

    def test_reasoning_signatures_cycle_within_each_operation(self) -> None:
        targets = native._build_stage64_canonical_targets(
            6,
            {"temporal_state_update": 0.5, "bridge": 0.5},
            None,
        )
        by_operation = {
            operation: [
                row["stage64_target"]
                for row in targets
                if row["stage64_target"]["qa_operation"] == operation
            ]
            for operation in ("temporal_state_update", "bridge")
        }
        for operation, rows in by_operation.items():
            self.assertEqual(
                [row["reasoning_subtype"] for row in rows],
                [
                    spec["reasoning_subtype"]
                    for spec in native.STAGE64_REASONING_TARGET_SPECS[operation]
                ],
            )
        self.assertTrue(
            all(row["forbid_max_time_shortcut"] for row in by_operation["temporal_state_update"])
        )
        self.assertTrue(
            all("forbid_max_time_shortcut" not in row for row in by_operation["bridge"])
        )

    def test_reasoning_target_binding_rejects_drift_and_temporal_shortcut(self) -> None:
        target = native._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        identity = {
            field: target[field] for field in native.STAGE64_REASONING_TARGET_FIELDS
        }
        proposal = {
            **identity,
            "question": (
                "After disregarding the canceled proposal, what is the current "
                "destination of Project Atlas?"
            ),
        }
        plan = {
            "events": [
                {"event_instance_id": "E1", "entity": "Project Atlas", "slot": "destination", "value": "Paris", "status": "outdated", "time_index": 1, "depends_on": []},
                {"event_instance_id": "E2", "entity": "Project Atlas", "slot": "destination", "value": "Lyon", "status": "active", "time_index": 2, "previous_value": "Paris", "depends_on": ["E1"]},
                {"event_instance_id": "E3", "entity": "Project Atlas", "slot": "destination", "value": "Berlin", "status": "belief", "time_index": 3, "previous_value": "Lyon", "depends_on": ["E2"]},
                {"event_instance_id": "E4", "entity": "Project Atlas", "slot": "destination", "value": "Rome", "status": "canceled", "time_index": 4, "previous_value": "Berlin", "depends_on": ["E3"]},
                {"event_instance_id": "E5", "entity": "Maya", "slot": "hotel", "value": "Orchid", "status": "active", "time_index": 5},
                {"event_instance_id": "E6", "entity": "Noah", "slot": "train", "value": "T18", "status": "active", "time_index": 6},
            ]
        }
        query_plan = {
            **identity,
            "dependency_closure_refs": ["E1", "E2", "E3", "E4"],
            "answer_projection": "current_value",
        }
        self.assertEqual(
            native._validate_stage64_reasoning_target_binding(
                stage64_target=target,
                qa_proposal=proposal,
                query_plan=query_plan,
                current_event_plan=plan,
                resolved_reasoning_event_ids=["E1", "E2", "E3", "E4"],
                resolved_distractor_event_ids=["E1", "E3", "E4"],
            )["reasoning_subtype"],
            target["reasoning_subtype"],
        )

        drifted = copy.deepcopy(query_plan)
        drifted["reasoning_template"] = "direct_lookup"
        with self.assertRaisesRegex(native.NativeGraphDialogueReject, "reasoning target drift"):
            native._validate_stage64_reasoning_target_binding(
                stage64_target=target,
                qa_proposal=proposal,
                query_plan=drifted,
                current_event_plan=plan,
                resolved_reasoning_event_ids=["E1", "E2", "E3", "E4"],
                resolved_distractor_event_ids=["E1", "E3", "E4"],
            )

        shortcut = dict(proposal, question="What is the latest destination of Project Atlas?")
        with self.assertRaisesRegex(native.NativeGraphDialogueReject, "max-time-only"):
            native._validate_stage64_reasoning_target_binding(
                stage64_target=target,
                qa_proposal=shortcut,
                query_plan=copy.deepcopy(query_plan),
                current_event_plan=plan,
                resolved_reasoning_event_ids=["E1", "E2", "E3", "E4"],
                resolved_distractor_event_ids=["E1", "E3", "E4"],
            )

    def test_skip_requeues_same_target_for_new_physical_slot(self) -> None:
        plan = _plan()
        ledger = native._Stage64TargetLedger(plan)
        self.assertEqual(ledger.claim(0)["target_index"], 0)
        skipped_target = ledger.claim(1)
        self.assertEqual(skipped_target["target_index"], 1)
        self.assertEqual(ledger.claim(2)["target_index"], 2)
        ledger.skip(1)
        replacement = ledger.claim(3)
        self.assertEqual(replacement["target_index"], 1)
        self.assertEqual(replacement["stage64_target"]["qa_operation"], "comparison")
        self.assertEqual(ledger.reallocated_slot_count, 1)

    def test_out_of_order_accept_preserves_operation_quota(self) -> None:
        plan = _plan()
        ledger = native._Stage64TargetLedger(plan)
        assignments = {slot: ledger.claim(slot) for slot in range(3)}
        ledger.accept(2, _sample(assignments[2], 2))
        ledger.accept(0, _sample(assignments[0], 0))
        ledger.skip(1)
        replacement = ledger.claim(3)
        ledger.accept(3, _sample(replacement, 3))
        summary = ledger.quota_summary(
            [_sample(assignments[2], 2), _sample(assignments[0], 0), _sample(replacement, 3)]
        )
        self.assertEqual(summary["stage64_accepted_operation_counts"], {
            "comparison": 1,
            "count": 1,
            "temporal_state_update": 1,
        })
        self.assertEqual(summary["stage64_operation_shortfall_counts"], {})
        self.assertEqual(summary["stage64_operation_excess_counts"], {})
        self.assertEqual(summary["stage64_reasoning_target_match_count"], 3)
        self.assertEqual(summary["stage64_reasoning_target_mismatch_count"], 0)
        self.assertEqual(
            sum(summary["stage64_accepted_reasoning_subtype_counts"].values()),
            3,
        )
        self.assertEqual(
            summary["stage64_accepted_hop_count_counts"],
            summary["stage64_target_hop_count_counts"],
        )
        self.assertTrue(summary["stage64_operation_quota_complete"])

    def test_resume_slot_zero_and_two_only_claims_missing_comparison(self) -> None:
        plan = _plan()
        resumed = [_sample(plan[0], 0), _sample(plan[2], 2)]
        ledger = native._Stage64TargetLedger(plan, resumed)
        self.assertEqual(ledger.next_generation_slot, 3)
        replacement = ledger.claim(3)
        self.assertEqual(replacement["target_index"], 1)
        self.assertEqual(replacement["stage64_target"]["qa_operation"], "comparison")

    def test_resume_without_provenance_fails_closed(self) -> None:
        plan = _plan()
        legacy_row = {"sample_id": "old-0000", "qa_operation": "temporal_state_update"}
        with self.assertRaisesRegex(RuntimeError, "generation_slot_id|target_index"):
            native._Stage64TargetLedger(plan, [legacy_row])

    def test_formal_generation_rejects_old_partial_before_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            partial = output / "samples.partial.jsonl"
            native.write_jsonl(
                partial,
                [{"sample_id": "old-0000", "qa_operation": "temporal_state_update"}],
            )
            args = SimpleNamespace(
                native_curriculum_mode=native.EASY_V2_MODE,
                qa_operation_answer_type_diversification=True,
            )
            with mock.patch.object(
                native,
                "call_raw_json_api",
                side_effect=AssertionError("stale resume must fail before API"),
            ):
                with self.assertRaisesRegex(RuntimeError, "generation_slot_id|target_index"):
                    native.generate_graph_conditioned_api_dialogue_samples(
                        seed_rows=[{"sample_id": "seed-1"}],
                        policy={
                            "stage3_operation_plan": {
                                "operation_shares": {
                                    "temporal_state_update": 1,
                                    "comparison": 1,
                                    "count": 1,
                                }
                            }
                        },
                        n_samples=3,
                        output_dir=output,
                        api_args=args,
                        partial_samples=partial,
                    )

    def test_resume_rejects_stale_plan_hash_and_duplicate_target(self) -> None:
        plan = _plan()
        stale = _sample(plan[0], 0)
        stale["stage64_target_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "target plan"):
            native._Stage64TargetLedger(plan, [stale])
        duplicate = [_sample(plan[0], 0), _sample(plan[0], 1)]
        with self.assertRaisesRegex(RuntimeError, "duplicate target_index"):
            native._Stage64TargetLedger(plan, duplicate)

    def test_shortfall_and_excess_are_explicit(self) -> None:
        plan = _plan()
        ledger = native._Stage64TargetLedger(plan)
        first = ledger.claim(0)
        third = ledger.claim(1)
        ledger.accept(0, _sample(first, 0))
        ledger.accept(1, _sample(third, 1))
        summary = ledger.quota_summary([_sample(first, 0), _sample(third, 1)])
        self.assertEqual(summary["stage64_operation_shortfall_counts"], {"count": 1})
        self.assertEqual(summary["stage64_operation_excess_counts"], {})
        self.assertFalse(summary["stage64_operation_quota_complete"])

    def test_formal_mode_disables_only_legacy_caps(self) -> None:
        self.assertFalse(native._legacy_easy_v2_caps_enabled(native.EASY_V2_MODE, True))
        self.assertTrue(native._legacy_easy_v2_caps_enabled(native.EASY_V2_MODE, False))
        self.assertFalse(native._legacy_easy_v2_caps_enabled("default", False))


if __name__ == "__main__":
    unittest.main()
