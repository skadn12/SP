from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_auto_graph_expansion_round as runner
from scripts.current_method import native_api_graph_dialogue_expansion as native
from scripts.current_method import stage3_operation_controller as controller


def audited_sample(operation: str, answer_type: str, status: str, counts: list[int]) -> dict:
    signature = {
        "reasoning_subtype": f"{operation}_test_subtype",
        "reasoning_template": f"{operation}_test_template",
        "hop_count": 2,
        "hard_negative_strategy": f"{operation}_test_hard_negative",
    }
    return {
        "sample_id": f"{operation}-{answer_type}-{status}-{counts}",
        "qa_operation": operation,
        "answer_type": answer_type,
        **signature,
        "query_plan": dict(signature),
        "qa": {**signature, "query_plan": dict(signature)},
        "stage64_target": {"qa_operation": operation, **signature},
        "stage64_target_identity": f"formal-v2-{operation}-{answer_type}",
        "stage3_generation_protocol": controller.generation_protocol(),
        "stage64_permutation_eval": {
            "status": status,
            "k_eval_per_permutation": 5,
            "required_frontier_permutation_share": 1.0,
            "permutations": [
                {"shift": index, "exact_correct_count": count}
                for index, count in enumerate(counts)
            ],
        },
    }


class Stage3OperationControllerTest(unittest.TestCase):
    def test_native_graph_gate_and_controller_protocol_versions_match(self) -> None:
        self.assertEqual(
            native.GRAPH_CACHE_SCHEMA_VERSION,
            controller.REQUIRED_GRAPH_GATE_VERSION,
        )

    def test_explicit_smoke_plan_allocates_one_operation_without_state(self) -> None:
        plan = runner.explicit_stage3_smoke_operation_plan("bridge")
        self.assertTrue(plan["smoke_only"])
        self.assertEqual(plan["operation_shares"]["bridge"], 1.0)
        self.assertEqual(
            sum(plan["operation_shares"].values()),
            1.0,
        )
        target = native._build_stage64_canonical_targets(
            1,
            plan["operation_shares"],
            None,
        )[0]["stage64_target"]
        self.assertEqual(target["qa_operation"], "bridge")
        self.assertEqual(target["preferred_answer_type"], "Person")
        with self.assertRaises(ValueError):
            runner.explicit_stage3_smoke_operation_plan("unknown")

    def test_native_and_controller_answer_type_domains_match(self) -> None:
        self.assertEqual(
            native.STAGE64_OPERATION_ANSWER_TYPE_CANDIDATES,
            controller.ANSWER_TYPE_CANDIDATES,
        )

    def test_native_allocator_materializes_exact_controller_quotas(self) -> None:
        operation_shares = {"bridge": 0.6, "comparison": 0.4}
        answer_shares = {"bridge": {"Event": 0.5, "Person": 0.5}}
        targets = [
            native._stage64_target_for_slot(index, 10, operation_shares, answer_shares)
            for index in range(10)
        ]
        self.assertEqual(
            {operation: sum(item["qa_operation"] == operation for item in targets) for operation in operation_shares},
            {"bridge": 6, "comparison": 4},
        )
        bridge_answers = [
            item["preferred_answer_type"] for item in targets if item["qa_operation"] == "bridge"
        ]
        self.assertEqual(bridge_answers.count("Event"), 3)
        self.assertEqual(bridge_answers.count("Person"), 3)
        self.assertTrue(all(item["target_basis"].startswith("stage3_operation_controller") for item in targets))

    def test_uses_cyclic_aggregate_and_preserves_operation_floor(self) -> None:
        samples = [
            audited_sample("bridge", "Event", "robust_frontier", [1, 2, 3, 4]),
            audited_sample("bridge", "Person", "robust_frontier", [2, 2, 2, 2]),
            audited_sample("comparison", "Event", "robust_easy5", [5, 5, 5, 5]),
        ]
        summary = controller.summarize_batch(samples)
        plan = controller.build_plan(summary, minimum_operation_share=0.08)
        self.assertEqual(summary["evaluated_permutation_count"], 12)
        self.assertGreater(plan["operation_shares"]["bridge"], plan["operation_shares"]["comparison"])
        self.assertTrue(all(value >= 0.08 - 1e-8 for value in plan["operation_shares"].values()))
        self.assertAlmostEqual(sum(plan["operation_shares"].values()), 1.0)
        self.assertFalse(plan["semantic_rejects_used_as_solver_signal"])
        self.assertFalse(plan["extra_solver_evaluation_run"])

    def test_reasoning_signature_metrics_use_target_and_rollout_weighting(self) -> None:
        frontier = audited_sample(
            "bridge", "Event", "robust_frontier", [1, 2, 3, 4]
        )
        easy = audited_sample("bridge", "Event", "robust_easy5", [5, 5, 5, 5])
        summary = controller.summarize_batch([frontier, easy])
        self.assertEqual(len(summary["reasoning_signatures"]), 1)
        metric = next(iter(summary["reasoning_signatures"].values()))
        self.assertEqual(metric["total"], 2)
        self.assertEqual(metric["status_counts"]["robust_frontier"], 1)
        self.assertEqual(metric["status_counts"]["robust_easy5"], 1)
        self.assertEqual(metric["exact_correct_rollout_count"], 30)
        self.assertEqual(metric["evaluated_rollout_count"], 40)
        self.assertEqual(metric["weighted_rollout_exact_accuracy"], 0.75)
        self.assertEqual(metric["frontier_yield"], 0.5)
        self.assertEqual(metric["signature"]["operation"], "bridge")

    def test_reasoning_signature_is_fail_closed_and_target_authoritative(self) -> None:
        missing_target = audited_sample(
            "bridge", "Event", "robust_frontier", [1, 2, 3, 4]
        )
        missing_target.pop("stage64_target")
        with self.assertRaisesRegex(ValueError, "authoritative sample.stage64_target"):
            controller.summarize_batch([missing_target])

        incomplete_target = audited_sample(
            "bridge", "Event", "robust_frontier", [1, 2, 3, 4]
        )
        incomplete_target["stage64_target"].pop("reasoning_template")
        with self.assertRaisesRegex(ValueError, "missing or invalid reasoning_template"):
            controller.summarize_batch([incomplete_target])

        drifted = audited_sample(
            "bridge", "Event", "robust_frontier", [1, 2, 3, 4]
        )
        drifted["qa"]["query_plan"]["hop_count"] = 3
        with self.assertRaisesRegex(ValueError, "qa.query_plan.hop_count"):
            controller.summarize_batch([drifted])

        legacy = {
            "qa_operation": "bridge",
            "qa": {"qa_operation": "bridge"},
            "stage64_target": {"qa_operation": "bridge"},
        }
        self.assertIsNone(controller.sample_reasoning_signature(legacy))

    def test_predefined_answer_candidates_preserve_exploration(self) -> None:
        summary = controller.summarize_batch(
            [
                audited_sample("bridge", "Event", "robust_easy5", [5, 5, 5, 5]),
                audited_sample("bridge", "YesNo", "robust_easy5", [5, 5, 5, 5]),
            ]
        )
        plan = controller.build_plan(summary, minimum_operation_share=0.08)
        answer_plans = plan["answer_type_shares_by_operation"]
        self.assertEqual(answer_plans["count"], {"Number": 1.0})
        for operation in controller.OPERATIONS:
            self.assertIn(operation, answer_plans)
            self.assertAlmostEqual(sum(answer_plans[operation].values()), 1.0)
            if operation not in controller.REQUIRED_ANSWER_TYPES:
                self.assertGreater(len(answer_plans[operation]), 1)
                self.assertTrue(
                    all(value >= plan["minimum_answer_type_share"] - 1e-8 for value in answer_plans[operation].values())
                )
        self.assertLess(answer_plans["bridge"]["Event"], 1.0)
        self.assertNotIn("YesNo", answer_plans["bridge"])

    def test_difficulty_hardens_easy_low_yield_and_backs_off_hard0(self) -> None:
        easy_batch = controller.summarize_batch(
            [
                audited_sample("bridge", "Event", "robust_easy5", [5, 5, 5, 5]),
                audited_sample("comparison", "Number", "robust_easy5", [5, 5, 5, 5]),
                audited_sample("visibility", "YesNo", "robust_easy5", [5, 5, 5, 5]),
                audited_sample("relation_lookup", "Person", "position_sensitive", [5, 4, 5, 4]),
            ]
        )
        easy_plan = controller.build_plan(easy_batch, minimum_operation_share=0.08)
        self.assertEqual(easy_plan["item_difficulty_target"], "harder_boundary")
        self.assertEqual(
            easy_plan["difficulty_control"]["reason"],
            "mostly_robust_easy5_with_low_frontier_yield",
        )

        hard_batch = controller.summarize_batch(
            [
                audited_sample("bridge", "Event", "robust_hard0", [0, 0, 0, 0]),
                audited_sample("comparison", "Number", "robust_hard0", [0, 0, 0, 0]),
                audited_sample("visibility", "YesNo", "robust_frontier", [1, 2, 3, 4]),
                audited_sample("relation_lookup", "Person", "position_sensitive", [5, 4, 5, 4]),
            ]
        )
        hard_plan = controller.build_plan(hard_batch, minimum_operation_share=0.08)
        self.assertEqual(hard_plan["item_difficulty_target"], "easy")
        self.assertEqual(hard_plan["difficulty_control"]["reason"], "high_robust_hard0_backoff")
        self.assertFalse(hard_plan["difficulty_control"]["semantic_rejects_used_as_solver_signal"])

    def test_runner_uses_controller_difficulty_unless_cli_overrides(self) -> None:
        plan = {"operation_shares": {"bridge": 1.0}, "item_difficulty_target": "harder_boundary"}
        automatic = runner.inject_stage3_operation_plan(
            {"policy_id": "test"},
            plan,
            explicit_item_difficulty_target=None,
        )
        self.assertEqual(automatic["item_difficulty_target"], "harder_boundary")
        self.assertEqual(automatic["item_difficulty_target_source"], "stage3_operation_controller")

        explicit = runner.inject_stage3_operation_plan(
            {"policy_id": "test"},
            plan,
            explicit_item_difficulty_target="boundary",
        )
        self.assertEqual(explicit["item_difficulty_target"], "boundary")
        self.assertEqual(explicit["item_difficulty_target_source"], "explicit_cli")

    def test_runner_attaches_current_protocol_only_from_matching_v6_raw_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.jsonl"
            raw_path.write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "sample_id": "sample-1",
                        "api_graph_meta": {
                            "semantic_graph_v2_quality": {
                                "pass": True,
                                "gate_version": controller.REQUIRED_GRAPH_GATE_VERSION,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = [
                {
                    "sample_id": "sample-1",
                    "native_generator_version": controller.REQUIRED_NATIVE_GENERATOR_VERSION,
                }
            ]
            audit = runner.attach_stage3_generation_protocol_from_raw_audit(samples, raw_path)
            self.assertEqual(audit["sample_count"], 1)
            self.assertEqual(
                samples[0]["stage3_generation_protocol"],
                controller.generation_protocol(),
            )

    def test_runner_rejects_stale_or_mismatched_raw_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.jsonl"
            raw_path.write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "sample_id": "sample-1",
                        "api_graph_meta": {
                            "semantic_graph_v2_quality": {
                                "pass": True,
                                "gate_version": "semantic_graph_v16",
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = [
                {
                    "sample_id": "sample-1",
                    "native_generator_version": controller.REQUIRED_NATIVE_GENERATOR_VERSION,
                }
            ]
            with self.assertRaisesRegex(RuntimeError, "refuses stale"):
                runner.attach_stage3_generation_protocol_from_raw_audit(samples, raw_path)

            samples[0]["sample_id"] = "other-sample"
            with self.assertRaisesRegex(RuntimeError, "sample mismatch"):
                runner.attach_stage3_generation_protocol_from_raw_audit(samples, raw_path)

    def test_runner_skip_generation_does_not_backfill_missing_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.jsonl"
            raw_path.write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "sample_id": "sample-1",
                        "api_graph_meta": {
                            "semantic_graph_v2_quality": {
                                "pass": True,
                                "gate_version": controller.REQUIRED_GRAPH_GATE_VERSION,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sample = {
                "sample_id": "sample-1",
                "native_generator_version": controller.REQUIRED_NATIVE_GENERATOR_VERSION,
            }
            with self.assertRaisesRegex(ValueError, "Missing stage3_generation_protocol"):
                runner.attach_stage3_generation_protocol_from_raw_audit(
                    [sample],
                    raw_path,
                    require_existing_fingerprint=True,
                )

            controller.attach_generation_protocol(
                sample,
                graph_gate_version=controller.REQUIRED_GRAPH_GATE_VERSION,
            )
            audit = runner.attach_stage3_generation_protocol_from_raw_audit(
                [sample],
                raw_path,
                require_existing_fingerprint=True,
            )
            self.assertEqual(audit["sample_count"], 1)

    def test_rejects_missing_or_non_k5_audit(self) -> None:
        sample = audited_sample("bridge", "Event", "robust_frontier", [1, 2, 3, 4])
        sample["stage64_permutation_eval"]["k_eval_per_permutation"] = 1
        with self.assertRaisesRegex(ValueError, "requires k=5"):
            controller.summarize_batch([sample])

        sample = audited_sample("bridge", "Event", "robust_frontier", [1, 2, 3, 4])
        sample["stage64_permutation_eval"]["required_frontier_permutation_share"] = 0.75
        with self.assertRaisesRegex(ValueError, "strict all-position"):
            controller.summarize_batch([sample])

    def test_state_is_idempotent_and_solver_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "clean_samples.jsonl"
            rows = [audited_sample("count", "Number", "robust_frontier", [1, 2, 3, 4])]
            samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            state_path = root / "state.json"
            first = controller.update_state(
                samples_path=samples_path,
                state_path=state_path,
                accepted_solver_model="/model/stage2",
            )
            second = controller.update_state(
                samples_path=samples_path,
                state_path=state_path,
                accepted_solver_model="/model/stage2",
            )
            self.assertEqual(first, second)
            self.assertEqual(second["aggregate"]["sample_count"], 1)
            self.assertEqual(second["stage3_operation_plan"]["answer_type_shares_by_operation"]["count"], {"Number": 1.0})
            with self.assertRaisesRegex(RuntimeError, "solver mismatch"):
                controller.update_state(
                    samples_path=samples_path,
                    state_path=state_path,
                    accepted_solver_model="/model/other",
                )

    def test_stale_controller_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "clean_samples.jsonl"
            row = audited_sample("bridge", "Event", "robust_easy5", [5, 5, 5, 5])
            samples_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            state_path = root / "state.json"
            current = controller.update_state(
                samples_path=samples_path,
                state_path=state_path,
                accepted_solver_model="/model/stage2",
            )
            current["schema_version"] = "stage3_operation_controller_v1"
            current.pop("generation_protocol", None)
            current["stage3_operation_plan"] = {
                "schema_version": "stage3_operation_controller_v1",
                "answer_type_shares_by_operation": {"bridge": {"Event": 1.0}},
            }
            state_path.write_text(json.dumps(current), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing or stale"):
                controller.update_state(
                    samples_path=samples_path,
                    state_path=state_path,
                    accepted_solver_model="/model/stage2",
                )

    def test_missing_or_stale_generation_protocol_fails_closed(self) -> None:
        missing = audited_sample("bridge", "Event", "robust_easy5", [5, 5, 5, 5])
        missing.pop("stage3_generation_protocol")
        with self.assertRaisesRegex(ValueError, "Missing stage3_generation_protocol"):
            controller.summarize_batch([missing])

        stale = audited_sample("bridge", "Event", "robust_easy5", [5, 5, 5, 5])
        stale["stage3_generation_protocol"] = controller.generation_protocol(
            graph_gate_version="semantic_graph_v16"
        )
        with self.assertRaisesRegex(ValueError, "Unsupported Stage3 generation protocol"):
            controller.summarize_batch([stale])

    def test_current_semantic_quality_is_rechecked_before_aggregation(self) -> None:
        sample = audited_sample("count", "Number", "robust_frontier", [1, 2, 3, 4])
        sample["qa"] = {
            "qa_operation": "count",
            "answer_type": "Number",
            "question": "How many items are included?",
            "options": [
                {"id": "A", "text": "1"},
                {"id": "B", "text": "1"},
                {"id": "C", "text": "2"},
                {"id": "D", "text": "3"},
            ],
            "gold": ["A"],
        }
        with self.assertRaisesRegex(ValueError, "failed current semantic quality"):
            controller.summarize_batch([sample])


if __name__ == "__main__":
    unittest.main()
