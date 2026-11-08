import copy
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import stage3_solver_feedback_mining as subject


def prediction(label: str, gold: str, index: int) -> dict:
    return {
        "sample_index": index,
        "prediction": label,
        "parsed": label,
        "exact_correct": label == gold,
        "partial_credit": 1.0 if label == gold else 0.0,
    }


def audited_sample(sample_id: str, status: str, counts: list[int]) -> dict:
    events = []
    options = []
    for index, label in enumerate("ABCD", start=1):
        event_id = f"{sample_id}:E{index}"
        events.append(
            {
                "event_id": event_id,
                "event_instance_id": event_id,
                "node_id": f"node-{event_id}",
                "entity": f"Person{index}",
                "slot": "job",
                "value": f"Role{index}",
                "status": "active",
                "time_index": index,
                "turn_ids": [f"D1:{index}"],
                "evidence_turn_ids": [f"D1:{index}"],
            }
        )
        options.append(
            {
                "id": label,
                "option_id": label,
                "text": f"Person{index}",
                "answer_type": "Person",
                "source_event_instance_id": event_id,
                "anchor_ref": event_id,
                "anchor_kind": "event",
                "truth_value": label == "A",
                "is_gold": label == "A",
                "turn_ids": [f"D1:{index}"],
            }
        )
    qa = {
        "question_type": "single_choice",
        "question": "Who has Role1?",
        "question_text": "Who has Role1?",
        "qa_operation": "relation_lookup",
        "qa_subtype": "attribute_lookup",
        "answer_type": "Person",
        "answer_format": "single_choice_mcq",
        "options": options,
        "gold": ["A"],
        "gold_answer_text": "Person1",
        "normalized_gold_answer": "person1",
        "query_plan": {
            "answer_source": "graph_anchor",
            "qa_operation": "relation_lookup",
            "qa_subtype": "attribute_lookup",
            "answer_type": "Person",
            "resolved_gold_event_instance_ids": [f"{sample_id}:E1"],
        },
        "anchor_refs": {
            "support": [f"{sample_id}:E1"],
            "gold": [f"{sample_id}:E1"],
            "distractors": [f"{sample_id}:E2", f"{sample_id}:E3", f"{sample_id}:E4"],
        },
        "evidence_turn_ids": ["D1:1", "D1:2", "D1:3", "D1:4"],
        "difficulty_subtype": "hard_single_choice",
        "hard_single_choice_template": "api_stage64_graph_anchor",
        "distractor_error_types": [],
        "graph_validation_pass": True,
    }
    permutations = []
    labels = list("ABCD")
    for shift, (gold, count) in enumerate(zip(labels, counts)):
        predictions = [
            prediction(gold if index < count else labels[(shift + 1) % 4], gold, index)
            for index in range(5)
        ]
        permutations.append(
            {
                "shift": shift,
                "gold": gold,
                "exact_correct_count": count,
                "bucket": f"{count}/5",
                "soft_accuracy": count / 5,
                "partial_credit": count / 5,
                "predictions": predictions,
            }
        )
    graph_hash = f"graph-{sample_id}"
    dialogue_hash = f"dialogue-{sample_id}"
    alignment_hash = f"alignment-{sample_id}"
    dialogue = {
        "dialogue_hash": dialogue_hash,
        "turns": [
            {"turn_id": f"D1:{index}", "speaker": "A", "text": f"Person{index} has Role{index}."}
            for index in range(1, 5)
        ],
    }
    row = {
        "sample_id": sample_id,
        "lineage_group": f"lineage-{sample_id}",
        "graph_update": {
            "graph_snapshot_hash": graph_hash,
            "events": events,
            "depends_on_graph_snapshot_hash": graph_hash,
            "depends_on_dialogue_hash": dialogue_hash,
            "depends_on_alignment_hash": alignment_hash,
        },
        "dialogue": dialogue,
        "expanded_dialogue_text": "\n".join(turn["text"] for turn in dialogue["turns"]),
        "qa": qa,
        "qa_operation": "relation_lookup",
        "qa_subtype": "attribute_lookup",
        "answer_type": "Person",
        "answer_format": "single_choice_mcq",
        "question": qa["question"],
        "question_text": qa["question"],
        "question_type": "single_choice",
        "options": copy.deepcopy(options),
        "gold_answer": ["A"],
        "gold_answers": ["A"],
        "gold_options": ["A"],
        "gold_option": "A",
        "gold_answer_text": "Person1",
        "normalized_gold_answer": "person1",
        "query_plan": copy.deepcopy(qa["query_plan"]),
        "anchor_refs": copy.deepcopy(qa["anchor_refs"]),
        "attack_skill": "relation_lookup",
        "attack_plan": {"attack_skill": "relation_lookup"},
        "metadata": {
            "graph_snapshot_hash": graph_hash,
            "dialogue_hash": dialogue_hash,
            "alignment_hash": alignment_hash,
            "qa_operation": "relation_lookup",
            "answer_type": "Person",
            "option_shuffle_applied": True,
        },
        "solver_eval": {"k": 5},
        "frontier_label": {"category": "easy"},
        "option_shuffle": {"seed": 7},
        "stage64_permutation_eval": {
            "status": status,
            "k_eval_per_permutation": 5,
            "permutation_count": 4,
            "frontier_permutation_share": sum(0 < value < 5 for value in counts) / 4,
            "required_frontier_permutation_share": 1.0,
            "permutations": permutations,
        },
    }
    row["stage3_generation_protocol"] = (
        subject.stage3_operations.generation_protocol()
    )
    return row


def reasoning_target_sample(sample_id: str = "reasoning") -> dict:
    row = audited_sample(sample_id, "robust_easy5", [5, 5, 5, 5])
    target = {
        "qa_operation": "temporal_state_update",
        "preferred_answer_type": "Person",
        **copy.deepcopy(
            subject.native.STAGE64_REASONING_TARGET_SPECS["temporal_state_update"][0]
        ),
        "target_basis": "unit_test_reasoning_ledger",
    }
    target_index = 2
    target_plan_sha256 = "a" * 64
    target_identity = subject.native._stage64_target_identity(  # noqa: SLF001
        target_index,
        target,
        target_plan_sha256,
    )
    qa = row["qa"]
    qa["qa_operation"] = target["qa_operation"]
    qa["query_plan"]["qa_operation"] = target["qa_operation"]
    for field in subject.STAGE64_REASONING_TARGET_FIELDS:
        qa[field] = copy.deepcopy(target[field])
        qa["query_plan"][field] = copy.deepcopy(target[field])
        row[field] = copy.deepcopy(target[field])
    qa["forbid_max_time_shortcut"] = True
    qa["query_plan"]["forbid_max_time_shortcut"] = True
    row["forbid_max_time_shortcut"] = True
    row.update(
        {
            "qa_operation": target["qa_operation"],
            "stage64_target": target,
            "generation_slot_id": 7,
            "target_index": target_index,
            "stage64_target_identity": target_identity,
            "stage64_target_id": target_identity,
            "stage64_target_plan_sha256": target_plan_sha256,
        }
    )
    row["metadata"].update(
        {
            "qa_operation": target["qa_operation"],
            "stage64_target_index": target_index,
            "stage64_target_identity": target_identity,
            "stage64_target_id": target_identity,
            "stage64_target_plan_sha256": target_plan_sha256,
        }
    )
    return row


def marked_bundle(row: dict) -> dict:
    events = copy.deepcopy(row["graph_update"]["events"])
    plan = {
        "schema_version": "semantic_graph_v2",
        "graph_snapshot_hash": row["metadata"]["graph_snapshot_hash"],
        "events": events,
    }
    alignment = {
        event["event_instance_id"]: {"turn_ids": event["turn_ids"]}
        for event in events
    }
    hashes = {
        "graph_snapshot_hash": row["metadata"]["graph_snapshot_hash"],
        "dialogue_hash": row["metadata"]["dialogue_hash"],
        "alignment_hash": row["metadata"]["alignment_hash"],
    }
    clean_sample = copy.deepcopy(row)
    clean_sample.pop("solver_eval", None)
    clean_sample.pop("frontier_label", None)
    clean_sample.pop("stage64_permutation_eval", None)
    clean_sample.pop("option_shuffle", None)
    return {
        "clean_sample": clean_sample,
        "current_event_plan": plan,
        "current_alignment": alignment,
        "clean_dialogue": {**copy.deepcopy(row["dialogue"]), "frozen_dialogue": True},
        "hashes": hashes,
        "current_anchor_to_event_instance_id": {},
        "current_event_instance_to_anchor": {},
    }


def recalibration_summary(accepted: Path, clean_count: int) -> dict:
    resolved = str(accepted.resolve())
    return {
        "clean_count": clean_count,
        "k_eval": 5,
        "permutation_mode": "cyclic",
        "min_frontier_permutation_share": 1.0,
        "quality": {"semantic_quality_gate_enabled": True},
        "decoding_health": {"healthy": True},
        "accepted_stage2_solver_resolved": resolved,
        "loaded_solver_model_resolved": resolved,
        "loaded_solver_adapter": None,
        "base_model": "/models/base",
        "max_model_len": 32768,
        "max_new_tokens": 80,
        "temperature": 0.7,
        "top_p": 0.9,
        "solver_eval_seed": 20260711,
        "generation_seed": 20260710,
    }


class SolverFeedbackMiningTest(unittest.TestCase):
    def test_locate_marked_samples_prefers_truth_first_qa_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            generation = run / "generation"
            recalibration = run / "stage64_mining" / "recalibrated"
            generation.mkdir(parents=True)
            recalibration.mkdir(parents=True)
            legacy = generation / "marked_samples.jsonl"
            truth_first = generation / "claim_first_marked_qa_rows.jsonl"
            legacy.write_text("{}\n", encoding="utf-8")
            truth_first.write_text("{}\n", encoding="utf-8")

            located = subject.locate_marked_samples(recalibration)

        self.assertEqual(located, truth_first)

    def make_batch(self, root: Path, rows: list[dict], accepted: Path) -> Path:
        generation = root / "generation"
        recalibration = root / "stage3_mining" / "recalibrated"
        subject.write_jsonl(generation / "marked_samples.jsonl", [marked_bundle(row) for row in rows])
        subject.write_jsonl(recalibration / "clean_samples.jsonl", rows)
        subject.write_json(
            recalibration / "recalibration_summary.json",
            recalibration_summary(accepted, len(rows)),
        )
        return recalibration

    def test_export_carries_full_cyclic_feedback_and_never_authorizes_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            rows = [
                reasoning_target_sample("easy"),
                audited_sample("position", "position_sensitive", [2, 1, 4, 5]),
                audited_sample("frontier", "robust_frontier", [2, 2, 2, 2]),
            ]
            recalibration = self.make_batch(root / "run", rows, accepted)
            output = root / "queue"
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(
                    subject.native,
                    "semantic_graph_plan_quality_issues",
                    return_value=[],
                ) as graph_gate,
            ):
                manifest = subject.export_queue(
                    [recalibration],
                    output,
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )
            queue = subject.read_jsonl(output / "hardening_queue.jsonl")
            frozen = subject.read_jsonl(output / "frozen_samples.jsonl")
            feedback_manifest = subject.read_jsonl(output / "feedback_manifest.jsonl")
            retained = subject.read_jsonl(output / "robust_frontier_candidates.jsonl")
            self.assertEqual(manifest["queue_count"], 2)
            self.assertEqual(
                manifest["status_counts"],
                {"position_sensitive": 1, "robust_easy5": 1, "robust_frontier": 1},
            )
            self.assertEqual(
                manifest["rewrite_status_counts"],
                {"position_sensitive": 1, "robust_easy5": 1},
            )
            self.assertEqual(manifest["feedback_manifest_count"], 3)
            self.assertEqual(manifest["retained_frontier_count"], 1)
            self.assertFalse(manifest["automatic_api_calls"])
            self.assertFalse(manifest["automatic_solver_calls"])
            self.assertFalse(manifest["automatic_training"])
            self.assertFalse(manifest["training_authorized"])
            self.assertEqual(manifest["graph_contract_gate"], "semantic_graph_v17")
            self.assertEqual(
                graph_gate.call_args_list[0].args[2],
                rows[0]["stage64_target"],
            )
            self.assertEqual(manifest["search_eval_seed"], 20260711)
            self.assertEqual(manifest["generation_seed"], 20260710)
            self.assertEqual(manifest["final_eval_seed"], 20260712)
            self.assertTrue(manifest["independent_final_eval_seed"])
            self.assertEqual(len(queue), 2)
            self.assertEqual(len(frozen), 2)
            self.assertEqual(len(feedback_manifest), 3)
            self.assertEqual([row["sample_id"] for row in retained], ["frontier"])
            for item in queue:
                feedback = item["challenger_request"]["fixed_stage2_feedback"]
                self.assertEqual(feedback["k_eval_per_permutation"], 5)
                self.assertNotIn("permutations", feedback)
                self.assertNotIn("position_results", feedback)
                self.assertNotIn("wrong_predictions", feedback)
                self.assertNotIn("predicted_label_histogram", feedback)
                self.assertTrue(feedback["raw_shifted_labels_omitted"])
                self.assertTrue(feedback["raw_predictions_omitted"])
                self.assertEqual(
                    feedback["cyclic_exact_count_summary"]["counts_sorted"],
                    [1, 2, 4, 5]
                    if item["source"]["status"] == "position_sensitive"
                    else [5, 5, 5, 5],
                )
                self.assertEqual(
                    item["challenger_request"]["required_output_schema"]["frozen_bundle_sha256"],
                    item["frozen_contract"]["frozen_bundle_sha256"],
                )
                self.assertEqual(item["source"]["generation_seed"], 20260710)
                self.assertEqual(feedback["aggregate_rollout_count"], 20)
                self.assertIn("difficulty_score", feedback)
                self.assertEqual(item["target"]["target_band"], subject.TARGET_BAND)
            commands = [shlex.split(command) for command in manifest["closed_loop_commands"]]
            commands_by_script = {Path(command[1]).name: command for command in commands}
            self.assertEqual(
                set(commands_by_script),
                {
                    "stage3_solver_feedback_mining.py",
                    "shuffle_stage64_options.py",
                    "recalibrate_stage64_shuffled_frontier.py",
                },
            )
            shuffle_command = commands_by_script["shuffle_stage64_options.py"]
            recalibrate_command = commands_by_script["recalibrate_stage64_shuffled_frontier.py"]
            self.assertEqual(
                shuffle_command[shuffle_command.index("--seed") + 1],
                "20260712",
            )
            self.assertEqual(
                recalibrate_command[recalibrate_command.index("--seed") + 1],
                "20260712",
            )
            self.assertNotIn("20260711", recalibrate_command)
            self.assertEqual(
                recalibrate_command[recalibrate_command.index("--generation-seed") + 1],
                "20260710",
            )
            self.assertNotIn("train", " ".join(" ".join(command) for command in commands).lower())

    def test_cyclic_feedback_classifies_all_outcomes_and_scores_boundary(self) -> None:
        cases = [
            ("easy", "robust_easy5", [5, 5, 5, 5], "harden", 1.0),
            ("hard", "robust_hard0", [0, 0, 0, 0], "soften_or_diagnostic", 0.0),
            ("position", "position_sensitive", [0, 1, 4, 5], "repair_position_sensitivity", 0.5),
            ("frontier", "robust_frontier", [1, 2, 3, 4], "retain_frontier", 0.5),
        ]
        for sample_id, status, counts, action, expected_accuracy in cases:
            with self.subTest(status=status):
                feedback = subject.validate_cyclic_audit(audited_sample(sample_id, status, counts))
                self.assertEqual(feedback["decision"]["action"], action)
                self.assertEqual(feedback["aggregate_rollout_count"], 20)
                self.assertEqual(feedback["aggregate_exact_accuracy"], expected_accuracy)
                self.assertEqual(len(feedback["position_results"]), 4)
                self.assertEqual(sum(row["selected_count"] for row in feedback["semantic_option_confusion"]), 20)
                self.assertEqual(
                    feedback["all_positions_inside_target_band"],
                    status == "robust_frontier",
                )
        frontier = subject.validate_cyclic_audit(
            audited_sample("duplicate", "robust_frontier", [1, 2, 3, 4]),
            duplicate_count=2,
        )
        self.assertEqual(frontier["difficulty_score"]["boundary_reward"], 1.0)
        self.assertEqual(frontier["difficulty_score"]["duplicate_penalty"], 0.075)
        self.assertEqual(frontier["difficulty_score"]["position_penalty"], 0.15)
        self.assertEqual(frontier["difficulty_score"]["score"], 0.775)

    def test_export_queues_hard0_for_softening_and_keeps_diagnostic_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            row = audited_sample("hard", "robust_hard0", [0, 0, 0, 0])
            recalibration = self.make_batch(root / "run", [row], accepted)
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
            ):
                manifest = subject.export_queue(
                    [recalibration],
                    root / "queue",
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )
            queue = subject.read_jsonl(root / "queue" / "hardening_queue.jsonl")
            diagnostic = subject.read_jsonl(root / "queue" / "diagnostic_hard0_samples.jsonl")
            self.assertEqual(manifest["diagnostic_hard0_count"], 1)
            self.assertEqual(len(queue), 1)
            self.assertEqual(len(diagnostic), 1)
            self.assertEqual(queue[0]["target"]["action"], "soften_or_diagnostic")
            self.assertEqual(queue[0]["target"]["difficulty_target"], "softer_boundary")
            self.assertTrue(
                queue[0]["challenger_request"]["fixed_stage2_feedback"]["decision"][
                    "diagnostic_only_allowed"
                ]
            )

    def test_export_can_retain_frontier_without_creating_api_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            rows = [
                audited_sample("frontier1", "robust_frontier", [2, 2, 2, 2]),
                audited_sample("frontier2", "robust_frontier", [2, 2, 2, 2]),
            ]
            recalibration = self.make_batch(root / "run", rows, accepted)
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
            ):
                manifest = subject.export_queue(
                    [recalibration],
                    root / "queue",
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )
            self.assertEqual(manifest["queue_count"], 0)
            self.assertEqual(manifest["retained_frontier_count"], 2)
            self.assertEqual(manifest["closed_loop_commands"], [])
            self.assertEqual(subject.read_jsonl(root / "queue" / "hardening_queue.jsonl"), [])
            self.assertEqual(
                len(subject.read_jsonl(root / "queue" / "robust_frontier_candidates.jsonl")),
                2,
            )
            feedback = subject.read_jsonl(root / "queue" / "feedback_manifest.jsonl")
            self.assertTrue(all(row["solver_feedback"]["duplicate_audit"]["is_duplicate"] for row in feedback))
            self.assertTrue(all(row["solver_feedback"]["duplicate_audit"]["penalty"] == 0.075 for row in feedback))

    def test_generate_qa_responses_checkpoints_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_dir = root / "queue"
            queue_dir.mkdir()
            rows = [
                {
                    "queue_id": f"q{index}",
                    "challenger_request": {"task": "qa-only", "index": index},
                }
                for index in range(2)
            ]
            subject.write_json(
                queue_dir / "queue_ready_manifest.json",
                {"schema_version": subject.SCHEMA_VERSION, "training_authorized": False},
            )
            subject.write_jsonl(queue_dir / "hardening_queue.jsonl", rows)
            config = root / "config.json"
            subject.write_json(
                config,
                {"api": {"api_key": "secret", "base_url": "https://example.test/v1", "model": "model"}},
            )

            def response(_runtime, prompt, **_kwargs):
                index = prompt["index"]
                return {
                    "queue_id": f"q{index}",
                    "frozen_bundle_sha256": f"hash{index}",
                    "qa_proposal": {"question": f"question {index}"},
                }, {"used": True}

            responses = queue_dir / "challenger_qa_responses.jsonl"
            with mock.patch.object(subject.native, "call_raw_json_api", side_effect=response) as call:
                summary = subject.generate_qa_responses(queue_dir, config, responses)
                resumed = subject.generate_qa_responses(queue_dir, config, responses)
            self.assertEqual(summary["successful_response_count"], 2)
            self.assertEqual(summary["remaining_count"], 0)
            self.assertEqual(resumed["attempted_count"], 0)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(len(subject.read_jsonl(responses)), 2)
            self.assertFalse(summary["automatic_solver_calls"])
            self.assertFalse(summary["training_authorized"])

    def test_export_fails_closed_when_eligible_sample_lacks_marked_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            rows = [audited_sample("easy", "robust_easy5", [5, 5, 5, 5])]
            recalibration = self.make_batch(root / "run", rows, accepted)
            subject.write_jsonl(root / "run" / "generation" / "marked_samples.jsonl", [])
            with self.assertRaisesRegex(RuntimeError, "no original marked generation bundle"):
                with mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]):
                    subject.export_queue(
                        [recalibration],
                        root / "queue",
                        accepted_stage2_solver_model=accepted,
                        final_eval_seed=20260712,
                    )

    def test_export_fails_closed_on_missing_or_stale_generation_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            cases = {
                "missing": None,
                "stale": subject.stage3_operations.generation_protocol(
                    graph_gate_version="semantic_graph_v16"
                ),
            }
            for case_name, protocol in cases.items():
                with self.subTest(case=case_name):
                    row = audited_sample(
                        f"easy-{case_name}", "robust_easy5", [5, 5, 5, 5]
                    )
                    if protocol is None:
                        row.pop("stage3_generation_protocol")
                    else:
                        row["stage3_generation_protocol"] = protocol
                    recalibration = self.make_batch(
                        root / f"run-{case_name}", [row], accepted
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "missing or stale Stage3 generation provenance"
                    ):
                        subject.export_queue(
                            [recalibration],
                            root / f"queue-{case_name}",
                            accepted_stage2_solver_model=accepted,
                            final_eval_seed=20260712,
                        )

    def test_export_requires_independent_final_eval_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            row = audited_sample("easy", "robust_easy5", [5, 5, 5, 5])
            recalibration = self.make_batch(root / "run", [row], accepted)
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
                self.assertRaisesRegex(RuntimeError, "must differ"),
            ):
                subject.export_queue(
                    [recalibration],
                    root / "queue",
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260711,
                )

    def test_export_requires_one_recorded_generation_seed_per_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            first = self.make_batch(
                root / "run1",
                [audited_sample("easy1", "robust_easy5", [5, 5, 5, 5])],
                accepted,
            )
            second = self.make_batch(
                root / "run2",
                [audited_sample("easy2", "robust_easy5", [5, 5, 5, 5])],
                accepted,
            )
            second_summary_path = second / "recalibration_summary.json"
            second_summary = subject.read_json(second_summary_path)
            second_summary["generation_seed"] = 20260720
            subject.write_json(second_summary_path, second_summary)
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
                self.assertRaisesRegex(RuntimeError, "one challenger generation seed per queue"),
            ):
                subject.export_queue(
                    [first, second],
                    root / "queue",
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )

            first_summary_path = first / "recalibration_summary.json"
            first_summary = subject.read_json(first_summary_path)
            first_summary.pop("generation_seed")
            subject.write_json(first_summary_path, first_summary)
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
                self.assertRaisesRegex(RuntimeError, "lacks an integer challenger generation_seed"),
            ):
                subject.export_queue(
                    [first],
                    root / "queue_missing",
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )

    def test_unhealthy_or_nonaccepted_solver_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            row = audited_sample("easy", "robust_easy5", [5, 5, 5, 5])
            recalibration = self.make_batch(root / "run", [row], accepted)
            summary = subject.read_json(recalibration / "recalibration_summary.json")
            summary["decoding_health"]["healthy"] = False
            subject.write_json(recalibration / "recalibration_summary.json", summary)
            with self.assertRaisesRegex(RuntimeError, "Untrusted Stage3 recalibration batch"):
                subject.export_queue(
                    [recalibration],
                    root / "queue",
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )

    def test_frozen_bundle_tampering_is_detected(self) -> None:
        row = audited_sample("easy", "robust_easy5", [5, 5, 5, 5])
        bundle = marked_bundle(row)
        wrapper = subject.queue_item(
            row,
            bundle,
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )[1]
        wrapper["frozen_bundle"]["dialogue"]["turns"][0]["text"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            subject.verify_frozen_wrapper(wrapper)

    def test_queue_freezes_complete_stage64_reasoning_target_and_provenance(self) -> None:
        row = reasoning_target_sample()
        item, wrapper = subject.queue_item(
            row,
            marked_bundle(row),
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )
        frozen_target = item["frozen_contract"]["stage64_target_contract"]
        self.assertEqual(frozen_target, wrapper["frozen_bundle"]["stage64_target_contract"])
        self.assertEqual(item["target"]["stage64_target"], row["stage64_target"])
        self.assertEqual(
            item["target"]["stage64_reasoning_contract"],
            {
                **{
                    field: row["stage64_target"][field]
                    for field in subject.STAGE64_REASONING_TARGET_FIELDS
                },
                "forbid_max_time_shortcut": True,
            },
        )
        self.assertEqual(
            item["source"]["stage64_target_provenance"]["stage64_target_identity"],
            row["stage64_target_identity"],
        )
        self.assertEqual(
            item["challenger_request"]["immutable_stage64_target_contract"],
            frozen_target,
        )
        requirements = " ".join(item["challenger_request"]["requirements"])
        for field in (*subject.STAGE64_REASONING_TARGET_FIELDS, "forbid_max_time_shortcut"):
            self.assertIn(field, requirements)
        feedback = item["challenger_request"]["fixed_stage2_feedback"]
        self.assertTrue(feedback["raw_shifted_labels_omitted"])
        self.assertTrue(feedback["raw_predictions_omitted"])
        for forbidden in (
            "permutations",
            "position_results",
            "wrong_predictions",
            "predicted_label_histogram",
        ):
            self.assertNotIn(forbidden, feedback)
        compact_dialogue = item["challenger_request"]["frozen_dialogue"]
        self.assertTrue(compact_dialogue["evidence_only"])
        self.assertIsInstance(
            item["challenger_request"]["required_output_schema"]["qa_proposal"],
            dict,
        )
        self.assertIn("query_plan", item["challenger_request"]["required_output_schema"]["qa_proposal"])
        self.assertIn("metadata", requirements.lower())
        self.assertIn("stock answer-leading wording", requirements)
        self.assertIn("complete, self-contained qa_proposal", requirements)
        self.assertIn("gold must be a JSON array", requirements)
        self.assertIn("Keep all four same-slot chain option values", requirements)
        self.assertNotIn("replace weak distractors", requirements)

    def test_reasoning_target_drift_is_rejected_before_native_builder(self) -> None:
        row = reasoning_target_sample("drift")
        wrapper = subject.queue_item(
            row,
            marked_bundle(row),
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )[1]
        proposal = copy.deepcopy(row["qa"])
        proposal["reasoning_template"] = "direct_lookup"
        with (
            mock.patch.object(subject.native, "_build_stage64_api_qa_from_proposal") as builder,
            self.assertRaisesRegex(RuntimeError, "explicit conflict in qa_proposal.reasoning_template"),
        ):
            subject.build_repaired_sample(wrapper, proposal)
        builder.assert_not_called()

    def test_hardening_metadata_completion_is_deterministic_and_conflicts_fail(self) -> None:
        row = audited_sample("metadata", "robust_easy5", [5, 5, 5, 5])
        wrapper = subject.queue_item(
            row,
            marked_bundle(row),
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )[1]
        proposal = copy.deepcopy(row["qa"])
        proposal["question"] = "After comparing the role evidence, who has Role1?"
        proposal["question_text"] = proposal["question"]
        proposal["options"][1]["text"] = "Person2 with the competing role record"
        proposal["options"] = [
            proposal["options"][3],
            proposal["options"][0],
            proposal["options"][1],
            proposal["options"][2],
        ]
        for field in (
            "qa_operation",
            "qa_subtype",
            "answer_type",
            "question_type",
            "answer_format",
            "normalized_gold_answer",
            "query_plan",
        ):
            proposal.pop(field, None)
        original = copy.deepcopy(proposal)

        completed = subject.complete_qa_hardening_metadata(wrapper, proposal)

        self.assertEqual(proposal, original)
        self.assertEqual(completed["qa_operation"], "relation_lookup")
        self.assertEqual(completed["qa_subtype"], "attribute_lookup")
        self.assertEqual(completed["answer_type"], "Person")
        self.assertEqual(
            [option["id"] for option in completed["options"]],
            list("ABCD"),
        )
        self.assertEqual(completed["query_plan"]["answer_source"], "graph_anchor")
        self.assertEqual(
            completed["query_plan"]["required_anchor_refs"],
            row["qa"]["anchor_refs"]["support"],
        )

        conflict = copy.deepcopy(proposal)
        conflict["qa_operation"] = "count"
        with self.assertRaisesRegex(RuntimeError, "qa_proposal.qa_operation"):
            subject.complete_qa_hardening_metadata(wrapper, conflict)

        option_label_conflict = copy.deepcopy(proposal)
        option_label_conflict["options"][0]["option_id"] = "A"
        with self.assertRaisesRegex(RuntimeError, "option.id and option.option_id"):
            subject.complete_qa_hardening_metadata(wrapper, option_label_conflict)

        duplicate_label = copy.deepcopy(proposal)
        duplicate_label["options"][-1]["id"] = "D"
        duplicate_label["options"][-1]["option_id"] = "D"
        duplicate_completed = subject.complete_qa_hardening_metadata(
            wrapper,
            duplicate_label,
        )
        self.assertEqual(
            [option["id"] for option in duplicate_completed["options"]],
            ["D", "A", "B", "D"],
        )

        missing_label = copy.deepcopy(proposal)
        missing_label["options"][0].pop("id")
        missing_label["options"][0].pop("option_id")
        missing_completed = subject.complete_qa_hardening_metadata(
            wrapper,
            missing_label,
        )
        self.assertNotIn("id", missing_completed["options"][0])
        self.assertEqual(
            [option.get("id") for option in missing_completed["options"]],
            [None, "A", "B", "C"],
        )

    def test_hardening_rejects_metadata_only_and_reordered_visible_noop(self) -> None:
        row = audited_sample("noop", "robust_easy5", [5, 5, 5, 5])
        wrapper = subject.queue_item(
            row,
            marked_bundle(row),
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )[1]
        proposal = copy.deepcopy(row["qa"])
        proposal["question"] = "  WHO has Role1?!  "
        proposal["question_text"] = proposal["question"]
        proposal["options"] = list(reversed(proposal["options"]))
        proposal.pop("query_plan")
        for field in ("qa_operation", "qa_subtype", "answer_type"):
            proposal.pop(field)
        with (
            mock.patch.object(
                subject.native,
                "_build_stage64_api_qa_from_proposal",
            ) as builder,
            self.assertRaisesRegex(RuntimeError, "metadata-only or reordered no-op"),
        ):
            subject.build_repaired_sample(wrapper, proposal)
        builder.assert_not_called()

        source_only = copy.deepcopy(row["qa"])
        source_only["options"][1]["source_event_instance_id"] = "alternate:E2"
        with (
            mock.patch.object(
                subject.native,
                "_build_stage64_api_qa_from_proposal",
            ) as builder,
            self.assertRaisesRegex(RuntimeError, "metadata-only or reordered no-op"),
        ):
            subject.build_repaired_sample(wrapper, source_only)
        builder.assert_not_called()

    def test_stage64_target_provenance_tampering_is_detected(self) -> None:
        row = reasoning_target_sample("provenance")
        wrapper = subject.queue_item(
            row,
            marked_bundle(row),
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )[1]
        wrapper["original_recalibrated_sample"]["stage64_target_plan_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "invalid Stage64 target identity provenance"):
            subject.verify_frozen_wrapper(wrapper)

    def test_native_builder_receives_full_target_and_output_preserves_identity(self) -> None:
        row = reasoning_target_sample("materialize")
        wrapper = subject.queue_item(
            row,
            marked_bundle(row),
            batch_dir=Path("/batch"),
            generation_seed=20260710,
            solver_identity={},
            solver_feedback=subject.validate_cyclic_audit(row),
        )[1]
        proposal = copy.deepcopy(row["qa"])
        proposal["question"] = "After rejecting the canceled update, who still has Role1?"
        proposal["question_text"] = proposal["question"]
        proposal["options"][1]["text"] = "Person2 from the canceled update"
        built_qa = copy.deepcopy(proposal)
        with (
            mock.patch.object(
                subject.native,
                "_build_stage64_api_qa_from_proposal",
                return_value=built_qa,
            ) as builder,
            mock.patch.object(
                subject.native,
                "semantic_graph_plan_quality_issues",
                return_value=[],
            ) as graph_gate,
            mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
            mock.patch.object(subject.native.base, "validate_sample", return_value={}),
            mock.patch.object(subject.native, "validate_final_clean_sample"),
        ):
            materialized = subject.build_repaired_sample(wrapper, proposal)
        self.assertEqual(graph_gate.call_args.args[2], row["stage64_target"])
        self.assertEqual(builder.call_args.kwargs["stage64_target"], row["stage64_target"])
        self.assertEqual(materialized["stage64_target"], row["stage64_target"])
        self.assertEqual(
            materialized["stage64_target_identity"],
            row["stage64_target_identity"],
        )
        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            self.assertEqual(materialized[field], row["stage64_target"][field])
            self.assertEqual(
                materialized["qa"]["query_plan"][field],
                row["stage64_target"][field],
            )
        self.assertTrue(materialized["forbid_max_time_shortcut"])
        self.assertTrue(materialized["qa"]["query_plan"]["forbid_max_time_shortcut"])

    def test_validate_rejects_any_graph_or_dialogue_response_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            row = audited_sample("easy", "robust_easy5", [5, 5, 5, 5])
            recalibration = self.make_batch(root / "run", [row], accepted)
            queue_dir = root / "queue"
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
            ):
                subject.export_queue(
                    [recalibration],
                    queue_dir,
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )
            queue = subject.read_jsonl(queue_dir / "hardening_queue.jsonl")[0]
            response = {
                "queue_id": queue["queue_id"],
                "frozen_bundle_sha256": queue["frozen_contract"]["frozen_bundle_sha256"],
                "qa_proposal": {"dialogue": [], "qa_operation": "relation_lookup"},
            }
            responses = root / "responses.jsonl"
            subject.write_jsonl(responses, [response])
            summary = subject.validate_responses(queue_dir, responses, root / "validated")
            self.assertFalse(summary["ready_for_shuffle"])
            self.assertEqual(summary["rejected_count"], 1)
            self.assertFalse((root / "validated" / "revalidated_samples.jsonl").exists())

    def test_validate_publishes_only_after_qa_gate_and_removes_stale_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_stage2"
            accepted.mkdir()
            row = audited_sample("easy", "robust_easy5", [5, 5, 5, 5])
            recalibration = self.make_batch(root / "run", [row], accepted)
            queue_dir = root / "queue"
            with (
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
            ):
                subject.export_queue(
                    [recalibration],
                    queue_dir,
                    accepted_stage2_solver_model=accepted,
                    final_eval_seed=20260712,
                )
            queue = subject.read_jsonl(queue_dir / "hardening_queue.jsonl")[0]
            hardened_qa = copy.deepcopy(row["qa"])
            hardened_qa["question"] = "After resolving the competing job references, who has Role1?"
            hardened_qa["question_text"] = hardened_qa["question"]
            hardened_qa["options"][1]["text"] = "Person2 after the later role update"
            response = {
                "queue_id": queue["queue_id"],
                "frozen_bundle_sha256": queue["frozen_contract"]["frozen_bundle_sha256"],
                "qa_proposal": copy.deepcopy(hardened_qa),
            }
            responses = root / "responses.jsonl"
            subject.write_jsonl(responses, [response])
            built_qa = copy.deepcopy(hardened_qa)
            with (
                mock.patch.object(subject.native, "_build_stage64_api_qa_from_proposal", return_value=built_qa),
                mock.patch.object(subject.native, "semantic_graph_plan_quality_issues", return_value=[]),
                mock.patch.object(subject.stage64_quality, "semantic_quality_issues", return_value=[]),
                mock.patch.object(subject.native.base, "validate_sample", return_value={
                    "graph_dialogue_consistent": True,
                    "gold_derivable": True,
                    "option_valid": True,
                    "anti_shortcut_pass": True,
                    "warnings": [],
                }),
                mock.patch.object(subject.native, "validate_final_clean_sample"),
            ):
                summary = subject.validate_responses(queue_dir, responses, root / "validated")
            self.assertTrue(summary["ready_for_shuffle"])
            output = subject.read_jsonl(root / "validated" / "revalidated_samples.jsonl")
            self.assertEqual(len(output), 1)
            materialized = output[0]
            self.assertNotIn("solver_eval", materialized)
            self.assertNotIn("stage64_permutation_eval", materialized)
            self.assertNotIn("frontier_label", materialized)
            self.assertNotIn("option_shuffle", materialized)
            self.assertEqual(materialized["graph_update"], row["graph_update"])
            self.assertEqual(materialized["dialogue"], row["dialogue"])
            self.assertEqual(materialized["qa_operation"], "relation_lookup")
            self.assertEqual(materialized["answer_type"], "Person")
            self.assertTrue(materialized["qa_hardening_provenance"]["requires_option_shuffle_and_fixed_stage2_cyclic_k5"])


if __name__ == "__main__":
    unittest.main()
