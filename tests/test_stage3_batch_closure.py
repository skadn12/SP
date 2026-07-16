import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import stage3_batch_closure as subject


SEARCH_SEED = 20260731
FINAL_SEED = 20260801
GENERATION_SEED = 20260802


def option(label: str, text: str, *, gold: bool = False) -> dict:
    return {
        "id": label,
        "option_id": label,
        "text": text,
        "source_event_instance_id": f"event-{label}",
        "truth_value": gold,
        "is_gold": gold,
    }


def sample(sample_id: str, status: str, *, question: str | None = None, gold: str = "A") -> dict:
    options = [option(label, f"Person {label}", gold=label == gold) for label in "ABCD"]
    if status == "robust_frontier":
        counts = [1, 2, 3, 4]
    elif status == "robust_easy5":
        counts = [5, 5, 5, 5]
    elif status == "position_sensitive":
        counts = [1, 5, 2, 5]
    else:
        counts = [0, 0, 0, 0]
    permutations = []
    for shift, (label, count) in enumerate(zip("ABCD", counts)):
        permutations.append(
            {
                "shift": shift,
                "gold": label,
                "exact_correct_count": count,
                "predictions": [],
            }
        )
    return {
        "sample_id": sample_id,
        "lineage_group": f"lineage-{sample_id}",
        "qa": {
            "question": question or f"Who is correct for {sample_id}?",
            "question_text": question or f"Who is correct for {sample_id}?",
            "options": options,
            "gold": [gold],
            "qa_operation": "relation_lookup",
            "answer_type": "Person",
        },
        "stage64_permutation_eval": {
            "status": status,
            "k_eval_per_permutation": 5,
            "permutation_count": 4,
            "required_frontier_permutation_share": 1.0,
            "permutations": permutations,
        },
    }


def summary(accepted: Path, base: Path, clean_count: int) -> dict:
    return {
        "clean_count": clean_count,
        "k_eval": 5,
        "max_model_len": 32768,
        "max_new_tokens": 80,
        "temperature": 1.0,
        "top_p": 0.9,
        "seed": SEARCH_SEED,
        "solver_eval_seed": SEARCH_SEED,
        "generation_seed": GENERATION_SEED,
        "permutation_mode": "cyclic",
        "min_frontier_permutation_share": 1.0,
        "quality": {"semantic_quality_gate_enabled": True},
        "decoding_health": {"healthy": True},
        "accepted_stage2_solver_resolved": str(accepted.resolve()),
        "loaded_solver_model_resolved": str(accepted.resolve()),
        "loaded_solver_adapter": None,
        "base_model": str(base.resolve()),
    }


class Stage3BatchClosureTest(unittest.TestCase):
    def run_trusted(self, args: argparse.Namespace) -> dict:
        with mock.patch.object(subject.controller, "validate_controller_sample", return_value={}):
            return subject.run(args)

    def make_search(self, root: Path, rows: list[dict], *, marked: bool = False) -> tuple[Path, Path, Path, Path]:
        batch = root / "batch"
        search = batch / "search_recalibrated"
        accepted = root / "accepted_stage2"
        base = root / "base"
        accepted.mkdir(parents=True)
        base.mkdir(parents=True)
        subject.write_jsonl_atomic(search / "clean_samples.jsonl", rows)
        subject.write_json_atomic(search / "recalibration_summary.json", summary(accepted, base, len(rows)))
        # Deliberately empty: closure must source robust rows from clean_samples,
        # never from this diversity-capped artifact.
        subject.write_jsonl_atomic(search / "frontier.jsonl", [])
        if marked:
            subject.write_jsonl_atomic(batch / "generation" / "marked_samples.jsonl", [])
        return batch, search, accepted, base

    def args(
        self,
        root: Path,
        batch: Path,
        search: Path,
        accepted: Path,
        base: Path,
        *,
        config: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            batch_run_dir=batch,
            search_recalibration_dir=search,
            output_dir=root / "closure",
            generation_seed=GENERATION_SEED,
            search_eval_seed=SEARCH_SEED,
            final_eval_seed=FINAL_SEED,
            accepted_stage2_solver_model=accepted,
            base_model=base,
            config=config,
            max_hardening_retries=2,
            gpu_memory_utilization=0.9,
            dry_run=True,
            execute=False,
        )

    def test_empty_hardening_eligible_keeps_all_original_robust_from_clean_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, search, accepted, base = self.make_search(
                root,
                [sample("frontier-one", "robust_frontier")],
            )
            manifest = self.run_trusted(self.args(root, batch, search, accepted, base))
            self.assertEqual(manifest["counts"]["search_original_robust_frontier"], 1)
            self.assertEqual(manifest["counts"]["search_hardening_eligible"], 0)
            stages = [command["stage"] for command in manifest["commands"]]
            self.assertNotIn("qa_export", stages)
            self.assertEqual(stages, ["option_shuffle", "independent_final_cyclic_k5"])
            self.assertEqual(subject.read_jsonl(search / "frontier.jsonl"), [])

    def test_original_frontier_and_hardened_candidate_merge(self) -> None:
        original = sample("same-readable-id", "robust_frontier")
        hardened = sample("hardened-source", "position_sensitive", question="A harder distinct question?")
        hardened["qa_hardening_provenance"] = {"queue_id": "queue-1"}
        combined, duplicates = subject.namespace_and_deduplicate_candidates(
            [original],
            [hardened],
            generation_seed=GENERATION_SEED,
            batch_namespace="batch-namespace",
        )
        self.assertEqual(len(combined), 2)
        self.assertEqual(duplicates, [])
        self.assertEqual(
            {row["stage3_batch_closure"]["source_kind"] for row in combined},
            {"original", "hardened"},
        )
        self.assertEqual(len({row["sample_id"] for row in combined}), 2)

    def test_duplicate_source_id_conflict_fails_closed(self) -> None:
        first = sample("duplicate", "robust_frontier", question="Question one?")
        second = sample("duplicate", "robust_frontier", question="Question two?")
        with self.assertRaisesRegex(RuntimeError, "Conflicting duplicate original candidate ID"):
            subject.namespace_and_deduplicate_candidates(
                [first, second],
                [],
                generation_seed=GENERATION_SEED,
                batch_namespace="batch-namespace",
            )

    def test_namespaced_ids_are_stable_across_input_order(self) -> None:
        rows = [sample("b", "robust_frontier"), sample("a", "robust_frontier")]
        first, _ = subject.namespace_and_deduplicate_candidates(
            rows,
            [],
            generation_seed=GENERATION_SEED,
            batch_namespace="stable-batch",
        )
        second, _ = subject.namespace_and_deduplicate_candidates(
            list(reversed(rows)),
            [],
            generation_seed=GENERATION_SEED,
            batch_namespace="stable-batch",
        )
        self.assertEqual(
            [(row["sample_id"], row["stage3_batch_closure"]["source_sample_id"]) for row in first],
            [(row["sample_id"], row["stage3_batch_closure"]["source_sample_id"]) for row in second],
        )

    def test_dry_run_commands_fix_final_seed_temperature_and_full_cyclic_share(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, search, accepted, base = self.make_search(
                root,
                [sample("easy", "robust_easy5")],
                marked=True,
            )
            config = root / "config.json"
            subject.write_json_atomic(config, {"api": {"api_key": "unused"}})
            manifest = self.run_trusted(self.args(root, batch, search, accepted, base, config=config))
            stages = [command["stage"] for command in manifest["commands"]]
            self.assertIn("qa_export", stages)
            self.assertIn("qa_generate_attempt_03", stages)
            final = next(
                command["argv"]
                for command in manifest["commands"]
                if command["stage"] == "independent_final_cyclic_k5"
            )
            self.assertEqual(final[final.index("--temperature") + 1], "1.0")
            self.assertEqual(final[final.index("--seed") + 1], str(FINAL_SEED))
            self.assertNotIn(str(SEARCH_SEED), final)
            self.assertEqual(final[final.index("--generation-seed") + 1], str(GENERATION_SEED))
            self.assertEqual(final[final.index("--permutation-mode") + 1], "cyclic")
            self.assertEqual(final[final.index("--min-frontier-permutation-share") + 1], "1.0")
            self.assertTrue((root / "closure" / subject.STATE_FILE).is_file())
            self.assertTrue((root / "closure" / subject.MANIFEST_FILE).is_file())

    def test_resume_reuses_state_and_rejects_changed_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, search, accepted, base = self.make_search(
                root,
                [sample("frontier", "robust_frontier")],
            )
            args = self.args(root, batch, search, accepted, base)
            first = self.run_trusted(args)
            second = self.run_trusted(args)
            self.assertEqual(first["config_sha256"], second["config_sha256"])
            state = subject.read_json(root / "closure" / subject.STATE_FILE)
            self.assertEqual(
                sum(item.get("event") == "dry_run_plan_published" for item in state["history"]),
                2,
            )
            changed = sample("frontier", "robust_frontier")
            changed["qa"]["question"] = "Input changed after the dry run"
            subject.write_jsonl_atomic(search / "clean_samples.jsonl", [changed])
            with self.assertRaisesRegex(RuntimeError, "input artifacts changed"):
                self.run_trusted(args)

    def test_stale_generation_protocol_fails_search_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = sample("stale-protocol", "robust_frontier")
            row["stage3_generation_protocol"] = subject.controller.generation_protocol()
            row["stage3_generation_protocol"]["fingerprint"] = "stale-native-source-fingerprint"
            batch, search, accepted, base = self.make_search(root, [row])
            with self.assertRaisesRegex(RuntimeError, "generation protocol/semantic gate"):
                subject.run(self.args(root, batch, search, accepted, base))

    def test_resume_rebuilds_partial_attempt_before_retrying_only_pending_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hardening = root / "hardening"
            attempt = hardening / "attempt_01"
            validated = attempt / "validated"
            queue_rows = [
                {"queue_id": "q1", "source": {"sample_id": "source-1"}},
                {"queue_id": "q2", "source": {"sample_id": "source-2"}},
            ]
            accepted = sample("accepted-hardened", "position_sensitive")
            accepted["qa_hardening_provenance"] = {"queue_id": "q1"}
            rejected = [{"queue_id": "q2", "source_sample_id": "source-2", "reason": "bad proposal"}]
            subject.write_jsonl_atomic(attempt / "hardening_queue.jsonl", queue_rows)
            subject.write_jsonl_atomic(attempt / "challenger_qa_responses.jsonl", [{"queue_id": "q1"}, {"queue_id": "q2"}])
            subject.write_jsonl_atomic(validated / "accepted_revalidated_samples.jsonl", [accepted])
            subject.write_jsonl_atomic(validated / "rejected_repairs.jsonl", rejected)
            subject.write_json_atomic(
                validated / "validation_summary.json",
                {
                    "queue_count": 2,
                    "response_count": 2,
                    "accepted_count": 1,
                    "rejected_count": 1,
                    "missing_response_count": 0,
                    "missing_queue_ids": [],
                    "allow_partial": True,
                    "ready_for_shuffle": False,
                },
            )
            attempt_record = {
                "attempt": 1,
                "queue_count": 2,
                "accepted_queue_ids": ["q1"],
                "retry_queue_ids": ["q2"],
                "validation_summary": subject.artifact_record(validated / "validation_summary.json"),
                "responses": subject.artifact_record(attempt / "challenger_qa_responses.jsonl"),
                "accepted_revalidated_samples": subject.artifact_record(
                    validated / "accepted_revalidated_samples.jsonl"
                ),
                "rejected_repairs": subject.artifact_record(validated / "rejected_repairs.jsonl"),
            }
            state = {"hardening": {"attempts": [attempt_record]}}
            rebuilt, pending, history, completed = subject.rebuild_completed_hardening(
                state,
                hardening_dir=hardening,
                original_queue_ids={"q1", "q2"},
            )
            self.assertEqual(set(rebuilt), {"q1"})
            self.assertEqual(pending, {"q2"})
            self.assertEqual(completed, 1)
            self.assertEqual(history["q2"][0]["reason"], "bad proposal")
            self.assertEqual(history["q2"][0]["attempt"], 1)

    def test_retry_queue_injects_previous_rejection_without_changing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt_one = root / "hardening" / "attempt_01"
            retry_dir = root / "hardening" / "attempt_02"
            frozen_contract = {
                "frozen_bundle_sha256": "frozen-sha",
                "gold": {"answer_text": "Person A"},
            }
            original_queue = [
                {
                    "queue_id": "q-retry",
                    "frozen_contract": frozen_contract,
                    "challenger_request": {
                        "task": "rewrite qa only",
                        "requirements": ["Return a QA-only response."],
                    },
                }
            ]
            original_frozen = [
                {
                    "queue_id": "q-retry",
                    "frozen_contract": frozen_contract,
                    "frozen_bundle": {"dialogue": {"turns": []}},
                }
            ]
            subject.write_jsonl_atomic(
                attempt_one / "hardening_queue.jsonl",
                original_queue,
            )
            subject.write_jsonl_atomic(
                attempt_one / "frozen_samples.jsonl",
                original_frozen,
            )
            subject.write_json_atomic(
                attempt_one / "queue_ready_manifest.json",
                {"schema_version": "test", "queue_count": 1},
            )
            subject.write_jsonl_atomic(
                attempt_one / "validated" / "rejected_repairs.jsonl",
                [
                    {
                        "queue_id": "q-retry",
                        "source_sample_id": "source-1",
                        "reason": "QA hardening changed answer_type",
                    }
                ],
            )
            subject.write_json_atomic(
                attempt_one / "validated" / "validation_summary.json",
                {"missing_queue_ids": []},
            )

            subject._retry_queue(
                original_attempt_dir=attempt_one,
                previous_attempt_dir=attempt_one,
                retry_dir=retry_dir,
                queue_ids={"q-retry"},
                retry_index=2,
            )

            retry_queue = subject.read_jsonl(retry_dir / "hardening_queue.jsonl")
            self.assertEqual(len(retry_queue), 1)
            retry = retry_queue[0]
            self.assertEqual(retry["queue_id"], original_queue[0]["queue_id"])
            self.assertEqual(retry["frozen_contract"], frozen_contract)
            self.assertEqual(
                retry["challenger_request"]["previous_validation_failure"],
                {
                    "attempt": 1,
                    "queue_id": "q-retry",
                    "kind": "rejected_repair",
                    "reason": "QA hardening changed answer_type",
                },
            )
            self.assertIn(
                "complete self-contained qa_proposal object",
                " ".join(retry["challenger_request"]["requirements"]),
            )
            self.assertNotIn(
                "previous_validation_failure",
                subject.read_jsonl(attempt_one / "hardening_queue.jsonl")[0][
                    "challenger_request"
                ],
            )
            self.assertEqual(
                subject.read_jsonl(retry_dir / "frozen_samples.jsonl"),
                original_frozen,
            )
            manifest = subject.read_json(retry_dir / "queue_ready_manifest.json")
            self.assertEqual(manifest["retry_attempt"], 2)
            self.assertEqual(manifest["previous_validation_attempt"], 1)
            self.assertEqual(manifest["retry_queue_ids"], ["q-retry"])

    def test_hard0_is_planned_for_controlled_softening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, search, accepted, base = self.make_search(
                root,
                [sample("hard", "robust_hard0")],
                marked=True,
            )
            config = root / "config.json"
            subject.write_json_atomic(config, {"api": {"api_key": "unused"}})
            manifest = self.run_trusted(
                self.args(root, batch, search, accepted, base, config=config)
            )
            self.assertEqual(manifest["counts"]["search_hardening_eligible"], 1)
            self.assertIn("qa_export", [item["stage"] for item in manifest["commands"]])

    def test_both_candidate_sources_empty_is_zero_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, search, accepted, base = self.make_search(root, [])
            args = self.args(root, batch, search, accepted, base)
            args.dry_run = False
            args.execute = True
            manifest = self.run_trusted(args)
            self.assertEqual(manifest["status"], "zero_candidate")
            self.assertEqual(manifest["counts"]["combined_candidates"], 0)
            self.assertEqual(manifest["commands_executed"], [])


if __name__ == "__main__":
    unittest.main()
