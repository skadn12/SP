from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.current_method.stage3_run_isolation import validate_fresh_stage3_run


class Stage3RunIsolationTest(unittest.TestCase):
    def test_fresh_run_accepts_only_local_empty_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "174_stage3_backfilled_smoke_n1"
            output = run / "generation"
            audit = validate_fresh_stage3_run(
                run_root=run,
                output_dir=output,
                partial_samples=output / "samples.partial.jsonl",
                failed_api_calls=output / "failed_api_calls.jsonl",
                graph_proposal_cache_dir=None,
                fixed_stage3_graph_artifact=None,
                stage3_operation_state_path=run / "state" / "stage3_operation.json",
                no_resume=True,
            )
            self.assertTrue(audit["pass"])
            self.assertTrue(audit["no_resume"])

    def test_fresh_run_rejects_resume_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "truth_first_n1"
            output = run / "generation"
            with self.assertRaisesRegex(RuntimeError, "requires_no_resume"):
                validate_fresh_stage3_run(
                    run_root=run,
                    output_dir=output,
                    partial_samples=output / "samples.partial.jsonl",
                    failed_api_calls=output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=run / "state" / "stage3_operation.json",
                    no_resume=False,
                )

    def test_fresh_run_rejects_fixed_graph_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "truth_first_n1"
            output = run / "generation"
            old_artifact = root / "229_old" / "graph_artifacts" / "old.json"
            with self.assertRaisesRegex(RuntimeError, "fixed_artifact_forbidden"):
                validate_fresh_stage3_run(
                    run_root=run,
                    output_dir=output,
                    partial_samples=output / "samples.partial.jsonl",
                    failed_api_calls=output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=old_artifact,
                    stage3_operation_state_path=run / "state" / "stage3_operation.json",
                    no_resume=True,
                )

    def test_fresh_run_rejects_cache_outside_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "truth_first_n1"
            output = run / "generation"
            old_cache = root / "229_old" / "generation" / "graph_proposal_cache"
            with self.assertRaisesRegex(RuntimeError, "cache_outside_output"):
                validate_fresh_stage3_run(
                    run_root=run,
                    output_dir=output,
                    partial_samples=output / "samples.partial.jsonl",
                    failed_api_calls=output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=old_cache,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=run / "state" / "stage3_operation.json",
                    no_resume=True,
                )

    def test_old_partial_path_is_rejected_without_reading_or_overwriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_partial = root / "216_old" / "generation" / "samples.partial.jsonl"
            old_partial.parent.mkdir(parents=True)
            old_partial.write_text("old-state\n", encoding="utf-8")
            new_output = root / "174_new" / "generation"
            with self.assertRaisesRegex(RuntimeError, "state_path_outside_output"):
                validate_fresh_stage3_run(
                    run_root=root / "174_new",
                    output_dir=new_output,
                    partial_samples=old_partial,
                    failed_api_calls=new_output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=root / "174_new" / "state" / "stage3_operation.json",
                    no_resume=True,
                )
            self.assertEqual(old_partial.read_text(encoding="utf-8"), "old-state\n")

    def test_existing_new_run_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "174_new"
            output = run / "generation"
            output.mkdir(parents=True)
            (output / "samples.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "output_contains_state"):
                validate_fresh_stage3_run(
                    run_root=run,
                    output_dir=output,
                    partial_samples=output / "samples.partial.jsonl",
                    failed_api_calls=output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=run / "state" / "stage3_operation.json",
                    no_resume=True,
                )

    def test_output_directory_cannot_point_at_an_old_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new_run = root / "174_new"
            old_output = root / "216_old" / "generation"
            old_output.mkdir(parents=True)
            old_sample = old_output / "samples.partial.jsonl"
            old_sample.write_text("old-state\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "output_outside_run_root"):
                validate_fresh_stage3_run(
                    run_root=new_run,
                    output_dir=old_output,
                    partial_samples=old_output / "samples.partial.jsonl",
                    failed_api_calls=old_output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=new_run / "state" / "stage3_operation.json",
                    no_resume=True,
                )
            self.assertEqual(old_sample.read_text(encoding="utf-8"), "old-state\n")

    def test_nonempty_run_root_is_rejected_even_without_generation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "174_new"
            (run / "reports").mkdir(parents=True)
            (run / "reports" / "prior.json").write_text("{}\n", encoding="utf-8")
            output = run / "generation"
            with self.assertRaisesRegex(RuntimeError, "run_root_not_empty"):
                validate_fresh_stage3_run(
                    run_root=run,
                    output_dir=output,
                    partial_samples=output / "samples.partial.jsonl",
                    failed_api_calls=output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=run / "state" / "stage3_operation.json",
                    no_resume=True,
                )

    def test_shared_operation_controller_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "174_new"
            output = run / "generation"
            shared_state = root / "stage3_operation_controller_state_v5.json"
            with self.assertRaisesRegex(RuntimeError, "operation_state_outside_run_root"):
                validate_fresh_stage3_run(
                    run_root=run,
                    output_dir=output,
                    partial_samples=output / "samples.partial.jsonl",
                    failed_api_calls=output / "failed_api_calls.jsonl",
                    graph_proposal_cache_dir=None,
                    fixed_stage3_graph_artifact=None,
                    stage3_operation_state_path=shared_state,
                    no_resume=True,
                )


if __name__ == "__main__":
    unittest.main()
