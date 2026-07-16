from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.current_method import native_api_graph_dialogue_expansion as subject


def clean_sample(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "attack_skill": "relation_lookup",
        "qa": {"question_type": "single_choice"},
        "metadata": {},
        "solver_eval": {"note": "existing data is preserved, never reconstructed"},
    }


def marked_sample(sample_id: str) -> dict:
    return {
        "clean_sample": clean_sample(sample_id),
        "current_event_plan": {"events": []},
    }


def accepted_raw(sample_id: str, *, detail: str = "first") -> dict:
    return {
        "attempt": 1,
        "target_sample_index": 0,
        "accepted": True,
        "sample_id": sample_id,
        "detail": detail,
    }


def rejected_raw(attempt: int, detail: str) -> dict:
    return {
        "attempt": attempt,
        "target_sample_index": 1,
        "accepted": False,
        "final_reject_reason": "semantic_graph",
        "detail": detail,
    }


def reject_row(attempt: int, detail: str) -> dict:
    raw = rejected_raw(attempt, detail)
    return {
        "attempt": attempt,
        "target_sample_index": 1,
        "reason": "semantic_graph",
        "raw_record": raw,
    }


class NativeGraphDialogueResumeTest(unittest.TestCase):
    def _prepare_artifacts(self, output: Path) -> tuple[list[dict], list[dict], list[dict]]:
        marked = marked_sample("accepted-1")
        raw_rows = [
            accepted_raw("accepted-1"),
            accepted_raw("accepted-1", detail="duplicate accepted sample id"),
            rejected_raw(2, "first reject"),
            rejected_raw(2, "first reject"),
            rejected_raw(3, "second reject"),
        ]
        reject_rows = [
            reject_row(2, "first reject"),
            reject_row(2, "first reject"),
            reject_row(3, "second reject"),
        ]
        subject.write_jsonl(output / "marked_samples.jsonl", [marked, marked])
        subject.write_jsonl(output / "graph_conditioned_api_dialogue_validated.jsonl", [marked])
        subject.write_jsonl(output / "graph_conditioned_api_dialogue_raw.jsonl", raw_rows)
        subject.write_jsonl(output / "graph_conditioned_api_dialogue_rejects.jsonl", reject_rows)
        with (output / "graph_conditioned_api_dialogue_raw.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"interrupted":')
        return [marked], [raw_rows[0], raw_rows[2], raw_rows[4]], [reject_rows[0], reject_rows[2]]

    def test_resume_loader_preserves_valid_rows_and_deduplicates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            expected_marked, expected_raw, expected_rejects = self._prepare_artifacts(output)

            marked, raw, rejects, diagnostics = subject._load_resume_audit_artifacts(
                output,
                [clean_sample("accepted-1"), clean_sample("accepted-2")],
            )

        self.assertEqual(marked, expected_marked)
        self.assertEqual(raw, expected_raw)
        self.assertEqual(rejects, expected_rejects)
        self.assertEqual(diagnostics["resume_marked_loaded_count"], 1)
        self.assertEqual(diagnostics["resume_marked_duplicate_row_count"], 2)
        self.assertEqual(diagnostics["resume_raw_loaded_count"], 3)
        self.assertEqual(diagnostics["resume_raw_duplicate_row_count"], 2)
        self.assertEqual(diagnostics["resume_raw_file_invalid_row_count"], 1)
        self.assertEqual(diagnostics["resume_reject_loaded_count"], 2)
        self.assertEqual(diagnostics["resume_reject_duplicate_row_count"], 1)
        self.assertEqual(diagnostics["resume_clean_without_marked_count"], 1)
        self.assertEqual(diagnostics["resume_clean_without_accepted_raw_count"], 1)
        self.assertEqual(diagnostics["resume_audit_reconstructed_count"], 0)

    def test_repeated_resume_preserves_audits_without_api_or_row_growth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            partial = output / "samples.partial.jsonl"
            resumed = [clean_sample("accepted-1"), clean_sample("accepted-2")]
            subject.write_jsonl(partial, resumed)
            with partial.open("a", encoding="utf-8") as handle:
                handle.write('{"interrupted":')
            expected_marked, expected_raw, expected_rejects = self._prepare_artifacts(output)
            args = SimpleNamespace(native_curriculum_mode="default")

            with mock.patch.object(
                subject,
                "call_raw_json_api",
                side_effect=AssertionError("resume at target must not call the API"),
            ):
                first = subject.generate_graph_conditioned_api_dialogue_samples(
                    seed_rows=[{"sample_id": "seed-1"}],
                    policy={},
                    n_samples=2,
                    output_dir=output,
                    api_args=args,
                    partial_samples=partial,
                )
                second = subject.generate_graph_conditioned_api_dialogue_samples(
                    seed_rows=[{"sample_id": "seed-1"}],
                    policy={},
                    n_samples=2,
                    output_dir=output,
                    api_args=args,
                    partial_samples=partial,
                )

            summary = json.loads(
                (output / "graph_conditioned_api_dialogue_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first, resumed)
            self.assertEqual(second, resumed)
            self.assertEqual(subject.read_jsonl_if_exists(partial), resumed)
            self.assertEqual(subject.read_jsonl_if_exists(output / "marked_samples.jsonl"), expected_marked)
            self.assertEqual(
                subject.read_jsonl_if_exists(output / "graph_conditioned_api_dialogue_validated.jsonl"),
                expected_marked,
            )
            self.assertEqual(subject.read_jsonl_if_exists(output / "graph_conditioned_api_dialogue_raw.jsonl"), expected_raw)
            self.assertEqual(
                subject.read_jsonl_if_exists(output / "graph_conditioned_api_dialogue_rejects.jsonl"),
                expected_rejects,
            )
            self.assertEqual(summary["resume_loaded_count"], 2)
            self.assertEqual(summary["resume_marked_loaded_count"], 1)
            self.assertEqual(summary["resume_raw_loaded_count"], 3)
            self.assertEqual(summary["resume_reject_loaded_count"], 2)
            self.assertEqual(summary["resume_clean_without_marked_count"], 1)
            self.assertEqual(summary["resume_clean_without_accepted_raw_count"], 1)
            self.assertEqual(summary["resume_audit_reconstructed_count"], 0)
            self.assertEqual(summary["audit_marked_sample_count"], 1)
            self.assertEqual(summary["audit_raw_record_count"], 3)
            self.assertEqual(summary["audit_reject_count"], 2)


if __name__ == "__main__":
    unittest.main()
