import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.prepare_quality_exemplar_curation import (
    DEFAULT_INPUT,
    CurationPreparationError,
    load_quality_rows,
    prepare_curation,
    select_candidates,
    sha256_file,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class QualityExemplarCurationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.input_path = self.root / "input.jsonl"
        self.summary_path = self.root / "summary.json"
        self.history_dir = self.root / "history"
        self.history_dir.mkdir()
        self.output_dir = self.root / "packet"

        rows = []
        bucket_tokens = {"short": 4000, "medium": 6000, "long": 8000}
        index = 0
        for bucket, tokens in bucket_tokens.items():
            for article_index in range(8):
                index += 1
                rows.append(
                    {
                        "id": f"quality:validation:{bucket}_{article_index}",
                        "dataset": "quality",
                        "context": f"unique context {bucket} {article_index}",
                        "question": f"question {index}?",
                        "options": {
                            "A": f"option A {index}",
                            "B": f"option B {index}",
                            "C": f"option C {index}",
                            "D": f"option D {index}",
                        },
                        "answer": ("A", "B", "C", "D")[index % 4],
                        "task_type": "quality_mcq",
                        "prompt_tokens": tokens,
                    }
                )
        rows.append(
            {
                "id": "other:validation:ignored",
                "dataset": "other",
                "context": "must be ignored",
                "question": "ignored?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer": "A",
                "task_type": "other",
                "prompt_tokens": 100,
            }
        )
        self.rows = rows
        write_jsonl(self.input_path, rows)
        self.summary_path.write_text(
            json.dumps(
                {
                    "scored_has_gold_answer": {
                        "under_32k_by_dataset": {"quality": 24}
                    }
                }
            ),
            encoding="utf-8",
        )

        quality_rows = [row for row in rows if row["dataset"] == "quality"]
        report = {
            "model": "/models/historical",
            "results": [
                {
                    "source_record_id": row["id"],
                    "dataset": "quality",
                    "gold_label": row["answer"],
                    "pred_label": row["answer"],
                    "prediction": "must not enter blind packet",
                    "correct": True,
                }
                for row in quality_rows
            ],
        }
        (self.history_dir / "historical.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def prepare(self) -> dict:
        return prepare_curation(
            input_path=self.input_path,
            summary_path=self.summary_path,
            history_dir=self.history_dir,
            output_dir=self.output_dir,
            seed=20260718,
            expected_quality_count=24,
            expected_context_count=24,
            per_bucket=8,
        )

    def test_filters_quality_and_selects_reproducibly(self):
        quality_rows, stats = load_quality_rows(
            self.input_path,
            expected_quality_count=24,
            expected_context_count=24,
        )
        self.assertEqual(len(quality_rows), 24)
        self.assertEqual(stats["split"], "validation")
        first = select_candidates(quality_rows, seed=20260718, per_bucket=8)
        second = select_candidates(quality_rows, seed=20260718, per_bucket=8)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertNotIn("other:validation:ignored", {row["id"] for row in first})

    def test_packet_has_24_items_three_buckets_and_unique_articles(self):
        manifest = self.prepare()
        self.assertEqual(
            manifest["candidate_count_by_bucket"],
            {"short": 8, "medium": 8, "long": 8},
        )
        self.assertEqual(manifest["total_candidate_count"], 24)
        self.assertEqual(manifest["distinct_article_count"], 24)
        candidates = [
            json.loads(line)
            for line in (self.output_dir / "candidate_items.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(candidates), 24)
        context_hashes = {
            hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
            for row in candidates
        }
        self.assertEqual(len(context_hashes), 24)

    def test_blind_packet_does_not_leak_answer_and_key_matches_source(self):
        self.prepare()
        candidates = [
            json.loads(line)
            for line in (self.output_dir / "candidate_items.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        source_by_id = {
            row["id"]: row for row in self.rows if row["dataset"] == "quality"
        }
        answer_key = json.loads(
            (self.output_dir / "answer_key.json").read_text(encoding="utf-8")
        )
        for candidate, key in zip(candidates, answer_key):
            self.assertNotIn("answer", candidate)
            self.assertNotIn("correct_answer", candidate)
            self.assertEqual(candidate["options"], [
                source_by_id[candidate["id"]]["options"][letter]
                for letter in ("A", "B", "C", "D")
            ])
            for result in candidate["historical_solver_results"]:
                self.assertEqual(set(result), {
                    "report_file", "report_sha256", "model", "correct"
                })
            source = source_by_id[key["id"]]
            self.assertEqual(key["correct_answer"], source["answer"])
            self.assertEqual(
                key["correct_option_text"], source["options"][source["answer"]]
            )
        markdown = (self.output_dir / "candidate_items.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("正确答案：", markdown)
        annotations = [
            json.loads(line)
            for line in (self.output_dir / "annotation_template.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertTrue(all(row["keep_as_exemplar"] is None for row in annotations))
        self.assertTrue(all(row["evidence_a_quote"] == "" for row in annotations))

    def test_source_is_unchanged_and_existing_output_is_rejected(self):
        before = sha256_file(self.input_path)
        self.prepare()
        self.assertEqual(sha256_file(self.input_path), before)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_count_mismatch_stops(self):
        with self.assertRaises(CurationPreparationError):
            load_quality_rows(
                self.input_path,
                expected_quality_count=25,
                expected_context_count=24,
            )

    @unittest.skipUnless(DEFAULT_INPUT.exists(), "authoritative local QuALITY input absent")
    def test_authoritative_local_quality_statistics(self):
        rows, stats = load_quality_rows(DEFAULT_INPUT)
        self.assertEqual(len(rows), 2086)
        self.assertEqual(stats["unique_context_count"], 115)
        self.assertTrue(stats["all_ids_validation"])


if __name__ == "__main__":
    unittest.main()
