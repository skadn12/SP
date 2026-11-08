"""Synthetic gate-safety tests; no model calls or real root interpretation."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import codex_baseline_screen as screen


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BaselineScreenTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        policy = self.folder / "BASELINE_POLICY.md"
        policy.write_text("Synthetic policy fixture only.\n", encoding="utf-8")
        bound = self.folder / "bound-input.json"
        write(bound, {"frozen": True})
        self.plan = {
            "schema_version": "codex_baseline_screen_plan_v1", "batch_id": "synthetic_v116",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "status": "FROZEN_BEFORE_THIS_BATCH_FULL_TEXT_SEMANTIC_READS",
            "policy_path": str(policy), "policy_sha256": sha(policy),
            "candidate_denominator": 8, "selection_max": 4, "reviews_per_candidate": 2,
            "candidate_order": [], "candidates": [],
            "bindings": [{"path": str(bound), "sha256": sha(bound)}],
        }
        for index in range(1, 9):
            root = f"cc-train-episode-{index}"
            baseline = self.folder / f"inputs/root{index}_BASELINE.json"
            turns = [
                {"turn_ref": f"{root}:P01:T001", "scene_ref": None,
                 "speaker": "Alice", "text": "Last week I planned a move."},
                {"turn_ref": "R01:S01:T001", "scene_ref": "R01:S01",
                 "speaker": "Bob", "text": "Today you confirmed that you moved yesterday."},
                {"turn_ref": "R01:S02:T001", "scene_ref": "R01:S02",
                 "speaker": "Alice", "text": "I am your friend and mentor."},
                {"turn_ref": "R01:S03:T001", "scene_ref": "R01:S03",
                 "speaker": "Bob", "text": "We can discuss our next plan."},
            ]
            write(baseline, {"root_id": root, "turns": turns})
            self.plan["candidate_order"].append(root)
            self.plan["candidates"].append({
                "root_id": root, "baseline_path": str(baseline), "baseline_sha256": sha(baseline),
                "baseline_turn_count": len(turns), "baseline_speakers": ["Alice", "Bob"],
            })
        self.plan_path = self.folder / "SCREEN_PLAN.json"
        write(self.plan_path, self.plan)
        self.review_paths = []
        for index in range(8):
            for reader in ("/reader_a", "/reader_b"):
                path = self.folder / "reviews" / f"root{index + 1}_{reader.rsplit('_', 1)[-1]}.json"
                write(path, self.make_review(index, reader))
                self.review_paths.append(path)

    def make_review(self, index=0, reviewer="/reader_a"):
        candidate = self.plan["candidates"][index]
        turns = json.loads(Path(candidate["baseline_path"]).read_text())["turns"]
        return {
            "schema_version": "codex_baseline_review_v1", "policy_version": "v116",
            "root_id": candidate["root_id"], "baseline_path": candidate["baseline_path"],
            "baseline_sha256": candidate["baseline_sha256"], "plan_sha256": sha(self.plan_path),
            "policy_sha256": self.plan["policy_sha256"], "reviewer": reviewer,
            "started_at_utc": "2026-01-01T00:00:01+00:00",
            "finished_at_utc": "2026-01-01T00:00:02+00:00", "full_source_read": True,
            "covered_turn_refs": [turn["turn_ref"] for turn in turns],
            "checks": {category: {"complete": True, "status": "PASS",
                                  "reason": "Synthetic fixture declares no conflicting claims."}
                       for category in screen.CHECKS},
            "findings": [], "continuation_constraints": [], "decision": "PASS",
        }

    def add_finding(self, review, classification="UNRESOLVED_STATE_TIME"):
        disposition = screen.CLASSIFICATIONS[classification]
        turns = json.loads(Path(review["baseline_path"]).read_text())["turns"]
        citations = [{key: turn[key] for key in ("turn_ref", "speaker")} | {"quote": turn["text"]}
                     for turn in turns[:2]]
        review["findings"].append({
            "id": f"finding-{len(review['findings']) + 1}", "category": "fact_state_time",
            "classification": classification, "disposition": disposition,
            "reason": "Synthetic fixture exercises a declared finding, not semantic truth.",
            "citations": citations, "temporal_evidence": citations[1:] if classification == "COMPATIBLE_TIME_CHANGE" else [],
        })
        status = max([review["decision"], disposition], key=screen.RANK.__getitem__)
        review["checks"]["fact_state_time"]["status"] = status
        review["decision"] = status
        if status != "PASS":
            review["continuation_constraints"] = ["Do not infer a date or a move beyond the original text."]
        return review

    def assert_invalid(self, review):
        path = self.folder / "invalid_review.json"
        write(path, review)
        with self.assertRaises(screen.ReviewValidationError):
            screen.validate_review(self.plan_path, path)

    def test_valid_review_returned_unchanged(self):
        review = self.make_review()
        self.assertEqual(screen.validate_review(self.plan_path, self.review_paths[0]), review)

    def test_coverage_missing_duplicate_and_reordered_turns_rejected(self):
        original = self.make_review()
        for refs in [original["covered_turn_refs"][:-1], original["covered_turn_refs"] + original["covered_turn_refs"][:1],
                     original["covered_turn_refs"][::-1]]:
            review = copy.deepcopy(original)
            review["covered_turn_refs"] = refs
            self.assert_invalid(review)

    def test_wrong_hashes_versions_paths_and_root_rejected(self):
        for field, value in [("baseline_sha256", "0" * 64), ("plan_sha256", "0" * 64),
                             ("policy_sha256", "0" * 64), ("schema_version", "other"),
                             ("policy_version", "v115"), ("baseline_path", "/wrong/path"),
                             ("root_id", "cc-train-episode-999")]:
            with self.subTest(field=field):
                review = self.make_review()
                review[field] = value
                self.assert_invalid(review)

    def test_bound_source_policy_and_plan_input_changes_rejected(self):
        for path in [Path(self.plan["candidates"][0]["baseline_path"]),
                     Path(self.plan["policy_path"]), Path(self.plan["bindings"][0]["path"])]:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaises(screen.ReviewValidationError):
                screen.validate_review(self.plan_path, self.review_paths[0])
            path.write_bytes(original)

    def test_check_completeness_and_truthful_aggregate_required(self):
        for mutation in [lambda review: review.update(full_source_read=False),
                         lambda review: review["checks"].pop("identity_coreference"),
                         lambda review: review["checks"]["identity_coreference"].update(complete=False),
                         lambda review: review["checks"]["identity_coreference"].update(reason=" "),
                         lambda review: review["checks"]["identity_coreference"].update(status="WARNING"),
                         lambda review: review.update(decision="HOLD")]:
            review = self.make_review()
            mutation(review)
            self.assert_invalid(review)

    def test_forged_quotes_speakers_and_turns_rejected(self):
        original = self.add_finding(self.make_review())
        for field, value in [("quote", "Text absent from source."), ("speaker", "Other"),
                             ("turn_ref", "missing"), ("quote", " ")]:
            review = copy.deepcopy(original)
            review["findings"][0]["citations"][0][field] = value
            self.assert_invalid(review)

    def test_classification_status_and_category_cannot_be_downgraded(self):
        original = self.add_finding(self.make_review(), "HARD_CONTRADICTION")
        for mutation in [lambda review: review["findings"][0].update(disposition="WARNING"),
                         lambda review: review["findings"][0].update(classification="MADE_UP"),
                         lambda review: review["findings"][0].update(category="made_up"),
                         lambda review: review["checks"]["fact_state_time"].update(status="PASS"),
                         lambda review: review.update(decision="WARNING")]:
            review = copy.deepcopy(original)
            mutation(review)
            self.assert_invalid(review)

    def test_hold_needs_two_distinct_turns(self):
        for classification in ("HARD_CONTRADICTION", "UNRESOLVED_IDENTITY_OR_TITLE", "UNRESOLVED_CORE_EVENT"):
            review = self.add_finding(self.make_review(), classification)
            finding = review["findings"][0]
            finding["citations"] = [finding["citations"][0]] * 2
            self.assert_invalid(review)

    def test_nonpass_needs_string_constraints(self):
        for constraints in [[], "Do not infer.", [{}], [""]]:
            review = self.add_finding(self.make_review())
            review["continuation_constraints"] = constraints
            self.assert_invalid(review)

    def test_time_change_needs_valid_temporal_evidence(self):
        review = self.add_finding(self.make_review(), "COMPATIBLE_TIME_CHANGE")
        path = self.folder / "time_review.json"
        write(path, review)
        self.assertEqual(screen.validate_review(self.plan_path, path), review)
        review["findings"][0]["temporal_evidence"] = []
        self.assert_invalid(review)
        review["findings"][0]["temporal_evidence"] = [{"turn_ref": "missing", "speaker": "Alice", "quote": "yesterday"}]
        self.assert_invalid(review)

    def test_invalid_and_nonutc_time_order_rejected(self):
        for start, finish in [("2025-12-31T23:59:59+00:00", "2026-01-01T00:00:02+00:00"),
                              ("2026-01-01T00:00:03+00:00", "2026-01-01T00:00:02+00:00"),
                              ("2026-01-01T00:00:01", "2026-01-01T00:00:02+00:00"),
                              ("2026-01-01T00:00:01+01:00", "2026-01-01T00:00:02+00:00"),
                              ("2099-01-01T00:00:01+00:00", "2099-01-01T00:00:02+00:00")]:
            review = self.make_review()
            review.update(started_at_utc=start, finished_at_utc=finish)
            self.assert_invalid(review)

    def test_missing_duplicate_identity_and_duplicate_paths_block_all_output(self):
        for paths in [self.review_paths[:-1], self.review_paths[:-1] + self.review_paths[:1]]:
            out = self.folder / "blocked"
            with self.assertRaises(screen.ReviewValidationError):
                screen.finalize_screen(self.plan_path, paths, out)
            self.assertFalse(out.exists())
        duplicate = self.make_review(0, "/reader_a")
        write(self.review_paths[1], duplicate)
        with self.assertRaises(screen.ReviewValidationError):
            screen.finalize_screen(self.plan_path, self.review_paths, self.folder / "blocked")
        self.assertFalse((self.folder / "blocked").exists())

    def test_invalid_review_is_not_silently_skipped(self):
        review = self.make_review(7, "/reader_b")
        review["covered_turn_refs"] = []
        write(self.review_paths[-1], review)
        with self.assertRaises(screen.ReviewValidationError):
            screen.finalize_screen(self.plan_path, self.review_paths, self.folder / "blocked")
        self.assertFalse((self.folder / "blocked").exists())

    def test_first_four_eligible_in_plan_order_and_constraints_preserved(self):
        write(self.review_paths[0], self.add_finding(self.make_review(0), "HARD_CONTRADICTION"))
        warning_a = self.add_finding(self.make_review(1))
        warning_b = self.add_finding(self.make_review(1, "/reader_b"))
        warning_b["continuation_constraints"] = ["Preserve this second reader's independent boundary."]
        write(self.review_paths[2], warning_a)
        write(self.review_paths[3], warning_b)
        results, selected = screen.finalize_screen(self.plan_path, reversed(self.review_paths), self.folder / "out")
        self.assertEqual([row["root_id"] for row in selected["roots"]], self.plan["candidate_order"][1:5])
        self.assertEqual(results["eligible_root_count"], 7)
        self.assertEqual(results["roots"][0]["decision"], "HOLD")
        self.assertFalse(results["roots"][0]["eligible"])
        self.assertEqual(len(selected["roots"][0]["continuation_constraints"]), 2)
        self.assertEqual(selected["screen_results_sha256"], sha(self.folder / "out/SCREEN_RESULTS.json"))
        self.assertFalse(results["training_started"])
        self.assertFalse(results["semantic_correctness_certified_by_program"])

    def test_shortfall_does_not_replenish_or_select_hold(self):
        for index in range(6):
            write(self.review_paths[2 * index], self.add_finding(self.make_review(index), "UNRESOLVED_CORE_EVENT"))
        results, selected = screen.finalize_screen(self.plan_path, self.review_paths, self.folder / "out")
        self.assertEqual(results["candidate_denominator"], 8)
        self.assertEqual(selected["selected_root_count"], 2)
        self.assertEqual([row["root_id"] for row in selected["roots"]], self.plan["candidate_order"][6:])

    def test_outputs_never_overwritten_or_partially_created(self):
        out = self.folder / "out"
        out.mkdir()
        existing = out / "SELECTED_ROOTS.json"
        existing.write_text("Existing immutable artifact.", encoding="utf-8")
        before = existing.read_bytes()
        with self.assertRaises(screen.ReviewValidationError):
            screen.finalize_screen(self.plan_path, self.review_paths, out)
        self.assertEqual(existing.read_bytes(), before)
        self.assertFalse((out / "SCREEN_RESULTS.json").exists())

    def test_duplicate_json_keys_and_finding_ids_rejected(self):
        path = self.folder / "duplicate.json"
        path.write_text('{"decision":"HOLD","decision":"PASS"}', encoding="utf-8")
        with self.assertRaises(screen.ReviewValidationError):
            screen.validate_review(self.plan_path, path)
        review = self.add_finding(self.make_review())
        review["findings"].append(copy.deepcopy(review["findings"][0]))
        self.assert_invalid(review)


if __name__ == "__main__":
    unittest.main()
