"""Offline contract tests; no provider, tokenizer, Solver or training is started."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import qa


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def question(kind, root="test-root"):
    labels = "ABCDEF" if kind == "multi_select" else [f"I{i}" for i in range(1, 7)]
    field = "options" if kind == "multi_select" else "items"
    return {"candidate_id": f"{root}-{kind}", "root_id": root, "question_type": kind,
        "question": "Question", field: {label: f"statement {index}" for index, label in enumerate(labels)},
        "gold": list(labels)[:3] if kind == "multi_select" else list(labels),
        "option_evidence" if kind == "multi_select" else "item_evidence": {
            label: {"turn_ref": f"R02:A:S01:T{index:03}", "quote": f"statement {index}"}
            for index, label in enumerate(labels)}}


class QAContractTest(unittest.TestCase):
    def setUp(self):
        self.root = "test-root"
        self.source = {"root_id": self.root, "turns": [{"turn_ref": "P:T01", "speaker": "A", "text": "Base"}]}
        self.probe = SimpleNamespace(contract=SimpleNamespace(
            structured_choices=lambda kind, variant: range(63 if kind == "multi_select" else 720)))

    def plan(self, questions):
        return qa.plan_requests(questions, {self.root: {"source": self.source}}, self.probe, None)

    def test_budget_five_roots_two_routes_both_types(self):
        b = qa.budget(5, 4)
        self.assertEqual(b["candidate_slots"] * 2, 80)
        self.assertEqual(b["http_attempts_max"] * 2, 80)
        self.assertEqual(b["solver_full_calls_max"] * 2, 1200)

    def test_current_per_type_call_budgets_and_unique_gold_blind_random_permutations(self):
        questions = [question(kind) for kind in qa.TYPES]
        requests, prompts, bindings, public, gold = self.plan(questions)
        self.assertEqual(len(requests), 30)
        for q, expected_variants, expected_slots in ((questions[0], 6, 3), (questions[1], 12, 1)):
            rows = [r for r in requests if r["candidate_id"] == q["candidate_id"]]
            self.assertEqual(len(rows), expected_variants * expected_slots)
            variants = {r["variant"]["variant_id"]: r["variant"] for r in rows}
            self.assertEqual(len(variants), expected_variants)
            self.assertEqual(len({tuple(v["display_order_content_ids"]) for v in variants.values()}), expected_variants)
            self.assertTrue(all(r["mode"] == "full_context" for r in rows))
            self.assertEqual({r["slot"] for r in rows}, set(range(expected_slots)))
        changed = deepcopy(questions)
        changed[0]["gold"] = ["B", "D"]
        changed[1]["gold"] = list(reversed(changed[1]["gold"]))
        changed_requests, changed_prompts, *_ = self.plan(changed)
        self.assertEqual(requests, changed_requests)
        self.assertEqual(prompts, changed_prompts)
        self.assertTrue(all("gold" not in q and "gold_content_ids" not in q for q in public))
        self.assertEqual(gold[questions[0]["candidate_id"]], bindings[0]["gold_content_ids"])

    def test_canonical_mapping_and_signed_partial_scores(self):
        questions = [question(kind) for kind in qa.TYPES]
        rows, _, _, _, gold = self.plan(questions)
        for q in questions:
            row = next(row for row in rows if row["candidate_id"] == q["candidate_id"])
            content_to_display = {content: label for label, content in row["variant"]["display_to_content"].items()}
            q = {**q, "gold_content_ids": gold[q["candidate_id"]]}
            canonical_prediction = q["gold_content_ids"]
            raw = "<box>" + ",".join(content_to_display[c] for c in canonical_prediction) + "</box>"
            self.assertEqual(qa.evaluator.score_response(q, row, raw)["partial_score_fraction"], "1")
            if q["question_type"] == "ordering":
                raw = "<box>" + ",".join(content_to_display[c] for c in reversed(canonical_prediction)) + "</box>"
                self.assertEqual(qa.evaluator.score_response(q, row, raw)["partial_score_fraction"], "-1")

    def test_selection_does_not_restore_ground_exact_or_range_gates(self):
        variants = [f"perm_{i:02}" for i in range(6)]
        scores = {v: [Fraction(0)] * 3 for v in variants}
        scores[variants[1]] = [Fraction(1)] * 3
        result = qa.evaluator.select(scores, variants, exact_rate=Fraction(0), no_context="diagnostic only")
        self.assertTrue(result["selected_by_policy"])
        self.assertEqual(result["permutation_score_range"], "1")
        self.assertFalse(result["ground_used"])
        self.assertFalse(qa.evaluator.select({v: [Fraction(1)] * 3 for v in variants}, variants)["selected_by_policy"])
        self.assertIsNone(qa.evaluator.select({v: [] for v in variants}, variants)["selected_by_policy"])

    def branch_fixture(self, directory, letter="A", count=1):
        frozen = self.source
        suffix = [{"turn_ref": f"R02:{letter}:S01:T001", "speaker": "A", "text": "New"}] if count else []
        turns = frozen["turns"] + suffix
        branch = directory / "roots" / self.root / letter
        write(directory / "sources" / f"{self.root}.json", frozen)
        write(branch / "cumulative_dialogue.json", {"root_id": self.root, "route": qa.ROUTES[letter],
            "branch_independent": True, "turns": turns})
        write(branch / "RESULT.json", {"root_id": self.root, "route": qa.ROUTES[letter],
            "added_turns": len(suffix), "accepted_scenes": count,
            "scene_metadata": [{"scene_id": "S01"}] if count else [],
            "output_history_sha256": qa.base.digest(turns)})

    def test_sources_keep_branches_independent_and_allow_expansion_shortfall(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.branch_fixture(folder)
            _, source, _, result = qa.source_for_branch(folder, self.root, "A")
            self.assertEqual(result["accepted_scenes"], 1)
            source["turns"][-1]["turn_ref"] = "R02:B:S01:T001"
            write(folder / "roots" / self.root / "A/cumulative_dialogue.json", source)
            with self.assertRaisesRegex(ValueError, "other_branch"):
                qa.source_for_branch(folder, self.root, "A")
            self.branch_fixture(folder, "B", count=0)
            _, _, _, result = qa.source_for_branch(folder, self.root, "B")
            self.assertEqual(result["accepted_scenes"], 0)

    def test_empty_branch_skips_both_generation_types(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            generation = run / "A/generation"
            write(generation / "MANIFEST.json", {"roots": [self.root], "route": qa.ROUTES["A"], "branch": "A",
                "files": [], "budget": qa.budget(1, 4), "candidate_slots_per_type_per_root": 4,
                "source_rows": [{"root_id": self.root, "accepted_scenes": 0}]})
            with patch.object(qa.legacy, "Caller") as caller:
                caller.return_value.counter = 0
                result = qa.generate_route(run, "A")
                caller.return_value.call.assert_not_called()
            self.assertEqual(len(result["roots"]), 2)
            self.assertEqual(result["candidate_slots"], 8)
            self.assertEqual(result["ready_for_solver"], 0)
            self.assertTrue(all(row["qa_status"] == "EXPANSION_SHORTFALL_NO_QA" for row in result["roots"]))

    def test_feedback_is_bound_to_actual_request_and_previous_model(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory)
            write(generation / "MANIFEST.json", {"solver_model": qa.TARGET})
            context = {"root_id": self.root, "candidate_ids": [], "source_turns": []}
            with patch.object(qa.legacy.feedback, "append_feedback_to_request", wraps=qa.legacy.feedback.append_feedback_to_request):
                with patch.object(qa.legacy.feedback, "build_feedback_suffix", return_value="\nFROZEN_PREVIOUS_FEEDBACK"):
                    write(generation / "feedback/source_lock.json", {"round": "previous"})
                    payload, receipt = qa.make_generation_payload(generation, self.root, "multi_select", context)
            self.assertTrue(payload["messages"][0]["content"].endswith("FROZEN_PREVIOUS_FEEDBACK"))
            self.assertEqual(receipt["target_solver_model"], qa.TARGET)
            self.assertTrue(receipt["feedback_appended"])
            self.assertIn("only this independent R02 branch", payload["messages"][0]["content"])

    def test_report_incomplete_answers_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows, _, _, public, _ = self.plan([question("ordering")])
            folder = run / "A/solver"
            write(folder / "manifest.json", {"policy_version": qa.POLICY, "model": "frozen-clean", "requests": rows})
            write(folder / "questions.json", {"candidates": public})
            result = qa.solver_report(run, "A")
            self.assertEqual(result["selected_questions"], [])
            self.assertEqual(result["questions"][0]["expected_full_calls"], 12)
            self.assertIsNone(result["questions"][0]["selected_by_policy"])
            self.assertTrue(result["call_set_errors"])


if __name__ == "__main__":
    unittest.main()
