from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import atomic_json
from scripts.current_method.restart_n5_spell.core import balanced_grounding_decision
from scripts.current_method.restart_n5_spell.minimal_admission import (
    SPELL_STYLE_MINIMAL_MODE,
)
from scripts.current_method.restart_n5_spell.questioner_contract import (
    build_span_catalog, span_catalog_sha256,
)
from scripts.current_method.restart_n5_spell.prompts import (
    SOLVER_POSITION_NEUTRAL_PARSE_RETRY_SUFFIX,
    SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.run_mcq_spell_filter import (
    SOLVER_MAX_MODEL_LEN, classify_spell_admission, preflight,
)


class V11MCQSPELLFilterTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        mcq_run = root / "mcq"
        qa_run = root / "qa"
        run_root = root / "new-filter"
        model = root / "model"
        tokenizer = root / "tokenizer"
        model.mkdir()
        tokenizer.mkdir()
        turns = [
            {"turn_id": "root:P01:T001", "session_id": "parent",
             "speaker": "A", "text": "old fact"},
            {"turn_id": "R01:S01:T001", "session_id": "child",
             "speaker": "A", "text": "new fact"},
        ]
        catalog = build_span_catalog(turns)
        qa = {
            "qa_id": "qa-1", "question": "What changed?",
            "options": {"A": "x", "B": "y", "C": "old then new", "D": "z"},
            "reference_answer_letter": "C",
            "reference_answer_text": "old then new",
            "canonical_answer_text": "old then new",
            "evidence_span_ids": [catalog["spans"][0]["span_id"],
                                  catalog["spans"][1]["span_id"]],
            "evidence_turn_ids": ["root:P01:T001", "R01:S01:T001"],
            "admission_mode": SPELL_STYLE_MINIMAL_MODE,
            "span_catalog_sha256": span_catalog_sha256(catalog),
        }
        atomic_json(mcq_run / "qa/final_mcq.json", qa)
        atomic_json(mcq_run / "reports/MCQ_CANARY_SUMMARY.json", {
            "status": "completed", "readiness": {"mcq_ready": True}})
        atomic_json(qa_run / "frozen/qa_source_view.json", {
            "combined_dialogue": turns})
        atomic_json(qa_run / "frozen/span_catalog.json", catalog)
        return mcq_run, qa_run, run_root, model, tokenizer

    def test_preflight_passes_without_creating_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            report = preflight(mcq_run=values[0], qa_source_run=values[1],
                               run_root=values[2], solver_model=values[3],
                               tokenizer_path=values[4])
            self.assertEqual("pass", report["decision"])
            self.assertFalse(values[2].exists())
            self.assertEqual(0, report["network_api_calls"])

    def test_preflight_rejects_non_spell_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            path = values[0] / "qa/final_mcq.json"
            qa = json.loads(path.read_text(encoding="utf-8"))
            qa["admission_mode"] = "simplified_solver_only"
            atomic_json(path, qa)
            report = preflight(mcq_run=values[0], qa_source_run=values[1],
                               run_root=values[2], solver_model=values[3],
                               tokenizer_path=values[4])
            self.assertEqual("fail", report["decision"])
            self.assertIn("spell_minimal_admission", report["failed_checks"])

    def test_original_solver_context_budget_is_frozen(self) -> None:
        self.assertEqual(16384, SOLVER_MAX_MODEL_LEN)

    def test_position_neutral_instructions_do_not_exemplify_one_answer(self) -> None:
        self.assertNotIn("<box>A</box>", SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT)
        self.assertNotIn("<box>A</box>", SOLVER_POSITION_NEUTRAL_PARSE_RETRY_SUFFIX)
        self.assertIn("no letter is preferred",
                      SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT.casefold())

    @staticmethod
    def _orders(answer_rows: list[list[str]]) -> list[dict[str, object]]:
        orders = []
        for index, gold in enumerate("ABCD"):
            answers = answer_rows[index]
            orders.append({"order_id":f"o{index}", "gold_letter":gold,
                           "valid_answer_letters":answers,
                           "valid_rewards":[int(value == gold) for value in answers]})
        return orders

    def test_balanced_grounding_follows_semantic_text(self) -> None:
        result = balanced_grounding_decision(self._orders([
            ["A"] * 3, ["B"] * 3, ["C"] * 3, ["D"] * 3]))
        self.assertEqual("bad_ground_semantic_leakage", result["decision"])

    def test_balanced_grounding_separates_position_bias(self) -> None:
        result = balanced_grounding_decision(self._orders([["A"] * 3] * 4))
        self.assertEqual("no_context_position_bias", result["decision"])

    def test_balanced_grounding_random_level_can_pass(self) -> None:
        result = balanced_grounding_decision(self._orders([
            ["A", "C", "D"], ["A", "C", "D"],
            ["A", "B", "D"], ["A", "B", "C"]]))
        self.assertEqual("pass_grounding", result["decision"])

    def test_marked_evidence_failure_is_nonblocking_in_mark_only_mode(self) -> None:
        result = classify_spell_admission(
            {"final_decision": "bad_evidence_relevance",
             "evidence_audit": {"decision": "bad_evidence_relevance"},
             "option_audit": {"decision": "pass_option_audit"}},
            evidence_alignment={"evidence_mismatch_marked": True},
            evidence_mismatch_policy="mark_only")
        self.assertTrue(result["spell_filter_passed"])
        self.assertTrue(result["evidence_mismatch_override_applied"])
        self.assertEqual("bad_evidence_relevance",
                         result["solver_raw_final_decision"])

    def test_mark_only_does_not_override_grounding_or_option_failures(self) -> None:
        alignment = {"evidence_mismatch_marked": True}
        for final, evidence, option in (
                ("bad_ground_semantic_leakage", "", ""),
                ("bad_option_bias", "pass_evidence_audit", "bad_option_bias"),
                ("frontier_audit_inconclusive", "evidence_audit_inconclusive",
                 "option_audit_inconclusive")):
            with self.subTest(final=final):
                result = classify_spell_admission(
                    {"final_decision": final,
                     "evidence_audit": {"decision": evidence},
                     "option_audit": {"decision": option}},
                    evidence_alignment=alignment,
                    evidence_mismatch_policy="mark_only")
                self.assertFalse(result["spell_filter_passed"])
                self.assertFalse(result["evidence_mismatch_override_applied"])


if __name__ == "__main__":
    unittest.main()
