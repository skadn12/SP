"""Zero-API tests for the K=3 probe's deterministic bookkeeping helpers."""

from __future__ import annotations

import unittest

from .diagnostics import aggregate_position, classify_candidate, context_diagnostic
from .permutation import audit_gold_rebinding, position_variants
from .seed_schedule import SEED_SCHEDULE_VERSION, seed_schedule, stable_seed
from .validator import validate_candidate


def _rollout(*, correct: bool, failure_type: str | None = None) -> dict[str, object]:
    # A reasoning error is a successfully scored semantic answer.  Only
    # transport/protocol failures are excluded from the accuracy denominator.
    scored = failure_type not in {"parse_failure", "runtime_error", "timeout", "OOM", "guided_decoding_error"}
    return {
        "scored": scored,
        "correct": correct,
        "failure_type": failure_type,
    }


class SeedScheduleTests(unittest.TestCase):
    def test_seed_is_stable_and_namespaced(self) -> None:
        first = stable_seed("root-1", "C-1", 0)
        self.assertEqual(first, stable_seed("root-1", "C-1", 0))
        self.assertNotEqual(first, stable_seed("root-1", "C-1", 1))
        self.assertNotEqual(first, stable_seed("root-2", "C-1", 0))

    def test_schedule_has_exactly_three_slots(self) -> None:
        rows = seed_schedule("root-1", "C-1", rollout_count=3)
        self.assertEqual([row["slot_id"] for row in rows], [0, 1, 2])
        self.assertEqual(len({row["seed"] for row in rows}), 3)
        self.assertTrue(
            all(row["paired_factors"] == ["context_mode", "display_position"] for row in rows)
        )
        self.assertEqual(SEED_SCHEDULE_VERSION, "h4_k3_paired_seed_v1")

    def test_context_and_position_are_not_seed_inputs(self) -> None:
        # The schedule API has no mode or display-position argument.  The two
        # requests below intentionally share a seed for the same candidate/slot.
        self.assertEqual(stable_seed("root-1", "C-1", 2), stable_seed("root-1", "C-1", 2))


class K3DenominatorTests(unittest.TestCase):
    def test_three_semantic_rollouts_classify_partial(self) -> None:
        summary = aggregate_position([
            _rollout(correct=True),
            _rollout(correct=False, failure_type="reasoning_error"),
            _rollout(correct=True),
        ])
        self.assertEqual(summary["requested_total"], 3)
        self.assertEqual(summary["attempted_total"], 3)
        self.assertEqual(summary["scored_total"], 3)
        self.assertEqual(summary["accuracy"], 2 / 3)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["nonunanimous"])
        self.assertEqual(summary["reasoning_error_count"], 1)
        self.assertEqual(summary["parse_failure_count"], 0)
        self.assertEqual(summary["runtime_failure_count"], 0)

    def test_parse_failure_is_not_zero_of_three(self) -> None:
        summary = aggregate_position([
            _rollout(correct=False, failure_type="parse_failure"),
            _rollout(correct=False, failure_type="reasoning_error"),
            _rollout(correct=True),
        ])
        self.assertEqual(summary["attempted_total"], 3)
        self.assertEqual(summary["scored_total"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["parse_failure_count"], 1)
        self.assertFalse(summary["complete"])
        # An incomplete position must not be promoted to a 0/3 failure.
        candidate = classify_candidate([{"variant": "shift_0", **summary}])
        self.assertEqual(candidate["category"], "diagnostic_incomplete")

    def test_runtime_failure_is_not_reasoning_error(self) -> None:
        summary = aggregate_position([
            _rollout(correct=False, failure_type="runtime_error"),
            _rollout(correct=False, failure_type="reasoning_error"),
            _rollout(correct=True),
        ])
        self.assertEqual(summary["scored_total"], 2)
        self.assertEqual(summary["runtime_failure_count"], 1)
        self.assertEqual(summary["reasoning_error_count"], 1)
        self.assertFalse(summary["complete"])

    def test_all_supported_technical_aliases_are_excluded(self) -> None:
        for failure_type in (
            "runtime_failure", "technical_failure", "timeout", "OOM",
            "guided_decoding_error", "backend_failure", "parse_error",
        ):
            summary = aggregate_position([
                _rollout(correct=False, failure_type=failure_type),
                _rollout(correct=True),
                _rollout(correct=True),
            ])
            self.assertEqual(summary["scored_total"], 2, failure_type)
            self.assertEqual(summary["correct"], 2, failure_type)
            self.assertEqual(summary["technical_failure_count"], 1, failure_type)
            self.assertFalse(summary["complete"], failure_type)

    def test_missing_attempt_is_incomplete(self) -> None:
        summary = aggregate_position([_rollout(correct=False), _rollout(correct=False)])
        self.assertEqual(summary["attempted_total"], 2)
        self.assertEqual(summary["scored_total"], 2)
        self.assertFalse(summary["complete"])

    def test_all_incorrect_requires_three_evaluable_rollouts(self) -> None:
        summary = aggregate_position([
            _rollout(correct=False, failure_type="reasoning_error"),
            _rollout(correct=False, failure_type="reasoning_error"),
            _rollout(correct=False, failure_type="reasoning_error"),
        ])
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["nonunanimous"])
        candidate = classify_candidate([
            {"variant": f"shift_{index}", **summary} for index in range(4)
        ])
        self.assertEqual(candidate["category"], "position_invariant_failure")


class ContextDiagnosticTests(unittest.TestCase):
    def test_context_uses_only_complete_paired_positions(self) -> None:
        no_rows = [
            {"variant": "shift_0", **aggregate_position([_rollout(correct=False)] * 3)},
            {"variant": "shift_1", **aggregate_position([_rollout(correct=False)] * 2)},
        ]
        full_rows = [
            {"variant": "shift_0", **aggregate_position([_rollout(correct=True)] * 3)},
            {"variant": "shift_1", **aggregate_position([_rollout(correct=True)] * 3)},
        ]
        result = context_diagnostic(no_rows, full_rows)
        self.assertEqual(result["paired_position_count"], 1)
        self.assertEqual(result["no_context"]["scored_total"], 5)
        self.assertEqual(result["full_context"]["scored_total"], 6)
        self.assertTrue(result["context_helpful"])


class GoldRebindingTests(unittest.TestCase):
    def test_single_gold_moves_with_option_text(self) -> None:
        candidate = {
            "question_type": "single_choice",
            "options": {"A": "wrong 1", "B": "right", "C": "wrong 2", "D": "wrong 3"},
            "correct_option": "B",
        }
        variants = position_variants(candidate)
        self.assertEqual(len(variants), 4)
        self.assertEqual(
            [variant["options"][variant["gold"][0]] for variant in variants],
            ["right"] * 4,
        )
        self.assertEqual(audit_gold_rebinding(candidate, variants)["gold_rebinding_errors"], 0)

    def test_multi_gold_moves_with_option_text(self) -> None:
        candidate = {
            "question_type": "multi_select",
            "options": {"A": "yes 1", "B": "no", "C": "yes 2", "D": "no 2", "E": "no 3"},
            "gold": ["A", "C"],
        }
        variants = position_variants(candidate)
        self.assertEqual(len(variants), 5)
        for variant in variants:
            self.assertEqual(
                {variant["options"][key] for key in variant["gold"]},
                {"yes 1", "yes 2"},
            )
        self.assertEqual(audit_gold_rebinding(candidate, variants)["gold_rebinding_errors"], 0)

    def test_ordering_gold_ids_stay_stable(self) -> None:
        candidate = {
            "question_type": "ordering",
            "items": {"I1": "one", "I2": "two", "I3": "three", "I4": "four"},
            "gold": ["I2", "I4", "I1", "I3"],
        }
        variants = position_variants(candidate)
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(variant["gold"] == candidate["gold"] for variant in variants))
        self.assertEqual(audit_gold_rebinding(candidate, variants)["gold_rebinding_errors"], 0)

    def test_dynamic_multi_and_ordering_arity(self) -> None:
        for count in (4, 5, 6):
            letters = [chr(ord("A") + index) for index in range(count)]
            multi = {
                "question_type": "multi_select",
                "options": {key: f"option-{key}" for key in letters},
                "gold": list(letters),
            }
            ordering_keys = [f"I{index + 1}" for index in range(count)]
            ordering = {
                "question_type": "ordering",
                "items": {key: f"item-{key}" for key in ordering_keys},
                "gold": list(reversed(ordering_keys)),
            }
            for candidate in (multi, ordering):
                variants = position_variants(candidate)
                self.assertEqual(len(variants), count)
                self.assertTrue(audit_gold_rebinding(candidate, variants)["pass"])


class CandidateValidatorTests(unittest.TestCase):
    def test_dynamic_contract_and_evidence_uniqueness(self) -> None:
        for count in (4, 5, 6):
            keys = [chr(ord("A") + index) for index in range(count)]
            candidate = {
                "candidate_id": f"candidate-{count}",
                "question_type": "multi_select",
                "question": "Which options apply?",
                "options": {key: f"option-{key}" for key in keys},
                "gold": list(keys),
                "evidence_refs": ["R03:S01:T001"],
            }
            result = validate_candidate(candidate, valid_turn_refs={"R03:S01:T001"})
            self.assertEqual(result["decision"], "pass")
        candidate["evidence_refs"] = ["R03:S01:T001", "R03:S01:T001"]
        result = validate_candidate(candidate, valid_turn_refs={"R03:S01:T001"})
        self.assertEqual(result["decision"], "reject")
        self.assertIn("duplicate_evidence_refs", result["blocking_errors"])

    def test_duplicate_option_text_fails_closed(self) -> None:
        candidate = {
            "question_type": "single_choice",
            "options": {"A": "same", "B": "same", "C": "other", "D": "third"},
            "correct_option": "A",
        }
        with self.assertRaisesRegex(ValueError, "duplicate_option_text"):
            position_variants(candidate)

    def test_audit_rejects_tampered_rebound_gold(self) -> None:
        candidate = {
            "question_type": "single_choice",
            "options": {"A": "wrong 1", "B": "right", "C": "wrong 2", "D": "wrong 3"},
            "correct_option": "B",
        }
        variants = position_variants(candidate)
        variants[0] = {**variants[0], "gold": ["A"]}
        audit = audit_gold_rebinding(candidate, variants)
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["gold_rebinding_errors"], 1)


if __name__ == "__main__":
    unittest.main()
