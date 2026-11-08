from __future__ import annotations

import json
import unittest

from .run import (
    CRITERIA,
    ResponseValidationError,
    _balanced_assignments,
    paired_seed,
    validate_response,
)


def _valid_response() -> dict:
    criteria = {name: 4 for name in CRITERIA}
    return {
        "candidate_a": {"criteria": dict(criteria)},
        "candidate_b": {"criteria": dict(criteria)},
        "pairwise_preference": "TIE",
        "confidence": 0.8,
        "blocking_issues": [],
        "minor_issues": [],
        "evidence_turn_refs": ["A:T001"],
        "reason": "Both continuations are coherent.",
    }


class ContractTests(unittest.TestCase):
    def test_seed_is_stable_31_bit(self) -> None:
        seed = paired_seed("a" * 64, "b" * 64)
        self.assertEqual(seed, paired_seed("a" * 64, "b" * 64))
        self.assertTrue(0 <= seed < 2**31)

    def test_assignments_are_balanced(self) -> None:
        rows = [{"root_id": f"root-{index:02d}"} for index in range(70)]
        values = list(_balanced_assignments(rows).values())
        self.assertEqual(values.count("R01"), 35)
        self.assertEqual(values.count("R02"), 35)

    def test_strict_response_accepts_contract(self) -> None:
        parsed = validate_response(json.dumps(_valid_response()), {"A:T001"})
        self.assertEqual(parsed["confidence"], 0.8)

    def test_strict_response_rejects_extra_key(self) -> None:
        value = _valid_response()
        value["extra"] = True
        with self.assertRaisesRegex(ResponseValidationError, "top_level_schema"):
            validate_response(json.dumps(value), {"A:T001"})

    def test_strict_response_rejects_bad_score(self) -> None:
        value = _valid_response()
        value["candidate_a"]["criteria"]["qa_readiness"] = 6
        with self.assertRaisesRegex(ResponseValidationError, "criteria_range"):
            validate_response(json.dumps(value), {"A:T001"})


if __name__ == "__main__":
    unittest.main()
