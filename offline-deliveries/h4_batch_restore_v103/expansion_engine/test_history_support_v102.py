"""History-evidence gate boundaries, using immutable v101 raw claim/history.

Checker verdicts constructed here are unit-test stubs, never model results,
accepted training data, or proof of semantic reliability. In particular a
literal valid quote cannot make the validator an entailment oracle.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import history_support as support


V101 = Path(__file__).resolve().parents[2] / "h4_review_wrapper_correction_v101"
INPUT = V101 / "regression/INPUT.json"
RAW_REVIEW = V101 / "regression/requests/002-cc-train-episode-19053-B-S01-event-state-1-1/RESPONSE.json"
REVIEW = V101 / "regression/POST_REPLAN_REVIEW.json"


def read(path):
    return json.loads(path.read_text())


def digest(claim):
    value = {key: value for key, value in claim.items() if key != "claim_sha256"}
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class StubCaller:
    """Raw API boundary only; no network or production semantic oracle."""
    def __init__(self, directory, responses):
        self.directory, self.responses, self.calls = Path(directory), list(responses), []

    def call(self, label, payload, proof):
        if proof["role"] != "history_support_checker":
            raise AssertionError("Only history-support checker calls are allowed")
        directory = self.directory / "requests" / (f"{len(self.calls) + 1:03d}-" + label)
        directory.mkdir(parents=True)
        self.calls.append({"payload": deepcopy(payload), "proof": deepcopy(proof), "directory": directory})
        return self.responses.pop(0), directory


class HistorySupportV102Tests(unittest.TestCase):
    def setUp(self):
        self.paths = [INPUT, RAW_REVIEW, REVIEW]
        self.hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths}
        source = read(INPUT)
        self.root = source["context"]["root_id"]
        self.history = source["context"]["cumulative_history"]
        self.raw_review = json.loads(read(RAW_REVIEW)["text"])
        self.assertEqual(self.raw_review, read(REVIEW))
        self.claims = []
        for row in self.raw_review["timeline"]:
            claim = {key: deepcopy(row[key]) for key in ("statement", "status", "evidence_basis", "evidence")}
            claim["claim_id"] = row["constraint_id"]
            if "uncertainty" in row:
                claim["uncertainty"] = deepcopy(row["uncertainty"])
            claim["claim_sha256"] = digest(claim)
            self.claims.append(claim)

    def tearDown(self):
        self.assertEqual(self.hashes, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in self.paths})

    def response(self, verdicts=None, claims=None):
        claims = self.claims if claims is None else claims
        verdicts = ["supported"] * len(claims) if verdicts is None else verdicts
        return {"root_id": self.root, "checks": [
            {"claim_id": row["claim_id"], "claim_sha256": row["claim_sha256"], "verdict": verdict,
             "supported_meaning": "The cited speakers state the words shown in the exact evidence.",
             "reason": "Unit-test stub; the application must still assess the actual meaning.",
             "evidence": deepcopy(row["evidence"])}
            for row, verdict in zip(claims, verdicts)
        ]}

    def validate(self, value, *, claims=None, history=None):
        return support.validate(value, self.root, self.history if history is None else history,
                                self.claims if claims is None else claims)

    def assert_invalid(self, value, *, claims=None, history=None):
        with self.assertRaises(support.HistorySupportValidationError) as caught:
            self.validate(value, claims=claims, history=history)
        self.assertTrue(caught.exception.errors)
        return caught.exception

    def repair(self, previous):
        error = self.assert_invalid(previous)
        return support.build_repair(error, previous, self.root, self.history, self.claims)

    def progress(self, current, repair):
        return support.correction_progress(current, repair, self.root, self.history, self.claims)

    def test_real_c3_quote_exists_but_unsupported_verdict_blocks_autoaccept(self):
        c3 = self.claims[2]
        self.assertEqual(c3["claim_id"], "c3")
        self.assertEqual(c3["statement"], "Emily and James discussed the cooking event during their conversation.")
        turn = next(row for row in self.history if row["turn_ref"] == "R01:S01:T011")
        self.assertIn(c3["evidence"][0]["quote"], turn["text"])
        value = self.validate(self.response(["supported", "supported", "unsupported"]))
        assessment = support.assess(value)
        self.assertFalse(assessment["all_supported"])
        self.assertIn("c3", assessment["blocking_claim_ids"])
        self.assertFalse(assessment["semantic_certified"])

    def test_real_quote_and_supported_label_never_certify_semantic_truth(self):
        # A model could make this very semantic mistake. Mechanical validation
        # must not be misreported as proving that the snacks quote entails c3.
        value = self.validate(self.response())
        assessment = support.assess(value)
        self.assertTrue(assessment["all_supported"])
        self.assertFalse(assessment["semantic_certified"])

    def test_uncertain_is_blocking_and_not_silently_supported(self):
        assessment = support.assess(self.validate(self.response(["uncertain", "supported", "unsupported"])))
        self.assertFalse(assessment["all_supported"])
        self.assertEqual(set(assessment["blocking_claim_ids"]), {"c1", "c3"})
        self.assertFalse(assessment["semantic_certified"])

    def test_no_claims_does_not_invent_history_obligation_for_new_event(self):
        assessment = support.assess(self.validate({"root_id": self.root, "checks": []}, claims=[]))
        self.assertTrue(assessment["all_supported"])
        self.assertEqual(assessment["blocking_claim_ids"], [])
        self.assertFalse(assessment["semantic_certified"])
        self.assert_invalid(self.response(), claims=[])

    def test_request_contains_complete_history_and_claims_without_candidate_or_expected(self):
        before = deepcopy((self.history, self.claims))
        request = support.build_request(self.root, self.history, self.claims)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(request["temperature"], .25)
        self.assertFalse(request["enable_thinking"])
        text = request["messages"][0]["content"]
        payload, _ = json.JSONDecoder().raw_decode(text.split("HISTORY_SUPPORT_INPUT_JSON\n", 1)[1])
        self.assertEqual(set(payload), {"root_id", "cumulative_history", "claims", "output_contract"})
        self.assertEqual(payload["cumulative_history"], self.history)
        self.assertEqual(len(payload["cumulative_history"]), 80)
        self.assertEqual(payload["claims"], self.claims)
        self.assertEqual((self.history, self.claims), before)

    def test_claim_hash_is_canonical_and_binds_each_substantive_field(self):
        claim = self.claims[2]
        self.assertEqual(support.claim_digest(claim), digest(claim))
        reordered = dict(reversed(list(claim.items())))
        self.assertEqual(support.claim_digest(reordered), digest(claim))
        for field, replacement in [("claim_id", "other"), ("statement", "New claim."),
                                   ("status", "planned"), ("evidence_basis", "inferred"),
                                   ("evidence", [deepcopy(self.claims[0]["evidence"][0])])]:
            with self.subTest(field=field):
                changed = deepcopy(claim)
                changed[field] = replacement
                self.assertNotEqual(digest(changed), digest(claim))
                changed_claims = deepcopy(self.claims)
                changed_claims[2] = changed
                self.assert_invalid(self.response(), claims=changed_claims)

    def test_claim_inputs_cannot_carry_plan_card_or_expected_gold(self):
        for name in ["plan", "scene_plan", "card", "expected", "expected_verdict"]:
            with self.subTest(field=name):
                claims = deepcopy(self.claims)
                claims[2][name] = "supported"
                claims[2]["claim_sha256"] = digest(claims[2])
                with self.assertRaises(support.HistorySupportValidationError):
                    support.build_request(self.root, self.history, claims)

    def test_input_claim_id_coverage_and_hash_are_required(self):
        for mode in ["duplicate_id", "missing_hash", "wrong_hash", "extra_field"]:
            with self.subTest(mode=mode):
                claims = deepcopy(self.claims)
                if mode == "duplicate_id":
                    claims[2]["claim_id"] = claims[1]["claim_id"]
                    claims[2]["claim_sha256"] = digest(claims[2])
                elif mode == "missing_hash": del claims[2]["claim_sha256"]
                elif mode == "wrong_hash": claims[2]["claim_sha256"] = "0" * 64
                else: claims[2]["foreign"] = True
                self.assert_invalid(self.response(), claims=claims)

    def test_selected_input_citations_must_exist_exactly_in_unique_history(self):
        for mode in ["missing_turn", "inexact_quote", "empty_quote"]:
            with self.subTest(mode=mode):
                claims = deepcopy(self.claims)
                quote = claims[2]["evidence"][0]
                if mode == "missing_turn": quote["turn_ref"] = "other-root:T011"
                elif mode == "inexact_quote": quote["quote"] += " Not actually said."
                else: quote["quote"] = ""
                claims[2]["claim_sha256"] = digest(claims[2])
                self.assert_invalid(self.response(), claims=claims)
        history = deepcopy(self.history)
        history.append(deepcopy(history[-1]))
        self.assert_invalid(self.response(), history=history)

    def test_each_claim_checked_once_by_identity_with_reordering_allowed(self):
        value = self.response(["unsupported", "supported", "uncertain"])
        value["checks"].reverse()
        self.validate(value)
        for mode in ["dropped", "duplicate", "foreign"]:
            with self.subTest(mode=mode):
                changed = self.response()
                if mode == "dropped": changed["checks"].pop()
                elif mode == "duplicate": changed["checks"].append(deepcopy(changed["checks"][0]))
                else: changed["checks"][2]["claim_id"] = "not-an-input-claim"
                self.assert_invalid(changed)

    def test_response_root_hash_required_fields_and_extra_fields_fail_closed(self):
        for mode in ["wrong_root", "wrong_hash", "missing_meaning", "missing_reason", "extra_expected", "illegal_verdict"]:
            with self.subTest(mode=mode):
                changed = self.response()
                row = changed["checks"][2]
                if mode == "wrong_root": changed["root_id"] = "foreign-root"
                elif mode == "wrong_hash": row["claim_sha256"] = "f" * 64
                elif mode == "missing_meaning": del row["supported_meaning"]
                elif mode == "missing_reason": del row["reason"]
                elif mode == "extra_expected": row["expected"] = "supported"
                else: row["verdict"] = "compatible"
                self.assert_invalid(changed)

    def test_duplicate_json_keys_nested_duplicates_and_nonfinite_json_rejected(self):
        raw = json.dumps(self.response())
        top = raw.replace('"root_id":', '"root_id":"ignored", "root_id":', 1)
        nested = raw.replace('"verdict": "supported"', '"verdict":"unsupported", "verdict":"supported"', 1)
        for bad in [top, nested, raw.replace('"supported_meaning":', '"supported_meaning":NaN, "foreign":', 1),
                    "```json\n" + raw + "\n```", raw + " trailing"]:
            with self.subTest(raw=bad[:100]): self.assert_invalid(bad)

    def test_supported_evidence_must_retain_selected_meaning_not_other_true_snippet(self):
        for mode in ["other_turn", "shorter_same_turn", "unrelated_same_turn", "empty"]:
            with self.subTest(mode=mode):
                changed = self.response()
                if mode == "other_turn": changed["checks"][2]["evidence"] = [deepcopy(self.claims[0]["evidence"][0])]
                elif mode == "shorter_same_turn": changed["checks"][2]["evidence"][0]["quote"] = "bring some snacks"
                elif mode == "unrelated_same_turn": changed["checks"][2]["evidence"][0]["quote"] = "It’s been a while since we had a break like that."
                else: changed["checks"][2]["evidence"] = []
                self.assert_invalid(changed)
        changed = self.response()
        turn = next(row for row in self.history if row["turn_ref"] == "R01:S01:T011")
        changed["checks"][2]["evidence"][0]["quote"] = turn["text"]
        self.validate(changed)  # Adding exact surrounding context remains allowed.

    def test_negative_checker_can_cite_other_history_to_explain_non_support(self):
        changed = self.response(["supported", "supported", "unsupported"])
        changed["checks"][2]["evidence"] = [deepcopy(self.claims[0]["evidence"][0])]
        result = self.validate(changed)
        self.assertFalse(support.assess(result)["all_supported"])

    def test_evidence_inexact_quote_foreign_reference_and_too_many_fail(self):
        for mode in ["inexact", "foreign", "too_many", "duplicate"]:
            with self.subTest(mode=mode):
                changed = self.response(["supported", "supported", "unsupported"])
                row = changed["checks"][2]
                if mode == "inexact": row["evidence"][0]["quote"] += " Changed."
                elif mode == "foreign": row["evidence"][0]["turn_ref"] = "other-root:T011"
                elif mode == "too_many": row["evidence"] = [deepcopy(x) for x in self.claims[0]["evidence"] + self.claims[1]["evidence"]]
                else: row["evidence"].append(deepcopy(row["evidence"][0]))
                self.assert_invalid(changed)

    def test_format_retry_can_fix_reason_without_erasing_negative_or_input(self):
        previous = self.response(["supported", "supported", "unsupported"])
        del previous["checks"][2]["reason"]
        repair = self.repair(previous)
        current = self.response(["supported", "supported", "unsupported"])
        self.validate(current)
        self.assertEqual(self.progress(current, repair)["status"], "PASS")
        request = support.build_request(self.root, self.history, self.claims, repair=repair)
        text = request["messages"][0]["content"]
        self.assertIn("FORMAT_REPAIR", text)
        self.assertIn("unsupported", text)
        self.assertEqual(self.claims[2]["statement"], self.raw_review["timeline"][2]["statement"])

    def test_format_retry_never_flips_unsupported_or_uncertain_to_supported(self):
        for verdict in ["unsupported", "uncertain"]:
            with self.subTest(verdict=verdict):
                previous = self.response(["supported", "supported", verdict])
                del previous["checks"][2]["reason"]
                repair = self.repair(previous)
                current = self.response()
                self.validate(current)
                self.assertEqual(self.progress(current, repair)["status"], "REJECT")

    def test_retry_cannot_remove_claim_evidence_or_forge_empty_guard_contract(self):
        previous = self.response(["supported", "supported", "unsupported"])
        previous["checks"][2]["evidence"] = deepcopy(self.claims[0]["evidence"])
        del previous["checks"][2]["reason"]
        repair = self.repair(previous)
        for mode in ["drop_claim", "drop_evidence", "shrink_evidence", "forged_contract"]:
            with self.subTest(mode=mode):
                current = deepcopy(previous)
                current["checks"][2]["reason"] = "Fixed the missing required reason."
                local_repair = deepcopy(repair)
                if mode == "drop_claim": current["checks"].pop()
                elif mode == "drop_evidence": current["checks"][2]["evidence"] = []
                elif mode == "shrink_evidence": current["checks"][2]["evidence"].pop()
                else:
                    local_repair["correction_contract"] = {"protected_judgments": [], "evidence_obligations": []}
                    current["checks"][2]["verdict"] = "supported"
                    current["checks"][2]["evidence"] = deepcopy(self.claims[2]["evidence"])
                self.assertEqual(self.progress(current, local_repair)["status"], "REJECT")

    def test_run_check_valid_negative_stops_without_spending_format_retry(self):
        raw = json.dumps(self.response(["supported", "supported", "unsupported"]))
        with tempfile.TemporaryDirectory() as directory:
            caller = StubCaller(directory, [raw])
            outcome = support.run_check(caller, Path(directory), self.root, self.history, self.claims)
            self.assertEqual(len(caller.calls), 1)
            self.assertEqual(len(outcome["attempts"]), 1)
            self.assertIsNotNone(outcome["result"])
            self.assertFalse(outcome["assessment"]["all_supported"])
            self.assertFalse(outcome["assessment"]["semantic_certified"])
            receipt = read(caller.calls[0]["directory"] / "VALIDATION.json")
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(caller.calls[0]["payload"]["max_tokens"], 4096)

    def test_run_check_format_retry_cannot_turn_failure_into_supported(self):
        first = self.response(["supported", "supported", "unsupported"])
        del first["checks"][2]["reason"]
        raw = [json.dumps(first), json.dumps(self.response())]
        with tempfile.TemporaryDirectory() as directory:
            caller = StubCaller(directory, raw)
            outcome = support.run_check(caller, Path(directory), self.root, self.history, self.claims)
            self.assertEqual(len(caller.calls), 2)
            self.assertEqual(caller.responses, [])
            self.assertIsNone(outcome["result"])
            receipt = read(caller.calls[1]["directory"] / "CORRECTION_PROGRESS.json")
            self.assertEqual(receipt["status"], "REJECT")
            self.assertIn("FORMAT_REPAIR", caller.calls[1]["payload"]["messages"][0]["content"])

    def test_run_check_empty_claims_does_not_call_provider_or_certify_meaning(self):
        with tempfile.TemporaryDirectory() as directory:
            caller = StubCaller(directory, [])
            outcome = support.run_check(caller, Path(directory), self.root, self.history, [])
            self.assertEqual(caller.calls, [])
            self.assertEqual(outcome["attempts"], [])
            self.assertTrue(outcome["assessment"]["all_supported"])
            self.assertFalse(outcome["assessment"]["semantic_certified"])


if __name__ == "__main__":
    unittest.main()
