"""Probe contract checks with fictional fixtures; no semantic scoring or API."""

from copy import deepcopy
import json
import unittest

import probe_contract as contract


ROOT = "fixture-root"
HISTORY = [
    {"turn_ref": "fixture-root:P:T001", "speaker": "Alex", "text": "I may book the room tomorrow."},
    {"turn_ref": "R01:S01:T002", "speaker": "Casey", "text": "The room is available after noon."},
    {"turn_ref": "R01:S02:T003", "speaker": "Alex", "text": "I have booked the room for two o'clock."},
]
CLAIMS = [{"claim_id": f"claim_{index}", "statement": f"Candidate statement {index}.",
           "claimed_status": "completed"} for index in range(1, 6)]


def output(verdict="supported"):
    return {"root_id": ROOT, "checks": [{
        "claim_id": claim["claim_id"], "verdict": verdict,
        "atoms": [{"statement": claim["statement"], "verdict": verdict,
                   "evidence": [{"turn_ref": HISTORY[-1]["turn_ref"], "speaker": "Alex",
                                 "quote": "booked the room"}],
                   "reason": "Fixture evidence reason."}],
        "uncertainty": "", "reason": "Fixture check reason.",
    } for claim in CLAIMS]}


def context(payload):
    return json.loads(payload["messages"][0]["content"].split(contract.INPUT_MARKER, 1)[1])


class ProbeContractTests(unittest.TestCase):
    def assert_rejected(self, raw, code):
        with self.assertRaises(contract.ProbeValidationError) as caught:
            contract.validate_response(raw, ROOT, HISTORY, CLAIMS)
        self.assertIn(code, [error["code"] for error in caught.exception.errors])

    def test_two_conditions_differ_only_in_prior_summary(self):
        summary = {"facts": [{"statement": "A prior candidate description."}]}
        bare = contract.build_request(ROOT, HISTORY, CLAIMS)
        summarized = contract.build_request(ROOT, HISTORY, CLAIMS, summary)
        bare_context, summary_context = context(bare), context(summarized)
        self.assertIsNone(bare_context.pop("prior_summary"))
        self.assertEqual(summary_context.pop("prior_summary"), summary)
        self.assertEqual(bare_context, summary_context)
        for payload in (bare, summarized):
            payload["messages"][0]["content"] = contract.PROMPT + contract.INPUT_MARKER + json.dumps(bare_context)
        self.assertEqual(bare, summarized)
        self.assertNotIn("condition", bare_context)
        self.assertNotIn("replica", bare_context)
        self.assertEqual(bare["model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(bare["temperature"], .25)
        self.assertEqual(bare["max_tokens"], 4096)
        self.assertFalse(bare["enable_thinking"])
        self.assertEqual(bare["response_format"], {"type": "json_object"})

    def test_request_preserves_all_original_values_and_mixed_reference_namespaces(self):
        history, claims, summary = deepcopy(HISTORY), deepcopy(CLAIMS), {"facts": ["Unverified claim."]}
        before = deepcopy((history, claims, summary))
        request = contract.build_request(ROOT, history, claims, summary)
        self.assertEqual((history, claims, summary), before)
        ctx = context(request)
        self.assertEqual(ctx["history"], history)
        self.assertEqual(ctx["claims"], claims)
        self.assertEqual(ctx["prior_summary"], summary)
        self.assertEqual(ctx["history_reference_index"], [turn["turn_ref"] for turn in HISTORY])
        self.assertEqual(ctx["history_reference_examples"], [HISTORY[0]["turn_ref"], HISTORY[1]["turn_ref"]])
        self.assertTrue(all(set(claim) == {"claim_id", "statement", "claimed_status"} for claim in ctx["claims"]))

    def test_positive_and_not_established_verdicts_both_validate(self):
        for verdict in ("supported", "contradicted", "not_established"):
            with self.subTest(verdict=verdict):
                raw = output(verdict)
                self.assertEqual(contract.validate_response(raw, ROOT, HISTORY, CLAIMS), raw)

    def test_not_established_can_have_no_citations_with_nonempty_reason(self):
        raw = output("not_established")
        for check in raw["checks"]:
            check["atoms"][0]["evidence"] = []
        self.assertEqual(contract.validate_response(raw, ROOT, HISTORY, CLAIMS), raw)
        raw["checks"][0]["atoms"][0]["reason"] = ""
        self.assert_rejected(raw, "probe_text_required")

    def test_supported_and_contradicted_require_at_least_one_citation(self):
        for verdict in ("supported", "contradicted"):
            raw = output(verdict)
            raw["checks"][0]["atoms"][0]["evidence"] = []
            self.assert_rejected(raw, "probe_evidence_count_invalid")

    def test_citation_speaker_must_match_the_original_history_turn(self):
        raw = output()
        raw["checks"][0]["atoms"][0]["evidence"][0]["speaker"] = "Casey"
        self.assert_rejected(raw, "probe_evidence_speaker_invalid")
        raw["checks"][0]["atoms"][0]["evidence"][0].pop("speaker")
        self.assert_rejected(raw, "probe_fields_invalid")

    def test_validation_preserves_original_raw_history_and_claims(self):
        raw, history, claims = output(), deepcopy(HISTORY), deepcopy(CLAIMS)
        before = deepcopy((raw, history, claims))
        checked = contract.validate_response(raw, ROOT, history, claims)
        self.assertEqual((raw, history, claims), before)
        checked["checks"][0]["atoms"][0]["evidence"][0]["quote"] = "Changed only in returned copy"
        self.assertEqual((raw, history, claims), before)

    def test_json_text_parses_without_changing_model_values(self):
        raw = output("not_established")
        self.assertEqual(contract.validate_response(json.dumps(raw), ROOT, HISTORY, CLAIMS), raw)

    def test_aggregate_verdict_is_not_mechanically_inferred_from_atom_verdicts(self):
        raw = output()
        raw["checks"][0]["verdict"] = "mixed"
        raw["checks"][1]["verdict"] = "not_established"
        self.assertEqual(contract.validate_response(raw, ROOT, HISTORY, CLAIMS), raw)

    def test_wrong_reference_rejected_without_normalizing_or_changing_raw(self):
        raw = output()
        raw["checks"][0]["atoms"][0]["evidence"][0]["turn_ref"] = ROOT + ":" + HISTORY[-1]["turn_ref"]
        before = deepcopy(raw)
        self.assert_rejected(raw, "probe_evidence_reference_invalid")
        self.assertEqual(raw, before)

    def test_wrong_quote_and_quote_from_another_turn_rejected(self):
        for quote in ("Booked the room", HISTORY[0]["text"], "", "   "):
            with self.subTest(quote=quote):
                raw = output()
                raw["checks"][0]["atoms"][0]["evidence"][0]["quote"] = quote
                self.assert_rejected(raw, "probe_evidence_quote_not_exact" if quote.strip()
                                     else "probe_evidence_quote_invalid")

    def test_missing_unknown_and_duplicate_claims_rejected(self):
        raw = output()
        raw["checks"].pop()
        self.assert_rejected(raw, "probe_claims_missing")
        raw = output()
        raw["checks"][0]["claim_id"] = "unknown"
        self.assert_rejected(raw, "probe_claim_id_invalid")
        raw = output()
        raw["checks"][0]["claim_id"] = raw["checks"][1]["claim_id"]
        self.assert_rejected(raw, "probe_claim_id_duplicate")

    def test_extra_fields_at_each_schema_level_rejected(self):
        for level in ("response", "check", "atom", "citation"):
            with self.subTest(level=level):
                raw = output()
                target = {"response": raw, "check": raw["checks"][0],
                          "atom": raw["checks"][0]["atoms"][0],
                          "citation": raw["checks"][0]["atoms"][0]["evidence"][0]}[level]
                target["unexpected"] = True
                self.assert_rejected(raw, "probe_fields_invalid")

    def test_invalid_counts_and_atom_mixed_verdict_rejected(self):
        for atoms in ([], [output()["checks"][0]["atoms"][0]] * 5):
            raw = output()
            raw["checks"][0]["atoms"] = atoms
            self.assert_rejected(raw, "probe_atoms_count_invalid")
        for citations in ([], output()["checks"][0]["atoms"][0]["evidence"] * 4):
            raw = output()
            raw["checks"][0]["atoms"][0]["evidence"] = citations
            self.assert_rejected(raw, "probe_evidence_count_invalid")
        raw = output()
        raw["checks"][0]["atoms"][0]["verdict"] = "mixed"
        self.assert_rejected(raw, "probe_atom_verdict_invalid")

    def test_duplicate_json_keys_rejected_at_nested_and_top_levels(self):
        raw = json.dumps(output())
        self.assert_rejected(raw.replace('"root_id":', '"root_id":"duplicate", "root_id":', 1),
                             "probe_duplicate_json_key")
        self.assert_rejected(raw.replace('"verdict":', '"verdict":"duplicate", "verdict":', 1),
                             "probe_duplicate_json_key")

    def test_malformed_json_wrong_identity_and_missing_reasons_rejected(self):
        self.assert_rejected("{} trailing", "probe_response_invalid_json")
        self.assert_rejected("[]", "probe_response_not_object")
        raw = output()
        raw["root_id"] = "different-root"
        self.assert_rejected(raw, "probe_root_mismatch")
        raw = output()
        raw["checks"][0]["reason"] = ""
        self.assert_rejected(raw, "probe_reason_required")

    def test_request_rejects_wrong_claim_count_unexpected_fields_and_duplicate_ids(self):
        for claims in (CLAIMS[:4], CLAIMS + CLAIMS[:1],
                       [{**CLAIMS[0], "expected": "supported"}] + CLAIMS[1:],
                       [CLAIMS[0]] + CLAIMS[:4]):
            with self.subTest(claims=claims), self.assertRaises(ValueError):
                contract.build_request(ROOT, HISTORY, claims)

    def test_duplicate_history_refs_rejected(self):
        with self.assertRaisesRegex(ValueError, "probe_history_duplicate_reference"):
            contract.build_request(ROOT, HISTORY + HISTORY[:1], CLAIMS)


if __name__ == "__main__":
    unittest.main()
