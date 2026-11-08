from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import blind_request_control as control


EXPERIMENT = control.PROJECT_ROOT / "research-protocols/selfplay_week_20261019/base_qa_v2_2_1_acceptance_20261102"
FROZEN_FIXTURE = EXPERIMENT / "outputs/attempt_03/cc-train-episode-104814/frozen_review"
EVIDENCE_DIR = EXPERIMENT / "source_view_refresh_20261102/cc-train-episode-104814"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_keys(value):
    if type(value) is dict:
        for key, child in value.items():
            yield key
            yield from object_keys(child)
    elif type(value) is list:
        for child in value:
            yield from object_keys(child)


def synthetic_candidate(root_id: str) -> dict:
    return {
        "candidate_id": "offline-control-fixture-01",
        "author_id": "fixture-author",
        "root_id": root_id,
        "question_type": "single_choice",
        "question": "Which description matches the cited opening turn?",
        "options": {
            "opening": "Emily introduces an agenda for a team meeting.",
            "other": "Emily confirms a restaurant booking.",
        },
        "semantic_answer": {"kind": "scalar", "value": "opening"},
        "item_analysis": {
            "opening": {"semantic_value": "opening", "status": "supported"},
            "other": {"semantic_value": "other", "status": "distractor"},
        },
        "gold_answer": {"kind": "single_option", "option_id": "opening"},
    }


class BlindRequestControlTests(unittest.TestCase):
    def setUp(self):
        self.original_bundle = control.freeze.load_bundle(FROZEN_FIXTURE)
        self.temp = tempfile.TemporaryDirectory(prefix="blind-request-control-")
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "synthetic_bundle"
        control.freeze.freeze_bundle(
            self.run_dir,
            synthetic_candidate(self.original_bundle["candidate"]["root_id"]),
            self.original_bundle["source"],
            self.original_bundle["source_scope"],
            reviewer_id="fixture-review-assignment-01",
            context_id="fixture-fresh-context-01",
            issuer_id="fixture-coordinator",
            receipt_id="fixture-receipt-01",
        )
        self.packet_sha = digest(self.run_dir / control.freeze.PACKET)
        self.candidate_sha = digest(self.run_dir / "AUTHOR_CANDIDATE.json")
        self.freeze_sha = digest(self.run_dir / control.freeze.ENVELOPE)
        self.evidence_source_path = EVIDENCE_DIR / "EVIDENCE_SOURCE.json"
        self.evidence_scope_path = EVIDENCE_DIR / "EVIDENCE_SCOPE.json"
        self.request = control.freeze_control_envelope(
            self.run_dir, self.evidence_source_path, self.evidence_scope_path)
        self.bundle = control.freeze.load_bundle(self.run_dir)

    def test_control_request_is_bound_and_gold_free(self):
        self.assertEqual(control.verify_control_envelope(
            self.run_dir, self.evidence_source_path, self.evidence_scope_path), self.request["control"])
        request = json.loads((self.run_dir / control.REQUEST_FILE).read_bytes())
        self.assertEqual(len(request["source"]["turns"]), 82)
        self.assertEqual(
            [row["turn_ref"] for row in request["source"]["turns"]],
            self.bundle["source_scope"]["formal_qa_eligible_turn_refs"],
        )
        self.assertEqual(request["source_scope"]["included_turn_count"], 82)
        self.assertEqual(request["source_scope"]["excluded_context_turn_count"], 96)
        self.assertEqual(request["source_scope"]["included_turn_refs"],
                         self.bundle["source_scope"]["formal_qa_eligible_turn_refs"])
        self.assertEqual(request["control"]["reviewer_scope_payload_sha256"],
                         control.freeze.fingerprint(request["source_scope"]))
        request_keys = set(object_keys(request))
        self.assertTrue({"gold_answer", "semantic_answer", "item_analysis"}.isdisjoint(request_keys))
        self.assertNotIn("all_source_view_turns_formally_allowed", request_keys)
        self.assertFalse(request["author_gold_included"])
        self.assertFalse(request["author_certificate_included"])
        self.assertEqual(self.packet_sha, digest(self.run_dir / control.freeze.PACKET))
        self.assertEqual(self.candidate_sha, digest(self.run_dir / "AUTHOR_CANDIDATE.json"))
        self.assertEqual(self.freeze_sha, digest(self.run_dir / control.freeze.ENVELOPE))

    def _valid_response(self):
        packet = self.bundle["packet"]
        control_record = self.request["control"]
        mapping = self.bundle["render_contract"]["display_to_canonical"]
        label = next(label for label, option_id in mapping.items() if option_id == "opening")
        source_turn = self.request["source"]["turns"][0]
        return {
            "schema_version": control.freeze.BLIND_SCHEMA,
            "candidate_id": self.bundle["candidate"]["candidate_id"],
            "packet_sha256": packet["packet_sha256"],
            "source_scope_sha256": packet["source_scope_sha256"],
            "reviewer_id": control_record["reviewer_id"],
            "model_id": "offline_fixture_only",
            "reviewer_mode": "same_model_blind",
            "decision": "SOLVED",
            "independent_answer": {"kind": "single_option", "option_id": label},
            "evidence": [{"turn_ref": source_turn["turn_ref"], "speaker": source_turn["speaker"],
                          "quote": source_turn["text"]}],
            "reasoning_steps": ["The quoted turn states the agenda topic."],
            "ambiguities": [],
            "source_limitations": [],
            "isolation_receipt": control_record["isolation_receipt"],
        }

    def test_exact_echo_parses_valid_and_gold_comparison_stays_separate(self):
        raw = json.dumps(self._valid_response(), ensure_ascii=False).encode("utf-8")
        parsed = control.freeze.parse_blind_response(raw, self.bundle)
        self.assertEqual(parsed["status"], "VALID", parsed["errors"])
        self.assertEqual(parsed["response"]["reviewer_id"], self.request["control"]["reviewer_id"])
        # Parsing establishes an independent answer; it does not compare against author Gold.
        self.assertNotIn("author_gold", parsed)

    def test_missing_or_mismatched_control_fields_remain_invalid(self):
        missing_reviewer = self._valid_response()
        del missing_reviewer["reviewer_id"]
        parsed_missing = control.freeze.parse_blind_response(
            json.dumps(missing_reviewer).encode("utf-8"), self.bundle)
        self.assertEqual(parsed_missing["status"], "INVALID")
        self.assertIn("FIELD_MISSING:reviewer_id", parsed_missing["errors"])

        wrong_reviewer = self._valid_response()
        wrong_reviewer["reviewer_id"] = "runtime-agent-path-is-not-assignment-id"
        parsed_wrong = control.freeze.parse_blind_response(
            json.dumps(wrong_reviewer).encode("utf-8"), self.bundle)
        self.assertEqual(parsed_wrong["status"], "INVALID")
        self.assertIn("BINDING_MISMATCH:reviewer_id", parsed_wrong["errors"])

        wrong_receipt = self._valid_response()
        wrong_receipt["isolation_receipt"]["receipt_id"] = "invented"
        parsed_receipt = control.freeze.parse_blind_response(
            json.dumps(wrong_receipt).encode("utf-8"), self.bundle)
        self.assertEqual(parsed_receipt["status"], "INVALID")
        self.assertIn("ISOLATION_RECEIPT:RESPONSE_MISMATCH", parsed_receipt["errors"])

    def test_evidence_projection_hash_drift_is_rejected(self):
        tampered_path = Path(self.temp.name) / "tampered_evidence_source.json"
        tampered_path.write_bytes(self.evidence_source_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "EVIDENCE_SOURCE_BINDING_MISMATCH"):
            control.build_reviewer_request(self.run_dir, tampered_path, self.evidence_scope_path)


if __name__ == "__main__":
    unittest.main()
