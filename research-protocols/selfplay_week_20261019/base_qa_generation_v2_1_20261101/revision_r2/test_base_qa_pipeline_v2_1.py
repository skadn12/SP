from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import base_qa_pipeline_v1 as v1
    from . import base_qa_pipeline_v2_1 as qa
except ImportError:
    import base_qa_pipeline_v1 as v1
    import base_qa_pipeline_v2_1 as qa


class V21BlindProtocolTests(unittest.TestCase):
    def setUp(self):
        self.source = {"root_id": "root-a", "source_id": "source-a", "turns": [
            {"turn_ref": "T1", "speaker": "A", "text": "The first event was moved to Tuesday."},
            {"turn_ref": "T2", "speaker": "B", "text": "The new time is 14:00."},
            {"turn_ref": "T3", "speaker": "C", "text": "An unapproved note says 15:00."},
        ]}
        self.source_scope = {
            "source_id": "source-a",
            "source_sha256": qa._source_sha256(self.source, None),
            "source_admission_complete": True,
            "turns": [
                {"turn_id": "T1", "formal_qa_allowed": True},
                {"turn_id": "T2", "formal_qa_allowed": True},
                {"turn_id": "T3", "formal_qa_allowed": False},
            ],
        }
        self.scope_hash = qa.fingerprint(self.source_scope)
        self.candidate = {
            "candidate_id": "q1", "author_id": "challenger", "question_type": "single_choice",
            "options": {"o1": "Tuesday at 14:00", "o2": "Monday at 14:00"}, "gold": ["o1"],
        }
        self.packet_hash = "a" * 64
        self.response = {
            "schema_version": qa.BLIND_SCHEMA,
            "candidate_id": "q1",
            "packet_sha256": self.packet_hash,
            "source_scope_sha256": self.scope_hash,
            "reviewer_id": "isolated-reviewer",
            "model_id": "Codex, exact deployment identifier unavailable",
            "reviewer_mode": "same_model_blind",
            "decision": "SOLVED",
            "independent_gold": ["o1"],
            "evidence": [{"turn_ref": "T1", "speaker": "A", "quote": "The first event was moved to Tuesday."},
                         {"turn_ref": "T2", "speaker": "B", "quote": "The new time is 14:00."}],
            "reasoning_steps": ["The second statement gives the updated time for the moved event."],
            "ambiguities": [],
            "source_limitations": ["The source reports a planned time, not completion."],
            "isolation_receipt": None,
        }
        self.orchestration_receipt = qa.build_orchestration_receipt(
            "q1", self.packet_hash, self.scope_hash, "isolated-reviewer", "fresh-1", "context-orchestrator")
        self.response["isolation_receipt"] = self.orchestration_receipt

    def parse(self, response=None, raw=None):
        if raw is None:
            raw = json.dumps(response or self.response, ensure_ascii=False).encode("utf-8")
        return qa.parse_blind_response(raw, self.packet_hash, self.scope_hash,
                                       self.candidate, self.source, self.source_scope,
                                       self.orchestration_receipt)

    def freeze(self, run_dir, response=None, raw=None, receipt=None):
        if raw is None:
            raw = json.dumps(response or self.response, ensure_ascii=False).encode("utf-8")
        qa.freeze_author_candidate(run_dir, self.candidate, self.packet_hash, self.scope_hash)
        return qa.freeze_blind_response(run_dir, raw, self.packet_hash, self.scope_hash,
                                        self.candidate, self.source, self.source_scope,
                                        receipt or self.orchestration_receipt)

    def disposition(self, run_dir, gold=None):
        return qa.blind_disposition(run_dir, self.packet_hash, self.scope_hash,
                                    self.source, self.source_scope,
                                    self.orchestration_receipt)

    def test_valid_solved_with_nonblocking_limitation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.freeze(tmp)
            disposition = self.disposition(tmp)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["response"]["source_limitations"], self.response["source_limitations"])
        self.assertEqual(disposition["decision"], "REVIEWER_MATCH")

    def test_missing_field_is_invalid_and_raw_bytes_are_preserved(self):
        response = dict(self.response)
        response.pop("source_scope_sha256")
        raw = json.dumps(response).encode()
        with tempfile.TemporaryDirectory() as tmp:
            qa.freeze_author_candidate(tmp, self.candidate, self.packet_hash, self.scope_hash)
            result = qa.freeze_blind_response(tmp, raw, self.packet_hash, self.scope_hash,
                                              self.candidate, self.source, self.source_scope,
                                              self.orchestration_receipt)
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), raw)
            self.assertEqual(result["raw_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["status"], "INVALID")
            self.assertFalse(Path(tmp, "BLIND_SOLUTION.json").exists())

    def test_wrong_field_types_and_unhashable_evidence_do_not_raise(self):
        for field, value in (("independent_gold", "o1"), ("ambiguities", "none"),
                             ("evidence", [{"turn_ref": [], "speaker": "A", "quote": "x"}]),
                             ("isolation_receipt", {"context_mode": "fresh_context_no_history",
                                                     "author_material_visible": 0, "context_id": "x"})):
            response = dict(self.response)
            response[field] = value
            result = self.parse(response)
            self.assertEqual(result["status"], "INVALID", field)

    def test_extra_field_is_rejected(self):
        response = dict(self.response, unexpected="value")
        result = self.parse(response)
        self.assertIn("FIELD_EXTRA:unexpected", result["errors"])

    def test_invalid_utf8_malformed_duplicate_and_trailing_text_are_invalid(self):
        payloads = [b"\xff", b"{", b'{"x":1,"x":2}',
                    json.dumps(self.response).encode() + b" trailing"]
        for raw in payloads:
            self.assertEqual(self.parse(raw=raw)["status"], "INVALID")

    def test_solved_with_unresolved_ambiguity_is_invalid(self):
        response = dict(self.response, ambiguities=["Either answer may fit."])
        result = self.parse(response)
        self.assertIn("AMBIGUITIES:UNRESOLVED_SOLVED", result["errors"])

    def test_gold_conflict_holds_without_rewriting_answers(self):
        response = dict(self.response, independent_gold=["o2"])
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp, response=response)
            disposition = self.disposition(tmp)
        self.assertEqual(disposition["decision"], "HOLD")
        self.assertTrue(disposition["gold_compared"])
        self.assertEqual(disposition["author_gold"], ["o1"])
        self.assertEqual(disposition["reviewer_gold"], ["o2"])

    def test_reviewer_hold_never_compares_gold(self):
        response = dict(self.response, decision="HOLD", independent_gold=[],
                        ambiguities=["The source does not establish the event identity."])
        with tempfile.TemporaryDirectory() as tmp:
            result = self.freeze(tmp, response=response)
            disposition = self.disposition(tmp)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(disposition["decision"], "HOLD")
        self.assertFalse(disposition["gold_compared"])

    def test_disposition_rejects_unfrozen_parse_result(self):
        result = self.parse()
        disposition = qa.blind_disposition(result, self.packet_hash, self.scope_hash,
                                           self.source, self.source_scope,
                                           self.orchestration_receipt)
        self.assertEqual(disposition["decision"], "HOLD")
        self.assertFalse(disposition["gold_compared"])
        self.assertIn("BLIND_FREEZE_DIRECTORY_REQUIRED", disposition["errors"])

    def test_unapproved_source_turn_cannot_be_used_as_blind_evidence(self):
        response = dict(self.response, evidence=[{
            "turn_ref": "T3", "speaker": "C", "quote": "An unapproved note says 15:00."}])
        result = self.parse(response)
        self.assertEqual(result["status"], "INVALID")
        self.assertTrue(any("TURN_NOT_FORMALLY_ALLOWED:T3" in error for error in result["errors"]))

    def test_review_reply_cannot_supply_or_replace_orchestration_receipt(self):
        response = dict(self.response, isolation_receipt=dict(self.orchestration_receipt,
                                                               context_id="invented-context"))
        result = self.parse(response)
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("ISOLATION_RECEIPT:RESPONSE_MISMATCH", result["errors"])

    def test_tampered_frozen_solution_cannot_reach_gold_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            Path(tmp, "BLIND_SOLUTION.json").write_text(json.dumps(dict(self.response, independent_gold=["o2"])))
            disposition = self.disposition(tmp)
        self.assertEqual(disposition["decision"], "HOLD")
        self.assertFalse(disposition["gold_compared"])
        self.assertIn("BLIND_SOLUTION_FILE_HASH_MISMATCH", disposition["errors"])

    def test_gold_comparison_uses_pre_review_candidate_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            self.candidate["gold"] = ["o2"]
            disposition = self.disposition(tmp)
        self.assertEqual(disposition["decision"], "REVIEWER_MATCH")
        self.assertEqual(disposition["author_gold"], ["o1"])

    def test_tampered_candidate_gold_freeze_blocks_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            Path(tmp, "AUTHOR_CANDIDATE.json").write_text(
                json.dumps(dict(self.candidate, gold=["o2"])), encoding="utf-8")
            disposition = self.disposition(tmp)
        self.assertEqual(disposition["decision"], "HOLD")
        self.assertFalse(disposition["gold_compared"])
        self.assertIn("AUTHOR_CANDIDATE_FREEZE_HASH_MISMATCH", disposition["errors"])

    def test_historical_first_blind_artifact_is_not_repaired_or_reinterpreted(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "research-protocols/selfplay_week_20261019/base_qa_generation_v2_20261031/real_root/state/BLIND_SOLUTION.json"
        historical_bytes = path.read_bytes()
        historical = json.loads(historical_bytes)
        result = self.parse(historical)
        self.assertEqual(result["status"], "INVALID")
        self.assertTrue(any(code.startswith("FIELD_MISSING:") for code in result["errors"]))
        self.assertIn("FIELD_EXTRA:hash_verification", result["errors"])
        self.assertEqual(path.read_bytes(), historical_bytes)


class V21SourceAdmissionTests(unittest.TestCase):
    def make_source_and_receipt(self, tmp, review_scope="formal_qa_material_admission",
                                reviewer_id="independent-material-reviewer", source_hash_override=None,
                                include_warn_turn=False):
        turns = [{"turn_ref": "T1", "source_id": "source-a", "source_version": "v1",
                  "origin": "source_grounded", "speaker": "A", "text": "A source fact."}]
        if include_warn_turn:
            turns.append({"turn_ref": "T2", "source_id": "source-a", "source_version": "v1",
                          "origin": "source_grounded", "speaker": "B", "text": "A reviewed but unapproved fact."})
        source = {"root_id": "root-a", "source_id": "source-a", "source_version": "v1",
                  "turns": turns}
        source_path = Path(tmp, "SOURCE.json")
        source_path.write_text(json.dumps(source), encoding="utf-8")
        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        source_binding = {"path": str(source_path), "sha256": source_hash, "bytes": len(source_bytes)}
        review_rows = []
        for turn in turns:
            turn_id = turn["turn_ref"]
            status = "WARN" if turn_id == "T2" else "PASS"
            allowed = status == "PASS"
            review_record = {"schema_version": "base_qa_turn_material_review_v2_1",
                             "source_sha256": source_hash, "source_id": "source-a", "turn_id": turn_id,
                             "provenance_type": "original_root_dialogue", "source_version": "v1",
                             "reviewer_id": reviewer_id, "review_status": status,
                             "formal_qa_allowed": allowed}
            review_path = Path(tmp, "REVIEW_" + turn_id + ".json")
            review_path.write_text(json.dumps(review_record), encoding="utf-8")
            review_bytes = review_path.read_bytes()
            review_binding = {"path": str(review_path), "sha256": hashlib.sha256(review_bytes).hexdigest(),
                              "bytes": len(review_bytes)}
            review_rows.append({"turn_id": turn_id, "source_id": "source-a", "source_version": "v1",
                                "provenance_type": "original_root_dialogue", "review_status": status,
                                "review_record": str(review_path), "review_record_binding": review_binding,
                                "reviewer_id": reviewer_id, "formal_qa_allowed": allowed})
        receipt = {"schema_version": qa.RECEIPT_SCHEMA, "source_id": "source-a",
                   "source_sha256": source_hash_override or source_hash,
                   "review_scope": review_scope, "reviewer_id": reviewer_id, "decision": "READY",
                   "turns": review_rows}
        receipt_path = Path(tmp, "ADMISSION.json")
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        raw = receipt_path.read_bytes()
        receipt_binding = {"path": str(receipt_path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return source, source_binding, receipt, receipt_binding

    def test_verified_hash_bound_per_turn_approval_is_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_source_and_receipt(tmp)
            manifest = qa.source_manifest(*args[:3], admission_binding=args[3], challenger_id="challenger")
        self.assertTrue(manifest["formal_qa_eligible"])
        self.assertEqual(manifest["turns"][0]["provenance_type"], "original_root_dialogue")
        self.assertTrue(qa.formal_evidence_gate(manifest, ["T1"])["eligible"])

    def test_approved_turn_remains_usable_when_other_turn_is_explicitly_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, binding, receipt, receipt_binding = self.make_source_and_receipt(
                tmp, include_warn_turn=True)
            manifest = qa.source_manifest(source, binding, receipt, receipt_binding, "challenger")
        self.assertTrue(manifest["source_admission_complete"])
        self.assertFalse(manifest["formal_qa_eligible"])
        self.assertTrue(qa.formal_evidence_gate(manifest, ["T1"])["eligible"])
        self.assertFalse(qa.formal_evidence_gate(manifest, ["T2"])["eligible"])

    def test_missing_or_wrong_scope_or_wrong_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, binding, receipt, receipt_binding = self.make_source_and_receipt(tmp)
            missing = qa.source_manifest(source, binding)
            wrong_scope_receipt = dict(receipt, review_scope="source_consistency_only")
            wrong_scope_path = Path(tmp, "WRONG_SCOPE.json")
            wrong_scope_path.write_text(json.dumps(wrong_scope_receipt), encoding="utf-8")
            wrong_scope_bytes = wrong_scope_path.read_bytes()
            wrong_scope_binding = {"path": str(wrong_scope_path), "sha256": hashlib.sha256(wrong_scope_bytes).hexdigest(), "bytes": len(wrong_scope_bytes)}
            wrong_scope = qa.source_manifest(source, binding, wrong_scope_receipt, wrong_scope_binding, "challenger")
            wrong_hash = dict(receipt, source_sha256="0" * 64)
            wrong_hash_path = Path(tmp, "WRONG_HASH.json")
            wrong_hash_path.write_text(json.dumps(wrong_hash), encoding="utf-8")
            wrong_hash_bytes = wrong_hash_path.read_bytes()
            wrong_hash_binding = {"path": str(wrong_hash_path), "sha256": hashlib.sha256(wrong_hash_bytes).hexdigest(), "bytes": len(wrong_hash_bytes)}
            wrong_source = qa.source_manifest(source, binding, wrong_hash, wrong_hash_binding, "challenger")
        self.assertFalse(missing["formal_qa_eligible"])
        self.assertFalse(wrong_scope["formal_qa_eligible"])
        self.assertFalse(wrong_source["formal_qa_eligible"])

    def test_challenger_cannot_approve_its_own_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, binding, receipt, receipt_binding = self.make_source_and_receipt(
                tmp, reviewer_id="challenger")
            manifest = qa.source_manifest(source, binding, receipt, receipt_binding,
                                          challenger_id="challenger")
        self.assertFalse(manifest["formal_qa_eligible"])
        self.assertIn("SOURCE_ADMISSION_SELF_APPROVAL", manifest["admission_errors"])

    def test_formal_admission_requires_file_binding_and_explicit_challenger_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, binding, receipt, receipt_binding = self.make_source_and_receipt(tmp)
            unbound = qa.source_manifest(source, None, receipt, receipt_binding, "challenger")
            no_challenger = qa.source_manifest(source, binding, receipt, receipt_binding)
        self.assertFalse(unbound["formal_qa_eligible"])
        self.assertIn("SOURCE_BINDING_REQUIRED_FOR_ADMISSION", unbound["admission_errors"])
        self.assertFalse(no_challenger["formal_qa_eligible"])
        self.assertIn("CHALLENGER_ID_REQUIRED", no_challenger["admission_errors"])

    def test_generated_extension_requires_explicit_subtype(self):
        source = {"root_id": "root-a", "source_id": "source-a", "turns": [
            {"turn_ref": "T1", "origin": "generated_extension", "speaker": "A", "text": "x"}]}
        manifest = qa.source_manifest(source)
        self.assertEqual(manifest["turns"][0]["provenance_type"], "generated_extension_unclassified")
        self.assertFalse(manifest["turns"][0]["formal_qa_allowed"])

    def test_source_file_binding_drift_invalidates_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, binding, receipt, receipt_binding = self.make_source_and_receipt(tmp)
            binding = dict(binding, sha256="0" * 64)
            manifest = qa.source_manifest(source, binding, receipt, receipt_binding, "challenger")
        self.assertFalse(manifest["formal_qa_eligible"])
        self.assertIn("SOURCE_BINDING_DRIFT", manifest["admission_errors"])


class V21OptionAuditTests(unittest.TestCase):
    def test_first_hold_numeric_shortcuts_are_detected(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "research-protocols/selfplay_week_20261019/base_qa_generation_v2_20261031/real_root/state/CANDIDATE.json"
        candidate = qa.v2.read(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        audit = qa.style_audit(candidate)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertIn("GOLD_EXTREME_MATCHES_STEM", audit["flags"])
        self.assertIn("GOLD_NUMERIC_DIRECTION_OUTLIER", audit["flags"])
        self.assertIn("GOLD_UNIQUE_NUMERIC_WIDTH", audit["flags"])

    def test_non_numeric_choices_still_compare_length_and_semantics(self):
        candidate = {"question_type": "single_choice", "question": "Which appointment was cancelled?",
                     "gold": ["a"], "options": {
                         "a": "The old appointment, cancelled", "b": "Tuesday", "c": "Friday"}}
        audit = qa.style_audit(candidate)
        self.assertIn("GOLD_UNIQUE_SEMANTIC_CUE", audit["flags"])
        self.assertFalse(audit["static_shortcuts_proved_absent"])

    def test_multiselect_gold_set_semantic_cue_pattern_is_flagged(self):
        candidate = {"question_type": "multi_select", "question": "Which updates were completed?",
                     "gold": ["a", "b"], "options": {
                         "a": "The meeting was already moved",
                         "b": "The report was already sent",
                         "c": "The meeting may move",
                         "d": "The report may be sent"}}
        audit = qa.style_audit(candidate)
        self.assertIn("GOLD_SET_SEMANTIC_CUE_PATTERN", audit["flags"])

    def test_multiselect_semantic_cue_on_gold_subset_is_flagged(self):
        candidate = {"question_type": "multi_select", "question": "Which updates are current?",
                     "gold": ["a", "b"], "options": {
                         "a": "The Tuesday plan is still active",
                         "b": "The report was sent",
                         "c": "The Monday plan was cancelled",
                         "d": "The Friday plan was completed"}}
        audit = qa.style_audit(candidate)
        self.assertIn("GOLD_OPTION_UNIQUE_SEMANTIC_CUE", audit["flags"])

    def test_multiselect_numeric_class_separation_is_flagged(self):
        candidate = {"question_type": "multi_select", "question": "Which two totals are supported?",
                     "gold": ["a", "b"], "options": {
                         "a": "10 units after 9:00",
                         "b": "12 units after 9:15",
                         "c": "90 units after 10:00",
                         "d": "95 units after 10:15"}}
        audit = qa.style_audit(candidate)
        self.assertIn("GOLD_SET_NUMERIC_CLASS_SEPARATION", audit["flags"])

    def test_relevant_semantic_marker_is_only_a_review_trigger(self):
        candidate = {"question_type": "single_choice", "question": "Which plan remains active?",
                     "gold": ["a"], "options": {
                         "a": "The Tuesday plan is still active",
                         "b": "The Monday plan was cancelled",
                         "c": "The Friday plan was completed"}}
        audit = qa.style_audit(candidate)
        self.assertIn("GOLD_UNIQUE_SEMANTIC_CUE", audit["flags"])
        self.assertEqual(audit["decision"], "HOLD_REQUIRES_OPTION_ONLY_REVIEW")
        self.assertFalse(audit["static_shortcuts_proved_absent"])

    def test_numeric_shortcut_detection_handles_multiple_numbers_per_option(self):
        candidate = {"question_type": "single_choice", "question": "Which total is the minimum?",
                     "gold": ["a"], "options": {
                         "a": "61 minutes, starting at 9:00",
                         "b": "72 minutes, starting at 10:00",
                         "c": "83 minutes, starting at 11:00"}}
        audit = qa.style_audit(candidate)
        self.assertIn("GOLD_EXTREME_MATCHES_STEM", audit["flags"])
        self.assertIn("GOLD_NUMERIC_DIRECTION_OUTLIER", audit["flags"])

    def test_numeric_range_gap_check_scans_each_shared_number_position(self):
        candidate = {"question_type": "single_choice", "question": "Which value is closest to 10?",
                     "gold": ["a"], "options": {
                         "a": "10 units at 9:00", "b": "11 units at 9:15",
                         "c": "12 units at 9:30", "d": "13 units at 9:45",
                         "e": "100 units at 10:00"}}
        audit = qa.style_audit(candidate)
        self.assertIn("NUMERIC_RANGE_GAP_OUTLIER", audit["flags"])

    def test_malformed_gold_ids_return_hold_without_raising(self):
        candidate = {"question_type": "single_choice", "gold": [[]],
                     "options": {"a": "First", "b": "Second"}}
        audit = qa.style_audit(candidate)
        self.assertEqual(audit["decision"], "HOLD_INVALID_GOLD")


if __name__ == "__main__":
    unittest.main()
