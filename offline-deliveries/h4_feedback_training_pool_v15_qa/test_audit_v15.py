from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_run as audit
import ordering_gate as gate
import test_ordering_spans as fixtures


class AuditV15Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_valid_slot_survives_duplicate_abstention_for_another_id(self):
        raw = {"root_id": "r", "questions": [{"candidate_id": "good", "question": "unchanged"}],
               "abstentions": [{"candidate_id": "other"}, {"candidate_id": "other"}], "extra": "diagnostic"}
        result = audit.raw_candidate(json.dumps(raw), "good", "r", ["good", "other"])
        self.assertEqual(result, raw["questions"][0])
        for invalid in (
            {**raw, "questions": raw["questions"] * 2},
            {**raw, "abstentions": [{"candidate_id": "good"}]},
            {**raw, "root_id": "wrong"},
        ):
            with self.assertRaises(ValueError):
                audit.raw_candidate(json.dumps(invalid), "good", "r", ["good", "other"])
        with self.assertRaisesRegex(ValueError, "was_not_requested"):
            audit.raw_candidate(json.dumps(raw), "good", "r", ["other"])

    def ordering_fixture(self, *, singleton=False):
        fixture = fixtures.OrderingSpanContractTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # These are annotation-only differences and must not revive the old gate.
        fixture.row["item_evidence"]["I1"].update(status="proposal", reason="")
        errors, candidate = gate.validate_generated_ordering(fixture.row, "q1", "root", fixture.source, fixture.source_path)
        self.assertEqual(errors, [])
        response_path = fixture.directory / "generation/RESPONSE.json"
        request_path = response_path.with_name("REQUEST.json")
        candidate.update(generator_model=audit.MODEL, raw_question_content_preserved=True,
                         generation_response_path=str(response_path), generation_request_path=str(request_path))
        fixture.candidate = candidate
        self.write(response_path, {"text": json.dumps({"root_id": "root", "questions": [fixture.row],
                    "abstentions": [{"candidate_id": "q2"}, {"candidate_id": "q2"}]}),
                    "metadata": {"finish_reason": "stop", "http_status": 200}})
        self.write(request_path, {})
        self.write(response_path.with_name("PROOF.json"), {"candidate_ids": ["q1", "q2"]})
        review = fixture.review(singleton=singleton)
        self.write(fixture.directory / "REVIEW.json", review)
        return fixture, review["accepted"][0]

    def test_real_candidate_audit_accepts_singleton_and_annotation_diagnostics(self):
        fixture, reviewed = self.ordering_fixture(singleton=True)
        auditor = audit.Audit()
        sources = {"root": fixture.source}
        auditor.candidate(fixture.candidate, sources)
        auditor.candidate(reviewed, sources, reviewed=True)
        self.assertEqual(auditor.counts["singleton_review_wrappers_verified"], 1)
        self.assertEqual(auditor.counts["same_turn_ordering_candidates_checked"], 1)
        self.assertEqual(reviewed["item_evidence"]["I1"]["status"], "proposal")

    def test_unchanged_wrapped_review_remains_supported(self):
        fixture, reviewed = self.ordering_fixture()
        auditor = audit.Audit()
        auditor.candidate(reviewed, {"root": fixture.source}, reviewed=True)
        self.assertEqual(auditor.counts["singleton_review_wrappers_verified"], 0)

    def test_reviewer_raw_proof_or_claim_rewrite_is_rejected(self):
        fixture, reviewed = self.ordering_fixture(singleton=True)
        altered = deepcopy(reviewed)
        altered["qwen_review"]["raw_response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "review_raw_or_request_proof_mismatch"):
            audit.Audit().candidate(altered, {"root": fixture.source}, reviewed=True)
        altered = deepcopy(reviewed)
        altered["items"]["I1"] = "An edited claim."
        with self.assertRaisesRegex(ValueError, "ordering_items_rewritten"):
            audit.Audit().candidate(altered, {"root": fixture.source}, reviewed=True)

    def test_singleton_with_wrong_id_extra_fields_or_duplicate_json_keys_rejected(self):
        fixture, reviewed = self.ordering_fixture()
        raw = deepcopy(fixture.verdict)
        for value in ({**raw, "candidate_id": "other"}, {**raw, "gold": ["I1"]}):
            with self.assertRaises(ValueError):
                audit.independent_review_object(json.dumps(value), "q1")
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            audit.independent_review_object(json.dumps(raw)[:-1] + ',"candidate_id":"q1"}', "q1")

    def test_independent_audit_keeps_modality_and_simultaneous_alternative_checks(self):
        fixture, reviewed = self.ordering_fixture()
        spans = reviewed["order_anchor_spans"]
        bad = deepcopy(fixture.verdict)
        bad["items"]["I1"]["modality_ok"] = False
        with self.assertRaisesRegex(ValueError, "review_item_not_pass"):
            audit.independent_review_findings(bad, spans, fixture.source)
        bad = deepcopy(fixture.verdict)
        bad["items"]["I2"]["alternative_matching_anchors"] = [fixture.alternative("Payment is planned.")]
        bad["items"]["I1"]["alternative_matching_anchors"] = [fixture.alternative("The amount is confirmed.")]
        with self.assertRaisesRegex(ValueError, "alternative_spans_cross_or_overlap"):
            audit.independent_review_findings(bad, spans, fixture.source)

    def test_cli_requires_explicit_run(self):
        result = subprocess.run([sys.executable, str(Path(audit.__file__))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--run", result.stderr)

    def test_empty_completed_batch_accepts_arbitrary_five_roots_and_no_request_dirs(self):
        run, expansion = self.directory / "qa", self.directory / "fresh_expansion"
        roots = [f"new-root-{i}" for i in range(5)]
        self.write(expansion / "MANIFEST.json", {"roots": roots, "target_solver_model": audit.TARGET,
                   "selected_routes": list(audit.ROUTES.values()), "files": []})
        self.write(expansion / "REPORT.json", {"complete": True})
        self.write(run / "MANIFEST.json", {"version": "h4_feedback_training_pool_v15_qa", "roots": roots,
                   "files": [], "expansion_run": str(expansion)})
        per_root = {root: {"selected_unique_exact": 0, "selected_multi_select": 0, "selected_ordering": 0} for root in roots}
        self.write(run / "REPORT.json", {"status": "EVALUATION_COMPLETE", "per_root": per_root,
                   "selected_total_before_cross_branch_exact_dedup": 0, "selected_unique_exact": 0,
                   "cross_branch_exact_duplicates": [], "selected_multi_select": 0, "selected_ordering": 0,
                   "generation_http_attempts": 0, "full_solver_calls": 0, "ground_calls": 0, "training_calls": 0})
        self.write(run / "SELECTED_QUESTIONS.json", {"questions": [], "training_started": False,
                   "training_target": "complete_canonical_gold", "loss_weight_from_partial_score": False})
        for branch, route in audit.ROUTES.items():
            gen = run / branch / "generation"
            rows, outcomes = [], []
            for root in roots:
                frozen = {"root_id": root, "turns": [{"turn_ref": "P1", "speaker": "A", "text": "Original."}]}
                self.write(expansion / "sources" / (root + ".json"), frozen)
                source = {**frozen, "route": route, "branch_independent": True}
                original = expansion / "roots" / root / branch / "cumulative_dialogue.json"
                self.write(original, source)
                self.write(original.with_name("RESULT.json"), {"root_id": root, "route": route,
                    "accepted_scenes": 0, "added_turns": 0, "scene_metadata": []})
                target = gen / "sources" / (root + ".json")
                self.write(target, source)
                self.write(gen / "prior" / (root + ".json"), [])
                rows.append({"root_id": root, "source_path": str(target), "source_sha256": audit.sha(target),
                             "added_turns": 0, "accepted_scenes": 0})
                outcomes += [{"root_id": root, "question_type": kind, "mechanical_rejected": []}
                             for kind in ("multi_select", "ordering")]
            self.write(gen / "MANIFEST.json", {"files": [], "branch": branch, "route": route, "roots": roots,
                       "question_types": ["multi_select", "ordering"], "solver_model": audit.TARGET,
                       "ground_calls": 0, "training_calls": 0, "source_rows": rows})
            self.write(gen / "REPORT.json", {"mechanical_pass": 0, "http_attempts": 0, "ready_for_solver": 0, "roots": outcomes})
            self.write(gen / "MECHANICAL_POOL.json", {"questions": []})
            self.write(gen / "SOLVER_POOL.json", {"questions": []})
            self.write(run / branch / "solver/SKIPPED.json", {"request_count": 0})
            self.write(run / branch / "delivery/SELECTED_QUESTIONS.json", {"questions": [], "training_started": False})
        # This case exercises root/route/empty-pool accounting, independently
        # of the feedback chain covered by the dedicated real-lock test.
        with patch.object(audit.Audit, "feedback_chain"):
            result = audit.Audit().run(run)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_complete_gold_count"], 0)

    def test_real_previous_round_feedback_chain_and_changed_lock_rejected(self):
        deliveries = Path(audit.__file__).resolve().parent.parent
        expansion = deliveries / "h4_r02_yield_pilot_v12/runs/contract_smoke_2roots_01"
        generation = deliveries / "h4_r02_question_contract_v14/runs/frozen_v12_2roots_01/A/generation"
        auditor = audit.Audit()
        auditor.feedback_chain(expansion, generation)
        self.assertEqual(auditor.counts["expansion_to_qa_feedback_chains_checked"], 1)
        fake_generation = self.directory / "bad_feedback"
        self.write(fake_generation / "feedback/source_lock.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "file_hash_changed"):
            audit.Audit().feedback_chain(expansion, fake_generation)


if __name__ == "__main__":
    unittest.main()
