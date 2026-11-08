"""Independent source-repair workflow tests using temporary synthetic inputs.

These fixtures use the real helper at a copied, frozen runtime path. They do
not read/write real candidate payloads, perform semantic certification, or call
any model. The reviewer who owns this file is not the tool's author.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = "cc-train-episode-1"
AUTHOR = "/root/synthetic_author"
READERS = ["/root/synthetic_reader_a", "/root/synthetic_reader_b"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


class SourceRepairWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="v121-source-independent-")
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name)
        self.tool = self.batch / "tools/source_repair_v121.py"
        self.tool.parent.mkdir()
        shutil.copyfile(HERE / "source_repair_v121.py", self.tool)
        spec = importlib.util.spec_from_file_location("synthetic_source_repair", self.tool)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.unit = self.batch / "units/root1"
        self.folder = self.unit / "source_repair"
        self.original = self.batch / "upstream/BASELINE.json"
        self.source = {
            "root_id": ROOT,
            "expanded_dialogue_version": "synthetic_old_wrapper",
            "parent_dialogue_sha256": "old-parent-hash-for-provenance-only",
            "complete_dialogue_sha256": "old-complete-hash-for-provenance-only",
            "turn_count": 3,
            "turns": [
                {"turn_ref": "R01:S01:T001", "session_ref": "R01:S01",
                 "scene_ref": "R01:S01", "speaker": "Alex", "text": "I sent the note.", "origin": "synthetic_fixture"},
                {"turn_ref": "R01:S01:T002", "session_ref": "R01:S01",
                 "scene_ref": "R01:S01", "speaker": "Robin", "text": "I received the note.", "origin": "synthetic_fixture"},
                {"turn_ref": "R01:S01:T003", "session_ref": "R01:S01",
                 "scene_ref": "R01:S01", "speaker": "Alex", "text": "I have not sent the note.", "origin": "synthetic_fixture"},
            ],
        }
        write(self.original, self.source)
        self.policy = self.batch / "SOURCE_REPAIR_POLICY.md"
        self.policy.write_text("Synthetic test: only exact text repair, never recovered facts.\n")
        self.roles_path = self.batch / "ROLE_ASSIGNMENTS.json"
        write(self.roles_path, {"synthetic_fixture_only": True, "roots": [
            {"root_id": ROOT, "author": AUTHOR, "baseline_reviewers": READERS}
        ]})
        self.inputs_path = self.batch / "SOURCE_INPUTS.json"
        self.inputs = {
            "role_assignments_binding": self.m.binding(self.roles_path),
            "protocol_inputs": [{"label": "source_policy", **self.m.binding(self.policy)}],
            "roots": [{"root_id": ROOT, "allow_source_edit": True,
                       "original_baseline_binding": self.m.binding(self.original),
                       "inputs": [{"label": "baseline", **self.m.binding(self.original)}]}],
        }
        write(self.inputs_path, self.inputs)
        self.freeze()

    def freeze(self):
        paths = [self.tool, self.roles_path, self.inputs_path, self.original, self.policy]
        write(self.batch / "SOURCE_PROTOCOL_FREEZE.json", {
            "synthetic_fixture_only": True,
            "status": "FROZEN_BEFORE_V122_SOURCE_AUTHORING",
            "created_at_utc": (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
            "bindings": [self.m.binding(p) for p in paths],
        })

    def load(self):
        return self.m.load(self.batch, ROOT)

    def quiet(self, func, *args):
        with redirect_stdout(io.StringIO()):
            return func(*args)

    def author_payload(self, edited=True, decision="READY"):
        refs = [t["turn_ref"] for t in self.source["turns"]]
        return {
            "root_id": ROOT, "executor": AUTHOR,
            "status": "EDITED_SYNTHETIC" if edited else "UNCHANGED",
            "author_decision": decision,
            "patches": [{"turn_ref": refs[2], "old_text": self.source["turns"][2]["text"],
                         "new_text": "The note has been sent.",
                         "reason": "Align the synthetic final state with the first two turns.",
                         "change_kind": "synthetic_state_alignment"}] if edited else [],
            "continuation_constraints": ["The note is sent and received; do not regress it to unsent."],
            "unresolved_issues": ["Synthetic unresolved issue."] if decision == "HOLD" else [],
            "failure_explanation": "Synthetic author HOLD." if decision == "HOLD" else "",
            "read_attestation": {**{k: True for k in self.m.AUTHOR_ATTESTATIONS}, "turn_refs_read": refs},
        }

    def author_read(self):
        self.quiet(self.m.author_read, self.load(), AUTHOR)

    def author_save(self, payload=None):
        payload = self.author_payload() if payload is None else payload
        path = self.batch / "synthetic_author_submission.json"
        write(path, payload)
        self.quiet(self.m.author_save, self.load(), AUTHOR, path)

    def authored(self, payload=None):
        self.author_read()
        self.author_save(payload)

    def review_payload(self, reader, decision="READY"):
        candidate, author, _ = self.m.checked_candidate(self.load())
        refs = [t["turn_ref"] for t in candidate["turns"]]
        return {
            "root_id": ROOT, "executor": reader, "decision": decision,
            "checks": {k: {"status": "PASS", "reason": "Synthetic fixture check."} for k in self.m.SOURCE_CHECKS},
            "baseline_judgment": {
                "full_source_read": True, "covered_turn_refs": list(refs),
                "checks": {k: {"complete": True, "status": "PASS", "reason": "Synthetic fixture check."}
                           for k in self.m.BASELINE_CHECKS},
                "findings": [], "continuation_constraints": ["Keep the three synthetic turns in their source order."],
                "decision": "PASS",
            },
            "source_repair_assertions": {k: decision == "READY" for k in self.m.READY_ASSERTIONS},
            "read_attestation": {
                **{k: True for k in self.m.REVIEW_ATTESTATIONS},
                "original_turn_refs_read": list(refs), "candidate_turn_refs_read": list(refs),
                "author_constraints_read": author["continuation_constraints"],
            },
            "failure_explanation": "Synthetic reviewer HOLD." if decision == "HOLD" else "",
        }

    def review_read(self, reader):
        self.quiet(self.m.review_read, self.load(), reader)

    def review_save(self, reader, payload=None):
        payload = self.review_payload(reader) if payload is None else payload
        path = self.batch / ("synthetic_judgment_" + str(READERS.index(reader)) + ".json")
        write(path, payload)
        self.quiet(self.m.review_save, self.load(), reader, path)

    def reviewed(self, decisions=("READY", "READY")):
        for reader, decision in zip(READERS, decisions):
            self.review_read(reader)
            self.review_save(reader, self.review_payload(reader, decision))

    def finalize(self):
        self.quiet(self.m.finalize_source, self.load(), "/root")

    def rebind_candidate_outputs(self):
        manifest = read(self.folder / "PATCH_MANIFEST.json")
        manifest["candidate_binding"] = self.m.binding(self.folder / "CANDIDATE_BASELINE.json")
        write(self.folder / "PATCH_MANIFEST.json", manifest)
        receipt = read(self.folder / "AUTHOR_RECEIPT.json")
        receipt["candidate_binding"] = manifest["candidate_binding"]
        receipt["patch_manifest_binding"] = self.m.binding(self.folder / "PATCH_MANIFEST.json")
        write(self.folder / "AUTHOR_RECEIPT.json", receipt)

    def test_real_cli_author_read(self):
        result = subprocess.run([sys.executable, "-B", str(self.tool), "author-read", "--batch", str(self.batch),
                                 "--root", ROOT, "--executor", AUTHOR], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("I sent the note.", result.stdout)
        self.assertTrue((self.folder / "AUTHOR_READ_CONTEXT.json").is_file())

    def test_full_edited_read_save_two_reviews_finalize(self):
        self.authored()
        self.reviewed()
        self.finalize()
        self.assertEqual(read(self.folder / "FINALIZE_RECEIPT.json")["status"], "SOURCE_READY")
        output = read(self.unit / "inputs/CONTINUATION_CONSTRAINTS.json")
        self.assertEqual(output["decision"], "PASS")
        self.assertTrue(output["source_is_edited_synthetic"])
        self.assertFalse(output["original_facts_recovered"])
        self.assertEqual(len(output["review_bindings"]), 2)
        self.assertEqual(self.m.verify(output["baseline_binding"]), self.unit / "inputs/BASELINE.json")

    def test_edited_metadata_wrapper_rehash(self):
        self.authored()
        candidate = read(self.folder / "CANDIDATE_BASELINE.json")
        self.assertNotIn("expanded_dialogue_version", candidate)
        self.assertNotIn("parent_dialogue_sha256", candidate)
        self.assertEqual(candidate["original_source_metadata"]["parent_dialogue_sha256"], self.source["parent_dialogue_sha256"])
        self.assertEqual(candidate["complete_dialogue_sha256"], self.m.canonical_hash(candidate))
        self.assertNotEqual(candidate["complete_dialogue_sha256"], self.source["complete_dialogue_sha256"])
        self.assertFalse(candidate["original_facts_recovered"])
        for old, new in zip(self.source["turns"], candidate["turns"]):
            self.assertEqual({k: v for k, v in old.items() if k != "text"}, {k: v for k, v in new.items() if k != "text"})

    def test_unchanged_source_preserves_exact_original_bytes(self):
        self.authored(self.author_payload(edited=False))
        self.assertEqual((self.folder / "CANDIDATE_BASELINE.json").read_bytes(), self.original.read_bytes())
        self.reviewed()
        self.finalize()
        self.assertEqual((self.unit / "inputs/BASELINE.json").read_bytes(), self.original.read_bytes())
        self.assertFalse(read(self.unit / "inputs/CONTINUATION_CONSTRAINTS.json")["source_is_edited_synthetic"])

    def test_source_edit_forbidden_root(self):
        self.inputs["roots"][0]["allow_source_edit"] = False
        write(self.inputs_path, self.inputs)
        self.freeze()
        self.author_read()
        with self.assertRaisesRegex(ValueError, "SOURCE_EDIT_FORBIDDEN_FOR_ROOT"):
            self.author_save()
        self.assertTrue((self.folder / "AUTHOR_FAILURE.json").exists())

    def test_author_wrong_role(self):
        with self.assertRaisesRegex(ValueError, "ACTUAL_ASSIGNED_AUTHOR_REQUIRED"):
            self.m.author_read(self.load(), READERS[0])

    def test_reviewer_wrong_role(self):
        self.authored()
        with self.assertRaisesRegex(ValueError, "ACTUAL_ASSIGNED_BASELINE_REVIEWER_REQUIRED"):
            self.m.review_read(self.load(), AUTHOR)

    def test_parent_only_finalizer(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_FINALIZER_MUST_BE_PARENT_CONTROLLER"):
            self.m.finalize_source(self.load(), AUTHOR)

    def test_old_text_mismatch_consumes_first_attempt(self):
        payload = self.author_payload()
        payload["patches"][0]["old_text"] = "Not the exact old source."
        self.author_read()
        with self.assertRaisesRegex(ValueError, "PATCH_OLD_TEXT_NOT_EXACT"):
            self.author_save(payload)
        with self.assertRaisesRegex(ValueError, "SOURCE_FIRST_AUTHOR_ATTEMPT_ALREADY_SAVED"):
            self.author_save()
        self.assertEqual(read(self.folder / "AUTHOR_PAYLOAD.json"), payload)

    def test_patch_cannot_change_speaker(self):
        payload = self.author_payload()
        payload["patches"][0]["speaker"] = "Invented"
        self.author_read()
        with self.assertRaisesRegex(ValueError, "ONLY_EXACT_TURN_TEXT_PATCH_ALLOWED"):
            self.author_save(payload)

    def test_payload_cannot_add_turns(self):
        payload = self.author_payload()
        payload["turns"] = self.source["turns"] + [self.source["turns"][0]]
        self.author_read()
        with self.assertRaisesRegex(ValueError, "SOURCE_AUTHOR_PAYLOAD_FIELDS"):
            self.author_save(payload)

    def test_duplicate_patch_turn_rejected(self):
        payload = self.author_payload()
        payload["patches"].append(deepcopy(payload["patches"][0]))
        self.author_read()
        with self.assertRaisesRegex(ValueError, "PATCH_TURN_UNKNOWN_OR_DUPLICATE"):
            self.author_save(payload)

    def test_duplicate_semantic_author_save_rejected(self):
        self.authored()
        with self.assertRaisesRegex(ValueError, "SOURCE_FIRST_AUTHOR_ATTEMPT_ALREADY_SAVED"):
            self.author_save()

    def test_author_context_missing_binding_rejected(self):
        self.author_read()
        context = read(self.folder / "AUTHOR_READ_CONTEXT.json")
        context["input_bindings"].pop()
        write(self.folder / "AUTHOR_READ_CONTEXT.json", context)
        with self.assertRaisesRegex(ValueError, "SOURCE_AUTHOR_INPUT_SET_CHANGED"):
            self.author_save()

    def test_author_context_incomplete_render_rejected(self):
        self.author_read()
        path = self.folder / "AUTHOR_FULL_INPUT.txt"
        path.write_text("Partial fixture render.\n")
        context = read(self.folder / "AUTHOR_READ_CONTEXT.json")
        context["render_binding"] = self.m.binding(path)
        write(self.folder / "AUTHOR_READ_CONTEXT.json", context)
        with self.assertRaisesRegex(ValueError, "SOURCE_AUTHOR_RENDER_CHANGED_OR_INCOMPLETE"):
            self.author_save()

    def test_frozen_tool_tampering_rejected(self):
        with self.tool.open("a") as stream:
            stream.write("\n# synthetic tampering\n")
        with self.assertRaisesRegex(ValueError, "BOUND_FILE_CHANGED"):
            self.load()

    def test_frozen_policy_tampering_rejected(self):
        self.policy.write_text("Changed synthetic policy.\n")
        with self.assertRaisesRegex(ValueError, "BOUND_FILE_CHANGED"):
            self.load()

    def test_missing_runtime_freeze_binding_rejected(self):
        path = self.batch / "SOURCE_PROTOCOL_FREEZE.json"
        freeze = read(path)
        freeze["bindings"] = [x for x in freeze["bindings"] if x["path"] != str(self.tool)]
        write(path, freeze)
        with self.assertRaisesRegex(ValueError, "SOURCE_PROTOCOL_RUNTIME_NOT_FROZEN"):
            self.load()

    def test_missing_declared_policy_freeze_binding_rejected(self):
        path = self.batch / "SOURCE_PROTOCOL_FREEZE.json"
        freeze = read(path)
        freeze["bindings"] = [x for x in freeze["bindings"] if x["path"] != str(self.policy)]
        write(path, freeze)
        with self.assertRaisesRegex(ValueError, "DECLARED_INPUT_NOT_FROZEN"):
            self.load()

    def test_rebound_candidate_speaker_tamper_rejected(self):
        self.authored()
        path = self.folder / "CANDIDATE_BASELINE.json"
        candidate = read(path)
        candidate["turns"][1]["speaker"] = "Invented"
        candidate["complete_dialogue_sha256"] = self.m.canonical_hash(candidate)
        write(path, candidate)
        self.rebind_candidate_outputs()
        with self.assertRaisesRegex(ValueError, "CANDIDATE_NOT_EXACT_TEXT_PATCH_RESULT"):
            self.m.checked_candidate(self.load())

    def test_rebound_candidate_added_turn_tamper_rejected(self):
        self.authored()
        path = self.folder / "CANDIDATE_BASELINE.json"
        candidate = read(path)
        extra = deepcopy(candidate["turns"][0])
        extra["turn_ref"] = "R01:S01:T004"
        candidate["turns"].append(extra)
        candidate["turn_count"] = 4
        candidate["complete_dialogue_sha256"] = self.m.canonical_hash(candidate)
        write(path, candidate)
        self.rebind_candidate_outputs()
        with self.assertRaisesRegex(ValueError, "CANDIDATE_NOT_EXACT_TEXT_PATCH_RESULT"):
            self.m.checked_candidate(self.load())

    def test_review_context_missing_binding_rejected(self):
        self.authored()
        self.review_read(READERS[0])
        path = self.unit / "baseline_reviews/reader1/READ_CONTEXT.json"
        context = read(path)
        context["input_bindings"].pop()
        write(path, context)
        with self.assertRaisesRegex(ValueError, "SOURCE_REVIEW_EXACT_INPUT_SET_CHANGED"):
            self.review_save(READERS[0])

    def test_reviewer_read_before_author_finish_rejected(self):
        self.authored()
        self.review_read(READERS[0])
        path = self.unit / "baseline_reviews/reader1/READ_CONTEXT.json"
        context = read(path)
        context["started_at_utc"] = read(self.batch / "SOURCE_PROTOCOL_FREEZE.json")["created_at_utc"]
        write(path, context)
        with self.assertRaisesRegex(ValueError, "SOURCE_REVIEW_PREDATES_AUTHOR_COMPLETION"):
            self.review_save(READERS[0])

    def test_ready_with_failed_source_check_rejected(self):
        self.authored()
        reader = READERS[0]
        self.review_read(reader)
        payload = self.review_payload(reader)
        payload["checks"]["cross_scene_continuity"]["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "SOURCE_READY_WITH_FAILED_CHECK"):
            self.review_save(reader, payload)
        self.assertTrue((self.unit / "baseline_reviews/reader1/REVIEW_FAILURE.json").exists())
        with self.assertRaisesRegex(ValueError, "SOURCE_FIRST_REVIEW_ALREADY_SAVED"):
            self.review_save(reader)

    def test_native_baseline_incomplete_coverage_rejected(self):
        self.authored()
        reader = READERS[0]
        self.review_read(reader)
        payload = self.review_payload(reader)
        payload["baseline_judgment"]["covered_turn_refs"].pop()
        with self.assertRaisesRegex(ValueError, "NATIVE_BASELINE_FULL_COVERAGE"):
            self.review_save(reader, payload)

    def test_native_baseline_finding_quote_not_exact_rejected(self):
        self.authored()
        reader = READERS[0]
        self.review_read(reader)
        payload = self.review_payload(reader, "HOLD")
        judgment = payload["baseline_judgment"]
        judgment["findings"] = [{
            "id": "fixture-hold", "category": "fact_state_time", "classification": "HARD_CONTRADICTION",
            "disposition": "HOLD", "reason": "Synthetic failure.",
            "citations": [
                {"turn_ref": "R01:S01:T001", "speaker": "Alex", "quote": "Incorrect synthetic quote."},
                {"turn_ref": "R01:S01:T003", "speaker": "Alex", "quote": "The note has been sent."},
            ], "temporal_evidence": [],
        }]
        judgment["checks"]["fact_state_time"]["status"] = "HOLD"
        judgment["decision"] = "HOLD"
        with self.assertRaisesRegex(ValueError, "NATIVE_BASELINE_CITATION_NOT_EXACT"):
            self.review_save(reader, payload)

    def test_single_reader_cannot_finalize(self):
        self.authored()
        self.review_read(READERS[0])
        self.review_save(READERS[0])
        with self.assertRaises(FileNotFoundError):
            self.finalize()
        self.assertFalse((self.unit / "inputs/CONTINUATION_CONSTRAINTS.json").exists())

    def test_either_reviewer_hold_prevents_eligible_outputs(self):
        self.authored()
        self.reviewed(("READY", "HOLD"))
        self.finalize()
        self.assertEqual(read(self.folder / "FINALIZE_RECEIPT.json")["status"], "SOURCE_HOLD")
        self.assertEqual(read(self.folder / "SOURCE_HOLD.json")["decision"], "HOLD")
        self.assertFalse((self.unit / "inputs/BASELINE.json").exists())
        self.assertFalse((self.unit / "inputs/CONTINUATION_CONSTRAINTS.json").exists())

    def test_author_hold_cannot_be_overridden_by_ready_review(self):
        self.authored(self.author_payload(decision="HOLD"))
        reader = READERS[0]
        self.review_read(reader)
        with self.assertRaisesRegex(ValueError, "SOURCE_READY_WITH_BASELINE_OR_AUTHOR_HOLD"):
            self.review_save(reader)

    def test_author_and_reviewers_hold_are_recorded_without_eligible_outputs(self):
        self.authored(self.author_payload(decision="HOLD"))
        self.reviewed(("HOLD", "HOLD"))
        self.finalize()
        self.assertEqual(read(self.folder / "FINALIZE_RECEIPT.json")["status"], "SOURCE_HOLD")
        self.assertFalse((self.unit / "inputs/CONTINUATION_CONSTRAINTS.json").exists())

    def test_duplicate_review_save_rejected(self):
        self.authored()
        self.review_read(READERS[0])
        self.review_save(READERS[0])
        with self.assertRaisesRegex(ValueError, "SOURCE_FIRST_REVIEW_ALREADY_SAVED"):
            self.review_save(READERS[0])

    def test_duplicate_finalize_rejected(self):
        self.authored()
        self.reviewed()
        self.finalize()
        with self.assertRaisesRegex(ValueError, "SOURCE_FINALIZATION_ALREADY_ATTEMPTED_OR_OUTPUT_EXISTS"):
            self.finalize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
