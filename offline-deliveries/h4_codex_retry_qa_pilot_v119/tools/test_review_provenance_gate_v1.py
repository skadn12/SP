"""Synthetic metadata-chain tests, not semantic review or generation tests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import review_provenance_gate_v1 as gate
import run_guarded_questions_v1 as wrapper

ROOT = "cc-train-episode-1"
AUTHOR, REVIEWER = "/root/fixture_author", "/root/fixture_reviewer"
HERE = Path(__file__).resolve().parent


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixture(batch, phase="asset"):
    unit = batch / "units/root1"
    folder = unit / "roots/root1"
    helper = batch / "tools/review_v3.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(HERE / "review_v3.py", helper)
    role = {"root_id": ROOT, "author": AUTHOR, "reviewer": REVIEWER,
            "baseline_reviewers": ["/root/fixture_base1", "/root/fixture_base2"]}
    save(batch / "ROLE_ASSIGNMENTS.json", {"roots": [role]})
    save(batch / "MASTER_FREEZE.json", {"files": [gate.binding(helper), gate.binding(batch / "ROLE_ASSIGNMENTS.json")]})
    save(unit / "RUN_CONFIG.json", {"roots": [{"root_id": ROOT}]})
    save(unit / "inputs/BASELINE.json", {"synthetic_metadata_only": True})
    baseline = []
    for index, reviewer in enumerate(role["baseline_reviewers"], 1):
        p = unit / f"baseline_reviews/reader{index}/REVIEW.json"
        save(p, {"reviewer": reviewer, "synthetic_metadata_only": True})
        baseline.append({**gate.binding(p), "reviewer": reviewer})
    constraints = {"root_id": ROOT, "decision": "PASS", "continuation_constraints": ["Synthetic metadata constraint."],
                   "baseline_binding": gate.binding(unit / "inputs/BASELINE.json"), "review_bindings": baseline}
    save(unit / "inputs/CONTINUATION_CONSTRAINTS.json", constraints)
    source = {"root_id": ROOT, "turns": [{"turn_ref": "fixture:T001", "speaker": "Fixture", "text": "Synthetic source text."}]}
    save(folder / "CUMULATIVE_SOURCE.json", source)
    save(folder / "QA_AUTHOR_RECEIPT.json", {"author": AUTHOR, "source_sha256": gate.sha(folder / "CUMULATIVE_SOURCE.json")})
    for scene in gate.SCENES:
        save(folder / "scenes" / scene / "SOURCE_REVIEW.json", {"synthetic_metadata_only": True})
    save(folder / "PUBLIC_ASSETS.json", {"root_id": ROOT, "source_sha256": gate.sha(folder / "CUMULATIVE_SOURCE.json"), "synthetic_metadata_only": True})
    save(folder / "PUBLIC_QUESTIONS.json", {"questions": [{"root_id": ROOT, "source_sha256": gate.sha(folder / "CUMULATIVE_SOURCE.json"),
                                                          "source_path": str(folder / "CUMULATIVE_SOURCE.json"), "synthetic_metadata_only": True}]})
    build_record(batch, "asset", role, source, constraints)
    if phase == "public":
        build_record(batch, "public", role, source, constraints)
    return unit, folder


def paths(batch, phase):
    unit = batch / "units/root1"
    digest = hashlib.sha256(REVIEWER.encode()).hexdigest()[:16]
    directory = unit / "evidence/reads" / f"{phase}_root_{digest}"
    return unit / "roots/root1" / gate.OUTPUTS[phase], directory


def build_record(batch, phase, role, source, constraints):
    unit = batch / "units/root1"
    folder = unit / "roots/root1"
    review_path, directory = paths(batch, phase)
    source_path = folder / "CUMULATIVE_SOURCE.json"
    public_path = folder / ("PUBLIC_ASSETS.json" if phase == "asset" else "PUBLIC_QUESTIONS.json")
    helper = batch / "tools/review_v3.py"
    public = read(public_path)
    literals = gate.helper_literals(gate.Audit(), helper)
    lines = ["REVIEW_ASSIGNMENT", json.dumps({"root_id": ROOT, "phase": phase, "scene_ref": None, "executor_id": REVIEWER,
             "assigned_roles": role}, ensure_ascii=False), "ALL_CONTINUATION_CONSTRAINTS", json.dumps(constraints, ensure_ascii=False, indent=2), "FULL_CUMULATIVE_SOURCE"]
    lines += [t["turn_ref"] + " | " + t["speaker"] + ": " + t["text"] for t in source["turns"]]
    lines += ["PUBLIC_CONTENT_FULL", json.dumps(public, ensure_ascii=False, indent=2), "EVIDENCE_LIMITATIONS",
              literals["IDENTITY_LIMITATION"], literals["READ_LIMITATION"]]
    directory.mkdir(parents=True)
    (directory / "INPUT_RENDER.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    inputs = [batch / "ROLE_ASSIGNMENTS.json", unit / "RUN_CONFIG.json", unit / "inputs/CONTINUATION_CONSTRAINTS.json",
              source_path, helper, unit / "inputs/BASELINE.json", folder / "QA_AUTHOR_RECEIPT.json", public_path]
    inputs += [Path(row["path"]) for row in constraints["review_bindings"]]
    inputs += [folder / "scenes" / scene / "SOURCE_REVIEW.json" for scene in gate.SCENES]
    asset_binding = gate.binding(folder / gate.OUTPUTS["asset"]) if phase == "public" else None
    if phase == "public":
        inputs.append(folder / gate.OUTPUTS["asset"])
    context = {"schema_version": "codex_assigned_read_context_v2", "root_id": ROOT, "phase": phase,
               "scene_ref": None, "executor_id": REVIEWER, "started_at_utc": "2026-01-01T00:00:00+00:00",
               "source_binding": gate.binding(source_path), "public_binding": gate.binding(public_path),
               "constraints_binding": gate.binding(unit / "inputs/CONTINUATION_CONSTRAINTS.json"),
               "role_assignments_binding": gate.binding(batch / "ROLE_ASSIGNMENTS.json"),
               "input_render_binding": gate.binding(directory / "INPUT_RENDER.txt"), "input_bindings": [gate.binding(p) for p in inputs],
               "turn_refs_rendered": ["fixture:T001"], "constraint_count_rendered": 1,
               "identity_verification_scope": literals["IDENTITY_LIMITATION"], "read_attestation_scope": literals["READ_LIMITATION"],
               "native_event_binding": None, "selection_binding": None, "asset_judgments_binding": asset_binding}
    save(directory / "READ_CONTEXT.json", context)
    attestation = {"full_source_read": True, "all_continuation_constraints_read": True, "private_answers_read_before_save": False}
    if phase == "public":
        attestation["final_public_questions_actually_reread"] = True
    semantic = {"multiselect_stem_valid": True, "statements": {}, "ordering_questions": [], "findings": []} if phase == "asset" else {"questions": {}, "findings": []}
    # Empty semantic fields are intentional: these tests exercise metadata
    # provenance only and make no claim that the fixture is valid QA content.
    save(directory / "RAW_JUDGMENT.json", {"read_attestation": attestation, **semantic})
    review = {"schema_version": "codex_assigned_source_review_v2", "root_id": ROOT, "reviewer": REVIEWER,
              "review_role": gate.ROLES[phase], "backend": "CURRENT_CODEX_SESSION",
              "external_semantic_certification": False, "semantic_review_is_programmatic": False,
              "identity_verification_scope": literals["IDENTITY_LIMITATION"], "read_attestation_scope": literals["READ_LIMITATION"],
              "source_sha256": gate.sha(source_path), "public_sha256": gate.sha(public_path),
              "full_source_read": True, "all_continuation_constraints_read": True, "independent_of_question_author": True,
              "private_answers_read_before_save": False, "turn_refs_read": ["fixture:T001"],
              "helper_binding": gate.binding(helper), "read_context_binding": gate.binding(directory / "READ_CONTEXT.json"),
              "input_render_binding": gate.binding(directory / "INPUT_RENDER.txt"), "raw_judgment_binding": gate.binding(directory / "RAW_JUDGMENT.json"),
              "role_assignments_binding": gate.binding(batch / "ROLE_ASSIGNMENTS.json"), "constraints_binding": gate.binding(unit / "inputs/CONTINUATION_CONSTRAINTS.json"),
              "executor_read_attestation": attestation, "started_at_utc": context["started_at_utc"],
              "finished_at_utc": "2026-01-01T00:01:00+00:00", "created_at_utc": "2026-01-01T00:01:00+00:00", **semantic}
    if phase == "public":
        review.update(final_public_questions_actually_reread=True, asset_judgments_sha256=asset_binding["sha256"])
    save(review_path, review)
    save(directory / "SAVE_RECEIPT.json", {"status": "RECORDED_EXECUTOR_JUDGMENT_NOT_EXTERNAL_CERTIFICATION", "created_at_utc": "2026-01-01T00:01:01+00:00",
         "executor_id": REVIEWER, "root_id": ROOT, "phase": phase, "read_context_binding": gate.binding(directory / "READ_CONTEXT.json"),
         "raw_judgment_binding": gate.binding(directory / "RAW_JUDGMENT.json"), "output_binding": gate.binding(review_path),
         "semantic_judgment_supplied_by_executor": True, "private_file_access_proven_by_program": False, "internal_reading_proven_by_program": False})


def refresh_chain(batch, phase="asset"):
    review_path, directory = paths(batch, phase)
    review = read(review_path)
    for key, name in (("read_context_binding", "READ_CONTEXT.json"), ("raw_judgment_binding", "RAW_JUDGMENT.json"), ("input_render_binding", "INPUT_RENDER.txt")):
        review[key] = gate.binding(directory / name)
    save(review_path, review)
    receipt = read(directory / "SAVE_RECEIPT.json")
    receipt.update(read_context_binding=gate.binding(directory / "READ_CONTEXT.json"), raw_judgment_binding=gate.binding(directory / "RAW_JUDGMENT.json"), output_binding=gate.binding(review_path))
    save(directory / "SAVE_RECEIPT.json", receipt)


class MetadataGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name) / "synthetic_batch"

    def codes(self, phase="asset"):
        return {e["code"] for e in gate.check(self.batch, ROOT, phase)["errors"]}

    def test_valid_metadata_asset(self):
        fixture(self.batch)
        result = gate.check(self.batch, ROOT, "asset")
        self.assertTrue(result["pass"], result["errors"])
        self.assertFalse(result["semantic_quality_certified"])

    def test_valid_metadata_public_and_asset_lineage(self):
        fixture(self.batch, "public")
        result = gate.check(self.batch, ROOT, "public")
        self.assertTrue(result["pass"], result["errors"])

    def test_actual_reviewer_mismatch(self):
        fixture(self.batch)
        p, _ = paths(self.batch, "asset")
        obj = read(p); obj["reviewer"] = "/root/unassigned"; save(p, obj); refresh_chain(self.batch)
        self.assertIn("REVIEW:REVIEWER_NOT_FROZEN_ASSIGNEE", self.codes())

    def test_missing_context_is_blocked(self):
        fixture(self.batch)
        _, directory = paths(self.batch, "asset")
        (directory / "READ_CONTEXT.json").unlink()
        self.assertIn("READ_CONTEXT:FILE_MISSING", self.codes())

    def test_wrong_context_executor_even_with_refreshed_hashes(self):
        fixture(self.batch)
        _, directory = paths(self.batch, "asset")
        p = directory / "READ_CONTEXT.json"; obj = read(p); obj["executor_id"] = AUTHOR; save(p, obj); refresh_chain(self.batch)
        self.assertIn("READ_CONTEXT:FIELD_MISMATCH:executor_id", self.codes())

    def test_tampered_render_even_with_refreshed_hashes(self):
        fixture(self.batch)
        _, directory = paths(self.batch, "asset")
        p = directory / "INPUT_RENDER.txt"; p.write_text(p.read_text() + "omitted input\n")
        c = directory / "READ_CONTEXT.json"; obj = read(c); obj["input_render_binding"] = gate.binding(p); save(c, obj); refresh_chain(self.batch)
        self.assertIn("INPUT_RENDER:NOT_EXACT_CURRENT_INPUT_RENDER", self.codes())

    def test_source_first_attestation_required(self):
        fixture(self.batch)
        p, directory = paths(self.batch, "asset")
        raw_path = directory / "RAW_JUDGMENT.json"; raw = read(raw_path); raw["read_attestation"]["private_answers_read_before_save"] = True; save(raw_path, raw)
        review = read(p); review["executor_read_attestation"] = deepcopy(raw["read_attestation"]); save(p, review); refresh_chain(self.batch)
        self.assertIn("RAW_JUDGMENT:SOURCE_FIRST_ATTESTATION_REQUIRED", self.codes())

    def test_receipt_and_frozen_role_hash_required(self):
        fixture(self.batch)
        _, directory = paths(self.batch, "asset")
        p = directory / "SAVE_RECEIPT.json"; obj = read(p); obj["output_binding"]["sha256"] = "0" * 64; save(p, obj)
        self.assertIn("SAVE_RECEIPT:output_binding:HASH_MISMATCH", self.codes())
        p = self.batch / "ROLE_ASSIGNMENTS.json"; obj = read(p); obj["roots"][0]["reviewer"] = AUTHOR; save(p, obj)
        self.assertIn("FROZEN_ROLES:HASH_MISMATCH", self.codes())

    def test_public_cannot_hide_invalid_prior_asset(self):
        fixture(self.batch, "public")
        p, _ = paths(self.batch, "asset"); obj = read(p); obj["reviewer"] = "/root/unassigned"; save(p, obj)
        self.assertIn("PRIOR_ASSET_PROVENANCE:REVIEW:REVIEWER_NOT_FROZEN_ASSIGNEE", self.codes("public"))

    def test_failed_gate_never_invokes_downstream(self):
        fixture(self.batch)
        p, _ = paths(self.batch, "asset"); obj = read(p); obj["reviewer"] = "/root/unassigned"; save(p, obj)
        calls = []
        def downstream(*args):
            calls.append(args); raise AssertionError("downstream must never run")
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        result, code = wrapper.run(self.batch, ROOT, "compose", downstream=downstream)
        self.assertEqual(code, 2); self.assertEqual(calls, []); self.assertFalse(result["downstream_invoked"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_failed_public_gate_never_invokes_finalize(self):
        fixture(self.batch, "public")
        _, directory = paths(self.batch, "public"); (directory / "READ_CONTEXT.json").unlink()
        calls = []
        def downstream(*args):
            calls.append(args); raise AssertionError("finalize must never run")
        result, code = wrapper.run(self.batch, ROOT, "finalize", downstream=downstream)
        self.assertEqual(code, 2); self.assertEqual(calls, []); self.assertFalse(result["downstream_invoked"])

    def test_check_only_never_invokes_downstream(self):
        fixture(self.batch)
        calls = []
        def downstream(*args):
            calls.append(args); return SimpleNamespace(returncode=0, stdout="", stderr="")
        result, code = wrapper.run(self.batch, ROOT, "compose", check_only=True, downstream=downstream)
        self.assertEqual(code, 0); self.assertFalse(result["downstream_invoked"]); self.assertEqual(calls, [])
        result, code = wrapper.run(self.batch, ROOT, "compose", downstream=downstream)
        self.assertEqual(code, 0); self.assertTrue(result["downstream_invoked"]); self.assertEqual(len(calls), 1)

    def test_cli_blocked_is_nonzero_valid_json(self):
        fixture(self.batch)
        _, directory = paths(self.batch, "asset"); (directory / "READ_CONTEXT.json").unlink()
        completed = subprocess.run([sys.executable, "-B", str(HERE / "review_provenance_gate_v1.py"), "check", "--batch", str(self.batch), "--root", ROOT, "--phase", "asset"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2); self.assertFalse(json.loads(completed.stdout)["pass"])

    def test_cli_bad_arguments_are_nonzero_valid_json(self):
        completed = subprocess.run([sys.executable, "-B", str(HERE / "review_provenance_gate_v1.py"), "check"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["errors"][0]["code"], "CLI_ARGUMENT_ERROR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
