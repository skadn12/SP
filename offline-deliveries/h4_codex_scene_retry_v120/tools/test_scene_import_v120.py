"""Synthetic native-controller integration; never a formal semantic sample."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prefix_review_v1 as prefix
import scene_controller_v120 as adapter


ROOT = "cc-train-episode-1"
AUTHOR = "/root/fixture_author"
REVIEWER = "/root/fixture_reviewer"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def turns(scene, count):
    return [{"turn_ref": f"{scene}:T{i:03d}", "session_ref": scene, "scene_ref": scene,
             "speaker": "A" if i % 2 else "B", "text": f"Synthetic {scene} utterance {i}.",
             "origin": "codex_session_generated"} for i in range(1, count + 1)]


def config(baseline):
    return {"controller_version": adapter.native.VERSION, "batch_id": "SYNTHETIC_FIXTURE_ONLY",
            "roots": [{"root_id": ROOT, "baseline_path": str(baseline), "baseline_sha256": adapter.digest(baseline)}],
            "scene_sequence": list(adapter.SCENES), "new_turn_count": {"min": 12, "max": 16}}


def refresh_execution_freeze(unit):
    """Rebuild only a temporary synthetic fixture's post-matching bindings."""
    unit = Path(unit)
    master = unit.parents[1]
    placeholders = {"selection.json": {"root_id": ROOT}, "HISTORICAL_SELECTION.json": {"root_id": ROOT},
                    "inputs/CONTINUATION_CONSTRAINTS.json": {"root_id": ROOT},
                    "inputs/EVENT_POOL.json": {"events": []},
                    **{"inputs/B_S%02d_EVENT.json" % i: {"dialogue_id": "synthetic-%d" % i, "turns": []} for i in (1, 2, 3)},
                    "evidence/MATCHING/READ_CONTEXT.json": {"root_id": ROOT},
                    "evidence/MATCHING/MATCHING_PAYLOAD.json": {"synthetic": True}}
    for name, value in placeholders.items():
        if not (unit / name).exists():
            write(unit / name, value)
    render = unit / "evidence/MATCHING/FULL_INPUT.txt"
    if not render.exists():
        render.write_text("Synthetic fixture only.\n", encoding="utf-8")
    role = next(r for r in adapter.read(master / "ROLE_ASSIGNMENTS.json")["roots"] if r["root_id"] == ROOT)
    path = unit / "EXECUTION_INPUTS_FROZEN.json"
    document = {"root_id": ROOT, "status": "READY_FOR_FIXED_REMAINING_SCENES",
                "master_freeze_binding": adapter.binding(master / "MASTER_FREEZE.json"),
                "files": [adapter.binding(p) for p in adapter.execution_input_paths(unit)],
                "author": role["author"], "reviewer": role["reviewer"], "generation_started": False}
    path.write_text(json.dumps(document), encoding="utf-8")
    receipt = {"root_id": ROOT, "actual_executor": role["author"], "execution_freeze_binding": adapter.binding(path),
               "selection_binding": adapter.binding(unit / "selection.json"),
               "reservations_binding": adapter.binding(unit / "SOURCE_RESERVATIONS.json"),
               "raw_payload_binding": adapter.binding(unit / "evidence/MATCHING/MATCHING_PAYLOAD.json")}
    (unit / "evidence/MATCHING/SAVE_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")


def make_fixture(folder):
    """Exercise real native materialization and actual prefix read/save helper."""
    history = folder / "synthetic_history"
    baseline = history / "inputs/BASELINE.json"
    write(baseline, {"root_id": ROOT, "turns": turns("R01:S01", 2)})
    write(history / "RUN_CONFIG.json", config(baseline))
    write(history / "SOURCE_RESERVATIONS.json", {"allocation": []})
    old_scene = adapter.scene_folder(history, ROOT, adapter.SCENES[0])
    write(old_scene / "EXPANSION_DRAFT.json", {"root_id": ROOT, "scene_ref": adapter.SCENES[0],
          "author": "/root/historical_fixture_author", "input_source_sha256": adapter.digest(baseline),
          "turns": turns(adapter.SCENES[0], 12)})
    result = adapter.native.materialize_scene(history, ROOT, adapter.SCENES[0])
    assert result["status"] == "STRUCTURAL_PASS_ONLY", result
    source = adapter.read(old_scene / "CUMULATIVE_SOURCE.json")
    write(old_scene / "SOURCE_REVIEW.json", {"root_id": ROOT, "scene_ref": adapter.SCENES[0],
          "reviewer": "/root/historical_fixture_reviewer", "created_at_utc": prefix.now(),
          "source_sha256": adapter.digest(old_scene / "CUMULATIVE_SOURCE.json"),
          "full_source_read": True, "turn_refs_read": [t["turn_ref"] for t in source["turns"]],
          "decision": "READY", "checks": {k: {"status": "PASS", "reason": "Synthetic test answer, not a real semantic certification."} for k in adapter.CHECKS}, "findings": []})
    master = folder / "synthetic_batch"
    unit = master / "units/root1"
    write(master / "ROLE_ASSIGNMENTS.json", {"roots": [{"root_id": ROOT, "author": AUTHOR,
          "reviewer": REVIEWER, "prefix_reviewer": REVIEWER,
          "baseline_reviewers": ["/root/historical_fixture_r1", "/root/historical_fixture_r2"]}]})
    baseline_new = unit / "inputs/BASELINE.json"
    baseline_new.parent.mkdir(parents=True)
    shutil.copyfile(baseline, baseline_new)
    write(unit / "RUN_CONFIG.json", config(baseline_new))
    write(unit / "SOURCE_RESERVATIONS.json", {"allocation": []})
    constraints = master / "fixture_constraints.json"
    write(constraints, {"continuation_constraints": ["Synthetic fixture only; establish no real content judgment."]})
    candidate = old_scene / "CUMULATIVE_SOURCE.json"
    write(master / "PREFIX_INPUTS.json", {"role_assignments_binding": adapter.binding(master / "ROLE_ASSIGNMENTS.json"),
          "roots": [{"root_id": ROOT, "retained_scene_refs": [adapter.SCENES[0]],
                     "proposed_retry_start_scene": adapter.SCENES[1], "candidate_prefix_binding": adapter.binding(candidate),
                     "inputs": [{"label": "baseline", **adapter.binding(baseline)},
                                {"label": "constraints", **adapter.binding(constraints)},
                                {"label": "candidate_prefix", **adapter.binding(candidate)}]}]})
    args = SimpleNamespace(command="read", batch=str(master), root=ROOT, executor=REVIEWER)
    with redirect_stdout(io.StringIO()):
        prefix.run(args)
    audit_folder = master / "prefix_audits/root1"
    context = adapter.read(audit_folder / "READ_CONTEXT.json")
    raw = audit_folder / "RAW_JUDGMENT.json"
    write(raw, {"root_id": ROOT, "executor": REVIEWER, "decision": "PASS", "findings": [],
               "checks": {k: {"status": "PASS", "reason": "Synthetic fixture attestation only."} for k in prefix.CHECKS},
               "read_attestation": {**{k: True for k in ("full_baseline_read", "full_candidate_prefix_read", "all_constraints_read", "selection_read", "retained_native_events_read", "no_private_qa_or_prior_source_judgments_read")},
                                    "turn_refs_read": context["turn_refs_read"],
                                    "continuation_constraints_read": context["continuation_constraints"]}})
    args.command, args.judgment = "save", str(raw)
    with redirect_stdout(io.StringIO()):
        prefix.run(args)
    new_scene = adapter.scene_folder(unit, ROOT, adapter.SCENES[0])
    new_scene.mkdir(parents=True)
    names = ["CUMULATIVE_SOURCE.json", "EXPANSION_DRAFT.json", "SOURCE_REVIEW.json"]
    artifacts = {}
    for name in names:
        shutil.copyfile(old_scene / name, new_scene / name)
        artifacts[name] = {"historical_binding": adapter.binding(old_scene / name), "imported_binding": adapter.binding(new_scene / name)}
    write(unit / "PREFIX_IMPORT.json", {"schema_version": "historical_prefix_import_v1", "root_id": ROOT,
          "retained_scene_refs": [adapter.SCENES[0]], "role_assignments_binding": adapter.binding(master / "ROLE_ASSIGNMENTS.json"),
          "prefix_review_binding": adapter.binding(audit_folder / "PREFIX_SOURCE_REVIEW.json"),
          "prefix_read_receipt_binding": adapter.binding(audit_folder / "SAVE_RECEIPT.json"),
          "candidate_prefix_binding": adapter.binding(candidate),
          "scenes": [{"scene_ref": adapter.SCENES[0], "verbatim_artifacts": artifacts}]})
    tools = Path(adapter.__file__).parent
    files = [tools / name for name in ("scene_controller_v120.py", "question_controller_v120.py", "author_v120.py", "review_v4.py",
             "review_provenance_gate_v2.py", "scene_review_provenance_v1.py", "public_anchor_v1.py", "run_guarded_questions_v2.py")]
    files += [master / "ROLE_ASSIGNMENTS.json", unit / "PREFIX_IMPORT.json"]
    files += [adapter.MASTER / "scripts" / name for name in ("codex_sequential_ab.py", "codex_controlled_batch.py", "codex_sequential_questions.py", "codex_scene_payload.py")]
    write(master / "MASTER_FREEZE.json", {"status": "SYNTHETIC_FIXTURE_ONLY", "files": [adapter.binding(p) for p in files]})
    refresh_execution_freeze(unit)
    return master, unit, history


class HistoricalImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.master, self.unit, self.history = make_fixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_progress_preserves_historical_identities(self):
        state = adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)
        self.assertEqual(state["ready_scene_count"], 1)
        self.assertEqual(state["scenes"][0]["reviewer"], "/root/historical_fixture_reviewer")
        self.assertEqual(state["scenes"][1]["status"], "NOT_STARTED")
        self.assertFalse(state["full_expansion_complete"])
        self.assertFalse(list((adapter.scene_folder(self.unit, ROOT, adapter.SCENES[0]) / "receipts").glob("*.json")))

    def test_new_scene_real_native_materialization(self):
        scene = adapter.SCENES[1]
        prev = adapter.scene_folder(self.unit, ROOT, adapter.SCENES[0]) / "CUMULATIVE_SOURCE.json"
        write(adapter.scene_folder(self.unit, ROOT, scene) / "EXPANSION_DRAFT.json", {
              "root_id": ROOT, "scene_ref": scene, "author": AUTHOR,
              "input_source_sha256": adapter.digest(prev), "turns": turns(scene, 12)})
        out = adapter.materialize_scene(self.unit, ROOT, scene)
        self.assertEqual(out["status"], "STRUCTURAL_PASS_ONLY")
        source = adapter.read(out["source_path"])
        self.assertEqual(source["turns"][:14], adapter.read(prev)["turns"])
        state = adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)
        self.assertEqual(state["scenes"][1]["status"], "AWAITING_REVIEW")
        self.assertEqual(state["scenes"][2]["status"], "BLOCKED_BY_PREDECESSOR")

    def test_import_cannot_be_reauthored(self):
        with self.assertRaisesRegex(ValueError, "CANNOT_REAUTHOR_IMPORTED_SCENE"):
            adapter.materialize_scene(self.unit, ROOT, adapter.SCENES[0])

    def test_changed_import_rejected_before_controller(self):
        path = adapter.scene_folder(self.unit, ROOT, adapter.SCENES[0]) / "CUMULATIVE_SOURCE.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "PREFIX_BOUND_FILE_CHANGED"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_changed_prefix_judgment_rejected(self):
        path = self.master / "prefix_audits/root1/PREFIX_SOURCE_REVIEW.json"
        obj = adapter.read(path); obj["content_reuse_approved"] = False
        path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "PREFIX_BOUND_FILE_CHANGED"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_missing_save_receipt_rejected(self):
        (self.master / "prefix_audits/root1/SAVE_RECEIPT.json").unlink()
        with self.assertRaises(FileNotFoundError):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_wrong_new_author_rejected_before_materialization(self):
        scene = adapter.SCENES[1]
        prev = adapter.scene_folder(self.unit, ROOT, adapter.SCENES[0]) / "CUMULATIVE_SOURCE.json"
        write(adapter.scene_folder(self.unit, ROOT, scene) / "EXPANSION_DRAFT.json", {
              "root_id": ROOT, "scene_ref": scene, "author": REVIEWER,
              "input_source_sha256": adapter.digest(prev), "turns": turns(scene, 12)})
        with self.assertRaisesRegex(ValueError, "NEW_DRAFT_AUTHOR_NOT_ASSIGNED"):
            adapter.materialize_scene(self.unit, ROOT, scene)
        self.assertFalse((adapter.scene_folder(self.unit, ROOT, scene) / "CUMULATIVE_SOURCE.json").exists())

    def test_import_manifest_cannot_drop_artifact_bindings(self):
        path = self.unit / "PREFIX_IMPORT.json"
        obj = adapter.read(path); obj["scenes"][0]["verbatim_artifacts"] = {}
        path.write_text(json.dumps(obj))
        refresh_execution_freeze(self.unit)
        with self.assertRaisesRegex(ValueError, "PREFIX_IMPORT_NOT_FROZEN"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_incomplete_frozen_import_manifest_rejected(self):
        path = self.unit / "PREFIX_IMPORT.json"
        obj = adapter.read(path); obj["scenes"][0]["verbatim_artifacts"] = {}
        path.write_text(json.dumps(obj))
        freeze_path = self.master / "MASTER_FREEZE.json"
        freeze = adapter.read(freeze_path)
        for row in freeze["files"]:
            if row["path"] == str(path): row["sha256"] = adapter.digest(path)
        freeze_path.write_text(json.dumps(freeze))
        refresh_execution_freeze(self.unit)
        with self.assertRaisesRegex(ValueError, "IMPORTED_ARTIFACT_SET_INCOMPLETE"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_missing_execution_freeze_rejected_before_materialization(self):
        (self.unit / "EXECUTION_INPUTS_FROZEN.json").unlink()
        with self.assertRaisesRegex(ValueError, "EXECUTION_INPUTS_NOT_FROZEN"):
            adapter.materialize_scene(self.unit, ROOT, adapter.SCENES[1])
        self.assertFalse((adapter.scene_folder(self.unit, ROOT, adapter.SCENES[1]) / "CUMULATIVE_SOURCE.json").exists())

    def test_changed_execution_config_rejected_before_materialization(self):
        path = self.unit / "RUN_CONFIG.json"
        cfg = adapter.read(path); cfg["author_attempts_per_scene"] = 200
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "EXECUTION_INPUT_CHANGED"):
            adapter.materialize_scene(self.unit, ROOT, adapter.SCENES[1])
        self.assertFalse((adapter.scene_folder(self.unit, ROOT, adapter.SCENES[1]) / "CUMULATIVE_SOURCE.json").exists())

    def test_execution_freeze_cannot_omit_required_binding(self):
        path = self.unit / "EXECUTION_INPUTS_FROZEN.json"
        doc = adapter.read(path)
        doc["files"] = [r for r in doc["files"] if r["path"] != str(self.unit / "selection.json")]
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "EXECUTION_FREEZE_FILE_SET"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_execution_freeze_cannot_omit_required_input_file(self):
        (self.unit / "inputs/B_S02_EVENT.json").unlink()
        with self.assertRaisesRegex(ValueError, "EXECUTION_REQUIRED_INPUT_MISSING"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_execution_freeze_is_bound_to_matching_save_receipt(self):
        path = self.unit / "EXECUTION_INPUTS_FROZEN.json"
        doc = adapter.read(path); doc["created_at_utc"] = "2099-01-01T00:00:00Z"
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "EXECUTION_MATCHING_RECEIPT_CHANGED"):
            adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)

    def test_shadowed_gate_rejected_before_its_code_is_called(self):
        path = Path(self.tmp.name) / "foreign/review_provenance_gate_v2.py"
        path.parent.mkdir()
        path.write_text("called = False\ndef check_scene(*args):\n    global called\n    called = True\n    return {'pass': True}\n")
        spec = importlib.util.spec_from_file_location("review_provenance_gate_v2", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(sys.modules, {"review_provenance_gate_v2": module}):
            with self.assertRaisesRegex(ValueError, "FOREIGN_LOADED_RUNTIME:review_provenance_gate_v2"):
                adapter.inspect_root(self.unit, adapter.load_config(self.unit), ROOT)
        self.assertFalse(module.called)


if __name__ == "__main__":
    unittest.main()
