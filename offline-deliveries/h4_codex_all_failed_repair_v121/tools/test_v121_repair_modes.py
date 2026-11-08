"""Synthetic v121 repair-mode integration; no real-root semantic certification.

Run real prefix read/save, native historical materialization, author I/O,
anchor freeze and guarded QA stages in disposable fixtures.  Expected judgments
describe synthetic strings only; these tests never invoke a model or Solver.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest

import prepare_v120 as matching
import prefix_review_v1 as prefix
import scene_controller_v120 as scene
import test_native_retry_workflow_v120 as workflow
import test_prepare_v120 as matching_tests
from test_scene_import_v120 import AUTHOR, REVIEWER, ROOT, refresh_execution_freeze, turns


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class NativeRepairModes(unittest.TestCase):
    # Reuse fixture helpers, without inheriting or rerunning the existing suite.
    setUp = workflow.NativeRetryWorkflowTests.setUp
    call = workflow.NativeRetryWorkflowTests.call
    expand = workflow.NativeRetryWorkflowTests.expand
    scene_review = workflow.NativeRetryWorkflowTests.scene_review
    qa_payloads = workflow.NativeRetryWorkflowTests.qa_payloads
    statement_judgment = workflow.NativeRetryWorkflowTests.statement_judgment
    ordering_judgment = workflow.NativeRetryWorkflowTests.ordering_judgment
    qa_review = workflow.NativeRetryWorkflowTests.qa_review

    def blocked(self, tool, message, *args):
        result = subprocess.run([sys.executable, "-B", str(self.tools / tool), *map(str, args)],
                                cwd=self.batch, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def install_prefix(self, count, decision="PASS"):
        """Create real native historical artifacts, then import their exact bytes."""
        self.assertIn(count, (0, 6))
        baseline = self.unit / "inputs/BASELINE.json"
        if count:
            history = self.batch.parent / "synthetic_complete_history"
            old_baseline = history / "inputs/BASELINE.json"
            old_baseline.parent.mkdir(parents=True)
            shutil.copyfile(baseline, old_baseline)
            cfg = read(self.unit / "RUN_CONFIG.json")
            cfg["roots"][0].update(baseline_path=str(old_baseline), baseline_sha256=scene.digest(old_baseline))
            write(history / "RUN_CONFIG.json", cfg)
            write(history / "SOURCE_RESERVATIONS.json", read(self.unit / "SOURCE_RESERVATIONS.json"))
            for index, ref in enumerate(scene.SCENES):
                folder = scene.scene_folder(history, ROOT, ref)
                previous = old_baseline if not index else scene.scene_folder(history, ROOT, scene.SCENES[index - 1]) / "CUMULATIVE_SOURCE.json"
                draft = {"root_id": ROOT, "scene_ref": ref, "author": "/root/historical_fixture_author",
                         "input_source_sha256": scene.digest(previous), "turns": turns(ref, 12)}
                if ":B:" in ref:
                    draft["source_id"] = next(a["source_id"] for a in read(history / "SOURCE_RESERVATIONS.json")["allocation"] if a["scene_ref"] == ref)
                write(folder / "EXPANSION_DRAFT.json", draft)
                result = scene.native.materialize_scene(history, ROOT, ref)
                self.assertEqual(result["status"], "STRUCTURAL_PASS_ONLY", result)
                source = read(folder / "CUMULATIVE_SOURCE.json")
                write(folder / "SOURCE_REVIEW.json", {"root_id": ROOT, "scene_ref": ref,
                      "reviewer": "/root/historical_fixture_reviewer", "created_at_utc": prefix.now(),
                      "source_sha256": scene.digest(folder / "CUMULATIVE_SOURCE.json"),
                      "full_source_read": True, "turn_refs_read": [t["turn_ref"] for t in source["turns"]],
                      "decision": "READY", "findings": [],
                      "checks": {k: {"status": "PASS", "reason": "Synthetic fixture expectation only."} for k in scene.CHECKS}})
            state = scene.native.inspect_root(history, cfg, ROOT)
            self.assertTrue(state["full_expansion_complete"], state)
            self.history = history
            candidate = scene.scene_folder(history, ROOT, scene.SCENES[-1]) / "CUMULATIVE_SOURCE.json"
        else:
            candidate = baseline

        shutil.rmtree(self.folder)
        shutil.rmtree(self.batch / "prefix_audits")
        retained = list(scene.SCENES[:count])
        constraints = self.unit / "inputs/CONTINUATION_CONSTRAINTS.json"
        write(self.batch / "PREFIX_INPUTS.json", {"role_assignments_binding": scene.binding(self.batch / "ROLE_ASSIGNMENTS.json"),
              "roots": [{"root_id": ROOT, "retained_scene_refs": retained,
                         "proposed_retry_start_scene": "QA_ONLY" if count == 6 else scene.SCENES[0],
                         "candidate_prefix_binding": scene.binding(candidate),
                         "inputs": [{"label": "baseline", **scene.binding(baseline)},
                                    {"label": "constraints", **scene.binding(constraints)},
                                    {"label": "candidate_prefix", **scene.binding(candidate)}]}]})
        args = SimpleNamespace(command="read", batch=str(self.batch), root=ROOT, executor=REVIEWER)
        with redirect_stdout(io.StringIO()):
            prefix.run(args)
        audit = self.batch / "prefix_audits/root1"
        context = read(audit / "READ_CONTEXT.json")
        raw = {"root_id": ROOT, "executor": REVIEWER, "decision": decision, "findings": [],
               "checks": {k: {"status": "PASS", "reason": "Synthetic fixture only."} for k in prefix.CHECKS},
               "read_attestation": {**{k: True for k in ("full_baseline_read", "full_candidate_prefix_read", "all_constraints_read",
                    "selection_read", "retained_native_events_read", "no_private_qa_or_prior_source_judgments_read")},
                    "turn_refs_read": context["turn_refs_read"], "continuation_constraints_read": context["continuation_constraints"]}}
        if decision == "HOLD":
            raw["checks"]["prefix_boundary"] = {"status": "FAIL", "reason": "Synthetic blocking boundary."}
            raw["findings"] = [{"severity": "BLOCKING", "reason": "Synthetic rejection."}]
        write(audit / "RAW_JUDGMENT.json", raw)
        args.command, args.judgment = "save", str(audit / "RAW_JUDGMENT.json")
        with redirect_stdout(io.StringIO()):
            prefix.run(args)
        imported = []
        for ref in retained:
            old, new = scene.scene_folder(self.history, ROOT, ref), scene.scene_folder(self.unit, ROOT, ref)
            new.mkdir(parents=True)
            artifacts = {}
            for name in ("CUMULATIVE_SOURCE.json", "EXPANSION_DRAFT.json", "SOURCE_REVIEW.json"):
                shutil.copyfile(old / name, new / name)
                artifacts[name] = {"historical_binding": scene.binding(old / name), "imported_binding": scene.binding(new / name)}
            imported.append({"scene_ref": ref, "verbatim_artifacts": artifacts})
        write(self.unit / "PREFIX_IMPORT.json", {"schema_version": "historical_prefix_import_v1", "root_id": ROOT,
              "retained_scene_refs": retained, "role_assignments_binding": scene.binding(self.batch / "ROLE_ASSIGNMENTS.json"),
              "prefix_review_binding": scene.binding(audit / "PREFIX_SOURCE_REVIEW.json"),
              "prefix_read_receipt_binding": scene.binding(audit / "SAVE_RECEIPT.json"),
              "candidate_prefix_binding": scene.binding(candidate), "scenes": imported})
        frozen = [Path(r["path"]) for r in read(self.batch / "MASTER_FREEZE.json")["files"]]
        write(self.batch / "MASTER_FREEZE.json", {"status": "SYNTHETIC_FIXTURE_ONLY", "files": [scene.binding(p) for p in frozen]})
        refresh_execution_freeze(self.unit)

    def test_six_prefix_qa_only_full_guarded_chain(self):
        self.install_prefix(6)
        state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
        self.assertTrue(state["full_expansion_complete"], state)
        self.assertEqual(state["ready_scene_count"], 6)
        self.call("author_v120.py", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        plan, draft = self.qa_payloads()
        self.call("author_v120.py", "anchor-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--payload", plan)
        self.call("author_v120.py", "qa-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--payload", draft)
        self.call("question_controller_v120.py", "publish", "--batch-dir", self.unit)
        self.qa_review("asset")
        compose = self.call("run_guarded_questions_v2.py", "compose", "--batch", self.batch, "--root", ROOT, json_output=True)
        self.assertEqual(compose["status"], "GUARDED_STAGE_COMPLETED")
        self.qa_review("public")
        final = self.call("run_guarded_questions_v2.py", "finalize", "--batch", self.batch, "--root", ROOT, json_output=True)
        self.assertEqual(final["status"], "GUARDED_STAGE_COMPLETED")
        crosscheck = read(self.folder / "FINAL_CROSSCHECK.json")
        self.assertEqual(len(crosscheck["results"]), 7)
        self.assertTrue(all(r["status"] == "CONTENT_PASS" for r in crosscheck["results"]), crosscheck)
        questions = {q["candidate_id"]: q for q in read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]}
        gold = read(self.folder / "PRIVATE_GOLD.json")
        self.assertEqual(sorted(len(q["gold"]) for q in gold["questions"] if questions[q["candidate_id"]]["question_type"] == "multi_select"), [2, 3, 4, 5, 6])
        for ref in scene.SCENES:
            current, old = scene.scene_folder(self.unit, ROOT, ref), scene.scene_folder(self.history, ROOT, ref)
            for name in ("CUMULATIVE_SOURCE.json", "EXPANSION_DRAFT.json", "SOURCE_REVIEW.json"):
                self.assertEqual((current / name).read_bytes(), (old / name).read_bytes())
            self.assertFalse(list((current / "receipts").glob("*.json")))
        self.assertFalse(any(r["tool"] == "author_v120.py" and r["args"][0] == "scene-save" for r in self.stages))

    def test_zero_prefix_reaches_first_scene_and_preserves_qa_block(self):
        self.install_prefix(0)
        state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
        self.assertEqual(state["ready_scene_count"], 0)
        self.assertEqual(state["scenes"][0]["status"], "NOT_STARTED")
        self.blocked("author_v120.py", "ALL_SIX_SCENES_MUST_BE_READY", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        self.assertFalse((self.unit / "evidence/AUTHOR_QA_READ_CONTEXT.json").exists())
        self.expand([scene.SCENES[0]])

    def test_six_prefix_hold_blocks_qa_read(self):
        self.install_prefix(6, decision="HOLD")
        self.blocked("author_v120.py", "PREFIX_NOT_APPROVED", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        self.assertFalse((self.unit / "evidence/AUTHOR_QA_READ_CONTEXT.json").exists())
        self.assertFalse((self.folder / "QA_DRAFT.json").exists())

    def test_eligible_prefix_cannot_override_baseline_hold(self):
        self.install_prefix(6)
        path = self.unit / "inputs/CONTINUATION_CONSTRAINTS.json"
        doc = read(path); doc["decision"] = "HOLD"; write(path, doc)
        refresh_execution_freeze(self.unit)
        self.blocked("author_v120.py", "BASELINE_NOT_ELIGIBLE", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        self.assertFalse((self.unit / "evidence/AUTHOR_QA_READ_CONTEXT.json").exists())

    def test_six_prefix_frozen_artifact_tamper_blocks_qa(self):
        self.install_prefix(6)
        path = scene.scene_folder(self.unit, ROOT, scene.SCENES[4]) / "CUMULATIVE_SOURCE.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.blocked("author_v120.py", "PREFIX_BOUND_FILE_CHANGED", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        self.assertFalse((self.unit / "evidence/AUTHOR_QA_READ_CONTEXT.json").exists())

    def test_six_prefix_cannot_reauthor_any_imported_scene(self):
        self.install_prefix(6)
        for ref in (scene.SCENES[0], scene.SCENES[-1]):
            self.blocked("scene_controller_v120.py", "CANNOT_REAUTHOR_IMPORTED_SCENE", "materialize-scene", "--batch-dir", self.unit,
                         "--root-id", ROOT, "--scene-ref", ref)
        self.assertFalse(list(self.folder.glob("scenes/*/receipts/*.json")))


class QAOnlyMatching(unittest.TestCase):
    tearDown = matching_tests.RealMatchingChain.tearDown
    save_match = matching_tests.RealMatchingChain.save_match

    def setUp(self):
        matching_tests.RealMatchingChain.setUp(self)
        shutil.rmtree(self.unit / "evidence/MATCHING")
        oldunit = matching.B / "history"
        allocations = []
        self.old_bytes = {}
        for index, ref in enumerate(matching.SCENES[3:]):
            event = oldunit / "inputs" / f"B_S{index + 1:02d}_EVENT.json"
            write(event, {"dialogue_id": f"retained{index}.json", "turns": [{"turn_id": "0", "utterance": "Synthetic retained event only."}]})
            allocations.append({"root_id": self.root, "scene_ref": ref, "source_id": f"multiwoz22:train:retained{index}.json",
                                "source_item_path": str(event), "source_item_sha256": matching.sha(event)})
            self.old_bytes[event] = event.read_bytes()
        path = oldunit / "SOURCE_RESERVATIONS.json"
        write(path, {"allocation": allocations}); self.old_bytes[path] = path.read_bytes()
        self.row.update(retained_scene_refs=list(matching.SCENES), new_b_scene_refs=[], retry_start="QA_ONLY")
        write(matching.B / "MANIFEST.json", {"roots": [self.row]})
        write(matching.B / "evidence/SOURCE_POOL_FREEZE.json", {"consumed_source_ids": [a["source_id"] for a in allocations]})
        raw = read(self.payload); raw["allocations"] = []; write(self.payload, raw)
        frozen = matching.B / "MASTER_FREEZE.json"
        doc = read(frozen); doc["files"] = [matching.bind(r["path"]) for r in doc["files"]]; write(frozen, doc)
        with redirect_stdout(io.StringIO()):
            matching.match_read(self.root, self.author)

    def test_all_six_retained_matching_uses_zero_new_events(self):
        self.save_match()
        reservations = read(self.unit / "SOURCE_RESERVATIONS.json")
        self.assertEqual(reservations["new_consumption_count"], 0)
        self.assertEqual(len(reservations["allocation"]), 3)
        self.assertTrue(all(a["status"] == "HISTORICAL_PREFIX_REUSE_NO_NEW_CONSUMPTION" for a in reservations["allocation"]))
        receipt = read(self.unit / "evidence/MATCHING/SAVE_RECEIPT.json")
        self.assertEqual(receipt["new_source_ids_consumed"], [])
        self.assertEqual(len(receipt["retained_source_ids_reused"]), 3)
        self.assertEqual(read(self.unit / "RUN_CONFIG.json")["retry_start"], "QA_ONLY")
        for path, expected in self.old_bytes.items():
            self.assertEqual(path.read_bytes(), expected)

    def test_qa_only_cannot_allocate_any_new_slot(self):
        raw = read(self.payload)
        raw["allocations"] = [{"scene_ref": matching.SCENES[-1], "source_id": "multiwoz22:train:event0.json",
                               "compatibility_reason": "Synthetic forbidden allocation.", "main_purpose": "Synthetic only."}]
        write(self.payload, raw)
        with self.assertRaisesRegex(ValueError, "ONLY_EXACT_NEW_B_SLOTS_REQUIRED"):
            self.save_match()
        self.assertFalse((self.unit / "SOURCE_RESERVATIONS.json").exists())


class NewSelectionMatching(unittest.TestCase):
    tearDown = matching_tests.RealMatchingChain.tearDown
    save_match = matching_tests.RealMatchingChain.save_match

    def setUp(self):
        matching_tests.RealMatchingChain.setUp(self)
        shutil.rmtree(self.unit / "evidence/MATCHING")
        base = self.unit / "inputs/BASELINE.json"
        source = {"root_id": self.root, "turns": [
            {"turn_ref": f"R01:S{i:02d}:T001", "scene_ref": f"R01:S{i:02d}", "speaker": "A",
             "text": f"Synthetic baseline thread {i}."} for i in (1, 2, 3)]}
        write(base, source)
        cp = self.unit / "inputs/CONTINUATION_CONSTRAINTS.json"
        constraints = read(cp); constraints["baseline_binding"] = matching.bind(base); write(cp, constraints)
        write(self.unit / "A_PLAN_POLICY.json", {"plan_mode": "NEW_SELECTION", "a_continuations": []})
        self.row.update(a_plan_mode="NEW_SELECTION", baseline_binding=matching.bind(base), candidate_prefix_binding=matching.bind(base))
        write(matching.B / "MANIFEST.json", {"roots": [self.row]})
        raw = read(self.payload)
        raw["a_continuations"] = [{"scene_ref": ref, "baseline_scene_ref": f"R01:S{i:02d}", "thread": f"Synthetic continuation {i}.",
            "evidence": [{"turn_ref": source["turns"][i - 1]["turn_ref"], "speaker": "A", "quote": source["turns"][i - 1]["text"]}]}
            for i, ref in enumerate(matching.SCENES[:3], 1)]
        write(self.payload, raw)
        self.refreeze()
        with redirect_stdout(io.StringIO()):
            matching.match_read(self.root, self.author)

    def refreeze(self):
        path = matching.B / "MASTER_FREEZE.json"
        freeze = read(path)
        freeze["files"] = [matching.bind(r["path"]) for r in freeze["files"]]
        write(path, freeze)

    def validate(self, raw, source=True):
        return matching.validate_payload(raw, self.row, read(self.unit / "A_PLAN_POLICY.json"),
             {f"multiwoz22:train:event{i}.json": {} for i in range(6)}, set(),
             read(self.unit / "inputs/BASELINE.json") if source else None)

    def test_new_selection_real_read_save_and_policy_filename(self):
        self.save_match()
        self.assertEqual(read(self.unit / "selection.json")["a_plan_mode"], "NEW_SELECTION")
        self.assertEqual(read(self.unit / "SOURCE_RESERVATIONS.json")["new_consumption_count"], 3)
        frozen = read(self.unit / "EXECUTION_INPUTS_FROZEN.json")
        self.assertIn(str(self.unit / "A_PLAN_POLICY.json"), [r["path"] for r in frozen["files"]])
        self.assertFalse((self.unit / "HISTORICAL_SELECTION.json").exists())

    def test_new_selection_requires_full_source(self):
        with self.assertRaisesRegex(ValueError, "NEW_SELECTION_FULL_SOURCE_REQUIRED"):
            self.validate(read(self.payload), source=False)

    def test_three_r01_threads_cannot_be_permuted(self):
        raw = read(self.payload)
        raw["a_continuations"][0]["baseline_scene_ref"] = "R01:S02"
        with self.assertRaisesRegex(ValueError, "NEW_A_FIXED_R01_ALIGNMENT"):
            self.validate(raw)

    def test_citation_must_include_aligned_r01_scene(self):
        raw = read(self.payload)
        raw["a_continuations"][0]["evidence"] = copy.deepcopy(raw["a_continuations"][1]["evidence"])
        with self.assertRaisesRegex(ValueError, "NEW_A_FIXED_R01_EVIDENCE"):
            self.validate(raw)

    def test_new_selection_citation_speaker_and_quote_are_source_bound(self):
        for field, value in (("speaker", "B"), ("quote", "Invented sentence.")):
            with self.subTest(field=field):
                raw = read(self.payload); raw["a_continuations"][0]["evidence"][0][field] = value
                with self.assertRaisesRegex(ValueError, "NEW_A_CITATION_MISMATCH"):
                    self.validate(raw)

    def test_new_selection_rejects_unknown_turn_and_empty_evidence(self):
        raw = read(self.payload); raw["a_continuations"][0]["evidence"][0]["turn_ref"] = "R01:S99:T001"
        with self.assertRaisesRegex(ValueError, "NEW_A_CITATION_FIELDS"):
            self.validate(raw)
        raw = read(self.payload); raw["a_continuations"][0]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "NEW_A_EVIDENCE_REQUIRED"):
            self.validate(raw)

    def test_new_selection_rejects_extra_answer_field_and_missing_slot(self):
        raw = read(self.payload); raw["a_continuations"][0]["gold"] = ["A"]
        with self.assertRaisesRegex(ValueError, "NEW_A_PLAN_FIELDS"):
            self.validate(raw)
        raw = read(self.payload); raw["a_continuations"].pop()
        with self.assertRaisesRegex(ValueError, "NEW_A_PLAN_SCENE_SCOPE"):
            self.validate(raw)

    def test_baseline_mutation_after_match_read_is_rejected_before_output(self):
        base = self.unit / "inputs/BASELINE.json"
        base.write_bytes(base.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "FROZEN_INPUT_CHANGED"):
            self.save_match()
        self.assertFalse((self.unit / "selection.json").exists())
        self.assertFalse((self.unit / "SOURCE_RESERVATIONS.json").exists())

    def test_hold_manifest_cannot_enter_matching(self):
        self.row["prefix_reuse_approved"] = False
        write(matching.B / "MANIFEST.json", {"roots": [self.row]}); self.refreeze()
        with self.assertRaisesRegex(ValueError, "PREFIX_HOLD_NO_RETRY"):
            self.save_match()
        self.assertFalse((self.unit / "selection.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
