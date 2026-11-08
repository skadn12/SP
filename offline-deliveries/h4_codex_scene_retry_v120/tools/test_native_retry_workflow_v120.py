"""Full synthetic native retry integration; no mocks and no formal samples.

Reuse actual historical-A1 materialization and prefix read/save, then execute
new author read/save, native A2..B3 materialization, assigned source reviews,
scene gate, native QA publish, asset/public reviews, guarded compose/finalize.
Semantic judgments are known synthetic fixture expectations, not certification
of any real dialogue. All temporary writes stay beneath this tools directory.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import public_anchor_v1 as anchors
from test_scene_import_v120 import make_fixture, refresh_execution_freeze, ROOT, AUTHOR, REVIEWER

HERE = Path(__file__).resolve().parent


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path):
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class NativeRetryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic_native_retry_", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        with redirect_stdout(io.StringIO()):
            self.batch, self.unit, self.history = make_fixture(Path(self.tmp.name))
        self.folder = self.unit / "roots/root1"
        self.tools = self.batch / "tools"
        self.tools.mkdir()
        self.stages = []
        tool_names = ["author_v120.py", "review_v4.py", "public_anchor_v1.py", "review_provenance_gate_v2.py",
                      "scene_review_provenance_v1.py", "run_guarded_questions_v2.py", "scene_controller_v120.py",
                      "question_controller_v120.py"]
        for name in tool_names:
            shutil.copy2(HERE / name, self.tools / name)
        shutil.copytree(HERE.parent / "scripts", self.batch / "scripts")
        cfg = read(self.unit / "RUN_CONFIG.json")
        cfg.update(scope_scene_refs=list(anchors.SCENES), multiselect_target_cardinalities=[2, 3, 4, 5],
                   seed_namespace="synthetic_native_retry_only", gold6_control_per_ready_root=1, ordering_slots_per_ready_root=2)
        write(self.unit / "RUN_CONFIG.json", cfg)
        reviews = []
        for i, reviewer in enumerate(["/root/historical_fixture_r1", "/root/historical_fixture_r2"], 1):
            path = self.unit / f"baseline_reviews/reader{i}/REVIEW.json"
            write(path, {"reviewer": reviewer, "synthetic_fixture_only": True})
            reviews.append({**binding(path), "reviewer": reviewer})
        write(self.unit / "inputs/CONTINUATION_CONSTRAINTS.json", {"root_id": ROOT, "decision": "PASS",
              "continuation_constraints": ["Synthetic fixture only; establish no real content judgment."],
              "baseline_binding": binding(self.unit / "inputs/BASELINE.json"), "review_bindings": reviews})
        selection_path = self.unit / "selection.json"
        allocations, reservations = [], []
        for i, scene in enumerate(anchors.SCENES[3:], 1):
            source_id = f"synthetic-event-{i}"
            event = self.unit / "inputs" / (scene.removeprefix("R02:").replace(":", "_") + "_EVENT.json")
            write(event, {"synthetic_fixture_only": True, "source_id": source_id,
                         "turns": [{"turn_id": f"native-{i}-1", "speaker": "A", "text": f"Synthetic native event {i} begins."},
                                   {"turn_id": f"native-{i}-2", "speaker": "B", "text": f"Synthetic native event {i} is completed."}]})
            allocations.append({"root_id": ROOT, "scene_ref": scene, "source_id": source_id})
            reservations.append({"root_id": ROOT, "scene_ref": scene, "source_id": source_id,
                                 "source_item_path": str(event), "source_item_sha256": binding(event)["sha256"],
                                 "selection_path": str(selection_path)})
        write(selection_path, {"synthetic_fixture_only": True, "root_id": ROOT, "allocations": allocations,
              "a_continuations": [{"scene_ref": scene, "baseline_scene_ref": "R01:S01", "thread": "Synthetic thread.",
                                   "evidence": ["Synthetic evidence."]} for scene in anchors.SCENES[:3]]})
        for row in reservations:
            row["selection_sha256"] = binding(selection_path)["sha256"]
        write(self.unit / "SOURCE_RESERVATIONS.json", {"allocation": reservations})
        runtime = [self.batch / "ROLE_ASSIGNMENTS.json", self.unit / "PREFIX_IMPORT.json"] + [self.tools / n for n in tool_names]
        runtime.extend(p for p in (self.batch / "scripts").glob("*.py"))
        write(self.batch / "MASTER_FREEZE.json", {"status": "SYNTHETIC_FIXTURE_ONLY", "files": [binding(p) for p in runtime]})
        refresh_execution_freeze(self.unit)

    def call(self, tool, *args, json_output=False, expect=0):
        completed = subprocess.run([sys.executable, "-B", str(self.tools / tool), *map(str, args)],
                                   cwd=self.batch, capture_output=True, text=True)
        self.stages.append({"tool": tool, "args": list(map(str, args)), "returncode": completed.returncode})
        self.assertEqual(completed.returncode, expect, completed.stdout + "\n" + completed.stderr)
        return json.loads(completed.stdout) if json_output else completed.stdout

    def scene_review(self, scene):
        target = self.folder / "scenes" / scene.removeprefix("R02:").replace(":", "_")
        source = read(target / "CUMULATIVE_SOURCE.json")
        raw = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True},
               "decision": "READY", "checks": {name: {"status": "PASS", "reason": "Known synthetic fixture expectation only."}
                                                for name in ("time", "location", "actor_identity", "plan_vs_completion", "cross_scene_continuity",
                                                             "semantic_repetition", "naturalness", "event_fidelity")},
               "findings": [], "baseline_constraint_review": {"status": "PASS", "reason": "Synthetic expected judgment.",
                   "constraint_results": [{"index": 1, "constraint": "Synthetic fixture only; establish no real content judgment.",
                                           "status": "PASS", "reason": "Fixture only.", "evidence_turn_refs": [source["turns"][-1]["turn_ref"]]}],
                   "new_blocking_findings": []}}
        if ":B:" in scene:
            raw["read_attestation"]["native_event_full_read"] = True
        judgment = self.unit / f"synthetic_{scene.replace(':', '_')}_review.json"; write(judgment, raw)
        base = [ROOT, "scene-read", "--batch", self.batch, "--executor", REVIEWER, "--scene", scene]
        self.call("review_v4.py", *base)
        base[1] = "scene-save"
        self.call("review_v4.py", *base, "--judgment", judgment)
        result = self.call("review_provenance_gate_v2.py", "check", "--batch", self.batch, "--root", ROOT,
                           "--phase", "scene", "--scene", scene, json_output=True)
        self.assertTrue(result["pass"], result["errors"])

    def expand(self, scenes):
        for scene in scenes:
            self.call("author_v120.py", "scene-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--scene", scene)
            messages = [{"speaker": "A" if i % 2 else "B", "text": f"Synthetic {scene} utterance {i}."} for i in range(1, 13)]
            payload = self.unit / f"synthetic_{scene.replace(':', '_')}_payload.json"
            write(payload, {"turns": messages})
            self.call("author_v120.py", "scene-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR,
                      "--scene", scene, "--payload", payload)
            self.scene_review(scene)
            state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
            index = anchors.SCENES.index(scene)
            self.assertEqual(state["scenes"][index]["status"], "READY", state)
            self.assertEqual(state["ready_scene_count"], index + 1)

    def qa_payloads(self):
        source = read(self.folder / "CUMULATIVE_SOURCE.json")
        turns = {row["turn_ref"]: row for row in source["turns"]}
        source_sha = binding(self.folder / "CUMULATIVE_SOURCE.json")["sha256"]
        cfg = read(self.unit / "RUN_CONFIG.json")
        plan = {"schema_version": anchors.SCHEMA, "root_id": ROOT, "source_sha256": source_sha,
                "scope_scene_refs": list(anchors.SCENES), "slots": []}
        draft = {"stem": "Select the correctly attributed exact utterances within these six scenes: " + ", ".join(anchors.SCENES),
                 "facts": [], "ordering": []}
        self.statement_expected = {}
        for i, scene in enumerate(anchors.SCENES, 1):
            turn = turns[scene + ":T003"]
            true = f'In {scene}, speaker A says "{turn["text"]}".'
            false = f'In {scene}, speaker B says "{turn["text"]}".'
            draft["facts"].append({"fact_id": f"synthetic-f{i}", "source_turn_ref": turn["turn_ref"], "true_text": true, "false_text": false})
            self.statement_expected[true] = ("supports", turn)
            self.statement_expected[false] = ("contradicts", turn)
        for slot_i, turn_i in ((1, 4), (2, 8)):
            slot = {"slot_id": str(slot_i), "candidate_id": cfg["batch_id"] + "-root1-ordering" + str(slot_i), "anchors": []}
            ordering = {"slot_id": str(slot_i), "items": []}
            for i, scene in enumerate(anchors.SCENES):
                turn = turns[scene + f":T{turn_i:03d}"]
                anchor = {"item_id": f"I{(i + slot_i) % 6 + 1}", "anchor_id": f"synthetic-anchor-{slot_i}-{i}",
                          "turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "full_turn_quote": turn["text"],
                          "full_turn_sha256": anchors.text_sha(turn["text"])}
                slot["anchors"].append(anchor)
                ordering["items"].append({"item_id": anchor["item_id"], "anchor_id": anchor["anchor_id"],
                                          "text": turn["text"], "support_quote": turn["text"]})
            plan["slots"].append(slot); draft["ordering"].append(ordering)
        self.plan = plan
        plan_path, draft_path = self.unit / "synthetic_anchor_payload.json", self.unit / "synthetic_qa_payload.json"
        write(plan_path, plan); write(draft_path, draft)
        return plan_path, draft_path

    def statement_judgment(self, text):
        relation, turn = self.statement_expected[text]
        return {"proposition": text, "relation": relation, "scope_valid": True, "fact_distinct": True,
                "reason": "Synthetic known speaker attribution expectation.",
                "citations": [{"turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": turn["text"]}]}

    def ordering_judgment(self, question):
        slot = next(s for s in self.plan["slots"] if s["candidate_id"] == question["candidate_id"])
        by_item = {a["item_id"]: a for a in slot["anchors"]}
        result = {"candidate_id": question["candidate_id"], **{k: True for k in ("stem_valid", "scope_valid", "coherent", "unique_order", "neutral_stem_ok")},
                  "reason": "Exact synthetic events in distinct turns.", "derived_order": [a["item_id"] for a in slot["anchors"]],
                  "alternative_orders": [], "items": {}}
        for label, text in question["items"].items():
            a = by_item[label]
            result["items"][label] = {"proposition": text, **{k: True for k in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok", "single_event_ok")},
                "reason": "Synthetic exact event.", "anchor_id": a["anchor_id"], "frozen_full_turn_sha256": a["full_turn_sha256"],
                "order_anchor": {"turn_ref": a["turn_ref"], "speaker": a["speaker"], "quote": a["full_turn_quote"]},
                "support_refs": [a["turn_ref"]], "alternative_matching_anchors": []}
        return result

    def qa_review(self, phase):
        attest = {"full_source_read": True, "all_continuation_constraints_read": True, "private_answers_read_before_save": False,
                  "public_anchor_plan_full_read": True, "source_read_before_anchor_plan_and_public_content": True}
        raw = {"read_attestation": attest, "findings": []}
        if phase == "asset":
            assets = read(self.folder / "PUBLIC_ASSETS.json")
            raw.update(multiselect_stem_valid=True, statements={r["statement_id"]: self.statement_judgment(r["proposition"]) for r in assets["statements"]},
                       ordering_questions=[self.ordering_judgment(q) for q in assets["ordering_questions"]])
        else:
            attest["final_public_questions_actually_reread"] = True
            raw["questions"] = {}
            for q in read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]:
                if q["question_type"] == "ordering":
                    judgment = self.ordering_judgment(q)
                else:
                    judgment = {"candidate_id": q["candidate_id"], **{k: True for k in ("stem_valid", "scope_valid", "fact_distinct", "no_entangled_options")},
                                "reason": "Known synthetic attribution facts.", "options": {label: self.statement_judgment(text) for label, text in q["options"].items()}}
                raw["questions"][q["candidate_id"]] = judgment
        path = self.unit / f"synthetic_{phase}_judgment.json"; write(path, raw)
        self.call("review_v4.py", ROOT, phase + "-read", "--batch", self.batch, "--executor", REVIEWER)
        self.call("review_v4.py", ROOT, phase + "-save", "--batch", self.batch, "--executor", REVIEWER, "--judgment", path)

    def test_full_native_retry_expansion_and_qa(self):
        self.expand(anchors.SCENES[1:])
        self.call("author_v120.py", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        plan, draft = self.qa_payloads()
        self.call("author_v120.py", "anchor-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--payload", plan)
        self.call("author_v120.py", "qa-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--payload", draft)
        self.call("question_controller_v120.py", "publish", "--batch-dir", self.unit)
        self.qa_review("asset")
        compose = self.call("run_guarded_questions_v2.py", "compose", "--batch", self.batch, "--root", ROOT, json_output=True)
        self.assertTrue(compose["downstream_invoked"]); self.assertEqual(compose["status"], "GUARDED_STAGE_COMPLETED")
        self.assertEqual(len(read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]), 7)
        self.qa_review("public")
        finalize = self.call("run_guarded_questions_v2.py", "finalize", "--batch", self.batch, "--root", ROOT, json_output=True)
        self.assertTrue(finalize["downstream_invoked"]); self.assertEqual(finalize["status"], "GUARDED_STAGE_COMPLETED")
        crosscheck = read(self.folder / "FINAL_CROSSCHECK.json")
        self.assertEqual(len(crosscheck["results"]), 7)
        self.assertTrue(all(row["status"] == "CONTENT_PASS" for row in crosscheck["results"]), crosscheck)
        gold = read(self.folder / "PRIVATE_GOLD.json")
        self.assertTrue(gold["gold_constructed_after_complete_public_review"])
        by_id = {q["candidate_id"]: q for q in read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]}
        self.assertEqual(sorted(len(q["gold"]) for q in gold["questions"] if by_id[q["candidate_id"]]["question_type"] == "multi_select"), [2, 3, 4, 5, 6])
        old = read(self.history / "roots/root1/scenes/A_S01/SOURCE_REVIEW.json")
        imported = read(self.folder / "scenes/A_S01/SOURCE_REVIEW.json")
        self.assertEqual(old, imported); self.assertEqual(imported["reviewer"], "/root/historical_fixture_reviewer")
        self.assertGreaterEqual(len(self.stages), 40)

    def test_author_read_rejects_changed_execution_config_before_render(self):
        path = self.unit / "RUN_CONFIG.json"
        cfg = read(path); cfg["new_turn_count"] = {"min": 1, "max": 100}; write(path, cfg)
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        result = self.call("author_v120.py", "scene-read", "--batch-dir", self.batch, "--root", ROOT,
                           "--executor", AUTHOR, "--scene", "R02:A:S02", expect=1)
        self.assertEqual(result, "")
        self.assertFalse((self.unit / "evidence/AUTHOR_R02_A_S02_READ_CONTEXT.json").exists())
        self.assertFalse((self.unit / "evidence/AUTHOR_R02_A_S02_FULL_READ.md").exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_author_entry_rejects_foreign_self_or_serializer_origins(self):
        # A separate interpreter imports the real frozen author helper and then
        # substitutes only an origin; no production function implementation is mocked.
        for field, expected in (("self", "RUNTIME_LOADED_CODE_MISMATCH"), ("serializer", "FOREIGN_SCENE_SERIALIZER")):
            foreign = self.tools / ("FOREIGN_" + field + ".py")
            shutil.copy2(self.tools / "author_v120.py" if field == "self" else self.batch / "scripts/codex_scene_payload.py", foreign)
            before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
            script = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import author_v120 as author
if sys.argv[2] == "self": author.__file__ = sys.argv[3]
else: author.serializer.__file__ = sys.argv[3]
try:
    author.assigned(Path(sys.argv[4]),sys.argv[5],sys.argv[6])
except ValueError as exc:
    print(str(exc)); raise SystemExit(2)
raise SystemExit(9)
'''
            completed = subprocess.run([sys.executable, "-B", "-c", script, str(self.tools), field, str(foreign),
                                        str(self.batch), ROOT, AUTHOR], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn(expected, completed.stdout)
            self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_real_native_a2_review_unlocks_a3_then_missing_receipt_blocks(self):
        self.expand([anchors.SCENES[1]])
        scene = anchors.SCENES[2]
        self.call("author_v120.py", "scene-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--scene", scene)
        payload = self.unit / "synthetic_A3_payload.json"
        write(payload, {"turns": [{"speaker": "A" if i % 2 else "B", "text": f"Synthetic A3 check {i}."} for i in range(1, 13)]})
        self.call("author_v120.py", "scene-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--scene", scene, "--payload", payload)
        state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
        self.assertEqual(state["scenes"][2]["status"], "AWAITING_REVIEW")
        digest = hashlib.sha256(REVIEWER.encode()).hexdigest()[:16]
        receipt = self.unit / "evidence/reads" / f"scene_R02_A_S02_{digest}" / "SAVE_RECEIPT.json"
        receipt.unlink()
        state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
        self.assertEqual(state["scenes"][1]["status"], "INTEGRITY_FAILURE")
        self.assertIn("NEW_SCENE_REVIEW_PROVENANCE_BLOCKED", state["scenes"][1]["reason"])
        self.assertFalse(state["full_expansion_complete"])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main(verbosity=2)
