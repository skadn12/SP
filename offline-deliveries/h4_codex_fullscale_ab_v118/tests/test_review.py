"""Temporary-fixture tests: role enforcement, immutable reads, and QA integration."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.dont_write_bytecode = True
HELPER = Path(__file__).resolve().parents[1] / "tools/review.py"
spec = importlib.util.spec_from_file_location("v118_review", HELPER)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
sys.path.insert(0, str(review.PROJECT / "scripts"))
from codex_scene_payload import write_new
from codex_sequential_ab import materialize_scene, inspect_root, load_config
from codex_sequential_questions import publish_assets, compose, finalize
from codex_evidence_export import export_root_questions


ROOT = "cc-train-episode-42"
AUTHOR = "/root/test_author"
REVIEWER = "/root/test_reviewer"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quiet(function, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name) / "batch"
        self.unit = self.batch / "units/root42"
        self.folder = self.unit / "roots/root42"
        self.roles = {"roots": [{"root_id": ROOT, "author": AUTHOR, "reviewer": REVIEWER,
                                 "baseline_reviewers": [AUTHOR, "/root/test_third"]}]}
        write(self.batch / "ROLE_ASSIGNMENTS.json", self.roles)
        baseline = self.unit / "inputs/BASELINE.json"
        write(baseline, {"root_id": ROOT, "turns": [
            {"turn_ref": "P:T001", "speaker": "Emily", "text": "We have two separate notebooks.", "scene_ref": "P"},
            {"turn_ref": "P:T002", "speaker": "James", "text": "We will keep those notebooks separate.", "scene_ref": "P"}]})
        reviews = []
        for i in (1, 2):
            p = self.unit / f"inputs/BASELINE_REVIEW_{i}.json"
            write(p, {"decision": "WARNING", "reviewer": [AUTHOR, "/root/test_third"][i - 1]})
            reviews.append(review.binding(p))
        self.constraints = {"root_id": ROOT, "decision": "WARNING", "continuation_constraints": [
            "Do not merge the two notebooks.", "Do not infer gender from a name."],
            "baseline_binding": review.binding(baseline), "review_bindings": reviews}
        write(self.unit / "inputs/CONTINUATION_CONSTRAINTS.json", self.constraints)
        cfg = {"controller_version": "codex_sequential_ab_v1", "batch_id": "fixture_batch", "scene_sequence": list(review.SCENES),
               "new_turn_count": {"min": 12, "max": 16}, "roots": [{"root_id": ROOT, "baseline_path": str(baseline),
               "baseline_sha256": review.sha(baseline)}], "seed_namespace": "fixture", "multiselect_target_cardinalities": [2, 3, 4, 5]}
        write(self.unit / "RUN_CONFIG.json", cfg)
        reservations = []
        for n in (1, 2, 3):
            event = self.unit / f"inputs/EVENT_{n}.json"
            write(event, {"dialogue_id": f"fixture{n}", "services": ["train"], "turns": [
                {"turn_id": "0", "speaker": "USER", "utterance": f"Please quote service {n}."},
                {"turn_id": "1", "speaker": "SYSTEM", "utterance": f"Service {n} is available."}]})
            reservations.append({"root_id": ROOT, "scene_ref": f"R02:B:S{n:02d}", "source_id": f"fixture:{n}",
                                 "source_item_path": str(event), "source_item_sha256": review.sha(event)})
        write(self.unit / "SOURCE_RESERVATIONS.json", {"allocation": reservations})

    def draft(self, scene):
        n = review.SCENES.index(scene) + 1
        messages = [{"speaker": "Emily" if i % 2 else "James", "text": f"Scene {n} has explicit report number {i}."}
                    for i in range(1, 13)]
        quiet(write_new, self.unit, ROOT, scene, AUTHOR, {"messages": messages})
        state = quiet(materialize_scene, self.unit, ROOT, scene)
        self.assertEqual(state["status"], "STRUCTURAL_PASS_ONLY")

    def judgment(self, scene, **changes):
        source = review.read(self.folder / "scenes" / scene.removeprefix("R02:").replace(":", "_") / "CUMULATIVE_SOURCE.json")
        refs = [source["turns"][-1]["turn_ref"]]
        raw = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True},
               "decision": "READY", "checks": {name: {"status": "PASS", "reason": "Fixture-specific explicit reports preserve this check."} for name in review.CHECKS},
               "findings": [], "baseline_constraint_review": {"status": "PASS", "reason": "Neither inherited constraint is changed.",
               "constraint_results": [{"index": i, "constraint": c, "status": "PASS", "reason": "The explicit report keeps the existing boundary.",
                                       "evidence_turn_refs": refs} for i, c in enumerate(self.constraints["continuation_constraints"], 1)],
               "new_blocking_findings": []}}
        if ":B:" in scene:
            raw["read_attestation"]["native_event_full_read"] = True
        raw.update(changes)
        return raw

    def supplied(self, raw, name="judgment"):
        p = self.batch / (name + ".json")
        write(p, raw)
        return p

    def read_scene(self, scene):
        return quiet(review.start, self.batch, ROOT, "scene", REVIEWER, scene)

    def save_scene(self, scene, raw=None):
        return quiet(review.save_review, self.batch, ROOT, "scene", REVIEWER,
                     self.supplied(raw if raw is not None else self.judgment(scene), "judgment_" + scene.replace(":", "_")), scene)

    def ready_all(self):
        for scene in review.SCENES:
            self.draft(scene)
            self.read_scene(scene)
            self.save_scene(scene)
        final = self.folder / "scenes/B_S03"
        for name in ("CUMULATIVE_SOURCE.json", "SOURCE_REVIEW.json"):
            (self.folder / name).write_bytes((final / name).read_bytes())

    def make_qa(self):
        self.ready_all()
        source = review.read(self.folder / "CUMULATIVE_SOURCE.json")
        turns = {t["turn_ref"]: t for t in source["turns"]}
        stem = "Select all supported statements within these scenes: " + ", ".join(review.SCENES)
        facts = []
        for i, scene in enumerate(review.SCENES, 1):
            t = turns[scene + ":T001"]
            facts.append({"fact_id": f"F{i}", "source_turn_ref": t["turn_ref"], "true_text": "Emily reports: " + t["text"],
                          "false_text": f"Emily reports: Scene {i} has explicit report number 99."})
        slots = []; items = []
        for slot_id in (1, 2):
            anchors = []; rows = []
            for i, scene in enumerate(review.SCENES, 1):
                t = turns[scene + f":T{slot_id:03d}"]
                anchor_id = f"S{slot_id}A{i}"
                anchors.append({"anchor_id": anchor_id, "turn_ref": t["turn_ref"], "speaker": t["speaker"], "full_turn_quote": t["text"]})
                label = f"I{7-i}"
                rows.append({"item_id": label, "anchor_id": anchor_id, "text": t["speaker"] + " reports: " + t["text"], "support_quote": t["text"]})
            slots.append({"slot_id": str(slot_id), "anchors": anchors})
            items.append({"slot_id": str(slot_id), "items": sorted(rows, key=lambda row: row["item_id"])})
        write(self.folder / "ORDERING_ANCHOR_PLAN.json", {"source_sha256": review.sha(self.folder / "CUMULATIVE_SOURCE.json"), "slots": slots})
        write(self.folder / "QA_DRAFT.json", {"stem": stem, "facts": facts, "ordering": items})
        write(self.folder / "QA_AUTHOR_RECEIPT.json", {"author": AUTHOR, "source_sha256": review.sha(self.folder / "CUMULATIVE_SOURCE.json")})
        quiet(publish_assets, self.unit)
        self.turns = turns
        self.true_texts = {fact["true_text"] for fact in facts}
        self.proposition_judgments = {}
        assets = review.read(self.folder / "PUBLIC_ASSETS.json")
        for row in assets["statements"]:
            proposition = row["proposition"]
            n = int(proposition.split("Scene ")[1].split()[0])
            t = turns[review.SCENES[n - 1] + ":T001"]
            c = {"turn_ref": t["turn_ref"], "speaker": t["speaker"], "quote": t["text"]}
            self.proposition_judgments[proposition] = {"proposition": proposition,
                "relation": "supports" if proposition in self.true_texts else "contradicts", "scope_valid": True,
                "fact_distinct": True, "reason": "The report number is explicit in this source turn.", "citations": [c], "primary_citation": c}
        self.asset_raw = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True,
                          "private_answers_read_before_save": False}, "multiselect_stem_valid": True, "findings": [],
                          "statements": {row["statement_id"]: self.proposition_judgments[row["proposition"]] for row in assets["statements"]},
                          "ordering_questions": [self.ordering_judgment(q) for q in assets["ordering_questions"]]}

    def ordering_judgment(self, question):
        items = {}
        for label, proposition in question["items"].items():
            suffix = proposition.split(" reports: ", 1)[1]
            turn = next(t for t in self.turns.values() if t["text"] == suffix)
            items[label] = {"proposition": proposition, "supported": True, "actor_ok": True, "modality_ok": True,
                "state_ok": True, "neutral_stem_ok": True, "single_event_ok": True, "order_anchor": {
                "turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": turn["text"]},
                "support_refs": [turn["turn_ref"]], "alternative_matching_anchors": [], "reason": "This exact report has one source occurrence."}
        positions = {ref: i for i, ref in enumerate(self.turns)}
        return {"candidate_id": question["candidate_id"], "stem_valid": True, "scope_valid": True, "coherent": True,
                "unique_order": True, "neutral_stem_ok": True, "items": items,
                "derived_order": sorted(items, key=lambda k: positions[items[k]["order_anchor"]["turn_ref"]]),
                "alternative_orders": [], "reason": "Each report is tied to its explicit single scene."}


class SceneReviewTests(Fixture):
    def test_wrong_executor_and_root_orchestrator_rejected_before_artifacts(self):
        self.draft(review.SCENES[0])
        for actor, error in [(AUTHOR, "EXECUTOR_NOT_ASSIGNED"), ("/root", "EXPLICIT_CANONICAL")]:
            with self.assertRaisesRegex(review.ReviewError, error):
                quiet(review.start, self.batch, ROOT, "scene", actor, review.SCENES[0])
        self.assertFalse((self.unit / "evidence/reads").exists())

    def test_assignment_self_review_rejected(self):
        self.roles["roots"][0]["reviewer"] = AUTHOR
        write(self.batch / "ROLE_ASSIGNMENTS.json", self.roles)
        with self.assertRaisesRegex(review.ReviewError, "ASSIGNED_REVIEWER_IS_AUTHOR"):
            quiet(review.start, self.batch, ROOT, "scene", AUTHOR, review.SCENES[0])

    def test_actual_draft_author_must_match_assignment(self):
        self.draft(review.SCENES[0])
        p = self.folder / "scenes/A_S01/EXPANSION_DRAFT.json"
        d = review.read(p); d["author"] = REVIEWER; write(p, d)
        with self.assertRaisesRegex(review.ReviewError, "DRAFT_AUTHOR_NOT_ASSIGNED|SCENE_NOT_AWAITING"):
            self.read_scene(review.SCENES[0])

    def test_unchanged_scene_roundtrip_is_controller_compatible_and_exclusive(self):
        scene = review.SCENES[0]; self.draft(scene)
        context = self.read_scene(scene)
        with self.assertRaises(FileExistsError):
            self.read_scene(scene)
        output = self.save_scene(scene)
        saved = review.read(output)
        self.assertEqual(saved["reviewer"], REVIEWER)
        self.assertTrue(saved["all_continuation_constraints_read"])
        state = inspect_root(self.unit, load_config(self.unit), ROOT, record_reviews=False)
        self.assertEqual(state["scenes"][0]["status"], "READY")
        self.assertEqual(review.sha(context.parent / "INPUT_RENDER.txt"), saved["input_render_binding"]["sha256"])
        with self.assertRaisesRegex(review.ReviewError, "REVIEW_ALREADY_EXISTS"):
            self.save_scene(scene)

    def test_source_changed_after_read_is_rejected(self):
        scene = review.SCENES[0]; self.draft(scene); self.read_scene(scene)
        p = self.folder / "scenes/A_S01/CUMULATIVE_SOURCE.json"
        d = review.read(p); d["turns"][-1]["text"] += " changed"; write(p, d)
        with self.assertRaisesRegex(review.ReviewError, "SCENE_NOT_AWAITING|READ_INPUTS_CHANGED"):
            self.save_scene(scene)
        self.assertFalse((p.parent / "SOURCE_REVIEW.json").exists())

    def test_constraint_change_after_read_rejected(self):
        scene = review.SCENES[0]; self.draft(scene); self.read_scene(scene)
        self.constraints["continuation_constraints"].append("A new later condition.")
        write(self.unit / "inputs/CONTINUATION_CONSTRAINTS.json", self.constraints)
        with self.assertRaisesRegex(review.ReviewError, "READ_INPUTS_CHANGED"):
            self.save_scene(scene)

    def test_role_manifest_change_after_read_rejected(self):
        scene = review.SCENES[0]; self.draft(scene); self.read_scene(scene)
        self.roles["procedural_note"] = "added after this read"
        write(self.batch / "ROLE_ASSIGNMENTS.json", self.roles)
        with self.assertRaisesRegex(review.ReviewError, "READ_INPUTS_CHANGED"):
            self.save_scene(scene)

    def test_baseline_reviewers_must_match_fixed_assignment(self):
        scene = review.SCENES[0]; self.draft(scene)
        self.roles["roots"][0]["baseline_reviewers"] = [AUTHOR, "/root/unassigned_reader"]
        write(self.batch / "ROLE_ASSIGNMENTS.json", self.roles)
        with self.assertRaisesRegex(review.ReviewError, "BASELINE_REVIEWERS_NOT_ASSIGNED"):
            self.read_scene(scene)

    def test_render_changed_after_read_rejected(self):
        scene = review.SCENES[0]; self.draft(scene); context = self.read_scene(scene)
        with (context.parent / "INPUT_RENDER.txt").open("a") as stream:
            stream.write("replacement")
        with self.assertRaisesRegex(review.ReviewError, "HASH_CHANGED"):
            self.save_scene(scene)

    def test_missing_constraint_judgment_preserves_first_raw_and_failure(self):
        scene = review.SCENES[0]; self.draft(scene); context = self.read_scene(scene)
        raw = self.judgment(scene); raw["baseline_constraint_review"]["constraint_results"].pop()
        with self.assertRaisesRegex(review.ReviewError, "CONSTRAINT_COVERAGE_INCOMPLETE"):
            self.save_scene(scene, raw)
        self.assertEqual(review.read(context.parent / "RAW_JUDGMENT.json"), raw)
        self.assertTrue((context.parent / "SAVE_FAILURE.json").exists())
        with self.assertRaises(FileExistsError):
            self.save_scene(scene)

    def test_read_claim_must_be_explicit(self):
        scene = review.SCENES[0]; self.draft(scene); self.read_scene(scene)
        raw = self.judgment(scene); raw["read_attestation"]["all_continuation_constraints_read"] = False
        with self.assertRaisesRegex(review.ReviewError, "EXPLICIT_READ_ATTESTATION"):
            self.save_scene(scene, raw)

    def test_failed_check_cannot_claim_ready(self):
        scene = review.SCENES[0]; self.draft(scene); self.read_scene(scene)
        raw = self.judgment(scene); raw["checks"]["actor_identity"]["status"] = "FAIL"
        with self.assertRaisesRegex(review.ReviewError, "READY_WITH_FAILED_CHECK"):
            self.save_scene(scene, raw)

    def test_hold_is_saved_and_blocks_next_scene(self):
        scene = review.SCENES[0]; self.draft(scene); self.read_scene(scene)
        raw = self.judgment(scene, decision="HOLD")
        raw["checks"]["actor_identity"] = {"status": "FAIL", "reason": "This fixture deliberately reports an identity failure."}
        output = self.save_scene(scene, raw)
        self.assertEqual(review.read(output)["decision"], "HOLD")
        with self.assertRaisesRegex(ValueError, "ROOT_ALREADY_STOPPED"):
            quiet(write_new, self.unit, ROOT, review.SCENES[1], AUTHOR, {"messages": []})

    def test_b_native_full_input_is_bound_and_attested(self):
        for scene in review.SCENES[:3]:
            self.draft(scene); self.read_scene(scene); self.save_scene(scene)
        scene = review.SCENES[3]; self.draft(scene); context = self.read_scene(scene)
        text = (context.parent / "INPUT_RENDER.txt").read_text()
        self.assertIn("Please quote service 1.", text)
        self.assertIn("Service 1 is available.", text)
        raw = self.judgment(scene); raw["read_attestation"].pop("native_event_full_read")
        with self.assertRaisesRegex(review.ReviewError, "EXPLICIT_NATIVE"):
            self.save_scene(scene, raw)


class QAReviewTests(Fixture):
    def test_complete_asset_public_gold_export_roundtrip(self):
        self.make_qa()
        quiet(review.start, self.batch, ROOT, "asset", REVIEWER)
        quiet(review.save_review, self.batch, ROOT, "asset", REVIEWER, self.supplied(self.asset_raw, "asset"))
        quiet(compose, self.unit)
        public = review.read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]
        self.assertEqual(len(public), 7)
        raw = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True,
               "private_answers_read_before_save": False, "final_public_questions_actually_reread": True}, "questions": {}, "findings": []}
        for question in public:
            cid = question["candidate_id"]
            if question["question_type"] == "ordering":
                raw["questions"][cid] = self.ordering_judgment(question)
            else:
                raw["questions"][cid] = {"candidate_id": cid, "stem_valid": True, "scope_valid": True, "fact_distinct": True,
                    "no_entangled_options": True, "options": {label: self.proposition_judgments[prop] for label, prop in question["options"].items()},
                    "reason": "Each option was compared with its explicit source report."}
        quiet(review.start, self.batch, ROOT, "public", REVIEWER)
        quiet(review.save_review, self.batch, ROOT, "public", REVIEWER, self.supplied(raw, "public"))
        quiet(finalize, self.unit)
        exported = export_root_questions(self.folder)
        self.assertEqual(len(exported), 7)
        self.assertEqual(sorted(len(q["gold"]) for q in exported if q["question_type"] == "multi_select"), [2, 3, 4, 5, 6])

    def test_qa_receipt_author_cannot_be_reviewer(self):
        self.make_qa()
        p = self.folder / "QA_AUTHOR_RECEIPT.json"
        d = review.read(p); d["author"] = REVIEWER; write(p, d)
        with self.assertRaisesRegex(review.ReviewError, "QA_AUTHOR_NOT_ASSIGNED|QA_SELF_REVIEW"):
            quiet(review.start, self.batch, ROOT, "asset", REVIEWER)

    def test_multicitation_requires_explicit_primary(self):
        self.make_qa(); quiet(review.start, self.batch, ROOT, "asset", REVIEWER)
        raw = deepcopy(self.asset_raw); first = next(iter(raw["statements"].values()))
        first["citations"].append(deepcopy(first["citations"][0])); first.pop("primary_citation")
        with self.assertRaisesRegex(review.ReviewError, "EXPLICIT_REVIEWED_PRIMARY"):
            quiet(review.save_review, self.batch, ROOT, "asset", REVIEWER, self.supplied(raw))

    def test_private_blindness_is_attested_not_programmatically_proven(self):
        self.make_qa()
        # Deliberately access a private fixture here to demonstrate the boundary:
        # the helper cannot detect earlier file access in a shared environment.
        (self.folder / "QA_DRAFT.json").read_text()
        paths_read = []
        original = review.read
        def track(path):
            paths_read.append(Path(path).name)
            return original(path)
        with patch.object(review, "read", side_effect=track):
            context = quiet(review.start, self.batch, ROOT, "asset", REVIEWER)
            output = quiet(review.save_review, self.batch, ROOT, "asset", REVIEWER, self.supplied(self.asset_raw))
        for forbidden in ("QA_DRAFT.json", "PRIVATE_ASSET_DESIGN.json", "PRIVATE_DESIGN.json", "PRIVATE_GOLD.json"):
            self.assertNotIn(forbidden, paths_read)
        receipt = review.read(context.parent / "SAVE_RECEIPT.json")
        self.assertFalse(receipt["private_file_access_proven_by_program"])
        self.assertFalse(receipt["internal_reading_proven_by_program"])
        self.assertIn("cannot prove", review.read(output)["read_attestation_scope"])

    def test_private_answer_read_claim_rejected(self):
        self.make_qa(); quiet(review.start, self.batch, ROOT, "asset", REVIEWER)
        raw = deepcopy(self.asset_raw); raw["read_attestation"]["private_answers_read_before_save"] = True
        with self.assertRaisesRegex(review.ReviewError, "EXPLICIT_SOURCE_FIRST"):
            quiet(review.save_review, self.batch, ROOT, "asset", REVIEWER, self.supplied(raw))

    def test_existing_gold_rejected_before_qa_read(self):
        self.make_qa(); write(self.folder / "PRIVATE_GOLD.json", {"questions": []})
        with self.assertRaisesRegex(review.ReviewError, "PRIVATE_GOLD_ALREADY_EXISTS"):
            quiet(review.start, self.batch, ROOT, "asset", REVIEWER)


if __name__ == "__main__":
    unittest.main()
