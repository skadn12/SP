from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method import reason_material_feedback_pipeline as pipeline


class FeedbackPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.r1, self.r2 = "cc-train-episode-103866", "cc-train-episode-104814"
        self.sources = {}
        for root in (self.r1, self.r2):
            self.sources[root] = self.put(root + ".json", {"root_id": root, "turns": [
                {"turn_ref": "OLD:T001", "speaker": "Alex", "text": "We have not confirmed the date."},
                {"turn_ref": "OLD:T002", "speaker": "Sam", "text": "I will check."},
            ]})
        parsed = {"item_evidence": {"A": {"judgment": "supported", "turn_refs": ["OLD:T001"], "reason": "The date remains unconfirmed."}}, "answer_reason": "This is a proposal, not a confirmed booking.", "final_answer": "<box>A</box>"}
        raw = self.put("reason_raw.json", {"raw_output": json.dumps(parsed)})
        self.reason_entry = {"request_id": "reason1", "candidate_id": "question1", "root_id": self.r1, "question_type": "multi_select", "source_binding": self.sources[self.r1], "raw_binding": raw, "model_reported_explanations": parsed, "semantic_review": {"raw_binding": raw, "review_complete": True, "reviewer_type": "ai_agent", "item_reviews": {"A": {"source_support": "pass", "reason_support": "pass", "notes": "Read the unconfirmed date in source."}}, "answer_reason_consistency": "pass"}, "public_task": {"question": "Which claim is supported?"}, "mechanical_consistency": {"consistent": True}}
        self.reason_packet = self.put("reason_packet.json", {"scope": "train_only_exploratory", "model_reasons_included": True, "formal_B_admission": False, "entries": [self.reason_entry], "bindings": []})
        review = self.put("shortfall_review.json", {"slot_reviews": [{"slot_id": "time-gap", "recommendation": "PASS_SHORTFALL", "finding": "No common date establishes a unique timeline."}]})
        self.gap = {"slot_id": "time-gap", "root_id": self.r2, "status": "SHORTFALL", "relation_target": "ORDER_EXPLICIT_TIME", "source_sha256": self.sources[self.r2]["sha256"], "review_binding": review, "review_finding": "No common date establishes a unique timeline.", "target_gold_count": 4}
        self.ledger = self.put("ledger.json", {"slots": [self.gap]})
        self.config = {"schema_version": pipeline.VERSION + "_config", "batch_id": "test", "namespace": "RMF1", "reason_packet": self.reason_packet, "shortfall_ledger": self.ledger, "roots": [
            {"root_id": self.r1, "source_binding": self.sources[self.r1], "task_operations": [{"operation_id": "evidence_selection", "question_type": "multi_select"}]},
            {"root_id": self.r2, "source_binding": self.sources[self.r2], "task_operations": [{"operation_id": "explicit_event_time", "question_type": "ordering"}], "material_targets": [{"slot_id": "time-gap", "material_goal": "Explicitly confirm dates for the related future events."}]},
        ]}

    def put(self, name, data):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return pipeline.binding(path)

    def prepare(self, config=None, output="prepared"):
        cfg = self.put("config.json", config or self.config)
        destination = self.base / output
        result = pipeline.prepare(cfg["path"], destination)
        return destination, result

    def source_review(self, name, source_bound, author="writer", draft=None, time="2026-10-22T02:00:00+00:00"):
        source = pipeline.bound_json(source_bound)
        review = {"source_binding": source_bound, "reviewer_id": "independent-reader", "reviewer_type": "ai_agent", "independent_of_source_author": True, "full_source_read": True, "turn_refs_read": [t["turn_ref"] for t in source["turns"]], "decision": "READY", "checks": {name: {"status": "PASS", "reason": "Checked the complete cumulative source."} for name in pipeline.CHECKS}, "findings": [], "created_at_utc": time}
        if draft:
            review["draft_binding"] = draft
        return self.put(name, review)

    def reuse_submission(self, directory):
        root_dir = directory / self.r1
        review = self.source_review("reuse_review.json", self.sources[self.r1])
        sub = {"root_id": self.r1, "mode": "REUSE", "author_input_binding": pipeline.binding(root_dir / "SOURCE_AUTHOR_INPUT.json"), "source_binding": self.sources[self.r1], "source_author_id": "legacy-author", "review_binding": review}
        return root_dir, sub

    def extension_fixture(self):
        config = copy.deepcopy(self.config)
        config["roots"] = [config["roots"][1]]
        events = []
        for index in range(1, 4):
            events.append({"scene_ref": f"RMF1:B:S{index:02d}", "source_id": f"soda:train:{index}", "split": "train", "binding": self.put(f"event{index}.json", {"split": "train", "event": f"An independently reserved event {index}."}), "reserved_before_authoring": True})
        config["roots"][0]["native_events"] = events
        directory, _ = self.prepare(config)
        root_dir = directory / self.r2
        author_bound = pipeline.binding(root_dir / "SOURCE_AUTHOR_INPUT.json")
        prior_bound = self.sources[self.r2]
        scenes = []
        for index, scene in enumerate([f"RMF1:{route}:S{n:02d}" for route in "AB" for n in range(1, 4)]):
            # Strictly increasing author/reviewer times preserve predecessor READY.
            draft_time = f"2026-10-22T{2 + index:02d}:00:00+00:00"
            review_time = f"2026-10-22T{2 + index:02d}:30:00+00:00"
            turns = [{"turn_ref": f"{scene}:T{n:03d}", "scene_ref": scene, "session_ref": scene, "speaker": "Alex" if n % 2 else "Sam", "text": f"Confirmed detail {index}-{n}."} for n in range(1, 13)]
            draft = {"root_id": self.r2, "scene_ref": scene, "author_id": "writer", "created_at_utc": draft_time, "input_source_binding": prior_bound, "author_input_binding": author_bound, "turns": turns}
            if index >= 3:
                event = events[index - 3]
                draft.update(native_event_binding=event["binding"], native_event_source_id=event["source_id"])
            db = self.put(f"draft{index}.json", draft)
            prior = pipeline.bound_json(prior_bound)
            sb = self.put(f"cumulative{index}.json", {"root_id": self.r2, "turns": prior["turns"] + turns})
            rb = self.source_review(f"review{index}.json", sb, draft=db, time=review_time)
            scenes.append({"scene_ref": scene, "draft_binding": db, "source_binding": sb, "review_binding": rb})
            prior_bound = sb
        material = self.material_review(prior_bound)
        sub = {"root_id": self.r2, "mode": "EXTEND", "author_input_binding": author_bound, "scenes": scenes, "material_realization_review_binding": material}
        return root_dir, sub

    def material_review(self, source_bound):
        source = pipeline.bound_json(source_bound)
        turn = source["turns"][-1]
        return self.put("material_review.json", {"source_binding": source_bound, "reviewer_type": "ai_agent", "reviewer_id": "material-reader", "full_source_read": True, "targets": [{"slot_id": "time-gap", "status": "READY", "notes": "The new source explicitly supplies the previously missing relationship.", "evidence": [{"turn_ref": turn["turn_ref"], "quote": turn["text"]}]}]})

    def legacy_fixture(self):
        config = copy.deepcopy(self.config)
        config["roots"] = [config["roots"][1]]
        directory, _ = self.prepare(config)
        root_dir = directory / self.r2
        prior_bound = self.sources[self.r2]
        records, reservations = [], []
        for index, scene in enumerate([f"T4R:{route}:S{n:02d}" for route in "AB" for n in range(1, 4)]):
            turns = [{"turn_ref": f"{scene}:T{n:03d}", "speaker": "Alex", "text": f"Legacy fact {index}-{n}."} for n in range(1, 13)]
            native = None
            if index >= 3:
                native = self.put(f"legacy_event{index}.json", {"event": f"Train event {index}"})
                reservations.append({"root_id": self.r2, "scene_ref": scene, "split": "train", "status": "PERMANENTLY_RESERVED_BEFORE_GENERATION", "native_record": native})
            draft = self.put(f"legacy_draft{index}.json", {"root_id": self.r2, "scene_ref": scene, "input_source": prior_bound, "native_event": native, "turns": turns, "generation_metadata": {"started_at_utc": f"2026-10-20T{index + 1:02d}:00:00+00:00", "ended_at_utc": f"2026-10-20T{index + 1:02d}:10:00+00:00"}})
            review = self.put(f"legacy_parent_review{index}.json", {"status": "PASS", "draft": draft, "full_prefix_read": True, "reviewer": "parent", "created_at_utc": f"2026-10-20T{index + 1:02d}:20:00+00:00"})
            source = self.put(f"legacy_source{index}.json", {"root_id": self.r2, "turns": pipeline.bound_json(prior_bound)["turns"] + turns})
            acceptance = self.put(f"legacy_acceptance{index}.json", {"status": "ACCEPTED", "root_id": self.r2, "scene_ref": scene, "source": source, "draft": draft, "review": review, "created_at_utc": f"2026-10-20T{index + 1:02d}:30:00+00:00"})
            records.append({"scene_ref": scene, "source": source, "draft": draft, "parent_review": review, "acceptance": acceptance, "independent_content_status": "PASS"})
            prior_bound = source
        reservation = self.put("legacy_reservations.json", {"reservations": reservations})
        audit = self.put("legacy_independent_audit.json", {"source_content_gate": "PASS", "reviewer": "independent-original-reader", "independence": "Did not write these scenes.", "reservation_binding": reservation, "roots": [{"root_id": self.r2, "status": "PASS_WITH_DOCUMENTED_LIMITATIONS", "prefix": self.sources[self.r2], "final_source": prior_bound, "scene_records": records}]})
        freeze = self.put("legacy_source_freeze.json", {"schema_version": "targeted_four_root_source_freeze_v1", "status": "CONTENT_ACCEPTED_BEFORE_QUESTIONS", "sources": {self.r2: {"mode": "targeted_extension", "accepted_scenes": 6, "source": prior_bound}}, "independent_source_review": audit, "source_reservations": reservation})
        spec = {"namespace": "T4R", "source_freeze_binding": freeze, "source_author_id": "legacy-writer", "current_source_review_binding": self.source_review("current_import_review.json", prior_bound), "material_realization_review_binding": self.material_review(prior_bound)}
        return root_dir, {"root_id": self.r2, "mode": "EXTEND", "author_input_binding": pipeline.binding(root_dir / "SOURCE_AUTHOR_INPUT.json"), "existing_extension_import": spec}

    def accept(self, root_dir, submission):
        sub = self.put("submission.json", submission)
        return pipeline.accept_source(root_dir, sub["path"])

    def test_prepare_isolates_roots_and_sanitizes_source_author_input(self):
        directory, result = self.prepare()
        self.assertEqual([r["mode"] for r in result["roots"]], ["REUSE", "EXTEND"])
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["llm_consumption_proven"])
        planner1 = pipeline.read(directory / self.r1 / "CHALLENGER_INPUT.json")
        planner2 = pipeline.read(directory / self.r2 / "CHALLENGER_INPUT.json")
        self.assertEqual(len(planner1["reviewed_reason_observations"]), 1)
        self.assertEqual(planner2["reviewed_reason_observations"], [])
        author = pipeline.read(directory / self.r2 / "SOURCE_AUTHOR_INPUT.json")
        text = json.dumps(author)
        self.assertNotIn("target_gold_count", text)
        self.assertNotIn("reason1", text)
        self.assertEqual(author["task_operations"], self.config["roots"][1]["task_operations"])
        self.assertEqual(author["full_source"], pipeline.bound_json(self.sources[self.r2]))
        self.assertFalse(author["author_input_ready"])

    def test_reuse_becomes_question_ready_after_independent_review(self):
        directory, _ = self.prepare()
        root_dir, sub = self.reuse_submission(directory)
        result = self.accept(root_dir, sub)
        self.assertEqual(result["source_binding"], self.sources[self.r1])
        self.assertFalse(result["question_generation_started"])
        self.assertEqual(result["task_operations"], self.config["roots"][0]["task_operations"])

    def test_extension_six_scenes_become_question_ready(self):
        root_dir, sub = self.extension_fixture()
        result = self.accept(root_dir, sub)
        self.assertEqual(len(result["accepted_source_reviews"]), 6)
        self.assertEqual(len(pipeline.bound_json(result["source_binding"])["turns"]), 74)

    def test_rejects_unreviewed_reason_before_writing(self):
        data = pipeline.bound_json(self.reason_packet)
        data["entries"][0]["semantic_review"]["review_complete"] = False
        config = copy.deepcopy(self.config)
        config["reason_packet"] = self.put("unreviewed_packet.json", data)
        with self.assertRaisesRegex(ValueError, "REASON_REVIEW_INCOMPLETE"):
            self.prepare(config)
        self.assertFalse((self.base / "prepared").exists())

    def test_rejects_reason_backfill(self):
        data = pipeline.bound_json(self.reason_packet)
        data["entries"][0]["model_reported_explanations"]["answer_reason"] = "A fabricated explanation."
        config = copy.deepcopy(self.config)
        config["reason_packet"] = self.put("modified_packet.json", data)
        with self.assertRaisesRegex(ValueError, "REASONS_MODIFIED"):
            self.prepare(config)

    def test_rejects_cross_root_material_target(self):
        config = copy.deepcopy(self.config)
        config["roots"][0]["material_targets"] = config["roots"][1]["material_targets"]
        with self.assertRaisesRegex(ValueError, "MATERIAL_TARGET_ROOT_MISMATCH"):
            self.prepare(config)

    def test_rejects_unapproved_shortfall(self):
        gap = copy.deepcopy(self.gap)
        review = {"slot_reviews": [{"slot_id": "time-gap", "recommendation": "HOLD"}]}
        gap["review_binding"] = self.put("held_review.json", review)
        config = copy.deepcopy(self.config)
        config["shortfall_ledger"] = self.put("held_ledger.json", {"slots": [gap]})
        with self.assertRaisesRegex(ValueError, "SHORTFALL_NOT_REVIEWED"):
            self.prepare(config)

    def test_rejects_private_answer_field_in_operations(self):
        config = copy.deepcopy(self.config)
        config["roots"][1]["task_operations"][0]["target_gold_count"] = 4
        with self.assertRaisesRegex(ValueError, "ANSWER_DATA_IN_AUTHOR_INPUT"):
            self.prepare(config)

    def test_rejects_stale_source_binding(self):
        Path(self.sources[self.r1]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "BOUND_FILE_CHANGED"):
            self.prepare()

    def test_rejects_source_self_review(self):
        directory, _ = self.prepare()
        root_dir, sub = self.reuse_submission(directory)
        review = pipeline.bound_json(sub["review_binding"])
        review["reviewer_id"] = sub["source_author_id"]
        sub["review_binding"] = self.put("self_review.json", review)
        with self.assertRaisesRegex(ValueError, "SOURCE_AUTHOR_CANNOT_SELF_REVIEW"):
            self.accept(root_dir, sub)

    def test_rejects_modified_prepared_author_handoff(self):
        directory, _ = self.prepare()
        root_dir, sub = self.reuse_submission(directory)
        path = root_dir / "SOURCE_AUTHOR_INPUT.json"
        content = pipeline.read(path)
        content["task_operations"] = [{"operation_id": "substituted"}]
        path.write_text(json.dumps(content))
        sub["author_input_binding"] = pipeline.binding(path)
        with self.assertRaisesRegex(ValueError, "BOUND_FILE_CHANGED"):
            self.accept(root_dir, sub)

    def test_rejects_incomplete_source_review(self):
        directory, _ = self.prepare()
        root_dir, sub = self.reuse_submission(directory)
        review = pipeline.bound_json(sub["review_binding"])
        review["turn_refs_read"] = review["turn_refs_read"][:1]
        sub["review_binding"] = self.put("partial_review.json", review)
        with self.assertRaisesRegex(ValueError, "SOURCE_REVIEW_COVERAGE"):
            self.accept(root_dir, sub)

    def test_rejects_incomplete_scene_sequence(self):
        root_dir, sub = self.extension_fixture()
        sub["scenes"] = sub["scenes"][:-1]
        with self.assertRaisesRegex(ValueError, "ALL_SIX_SCENES"):
            self.accept(root_dir, sub)

    def test_rejects_scene_authored_before_predecessor_ready(self):
        root_dir, sub = self.extension_fixture()
        row = sub["scenes"][1]
        draft = pipeline.bound_json(row["draft_binding"])
        draft["created_at_utc"] = "2026-10-22T01:00:00+00:00"
        row["draft_binding"] = self.put("early_draft.json", draft)
        with self.assertRaisesRegex(ValueError, "DRAFT_CREATED_BEFORE_PREDECESSOR_READY"):
            self.accept(root_dir, sub)

    def test_rejects_modified_historical_prefix(self):
        root_dir, sub = self.extension_fixture()
        row = sub["scenes"][0]
        source = pipeline.bound_json(row["source_binding"])
        source["turns"][0]["text"] = "The date was already confirmed."
        row["source_binding"] = self.put("changed_prefix.json", source)
        with self.assertRaisesRegex(ValueError, "SOURCE_MUST_BE_EXACT_PREFIX"):
            self.accept(root_dir, sub)

    def test_rejects_native_event_substitution(self):
        root_dir, sub = self.extension_fixture()
        row = sub["scenes"][3]
        draft = pipeline.bound_json(row["draft_binding"])
        draft["native_event_source_id"] = "soda:train:999"
        row["draft_binding"] = self.put("wrong_native_event.json", draft)
        with self.assertRaisesRegex(ValueError, "DRAFT_NATIVE_EVENT_ID_MISMATCH"):
            self.accept(root_dir, sub)

    def test_imports_original_legacy_audit_without_claiming_new_generation(self):
        root_dir, sub = self.legacy_fixture()
        result = self.accept(root_dir, sub)
        self.assertEqual(result["mode"], "REUSE_EXISTING_EXTENSION")
        self.assertEqual(result["namespace"], "T4R")
        self.assertFalse(result["existing_source_generated_for_current_feedback"])
        self.assertEqual(len(result["accepted_source_reviews"]), 6)

    def test_import_rejects_missing_original_audit(self):
        root_dir, sub = self.legacy_fixture()
        spec = sub["existing_extension_import"]
        freeze = pipeline.bound_json(spec["source_freeze_binding"])
        audit = pipeline.bound_json(freeze["independent_source_review"])
        audit["roots"][0]["scene_records"][2]["independent_content_status"] = "HOLD"
        freeze["independent_source_review"] = self.put("held_legacy_audit.json", audit)
        spec["source_freeze_binding"] = self.put("held_legacy_freeze.json", freeze)
        with self.assertRaisesRegex(ValueError, "LEGACY_SCENE_INDEPENDENT_QA_NOT_PASS"):
            self.accept(root_dir, sub)

    def test_extension_rejects_unresolved_material_target(self):
        root_dir, sub = self.extension_fixture()
        review = pipeline.bound_json(sub["material_realization_review_binding"])
        review["targets"][0]["status"] = "HOLD"
        sub["material_realization_review_binding"] = self.put("unresolved_material.json", review)
        with self.assertRaisesRegex(ValueError, "MATERIAL_GAP_STILL_UNRESOLVED"):
            self.accept(root_dir, sub)


if __name__ == "__main__":
    unittest.main()
