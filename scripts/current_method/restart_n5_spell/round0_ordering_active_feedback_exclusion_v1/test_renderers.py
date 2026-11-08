"""Boundary regressions for offline feedback messages and source identities."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import NO_FEEDBACK, VERSION
from .renderers import backend_record, feedback_envelope, history_envelope, render_role, validate_envelope
from .sources import Sources, file_sha, scope_filter


ROOT = "cc-train-name-does-not-authorize-split"
SCHEMA = {"type": "object", "properties": {"fixture_schema_marker": {"type": "string"}}}
STRATEGY = {"strategy_space": ["fixture_strategy_marker"]}
DIALOGUE = {"root_id": ROOT, "turns": [
    {"turn_ref": "R01:T001", "speaker": "Alice", "text": "Authoritative dialogue unique marker."},
]}


def record(candidate_id="allowed", question_type="multi_select", missing_metrics=False):
    qa = {
        "candidate_id": candidate_id, "root_id": ROOT, "question_type": question_type,
        "question": "Complete source question unique marker?",
        "options": {"A": "Alpha full option", "B": "Beta full option", "C": "Gamma full option", "D": "Delta full option"},
        "gold": ["A", "C"], "evidence_refs": [],
    }
    if question_type == "ordering":
        qa.update({"options": {}, "items": {"I1": "First full item", "I2": "Second full item"}, "gold": ["I2", "I1"]})
    metric = {"full_correct": 2, "full_total": 12, "no_correct": 1, "no_total": 15}
    cluster = {
        "solver_prompt_fingerprint": "cluster-" + candidate_id,
        "root_id": ROOT, "canonical_representative_candidate_id": candidate_id,
        "metrics": {} if missing_metrics else {"M_clean_policy": dict(metric), "M_inclusive_policy": dict(metric)},
    }
    entry = {"candidate_id": candidate_id, "candidate": qa, "normalized_experiment_arm": "E1Q1_FEEDBACK_GUIDED"}
    return backend_record(entry, cluster, "explicit-manifest-sha")


def later_scope_inputs():
    candidate = record("later", "ordering")
    qa = candidate["historical_qa"]
    entry = {"candidate_id": "later", "split": "train", "corpus_id": "E1Q1",
             "normalized_experiment_arm": "E1Q1_FEEDBACK_GUIDED", "context_signature": "dialogue-sha",
             "candidate": {"candidate_id": "later", "root_id": ROOT,
                           "question_type": qa["question_type"], "question": qa["question"],
                           "items": qa["options_or_items"], "gold": qa["canonical_gold"]}}
    scope = {"candidate_id": "later", "root_id": ROOT, "split": "train", "corpus_id": "E1Q1",
             "question_type": "ordering", "normalized_experiment_arm": "E1Q1_FEEDBACK_GUIDED",
             "source_path": "explicit-source-path", "solver_prompt_fingerprint": "cluster-later"}
    return entry, scope, {ROOT: {"split": "train"}}


class RendererTests(unittest.TestCase):
    def test_fresh_history_mode_is_distinct_from_absent_feedback(self):
        envelope = history_envelope(ROOT)
        output = render_role("fresh", envelope, DIALOGUE, {"allowed"}, schema=SCHEMA)
        self.assertEqual(output["feedback_status"], "HISTORY_ONLY_NO_SOURCE_BINDING")
        with self.assertRaisesRegex(ValueError, "history_only_status_is_fresh_only"):
            render_role("targeted", envelope, DIALOGUE, {"allowed"}, schema=SCHEMA)

    def test_complete_qa_and_observations_reach_messages_unchanged(self):
        source = record()
        envelope = feedback_envelope(ROOT, [source])
        before = copy.deepcopy((source, envelope, DIALOGUE))
        for role in ("expansion", "targeted"):
            with self.subTest(role=role):
                output = render_role(role, envelope, DIALOGUE, {"allowed"}, schema=SCHEMA, strategy_schema=STRATEGY)
                view = output["feedback_view"]["feedback"][0] if role == "expansion" else output["feedback_view"]
                self.assertEqual(view["historical_qa"], source["historical_qa"])
                self.assertEqual(view["solver_observations"], source["performance"])
                metric = view["solver_observations"]["M_clean_policy"]
                self.assertEqual(metric["full_context"]["accuracy"], 2 / 12)
                self.assertEqual(metric["no_context"]["accuracy"], 1 / 15)
                prompt = output["messages"][0]["content"]
                self.assertIn("Alpha full option", prompt)
                self.assertIn('"canonical_gold":["A","C"]', prompt)
                self.assertEqual(prompt.count(DIALOGUE["turns"][0]["text"]), 1)
                self.assertNotIn("explicit-manifest-sha", prompt)
                self.assertFalse(output["api_call_authorized"])
        self.assertEqual((source, envelope, DIALOGUE), before)

    def test_unknown_metrics_and_missing_evidence_are_not_fabricated(self):
        source = record(missing_metrics=True)
        validate_envelope(feedback_envelope(ROOT, [source]), {"allowed"})
        self.assertEqual(source["historical_qa"]["evidence_refs"], [])
        self.assertEqual(source["performance"]["M_clean_policy"]["full_context"],
                         {"accuracy": "UNKNOWN", "sample_size": "UNKNOWN"})
        self.assertNotIn("M0", source["performance"])

    def test_later_ordering_is_preserved_for_targeted_and_composition(self):
        ordering, other = record("later", "ordering"), record("other")
        targeted = render_role("targeted", feedback_envelope(ROOT, [ordering]), DIALOGUE,
                               {"later"}, schema=SCHEMA, strategy_schema=STRATEGY)
        self.assertEqual(targeted["feedback_view"]["historical_qa"]["canonical_gold"], ["I2", "I1"])
        output = render_role("composition", feedback_envelope(ROOT, [ordering, other]), DIALOGUE,
                             {"later", "other"}, schema=SCHEMA)
        self.assertEqual(output["feedback_view"]["source_count"], 2)
        self.assertIn("First full item", output["messages"][0]["content"])
        self.assertNotIn("solver_observations", output["feedback_view"])

    def test_explicit_absence_continues_expansion_and_fresh_only(self):
        envelope = feedback_envelope(ROOT, [], "No sources pass the scope and existing eligibility rules.")
        for role in ("expansion", "fresh", "targeted", "composition"):
            with self.subTest(role=role):
                output = render_role(role, envelope, DIALOGUE, set(), schema=SCHEMA)
                self.assertEqual(output["feedback_status"], NO_FEEDBACK)
                if role in {"targeted", "composition"}:
                    self.assertEqual(output["dispatch_status"], "UNSCHEDULABLE_INSUFFICIENT_SOURCES")
                    self.assertEqual(output["messages"], [])
                else:
                    self.assertEqual(output["dispatch_status"], "OFFLINE_RENDERED_NOT_SENT")
                    self.assertEqual(len(output["messages"]), 1)

    def test_missing_or_inconsistent_absence_is_rejected(self):
        invalid = [{}, feedback_envelope(ROOT, []),
                   {**feedback_envelope(ROOT, [], "valid reason"), "records": [record()]},
                   {**feedback_envelope(ROOT, [record()]), "records": []}]
        for envelope in invalid:
            with self.subTest(envelope=envelope):
                with self.assertRaises(ValueError):
                    render_role("expansion", envelope, DIALOGUE, {"allowed"})

    def test_wrong_scope_root_duplicates_and_missing_schemas_are_rejected(self):
        source = record()
        cases = [([source], set(), DIALOGUE), ([source, source], {"allowed"}, DIALOGUE),
                 ([source], {"allowed"}, {**DIALOGUE, "root_id": "other"})]
        for sources, allowed, dialogue in cases:
            with self.subTest(allowed=allowed, sources=len(sources)):
                with self.assertRaises(ValueError):
                    render_role("expansion", feedback_envelope(ROOT, sources), dialogue, allowed)
        with self.assertRaisesRegex(ValueError, "qa_schema_required"):
            render_role("targeted", feedback_envelope(ROOT, [source]), DIALOGUE, {"allowed"})
        with self.assertRaisesRegex(ValueError, "targeted_strategy_schema_required"):
            render_role("targeted", feedback_envelope(ROOT, [source]), DIALOGUE, {"allowed"}, schema=SCHEMA)

    def test_fresh_exposes_only_bounded_coverage_fields(self):
        history = [{"root_id": ROOT, "normalized_stem_signature": "old-ordering-history-signature",
                    "prompt_cluster_count": 2, "candidate_count": 3, "family_saturation_multiplier": 0.5,
                    "gold": "secret-history-gold", "frontier": "secret-history-frontier", "accuracy": 0.123}]
        output = render_role("fresh", feedback_envelope(ROOT, [], "no active source"), DIALOGUE,
                             set(), schema=SCHEMA, history=history)
        self.assertEqual(output["feedback_view"]["historical_coverage"], [{
            "normalized_stem_signature": "old-ordering-history-signature", "prompt_cluster_count": 2, "candidate_count": 3}])
        prompt = output["messages"][0]["content"]
        for forbidden in ("secret-history-gold", "secret-history-frontier", "family_saturation_multiplier", "source_candidate_id"):
            self.assertNotIn(forbidden, prompt)
        with self.assertRaisesRegex(ValueError, "fresh_history_limit_exceeded"):
            render_role("fresh", feedback_envelope(ROOT, [], "no active source"), DIALOGUE,
                        set(), schema=SCHEMA, history=history * 11)

    def test_rendering_reads_no_files_and_opens_no_network(self):
        source = record()
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
                patch("pathlib.Path.open", side_effect=AssertionError("unexpected path read")), \
                patch("socket.socket", side_effect=AssertionError("unexpected network")):
            render_role("targeted", feedback_envelope(ROOT, [source]), DIALOGUE,
                        {"allowed"}, schema=SCHEMA, strategy_schema=STRATEGY)
            render_role("expansion", feedback_envelope(ROOT, [source]), DIALOGUE, {"allowed"})

    def test_inconsistent_source_data_status_is_rejected(self):
        source = record()
        source["historical_qa"]["source_data_status"]["options_or_items"] = "missing"
        with self.assertRaises(ValueError):
            render_role("expansion", feedback_envelope(ROOT, [source]), DIALOGUE, {"allowed"})

    def test_unexpected_performance_cannot_reintroduce_excluded_payload(self):
        source = record()
        source["performance"]["M0"] = {"excluded_candidate_id": "old-ordering", "gold": "excluded-secret"}
        with self.assertRaises(ValueError):
            render_role("expansion", feedback_envelope(ROOT, [source]), DIALOGUE, {"allowed"})

    def test_backend_record_requires_matching_observation_identity(self):
        entry, _, _ = later_scope_inputs()
        wrong_cluster = {"solver_prompt_fingerprint": "wrong", "root_id": "other-root",
                         "canonical_representative_candidate_id": "other-candidate", "metrics": {}}
        with self.assertRaises(ValueError):
            backend_record(entry, wrong_cluster, "explicit-manifest-sha")


class SourceIdentityTests(unittest.TestCase):
    def test_explicit_manifest_source_hash_is_verified_on_every_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "explicit.json"
            source.write_text('{"identity":"first"}', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"filter_version": VERSION, "inputs": {
                "bank": {"path": str(source), "sha256": file_sha(source)}}}), encoding="utf-8")
            sources = Sources(manifest)
            self.assertEqual(sources.get("bank"), {"identity": "first"})
            self.assertTrue(all(row["unchanged"] for row in sources.unchanged()))
            with self.assertRaises(KeyError):
                sources.get("missing_no_default_fallback")
            source.write_text('{"identity":"changed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_sha_mismatch"):
                sources.get("bank")

    def test_later_ordering_is_kept_without_heuristic_round_inference(self):
        entry, scope, splits = later_scope_inputs()
        excluded, kept = scope_filter([entry], [scope], {}, {}, splits)
        self.assertEqual(excluded, [])
        self.assertEqual([row["candidate_id"] for row in kept], ["later"])

    def test_train_name_cannot_override_authoritative_split(self):
        entry, scope, splits = later_scope_inputs()
        splits[ROOT]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "non_train_candidate"):
            scope_filter([entry], [scope], {}, {}, splits)

    def test_nested_candidate_identity_must_match_bank_and_scope(self):
        entry, scope, splits = later_scope_inputs()
        entry["candidate"]["candidate_id"] = "wrong-candidate"
        with self.assertRaises(ValueError):
            scope_filter([entry], [scope], {}, {}, splits)

    def test_later_lineage_must_match_frozen_scope(self):
        entry, scope, splits = later_scope_inputs()
        scope["normalized_experiment_arm"] = "E0Q0_CONTROL_BLIND"
        with self.assertRaises(ValueError):
            scope_filter([entry], [scope], {}, {}, splits)


if __name__ == "__main__":
    unittest.main()
