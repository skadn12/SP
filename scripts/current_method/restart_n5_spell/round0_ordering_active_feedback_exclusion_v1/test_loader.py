"""Mutation fixtures test loading boundaries, not a real new R03 snapshot."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from . import READY, VERSION
from .loader import BOUND, REQUIRED_ARTIFACTS, bind_snapshot, load_export, validate_export_data
from .renderers import backend_record, feedback_envelope
from .sources import value_sha


ROOT = "root-named-train-with-explicit-split"
MANIFEST_SHA = "a" * 64
REFERENCE_FILE_SHA = "b" * 64
REFERENCE_CONTENT_SHA = "c" * 64
NEW_FILE_SHA = "d" * 64
NEW_CONTENT_SHA = "e" * 64
SCHEMA = {"type": "object", "properties": {"fixture_marker": {"type": "string"}}}


def fixture():
    artifacts = {name: [] for name in REQUIRED_ARTIFACTS}
    records, scope, bank, clusters, pools = [], [], [], [], []
    for cid, qtype in (("single", "single_choice"), ("later", "ordering")):
        qa = {"candidate_id": cid, "root_id": ROOT, "question_type": qtype,
              "question": f"Complete fixture {cid} question?", "options": {"A": "Option A", "B": "Option B"},
              "gold": ["B", "A"] if qtype == "ordering" else ["A"], "evidence_refs": ["R02:S01:T001"]}
        entry = {"candidate_id": cid, "candidate": qa, "split": "train", "normalized_experiment_arm": "E1Q1_FEEDBACK_GUIDED", "context_signature": REFERENCE_CONTENT_SHA}
        metric = {"full_correct": 2, "full_total": 3, "no_correct": 0, "no_total": 3}
        cluster = {"solver_prompt_fingerprint": f"cluster-{cid}", "root_id": ROOT,
                   "member_candidate_ids": [cid], "canonical_representative_candidate_id": cid,
                   "active_observation_candidate_id": cid, "active_observation_status": "RETAINED_INDEPENDENT_OBSERVATION",
                   "metrics": {"M_clean_policy": dict(metric), "M_inclusive_policy": dict(metric)}}
        scope.append({"candidate_id": cid, "root_id": ROOT, "split": "train", "filter_version": VERSION,
                      "question_type": qtype, "active_feedback_scope_allowed": True,
                      "normalized_experiment_arm": "E1Q1_FEEDBACK_GUIDED", "solver_prompt_fingerprint": f"cluster-{cid}",
                      "candidate_record_sha256": value_sha(qa), "bank_record_sha256": value_sha(entry)})
        bank.append(entry)
        clusters.append(cluster)
        records.append(backend_record(entry, cluster, MANIFEST_SHA))
        pools.append({"root_id": ROOT, "canonical_representative_candidate_id": cid, "solver_prompt_fingerprint": f"cluster-{cid}"})
    artifacts.update({
        "retained_candidate_scope.jsonl": scope,
        "active_question_bank_projection.jsonl": bank,
        "active_clusters.jsonl": clusters,
        "active_shared_feedback_records.jsonl": records,
        "active_expansion_seed_pool.jsonl": copy.deepcopy(pools),
        "active_expansion_selection.jsonl": [copy.deepcopy(pools[0])],
        "active_targeted_pool.jsonl": copy.deepcopy(pools),
        "active_composition_parent_pool.jsonl": copy.deepcopy(pools),
        "active_composition_pairs.jsonl": [{"root_id": ROOT, "left_candidate_id": "single", "right_candidate_id": "later", "left_cluster_id": "cluster-single", "right_cluster_id": "cluster-later"}],
        "final_report.json": {"status": READY}, "offline_acceptance.json": {"passed": True},
        "implementation_identity.json": {"source_manifest": "/synthetic/source_manifest.json", "source_manifest_sha256": MANIFEST_SHA},
    })
    artifacts["role_feedback_envelopes.jsonl"] = [
        {"role": "expansion", "selection_key": f"expansion:{ROOT}", **feedback_envelope(ROOT, [records[0]])},
        {"role": "fresh", "selection_key": f"fresh:{ROOT}", **feedback_envelope(ROOT, [], "History only")},
        {"role": "targeted", "selection_key": "targeted:single", **feedback_envelope(ROOT, [records[0]])},
        {"role": "targeted", "selection_key": "targeted:later", **feedback_envelope(ROOT, [records[1]])},
        {"role": "composition", "selection_key": f"composition:{ROOT}:cluster-single:cluster-later", **feedback_envelope(ROOT, records)},
    ]
    return artifacts


def loaded_fixture():
    artifacts = fixture()
    loaded = {"artifacts": artifacts, "source_manifest_sha256": MANIFEST_SHA,
              "export_manifest": "/synthetic/export/artifact_inventory.json", "export_manifest_sha256": "f" * 64}
    loaded.update(validate_export_data(artifacts, MANIFEST_SHA))
    loaded["reference_identity"] = {ROOT: {"root_id": ROOT, "input_dialogue_path": "/synthetic/old.json",
        "input_dialogue_file_sha256": REFERENCE_FILE_SHA, "complete_dialogue_sha256": REFERENCE_CONTENT_SHA,
        "source_prefix": "Parent+R01+R02-feedback"}}
    loaded["validated_payload_sha256"] = value_sha({key: loaded[key] for key in (
        "artifacts", "scope", "bank", "clusters", "records", "envelopes", "roots", "reference_identity",
    )})
    old = {"root_id": ROOT, "complete_dialogue_sha256": REFERENCE_CONTENT_SHA,
           "turns": [{"turn_ref": "R02:S01:T001", "speaker": "Alice", "text": "Frozen feedback prefix."}]}
    new = {"root_id": ROOT, "complete_dialogue_sha256": NEW_CONTENT_SHA,
           "turns": old["turns"] + [{"turn_ref": "R03:S01:T001", "speaker": "Alice", "text": "New frozen R03 fixture turn."}]}
    identities = [{"root_id": ROOT, "input_dialogue_path": "/synthetic/new.json",
                   "input_dialogue_file_sha256": NEW_FILE_SHA, "complete_dialogue_sha256": NEW_CONTENT_SHA,
                   "source_prefix": "Parent+R01+R02-feedback+R03", "status": "FROZEN_NEW_R03"}]
    files = {"/synthetic/r03.json": identities, "/synthetic/old.json": old, "/synthetic/new.json": new}
    return loaded, files


class LoaderTests(unittest.TestCase):
    def test_record_and_role_membership_validated(self):
        result = validate_export_data(fixture(), MANIFEST_SHA)
        self.assertEqual(set(result["records"]), {"single", "later"})
        self.assertEqual(len(result["envelopes"]), 5)

    def test_record_performance_forgery_rejected(self):
        artifacts = fixture()
        artifacts["active_shared_feedback_records.jsonl"][0]["performance"]["M_clean_policy"]["full_context"]["accuracy"] = 0.123
        with self.assertRaisesRegex(ValueError, "export_feedback_record_sha_mismatch"):
            validate_export_data(artifacts, MANIFEST_SHA)

    def test_forged_no_feedback_with_available_target_is_rejected(self):
        artifacts = fixture()
        artifacts["role_feedback_envelopes.jsonl"][2] = {
            "role": "targeted", "selection_key": "targeted:single", **feedback_envelope(ROOT, [], "Forged empty feedback")}
        with self.assertRaisesRegex(ValueError, "forged_no_feedback_or_role_source_membership"):
            validate_export_data(artifacts, MANIFEST_SHA)

    def test_missing_target_envelope_is_rejected(self):
        artifacts = fixture()
        artifacts["role_feedback_envelopes.jsonl"].pop(2)
        with self.assertRaisesRegex(ValueError, "export_role_envelope_selection_coverage_mismatch"):
            validate_export_data(artifacts, MANIFEST_SHA)

    def test_forged_composition_pair_is_rejected(self):
        artifacts = fixture()
        artifacts["active_composition_pairs.jsonl"][0]["right_candidate_id"] = "single"
        with self.assertRaisesRegex(ValueError, "export_composition_pair_membership_mismatch"):
            validate_export_data(artifacts, MANIFEST_SHA)

    def test_dev_scope_rejected_despite_root_name(self):
        artifacts = fixture()
        artifacts["retained_candidate_scope.jsonl"][0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "export_non_train_or_disallowed_scope"):
            validate_export_data(artifacts, MANIFEST_SHA)

    def test_new_r03_fixture_renders_all_exported_roles_offline(self):
        loaded, files = loaded_fixture()
        with patch(__package__ + ".loader.read", side_effect=lambda path: files[str(path)]), \
             patch(__package__ + ".loader._checked", side_effect=lambda path, sha: Path(path)), \
             patch(__package__ + ".loader._source", side_effect=lambda obj, name: {"strategy_space": ["fixture"]} if name == "strategy_schema" else SCHEMA), \
             patch(__package__ + ".loader.file_sha", return_value="f" * 64):
            result = bind_snapshot(loaded, "/synthetic/r03.json")
        self.assertEqual(result["status"], BOUND)
        self.assertEqual(result["external_calls"], 0)
        self.assertEqual(len(result["rendered"]), 5)
        self.assertTrue(all(not row["api_call_authorized"] for row in result["rendered"]))
        self.assertTrue(all("New frozen R03 fixture turn." in row["messages"][0]["content"] for row in result["rendered"]))

    def test_current_r02_manifest_rejected(self):
        loaded, files = loaded_fixture()
        files["/synthetic/r03.json"][0]["source_prefix"] = "Parent+R01+R02-feedback"
        with patch(__package__ + ".loader.read", side_effect=lambda path: files[str(path)]):
            with self.assertRaisesRegex(ValueError, "new_frozen_r03_required_r02_rejected"):
                bind_snapshot(loaded, "/synthetic/r03.json")

    def test_new_r03_cannot_rewrite_feedback_prefix(self):
        loaded, files = loaded_fixture()
        files["/synthetic/new.json"]["turns"][0] = {"turn_ref": "R02:S01:T001", "speaker": "Alice", "text": "Rewritten control prefix."}
        with patch(__package__ + ".loader.read", side_effect=lambda path: files[str(path)]), \
             patch(__package__ + ".loader._checked", side_effect=lambda path, sha: Path(path)):
            with self.assertRaisesRegex(ValueError, "r03_feedback_prefix_not_preserved"):
                bind_snapshot(loaded, "/synthetic/r03.json")

    def test_missing_evidence_rejected_on_new_snapshot(self):
        loaded, files = loaded_fixture()
        loaded["envelopes"][f"expansion:{ROOT}"]["records"][0]["historical_qa"]["evidence_refs"] = ["R02:MISSING"]
        loaded["validated_payload_sha256"] = value_sha({key: loaded[key] for key in (
            "artifacts", "scope", "bank", "clusters", "records", "envelopes", "roots", "reference_identity",
        )})
        with patch(__package__ + ".loader.read", side_effect=lambda path: files[str(path)]), \
             patch(__package__ + ".loader._checked", side_effect=lambda path, sha: Path(path)):
            with self.assertRaisesRegex(ValueError, "new_r03_source_evidence_missing"):
                bind_snapshot(loaded, "/synthetic/r03.json")

    def test_loaded_payload_mutation_rejected(self):
        loaded, files = loaded_fixture()
        loaded["envelopes"][f"expansion:{ROOT}"]["records"] = []
        with self.assertRaisesRegex(ValueError, "loaded_export_mutated_after_validation"):
            bind_snapshot(loaded, "/synthetic/r03.json")

    def test_load_export_checks_inventory_and_source_sha(self):
        artifacts = fixture()
        manifest_path = "/synthetic/export/artifact_inventory.json"
        source = {"filter_version": VERSION, "real_calls_authorized": False, "inputs": {}}
        inventory = [{"path": "/synthetic/export/" + name, "sha256": value_sha(value)} for name, value in artifacts.items()]
        files = {row["path"]: artifacts[Path(row["path"]).name] for row in inventory}
        files[manifest_path] = inventory
        files["/synthetic/source_manifest.json"] = source
        hashes = {path: value_sha(value) for path, value in files.items()}
        hashes["/synthetic/source_manifest.json"] = MANIFEST_SHA
        reference = [{"root_id": ROOT}]
        with patch(__package__ + ".loader.read", side_effect=lambda path: files[str(path)]), \
             patch(__package__ + ".loader.file_sha", side_effect=lambda path: hashes[str(path)]), \
             patch(__package__ + ".loader._sidecar_sha", return_value=hashes[manifest_path]), \
             patch(__package__ + ".loader._source", side_effect=lambda obj, name: [{"root_id": ROOT, "split": "train"}] if name == "round0_split" else reference):
            result = load_export("/synthetic/export")
            self.assertEqual(result["roots"], [ROOT])
            hashes["/synthetic/source_manifest.json"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "export_or_source_sha_mismatch"):
                load_export(manifest_path)


if __name__ == "__main__":
    unittest.main()
