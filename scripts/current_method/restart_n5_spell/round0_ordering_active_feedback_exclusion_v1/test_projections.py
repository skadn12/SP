"""Projection tests covering alias leakage and aggregation denominators."""

import copy
import math
import unittest

from .projections import MODELS, project_clusters


def fixture():
    clusters, members, origins, eligibility, weights, candidates = [], [], [], [], [], {}
    for fp, ids, correct, total in (("a", ["old", "alias"], 1, 2), ("b", ["later"], 9, 10)):
        representative = ids[0]
        metric = {"full_correct": correct, "full_total": total, "no_correct": 0, "no_total": total, "full": correct / total, "no": 0.0}
        clusters.append({"solver_prompt_fingerprint": fp, "canonical_representative_candidate_id": representative, "member_candidate_ids": ids, "member_count": len(ids), "alias_count": len(ids) - 1, "root_id": "train-name-but-authoritative-root", "normalized_experiment_arm": "ROUND0_PARENT_PLUS_R01", "question_type": "ordering", "normalized_stem_signature": "stem", "evidence_refs": ["R01:T1"], "lifecycle_state": "ACTIVE_FRONTIER_BOTH", "metrics": {model: dict(metric) for model in MODELS}, "auxiliary_labels": {"old_performance": True}})
        for cid in ids:
            members.append({"candidate_id": cid, "solver_prompt_fingerprint": fp, "canonical_representative_candidate_id": representative, "alias": cid != representative})
            origins.append({"candidate_id": cid, "solver_prompt_fingerprint": fp, "canonical_representative_candidate_id": representative, "observation_origin": "INDEPENDENT_OBSERVATION" if cid == representative else representative, "source_independent_observation_available": True, "counted_as_independent_observation": cid == representative})
            candidates[cid] = {"candidate_id": cid, "root_id": "train-name-but-authoritative-root", "normalized_experiment_arm": "E1Q1_FEEDBACK_GUIDED" if cid != "old" else "ROUND0_PARENT_PLUS_R01", "question_type": "ordering", "evidence_refs": ["R02:T1"]}
        eligibility.append({"solver_prompt_fingerprint": fp, "questioner_targeted_evolution_eligible": True, "questioner_bridge_generation_eligible": False, "composition_parent_eligible": True})
        weights.append({"solver_prompt_fingerprint": fp, "canonical_representative_candidate_id": representative, "root_id": "train-name-but-authoritative-root", "eligible": True, "additive_score": 4.0, "evidence_multiplier": 1.0, "cooldown_multiplier": 1.0, "representative_multiplier": 1.0, "family_saturation_multiplier": 1 / math.sqrt(2), "raw_weight": 4 / math.sqrt(2)})
    return dict(clusters=clusters, members=members, origins=origins, eligibility=eligibility, target_weights=weights, composition_weights=copy.deepcopy(weights), candidates=candidates)


class ProjectionTests(unittest.TestCase):
    def test_independent_denominators_and_no_mutation(self):
        inputs = fixture()
        before = copy.deepcopy(inputs)
        result = project_clusters(**inputs, excluded_ids=set())
        model = result["root_aggregates"][0]["models"][MODELS[0]]
        self.assertEqual((model["full_correct"], model["full_total"]), (10, 12))
        self.assertAlmostEqual(model["full"], 10 / 12)
        self.assertEqual(model["independent_observation_count"], 2)
        self.assertEqual(inputs, before)

    def test_removed_representative_cannot_leak_through_alias(self):
        result = project_clusters(**fixture(), excluded_ids={"old"})
        mixed = result["clusters"][0]
        self.assertEqual(mixed["member_candidate_ids"], ["alias"])
        self.assertEqual(mixed["metrics"], {model: None for model in MODELS})
        self.assertEqual(mixed["auxiliary_labels"], {})
        self.assertFalse(result["eligibility"][0]["questioner_targeted_evolution_eligible"])
        self.assertEqual(result["target_weights"][0]["raw_weight"], 0.0)
        self.assertIsNone(result["target_weights"][0]["additive_score"])
        self.assertEqual(result["root_aggregates"][0]["models"][MODELS[0]]["status"], "UNKNOWN")

    def test_retained_independent_representative_survives_mixed_cluster(self):
        result = project_clusters(**fixture(), excluded_ids={"alias"})
        mixed = result["clusters"][0]
        self.assertEqual(mixed["member_candidate_ids"], ["old"])
        self.assertEqual(mixed["metrics"][MODELS[0]]["full_correct"], 1)
        self.assertTrue(result["eligibility"][0]["questioner_targeted_evolution_eligible"])

    def test_family_and_normalization_rebuilt_after_cluster_removed(self):
        result = project_clusters(**fixture(), excluded_ids={"old", "alias"})
        self.assertEqual(len(result["clusters"]), 1)
        weight = result["target_weights"][0]
        self.assertEqual(weight["family_saturation_multiplier"], 1.0)
        self.assertEqual(weight["raw_weight"], 4.0)
        self.assertEqual(weight["global_normalized_weight"], 1.0)
        self.assertEqual(weight["within_root_normalized_weight"], 1.0)

    def test_bad_membership_is_rejected(self):
        inputs = fixture()
        inputs["members"][0]["solver_prompt_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "cluster_membership_or_origin_mismatch"):
            project_clusters(**inputs, excluded_ids=set())

    def test_missing_denominator_disables_active_signal(self):
        inputs = fixture()
        inputs["clusters"][0]["metrics"][MODELS[0]]["full_total"] = None
        result = project_clusters(**inputs, excluded_ids=set())
        self.assertIsNone(result["clusters"][0]["metrics"][MODELS[0]])
        self.assertEqual(result["target_weights"][0]["raw_weight"], 0.0)
        self.assertFalse(result["eligibility"][0]["composition_parent_eligible"])
        self.assertEqual(result["root_aggregates"][0]["status"], "UNKNOWN")

    def test_omitted_declared_member_is_rejected(self):
        inputs = fixture()
        inputs["clusters"][0]["member_candidate_ids"] = ["old"]
        with self.assertRaisesRegex(ValueError, "cluster_membership_or_origin_mismatch"):
            project_clusters(**inputs, excluded_ids=set())


if __name__ == "__main__":
    unittest.main()
