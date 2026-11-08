"""A recorded explicit actor rejection must not trigger contradictory repair."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import realization_review as review
from test_actor_review_v29 import strict_fixture, validate

RECORDS = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v29/probes/frozen_cases_01/requests"
FIRST = RECORDS / "004-regression-review_s03_development_actor-1"
SECOND = RECORDS / "005-regression-review_s03_development_actor-2"


def recorded_case():
    payload = json.loads((FIRST / "REQUEST.json").read_text())
    context = json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split("\nREVIEW INPUT DATA:\n", 1)[1])[0]
    raw = json.loads((FIRST / "RESPONSE.json").read_text())["text"]
    return context, raw


def recorded_validate(context, raw):
    return review.validate_review(raw, context["root_id"], context["route"],
        context["scene_plan"], context["new_scene"],
        dataset_card=context["dataset_card_guidance"], latest_state=context["latest_state"],
        event_state=context["event_state"], history=context["branch_history"],
        require_actor_checks=True, require_goal_checks=True)


class NegativeActorReviewTests(unittest.TestCase):
    def test_recorded_004_is_hard_reject_without_schema_retry_or_raw_changes(self):
        context, raw = recorded_case()
        source_bytes = (FIRST / "RESPONSE.json").read_bytes()
        old = json.loads((FIRST / "EVALUATION.json").read_text())
        result = recorded_validate(context, raw)
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["schema_failures"], [])
        # Every existing rejection is retained, including both mechanically
        # observed attribution conflicts; only its retry classification changes.
        self.assertEqual(result["failures"], old["failures"])
        self.assertEqual(result["raw_parsed"], old["raw_parsed"])
        self.assertFalse(result["raw_parsed"]["checks"][3]["actor_checks"][0]["satisfied"])
        self.assertFalse(result["raw_parsed"]["card_checks"][2]["realized"])
        self.assertEqual((FIRST / "RESPONSE.json").read_bytes(), source_bytes)
        spec = review.build_format_repair(result, raw, context["scene_plan"], context["branch_history"],
            context["new_scene"], dataset_card=context["dataset_card_guidance"],
            latest_state=context["latest_state"], event_state=context["event_state"],
            require_actor_checks=True, require_goal_checks=True)
        self.assertEqual(spec["repair_fields"], [])
        self.assertTrue(all(not ids for ids in spec["repair_check_ids"].values()))

    def test_explicit_negative_does_not_bypass_unknown_or_invalid_citations(self):
        context, raw = recorded_case()
        candidate = json.loads(raw)
        candidate["checks"][3]["actor_checks"][0]["evidence"][0]["turn_ref"] = "NEW:T999"
        result = recorded_validate(context, candidate)
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("citation_reference_not_in_context" in e for e in result["schema_failures"]))

    def test_explicit_negative_does_not_bypass_missing_actor_fields(self):
        context, raw = recorded_case()
        candidate = json.loads(raw)
        candidate["checks"][3]["actor_checks"][0].pop("action")
        result = recorded_validate(context, candidate)
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("actor_checks[1]:fields_invalid" in e for e in result["schema_failures"]))

    def test_positive_attribution_conflicts_remain_ineligible_for_acceptance(self):
        scene, card, realization, raw = strict_fixture()
        row = raw["checks"][1]["actor_checks"][0]
        row["evidence"] = [{"turn_ref": "NEW:T001"}]
        result = validate(scene, card, realization, raw)
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("direct_speaker_actor_mismatch" in e for e in result["schema_failures"]))

    def test_recorded_005_protected_rewrite_has_stable_failure_across_hashseeds(self):
        # v29 iterated a set of top-level fields. With multiple illegal edits,
        # changing PYTHONHASHSEED changed the first error and broke audit replay.
        script = '''
import json
from pathlib import Path
import realization_review as review
p = Path(%r)
first = p / "004-regression-review_s03_development_actor-1"
second = p / "005-regression-review_s03_development_actor-2"
raw = json.loads((first / "RESPONSE.json").read_text())["text"]
repair = json.loads((second / "RESPONSE.json").read_text())["text"]
spec = json.loads((first / "FORMAT_REPAIR_SPEC.json").read_text())
try:
    review.merge_review_repair(raw, repair, spec)
except ValueError as error:
    print(str(error))
else:
    raise AssertionError("the protected rewrite unexpectedly merged")
''' % str(RECORDS)
        outcomes = []
        for seed in ("1", "2", "3", "7", "11", "23", "42", "100"):
            outcomes.append(subprocess.check_output([sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parent,
                env={**os.environ, "PYTHONHASHSEED": seed}, text=True).strip())
        self.assertEqual(set(outcomes), {"review_repair_protected_field_changed:card_actor_bindings"})


if __name__ == "__main__":
    unittest.main()
