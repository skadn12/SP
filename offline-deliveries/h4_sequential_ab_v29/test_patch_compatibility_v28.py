"""Replay exact Qwen patch responses; redundant fields cannot alter semantics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import event_state as scan


class RecordedPatchCompatibility(unittest.TestCase):
    def test_both_real_qwen_patch_forms_preserve_every_judgment(self):
        parent = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v27/runs/repair_probe_01"
        for name in ("absence_timeline", "new_development_timeline"):
            case = json.loads((parent / "cases" / (name + ".json")).read_text())
            request = next((parent / "requests").glob("*-" + name))
            patch = json.loads(json.loads((request / "RESPONSE.json").read_text())["text"])
            c, old = case["context"], json.loads(case["previous_raw"])
            merged = scan.merge_event_repair(old, patch, case["repair"], c["cumulative_history"])
            scan.validate_repair_preservation(old, merged, case["repair"])
            scan.validate(merged, c["root_id"], c["route"], c["scene_plan"], c["cumulative_history"])
            self.assertEqual(merged["route_fulfillment"], old["route_fulfillment"])
            for before, after in zip(old["plan_checks"], merged["plan_checks"]):
                self.assertEqual({k:v for k,v in before.items() if k != "constraint_ids"},
                                 {k:v for k,v in after.items() if k != "constraint_ids"})
            invalid = deepcopy(patch)
            invalid["plan_check_repairs"][0]["constraint_ids"] = ["invented"]
            with self.assertRaises(ValueError):
                scan.merge_event_repair(old, invalid, case["repair"], c["cumulative_history"])


if __name__ == "__main__": unittest.main()
