"""Truth/identity preservation under the new human/future-training display."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import display_export as d


def question(kind):
    return {"candidate_id": "fixture-" + kind, "root_id": "fixture", "question_type": kind,
            "question": "Synthetic question.",
            ("options" if kind == "multi_select" else "items"):
                {label: "Statement " + label for label in
                 ("ABCDEF" if kind == "multi_select" else ["I"+str(i) for i in range(1,7)])},
            "gold": list("ACF") if kind == "multi_select" else ["I3","I1","I6","I2","I4","I5"]}


class DisplayTests(unittest.TestCase):
    def test_mapping_preserves_true_statement_set_and_chronological_content_order(self):
        for kind in ("multi_select","ordering"):
            q=question(kind);before=deepcopy(q);field="options" if kind=="multi_select" else "items"
            for seed in ("first","second","third"):
                v=d.display_view(q,seed=seed)
                original=[q[field][k] for k in q["gold"]]
                actual=[v[field][k] for k in v["gold"]]
                self.assertEqual(set(actual),set(original))
                if kind=="ordering":self.assertEqual(actual,original)
                self.assertEqual(set(v[field].values()),set(q[field].values()))
                self.assertEqual({old:now for now,old in v["display_to_original"].items()},v["original_to_display"])
                self.assertTrue(d.replay_view(q,v))
            self.assertEqual(q,before)

    def test_permutation_is_gold_score_evidence_blind_and_input_dict_order_independent(self):
        for kind in ("multi_select","ordering"):
            q=question(kind);changed=deepcopy(q)
            changed["gold"]=list(reversed(q["gold"])) if kind=="ordering" else list("BD")
            changed["option_evidence"]={"SECRET":"PRIVATE"}
            changed["training_selection"]={"full_partial_mean":"1/2"}
            field="options" if kind=="multi_select" else "items"
            changed[field]=dict(reversed(list(changed[field].items())))
            v=d.display_view(q);v2=d.display_view(changed)
            self.assertEqual(v[field],v2[field])
            self.assertNotIn("SECRET",json.dumps(v2))
            self.assertNotIn("full_partial_mean",json.dumps(v2))

    def test_all_true_remains_all_true_and_is_flagged_not_silently_relabelled_false(self):
        q=question("multi_select");q["gold"]=list("ABCDEF")
        v=d.display_view(q)
        self.assertEqual(v["gold"],list("ABCDEF"))
        self.assertEqual(d.display_stats([v])["multi_all_six"],1)
        self.assertEqual(d.display_stats([v])["multi_outside_two_to_four"],1)
        self.assertIn("未修改旧题真假",d.markdown([v]))

    def test_tampered_display_or_gold_fails_replay(self):
        q=question("ordering")
        for target in ("gold","items","display_to_original"):
            v=d.display_view(q)
            if target=="gold":v["gold"].reverse()
            else:v[target]["I1"]="CORRUPTED"
            with self.assertRaises(ValueError):d.replay_view(q,v)

    def test_file_exports_have_same_mapping_and_remain_provisional(self):
        with tempfile.TemporaryDirectory() as folder:
            qs=[question("multi_select"),question("ordering")]
            views=d.write_views(folder,qs)
            saved=[json.loads(line) for line in (Path(folder)/"DISPLAY_CANDIDATES.jsonl").read_text().splitlines()]
            self.assertEqual(views,saved)
            for q,v in zip(qs,saved):
                self.assertFalse(v["training_allowed"]);d.replay_view(q,v)
            with self.assertRaises(FileExistsError):d.write_views(folder,qs)

    def test_bad_gold_and_duplicate_texts_rejected(self):
        for kind in ("multi_select","ordering"):
            q=question(kind);q["gold"]=["UNKNOWN"]
            with self.assertRaises(ValueError):d.display_view(q)
        q=question("multi_select");q["options"]["B"]=q["options"]["A"]
        with self.assertRaises(ValueError):d.display_view(q)


if __name__=="__main__":
    unittest.main()
