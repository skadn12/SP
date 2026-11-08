import json
import unittest
from pathlib import Path

from scripts.v3_core import OPERATIONS, build_ledger, compile_root, compile_slot, cross_check_catalog


def _ledger(tmp_path):
    turns = []
    for scene in range(1, 7):
        st = []
        for i in range(1, 9):
            st.append({"turn_ref": f"R03:A:S{scene:02d}:T{i:03d}", "speaker": "A" if i % 2 else "B", "text": f"Plan {i} at {i:02d}:00 with {i} items.", "origin": "codex_synthetic_extension"})
        turns.append({"scene_ref": f"R03:A:S{scene:02d}", "turns": st})
    p = tmp_path / "draft.json"
    p.write_text(json.dumps({"root_id": "root", "scenes": turns, "native_evidence": {}}))
    return build_ledger("root", p)


class V3CoreTests(unittest.TestCase):
    def make_ledger(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        return _ledger(Path(self.tmp.name))

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_truth_assignment_and_quotas(self):
        ledger = self.make_ledger()
        slot = compile_slot(ledger, "F", "MC4")
        self.assertEqual(sum(x["truth"] for x in slot["options"].values()), 4)
        self.assertEqual(slot["gold"], ["A", "B", "C", "D"])


    def test_all_operations_and_spans(self):
        ledger = self.make_ledger()
        catalog = compile_root(ledger, {"F": {}, "G": {}})
        self.assertEqual(len(catalog), 16)
        self.assertFalse(cross_check_catalog(catalog, ledger))
        self.assertEqual({x["slot_id"] for x in catalog}, set(OPERATIONS))


    def test_permutations_are_nonidentity(self):
        ledger = self.make_ledger()
        for slot in compile_root(ledger, {"F": {}, "G": {}}):
            self.assertTrue(all(p["nonidentity_permutation"] for p in slot["presentations"]))


    def test_span_tampering_rejected(self):
        ledger = self.make_ledger()
        slot = compile_slot(ledger, "G", "MC2")
        slot["options"]["A"]["evidence"][0]["source_substring"] = "tampered"
        errors = cross_check_catalog([slot], ledger)
        self.assertTrue(errors)
