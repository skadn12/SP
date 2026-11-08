import unittest
from copy import deepcopy

import evidence_refs as e


class EvidenceReferences(unittest.TestCase):
    def setUp(self):
        self.history = [{"turn_ref": "R02:A:S01:T001", "speaker": "Emily",
                         "text": "I’ll book it tomorrow — not today."}]
        self.new = [{"speaker": "Emily", "text": "I booked it this morning."}]

    def test_exact_unicode_roundtrip_and_provenance(self):
        raw = {"verdict": "contradicted", "history_evidence": [{"turn_ref": self.history[0]["turn_ref"]}],
               "new_evidence": [{"turn_ref": "NEW:T001"}]}
        original = deepcopy(raw)
        result = e.materialize_citations(raw, self.history, self.new)
        self.assertEqual(raw, original)
        self.assertEqual(result["history_evidence"][0]["quote"], self.history[0]["text"])
        self.assertEqual(result["new_evidence"][0]["quote"], self.new[0]["text"])
        self.assertEqual(result["verdict"], "contradicted")
        provenance = e.citation_provenance(raw, self.history, self.new)
        self.assertFalse(provenance["semantic_support_verified"])
        self.assertEqual(len(provenance["dereferenced_citations"]), 2)
        changed = deepcopy(self.history)
        changed[0]["text"] += " Changed."
        self.assertNotEqual(provenance["history_sha256"], e.citation_provenance(raw, changed, self.new)["history_sha256"])

    def test_unknown_id_rejected_without_guessing(self):
        with self.assertRaisesRegex(ValueError, "not_in_context"):
            e.materialize_citations({"turn_ref": "R02:A:S01:T002"}, self.history)

    def test_ambiguous_namespace_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate_reference"):
            e.materialize_citations({}, self.history, self.history)

    def test_model_written_invalid_quote_not_silently_repaired(self):
        raw = {"evidence": [{"turn_ref": self.history[0]["turn_ref"], "quote": "I’ll book..."}]}
        self.assertEqual(e.materialize_citations(raw, self.history), raw)
        self.assertEqual(e.citation_provenance(raw, self.history)["dereferenced_citations"], [])

    def test_no_default_verdict_or_missing_check_added(self):
        raw = {"checks": [], "reason": "uncertain"}
        self.assertEqual(e.materialize_citations(raw, self.history), raw)

    def test_bare_reference_is_equivalent_only_in_evidence_arrays(self):
        ref = self.history[0]["turn_ref"]
        raw = {"history_evidence": [ref], "constraint_ids": [ref], "reason": ref}
        checked = e.materialize_citations(raw, self.history)
        self.assertEqual(checked["history_evidence"], [{"turn_ref": ref, "quote": self.history[0]["text"]}])
        self.assertEqual(checked["constraint_ids"], [ref])
        self.assertEqual(checked["reason"], ref)
        self.assertEqual(len(e.citation_provenance(raw, self.history)["dereferenced_citations"]), 1)
        with self.assertRaisesRegex(ValueError, "not_in_context"):
            e.materialize_citations({"history_evidence": ["invented"]}, self.history)


if __name__ == "__main__":
    unittest.main()
