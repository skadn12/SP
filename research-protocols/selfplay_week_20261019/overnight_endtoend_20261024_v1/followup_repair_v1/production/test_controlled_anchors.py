"""CPU-only citation protocol regressions; fixtures are not new model calls."""
from copy import deepcopy
import unittest

from controlled_anchors import expand_review, make_catalog
from production_core_v3 import ProductionError, validate_review
from test_repair import fixture


class ControlledAnchors(unittest.TestCase):
    def setUp(self):
        self.packet, self.source, self.legacy = fixture(24719, "GENERIC", "OET1:B:S02", raw=True)
        self.scene = "OET1:B:S02"
        self.catalog = make_catalog(self.packet, self.source, self.scene)
        refs = {item["turn_ref"]: item["anchor_id"] for item in self.catalog["anchors"]}
        from production_core_v3 import EvidenceResolver
        resolver = EvidenceResolver(self.packet, self.source, self.scene)
        self.raw = deepcopy(self.legacy)
        self.raw["catalog_id"] = self.catalog["catalog_id"]
        for check in self.raw["checks"].values():
            check["evidence"] = [{"anchor_id": refs[resolver.resolve(e["turn_ref"])[0]]}
                                 for e in check["evidence"]]
        for finding in self.raw["findings"]:
            finding["anchor_ids"] = [refs[resolver.resolve(ref)[0]] for ref in finding.pop("turn_refs")]

    def expand(self):
        return expand_review(self.raw, self.catalog, self.packet, self.source, self.scene)

    def test_exact_selected_anchors_and_no_mutation(self):
        before = deepcopy(self.raw)
        expanded, mapping = self.expand()
        result = validate_review(expanded, self.packet, self.source, self.scene)
        self.assertTrue(result["semantic_admitted"])
        self.assertEqual(self.raw, before)
        self.assertEqual(len(mapping["mappings"]), 22)
        self.assertFalse(mapping["controller_chose_evidence"])
        self.assertEqual(self.catalog, make_catalog(self.packet, self.source, self.scene))

    def test_free_quote_or_reference_cannot_override_anchor(self):
        for key, value in (("quote", "wrong quote"), ("turn_ref", "R02:A:S03:T013")):
            with self.subTest(key=key):
                raw = deepcopy(self.raw)
                next(iter(raw["checks"].values()))["evidence"][0][key] = value
                with self.assertRaises(ProductionError):
                    expand_review(raw, self.catalog, self.packet, self.source, self.scene)

    def test_changed_or_foreign_catalog_rejected(self):
        for mutation in ("quote", "root", "native"):
            changed = deepcopy(self.catalog)
            if mutation == "quote":
                changed["anchors"][0]["quote"] = " "
            elif mutation == "root":
                changed["root_id"] = "different root"
            else:
                changed["native_event_binding"] = self.packet["native_events"][0]["native_event_binding"]
            with self.subTest(mutation=mutation), self.assertRaises(ProductionError):
                expand_review(self.raw, changed, self.packet, self.source, self.scene)

    def test_unknown_anchor_and_wrong_catalog_id_rejected(self):
        for value in ("UNKNOWN", " ", None):
            raw = deepcopy(self.raw)
            next(iter(raw["checks"].values()))["evidence"] = [{"anchor_id": value}]
            with self.subTest(value=value), self.assertRaises(ProductionError):
                expand_review(raw, self.catalog, self.packet, self.source, self.scene)
        self.raw["catalog_id"] = "wrong"
        with self.assertRaises(ProductionError):
            self.expand()

    def test_empty_source_cannot_become_anchor(self):
        self.source["turns"][0]["text"] = "\n  "
        with self.assertRaises(ProductionError):
            make_catalog(self.packet, self.source, self.scene)

    def test_semantic_gates_survive_expansion(self):
        for mutation in ("hold", "check_fail", "blocking", "ambiguity", "code", "count"):
            raw = deepcopy(self.raw)
            if mutation == "hold": raw["decision"] = "HOLD"
            elif mutation == "check_fail": next(iter(raw["checks"].values()))["status"] = "FAIL"
            elif mutation == "ambiguity": raw["source_or_gold_ambiguity_failure"] = True
            elif mutation == "code": raw["failure_codes"] = ["UNSUPPORTED_FACT"]
            elif mutation == "count": raw["source_turn_count"] += 1
            else:
                raw["findings"].append({"severity": "BLOCKING", "dimension": "time", "reason": "bad time",
                                        "anchor_ids": [self.catalog["anchors"][0]["anchor_id"]]})
            expanded, _ = expand_review(raw, self.catalog, self.packet, self.source, self.scene)
            with self.subTest(mutation=mutation), self.assertRaises(ProductionError):
                validate_review(expanded, self.packet, self.source, self.scene)


if __name__ == "__main__":
    unittest.main()
