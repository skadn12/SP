"""Regression coverage for contextual citations preceding direct evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from codex_evidence_export import export_question_evidence, export_root_questions


def multiselect_fixture():
    # These two citations reproduce the v113 Nirala truncation failure: the
    # first describes a different restaurant; the second is the direct rebuttal.
    background = {"turn_ref": "R02:B:S01:T005", "speaker": "Rosa",
                  "quote": "the adviser said it was expensive."}
    direct = {"turn_ref": "R02:B:S01:T007", "speaker": "Rosa",
              "quote": "They came back with The Nirala and described it as moderately priced and in the north."}
    source = {"root_id": "fixture-root", "turns": [
        {"turn_ref": background["turn_ref"], "speaker": "Rosa",
         "text": "City Stop was suggested first, but " + background["quote"]},
        {"turn_ref": direct["turn_ref"], "speaker": "Rosa", "text": direct["quote"]}]}
    options = {"A": "Rosa reports that the adviser described The Nirala as expensive and in the north."}
    judgments = {"A": {"proposition": options["A"], "relation": "contradicts", "scope_valid": True,
        "reason": "Expensive described City Stop, while The Nirala was moderate. The proposition attaches the earlier price to the wrong restaurant.",
        "citations": [background, direct], "primary_citation": deepcopy(direct),
        "review_note": "T005 is context; T007 directly contradicts the price attributed to Nirala."}}
    for label in "BCDEF":
        text = f"Rosa reports fact {label}."
        ref = "fixture:" + label
        source["turns"].append({"turn_ref": ref, "speaker": "Rosa", "text": text})
        options[label] = text
        judgments[label] = {"proposition": text, "relation": "supports", "scope_valid": True,
                            "reason": f"Rosa explicitly reports fact {label}.",
                            "citations": [{"turn_ref": ref, "speaker": "Rosa", "quote": text}]}
    question = {"candidate_id": "fixture-question", "root_id": source["root_id"],
                "question_type": "multi_select", "question": "Select all supported statements.", "options": options}
    gold = {"candidate_id": question["candidate_id"], "gold": list("BCDEF"), "role": "new_child",
            "evidence": {label: deepcopy(j["citations"]) for label, j in judgments.items()}}
    review = {"candidate_id": question["candidate_id"], "options": judgments}
    design = {"candidate_id": question["candidate_id"],
              "labels_to_statement_ids": {label: "statement-" + label for label in options}}
    assets = {"statements": {"statement-" + label: deepcopy(j) for label, j in judgments.items()}}
    return question, gold, source, review, assets, design


def ordering_fixture():
    source = {"root_id": "ordering-root", "turns": [
        {"turn_ref": f"T{i}", "speaker": "Rosa", "text": f"Completed action {i}."} for i in range(1, 7)]}
    question = {"candidate_id": "ordering-question", "root_id": source["root_id"], "question_type": "ordering",
                "question": "Order by appearance.", "items": {f"I{i}": f"Completed action {i}." for i in range(1, 7)}}
    js = {}
    for i in range(1, 7):
        js[f"I{i}"] = {"proposition": f"Completed action {i}.", "supported": True, "actor_ok": True,
            "modality_ok": True, "state_ok": True, "neutral_stem_ok": True, "single_event_ok": True,
            "reason": f"This is completed action {i}.", "order_anchor": {"turn_ref": f"T{i}", "quote": f"Completed action {i}."},
            "support_refs": [f"T{i}"], "alternative_matching_anchors": []}
    review = {"candidate_id": question["candidate_id"], "items": js, "derived_order": list(question["items"])}
    gold = {"candidate_id": question["candidate_id"], "gold": list(question["items"]), "role": "ordering",
            "evidence": {label: deepcopy(j["order_anchor"]) for label, j in js.items()}}
    design = {"candidate_id": question["candidate_id"], "fixed_turns": {f"I{i}": f"T{i}" for i in range(1, 7)}}
    return question, gold, source, review, {"statements": {}}, design


class EvidenceExportTests(unittest.TestCase):
    def export(self, fixture):
        q, g, s, j, assets, design = fixture
        return export_question_evidence(q, g, s, j, asset_source_judgments=assets, design=design)

    def test_nirala_retains_context_and_direct_rebuttal_with_real_reason(self):
        fixture = multiselect_fixture(); saved = deepcopy(fixture)
        candidate = self.export(fixture)
        entry = candidate["option_evidence"]["A"]
        self.assertEqual(entry["citations"], fixture[3]["options"]["A"]["citations"])
        self.assertEqual(len(entry["citations"]), 2)
        self.assertEqual(entry["turn_ref"], "R02:B:S01:T007")
        self.assertEqual(entry["quote"], entry["citations"][1]["quote"])
        self.assertEqual(entry["relation"], "contradicts")
        self.assertEqual(entry["reason"], fixture[3]["options"]["A"]["reason"])
        self.assertEqual(entry["asset_source_judgment"], fixture[4]["statements"]["statement-A"])
        self.assertEqual(fixture, saved)

    def test_missing_multi_citation_primary_fails_instead_of_picking_first(self):
        fixture = multiselect_fixture(); del fixture[3]["options"]["A"]["primary_citation"]
        with self.assertRaisesRegex(ValueError, "reviewed_primary_citation_required"):
            self.export(fixture)

    def test_explicit_reviewer_primary_can_migrate_old_review_without_mutation(self):
        q, g, s, j, _, _ = multiselect_fixture(); primary = j["options"]["A"].pop("primary_citation")
        candidate = export_question_evidence(q, g, s, j, primary_citations={"A": primary})
        self.assertEqual(candidate["option_evidence"]["A"]["primary_citation_origin"], "explicit_reviewer_designation")
        self.assertNotIn("primary_citation", j["options"]["A"])

    def test_primary_must_match_full_citation_and_cannot_be_a_new_span(self):
        for field, value in [("speaker", "Diego"), ("quote", "The Nirala")]:
            with self.subTest(field=field):
                fixture = multiselect_fixture(); fixture[3]["options"]["A"]["primary_citation"][field] = value
                with self.assertRaisesRegex(ValueError, "primary_not_in_reviewed_citations"):
                    self.export(fixture)

    def test_bad_secondary_reference_rejected_even_with_valid_primary(self):
        for field, value in [("speaker", "Diego"), ("quote", "Never uttered"), ("turn_ref", "missing")]:
            with self.subTest(field=field):
                fixture = multiselect_fixture()
                # The background citation is secondary despite appearing first.
                fixture[3]["options"]["A"]["citations"][0][field] = value
                fixture[1]["evidence"]["A"] = deepcopy(fixture[3]["options"]["A"]["citations"])
                with self.assertRaisesRegex(ValueError, "citation_(speaker_mismatch|quote_not_exact|turn_ref_invalid)"):
                    self.export(fixture)

    def test_cannot_fill_missing_relation_or_reason_from_gold(self):
        for field, value in [("relation", None), ("relation", "unknown"), ("reason", " ")]:
            with self.subTest(field=field, value=value):
                fixture = multiselect_fixture(); fixture[3]["options"]["A"][field] = value
                with self.assertRaisesRegex(ValueError, "review_(relation_invalid|reason_required)"):
                    self.export(fixture)

    def test_truncated_private_evidence_is_rejected(self):
        fixture = multiselect_fixture(); fixture[1]["evidence"]["A"] = fixture[1]["evidence"]["A"][:1]
        with self.assertRaisesRegex(ValueError, "private_review_citations_mismatch"):
            self.export(fixture)

    def test_wrong_asset_mapping_or_relation_is_rejected(self):
        for change in ["mapping", "relation"]:
            with self.subTest(change=change):
                fixture = multiselect_fixture()
                if change == "mapping": fixture[5]["labels_to_statement_ids"]["A"] = "statement-B"
                else: fixture[4]["statements"]["statement-A"]["relation"] = "supports"
                with self.assertRaisesRegex(ValueError, "asset_(proposition|relation)_mismatch"):
                    self.export(fixture)

    def test_ordering_preserves_fixed_anchor_support_refs_and_alternatives(self):
        fixture = ordering_fixture(); before = deepcopy(fixture)
        candidate = self.export(fixture)
        for label, item in fixture[3]["items"].items():
            exported = candidate["item_evidence"][label]
            self.assertEqual(exported["order_anchor"], item["order_anchor"])
            self.assertEqual(exported["support_refs"], item["support_refs"])
            self.assertEqual(exported["alternative_matching_anchors"], [])
            self.assertEqual(candidate["order_anchor_spans"][label], {**item["order_anchor"], "speaker": "Rosa"})
        self.assertEqual(fixture, before)

    def test_ordering_cannot_move_primary_away_from_fixed_anchor(self):
        fixture = ordering_fixture(); fixture[5]["fixed_turns"]["I1"] = "T2"
        with self.assertRaisesRegex(ValueError, "ordering_outside_fixed_anchor"):
            self.export(fixture)

    def test_ordering_alternative_quotes_are_also_checked(self):
        fixture = ordering_fixture()
        fixture[3]["items"]["I1"]["alternative_matching_anchors"] = [{"turn_ref": "T2", "quote": "invented"}]
        with self.assertRaisesRegex(ValueError, "alternative0:citation_quote_not_exact"):
            self.export(fixture)

    def test_root_export_binds_complete_review_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            q, gold, source, review, assets, design = multiselect_fixture()
            def write(name, value):
                data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
                (folder / name).write_bytes(data)
                return hashlib.sha256(data).hexdigest()
            source_sha = write("CUMULATIVE_SOURCE.json", source)
            q.update(source_path=str(folder / "CUMULATIVE_SOURCE.json"), source_sha256=source_sha)
            public_sha = write("PUBLIC_QUESTIONS.json", {"questions": [q]})
            asset_public_sha = write("PUBLIC_ASSETS.json", {"statements": []})
            assets.update(source_sha256=source_sha, public_sha256=asset_public_sha)
            assets_sha = write("ASSET_SOURCE_JUDGMENTS.json", assets)
            design_sha = write("PRIVATE_DESIGN.json", {"source_sha256": source_sha, "public_assets_sha256": asset_public_sha,
                "asset_judgments_sha256": assets_sha, "design": [design]})
            review_doc = {"source_sha256": source_sha, "public_sha256": public_sha,
                "asset_judgments_sha256": assets_sha, "questions": {q["candidate_id"]: review}}
            review_sha = write("PUBLIC_SOURCE_JUDGMENTS.json", review_doc)
            gold_sha = write("PRIVATE_GOLD.json", {"source_sha256": source_sha, "public_sha256": public_sha,
                "source_judgments_sha256": review_sha, "original_design_sha256": design_sha, "questions": [gold]})
            write("FINAL_CROSSCHECK.json", {"private_sha256": gold_sha, "public_sha256": public_sha,
                "source_judgments_sha256": review_sha, "results": [{"candidate_id": q["candidate_id"],
                "status": "CONTENT_PASS", "source_derived_answer": gold["gold"]}]})
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            exported = export_root_questions(folder)
            self.assertEqual(exported[0]["option_evidence"]["A"]["turn_ref"], "R02:B:S01:T007")
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            review_doc["questions"][q["candidate_id"]]["options"]["A"]["reason"] = "Altered after Gold was frozen."
            write("PUBLIC_SOURCE_JUDGMENTS.json", review_doc)
            with self.assertRaisesRegex(ValueError, "private_gold_hash_binding:source_judgments_sha256"):
                export_root_questions(folder)


if __name__ == "__main__":
    unittest.main()
