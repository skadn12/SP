"""Offline schema/facade integration; synthetic material receipts are not grants."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

try:
    from . import base_qa_pipeline_v2_2_1 as qa
    from . import base_qa_option_freeze_v2_2_1 as freeze
    from . import test_base_qa_pipeline_v2_2_1 as fixtures
except ImportError:
    import base_qa_pipeline_v2_2_1 as qa
    import base_qa_option_freeze_v2_2_1 as freeze
    import test_base_qa_pipeline_v2_2_1 as fixtures


class SchemaAndFacadeTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads(qa.SCHEMA_PATH.read_text())
        self.validator = Draft202012Validator(self.schema)

    def validate_def(self, name, value):
        self.validator.evolve(schema={"$ref": "#/$defs/" + name}).validate(value)

    def test_schema_self_valid_and_complete_fixture_contracts_match(self):
        Draft202012Validator.check_schema(self.schema)
        for single in (False, True):
            source, discovery, target, candidate, scope = fixtures.fixture(complete_sequence_single=single)
            self.validator.validate(discovery)
            self.validate_def("target", target)
            self.validate_def("candidate", candidate)
        source, discovery, target, candidate, scope = fixtures.plan_fixture()
        self.validator.validate(discovery)
        self.validate_def("target", target)
        self.validate_def("candidate", candidate)

    def test_schema_does_not_conflate_typed_numeric_set_values(self):
        value = qa.answers.semantic_answer("set", [True, 1, 1.0])
        self.validate_def("semantic_answer", value)
        self.assertEqual(len(qa.answers.validate_semantic(value)["value"]), 3)
        with self.assertRaises(ValueError):
            qa.answers.semantic_answer("set", [1, 1])

    def test_no_legacy_answer_or_extra_task_answer_field(self):
        source, discovery, target, candidate, scope = fixtures.fixture()
        target["expected_answer"] = ["untrusted replacement"]
        self.assertFalse(self.validator.evolve(schema={"$ref": "#/$defs/target"}).is_valid(target))
        candidate["gold_answer"]["unexpected"] = "A"
        self.assertFalse(self.validator.evolve(schema={"$ref": "#/$defs/candidate"}).is_valid(candidate))

    def test_explicit_fixture_upgrade_is_not_source_authorization(self):
        old = fixtures.previous.fixture()
        source, discovery, target, candidate, scope = qa.adapt_v22_fixture(*old)
        self.assertTrue(qa.validate_target(target, discovery, source, scope)["mechanical_pass"])
        self.assertTrue(qa.validate_candidate(candidate, target, discovery, source, scope)["mechanical_pass"])
        self.assertFalse(qa.validate_candidate(candidate, target, discovery, source, scope)["formal_qa_allowed"])

    def test_formal_facade_refuses_unapproved_fixture_before_bundle_creation(self):
        source, discovery, target, candidate, scope = fixtures.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            target_path = Path(tmp, "TARGET.json")
            qa.freeze_target(target_path, target, discovery, source, scope)
            run = Path(tmp, "public_run")
            with self.assertRaisesRegex(qa.ValidationError, "formal generation blocked"):
                qa.prepare_frozen_review(run, candidate, target_path, discovery, source, scope,
                    reviewer_id="controlled-reviewer", context_id="controlled-context",
                    issuer_id="controlled-orchestrator", receipt_id="controlled-receipt")
            self.assertFalse(run.exists())

    def test_controlled_formal_facade_binds_target_and_uses_same_frozen_map(self):
        # Temporary fake receipt tests file mechanics only, not material approval.
        source, discovery, target, candidate, _ = fixtures.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            scope = fixtures.previous.fixture_scope(tmp, source, candidate["author_id"])
            fixtures.previous.decorate(discovery, source, scope)
            discovery["schema_version"] = qa.VERSION + ":discovery"
            target["discovery_sha256"] = qa.fp(discovery)
            target["selection_sha256"] = qa.fp(qa.rank_relations(discovery, source, scope))
            fixtures.refresh_candidate(candidate, target, source)
            target_path = Path(tmp, "TARGET.json")
            qa.freeze_target(target_path, target, discovery, source, scope)
            run = Path(tmp, "controlled_freeze")
            qa.prepare_frozen_review(run, candidate, target_path, discovery, source, scope,
                reviewer_id="controlled-reviewer", context_id="controlled-context",
                issuer_id="controlled-orchestrator", receipt_id="controlled-receipt")
            bundle = freeze.load_bundle(run, source, scope)
            paths = {row["path"] for row in json.loads((run / freeze.MANIFEST).read_text())["files"]}
            self.assertIn(str(target_path.resolve()), paths)
            reverse = bundle["render_contract"]["canonical_to_display"]
            display = qa.answers.map_answer(candidate["gold_answer"], candidate["question_type"], reverse)
            scored = freeze.score_frozen_answer(run, display)
            self.assertTrue(scored["correct"])
            self.assertEqual(scored["mapping_sha256"], bundle["render_contract"]["mapping_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
