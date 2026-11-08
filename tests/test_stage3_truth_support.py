from __future__ import annotations

import ast
import copy
import inspect
import re
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import stage3_claim_ir as claim_ir  # noqa: E402
from scripts.current_method import stage3_truth_constructors as truth  # noqa: E402
from scripts.current_method import stage3_truth_support as support  # noqa: E402
from scripts.current_method import stage3_truth_support_validator as independent_validator  # noqa: E402
from tests.test_stage3_truth_constructors import immutable_target  # noqa: E402


EXPECTED_RECORD_FIELDS = {
    "schema_version",
    "claim_id",
    "claim_type",
    "operation",
    "edge_role",
    "qa_eligible",
    "event_binding",
    "quote",
    "semantic_path_manifest",
    "support_sha256",
}

# This is the real lexical gate used by Stage64 for an active temporal gold
# evidence turn.  Keep it here verbatim so a renderer change cannot quietly
# reintroduce a direct answer-status hint.
STAGE64_ACTIVE_CUE_RE = re.compile(
    r"\b(?:current(?:ly)?|now|confirm(?:ed|s|ation)?|active)\b",
    re.IGNORECASE,
)


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise AssertionError(f"mutation source is absent: {old!r} in {text!r}")
    return text.replace(old, new, 1)


def _swap_once(text: str, left: str, right: str) -> str:
    if left not in text or right not in text:
        raise AssertionError(f"swap source is absent: {left!r}, {right!r} in {text!r}")
    marker = "TRUTH SUPPORT SWAP MARKER"
    if marker in text:
        raise AssertionError("unexpected swap marker collision")
    return text.replace(left, marker, 1).replace(right, left, 1).replace(marker, right, 1)


def _rehash(record: dict[str, Any]) -> None:
    unhashed = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "support_sha256"
    }
    record["support_sha256"] = support.canonical_sha256(unhashed)


def _delete_path(root: dict[str, Any], path: tuple[str | int, ...]) -> None:
    cursor: Any = root
    for component in path[:-1]:
        cursor = cursor[component]
    del cursor[path[-1]]


class TypedFrozenSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.claims_by_operation: dict[str, list[dict[str, Any]]] = {}
        for operation in truth.OPERATIONS:
            skeleton = truth.construct_truth_skeleton(
                immutable_target(operation),
                graph_id=f"truth-support-{operation}",
                parent_graph_id="truth-support-parent",
                selected_parent_event_ids=["P17", "P29"],
                parent_max_global_time_index=100,
            )
            materialized = truth.materialize_truth_skeleton(
                skeleton,
                truth.example_content_bindings(skeleton),
            )
            compiled = claim_ir.compile_claim_ir(materialized["graph"])
            cls.claims_by_operation[operation] = compiled["claims"]

    @classmethod
    def claim(
        cls, operation: str, claim_type: str, index: int = 0
    ) -> dict[str, Any]:
        matches = [
            claim
            for claim in cls.claims_by_operation[operation]
            if claim["claim_type"] == claim_type
        ]
        return matches[index]

    def assert_quote_rejected(
        self,
        claim: dict[str, Any],
        mutate_quote,
        *,
        expected_codes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Assert semantic rejection after giving the mutation a valid hash.

        Rehashing is important: otherwise every semantic mutation can appear to
        be protected while only the envelope hash check is doing any work.
        """

        record = support.compile_typed_frozen_support(claim)
        tampered = copy.deepcopy(record)
        tampered["quote"] = mutate_quote(record["quote"])
        self.assertNotEqual(tampered["quote"], record["quote"])
        _rehash(tampered)
        validation = support.validate_typed_frozen_support(claim, tampered)
        self.assertFalse(validation["pass"], validation)
        self.assertNotIn("support_hash_mismatch", validation["issue_counts_by_code"])
        for code in expected_codes:
            self.assertIn(code, validation["issue_counts_by_code"], validation)
        return validation

    def assert_deleted_typed_field_rejected(
        self,
        claim: dict[str, Any],
        path: tuple[str | int, ...],
    ) -> None:
        original_record = support.compile_typed_frozen_support(claim)
        malformed = copy.deepcopy(claim)
        _delete_path(malformed["canonical_support_claim"], path)

        with self.assertRaises(support.TruthSupportError):
            support.compile_typed_frozen_support(malformed)

        validation = support.validate_typed_frozen_support(
            malformed, original_record
        )
        self.assertFalse(validation["pass"], validation)
        self.assertIn("typed_field_missing", validation["issue_counts_by_code"])

    def test_all_seven_operations_and_all_84_claims_compile_losslessly(self) -> None:
        seen_claim_types: set[str] = set()
        total = 0
        self.assertEqual(set(self.claims_by_operation), set(truth.OPERATIONS))
        self.assertEqual(len(self.claims_by_operation), 7)
        for operation, claims in self.claims_by_operation.items():
            self.assertGreater(len(claims), 0)
            for claim in claims:
                with self.subTest(operation=operation, claim_id=claim["claim_id"]):
                    before = copy.deepcopy(claim)
                    record = support.compile_typed_frozen_support(claim)
                    validation = independent_validator.validate_typed_frozen_support(
                        claim, record
                    )
                    self.assertTrue(validation["pass"], validation["issues"])
                    self.assertEqual(
                        validation["required_semantic_paths"],
                        validation["covered_semantic_paths"],
                    )
                    self.assertEqual(set(record), EXPECTED_RECORD_FIELDS)
                    self.assertEqual(record["operation"], operation)
                    self.assertEqual(record["claim_id"], claim["claim_id"])
                    self.assertEqual(
                        record["event_binding"],
                        support.claim_event_binding(claim),
                    )
                    self.assertEqual(claim, before)
                    seen_claim_types.add(record["claim_type"])
                    total += 1
        self.assertEqual(total, 84)
        self.assertEqual(
            seen_claim_types,
            {
                "node",
                "dependency",
                "status_transition",
                "comparison",
                "visibility",
                "membership",
                "cause",
            },
        )

    def test_validator_is_a_separate_module_with_no_renderer_or_legacy_import(self) -> None:
        self.assertNotEqual(Path(inspect.getfile(support)), Path(inspect.getfile(independent_validator)))
        source = inspect.getsource(independent_validator)
        tree = ast.parse(source)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")

        self.assertFalse(
            any(name.endswith("stage3_truth_support") for name in imported_modules),
            imported_modules,
        )
        self.assertFalse(
            any(name.endswith("stage3_claim_validator") for name in imported_modules),
            imported_modules,
        )
        self.assertNotIn("_validate_claim_quote_semantics", source)
        self.assertNotIn("compile_typed_frozen_support", source)
        self.assertFalse(any(name.startswith("_render_") for name in vars(independent_validator)))

    def test_bare_quote_is_never_admitted_as_a_support_proof(self) -> None:
        claim = self.claim("bridge", "node")
        record = support.compile_typed_frozen_support(claim)
        validation = support.validate_typed_frozen_support(claim, record["quote"])
        self.assertFalse(validation["pass"], validation)
        self.assertIn("support_record_required", validation["issue_counts_by_code"])

    def test_record_schema_is_exact_and_each_missing_field_is_rejected(self) -> None:
        claim = self.claim("relation_lookup", "dependency")
        record = support.compile_typed_frozen_support(claim)
        self.assertEqual(set(record), EXPECTED_RECORD_FIELDS)

        for field in sorted(EXPECTED_RECORD_FIELDS):
            with self.subTest(missing_field=field):
                tampered = copy.deepcopy(record)
                del tampered[field]
                if field != "support_sha256":
                    _rehash(tampered)
                validation = support.validate_typed_frozen_support(claim, tampered)
                self.assertFalse(validation["pass"], validation)
                self.assertIn("support_field_missing", validation["issue_counts_by_code"])

        unexpected = copy.deepcopy(record)
        unexpected["unchecked_note"] = "do not admit me"
        _rehash(unexpected)
        validation = support.validate_typed_frozen_support(claim, unexpected)
        self.assertFalse(validation["pass"], validation)
        self.assertIn("support_field_unexpected", validation["issue_counts_by_code"])

    def test_each_metadata_field_tamper_is_rejected_even_after_rehash(self) -> None:
        claim = self.claim("relation_lookup", "dependency")
        record = support.compile_typed_frozen_support(claim)
        mutations = {
            "schema_version": "stage3_typed_frozen_support_v999",
            "claim_id": "clm_wrong",
            "claim_type": "node",
            "operation": "bridge",
            "edge_role": "control",
            "qa_eligible": not record["qa_eligible"],
        }
        expected_codes = {
            "schema_version": "support_schema_mismatch",
            "claim_id": "claim_id_mismatch",
            "claim_type": "claim_type_mismatch",
            "operation": "operation_mismatch",
            "edge_role": "edge_role_mismatch",
            "qa_eligible": "qa_eligible_mismatch",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                tampered = copy.deepcopy(record)
                tampered[field] = value
                _rehash(tampered)
                validation = support.validate_typed_frozen_support(claim, tampered)
                self.assertFalse(validation["pass"], validation)
                self.assertIn(expected_codes[field], validation["issue_counts_by_code"])
                self.assertNotIn("support_hash_mismatch", validation["issue_counts_by_code"])

    def test_binding_manifest_and_hash_tampering_are_independently_rejected(self) -> None:
        claim = self.claim("relation_lookup", "dependency")
        record = support.compile_typed_frozen_support(claim)

        wrong_binding = copy.deepcopy(record)
        wrong_binding["event_binding"]["subject_event_id"] = "E999"
        _rehash(wrong_binding)
        validation = support.validate_typed_frozen_support(claim, wrong_binding)
        self.assertFalse(validation["pass"], validation)
        self.assertIn("event_binding_mismatch", validation["issue_counts_by_code"])
        self.assertNotIn("support_hash_mismatch", validation["issue_counts_by_code"])

        wrong_manifest = copy.deepcopy(record)
        wrong_manifest["semantic_path_manifest"] = wrong_manifest[
            "semantic_path_manifest"
        ][:-1]
        _rehash(wrong_manifest)
        validation = support.validate_typed_frozen_support(claim, wrong_manifest)
        self.assertFalse(validation["pass"], validation)
        self.assertIn(
            "semantic_path_manifest_mismatch", validation["issue_counts_by_code"]
        )
        self.assertNotIn("support_hash_mismatch", validation["issue_counts_by_code"])

        stale_hash = copy.deepcopy(record)
        stale_hash["quote"] += " A harmless-looking suffix."
        validation = support.validate_typed_frozen_support(claim, stale_hash)
        self.assertFalse(validation["pass"], validation)
        self.assertIn("support_hash_mismatch", validation["issue_counts_by_code"])

        invalid_hash = copy.deepcopy(record)
        invalid_hash["support_sha256"] = "not-a-sha256"
        validation = support.validate_typed_frozen_support(claim, invalid_hash)
        self.assertFalse(validation["pass"], validation)
        self.assertIn("support_hash_invalid", validation["issue_counts_by_code"])

    def test_typed_semantic_manifests_cover_each_operation_specific_role(self) -> None:
        expected_paths = {
            "status_transition": {
                "$.canonical_support_claim.slot",
                "$.canonical_support_claim.from_status",
                "$.canonical_support_claim.to_status",
                "$.canonical_support_claim.objects[0]",
                "$.canonical_support_claim.objects[1]",
                "$.canonical_support_claim.time.from_time_index",
                "$.canonical_support_claim.time.time_index",
                "$.canonical_support_claim.time.global_time_index",
            },
            "visibility": {
                "$.canonical_support_claim.subject",
                "$.canonical_support_claim.predicate",
                "$.canonical_support_claim.objects[0]",
                "$.canonical_support_claim.owner",
                "$.canonical_support_claim.access_class",
                "$.canonical_support_claim.condition",
                "$.canonical_support_claim.scope.observer",
                "$.canonical_support_claim.scope.permission",
                "$.canonical_support_claim.scope.condition",
            },
            "comparison": {
                "$.canonical_support_claim.slot",
                "$.canonical_support_claim.left_value",
                "$.canonical_support_claim.right_value",
                "$.canonical_support_claim.absolute_difference",
            },
            "membership": {
                "$.canonical_support_claim.included",
                "$.canonical_support_claim.entity",
                "$.canonical_support_claim.slot",
                "$.canonical_support_claim.polarity",
            },
            "cause": {
                "$.canonical_support_claim.cause_event.entity",
                "$.canonical_support_claim.cause_event.slot",
                "$.canonical_support_claim.cause_event.value",
                "$.canonical_support_claim.cause_event.status",
                "$.canonical_support_claim.cause_event.time_index",
                "$.canonical_support_claim.cause_event.global_time_index",
                "$.canonical_support_claim.effect_event.entity",
                "$.canonical_support_claim.effect_event.slot",
                "$.canonical_support_claim.effect_event.value",
                "$.canonical_support_claim.effect_event.status",
                "$.canonical_support_claim.effect_event.time_index",
                "$.canonical_support_claim.effect_event.global_time_index",
                "$.canonical_support_claim.time.cause_time_index",
            },
            "dependency": {
                "$.canonical_support_claim.subject_event.entity",
                "$.canonical_support_claim.subject_event.slot",
                "$.canonical_support_claim.subject_event.value",
                "$.canonical_support_claim.subject_event.status",
                "$.canonical_support_claim.object_events[0].entity",
                "$.canonical_support_claim.object_events[0].slot",
                "$.canonical_support_claim.object_events[0].value",
                "$.canonical_support_claim.object_events[0].status",
            },
        }
        representatives = {
            "status_transition": self.claim("temporal_state_update", "status_transition"),
            "visibility": self.claim("visibility", "visibility"),
            "comparison": self.claim("comparison", "comparison"),
            "membership": self.claim("count", "membership"),
            "cause": self.claim("cause_consequence", "cause"),
            "dependency": self.claim("relation_lookup", "dependency"),
        }
        for claim_type, claim in representatives.items():
            with self.subTest(claim_type=claim_type):
                paths = {
                    row["path"] for row in support.required_semantic_paths(claim)
                }
                self.assertTrue(expected_paths[claim_type].issubset(paths), paths)
                self.assertFalse(any(path.endswith(".event_id") for path in paths))

    def test_visibility_membership_dependency_and_transition_fields_fail_closed(self) -> None:
        cases = {
            ("visibility", "visibility"): (
                ("owner",),
                ("access_class",),
                ("condition",),
                ("scope",),
                ("scope", "observer"),
                ("scope", "permission"),
                ("scope", "condition"),
            ),
            ("count", "membership"): (
                ("included",),
                ("entity",),
                ("slot",),
            ),
            ("relation_lookup", "dependency"): (
                ("subject_event",),
                ("subject_event", "event_id"),
                ("subject_event", "entity"),
                ("subject_event", "slot"),
                ("subject_event", "value"),
                ("subject_event", "status"),
                ("subject_event", "time_index"),
                ("subject_event", "global_time_index"),
                ("object_events",),
                ("object_events", 0, "event_id"),
                ("object_events", 0, "entity"),
                ("object_events", 0, "slot"),
                ("object_events", 0, "value"),
                ("object_events", 0, "status"),
                ("object_events", 0, "time_index"),
                ("object_events", 0, "global_time_index"),
            ),
            ("temporal_state_update", "status_transition"): (
                ("slot",),
                ("objects",),
                ("from_status",),
                ("to_status",),
                ("status",),
                ("time",),
                ("time", "from_time_index"),
                ("time", "time_index"),
                ("time", "global_time_index"),
            ),
        }
        for (operation, claim_type), paths in cases.items():
            claim = self.claim(operation, claim_type)
            for path in paths:
                with self.subTest(operation=operation, claim_type=claim_type, path=path):
                    self.assert_deleted_typed_field_rejected(claim, path)

    def test_active_uncertainty_and_trailing_contradiction_are_rejected(self) -> None:
        claim = self.claim("bridge", "node")
        for prefix in ("Reportedly, ", "It might be that ", "Unconfirmed, "):
            with self.subTest(uncertainty=prefix):
                self.assert_quote_rejected(
                    claim,
                    lambda quote, prefix=prefix: prefix + quote,
                    expected_codes=("active_uncertainty_contradiction",),
                )

        for suffix in (
            " The preceding statement is false.",
            " Disregard the previous claim.",
            " The opposite is true.",
        ):
            with self.subTest(contradiction=suffix):
                self.assert_quote_rejected(
                    claim,
                    lambda quote, suffix=suffix: quote + suffix,
                    expected_codes=("trailing_contradiction",),
                )

    def test_stage64_real_active_cue_regex_never_appears_in_active_support(self) -> None:
        checked = 0
        for operation, claims in self.claims_by_operation.items():
            for claim in claims:
                canonical = claim["canonical_support_claim"]
                if canonical.get("status") != "active":
                    continue
                record = support.compile_typed_frozen_support(claim)
                with self.subTest(operation=operation, claim_id=claim["claim_id"]):
                    self.assertIsNone(
                        STAGE64_ACTIVE_CUE_RE.search(record["quote"]), record["quote"]
                    )
                checked += 1
        self.assertGreater(checked, 0)

    def test_natural_support_has_no_internal_labels_underscores_or_ids(self) -> None:
        machine_label_re = re.compile(
            r"\b(?:timing|claim status|scope (?:details|observer|permission|condition)|"
            r"membership (?:entity|slot|included)|resource owner|access class|"
            r"comparison slot|left value|right value|cause endpoint|effect endpoint|"
            r"dependency source|dependency target|local step|global step)\s*:",
            re.IGNORECASE,
        )
        for operation, claims in self.claims_by_operation.items():
            for claim in claims:
                record = support.compile_typed_frozen_support(claim)
                quote = record["quote"]
                with self.subTest(operation=operation, claim_id=claim["claim_id"]):
                    self.assertNotIn("_", quote)
                    self.assertIsNone(machine_label_re.search(quote), quote)
                    self.assertNotIn(claim["claim_id"], quote)
                    for event_id in record["event_binding"]["endpoint_event_ids"]:
                        self.assertIsNone(
                            re.search(rf"(?<!\w){re.escape(event_id)}(?!\w)", quote), quote
                        )

        claim = self.claim("relation_lookup", "dependency")
        record = support.compile_typed_frozen_support(claim)
        mutations = (
            (" canonical_support_claim.", "internal_ir_surface"),
            (f" Reference {claim['claim_id']}.", "internal_id_leak"),
            (
                f" Reference {record['event_binding']['subject_event_id']}.",
                "internal_id_leak",
            ),
        )
        for suffix, code in mutations:
            with self.subTest(leak=suffix):
                self.assert_quote_rejected(
                    claim,
                    lambda quote, suffix=suffix: quote + suffix,
                    expected_codes=(code,),
                )

    def test_seven_operation_role_direction_value_and_polarity_mutations(self) -> None:
        cases = (
            (
                "temporal_state_update",
                "status_transition",
                "direction",
                lambda quote: _swap_once(quote, "Cedar Hall", "Riverside Studio"),
            ),
            (
                "bridge",
                "node",
                "value",
                lambda quote: _replace_once(quote, "Maya", "Zoe"),
            ),
            (
                "comparison",
                "comparison",
                "derived_value",
                lambda quote: _replace_once(quote, "difference is 5", "difference is 6"),
            ),
            (
                "visibility",
                "visibility",
                "semantic_roles",
                lambda _quote: (
                    "For red notebook to view Noah, the requirement is after final approval; "
                    "Maya keeps the resource under Private access."
                ),
            ),
            (
                "count",
                "membership",
                "polarity",
                lambda quote: _replace_once(
                    quote, "is a member of", "is not a member of"
                ),
            ),
            (
                "cause_consequence",
                "cause",
                "causal_direction",
                lambda _quote: (
                    "The fact that Workshop Roster's replacement host is Maya resulted in "
                    "Spring Workshop's trigger reason becoming severe weather."
                ),
            ),
            (
                "relation_lookup",
                "dependency",
                "polarity",
                lambda quote: _replace_once(
                    quote, "depends on", "does not depend on"
                ),
            ),
        )
        self.assertEqual({row[0] for row in cases}, set(truth.OPERATIONS))
        for operation, claim_type, mutation_kind, mutation in cases:
            with self.subTest(operation=operation, mutation=mutation_kind):
                self.assert_quote_rejected(self.claim(operation, claim_type), mutation)


if __name__ == "__main__":
    unittest.main()
