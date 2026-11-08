"""Direct tests for question_type discrimination, Gold shape, and evidence policy."""

from __future__ import annotations

from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.contract import local_closure, schema_valid
from .fixtures import KNOWN_REFS, multipass, targeted
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.solver_compat import (
    audit_gold_rebinding, position_variants, requests_per_model,
)


def run_case(row, schema, mode, expected: str) -> None:
    valid, _ = schema_valid(row, schema)
    closure = local_closure(row, mode=mode, known_refs=KNOWN_REFS) if valid else {"status": "SCHEMA_FAIL"}
    actual = "PASS" if valid and closure["status"] in {"PASS", "NO_SAFE_EVOLUTION"} else "FAIL"
    assert actual == expected, (expected, valid, closure)


def test_question_type_discriminator(schema_targeted, schema_fresh):
    run_case(targeted("single_choice", 4, ["A"]), schema_targeted, "targeted", "PASS")
    run_case(targeted(None, no_safe=True), schema_targeted, "targeted", "PASS")
    bad = targeted(None, 4, ["A"]); run_case(bad, schema_targeted, "targeted", "FAIL")
    bad = targeted("single_choice", 4, ["A"]); bad["selected_strategy"] = "NO_SAFE_EVOLUTION"; bad["no_safe_evolution_reason"] = "reason"; run_case(bad, schema_targeted, "targeted", "FAIL")
    bad = multipass("ordering", 4, ["I1", "I2", "I3", "I4"]); bad["question_type"] = "multi_select"; run_case(bad, schema_fresh, "fresh", "FAIL")


def test_single_choice_fixtures(schema_fresh):
    run_case(multipass("single_choice", 4, ["A"]), schema_fresh, "fresh", "PASS")
    for row in [
        multipass("single_choice", 4, ["E"]),
        multipass("single_choice", 3, ["A"]),
        multipass("single_choice", 5, ["A"]),
        multipass("single_choice", 4, ["A", "B"]),
    ]:
        run_case(row, schema_fresh, "fresh", "FAIL")


def test_multi_select_four_to_six_and_full_select(schema_fresh):
    for count, gold in ((4, ["A", "B"]), (4, ["A", "B", "C", "D"]), (5, ["A", "B"]), (5, list("ABCDE")), (6, ["A", "B"]), (6, list("ABCDEF"))):
        run_case(multipass("multi_select", count, gold), schema_fresh, "fresh", "PASS")
    for row in [
        multipass("multi_select", 3, ["A", "B"]), multipass("multi_select", 7, ["A", "B"]),
        multipass("multi_select", 4, ["A"]), multipass("multi_select", 4, ["A", "A"]),
        multipass("multi_select", 4, ["A", "E"]),
    ]:
        run_case(row, schema_fresh, "fresh", "FAIL")
    noncontiguous = multipass("multi_select", 4, ["A", "B"]); noncontiguous["options_or_items"] = {"A": "a", "B": "b", "D": "d", "E": "e"}; run_case(noncontiguous, schema_fresh, "fresh", "FAIL")


def test_ordering_four_to_six_complete_permutations(schema_fresh):
    for count in (4, 5, 6):
        keys = [f"I{index + 1}" for index in range(count)]
        run_case(multipass("ordering", count, list(reversed(keys))), schema_fresh, "fresh", "PASS")
    for row in [
        multipass("ordering", 3, ["I1", "I2", "I3"]),
        multipass("ordering", 7, [f"I{index + 1}" for index in range(7)]),
        multipass("ordering", 4, ["I1", "I2", "I3"]),
        multipass("ordering", 4, ["I1", "I2", "I2", "I4"]),
        multipass("ordering", 4, ["I1", "I2", "I3", "I5"]),
    ]:
        run_case(row, schema_fresh, "fresh", "FAIL")


def test_no_safe_and_minimal_evidence(schema_fresh):
    run_case(multipass("single_choice", 4, ["A"], no_safe=True), schema_fresh, "fresh", "PASS")
    row = multipass("single_choice", 4, ["A"]); row["evidence_refs"] = ["R03:S01:T001"]; run_case(row, schema_fresh, "fresh", "PASS")
    row = multipass("single_choice", 4, ["A"]); row["evidence_refs"] = ["R03:S99:T999"]; run_case(row, schema_fresh, "fresh", "FAIL")


def test_dynamic_downstream_rebinding():
    for count in (4, 5, 6):
        multi = {"question_type": "multi_select", "options": {chr(65 + i): f"o{i}" for i in range(count)}, "gold": ["A", "B"]}
        variants = position_variants(multi); assert len(variants) == count; assert audit_gold_rebinding(multi, variants)["pass"]; assert requests_per_model(multi) == count * 6
        keys = [f"I{i + 1}" for i in range(count)]
        ordering = {"question_type": "ordering", "items": {key: f"v{key}" for key in keys}, "gold": list(reversed(keys))}
        variants = position_variants(ordering); assert len(variants) == count; assert audit_gold_rebinding(ordering, variants)["pass"]; assert requests_per_model(ordering) == count * 6
