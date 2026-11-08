"""Direct fixture tests for the offline v1.1 contract preflight."""

from __future__ import annotations

from .contract import build_schema, local_closure, schema_valid


def _base(qtype: str, options: dict[str, str], gold: list[str], *, pass_name: str = "fresh") -> dict:
    row = {
        "outcome": "QA",
        "question_type": qtype,
        "new_question": "Which result is supported by the dialogue?",
        "options_or_items": options,
        "canonical_gold": gold,
        "evidence_refs": ["P01:T001", "R03:S01:T001"],
        "selected_evidence_relations": ["relation_a", "relation_b"] if pass_name == "composition" else ["relation_a"],
        "expected_context_dependency": "The answer uses the supplied context.",
        "r03_semantic_contribution_claimed": True,
        "answerable_without_r03_claim": False,
    }
    if pass_name == "composition":
        row.update({"source_cluster_ids": ["cluster-a", "cluster-b"], "composition_relation": "joint relation"})
    else:
        row.update({"independent_new_qa_reason": "It targets a new relation.", "novelty_basis": "R03 state change."})
    return row


def _schema(pass_name: str = "fresh") -> dict:
    # The source schemas are only needed for their common required/properties contract.
    from pathlib import Path
    import json
    name = "fresh_exploration_output_schema_v1.json" if pass_name == "fresh" else "multi_qa_composition_output_schema_v1.json"
    source = json.loads((Path(__file__).parents[5] / "restart-n5-runs" / "h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2" / name).read_text())
    return build_schema(source, pass_name)


def test_single_four_options_passes():
    row = _base("single_choice", {"A": "one", "B": "two", "C": "three", "D": "four"}, ["A"])
    assert schema_valid(row, _schema())[0]
    assert local_closure(row, "fresh", {"P01:T001", "R03:S01:T001"})["status"] == "PASS"


def test_multi_four_five_six_options_pass():
    for count in (4, 5, 6):
        options = {chr(ord("A") + index): f"option-{index}" for index in range(count)}
        row = _base("multi_select", options, ["A", "B"])
        assert schema_valid(row, _schema())[0]
        assert local_closure(row, "fresh", {"P01:T001", "R03:S01:T001"})["status"] == "PASS"


def test_ordering_four_five_six_options_pass():
    for count in (4, 5, 6):
        options = {f"I{index + 1}": f"item-{index}" for index in range(count)}
        row = _base("ordering", options, list(options))
        assert schema_valid(row, _schema())[0]
        assert local_closure(row, "fresh", {"P01:T001", "R03:S01:T001"})["status"] == "PASS"


def test_invalid_counts_and_gold_fail():
    too_few = _base("multi_select", {"A": "a", "B": "b", "C": "c"}, ["A", "B"])
    assert not schema_valid(too_few, _schema())[0]
    too_many = _base("ordering", {f"I{index + 1}": str(index) for index in range(7)}, [f"I{index + 1}" for index in range(7)])
    assert not schema_valid(too_many, _schema())[0]
    all_gold = _base("multi_select", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A", "B", "C", "D"])
    assert local_closure(all_gold, "fresh", {"P01:T001", "R03:S01:T001"})["status"] == "FAIL"


def test_minimal_evidence_is_accepted_and_r03_is_derived():
    row = _base("single_choice", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A"])
    row["evidence_refs"] = ["R03:S01:T001"]
    checked = local_closure(row, "fresh", {"R03:S01:T001"})
    assert checked["status"] == "PASS"
    assert checked["evidence_completeness"] == "NOT_REQUIRED_FIRST_PASS"
    assert checked["derived_new_r03_evidence_refs"] == ["R03:S01:T001"]


def test_repeated_r03_field_is_rejected_by_additional_properties():
    row = _base("single_choice", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A"])
    row["new_r03_evidence_refs"] = ["R03:S01:T001"]
    assert not schema_valid(row, _schema())[0]


def test_no_safe_has_conditional_reason():
    row = {
        "outcome": "NO_SAFE_EVOLUTION", "question_type": None, "new_question": None,
        "options_or_items": None, "canonical_gold": None, "evidence_refs": [],
        "selected_evidence_relations": [], "expected_context_dependency": None,
        "r03_semantic_contribution_claimed": None, "answerable_without_r03_claim": None,
        "no_safe_evolution_reason": "No unambiguous successor is supported.",
        "independent_new_qa_reason": None, "novelty_basis": None,
    }
    assert schema_valid(row, _schema())[0]
    assert local_closure(row, "fresh", set())["status"] == "NO_SAFE_EVOLUTION"

