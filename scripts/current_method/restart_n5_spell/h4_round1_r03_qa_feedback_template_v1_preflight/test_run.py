"""Regression tests for the frozen offline QA feedback template."""

from functools import lru_cache

from .preflight import build_all, load_inputs


@lru_cache(maxsize=1)
def _built():
    return build_all(load_inputs())


def test_01_pass_and_next_stage():
    report = _built()["artifacts"]["feedback_template_preflight_report.json"]
    assert report["status"] == "R03_QA_FEEDBACK_TEMPLATE_V1_PREFLIGHT_PASS"
    assert report["next_status"] == "READY_FOR_FEEDBACK_GUIDED_TARGETED_EVOLUTION_PILOT"


def test_02_eligibility_counts():
    counts = _built()["artifacts"]["feedback_template_preflight_report.json"]["eligibility_counts"]
    assert counts == {"ELIGIBLE": 6, "QUARANTINE_EXCLUDED": 1, "NO_SAFE_QA_FEEDBACK_SEED": 1}


def test_03_common_data_contract_filled():
    rows = _built()["artifacts"]["common_feedback_data_manifest.jsonl"]
    assert len(rows) == 6
    assert all(set(row) == {"source_identity", "source_qa", "evidence", "solver_observations", "transition_behavior", "r03_context", "diagnostic_boundary"} for row in rows)
    assert all(row["r03_context"]["evidence_refs"] for row in rows)


def test_04_no_forbidden_leakage():
    report = _built()["artifacts"]["feedback_template_preflight_report.json"]
    assert report["dev_test_leakage_count"] == 0
    assert report["quarantine_leakage_count"] == 0
    assert report["unsafe_96462_targeted_seed_leakage_count"] == 0
    assert report["root_specific_recommended_action_count"] == 0
    assert report["automatic_strategy_assignment_count"] == 0


def test_05_strategy_space_and_no_safe_contract():
    artifacts = _built()["artifacts"]
    assert len(artifacts["qa_evolution_strategy_schema_v1.json"]["enum"]) == 11
    schema = artifacts["qa_feedback_output_schema_v1.json"]
    assert "NO_SAFE_EVOLUTION" in schema["allOf"][0]["if"]["properties"]["selected_strategy"]["const"]


def test_06_template_sha_singleton():
    artifacts = _built()["artifacts"]
    rows = artifacts["feedback_seed_eligibility_manifest.jsonl"]
    assert len({row["template_sha256"] for row in rows}) == 1


def test_07_zero_calls():
    audit = _built()["artifacts"]["call_closure_zero_call_audit.json"]
    assert audit["status"] == "PASS"
    assert all(value == 0 for value in audit["calls"].values())


def test_08_unknown_and_semantic_boundary():
    rows = _built()["artifacts"]["common_feedback_data_manifest.jsonl"]
    assert all(row["evidence"]["evidence_summary"] is None for row in rows)
    assert all(row["diagnostic_boundary"]["unknown_policy"] for row in rows)
    assert _built()["artifacts"]["feedback_template_preflight_report.json"]["semantic_family_claimed"] is False


def test_09_json_schema_validation():
    from jsonschema import validate
    artifacts = _built()["artifacts"]
    common_schema = artifacts["common_feedback_schema_v1.json"]
    for row in artifacts["common_feedback_data_manifest.jsonl"]:
        validate(row, common_schema)
    no_safe = {
        "source_question_id": "example",
        "source_cluster_id": "example-cluster",
        "root_id": "example-root",
        "selected_strategy": "NO_SAFE_EVOLUTION",
        "strategy_rationale": "No uniquely answerable successor is supported.",
        "feedback_observations_used": [],
        "tentative_hypothesis": "hypothesis: insufficient support",
        "new_question": None,
        "options_or_items": None,
        "canonical_gold": None,
        "evidence_refs": [],
        "new_r03_evidence_refs": [],
        "source_question_relation": "",
        "expected_context_dependency": "",
        "no_safe_evolution_reason": "No safe evolution is supported by the available evidence.",
    }
    validate(no_safe, artifacts["qa_feedback_output_schema_v1.json"])
