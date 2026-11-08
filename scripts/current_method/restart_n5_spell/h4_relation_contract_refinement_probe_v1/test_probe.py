from __future__ import annotations

from .audit import audit_relation
from .contracts import build_relation_contract
from .prompt import build_relation_scene_prompt


def _plan():
    return {
        "scenes": [
            {"before_state": "recipe: base honey mustard", "after_state": "recipe: honey mustard base"},
            {"trigger": {"type": "constraint", "content": "the glaze is too thick to coat the ham"},
             "after_state": "recipe: add water to thin the glaze"},
            {"after_state": "recipe: final spreadable glaze with water"},
        ]
    }


def _child():
    return {"scenes": [
        {"dialogue_turns": [{"turn_ref": "S1:T1", "text": "They agree on the honey mustard base recipe.", "speaker": "Classmates A"}]},
        {"dialogue_turns": [{"turn_ref": "S2:T1", "text": "Because the glaze is too thick to coat the ham, they add water to thin it.", "speaker": "Classmates B"}]},
        {"dialogue_turns": [{"turn_ref": "S3:T1", "text": "The final spreadable glaze with water is confirmed ready.", "speaker": "Classmates A"}]},
    ]}


def test_contract_has_typed_edges_and_dependency_test():
    contract = build_relation_contract(_plan())
    assert contract["relation_type"] == "constraint_replan"
    assert len(contract["typed_edges"]) == 3
    assert contract["dependency_test"]["full_context_requires"] == ["fact_A", "trigger", "fact_B"]


def test_audit_requires_trigger_and_causal_link():
    result = audit_relation(_child(), build_relation_contract(_plan()))
    assert result["relation_present"] is True
    assert result["relation_audit_pass"] is True
    assert result["trigger_supported"] is True
    assert result["state_chain_valid"] is True


def test_audit_rejects_early_relation_leakage():
    child = _child()
    child["scenes"][0]["dialogue_turns"].append({
        "turn_ref": "S1:T2", "speaker": "Classmates B",
        "text": "Because it is too thick, we might add water later.",
    })
    result = audit_relation(child, build_relation_contract(_plan()))
    assert result["relation_present"] is True
    assert result["early_relation_leakage"] is True
    assert result["state_chain_valid"] is False


def test_relation_prompt_releases_future_contract_by_scene():
    contract = build_relation_contract(_plan())
    scene_one = build_relation_scene_prompt(
        parent={"dialogue": {"turns": []}}, prior=[], current=_plan()["scenes"][0],
        index=1, contract=contract)["prompt"]
    scene_two = build_relation_scene_prompt(
        parent={"dialogue": {"turns": []}}, prior=[], current=_plan()["scenes"][1],
        index=2, contract=contract)["prompt"]
    assert "add water to thin the glaze" not in scene_one
    assert "add water to thin the glaze" in scene_two
