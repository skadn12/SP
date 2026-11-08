"""Offline contract tests for feedback compiler V4."""
from pathlib import Path

from compiler_v4 import ROOTS, VERSION, append_feedback_to_request, build_feedback_suffix, load_feedback_context


HERE = Path(__file__).resolve().parent
COMPILED = HERE / "compiled_v4"


def test_delivery_has_eight_fixed_roots_and_no_external_calls():
    manifest = load_json(COMPILED / "COMPILED_FEEDBACK.json")
    assert manifest["version"] == VERSION
    assert manifest["external_calls"] == 0
    assert manifest["root_order"] == list(ROOTS)
    assert manifest["selection_contract"]["ground"] == "removed"
    assert manifest["selection_contract"]["multi_select_gold_cardinality"] == [2, 3, 4, 5, 6]
    assert len(manifest["records"]) == 8


def test_each_candidate_has_history_and_every_model_suffix_is_bound():
    for root in ROOTS:
        history = load_json(COMPILED / root / "candidate_history.json")
        assert history
        for row in history:
            assert row["primary_transition"]
            assert row["training_exposure"]
            assert set(row["models"]) == {"M0", "M_clean_policy", "M_inclusive_policy"}
            for model in row["models"].values():
                assert set(model["full_context"]) == {"correct_count", "total_count", "accuracy"}
                assert set(model["no_context"]) == {"correct_count", "total_count", "accuracy"}
        for role in ("writer", "miner", "planner"):
            for model in ("M0", "M_clean_policy", "M_inclusive_policy"):
                suffix = build_feedback_suffix(COMPILED, root, role, model)
                assert "PREVIOUS_TRAINING_ROUND_FEEDBACK_V4" in suffix
                assert f'"target_solver_model":"{model}"' in suffix
                assert "four-item" not in suffix
                assert "real-world relative time" not in suffix
                assert "fixed quota" not in suffix


def test_append_is_copying_and_receipted():
    payload = {"messages": [{"role": "user", "content": "dialogue"}]}
    updated, receipt = append_feedback_to_request(payload, COMPILED, ROOTS[0], "writer", "M_clean_policy")
    assert updated is not payload
    assert payload["messages"][0]["content"] == "dialogue"
    assert "PREVIOUS_TRAINING_ROUND_FEEDBACK_V4" in updated["messages"][0]["content"]
    assert receipt["feedback_appended"] is True
    assert receipt["target_solver_model"] == "M_clean_policy"


def load_json(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    for name in ("test_delivery_has_eight_fixed_roots_and_no_external_calls",
                 "test_each_candidate_has_history_and_every_model_suffix_is_bound",
                 "test_append_is_copying_and_receipted"):
        globals()[name]()
    print("feedback_v4_tests: PASS")
