from __future__ import annotations

from .validator import attest_original_preservation, materialize_view, validate_generation


def test_materialization_preserves_original_evidence() -> None:
    turns = [{"turn_ref": "T1", "speaker": "Alice", "text": "She chose the car.",
              "origin": "source_grounded", "scene_ref": None}]
    qa = {"gold_text": "car"}
    generation = {"perturbation_type": "competing_state", "added_information": [{
        "type": "alternative_state", "content": "Earlier, Alice briefly compared a bicycle.",
        "relation_to_gold": "historical_non_final_alternative", "affects_answer": False,
        "speaker": "Alice", "insert_after_turn_ref": "T1", "temporal_relation": "before_final_state",
        "state_status": "historically_true_not_final"}], "preservation_claim": {
        "answer_unchanged": True, "world_state_unchanged": True, "timeline_unchanged": True,
        "entity_bindings_unchanged": True, "original_evidence_sufficient": True}}
    assert validate_generation(generation, perturbation_type="competing_state", qa=qa,
                               valid_turn_refs=["T1"], valid_speakers=["Alice"])["valid"]
    view, _ = materialize_view(turns, qa_id="Q1", perturbation_type="competing_state",
                               additions=generation["added_information"])
    assert attest_original_preservation(turns, view, ["T1"])["valid"]


def test_conflicting_relation_is_rejected() -> None:
    qa = {"gold_text": "car"}
    generation = {"perturbation_type": "competing_state", "added_information": [{
        "type": "alternative_state", "content": "Earlier, Alice briefly compared a bicycle.",
        "relation_to_gold": "contradicts", "affects_answer": False,
        "speaker": "Alice", "insert_after_turn_ref": "T1", "temporal_relation": "before_final_state",
        "state_status": "historically_true_not_final"}], "preservation_claim": {
        "answer_unchanged": True, "world_state_unchanged": True, "timeline_unchanged": True,
        "entity_bindings_unchanged": True, "original_evidence_sufficient": True}}
    audit = validate_generation(generation, perturbation_type="competing_state", qa=qa,
                                valid_turn_refs=["T1"], valid_speakers=["Alice"])
    assert not audit["valid"]
    assert "addition_1_relation_to_gold" in audit["errors"]


def test_new_interference_entity_can_be_kept_separate() -> None:
    qa = {"gold_text": "car"}
    generation = {"perturbation_type": "entity_event_interference", "added_information": [{
        "type": "entity_interference", "content": "Carla separately compared a bicycle.",
        "relation_to_gold": "other_entity_only", "affects_answer": False,
        "speaker": "Carla", "insert_after_turn_ref": "T1", "temporal_relation": "concurrent_background",
        "state_status": "other_entity_only"}], "preservation_claim": {
        "answer_unchanged": True, "world_state_unchanged": True, "timeline_unchanged": True,
        "entity_bindings_unchanged": True, "original_evidence_sufficient": True}}
    assert validate_generation(generation, perturbation_type="entity_event_interference", qa=qa,
                               valid_turn_refs=["T1"], valid_speakers=["Alice"])["valid"]
