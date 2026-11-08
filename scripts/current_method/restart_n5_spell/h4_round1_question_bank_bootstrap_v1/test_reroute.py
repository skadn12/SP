"""Pure-function regression tests for offline rerouting signatures."""

from .reroute import canonical_gold, evidence_signature, gate_fingerprint, ref_suffix, surface_signature


def _multi(gold: list[str]) -> dict:
    return {
        "question_type": "multi_select",
        "question": "Which facts apply?",
        "options": {"A": "First", "B": "Second", "C": "Third"},
        "items": {},
        "gold": gold,
        "evidence_refs": ["R02:S01:T001"],
        "evidence_groups": {"after": ["R02:S01:T001"]},
    }


def test_surface_canonicalizes_multiselect_gold_order() -> None:
    assert canonical_gold(_multi(["A", "C"])) == canonical_gold(_multi(["C", "A"]))
    assert surface_signature(_multi(["A", "C"])) == surface_signature(_multi(["C", "A"]))


def test_evidence_groups_are_part_of_signature() -> None:
    left = _multi(["A"])
    right = _multi(["A"])
    right["evidence_groups"] = {"before": ["R02:S01:T001"]}
    assert evidence_signature(left) != evidence_signature(right)


def test_prompt_fingerprint_uses_prompt_and_contract_not_pool_path() -> None:
    gate = {
        "question_type": "single_choice", "position_count": 1,
        "max_model_len": 100, "output_reserve": 10,
        "prompt_audit": [{"context_mode": "full_context", "variant_id": "shift_0", "prompt_sha256": "abc", "prompt_token_count": 5}],
    }
    drifted = {**gate, "candidate_pool_sha256": "unrelated", "dialogue_sha256": "unrelated"}
    assert gate_fingerprint(gate, "contract") == gate_fingerprint(drifted, "contract")
    assert gate_fingerprint(gate, "contract") != gate_fingerprint(gate, "other-contract")


def test_ref_suffix_accepts_root_prefixed_refs() -> None:
    assert ref_suffix("cc-train-episode-1:R02:S03:T007") == "R02:S03:T007"
    assert ref_suffix("R02:S03:T007") == "R02:S03:T007"
