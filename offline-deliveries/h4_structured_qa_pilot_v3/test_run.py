"""Offline contract tests. No test calls the provider."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run


@contextmanager
def raises(error_type, match: str):
    try:
        yield
    except error_type as error:
        assert match in str(error), str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {match}")


def fact(fact_id: str, ref: str, *, speaker: str = "Emily", quote: str = "sent the outline") -> dict:
    return {
        "fact_id": fact_id, "actor": speaker, "action": "send", "object_id": f"outline-{fact_id}",
        "version": "v1", "checkpoint": ref, "status": "performed_now",
        "claim": f"{speaker} sent outline {fact_id} during this exchange.",
        "opposite_claim": f"{speaker} did not send outline {fact_id} during this exchange.",
        "evidence": {"ref": ref, "speaker": speaker, "quote": quote},
    }


def view() -> dict:
    return {
        "root_id": "root", "history_sha256": "old", "g02_sha256": "new",
        "turns": [
            {"turn_ref": "R03:S03:T009", "speaker": "Emily", "text": "I sent the outline."},
            {"turn_ref": "G02:S01:T001", "speaker": "Emily", "text": "I sent the outline."},
        ],
    }


def test_surface_is_fixed_projection_without_graph_keys() -> None:
    visible = run.surface()
    assert len(visible["turns"]) == len(run.HISTORY_REFS) + 10
    assert [turn["turn_ref"] for turn in visible["turns"][:len(run.HISTORY_REFS)]] == list(run.HISTORY_REFS)
    assert all(set(turn) == {"turn_ref", "speaker", "text"} for turn in visible["turns"])
    assert "event_key" not in run.extraction_prompt(visible)


def test_duplicate_json_keys_rejected_even_when_nested() -> None:
    with raises(ValueError, "duplicate_json_key:E"):
        run.strict_json('{"option_evidence":{"E":1,"E":2}}')


def test_extraction_checks_speaker_quote_and_checkpoint() -> None:
    row = fact("F001", "G02:S01:T001", quote="sent the outline")
    raw = json.dumps({"facts": [row], "abstain_reason": ""})
    result = run.validate_extraction(raw, view())
    assert result["status"] == "PENDING_HUMAN_SEMANTIC_REVIEW"
    assert result["training_promotion"] is False

    row["evidence"]["speaker"] = "David"
    with raises(ValueError, "evidence_speaker_mismatch"):
        run.validate_extraction(json.dumps({"facts": [row], "abstain_reason": ""}), view())
    row["evidence"]["speaker"] = "Emily"
    row["evidence"]["quote"] = "Emily: I sent the outline."
    with raises(ValueError, "evidence_quote_not_verbatim"):
        run.validate_extraction(json.dumps({"facts": [row], "abstain_reason": ""}), view())
    row["evidence"]["quote"] = "sent the outline"
    row["checkpoint"] = "G02:S03:T001"
    with raises(ValueError, "evidence_ref_or_checkpoint_invalid"):
        run.validate_extraction(json.dumps({"facts": [row], "abstain_reason": ""}), view())


def test_extraction_preserves_valid_abstention() -> None:
    result = run.validate_extraction('{"facts":[],"abstain_reason":"No direct evidence."}', view())
    assert result["facts"] == []
    assert result["abstain_reason"] == "No direct evidence."


def test_review_manifest_only_accepts_ids_from_frozen_pending_ledger() -> None:
    with TemporaryDirectory() as directory, patch.object(run, "HERE", Path(directory)):
        tmp_path = Path(directory)
        (tmp_path / "extraction").mkdir()
        pending = run.validate_extraction(json.dumps({"facts": [fact("F001", "G02:S01:T001")], "abstain_reason": ""}), view())
        run.write_new(tmp_path / "extraction" / "pending_ledger.json", pending)
        digest = run.sha((tmp_path / "extraction" / "pending_ledger.json").read_bytes())
        review = {"status": "SOURCE_REVIEWED", "attestation": "SOURCE_AND_OPPOSITE_CLAIM_REVIEWED",
                  "reviewer": "human", "pending_ledger_sha256": digest, "accepted_fact_ids": ["F001"]}
        review_path = tmp_path / "reviewed_ids.json"
        run.write_new(review_path, review)
        accepted, review_sha = run.reviewed_facts(view())
        assert [row["fact_id"] for row in accepted] == ["F001"]
        assert review_sha == run.sha(review_path.read_bytes())
        review["attestation"] = "SOURCE_ONLY"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        with raises(ValueError, "review_not_attested"):
            run.reviewed_facts(view())
        review["attestation"] = "SOURCE_AND_OPPOSITE_CLAIM_REVIEWED"
        review["accepted_fact_ids"] = ["F999"]
        review_path.write_text(json.dumps(review), encoding="utf-8")
        with raises(ValueError, "accepted_fact_id_unknown"):
            run.reviewed_facts(view())


def composition_facts() -> list[dict]:
    refs = ["R03:S03:T009", "G02:S01:T001", "G02:S01:T003", "G02:S02:T001",
            "G02:S02:T002", "G02:S03:T001", "G02:S02:T003"]
    return [fact(f"F{i}", ref) for i, ref in enumerate(refs, 1)]


def composition_response() -> dict:
    def clause(fact_id, orientation="as_stated"):
        return {"fact_id": fact_id, "orientation": orientation}
    return {"selected_options": [
        {"clauses": [clause("F1"), clause("F2")]},
        {"clauses": [clause("F3"), clause("F4")]},
        {"clauses": [clause("F5"), clause("F6")]},
        {"clauses": [clause("F2", "opposite"), clause("F4")]},
        {"clauses": [clause("F3"), clause("F7", "opposite")]},
    ], "abstain_reason": ""}


def test_composition_derives_gold_and_evidence_from_reviewed_ids() -> None:
    result = run.validate_composition(json.dumps(composition_response()), composition_facts(), view())
    assert result["gold"] == ["A", "B", "C"]
    assert result["option_evidence"]["D"]["clauses"][0]["verdict"] == "FALSE"
    assert result["option_evidence"]["D"]["clauses"][0]["checkpoint"] == "G02:S01:T001"
    assert result["training_promotion"] is False
    assert result["status"] == "UNREVIEWED_CANDIDATE_NOT_GOLD"
    assert result["candidate_id"].startswith("v3-ms-")


def test_composition_rejects_unreviewed_id_and_same_turn_pair() -> None:
    response = composition_response()
    response["selected_options"][0]["clauses"][1]["fact_id"] = "F999"
    with raises(ValueError, "unreviewed_fact_or_orientation"):
        run.validate_composition(json.dumps(response), composition_facts(), view())
    response = composition_response()
    response["selected_options"][0]["clauses"][1]["fact_id"] = "F1"
    with raises(ValueError, "option_not_two_independent_facts"):
        run.validate_composition(json.dumps(response), composition_facts(), view())


def test_composition_abstain_is_retained() -> None:
    result = run.validate_composition('{"selected_options":[],"abstain_reason":"No audited pair."}', [], view())
    assert result == {"status": "ABSTAIN", "reason": "No audited pair.", "questions": [], "training_promotion": False}


def test_raw_refs_in_question_wording_are_rejected() -> None:
    row = fact("F001", "G02:S01:T001")
    row["claim"] = "Emily sent the outline at G02:S01:T001."
    with raises(ValueError, "turn_ref_leaked_into_claim"):
        run.validate_extraction(json.dumps({"facts": [row], "abstain_reason": ""}), view())


def test_modified_frozen_prompt_rejected_before_api() -> None:
    with TemporaryDirectory() as directory, patch.object(run, "HERE", Path(directory)):
        stage = Path(directory) / "extraction"
        stage.mkdir()
        run.write_new(stage / "prompt.txt", "modified", raw=True)
        run.write_new(stage / "manifest.json", {
            "stage": "extraction", "model": run.MODEL, "prompt_sha256": run.sha(b"original"),
            "history_sha256": run.R03_SHA256, "g02_sha256": run.G02_SHA256,
        })
        with patch.object(run, "credential_client", side_effect=AssertionError("API must not be reached")):
            with raises(ValueError, "frozen_prompt_mismatch"):
                run.run_stage("extraction")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"{len(tests)} offline tests passed")
