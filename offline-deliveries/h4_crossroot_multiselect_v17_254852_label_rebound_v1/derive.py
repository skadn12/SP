"""Label-only rebound of a content-audited Qwen v17 question."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "h4_crossroot_multiselect_batch_v17_revision/cc-train-episode-254852"
RESPONSE = PARENT / "RESPONSE.json"
PARSED = PARENT / "PARSED_RESPONSE.json"
SOURCE = PARENT / "source.json"
PROTOCOL = HERE.parent / "h4_crossroot_multiselect_solver_selection_v2/protocol.py"
EXPECTED_RESPONSE_SHA = "45b785382bfaa2cb0b3f681dd46e9b24c66787d0b61bb364b6f97f5354eb78cc"
OLD_ID = "254852-qwen36-multi-11"
NEW_ID = "254852-qwen36-multi-11-label-rebound-v1"
OLD_REASON = ("Options A, B, C, D, and E are directly supported by the cited turns from Robin and Jamie. "
              "Option F is contradicted by Jamie's statement that the walk is 'another way to connect', "
              "implying it is not the only way.")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def derive() -> Path:
    if sha(RESPONSE) != EXPECTED_RESPONSE_SHA:
        raise ValueError("parent_response_changed")
    parsed = json.loads(PARSED.read_text())
    if json.loads(json.loads(RESPONSE.read_text())["raw_text"]) != parsed:
        raise ValueError("parent_parse_changed")
    original = next(q for q in parsed["questions"] if q.get("candidate_id") == OLD_ID)
    if original["gold"] != list("ABCDE") or original["answer_reason"] != OLD_REASON:
        raise ValueError("parent_question_changed")
    derived = copy.deepcopy(original)
    derived["candidate_id"] = NEW_ID
    mapping = {**{label: label for label in "ABCD"}, "E": "F", "F": "E"}
    derived["options"] = {new: original["options"][mapping[new]] for new in "ABCDEF"}
    derived["option_evidence"] = {
        new: copy.deepcopy(original["option_evidence"][mapping[new]]) for new in "ABCDEF"}
    derived["gold"] = list("ABCDF")
    derived["answer_reason"] = OLD_REASON.replace(
        "Options A, B, C, D, and E", "Options A, B, C, D, and F").replace(
        "Option F is contradicted", "Option E is contradicted")
    for new, old in mapping.items():
        if derived["options"][new] != original["options"][old] or \
                derived["option_evidence"][new] != original["option_evidence"][old]:
            raise ValueError("content_changed_during_label_rebound")
        if (derived["option_evidence"][new]["relation"] == "supports") != (new in derived["gold"]):
            raise ValueError("gold_relation_mismatch")
    derived["source_path"] = str(SOURCE.resolve())
    derived["source_sha256"] = sha(SOURCE)
    audit = {
        "content_audit_status": "PASSED_FOR_THIS_DERIVED_QUESTION",
        "auditor": "Codex independent semantic review",
        "scope": "Single-turn direct entailment/contradiction, speaker, modality, and overlap with prior selected questions",
        "findings": [
            "A/B/C/D/E in the parent are directly supported by their own R02:S03 turns and preserve speaker and modality.",
            "Parent F is directly contradicted by Jamie's 'This is another way to connect' in R02:S03:T011.",
            "Six anchors are distinct and none is a prior-selected forbidden anchor.",
            "Prior selected walk question 254852-codex-multi-05 uses different concrete propositions and turns."
        ],
        "only_transformation": "Swap whole E/F option and evidence entries; update Gold and answer_reason label references",
        "new_label_to_parent_label": mapping,
        "parent_candidate_id": OLD_ID,
        "parent_response_path": str(RESPONSE.resolve()),
        "parent_response_sha256": EXPECTED_RESPONSE_SHA,
        "parent_parsed_path": str(PARSED.resolve()),
        "parent_parsed_sha256": sha(PARSED),
        "source_path": str(SOURCE.resolve()),
        "source_sha256": sha(SOURCE),
        "model_questions_modified_by_codex": "LABELS_ONLY",
        "solver_evaluated": False,
        "selected_for_training": False,
        "training_started": False,
    }
    pool = {
        "status": "CONTENT_AUDITED_READY_FOR_SOLVER",
        "candidate_count": 1,
        "questions": [derived],
        "provenance": {
            "author": "MODEL_AUTHORED_CONTENT_LABEL_REBOUND_BY_CODEX",
            "provider": "siliconflow",
            "requested_model": "Qwen/Qwen3.6-35B-A3B",
            "returned_model": "Qwen/Qwen3.6-35B-A3B",
            "parent_candidate_id": OLD_ID,
            "parent_response_sha256": EXPECTED_RESPONSE_SHA,
            "derivation": "WHOLE_OPTION_EVIDENCE_LABEL_SWAP_E_F_ONLY",
            "solver_evaluated": False,
            "selected_for_training": False,
            "training_started": False,
        },
    }
    spec = importlib.util.spec_from_file_location("v17_label_rebound_protocol", PROTOCOL)
    assert spec and spec.loader
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    path = HERE / "READY_FOR_EVALUATION.json"
    if path.exists() or (HERE / "AUDIT.json").exists():
        raise FileExistsError("no_overwrite")
    write_new(path, pool)
    try:
        protocol.validate_pool(path)
    except Exception:
        path.unlink()
        raise
    write_new(HERE / "AUDIT.json", audit)
    return path


if __name__ == "__main__":
    print(derive())
