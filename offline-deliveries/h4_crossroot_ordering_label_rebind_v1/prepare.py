"""Create an immutable, label-only derivative of two audited model questions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
ORIGINAL = DELIVERIES / "h4_crossroot_ordering_batch_v2"
LABELS = tuple(f"I{index}" for index in range(1, 7))
SPECS = (
    {
        "root_id": "cc-train-episode-136202",
        "candidate_id": "136202-qwen36-order-02",
        "source_sha256": "61a0127ac31e3837d3475e566c062a1b95e221be66f113b37b978a33ce4b7627",
        "display_original_ids": ("I4", "I1", "I6", "I2", "I5", "I3"),
        "expected_r02_anchors": 1,
    },
    {
        "root_id": "cc-train-episode-254852",
        "candidate_id": "254852-qwen36-order-02",
        "source_sha256": "8385b541a6d9325ece6aded7cb90ca7ddc2389ebaa86e0d7c24824c47a95f2fd",
        "display_original_ids": ("I3", "I6", "I2", "I5", "I1", "I4"),
        "expected_r02_anchors": 0,
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_one(spec):
    root = ORIGINAL / spec["root_id"]
    files = {name: root / name for name in (
        "PARSED_RESPONSE.json", "VALIDATED_CANDIDATES.json", "RESPONSE.json",
        "source.json", "prompt.txt", "request.json", "VALIDATION.json",
    )}
    source = read_json(files["source.json"])
    validated = read_json(files["VALIDATED_CANDIDATES.json"])
    parsed = read_json(files["PARSED_RESPONSE.json"])
    response = read_json(files["RESPONSE.json"])
    validation = read_json(files["VALIDATION.json"])
    provenance = validated["provenance"]
    original_id = spec["candidate_id"]
    if sha256(files["source.json"]) != spec["source_sha256"]:
        raise ValueError(f"source hash changed: {original_id}")
    if source["root_id"] != spec["root_id"] or provenance["root_id"] != spec["root_id"]:
        raise ValueError(f"root mismatch: {original_id}")
    if provenance["source_sha256"] != spec["source_sha256"]:
        raise ValueError(f"provenance source mismatch: {original_id}")
    if provenance["prompt_sha256"] != sha256(files["prompt.txt"]):
        raise ValueError(f"prompt hash mismatch: {original_id}")
    if provenance["requested_model"] != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError(f"requested model changed: {original_id}")
    if provenance["returned_model"] != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError(f"returned model changed: {original_id}")
    if (json.loads(response["raw_text"]) != parsed
            or response["metadata"]["returned_model"] != provenance["returned_model"]
            or response["source_sha256"] != spec["source_sha256"]
            or response["payload_sha256"] != provenance["payload_sha256"]):
        raise ValueError(f"HTTP response provenance mismatch: {original_id}")
    if validation["global_errors"]:
        raise ValueError(f"source validation failed: {original_id}")
    if not any(row["candidate_id"] == original_id and not row["errors"]
               for row in validation["checks"]):
        raise ValueError(f"candidate validation failed: {original_id}")
    originals = [q for q in validated["questions"] if q["candidate_id"] == original_id]
    parsed_originals = [q for q in parsed["questions"] if q["candidate_id"] == original_id]
    if len(originals) != 1 or parsed_originals != originals:
        raise ValueError(f"validated candidate differs from model parse: {original_id}")
    original = originals[0]
    if (original["question_type"] != "ordering" or original["root_id"] != spec["root_id"]
            or tuple(original["items"]) != LABELS or set(original["item_evidence"]) != set(LABELS)
            or original["gold"] != list(LABELS)):
        raise ValueError(f"unexpected original question contract: {original_id}")

    turns = source["turns"]
    index_by_ref = {turn["turn_ref"]: index for index, turn in enumerate(turns)}
    if len(index_by_ref) != len(turns):
        raise ValueError(f"duplicate source turn references: {original_id}")
    anchor_indexes = []
    anchor_checks = []
    for old_id in original["gold"]:
        evidence = original["item_evidence"][old_id]
        ref, quote = evidence["turn_ref"], evidence["quote"]
        if ref not in index_by_ref or not quote or quote not in turns[index_by_ref[ref]]["text"]:
            raise ValueError(f"quote is not exact at source turn: {original_id}:{old_id}:{ref}")
        anchor_indexes.append(index_by_ref[ref])
        anchor_checks.append({"original_id": old_id, "turn_ref": ref,
                              "source_turn_index_1_based": index_by_ref[ref] + 1,
                              "quote_exact_substring": True})
    if sorted(anchor_indexes) != anchor_indexes or len(set(anchor_indexes)) != 6:
        raise ValueError(f"Gold anchor order invalid: {original_id}")
    r02_count = sum(row["turn_ref"].startswith("R02:") for row in anchor_checks)
    if r02_count != spec["expected_r02_anchors"]:
        raise ValueError(f"R02 anchor count changed: {original_id}")

    display = spec["display_original_ids"]
    if set(display) != set(LABELS) or tuple(display) == LABELS:
        raise ValueError(f"display mapping invalid: {original_id}")
    new_to_original = dict(zip(LABELS, display))
    original_to_new = {old: new for new, old in new_to_original.items()}
    derived_gold = [original_to_new[old] for old in original["gold"]]
    if derived_gold == list(LABELS):
        raise ValueError(f"Gold still follows displayed labels: {original_id}")
    derived_id = f"{original_id}-labels-v1"
    derived = {key: original[key] for key in (
        "root_id", "question_type", "topic", "chronology_kind", "question"
    )}
    derived["candidate_id"] = derived_id
    derived["items"] = {new: original["items"][old] for new, old in new_to_original.items()}
    derived["gold"] = derived_gold
    derived["item_evidence"] = {
        new: original["item_evidence"][old] for new, old in new_to_original.items()
    }

    for new, old in new_to_original.items():
        if derived["items"][new] != original["items"][old]:
            raise AssertionError("item text changed")
        if derived["item_evidence"][new] != original["item_evidence"][old]:
            raise AssertionError("evidence text changed")
    if derived["question"] != original["question"]:
        raise AssertionError("question text changed")

    audit = {
        "original_candidate_id": original_id,
        "derived_candidate_id": derived_id,
        "root_id": spec["root_id"],
        "source_sha256": spec["source_sha256"],
        "source_turn_count": len(turns),
        "original_gold": original["gold"],
        "derived_gold": derived_gold,
        "new_label_to_original_label": new_to_original,
        "original_label_to_new_label": original_to_new,
        "anchor_checks": anchor_checks,
        "r02_anchor_count": r02_count,
        "r02_new_content_yield": r02_count > 0,
        "original_order_reason_retained_only_in_original": True,
        "original_order_reason": original["order_reason"],
        "order_reason_excluded_from_derivative": "It refers to original I-labels; no model text is rewritten.",
        "original_files_sha256": {str(path): sha256(path) for path in files.values()},
        "original_model_provenance": provenance,
        "codex_operation": "DETERMINISTIC_LABEL_REBIND_ONLY",
        "question_text_modified": False,
        "item_text_modified": False,
        "evidence_text_modified": False,
        "solver_evaluated": False,
        "selected_for_training": False,
    }
    return derived, audit


def main():
    outputs = [prepare_one(spec) for spec in SPECS]
    if (HERE / "READY_FOR_EVALUATION.json").exists() or (HERE / "LABEL_MAPPING_AUDIT.json").exists():
        raise FileExistsError("derived pool already exists")
    questions, audits = zip(*outputs)
    provenance = {
        "original_question_author": "REMOTE_LARGE_MODEL",
        "provider": "siliconflow",
        "requested_model": "Qwen/Qwen3.6-35B-A3B",
        "returned_model": "Qwen/Qwen3.6-35B-A3B",
        "codex_role": "DETERMINISTIC_LABEL_REBIND_ONLY",
        "original_batch": str(ORIGINAL),
        "original_content_review": str(ORIGINAL / "CONTENT_REVIEW.md"),
        "solver_evaluated": False,
        "selected_for_training": False,
        "training_started": False,
        "new_generation_calls": 0,
    }
    pool = {
        "status": "CONTENT_AUDITED_LABEL_REBOUND_NOT_SOLVER_SCREENED",
        "version": "h4_crossroot_ordering_label_rebind_v1",
        "provenance": provenance,
        "candidate_count": len(questions),
        "questions": questions,
    }
    audit = {"version": pool["version"], "candidate_count": len(audits), "candidates": audits}
    write_once(HERE / "READY_FOR_EVALUATION.json", pool)
    write_once(HERE / "LABEL_MAPPING_AUDIT.json", audit)
    print(json.dumps({"candidate_count": len(questions),
                      "pool_sha256": sha256(HERE / "READY_FOR_EVALUATION.json"),
                      "audit_sha256": sha256(HERE / "LABEL_MAPPING_AUDIT.json"),
                      "candidate_ids": [q["candidate_id"] for q in questions]}, indent=2))


if __name__ == "__main__":
    main()
