"""Versioned, source-bound relabeling of content-reviewed v5 ordering drafts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "h4_crossroot_ordering_batch_v5_fresh_roots"
sys.path.insert(0, str(V5))
from generate import check_candidate, frozen_request, read_json, sha, write_new


LABELS = [f"I{i}" for i in range(1, 7)]
REPAIR_VERSION = HERE.name


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def label_map(candidate_id: str) -> dict[str, str]:
    shuffled = sorted(LABELS, key=lambda label: sha(f"{REPAIR_VERSION}:{candidate_id}:{label}".encode("utf-8")))
    if sum(old != new for old, new in zip(LABELS, shuffled)) < 3:
        shuffled = shuffled[2:] + shuffled[:2]
    return dict(zip(LABELS, shuffled))


def rebind(question: dict, mapping: dict[str, str]) -> dict:
    result = deepcopy(question)
    result["items"] = {mapping[old]: question["items"][old] for old in LABELS}
    result["item_evidence"] = {mapping[old]: deepcopy(question["item_evidence"][old]) for old in LABELS}
    result["gold"] = [mapping[old] for old in question["gold"]]
    return result


def main() -> None:
    audit_path = HERE / "CONTENT_AUDIT.json"
    audit = read_json(audit_path)
    if audit.get("version") != REPAIR_VERSION or audit.get("raw_v5_mechanical_valid") != 0:
        raise ValueError("audit_version_or_raw_count_invalid")
    decisions = {row["candidate_id"]: row for row in audit["candidates"]}
    if len(decisions) != 16 or len(audit["candidates"]) != 16:
        raise ValueError("audit_must_cover_sixteen_unique_candidates")
    batch = read_json(V5 / "BATCH_PREPARED.json")
    ready, mapping_audit, status_changes, seen_ids = [], [], [], set()
    for entry in batch["roots"]:
        root_id = entry["root_id"]
        _, source, frozen, directory = frozen_request(root_id, "initial")
        parsed_path = directory / "PARSED_RESPONSE.json"
        response_path = directory / "RESPONSE.json"
        validation = read_json(directory / "VALIDATION.json")
        parsed = read_json(parsed_path)
        response = read_json(response_path)
        if (validation.get("mechanically_valid_count") != 0 or
                validation.get("candidate_count") != 4 or len(parsed.get("questions", [])) != 4 or
                response.get("returned_model") != "Qwen/Qwen3.6-35B-A3B" or
                response.get("payload_sha256") != frozen["payload_sha256"]):
            raise ValueError(f"raw_v5_binding_or_count_invalid:{root_id}")
        for question in parsed["questions"]:
            candidate_id = question["candidate_id"]
            if candidate_id in seen_ids or candidate_id not in decisions:
                raise ValueError(f"unreviewed_or_duplicate_candidate:{candidate_id}")
            seen_ids.add(candidate_id)
            decision = decisions[candidate_id]
            if not decision["content_accepted"]:
                continue
            raw_errors = check_candidate(question, root_id, source["turns"], set(frozen["candidate_ids"]))
            allowed_raw_errors = {"display_not_sufficiently_shuffled"}
            if root_id.endswith("96544"):
                allowed_raw_errors |= {f"{label}:invalid_status_or_reason" for label in LABELS}
            if not set(raw_errors) <= allowed_raw_errors or "display_not_sufficiently_shuffled" not in raw_errors:
                raise ValueError(f"accepted_candidate_has_unreviewed_mechanical_error:{candidate_id}:{raw_errors}")
            mapping = label_map(candidate_id)
            repaired = rebind(question, mapping)
            per_candidate_changes = []
            for old_label, new_label in mapping.items():
                evidence = repaired["item_evidence"][new_label]
                if evidence["status"] == "proposal":
                    if root_id != "cc-train-episode-96544":
                        raise ValueError(f"unexpected_proposal_status:{candidate_id}")
                    evidence["status"] = "statement"
                    per_candidate_changes.append({
                        "path": f"item_evidence.{new_label}.status", "old_label": old_label,
                        "before": "proposal", "after": "statement",
                        "reason": "The utterance states a proposal; item text and quote retain its prospective modality.",
                    })
            if root_id == "cc-train-episode-96544" and not per_candidate_changes:
                raise ValueError(f"expected_status_normalization_missing:{candidate_id}")
            errors = check_candidate(repaired, root_id, source["turns"], set(frozen["candidate_ids"]))
            if errors:
                raise ValueError(f"repaired_candidate_failed_mechanical_checks:{candidate_id}:{errors}")
            if sorted(repaired["items"].values()) != sorted(question["items"].values()):
                raise ValueError(f"item_text_changed:{candidate_id}")
            mapping_audit.append({
                "candidate_id": candidate_id, "root_id": root_id, "old_to_new_label": mapping,
                "old_gold": question["gold"], "new_gold": repaired["gold"],
                "original_question_sha256": sha(canonical(question)),
                "repaired_question_sha256": sha(canonical(repaired)),
                "status_change_count": len(per_candidate_changes),
            })
            status_changes.extend({"candidate_id": candidate_id, **change} for change in per_candidate_changes)
            ready.append({
                "candidate_id": candidate_id, "root_id": root_id, "question": repaired,
                "provenance": {
                    "author": "REMOTE_LARGE_MODEL_WITH_REVIEWED_LOCAL_REBIND",
                    "provider": response["provider"], "requested_model": response["requested_model"],
                    "returned_model": response["returned_model"], "response_id": response["response_id"],
                    "source_path": str((V5 / root_id / "source.json").resolve()),
                    "source_sha256": entry["source_sha256"],
                    "raw_response_path": str(response_path.resolve()),
                    "raw_response_file_sha256": sha(response_path.read_bytes()),
                    "parsed_response_path": str(parsed_path.resolve()),
                    "parsed_response_file_sha256": sha(parsed_path.read_bytes()),
                    "original_question_sha256": sha(canonical(question)),
                    "repaired_question_sha256": sha(canonical(repaired)),
                    "original_prompt_sha256": frozen["prompt_sha256"],
                    "original_payload_sha256": frozen["payload_sha256"],
                    "content_audit_path": str(audit_path.resolve()),
                    "content_audit_sha256": sha(audit_path.read_bytes()),
                    "solver_evaluated": False, "selected_for_training": False,
                },
            })
    if seen_ids != set(decisions) or len(ready) != 8:
        raise ValueError("candidate_coverage_or_accepted_count_changed")
    write_new(HERE / "REBIND_AUDIT.json", {
        "version": REPAIR_VERSION, "status": "DETERMINISTIC_LABEL_REBIND_COMPLETE",
        "content_audit_sha256": sha(audit_path.read_bytes()), "mappings": mapping_audit,
    })
    write_new(HERE / "STATUS_NORMALIZATION.json", {
        "version": REPAIR_VERSION, "status": "REVIEWED_ENUM_NORMALIZATION_ONLY",
        "content_audit_sha256": sha(audit_path.read_bytes()), "changes": status_changes,
    })
    write_new(HERE / "READY_FOR_EVALUATION.json", {
        "version": REPAIR_VERSION, "status": "READY_FOR_ORDERING_GROUND_FULL_EVALUATION",
        "source_batch_path": str((V5 / "BATCH_PREPARED.json").resolve()),
        "source_batch_sha256": sha((V5 / "BATCH_PREPARED.json").read_bytes()),
        "content_audit_path": str(audit_path.resolve()),
        "content_audit_sha256": sha(audit_path.read_bytes()),
        "rebind_audit_path": str((HERE / "REBIND_AUDIT.json").resolve()),
        "rebind_audit_sha256": sha((HERE / "REBIND_AUDIT.json").read_bytes()),
        "status_normalization_path": str((HERE / "STATUS_NORMALIZATION.json").resolve()),
        "status_normalization_sha256": sha((HERE / "STATUS_NORMALIZATION.json").read_bytes()),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "model": "Qwen/Qwen3.6-35B-A3B", "questions": ready,
        "raw_v5_mechanical_valid_count": 0, "ready_count": len(ready),
        "solver_evaluated": False, "training_started": False,
    })
    print(json.dumps({"status": "READY_FOR_EVALUATION", "count": len(ready),
                      "per_root": {entry["root_id"]: sum(row["root_id"] == entry["root_id"] for row in ready)
                                   for entry in batch["roots"]},
                      "status_normalizations": len(status_changes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
