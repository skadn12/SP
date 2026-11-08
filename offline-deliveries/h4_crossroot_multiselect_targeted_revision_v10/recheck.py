"""Offline reinterpretation of frozen v9 model revisions under the v10 field rule."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V9_DIR = HERE.parent / "h4_crossroot_multiselect_targeted_revision_v9"
V9_SCRIPT = V9_DIR / "revise.py"
LABELS = "ABCDEF"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency():
    batch = read_json(V9_DIR / "BATCH_PREPARED.json")
    if digest(V9_SCRIPT) != batch["script_sha256"]:
        raise ValueError("v9_script_changed")
    spec = importlib.util.spec_from_file_location("h4_multiselect_v9_dependency", V9_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("v9_dependency_unavailable")
    v9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v9)
    return v9, batch


def check_revision(revised: object, original: dict, root_id: str, turns: list[dict],
                   validator, v9) -> list[str]:
    if not isinstance(revised, dict):
        return ["revised_question_not_object"]
    label = v9.TARGETS[root_id]["label"]
    required = {f"options.{label}"}
    allowed = required | {"answer_reason", f"option_evidence.{label}.reason"}
    changed = v9.changed_paths(original, revised)
    errors = []
    if not required <= changed or not changed <= allowed:
        errors.append("changed_fields_outside_v10_rule:" + ",".join(sorted(changed)))
    errors.extend(validator(revised, root_id, turns))
    return errors


def check_position(gold: list[str]) -> dict:
    count = len(gold)
    if not 2 <= count <= 6 or len(set(gold)) != count or not set(gold) <= set(LABELS):
        raise ValueError("gold_cardinality_or_labels_invalid")
    pattern = "".join(label for label in LABELS if label in gold)
    six_correct = count == 6
    return {
        "gold_count": count,
        "gold_pattern": pattern,
        "non_prefix_position": pattern != LABELS[:count],
        "six_correct_position_exception": six_correct,
        "position_rule_pass": six_correct or pattern != LABELS[:count],
    }


def recheck_root(root_id: str, v9, batch: dict) -> tuple[dict, dict | None]:
    runtime, entry, source, original, _, _ = v9.frozen_inputs(root_id)
    directory = V9_DIR / root_id
    response = read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != entry["source_sha256"] or
            response.get("original_question_sha256") != entry["original_question_sha256"] or
            response.get("payload_sha256") != entry["payload_sha256"]):
        raise ValueError("response_binding_changed:" + root_id)
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid:" + root_id)
    raw = metadata.get("raw_response")
    if not isinstance(raw, dict) or len(raw.get("choices", [])) != 1 or \
            raw["choices"][0].get("message", {}).get("content") != response.get("raw_text"):
        raise ValueError("raw_response_content_mismatch:" + root_id)
    parsed = json.loads(response["raw_text"])
    if parsed != read_json(directory / "PARSED_RESPONSE.json"):
        raise ValueError("parsed_response_changed:" + root_id)
    if set(parsed) != {"revised_question", "abstentions"} or parsed["abstentions"] != []:
        raise ValueError("response_schema_or_abstention_invalid:" + root_id)
    if (metadata.get("returned_model") != batch["requested_model"] or
            metadata.get("finish_reason") != "stop"):
        raise ValueError("response_model_or_finish_invalid:" + root_id)
    prior = read_json(directory / "VALIDATION.json")
    expected_prior_error = ("changed_fields_not_exactly_allowed:answer_reason,options." +
                            entry["target_label"])
    if (prior.get("status") != "MECHANICAL_REJECTED" or
            prior.get("errors") != [expected_prior_error]):
        raise ValueError("unexpected_v9_decision:" + root_id)
    revised = parsed["revised_question"]
    errors = check_revision(revised, original, root_id, source["turns"],
                            runtime.check_candidate, v9)
    position = check_position(revised["gold"]) if isinstance(revised, dict) else None
    if position and not position["position_rule_pass"]:
        errors.append("gold_position_prefix")
    status = ("MECHANICALLY_VALID_CONTENT_AUDIT_PENDING" if not errors else
              "MECHANICAL_REJECTED")
    report = {
        "root_id": root_id, "status": status, "errors": errors,
        "target_label": entry["target_label"], "position": position,
        "changed_fields": sorted(v9.changed_paths(original, revised)) if isinstance(revised, dict) else [],
        "source_sha256": entry["source_sha256"],
        "original_question_sha256": entry["original_question_sha256"],
        "v9_response_sha256": digest(directory / "RESPONSE.json"),
        "v9_parsed_response_sha256": digest(directory / "PARSED_RESPONSE.json"),
        "v9_validation_sha256": digest(directory / "VALIDATION.json"),
        "returned_model": metadata["returned_model"],
        "response_id": raw.get("id"),
        "content_audited": False, "ready_for_evaluation": False,
    }
    candidate = None if errors else revised
    return report, candidate


def build() -> tuple[dict, dict]:
    v9, batch = dependency()
    reports, candidates = [], []
    for root_id in v9.TARGETS:
        report, candidate = recheck_root(root_id, v9, batch)
        reports.append(report)
        if candidate is not None:
            candidates.append(candidate)
    patterns = [row["position"]["gold_pattern"] for row in reports if row["position"]]
    distinct = len(patterns) == len(set(patterns))
    if not distinct:
        for row in reports:
            row["errors"].append("gold_pattern_not_distinct_across_roots")
            row["status"] = "MECHANICAL_REJECTED"
        candidates = []
    summary = {
        "version": HERE.name, "source_version": V9_DIR.name,
        "source_batch_prepared_sha256": digest(V9_DIR / "BATCH_PREPARED.json"),
        "requested_model": batch["requested_model"], "roots": reports,
        "root_count": len(reports), "mechanically_valid_count": len(candidates),
        "gold_patterns_distinct_across_roots": distinct,
        "policy": "gold_count_2_to_6; non_prefix_for_2_to_5; six_correct_exception",
        "model_calls": 0, "solver_calls": 0, "training_started": False,
        "content_audited": False, "ready_for_evaluation": False,
    }
    pool = {
        "version": HERE.name, "source_author": batch["requested_model"],
        "revision_author": batch["requested_model"], "codex_role": "OFFLINE_MECHANICAL_RECHECK",
        "content_audited": False, "ready_for_evaluation": False,
        "questions": candidates,
    }
    return summary, pool


if __name__ == "__main__":
    summary, pool = build()
    for name, value in (("BATCH_RECHECK.json", summary), ("CANDIDATES_PENDING_AUDIT.json", pool)):
        path = HERE / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OFFLINE_RECHECK_COMPLETE",
                      "mechanically_valid_count": summary["mechanically_valid_count"],
                      "gold_patterns": [row["position"]["gold_pattern"] for row in summary["roots"]]}))
