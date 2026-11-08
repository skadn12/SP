from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "postweek_source_citation_control_v1"
RUN_DIR = HERE / "outputs" / "B1"
SCORE_DIR = HERE / "outputs" / "scoring"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify_binding(row: dict) -> None:
    p = Path(row["path"])
    if not p.is_file() or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
        raise RuntimeError("BOUND_FILE_CHANGED:" + str(p))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    if SCORE_DIR.exists():
        raise RuntimeError("SCORING_DIRECTORY_EXISTS_NO_OVERWRITE")
    raw_freeze_path = RUN_DIR / "RAW_RESPONSE_FREEZE.json"
    if not raw_freeze_path.is_file():
        raise RuntimeError("RAW_RESPONSE_FREEZE_REQUIRED_BEFORE_GOLD_ACCESS")
    raw_freeze = read(raw_freeze_path)
    if raw_freeze.get("status") != "COMMITTED" or raw_freeze.get("planned_requests") != 24:
        raise RuntimeError("RAW_RESPONSE_FREEZE_NOT_COMPLETE")
    if len(raw_freeze["files"]) != 24 or len(raw_freeze["request_ids_in_frozen_order"]) != 24:
        raise RuntimeError("RAW_RESPONSE_DENOMINATOR_INVALID")
    for key in ("execution_manifest", "parent_freeze"):
        verify_binding(raw_freeze[key])
    execution = read(Path(raw_freeze["execution_manifest"]["path"]))
    for row in execution["bindings"]:
        verify_binding(row)
    parent_freeze = read(PARENT / "FREEZE.json")
    verify_binding({**parent_freeze["manifest_binding"], "path": str(PARENT / parent_freeze["manifest_binding"]["path"])})
    parent_manifest = read(PARENT / "FINAL_MANIFEST.json")
    for row in parent_manifest["local_artifacts"] + parent_manifest["external_dependencies"]:
        if not Path(row["path"]).is_absolute():
            row = {**row, "path": str(PARENT / row["path"])}
        verify_binding(row)
    response_by_id = {}
    for row in raw_freeze["files"]:
        if not Path(row["path"]).is_file() or Path(row["path"]).stat().st_size != row["bytes"] or sha(Path(row["path"])) != row["sha256"]:
            raise RuntimeError("RAW_RESPONSE_HASH_MISMATCH")
        response = read(Path(row["path"]))
        response_by_id[response["request_id"]] = response
    requests = read(PARENT / "REQUESTS.json")["requests"]
    if len(requests) != 24 or set(response_by_id) != {r["request_id"] for r in requests}:
        raise RuntimeError("RAW_RESPONSE_REQUEST_SET_MISMATCH")
    frozen_order = read(PARENT / "RUN_ORDER.json")["run_order_request_ids"]
    if raw_freeze["request_ids_in_frozen_order"] != frozen_order:
        raise RuntimeError("RAW_RESPONSE_EXECUTION_ORDER_DRIFT")
    request_by_id = {r["request_id"]: r for r in requests}
    for position, request_id in enumerate(frozen_order):
        raw = response_by_id[request_id]
        request = request_by_id[request_id]
        if raw.get("position") != position:
            raise RuntimeError("RAW_RESPONSE_POSITION_MISMATCH")
        if raw.get("status") == "RETURNED":
            if raw.get("prompt_sha256") != request["prompt_sha256"] or raw.get("schema_sha256") != request["schema_sha256"] or raw.get("sampling_seed") != request["sampling_seed"]:
                raise RuntimeError("RAW_RESPONSE_REQUEST_BINDING_MISMATCH")

    # This is the first point at which the scorer opens the private answer key.
    private_key_path = PARENT / "private" / "PRIVATE_SCORE_KEY.json"
    private_key = read(private_key_path)
    import jsonschema

    salt = secrets.token_hex(32)
    score_rows = []
    private_map = {}
    for request in requests:
        request_id = request["request_id"]
        response = response_by_id[request_id]
        anon = hashlib.sha256((salt + request_id).encode()).hexdigest()[:24]
        gold_row = private_key["roots"][request["root_id"]]["variants"][request["display_variant"]]
        schema = read(Path(request["schema_path"]))
        row = {
            "blind_id": anon,
            "status": response.get("status"),
            "json_parse_valid": False,
            "json_schema_valid": False,
            "answer_shape_valid": False,
            "exact_order_correct": False,
            "pairwise_correct": 0,
            "pairwise_denominator": 15,
            "pairwise_accuracy": 0.0,
            "all_judgments_supported": False,
            "all_source_refs_valid": False,
            "all_first_sufficient_anchors_cited": False,
            "supported_judgment_count": 0,
            "first_sufficient_anchor_hits": 0,
            "item_denominator": 6,
            "citation_anchor_hit_rate": 0.0,
            "errors": [],
        }
        output_text = response.get("raw_output")
        obj = None
        if response.get("status") == "RETURNED" and isinstance(output_text, str):
            try:
                obj = json.loads(output_text)
                row["json_parse_valid"] = True
            except Exception as exc:
                row["errors"].append("JSON_PARSE:" + type(exc).__name__)
        if obj is not None:
            try:
                jsonschema.validate(obj, schema)
                row["json_schema_valid"] = True
            except Exception as exc:
                row["errors"].append("JSON_SCHEMA:" + type(exc).__name__)
            answer = obj.get("final_answer") if isinstance(obj, dict) else None
            labels = request["labels"]
            row["answer_shape_valid"] = isinstance(answer, list) and all(isinstance(label, str) for label in answer) and len(answer) == len(labels) and len(set(answer)) == len(labels) and set(answer) == set(labels)
            substantive_valid = row["json_schema_valid"] and row["answer_shape_valid"]
            row["exact_order_correct"] = substantive_valid and answer == gold_row["gold_display_order"]
            if substantive_valid:
                predicted_rank = {label: i for i, label in enumerate(answer)}
                gold_rank = {label: i for i, label in enumerate(gold_row["gold_display_order"])}
                row["pairwise_correct"] = sum(
                    (predicted_rank[left] < predicted_rank[right]) == (gold_rank[left] < gold_rank[right])
                    for i, left in enumerate(labels) for right in labels[i + 1:]
                )
                row["pairwise_accuracy"] = row["pairwise_correct"] / row["pairwise_denominator"]
            evidence = obj.get("item_evidence") if isinstance(obj, dict) else None
            if row["json_schema_valid"] and isinstance(evidence, dict) and set(evidence) == set(labels):
                refs_valid = True
                anchors_hit = 0
                supported = 0
                for label in labels:
                    ev = evidence[label]
                    if not isinstance(ev, dict):
                        refs_valid = False
                        continue
                    if ev.get("judgment") == "supported":
                        supported += 1
                    refs = ev.get("turn_refs", [])
                    allowed_refs = schema["properties"]["item_evidence"]["properties"][label]["properties"]["turn_refs"]["items"]["enum"]
                    if not isinstance(refs, list) or any(ref not in allowed_refs for ref in refs):
                        refs_valid = False
                        refs = []
                    if gold_row["first_sufficient_turn_ref_by_display_label"][label] in refs:
                        anchors_hit += 1
                row["all_judgments_supported"] = supported == len(labels)
                row["all_source_refs_valid"] = refs_valid
                row["all_first_sufficient_anchors_cited"] = anchors_hit == len(labels)
                row["supported_judgment_count"] = supported
                row["first_sufficient_anchor_hits"] = anchors_hit
                row["citation_anchor_hit_rate"] = anchors_hit / len(labels)
            else:
                row["errors"].append("ITEM_EVIDENCE_LABEL_COVERAGE")
        score_rows.append(row)
        private_map[anon] = {"request_id": request_id, "pair_id": request["pair_id"],
            "root_id": request["root_id"], "condition": request["condition"],
            "display_variant": request["display_variant"], "replicate": request["replicate"]}

    secrets.SystemRandom().shuffle(score_rows)
    SCORE_DIR.mkdir(parents=True, exist_ok=False)
    public_path = SCORE_DIR / "BLIND_MECHANICAL_SCORES.json"
    save(public_path, {
        "schema_version": "source_citation_control_blind_mechanical_scores_v1",
        "status": "SCORED_BLINDED_CONDITION_LABELS",
        "raw_responses_committed_before_gold_access": True,
        "scored_at_unix": time.time(),
        "planned_request_denominator": 24,
        "item_denominator": 144,
        "rows": score_rows,
    })
    private_path = SCORE_DIR / "private" / "UNBLIND_KEY.json"
    private_path.parent.mkdir(parents=True)
    save(private_path, {"salt": salt, "mapping": private_map})
    os.chmod(private_path, 0o600)
    freeze = {
        "schema_version": "source_citation_control_blind_score_freeze_v1",
        "status": "BLIND_MECHANICAL_SCORES_COMMITTED",
        "raw_response_freeze": binding(raw_freeze_path),
        "score_file": binding(public_path),
        "private_unblind_key": binding(private_path),
        "private_gold_key": binding(private_key_path),
        "request_file": binding(PARENT / "REQUESTS.json"),
        "metric_contract": {
            "exact_order_correct": "Final answer exactly equals frozen gold display order; malformed or missing output is false.",
            "all_judgments_supported": "All six supported dialogue statements receive judgment supported.",
            "citation_anchor_hit_rate": "Fraction of six displayed labels whose item_evidence.turn_refs contains that item's independently adjudicated first sufficient source turn.",
            "invalid_or_missing_outputs": "Remain in 24-request denominator and fail all substantive metrics.",
        },
        "condition_labels_in_public_score_file": False,
    }
    save(SCORE_DIR / "BLIND_SCORE_FREEZE.json", freeze)
    print(json.dumps({"status": freeze["status"], "requests": len(score_rows),
        "gold_key_read_after_raw_freeze": True, "condition_labels_released": False}))


if __name__ == "__main__":
    main()
