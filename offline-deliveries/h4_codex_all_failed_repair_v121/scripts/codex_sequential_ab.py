"""Immutable A3 -> B3 artifact controller. This script never invokes a model.

One scene is materialized at a time. A different declared reader must review
the complete cumulative source before its successor can be materialized.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from codex_controlled_batch import digest, now, read, root_folder, save

VERSION = "codex_sequential_ab_v1"
SCENES = tuple(f"R02:{route}:S{number:02d}" for route in "AB" for number in range(1, 4))
CHECKS = {"time", "location", "actor_identity", "plan_vs_completion",
          "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity"}
TURN_FIELDS = {"turn_ref", "session_ref", "scene_ref", "speaker", "text", "origin"}
TERMINAL_RECEIPTS = {"STRUCTURAL_FAILURE", "INTEGRITY_FAILURE", "REVIEW_INVALID", "REVIEW_HOLD"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_from(batch, value):
    path = Path(value)
    return path if path.is_absolute() else Path(batch) / path


def load_config(batch):
    cfg = read(Path(batch) / "RUN_CONFIG.json")
    require(cfg["controller_version"] == VERSION, "controller_version")
    require(cfg.get("scene_sequence") == list(SCENES), "scene_sequence_must_be_A3_then_B3")
    require(cfg.get("new_turn_count") == {"min": 12, "max": 16}, "new_turn_count_must_be_12_to_16")
    roots = cfg["roots"]
    ids = [row["root_id"] if isinstance(row, dict) else row for row in roots]
    require(bool(ids) and len(ids) == len(set(ids)), "empty_or_duplicate_roots")
    for root_id in ids:
        root_folder(batch, root_id)
    return cfg


def root_config(cfg, root_id):
    for row in cfg["roots"]:
        if (row["root_id"] if isinstance(row, dict) else row) == root_id:
            return row if isinstance(row, dict) else {"root_id": row}
    raise ValueError("root_not_in_run_config")


def scene_folder(batch, root_id, scene_ref):
    require(scene_ref in SCENES, "invalid_scene_ref")
    return root_folder(batch, root_id) / "scenes" / scene_ref.removeprefix("R02:").replace(":", "_")


def append_receipt(folder, status, **fields):
    stamp = now()
    destination = Path(folder) / "receipts" / (stamp.replace(":", "") + "-" + uuid.uuid4().hex + ".json")
    save(destination, {"controller_version": VERSION, "created_at_utc": stamp,
                       "status": status, **fields})
    return destination


def receipts(folder):
    return [read(path) for path in sorted((Path(folder) / "receipts").glob("*.json"))]


def load_baseline(batch, cfg, root_id):
    row = root_config(cfg, root_id)
    short = root_id.rsplit("-", 1)[-1]
    path = path_from(batch, row.get("baseline_path", f"inputs/root{short}_BASELINE.json"))
    source = read(path)
    require(source["root_id"] == root_id, "baseline_root_id")
    require(isinstance(source["turns"], list) and bool(source["turns"]), "empty_baseline")
    require(len({t["turn_ref"] for t in source["turns"]}) == len(source["turns"]), "baseline_duplicate_turn_refs")
    if row.get("baseline_sha256"):
        require(digest(path) == row["baseline_sha256"], "baseline_hash")
    return path, source


def reservation_for(batch, cfg, root_id, scene_ref):
    if ":A:" in scene_ref:
        return None
    rows = read(Path(batch) / "SOURCE_RESERVATIONS.json")["allocation"]
    keys = [(row["root_id"], row["scene_ref"]) for row in rows]
    require(len(keys) == len(set(keys)), "duplicate_root_scene_reservation")
    require(len({row["source_id"] for row in rows}) == len(rows), "duplicate_reserved_source_id")
    root_ids = {row["root_id"] if isinstance(row, dict) else row for row in cfg["roots"]}
    expected = {(rid, scene) for rid in root_ids for scene in SCENES if ":B:" in scene}
    require(set(keys) == expected, "reservation_slots_incomplete_or_extra")
    row = next(row for row in rows if (row["root_id"], row["scene_ref"]) == (root_id, scene_ref))
    short = root_id.rsplit("-", 1)[-1]
    default = f"inputs/root{short}_{scene_ref.removeprefix('R02:').replace(':', '_')}_EVENT.json"
    path = path_from(batch, row.get("source_item_path", default))
    require(digest(path) == row["source_item_sha256"], "event_hash")
    return {**row, "source_item_path": str(path)}


def validate_draft(batch, cfg, root_id, scene_ref, input_path, baseline):
    folder = scene_folder(batch, root_id, scene_ref)
    path = folder / "EXPANSION_DRAFT.json"
    draft = read(path)
    require(draft["root_id"] == root_id, "draft_root_id")
    require(draft["scene_ref"] == scene_ref, "draft_scene_ref")
    require(isinstance(draft.get("author"), str) and draft["author"].strip(), "draft_author_required")
    require(draft["input_source_sha256"] == digest(input_path), "draft_input_source_hash")
    reservation = reservation_for(batch, cfg, root_id, scene_ref)
    if reservation:
        require(draft["source_id"] == reservation["source_id"], "draft_source_id")
    else:
        require(not draft.get("source_id"), "A_route_must_not_transplant_external_source")
    turns = draft["turns"]
    require(isinstance(turns, list) and 12 <= len(turns) <= 16, "turn_count")
    speakers = {turn["speaker"] for turn in baseline["turns"]}
    for index, turn in enumerate(turns, 1):
        require(set(turn) == TURN_FIELDS, "turn_fields")
        require(turn["turn_ref"] == f"{scene_ref}:T{index:03d}", "turn_ref")
        require(turn["session_ref"] == turn["scene_ref"] == scene_ref, "turn_scene_ref")
        require(turn["speaker"] in speakers, "speaker_not_established")
        require(isinstance(turn["text"], str) and bool(turn["text"].strip()), "empty_turn")
        require(turn["origin"] == "codex_session_generated", "turn_origin")
    previous = read(input_path)["turns"]
    cumulative = previous + turns
    require(len({turn["turn_ref"] for turn in cumulative}) == len(cumulative), "duplicate_turn_refs")
    return path, draft, cumulative, reservation


def expected_source(batch, cfg, root_id, scene_ref, input_path, baseline_path, baseline):
    draft_path, draft, turns, reservation = validate_draft(batch, cfg, root_id, scene_ref, input_path, baseline)
    result = {"schema_version": VERSION + "_source", "root_id": root_id, "scene_ref": scene_ref,
              "baseline_sha256": digest(baseline_path), "input_source_sha256": digest(input_path),
              "draft_sha256": digest(draft_path), "baseline_turn_count": len(baseline["turns"]),
              "previous_turn_count": len(read(input_path)["turns"]), "new_turn_count": len(draft["turns"]),
              "cumulative_new_turn_count": len(turns) - len(baseline["turns"]), "turns": turns}
    if reservation:
        result.update(source_id=reservation["source_id"], source_item_sha256=reservation["source_item_sha256"])
    return result


def validate_materialization(batch, cfg, root_id, scene_ref, input_path, baseline_path, baseline):
    folder = scene_folder(batch, root_id, scene_ref)
    path = folder / "CUMULATIVE_SOURCE.json"
    expected = expected_source(batch, cfg, root_id, scene_ref, input_path, baseline_path, baseline)
    require(read(path) == expected, "cumulative_source_not_exact_prefix_plus_draft")
    matching = [r for r in receipts(folder) if r["status"] == "STRUCTURAL_PASS"]
    require(len(matching) == 1, "exactly_one_structural_pass_receipt_required")
    receipt = matching[0]
    require(receipt["source_sha256"] == digest(path), "materialized_source_hash")
    require(receipt["draft_sha256"] == expected["draft_sha256"], "materialized_draft_hash")
    require(receipt["input_source_sha256"] == expected["input_source_sha256"], "materialized_input_hash")
    require(receipt["run_config_sha256"] == digest(Path(batch) / "RUN_CONFIG.json"), "run_config_changed_after_materialization")
    require(receipt["source_reservations_sha256"] == digest(Path(batch) / "SOURCE_RESERVATIONS.json"), "source_reservations_changed_after_materialization")
    return path, read(path)


def validate_review(folder, root_id, scene_ref, source_path, source):
    path = folder / "SOURCE_REVIEW.json"
    review = read(path)
    draft = read(folder / "EXPANSION_DRAFT.json")
    require(review["root_id"] == root_id and review["scene_ref"] == scene_ref, "review_identity")
    require(review["source_sha256"] == digest(source_path), "review_source_hash")
    reviewer = review.get("reviewer")
    require(isinstance(reviewer, str) and reviewer.strip() and reviewer != draft["author"], "reviewer_must_differ_from_author")
    require(review["full_source_read"] is True, "full_source_read_not_attested")
    require(review["turn_refs_read"] == [turn["turn_ref"] for turn in source["turns"]], "review_full_cumulative_coverage")
    timestamp = datetime.fromisoformat(review["created_at_utc"].replace("Z", "+00:00"))
    require(timestamp.tzinfo is not None, "review_timestamp_requires_timezone")
    materialized = next(receipt for receipt in receipts(folder) if receipt["status"] == "STRUCTURAL_PASS")
    require(timestamp >= datetime.fromisoformat(materialized["created_at_utc"]), "review_timestamp_before_materialization")
    require(review["decision"] in {"READY", "HOLD"}, "invalid_review_decision")
    require(isinstance(review["findings"], list), "findings_must_be_list")
    require(CHECKS.issubset(review["checks"]), "review_checks_incomplete")
    for name in CHECKS:
        check = review["checks"][name]
        require(isinstance(check.get("reason"), str) and check["reason"].strip(), "review_check_reason_missing:" + name)
        require(check.get("status") in {"PASS", "WARNING", "PASS_WITH_WARNING", "FAIL"}, "invalid_check_status:" + name)
    if review["decision"] == "READY":
        require(all(review["checks"][name]["status"] != "FAIL" for name in CHECKS), "READY_with_failed_check")
        require(not any(finding.get("severity") == "BLOCKING" for finding in review["findings"]), "READY_with_blocking_finding")
    prior = [r for r in receipts(folder) if r["status"] in {"REVIEW_READY", "REVIEW_HOLD"}]
    require(all(r["review_sha256"] == digest(path) for r in prior), "accepted_review_was_modified")
    return path, review, prior


def inspect_root(batch, cfg, root_id, record_reviews=True):
    root_dir = root_folder(batch, root_id)
    root_failures = [r for r in receipts(root_dir) if r["status"] in TERMINAL_RECEIPTS]
    try:
        require(not root_failures, root_failures[0].get("reason", "root_integrity_failure") if root_failures else "")
        baseline_path, baseline = load_baseline(batch, cfg, root_id)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        if record_reviews and not root_failures:
            append_receipt(root_dir, "INTEGRITY_FAILURE", root_id=root_id, reason=str(exc))
        states = [{"scene_ref": scene, "folder": str(scene_folder(batch, root_id, scene)),
                   "status": "INTEGRITY_FAILURE" if not index else "BLOCKED_BY_PREDECESSOR",
                   **({"reason": str(exc)} if not index else {})} for index, scene in enumerate(SCENES)]
        return {"root_id": root_id, "full_expansion_complete": False, "root_stopped": True,
                "ready_scene_count": 0, "expected_scene_count": 6, "scenes": states}
    states = []
    input_path = baseline_path
    predecessor_ready = True
    for scene_ref in SCENES:
        folder = scene_folder(batch, root_id, scene_ref)
        state = {"scene_ref": scene_ref, "folder": str(folder)}
        prior = receipts(folder)
        terminal = [r for r in prior if r["status"] in TERMINAL_RECEIPTS]
        if terminal:
            state.update(status=terminal[0]["status"], reason=terminal[0].get("reason", "review_on_hold"))
        elif not predecessor_ready:
            state["status"] = "BLOCKED_BY_PREDECESSOR"
            if (folder / "CUMULATIVE_SOURCE.json").exists():
                state.update(status="INTEGRITY_FAILURE", reason="materialized_after_non_ready_predecessor")
                if record_reviews:
                    append_receipt(folder, "INTEGRITY_FAILURE", root_id=root_id, scene_ref=scene_ref,
                                   reason="materialized_after_non_ready_predecessor")
        elif not (folder / "CUMULATIVE_SOURCE.json").exists():
            state["status"] = "DRAFT_WAITING_MATERIALIZATION" if (folder / "EXPANSION_DRAFT.json").exists() else "NOT_STARTED"
        else:
            try:
                source_path, source = validate_materialization(batch, cfg, root_id, scene_ref, input_path, baseline_path, baseline)
                state.update(source_sha256=digest(source_path), new_turn_count=source["new_turn_count"], cumulative_turn_count=len(source["turns"]))
                if not (folder / "SOURCE_REVIEW.json").exists():
                    state["status"] = "AWAITING_REVIEW"
                else:
                    review_path, review, prior_reviews = validate_review(folder, root_id, scene_ref, source_path, source)
                    state.update(status="READY" if review["decision"] == "READY" else "REVIEW_HOLD",
                                 reviewer=review["reviewer"], review_sha256=digest(review_path))
                    if record_reviews and not prior_reviews:
                        append_receipt(folder, "REVIEW_" + review["decision"], root_id=root_id, scene_ref=scene_ref,
                                       source_sha256=digest(source_path), review_sha256=digest(review_path), reviewer=review["reviewer"])
                input_path = source_path
            except (ValueError, KeyError, TypeError, OSError) as exc:
                state.update(status="INTEGRITY_FAILURE", reason=str(exc))
                if record_reviews:
                    append_receipt(folder, "INTEGRITY_FAILURE", root_id=root_id, scene_ref=scene_ref, reason=str(exc))
        states.append(state)
        predecessor_ready = predecessor_ready and state["status"] == "READY"
    complete = len(states) == 6 and all(state["status"] == "READY" for state in states)
    stopped = any(state["status"] in TERMINAL_RECEIPTS for state in states)
    return {"root_id": root_id, "full_expansion_complete": complete, "root_stopped": stopped,
            "ready_scene_count": sum(state["status"] == "READY" for state in states),
            "expected_scene_count": 6, "scenes": states}


def materialize_scene(batch, root_id, scene_ref):
    batch = Path(batch)
    cfg = load_config(batch)
    root_config(cfg, root_id)
    folder = scene_folder(batch, root_id, scene_ref)
    state = inspect_root(batch, cfg, root_id)
    index = SCENES.index(scene_ref)
    current = state["scenes"][index]
    if state["root_stopped"]:
        return {"root_id": root_id, "scene_ref": scene_ref, "status": "ROOT_STOPPED", "progress": state}
    if index and state["scenes"][index - 1]["status"] != "READY":
        append_receipt(folder, "PREDECESSOR_NOT_READY", root_id=root_id, scene_ref=scene_ref)
        return {"root_id": root_id, "scene_ref": scene_ref, "status": "PREDECESSOR_NOT_READY"}
    if current["status"] in {"READY", "AWAITING_REVIEW"}:
        return {"root_id": root_id, "scene_ref": scene_ref, "status": current["status"], "already_materialized": True}
    if not (folder / "EXPANSION_DRAFT.json").exists():
        return {"root_id": root_id, "scene_ref": scene_ref, "status": "NO_DRAFT"}
    try:
        baseline_path, baseline = load_baseline(batch, cfg, root_id)
        input_path = baseline_path if not index else scene_folder(batch, root_id, SCENES[index - 1]) / "CUMULATIVE_SOURCE.json"
        result = expected_source(batch, cfg, root_id, scene_ref, input_path, baseline_path, baseline)
        output = folder / "CUMULATIVE_SOURCE.json"
        save(output, result)
        receipt = append_receipt(folder, "STRUCTURAL_PASS", root_id=root_id, scene_ref=scene_ref,
                                 source_sha256=digest(output), draft_sha256=result["draft_sha256"],
                                 input_source_sha256=result["input_source_sha256"], prefix_exact=True,
                                 run_config_sha256=digest(batch / "RUN_CONFIG.json"),
                                 source_reservations_sha256=digest(batch / "SOURCE_RESERVATIONS.json"),
                                 source_semantic_ready=False, new_turn_count=result["new_turn_count"])
        return {"root_id": root_id, "scene_ref": scene_ref, "status": "STRUCTURAL_PASS_ONLY",
                "source_path": str(output), "receipt_path": str(receipt), "full_expansion_complete": False}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        receipt = append_receipt(folder, "STRUCTURAL_FAILURE", root_id=root_id, scene_ref=scene_ref,
                                 reason=str(exc), draft_sha256=digest(folder / "EXPANSION_DRAFT.json"))
        return {"root_id": root_id, "scene_ref": scene_ref, "status": "STRUCTURAL_FAILURE", "reason": str(exc), "receipt_path": str(receipt)}


def progress(batch, root_id=None):
    batch = Path(batch)
    cfg = load_config(batch)
    ids = [row["root_id"] if isinstance(row, dict) else row for row in cfg["roots"]]
    if root_id:
        root_config(cfg, root_id)
        ids = [root_id]
    roots = [inspect_root(batch, cfg, rid) for rid in ids]
    result = {"controller_version": VERSION, "created_at_utc": now(), "roots": roots,
              "full_expansion_complete": len(roots) == len(cfg["roots"]) and all(row["full_expansion_complete"] for row in roots),
              "solver_run": False, "training_run": False}
    receipt = append_receipt(batch, "PROGRESS_SNAPSHOT", progress=result)
    return {**result, "receipt_path": str(receipt)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["materialize-scene", "progress"])
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--root-id")
    parser.add_argument("--scene-ref", choices=SCENES)
    args = parser.parse_args()
    if args.command == "materialize-scene":
        if not (args.root_id and args.scene_ref):
            parser.error("materialize-scene requires --root-id and --scene-ref")
        result = materialize_scene(args.batch_dir, args.root_id, args.scene_ref)
    else:
        result = progress(args.batch_dir, args.root_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "materialize-scene" and result["status"] in {"STRUCTURAL_FAILURE", "ROOT_STOPPED", "PREDECESSOR_NOT_READY"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
