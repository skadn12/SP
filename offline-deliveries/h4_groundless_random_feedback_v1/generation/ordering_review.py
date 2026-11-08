"""Independent Qwen ordering review, frozen request preparation and bridge.

The module deliberately does not import historical DeepSeek/GLM/Codex verdicts.
Preparation is offline. ``live`` is the only command that can read credentials
and make one explicitly reserved request per root.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

try:
    from .prompts import LABELS, MODEL, build_ordering_review_prompt
except ImportError:  # Support direct script/unittest-discover execution.
    from prompts import LABELS, MODEL, build_ordering_review_prompt


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE_DIR = PROJECT / "offline-deliveries/h4_crossroot_ordering_batch_v5_fresh_roots"
CANDIDATE_FILE = PROJECT / "offline-deliveries/h4_crossroot_ordering_no_codex_pilot_v1/LABEL_REBOUND_UNREVIEWED.json"
RUNS_DIR = HERE / "ordering_reviews"
ROOT_IDS = tuple(f"cc-train-episode-{suffix}" for suffix in ("210046", "231899", "194259", "96544"))
ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXPECTED_MODEL = MODEL


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: bytes | Path) -> str:
    data = value.read_bytes() if isinstance(value, Path) else value
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    """Create an artifact without replacing a previous frozen result."""
    if path.exists():
        raise FileExistsError(f"artifact_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def root_candidates(require_all: bool = True) -> dict[str, list[dict[str, Any]]]:
    rows = read(CANDIDATE_FILE)
    if not isinstance(rows, list):
        raise ValueError("candidate_file_not_array")
    result = {root: [] for root in ROOT_IDS}
    for row in rows:
        if not isinstance(row, dict) or row.get("root_id") not in result:
            continue
        result[row["root_id"]].append(deepcopy(row))
    if require_all and any(not rows for rows in result.values()):
        raise ValueError("candidate_root_missing")
    return result


def source_for(root_id: str) -> dict[str, Any]:
    source = read(SOURCE_DIR / root_id / "source.json")
    if source.get("root_id") != root_id or not isinstance(source.get("turns"), list):
        raise ValueError("source_binding_invalid")
    refs = [turn.get("turn_ref") for turn in source["turns"]]
    if any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != len(refs):
        raise ValueError("source_refs_not_unique")
    return source


def _candidate_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    # Historical candidate text is copied into the request, never rewritten.
    return {key: deepcopy(row[key]) for key in
            ("candidate_id", "root_id", "question_type", "chronology_kind", "question", "items", "gold", "item_evidence")}


def _selected(rows: list[dict[str, Any]], candidate_ids: set[str] | None) -> list[dict[str, Any]]:
    selected = [row for row in rows if candidate_ids is None or row["candidate_id"] in candidate_ids]
    if not selected:
        raise ValueError("candidate_subset_empty")
    if any(row.get("question_type") != "ordering" or row.get("chronology_kind") != "conversation_order" for row in selected):
        raise ValueError("candidate_contract_invalid")
    return selected


def prepare(pass_id: str, candidate_ids: Iterable[str] | None = None,
            feedback_suffix: str = "") -> dict[str, Any]:
    """Freeze one independent request per root; this function never calls HTTP."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", pass_id):
        raise ValueError("pass_id_invalid")
    ids = set(candidate_ids) if candidate_ids is not None else None
    by_root = root_candidates(require_all=candidate_ids is None)
    if ids is not None:
        unknown = ids - {row["candidate_id"] for rows in by_root.values() for row in rows}
        if unknown:
            raise ValueError("candidate_id_not_found")
    selected_roots = [root_id for root_id in ROOT_IDS
                      if any(ids is None or row["candidate_id"] in ids for row in by_root[root_id])]
    if not selected_roots:
        raise ValueError("candidate_subset_empty")
    run = RUNS_DIR / pass_id
    if run.exists():
        raise FileExistsError(f"review_run_exists:{run}")
    run.mkdir(parents=True)
    written = []
    try:
        for root_id in selected_roots:
            rows = _selected(by_root[root_id], ids)
            source = source_for(root_id)
            prompt = build_ordering_review_prompt(root_id, [_candidate_for_prompt(r) for r in rows], source["turns"], feedback_suffix)
            # Keep the frozen request within the active SiliconFlow profile's
            # explicit output_limit.  A larger declaration is rejected before
            # HTTP, so it cannot improve review quality.
            # This mirrors the active SiliconFlow provider profile exactly;
            # transport constructs the HTTP payload from these fields.
            payload = {"model": EXPECTED_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.25, "max_tokens": 4096,
                       "response_format": {"type": "json_object"}, "enable_thinking": False}
            directory = run / root_id
            write_new(directory / "REQUEST.json", payload)
            write_new(directory / "INDEX.json", {
                "status": "PREPARED_NO_API_CALL", "pass_id": pass_id, "root_id": root_id,
                "reviewer_model_requested": EXPECTED_MODEL, "candidate_ids": [r["candidate_id"] for r in rows],
                "source_path": str(SOURCE_DIR / root_id / "source.json"),
                "source_sha256": sha(SOURCE_DIR / root_id / "source.json"),
                "candidate_file_sha256": sha(CANDIDATE_FILE),
                "prompt_sha256": sha(prompt.encode("utf-8")), "payload_sha256": sha(canonical(payload)),
                "feedback_suffix_sha256": sha(feedback_suffix.strip().encode("utf-8")),
                "historical_verdicts_in_prompt": False,
            })
            written.append(root_id)
    except Exception:
        # Preparation is all-or-nothing; a partial run must not be mistaken for a frozen batch.
        for path in sorted(run.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(run.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        run.rmdir()
        raise
    write_new(run / "RUN_INDEX.json", {"status": "PREPARED_NO_API_CALL", "pass_id": pass_id,
                                      "reviewer_model": EXPECTED_MODEL, "roots": written,
                                      "candidate_ids": sorted(ids) if ids is not None else None,
                                      "historical_verdicts_in_prompt": False,
                                      "feedback_suffix_present": bool(feedback_suffix.strip())})
    return {"status": "PREPARED_NO_API_CALL", "run": str(run), "roots": written, "http_calls": 0}


def _run_roots(pass_id: str) -> list[str]:
    """Return the roots actually frozen for a run, including subset passes."""
    index = read(RUNS_DIR / pass_id / "RUN_INDEX.json")
    roots = index.get("roots")
    if not isinstance(roots, list) or not roots or any(root not in ROOT_IDS for root in roots):
        raise ValueError("run_roots_invalid")
    return roots


def _transport():
    worktree = PROJECT / "worktrees/h4-writer-coverage-v1"
    import sys
    if str(worktree) not in sys.path:
        sys.path.insert(0, str(worktree))
    from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import credential_client, provider_profile
    return credential_client, provider_profile


def live(pass_id: str, root_id: str) -> dict[str, Any]:
    """Execute exactly one frozen root request. Credentials are read only here."""
    if root_id not in ROOT_IDS:
        raise ValueError("root_id_invalid")
    directory = RUNS_DIR / pass_id / root_id
    index, payload = read(directory / "INDEX.json"), read(directory / "REQUEST.json")
    prompt = payload["messages"][0]["content"]
    if index["payload_sha256"] != sha(canonical(payload)) or index["prompt_sha256"] != sha(prompt.encode("utf-8")):
        raise ValueError("frozen_request_changed")
    if (directory / "RESPONSE.json").exists() or (directory / "ERROR.json").exists():
        raise FileExistsError("review_attempt_already_completed")
    write_new(directory / "ATTEMPT.json", {"status": "RESERVED_ONE_HTTP_ATTEMPT", "created_at": datetime.now(timezone.utc).isoformat(),
                                           "requested_model": EXPECTED_MODEL, "payload_sha256": index["payload_sha256"]})
    credential_client, provider_profile = _transport()
    profile = provider_profile()
    attempts = []
    def on_attempt(actual: dict[str, Any]) -> None:
        if actual != payload:
            raise ValueError("actual_payload_changed")
        attempts.append(1)
        write_new(directory / "HTTP_ATTEMPT.json", {"attempt_number": 1, "requested_model": actual.get("model"),
                                                     "payload_sha256": sha(canonical(actual)), "maximum_http_retries": 0})
    try:
        response = credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=payload["max_tokens"], json_object=True)
    except Exception as error:
        write_new(directory / "ERROR.json", {"status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
                                              "payload_sha256": index["payload_sha256"], "attempts": len(attempts)})
        raise
    metadata = getattr(response, "metadata", {}) or {}
    raw = metadata.get("raw_response") or {}
    write_new(directory / "RESPONSE.json", {"status": "RESPONSE_RECEIVED_REVIEW_PENDING", "requested_model": EXPECTED_MODEL,
                                             "returned_model": metadata.get("returned_model"), "response_id": raw.get("id"),
                                             "usage": metadata.get("usage"), "raw_text": response.raw_text,
                                             "request_path": str(directory / "REQUEST.json"),
                                             "request_sha256": index["payload_sha256"],
                                             "response_sha256": sha(response.raw_text.encode("utf-8"))})
    return {"status": "RESPONSE_RECEIVED_REVIEW_PENDING", "root_id": root_id,
            "returned_model": metadata.get("returned_model"), "attempts": len(attempts)}


def _review_object(raw_text: str, expected_ids: list[str]) -> dict[str, Any]:
    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("review_json_invalid") from error
    if not isinstance(obj, dict) or set(obj) != {"reviews"} or not isinstance(obj["reviews"], list):
        raise ValueError("review_schema_top_level")
    if [row.get("candidate_id") for row in obj["reviews"] if isinstance(row, dict)] != expected_ids:
        raise ValueError("review_candidate_order_or_count")
    for row in obj["reviews"]:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "items", "coherent", "unique_order", "alternative_orders", "neutral_stem_ok", "reason"}:
            raise ValueError("review_question_schema")
        if set(row["items"]) != set(LABELS) or not isinstance(row["alternative_orders"], list):
            raise ValueError("review_items_schema")
        for label in LABELS:
            item = row["items"][label]
            required = {"supported", "order_anchor", "support_refs", "alternative_matching_turn_refs", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok", "reason"}
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("review_item_schema")
            anchor = item["order_anchor"]
            if not isinstance(anchor, dict) or set(anchor) != {"turn_ref", "reason"} or not isinstance(anchor["turn_ref"], str):
                raise ValueError("review_anchor_schema")
            for field in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok"):
                if not isinstance(item[field], bool):
                    raise ValueError("review_boolean_schema")
            if not isinstance(item["support_refs"], list) or not all(isinstance(v, str) for v in item["support_refs"]):
                raise ValueError("review_support_refs_schema")
            if not isinstance(item["alternative_matching_turn_refs"], list) or not all(isinstance(v, str) for v in item["alternative_matching_turn_refs"]):
                raise ValueError("review_alternatives_schema")
        for order in row["alternative_orders"]:
            if (not isinstance(order, list) or len(order) != len(LABELS)
                    or set(order) != set(LABELS)):
                raise ValueError("review_alternative_order_schema")
        for field in ("coherent", "unique_order", "neutral_stem_ok"):
            if not isinstance(row[field], bool):
                raise ValueError("review_question_boolean_schema")
    return obj


def review(pass_id: str, root_id: str | None = None) -> dict[str, Any]:
    run_roots = _run_roots(pass_id)
    roots = (root_id,) if root_id else tuple(run_roots)
    if any(root not in run_roots for root in roots):
        raise ValueError("root_not_in_run")
    result = []
    rejected = []
    for rid in roots:
        directory = RUNS_DIR / pass_id / rid
        index, response = read(directory / "INDEX.json"), read(directory / "RESPONSE.json")
        try:
            parsed = _review_object(response["raw_text"], index["candidate_ids"])
        except ValueError as error:
            # Keep malformed or incomplete model output auditable and let the
            # bridge reject this root without preventing other roots from
            # completing.  No partial candidate is admitted.
            artifact = {"status": "QWEN_REVIEW_SCHEMA_REJECTED", "pass_id": pass_id, "root_id": rid,
                        "reviewer_model": response.get("returned_model") or EXPECTED_MODEL,
                        "request_path": str(directory / "REQUEST.json"), "request_sha256": index["payload_sha256"],
                        "response_path": str(directory / "RESPONSE.json"), "response_sha256": sha((directory / "RESPONSE.json").read_bytes()),
                        "error": str(error), "reviews": []}
            write_new(directory / "REVIEW.json", artifact)
            rejected.append({"root_id": rid, "error": str(error)})
            continue
        artifact = {"status": "QWEN_REVIEW_PARSED", "pass_id": pass_id, "root_id": rid,
                    "reviewer_model": response.get("returned_model") or EXPECTED_MODEL,
                    "request_path": str(directory / "REQUEST.json"), "request_sha256": index["payload_sha256"],
                    "response_path": str(directory / "RESPONSE.json"), "response_sha256": sha((directory / "RESPONSE.json").read_bytes()),
                    "reviews": parsed["reviews"]}
        write_new(directory / "REVIEW.json", artifact)
        result.append(rid)
    return {"status": "QWEN_REVIEW_COMPLETED_WITH_REJECTIONS" if rejected else "QWEN_REVIEW_PARSED",
            "pass_id": pass_id, "roots": result, "rejected": rejected}


def _source_positions(root_id: str) -> dict[str, int]:
    return {turn["turn_ref"]: position for position, turn in enumerate(source_for(root_id)["turns"])}


def bridge(pass_id: str) -> dict[str, Any]:
    """Create a Solver-ready pool using only freshly parsed Qwen reviews."""
    by_root = root_candidates(require_all=False)
    pool: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    run = RUNS_DIR / pass_id
    for root_id in _run_roots(pass_id):
        directory = run / root_id
        index, review_artifact = read(directory / "INDEX.json"), read(directory / "REVIEW.json")
        positions = _source_positions(root_id)
        source_hash = sha(SOURCE_DIR / root_id / "source.json")
        rows = {row["candidate_id"]: row for row in by_root[root_id]}
        if review_artifact.get("status") != "QWEN_REVIEW_PARSED":
            for candidate_id in index["candidate_ids"]:
                rejected.append({"candidate_id": candidate_id, "root_id": root_id,
                                 "reasons": ["qwen_review_schema_rejected:" + str(review_artifact.get("error", "unknown"))]})
            continue
        reviews = {row["candidate_id"]: row for row in review_artifact["reviews"]}
        for candidate_id in index["candidate_ids"]:
            candidate, verdict = rows[candidate_id], reviews[candidate_id]
            reasons = []
            anchors = {}
            for label in LABELS:
                item = verdict["items"][label]
                ref = item["order_anchor"]["turn_ref"]
                if ref not in positions:
                    reasons.append(f"{label}:anchor_not_in_source")
                anchors[label] = ref
                if not (item["supported"] and item["actor_ok"] and item["modality_ok"] and item["state_ok"] and item["neutral_stem_ok"]):
                    reasons.append(f"{label}:qwen_item_check_failed")
                if any(ref2 not in positions for ref2 in item["support_refs"] + item["alternative_matching_turn_refs"]):
                    reasons.append(f"{label}:context_ref_not_in_source")
            if len(set(anchors.values())) != len(LABELS):
                reasons.append("order_anchors_not_distinct")
            computed = sorted(LABELS, key=lambda label: positions[anchors[label]]) if not reasons else []
            if not verdict["coherent"] or not verdict["unique_order"] or verdict["alternative_orders"] or not verdict["neutral_stem_ok"]:
                reasons.append("qwen_question_check_failed")
            if reasons:
                rejected.append({"candidate_id": candidate_id, "root_id": root_id, "reasons": sorted(set(reasons))})
                continue
            row = deepcopy(candidate)
            original_gold = row.get("gold")
            row["gold"] = computed
            row["original_gold"] = original_gold
            row["gold_recomputed_from_order_anchor"] = True
            row["order_anchor"] = anchors
            row["support_refs"] = {label: verdict["items"][label]["support_refs"] for label in LABELS}
            row["qwen_review"] = {"status": "PASS", "returned_model": review_artifact["reviewer_model"],
                                   "request_path": str(directory / "REQUEST.json"), "request_sha256": index["payload_sha256"],
                                   "response_path": str(directory / "RESPONSE.json"), "response_sha256": review_artifact["response_sha256"]}
            row["provenance"] = {"codex_question_edits": False, "review_owner_model": EXPECTED_MODEL,
                                  "review_pass_id": pass_id, "source_sha256": source_hash}
            row["status"] = "QWEN_REVIEWED_MECHANICALLY_VERIFIED_READY_FOR_SOLVER"
            pool.append(row)
    output = {"status": "QWEN_REVIEWED_MECHANICALLY_VERIFIED_READY_FOR_SOLVER", "pass_id": pass_id,
              "review_owner_model": EXPECTED_MODEL, "codex_question_edits": False,
              "historical_deepseek_glm_verdicts_used": False, "codex_semantic_review_used": False,
              "roots": _run_roots(pass_id), "selected_count": len(pool), "rejected_count": len(rejected),
              "questions": pool, "rejected": rejected}
    write_new(run / "REVIEWED_POOL.json", output)
    return {"status": output["status"], "pass_id": pass_id, "selected_count": len(pool), "rejected_count": len(rejected)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--pass-id", required=True); p.add_argument("--candidate-id", action="append", default=[]); p.add_argument("--feedback-suffix", default="")
    p = sub.add_parser("live"); p.add_argument("--pass-id", required=True); p.add_argument("--root-id", required=True)
    p = sub.add_parser("review"); p.add_argument("--pass-id", required=True); p.add_argument("--root-id")
    p = sub.add_parser("bridge"); p.add_argument("--pass-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.pass_id, args.candidate_id or None, args.feedback_suffix)
    elif args.command == "live":
        value = live(args.pass_id, args.root_id)
    elif args.command == "review":
        value = review(args.pass_id, args.root_id)
    else:
        value = bridge(args.pass_id)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
