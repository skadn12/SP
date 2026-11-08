"""Configurable, frozen ordering-question generation; no Solver or training calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


HERE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
WORKTREE = PROJECT / "worktrees/h4-writer-coverage-v1"
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (
    MODEL, credential_client, provider_profile,
)
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload


VERSION = HERE.name
OUTPUT_LIMIT = 8192
INITIAL_COUNT = 3
TOPUP_COUNT = 2
MAX_TOTAL = 5
LABELS = {f"I{i}" for i in range(1, 7)}
STATUSES = {"statement", "report", "decision", "advice", "offer", "promise", "action"}
ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    """Publish a complete file without replacing an existing artifact."""
    content = (str(value) if raw else json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def profile() -> dict:
    result = provider_profile()
    if MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("unexpected_generator_model")
    result.update(profile_id=VERSION, output_limit=OUTPUT_LIMIT, http_retry_limit=0)
    result["request_parameters"].update(
        temperature=0.1, max_tokens=OUTPUT_LIMIT, enable_thinking=False,
        response_format={"type": "json_object"},
    )
    return result


def load_manifest(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    data = json.loads(raw)
    rows = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest_sources_must_be_nonempty_array")
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest_row_not_object")
        root_id, source_path, expected_hash = row.get("root_id"), row.get("source_path"), row.get("source_sha256")
        if not isinstance(root_id, str) or not ROOT_ID_RE.fullmatch(root_id) or root_id in seen:
            raise ValueError("manifest_root_id_invalid_or_duplicate")
        if not isinstance(source_path, str) or not Path(source_path).is_absolute():
            raise ValueError(f"manifest_source_path_must_be_absolute:{root_id}")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"manifest_source_sha256_invalid:{root_id}")
        prior_path, prior_hash = row.get("prior_candidates_path"), row.get("prior_candidates_sha256")
        if (prior_path is None) != (prior_hash is None):
            raise ValueError(f"manifest_prior_candidate_path_and_hash_must_be_paired:{root_id}")
        if prior_path is not None:
            if not isinstance(prior_path, str) or not Path(prior_path).is_absolute():
                raise ValueError(f"manifest_prior_candidates_path_must_be_absolute:{root_id}")
            if not isinstance(prior_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", prior_hash):
                raise ValueError(f"manifest_prior_candidates_sha256_invalid:{root_id}")
        seen.add(root_id)
        result.append({
            "root_id": root_id, "source_path": source_path, "source_sha256": expected_hash,
            "prior_candidates_path": prior_path, "prior_candidates_sha256": prior_hash,
        })
    return raw, result


def candidate_rows(data: object) -> list[dict]:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in ("questions", "candidates", "valid_candidates", "candidates_valid"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        if rows is None and any(key in data for key in ("candidate_id", "question", "question_text")):
            rows = [data]
        if rows is None:
            raise ValueError("prior_candidate_container_missing_candidate_array")
    else:
        raise ValueError("prior_candidate_pool_invalid")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("prior_candidate_row_not_object")
    return rows


def normalize_candidate_pool(data: object, root_id: str) -> list[dict]:
    """Keep only the fields needed to avoid repeating prior topics and evidence."""
    normalized = []
    for index, row in enumerate(candidate_rows(data), 1):
        candidate_root = row.get("root_id")
        if candidate_root is not None and candidate_root != root_id:
            raise ValueError(f"prior_candidate_root_mismatch:{root_id}:{index}")
        evidence = row.get("item_evidence") or row.get("option_evidence") or row.get("evidence") or {}
        refs = set()
        if isinstance(evidence, dict):
            for value in evidence.values():
                if isinstance(value, dict) and isinstance(value.get("turn_ref"), str) and value["turn_ref"]:
                    refs.add(value["turn_ref"])
        direct_refs = row.get("evidence_refs")
        if isinstance(direct_refs, list):
            refs.update(value for value in direct_refs if isinstance(value, str) and value)
        item_text = row.get("items") if isinstance(row.get("items"), dict) else row.get("options")
        if not isinstance(item_text, dict):
            item_text = {}
        item_text = {
            str(key): value.strip() for key, value in sorted(item_text.items(), key=lambda pair: str(pair[0]))
            if isinstance(value, str) and value.strip()
        }
        normalized.append({
            "candidate_id": row.get("candidate_id") if isinstance(row.get("candidate_id"), str) else f"prior-row-{index:04d}",
            "question": next((row[key].strip() for key in ("question", "question_text", "prompt")
                              if isinstance(row.get(key), str) and row[key].strip()), ""),
            "topic": row.get("topic", "") if isinstance(row.get("topic"), str) else "",
            "question_type": row.get("question_type", "") if isinstance(row.get("question_type"), str) else "",
            "evidence_refs": sorted(refs),
            "item_text": item_text,
        })
    return normalized


def checked_source(raw: bytes, root_id: str, expected_hash: str) -> dict:
    if sha(raw) != expected_hash:
        raise ValueError(f"source_hash_mismatch:{root_id}")
    source = json.loads(raw)
    turns = source.get("turns") if isinstance(source, dict) else None
    if (source.get("root_id") != root_id or not isinstance(source.get("turn_count"), int) or
            not isinstance(turns, list) or source["turn_count"] != len(turns) or not turns):
        raise ValueError(f"source_identity_or_turn_count_invalid:{root_id}")
    refs = []
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("turn_ref"), str) or not turn["turn_ref"] or not isinstance(turn.get("text"), str):
            raise ValueError(f"source_turn_invalid:{root_id}")
        refs.append(turn["turn_ref"])
    if len(set(refs)) != len(refs):
        raise ValueError(f"source_turn_refs_not_unique:{root_id}")
    return source


def candidate_ids(root_id: str, start: int, count: int) -> list[str]:
    return [f"{root_id}-order-{index:03d}" for index in range(start, start + count)]


def prompt_for(root_id: str, source: dict, instructions: str, ids: list[str], dedup_summary: list[dict]) -> str:
    surface = {key: source[key] for key in ("root_id", "turn_count", "round_count", "round_boundaries", "turns") if key in source}
    text = instructions.replace("{ROOT_ID}", root_id).replace("{COUNT}", str(len(ids)))
    text = text.replace("{CANDIDATE_IDS}", json.dumps(ids, ensure_ascii=False))
    dedup = {"prior_candidate_dedup_summary": dedup_summary}
    return (text + "\nPreviously covered candidates for deduplication:\n" +
            json.dumps(dedup, ensure_ascii=False, indent=2) +
            "\nSource metadata and complete dialogue follow as JSON:\n" +
            json.dumps(surface, ensure_ascii=False, indent=2) + "\n")


def request_bundle(root_id: str, source: dict, instructions: str, count: int, start: int,
                   dedup_summary: list[dict]) -> dict:
    prompt = prompt_for(root_id, source, instructions, candidate_ids(root_id, start, count), dedup_summary)
    payload = build_payload(profile(), prompt)
    return {
        "count": count, "first_candidate_number": start, "candidate_ids": candidate_ids(root_id, start, count),
        "prompt": prompt, "prompt_sha256": sha(prompt.encode("utf-8")),
        "payload": payload, "payload_sha256": sha(canonical(payload)),
        "dedup_summary_sha256": sha(canonical(dedup_summary)),
        "requested_model": MODEL, "provider": profile()["provider"],
        "status": "PREPARED_NO_API_CALL",
    }


def read_summary_file(path: Path, root_id: str) -> list[dict]:
    data = read_json(path)
    if not isinstance(data, dict) or data.get("root_id") != root_id or not isinstance(data.get("candidates"), list):
        raise ValueError(f"dedup_summary_invalid:{root_id}")
    summary = data["candidates"]
    if any(not isinstance(row, dict) for row in summary):
        raise ValueError(f"dedup_summary_row_invalid:{root_id}")
    return summary


def prepare(manifest_path: Path) -> None:
    batch_path = HERE / "BATCH_PREPARED.json"
    if batch_path.exists():
        raise FileExistsError("batch_already_prepared_no_overwrite")
    manifest_raw, rows = load_manifest(manifest_path)
    instructions_raw = (HERE / "instructions.txt").read_bytes()
    frozen_roots = []
    prepared = []
    for row in rows:
        root_id = row["root_id"]
        source_raw = Path(row["source_path"]).read_bytes()
        source = checked_source(source_raw, root_id, row["source_sha256"])
        directory = HERE / root_id
        if directory.exists():
            raise FileExistsError(f"root_directory_exists_no_overwrite:{root_id}")
        prior_raw = b""
        prior_summary = []
        if row["prior_candidates_path"] is not None:
            prior_raw = Path(row["prior_candidates_path"]).read_bytes()
            if sha(prior_raw) != row["prior_candidates_sha256"]:
                raise ValueError(f"prior_candidate_pool_hash_mismatch:{root_id}")
            prior_data = json.loads(prior_raw)
            prior_summary = normalize_candidate_pool(prior_data, root_id)
        bundle = request_bundle(root_id, source, instructions_raw.decode("utf-8"), INITIAL_COUNT, 1, prior_summary)
        prepared.append((row, source_raw, source, prior_raw, prior_summary, directory, bundle))
    for row, source_raw, source, prior_raw, prior_summary, directory, bundle in prepared:
        root_id = row["root_id"]
        directory.mkdir()
        write_new(directory / "source.json", source_raw.decode("utf-8"), raw=True)
        if row["prior_candidates_path"] is not None:
            write_new(directory / "prior_candidates.json", prior_raw.decode("utf-8"), raw=True)
            write_new(directory / "PRIOR_CANDIDATE_SUMMARY.json", {
                "status": "NORMALIZED_DEDUP_CONTEXT_FROZEN", "root_id": root_id,
                "original_pool_path": row["prior_candidates_path"],
                "original_pool_sha256": sha(prior_raw),
                "summary_sha256": sha(canonical(prior_summary)), "candidates": prior_summary,
            })
        else:
            write_new(directory / "PRIOR_CANDIDATE_SUMMARY.json", {
                "status": "NO_PRIOR_POOL_SUPPLIED", "root_id": root_id,
                "original_pool_path": None, "original_pool_sha256": None,
                "summary_sha256": sha(canonical([])), "candidates": [],
            })
        write_new(directory / "initial" / "prompt.txt", bundle["prompt"], raw=True)
        write_new(directory / "initial" / "request.json", bundle["payload"])
        bundle.pop("prompt")
        bundle.pop("payload")
        frozen_roots.append({
            **row, "source_turns": len(source["turns"]), "source_round_count": source.get("round_count"),
            "prior_candidate_summary_sha256": sha(canonical(prior_summary)),
            "initial": bundle, "topup": None,
        })
    write_new(batch_path, {
        "version": VERSION, "requested_model": MODEL, "profile": profile(),
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha(manifest_raw),
        "instructions_sha256": sha(instructions_raw), "script_sha256": sha(Path(__file__).read_bytes()),
        "initial_candidates_requested_per_root": INITIAL_COUNT,
        "maximum_total_candidates_per_root": MAX_TOTAL,
        "roots": frozen_roots, "generator_calls": 0, "solver_calls": 0, "training_started": False,
    })
    print(json.dumps({"status": "PREPARED_NO_API_CALL", "roots": len(frozen_roots), "initial_candidates_per_root": INITIAL_COUNT, "model": MODEL}))


def verify_batch() -> tuple[dict, list[dict]]:
    batch = read_json(HERE / "BATCH_PREPARED.json")
    manifest_path = Path(batch["manifest_path"])
    manifest_raw, rows = load_manifest(manifest_path)
    if (batch.get("version") != VERSION or batch.get("requested_model") != MODEL or
            batch.get("manifest_sha256") != sha(manifest_raw) or
            batch.get("instructions_sha256") != sha((HERE / "instructions.txt").read_bytes()) or
            batch.get("script_sha256") != sha(Path(__file__).read_bytes()) or batch.get("profile") != profile()):
        raise ValueError("batch_freeze_changed")
    if [row["root_id"] for row in batch.get("roots", [])] != [row["root_id"] for row in rows]:
        raise ValueError("manifest_roots_changed")
    return batch, rows


def frozen_request(root_id: str, stage: str) -> tuple[dict, dict, dict, Path]:
    batch, rows = verify_batch()
    row = next((item for item in rows if item["root_id"] == root_id), None)
    entry = next((item for item in batch["roots"] if item["root_id"] == root_id), None)
    if row is None or entry is None:
        raise ValueError("root_not_prepared")
    source_raw = (HERE / root_id / "source.json").read_bytes()
    source = checked_source(source_raw, root_id, row["source_sha256"])
    if sha(source_raw) != entry["source_sha256"]:
        raise ValueError("frozen_source_changed")
    prior_summary = read_summary_file(HERE / root_id / "PRIOR_CANDIDATE_SUMMARY.json", root_id)
    if row["prior_candidates_path"] is not None:
        prior_raw = (HERE / root_id / "prior_candidates.json").read_bytes()
        if sha(prior_raw) != row["prior_candidates_sha256"]:
            raise ValueError("frozen_prior_candidate_pool_changed")
        prior_summary_check = normalize_candidate_pool(json.loads(prior_raw), root_id)
        if (sha(canonical(prior_summary_check)) != sha(canonical(prior_summary)) or
                entry.get("prior_candidate_summary_sha256") != sha(canonical(prior_summary))):
            raise ValueError("frozen_prior_candidate_summary_changed")
    elif prior_summary:
        raise ValueError("unexpected_prior_candidate_summary")
    if stage == "initial":
        bundle = entry["initial"]
        directory = HERE / root_id / "initial"
        dedup_summary = read_summary_file(HERE / root_id / "PRIOR_CANDIDATE_SUMMARY.json", root_id)
        if bundle.get("dedup_summary_sha256") != sha(canonical(dedup_summary)):
            raise ValueError("initial_dedup_summary_changed")
    else:
        directory = HERE / root_id / "topup"
        index_path = directory / "TOPUP_INDEX.json"
        if not index_path.exists():
            raise ValueError("topup_not_prepared")
        index = read_json(index_path)
        if index.get("main_manifest_sha256") != batch["manifest_sha256"] or index.get("root_id") != root_id:
            raise ValueError("topup_main_batch_binding_changed")
        bundle = index.get("bundle")
        if not isinstance(bundle, dict) or bundle.get("count") != TOPUP_COUNT or bundle.get("first_candidate_number") != INITIAL_COUNT + 1:
            raise ValueError("topup_bundle_invalid")
        dedup_summary = read_summary_file(directory / "TOPUP_DEDUP_SUMMARY.json", root_id)
        if bundle.get("dedup_summary_sha256") != sha(canonical(dedup_summary)):
            raise ValueError("topup_dedup_summary_changed")
    prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
    payload = read_json(directory / "request.json")
    expected_start = bundle["first_candidate_number"]
    expected_prompt = prompt_for(root_id, source, (HERE / "instructions.txt").read_text(encoding="utf-8"), bundle["candidate_ids"], dedup_summary)
    expected_payload = build_payload(profile(), expected_prompt)
    if (sha(prompt.encode("utf-8")) != bundle["prompt_sha256"] or prompt != expected_prompt or
            sha(canonical(payload)) != bundle["payload_sha256"] or payload != expected_payload or
            bundle["candidate_ids"] != candidate_ids(root_id, expected_start, bundle["count"])):
        raise ValueError("frozen_request_changed")
    return entry, source, {"prompt": prompt, "payload": payload, **bundle}, directory


def prepare_topup(root_id: str, reason: str) -> None:
    batch, _ = verify_batch()
    entry = next((item for item in batch["roots"] if item["root_id"] == root_id), None)
    if entry is None:
        raise ValueError("root_not_prepared")
    initial_validation = HERE / root_id / "initial" / "VALIDATION.json"
    if not initial_validation.exists():
        raise ValueError("initial_response_must_be_mechanically_validated_before_topup")
    if not (HERE / root_id / "initial" / "VALIDATED_CANDIDATES.json").exists():
        raise ValueError("initial_candidate_summary_missing_before_topup")
    initial_result = read_json(initial_validation)
    if initial_result.get("stage") != "initial":
        raise ValueError("initial_validation_stage_mismatch")
    topup_dir = HERE / root_id / "topup"
    if topup_dir.exists() or entry.get("topup") is not None:
        raise FileExistsError("topup_already_prepared_no_overwrite")
    if not reason.strip():
        raise ValueError("topup_reason_required")
    completed_candidates = candidate_ids(root_id, 1, INITIAL_COUNT)
    count_already_requested = INITIAL_COUNT
    if count_already_requested + TOPUP_COUNT > MAX_TOTAL:
        raise ValueError("topup_limit_exceeded")
    source = json.loads((HERE / root_id / "source.json").read_text(encoding="utf-8"))
    prior_summary = read_summary_file(HERE / root_id / "PRIOR_CANDIDATE_SUMMARY.json", root_id)
    initial_candidates = read_json(HERE / root_id / "initial" / "VALIDATED_CANDIDATES.json")
    initial_summary = normalize_candidate_pool(initial_candidates, root_id)
    dedup_summary = prior_summary + initial_summary
    bundle = request_bundle(root_id, source, (HERE / "instructions.txt").read_text(encoding="utf-8"),
                            TOPUP_COUNT, INITIAL_COUNT + 1, dedup_summary)
    topup_dir.mkdir()
    write_new(topup_dir / "prompt.txt", bundle["prompt"], raw=True)
    write_new(topup_dir / "request.json", bundle["payload"])
    write_new(topup_dir / "DECISION.json", {
        "status": "TOPUP_PREPARED_BY_OPERATOR",
        "reason": reason, "prior_candidate_ids": completed_candidates,
        "additional_candidates_requested": TOPUP_COUNT, "total_requested_after_topup": MAX_TOTAL,
        "no_solver_calls": True, "prepared_at": datetime.now(timezone.utc).isoformat(),
    })
    write_new(topup_dir / "TOPUP_DEDUP_SUMMARY.json", {
        "status": "NORMALIZED_DEDUP_CONTEXT_FROZEN", "root_id": root_id,
        "prior_summary_sha256": sha(canonical(prior_summary)),
        "initial_validation_path": str(initial_validation),
        "summary_sha256": sha(canonical(dedup_summary)), "candidates": dedup_summary,
    })
    bundle.pop("prompt")
    bundle.pop("payload")
    write_new(HERE / root_id / "topup" / "TOPUP_INDEX.json", {
        "root_id": root_id, "bundle": bundle,
        "batch_version": VERSION, "main_manifest_sha256": batch["manifest_sha256"],
        "status": "PREPARED_NO_API_CALL", "solver_calls": 0,
    })
    print(json.dumps({"status": "TOPUP_PREPARED_NO_API_CALL", "root_id": root_id, "candidate_ids": bundle["candidate_ids"]}))


def get_entry(root_id: str) -> dict:
    batch, _ = verify_batch()
    entry = next((item for item in batch["roots"] if item["root_id"] == root_id), None)
    if entry is None:
        raise ValueError("root_not_prepared")
    topup_path = HERE / root_id / "topup" / "TOPUP_INDEX.json"
    if topup_path.exists():
        index = read_json(topup_path)
        if index.get("main_manifest_sha256") != batch["manifest_sha256"]:
            raise ValueError("topup_main_batch_binding_changed")
        entry["topup"] = index["bundle"]
    return entry


def execute(root_id: str, stage: str) -> None:
    entry, _, request, directory = frozen_request(root_id, stage)
    write_new(directory / "ATTEMPT_RESERVED.json", {
        "status": "RESERVED_ONE_GENERATOR_CALL", "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": profile()["provider"], "requested_model": MODEL,
        "payload_sha256": request["payload_sha256"], "maximum_http_attempts": 1,
    })

    def on_attempt(actual: dict) -> None:
        if actual != request["payload"]:
            raise ValueError("actual_payload_changed")
        write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "provider": profile()["provider"],
            "payload_sha256": sha(canonical(actual)), "requested_model": actual["model"],
            "max_tokens": actual["max_tokens"], "maximum_http_attempts": 1,
            "maximum_http_retries": 0,
        })

    try:
        response = credential_client(profile(), on_http_attempt=on_attempt).complete(
            prompt=request["prompt"], max_tokens=OUTPUT_LIMIT, json_object=True,
        )
    except Exception as error:
        metadata = getattr(error, "metadata", None)
        write_new(directory / "ERROR.json", {
            "status": "ERROR_NO_RETRY", "provider": profile()["provider"],
            "requested_model": MODEL, "error_type": type(error).__name__,
            "metadata": metadata, "response_id": (metadata or {}).get("response_id"),
            "usage": (metadata or {}).get("usage"),
            "source_sha256": entry["source_sha256"], "prompt_sha256": request["prompt_sha256"],
            "payload_sha256": request["payload_sha256"],
        })
        print(json.dumps({"root_id": root_id, "stage": stage, "status": "ERROR_NO_RETRY", "error_type": type(error).__name__}))
        return
    metadata = response.metadata
    raw_response = metadata.get("raw_response") or {}
    write_new(directory / "RESPONSE.json", {
        "status": "RESPONSE_RECEIVED_CONTENT_REVIEW_PENDING", "provider": profile()["provider"],
        "requested_model": MODEL, "returned_model": metadata.get("returned_model"),
        "response_id": raw_response.get("id"), "usage": metadata.get("usage"),
        "source_sha256": entry["source_sha256"], "prompt_sha256": request["prompt_sha256"],
        "payload_sha256": request["payload_sha256"], "raw_text": response.raw_text,
        "metadata": metadata, "mechanical_validation_run": False,
        "solver_evaluated": False, "selected_for_training": False,
    })
    print(json.dumps({"root_id": root_id, "stage": stage, "status": "RESPONSE_SAVED_VALIDATION_NOT_RUN",
                      "provider": profile()["provider"], "requested_model": MODEL,
                      "returned_model": metadata.get("returned_model"), "response_id": raw_response.get("id"),
                      "usage": metadata.get("usage")}, ensure_ascii=False))


def check_candidate(row: object, root_id: str, turns: list[dict], allowed_ids: set[str]) -> list[str]:
    if not isinstance(row, dict):
        return ["candidate_not_object"]
    errors = []
    if row.get("candidate_id") not in allowed_ids:
        errors.append("candidate_id_invalid")
    for field in ("topic", "question", "order_reason"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(field + "_missing")
    if row.get("root_id") != root_id or row.get("question_type") != "ordering":
        errors.append("root_or_type_mismatch")
    kind = row.get("chronology_kind")
    if kind not in {"conversation_order", "within_scene_actions"}:
        errors.append("chronology_kind_invalid")
    if kind == "conversation_order" and "APPEAR IN THE SUPPLIED DIALOGUE" not in row.get("question", ""):
        errors.append("conversation_order_question_scope_missing")
    items = row.get("items")
    if not isinstance(items, dict) or set(items) != LABELS or any(not isinstance(value, str) or not value.strip() for value in items.values()):
        errors.append("six_items_required")
    gold = row.get("gold")
    if not isinstance(gold, list) or len(gold) != 6 or any(not isinstance(value, str) for value in gold) or set(gold) != LABELS:
        errors.append("gold_not_complete_permutation")
    evidence = row.get("item_evidence")
    if not isinstance(evidence, dict) or set(evidence) != LABELS:
        return errors + ["six_evidence_entries_required"]
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    positions = {turn["turn_ref"]: index for index, turn in enumerate(turns)}
    refs = {}
    for label in sorted(LABELS):
        value = evidence[label]
        if not isinstance(value, dict):
            errors.append(label + ":evidence_not_object")
            continue
        ref, quote = value.get("turn_ref"), value.get("quote")
        if not isinstance(ref, str) or ref not in by_ref:
            errors.append(label + ":invalid_turn_ref")
            continue
        refs[label] = ref
        if not isinstance(quote, str) or not quote.strip() or quote not in by_ref[ref]["text"]:
            errors.append(label + ":quote_not_verbatim")
        if value.get("status") not in STATUSES or not isinstance(value.get("reason"), str) or not value["reason"].strip():
            errors.append(label + ":invalid_status_or_reason")
    if len(refs) == 6:
        if len(set(refs.values())) != 6:
            errors.append("duplicate_anchor_turns")
        if not any(ref.startswith("R01:") for ref in refs.values()):
            errors.append("r01_anchor_required")
        if (kind == "conversation_order" and "gold_not_complete_permutation" not in errors and
                [positions[refs[label]] for label in gold] != sorted(positions[ref] for ref in refs.values())):
            errors.append("gold_conflicts_with_anchor_order")
        if kind == "within_scene_actions":
            scenes = {turn["scene_ref"] for turn in (by_ref[ref] for ref in refs.values()) if isinstance(turn.get("scene_ref"), str)}
            if len(scenes) != 1 or len(scenes) == 0:
                errors.append("action_sequence_spans_scenes")
    return errors


def validate(root_id: str, stage: str) -> dict:
    entry, source, request, directory = frozen_request(root_id, stage)
    response = read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != entry["source_sha256"] or
            response.get("prompt_sha256") != request["prompt_sha256"] or
            response.get("payload_sha256") != request["payload_sha256"]):
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata", {})
    try:
        parsed = json.loads(response["raw_text"])
    except (TypeError, ValueError):
        parsed = None
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    abstentions = parsed.get("abstentions") if isinstance(parsed, dict) else None
    global_errors = []
    if not isinstance(questions, list):
        global_errors.append("questions_array_missing")
        questions = []
    if len(questions) > request["count"]:
        global_errors.append("candidate_count_exceeds_request")
    if not isinstance(abstentions, list) or any(not isinstance(value, str) for value in abstentions):
        global_errors.append("abstentions_array_invalid")
    if metadata.get("returned_model") != MODEL:
        global_errors.append("returned_model_mismatch")
    checks, seen_ids, seen_event_sets, valid = [], set(), set(), []
    allowed_ids = set(request["candidate_ids"])
    for index, question in enumerate(questions, 1):
        errors = check_candidate(question, root_id, source["turns"], allowed_ids)
        candidate_id = question.get("candidate_id") if isinstance(question, dict) else None
        if candidate_id in seen_ids:
            errors.append("duplicate_candidate_id")
        seen_ids.add(candidate_id)
        evidence = question.get("item_evidence") if isinstance(question, dict) else None
        if isinstance(evidence, dict):
            event_set = tuple(sorted(value.get("turn_ref") for value in evidence.values() if isinstance(value, dict) and isinstance(value.get("turn_ref"), str)))
            if len(event_set) == 6 and event_set in seen_event_sets:
                errors.append("duplicate_event_set")
            other_stage = "topup" if stage == "initial" else "initial"
            other_path = HERE / root_id / other_stage / "VALIDATED_CANDIDATES.json"
            if len(event_set) == 6 and other_path.exists():
                other = read_json(other_path)
                for prior in other.get("questions", []):
                    prior_evidence = prior.get("item_evidence", {}) if isinstance(prior, dict) else {}
                    prior_set = tuple(sorted(value.get("turn_ref") for value in prior_evidence.values()
                                             if isinstance(value, dict) and isinstance(value.get("turn_ref"), str)))
                    if len(prior_set) == 6 and prior_set == event_set:
                        errors.append("duplicate_event_set_across_stages")
                        break
            seen_event_sets.add(event_set)
        checks.append({"index": index, "candidate_id": candidate_id, "errors": errors})
        if not errors:
            valid.append(question)
    if global_errors:
        valid = []
    provenance = {
        "author": "REMOTE_LARGE_MODEL", "provider": profile()["provider"],
        "requested_model": MODEL, "returned_model": metadata.get("returned_model"),
        "response_id": response.get("response_id"), "usage": response.get("usage"),
        "root_id": root_id, "source_sha256": entry["source_sha256"],
        "prompt_sha256": request["prompt_sha256"], "payload_sha256": request["payload_sha256"],
        "solver_evaluated": False, "selected_for_training": False, "training_started": False,
    }
    result = {
        "status": "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if not global_errors else "MECHANICAL_REJECTED",
        "root_id": root_id, "stage": stage, "candidate_count": len(questions),
        "mechanically_valid_count": len(valid), "global_errors": global_errors, "checks": checks,
        "semantic_gold_verified": False, "provenance": provenance,
    }
    write_new(directory / "PARSED_RESPONSE.json", parsed)
    write_new(directory / "VALIDATED_CANDIDATES.json", {
        "status": result["status"], "provenance": provenance, "questions": valid,
    })
    write_new(directory / "VALIDATION.json", result)
    return result


def summary() -> dict:
    batch, _ = verify_batch()
    roots = []
    for entry in batch["roots"]:
        root_id = entry["root_id"]
        stages = {}
        for stage in ("initial", "topup"):
            path = HERE / root_id / stage / "VALIDATION.json"
            if path.exists():
                stages[stage] = read_json(path)
        roots.append({"root_id": root_id, "stages": stages})
    result = {
        "version": VERSION, "requested_model": MODEL, "provider": profile()["provider"],
        "root_count": len(roots), "roots": roots, "solver_calls": 0, "training_started": False,
        "status": "GENERATION_AND_MECHANICAL_VALIDATION_ONLY_CONTENT_REVIEW_PENDING",
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--prepare-top-up", metavar="ROOT_ID")
    modes.add_argument("--execute", metavar="ROOT_ID")
    modes.add_argument("--execute-top-up", metavar="ROOT_ID")
    modes.add_argument("--validate", metavar="ROOT_ID")
    modes.add_argument("--validate-top-up", metavar="ROOT_ID")
    modes.add_argument("--summary", action="store_true")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.manifest.resolve())
    elif args.prepare_top_up:
        prepare_topup(args.prepare_top_up, args.reason)
    elif args.execute:
        execute(args.execute, "initial")
    elif args.execute_top_up:
        execute(args.execute_top_up, "topup")
    elif args.validate:
        print(json.dumps(validate(args.validate, "initial"), ensure_ascii=False))
    elif args.validate_top_up:
        print(json.dumps(validate(args.validate_top_up, "topup"), ensure_ascii=False))
    else:
        summary()
