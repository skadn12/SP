"""Offline plan builder for a 30-root H4 integration probe.

The plan is intentionally a boundary object between the cleaned input batch,
the single-root Memory-event continuation runner, and the isolated H4 K=3
multi-type QA runner.  It does not import either live runner and therefore
cannot make Challenger, Memorax, or local Solver calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, canonical_json
from ..batch_manifest_v1.build import SELECTION_MANIFEST_VERSION
from ..simple_append_v1.assembly import validate_appended_round
from ..unified_dialogue_root import (
    validate_unified_dialogue_root,
    validate_unified_root_collection,
)


BATCH_PLAN_VERSION = "h4_batch30_memory_event_k3_plan_v1"
EXPANSION_SOURCE_SELECTOR_VERSION = "h4_expansion_source_selector_v1"
CLEAN_MANIFEST_VERSION = "named_input_cleaning_batch_manifest_v1"
EXPANSION_MODULE = (
    "scripts.current_method.restart_n5_spell."
    "simple_append_memorax_v1_memory_event_continuation_probe_v1.live"
)
EXPANSION_REFERENCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_91716_named_speakers_event_bucket_only_"
    "expansion_v1_live_1"
)
QA_MODULE = (
    "scripts.current_method.restart_n5_spell."
    "h4_qa_multiroot_k3_t08_probe_v1.run"
)
DEFAULT_CLEANED_BATCH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/"
    "named_batch_v1_new30_h4_pilot_v2"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_batch30_memory_event_k3_probe_v1_offline_1"
)
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged"
)
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
DEFAULT_SHARD_SIZE = 10
DEFAULT_EXPECTED_COUNT = 30
DEFAULT_NAMESPACE_PREFIX = "gxy-selfplay:h4-batch30-memory-event-k3-v1"
EXPANSION_EVENT_BUCKET_ONLY = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value)).strip("-.")
    if not text:
        raise ValueError("empty_safe_id")
    return text


def _as_path(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"json_load_failed:{path}:{type(error).__name__}") from error


def _validate_manifest_shape(value: Mapping[str, Any], manifest_path: Path) -> None:
    manifest_version = value.get("manifest_version")
    if manifest_version not in {
        CLEAN_MANIFEST_VERSION,
        SELECTION_MANIFEST_VERSION,
    }:
        raise ValueError(
            f"clean_manifest_version_invalid:{manifest_version}"
        )
    if value.get("status") != "passed":
        raise ValueError(f"clean_manifest_status_invalid:{value.get('status')}")
    expected = value.get("expected_count")
    count = value.get("record_count")
    if not isinstance(expected, int) or not isinstance(count, int) or expected != count:
        raise ValueError("clean_manifest_count_inconsistent")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("clean_manifest_records_inconsistent")
    roots_dir = value.get("roots_directory")
    if not roots_dir:
        raise ValueError("clean_manifest_roots_directory_missing")
    global_checks = value.get("global_checks") or {}
    if manifest_version == SELECTION_MANIFEST_VERSION:
        required_checks = {
            "all_records_valid", "root_ids_unique", "root_sha256_match",
            "root_identity_match", "no_data_copied",
        }
    else:
        required_checks = {
            "all_records_valid", "all_repeatable", "expected_count",
            "mapping_coverage_exact", "names_unique_within_each_root",
            "root_ids_unique", "source_unchanged",
        }
    missing = sorted(key for key in required_checks if global_checks.get(key) is not True)
    if missing:
        raise ValueError("clean_manifest_global_checks_failed:" + ",".join(missing))


def _root_from_record(record: Mapping[str, Any], *, manifest_dir: Path) -> tuple[Path, dict[str, Any]]:
    root_id = str(record.get("root_id") or "")
    if not root_id:
        raise ValueError("clean_manifest_root_id_missing")
    # Cleaning manifests call this field output_path.  Reference-only batch
    # manifests use input_path while pointing at the same immutable root.
    raw_path = record.get("output_path") or record.get("input_path")
    if not raw_path:
        raise ValueError(f"clean_manifest_output_path_missing:{root_id}")
    path = _as_path(raw_path, base=manifest_dir)
    if not path.is_file():
        raise FileNotFoundError(f"cleaned_root_missing:{root_id}:{path}")
    actual_sha = sha256_file(path)
    expected_sha = str(record.get("output_sha256") or "")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"cleaned_root_hash_drift:{root_id}")
    root = _load_json(path)
    if not isinstance(root, dict):
        raise ValueError(f"cleaned_root_not_object:{root_id}")
    if str(root.get("root_id") or "") != root_id:
        raise ValueError(f"cleaned_root_identity_drift:{root_id}")
    validation = validate_unified_dialogue_root(root)
    if not validation["valid"]:
        raise ValueError(
            f"cleaned_root_invalid:{root_id}:{','.join(validation['errors'])}"
        )
    row = {
        "root_id": root_id,
        "input_path": str(path),
        "input_sha256": actual_sha,
        "parent_dialogue_sha256": str(
            (root.get("dialogue") or {}).get("dialogue_hash") or ""),
        "turn_count": int((root.get("dialogue") or {}).get("turn_count") or 0),
        "session_count": int((root.get("dialogue") or {}).get("session_count") or 0),
        "speaker_names": sorted({
            str(turn.get("speaker") or "")
            for turn in (root.get("dialogue") or {}).get("turns") or []
        }),
        "cleaning_report_path": str(path.with_name("name_cleaning_report.json")),
        "source_record_sha256": record.get("source_record_sha256"),
        "source_row_index": record.get("source_row_index"),
        "validation": validation,
    }
    return path, row


def load_cleaned_batch(
    cleaned_batch: Path,
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
) -> dict[str, Any]:
    """Load and fully validate the cleaned-root batch without writing files."""

    batch = cleaned_batch.resolve()
    manifest_path = batch / "manifest.json" if batch.is_dir() else batch
    if not manifest_path.is_file():
        raise FileNotFoundError(f"clean_manifest_missing:{manifest_path}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("clean_manifest_not_object")
    _validate_manifest_shape(manifest, manifest_path)
    if int(manifest["record_count"]) != int(expected_count):
        raise ValueError(
            f"clean_manifest_expected_count:{manifest['record_count']}:{expected_count}"
        )
    rows: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for record in manifest["records"]:
        if not isinstance(record, Mapping):
            raise ValueError("clean_manifest_record_not_object")
        _, row = _root_from_record(record, manifest_dir=manifest_path.parent)
        rows.append(row)
        roots.append(_load_json(Path(row["input_path"])))
    collection = validate_unified_root_collection(
        roots,
        expected_count=int(expected_count),
    )
    if not collection["valid"]:
        failed = [
            key for key, passed in (collection.get("global_checks") or {}).items()
            if not passed
        ]
        raise ValueError("clean_root_collection_invalid:" + ",".join(failed))
    ids = [str(row["root_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("clean_root_ids_not_unique")
    cleaned_jsonl = manifest.get("cleaned_jsonl")
    cleaned_jsonl_sha = manifest.get("cleaned_jsonl_sha256")
    cleaned_jsonl_path = _as_path(cleaned_jsonl, base=manifest_path.parent) if cleaned_jsonl else None
    if cleaned_jsonl_path is not None:
        if not cleaned_jsonl_path.is_file():
            raise FileNotFoundError(f"cleaned_jsonl_missing:{cleaned_jsonl_path}")
        if cleaned_jsonl_sha and sha256_file(cleaned_jsonl_path) != cleaned_jsonl_sha:
            raise ValueError("cleaned_jsonl_hash_drift")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "rows": rows,
        "roots": roots,
        "collection_validation": collection,
        "cleaned_jsonl_path": str(cleaned_jsonl_path) if cleaned_jsonl_path else None,
        "cleaned_jsonl_sha256": sha256_file(cleaned_jsonl_path) if cleaned_jsonl_path else None,
    }


def _selector_rejection(
    *, expansion_root: Path, expected_root_id: str, reasons: Sequence[str],
    checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Return a read-only quarantine decision for an expansion artifact."""

    return {
        "selector_version": EXPANSION_SOURCE_SELECTOR_VERSION,
        "eligible": False,
        "status": "quarantined",
        "root_id": expected_root_id,
        "expansion_root": str(expansion_root),
        "selected_path": None,
        "selected_sha256": None,
        "complete_dialogue_sha256": None,
        "source_kind": None,
        "validation_path": None,
        "reason_codes": list(reasons),
        "checks": dict(checks or {}),
    }


def _read_selector_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(value, dict):
        return None, "not_object"
    return value, None


def _expanded_dialogue_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("complete_dialogue_sha256", None)
    return _sha256_json(body)


def _multi_round_source_checks(
    expanded: Mapping[str, Any], validation: Mapping[str, Any],
    freeze: Mapping[str, Any], *, expected_root: str, expected_input_hash: str,
) -> dict[str, bool]:
    """Bind an appended round to its frozen prior transcript, read-only.

    Multi-round artifacts retain the original Parent count/hash, whereas
    their validation describes the cumulative prior and the latest addition.
    Check that boundary directly instead of treating it as a first append.
    """
    prior_path = Path(str(freeze.get("input_path") or ""))
    try:
        pinned = prior_path.is_file() and sha256_file(prior_path) == expected_input_hash
    except OSError:
        pinned = False
    checks = {"selected_prior_file_hash_matches": pinned}
    if not pinned:
        return checks
    prior, prior_error = _read_selector_json(prior_path)
    checks["selected_prior_object_valid"] = prior_error is None
    if prior is None:
        return checks
    prior_hash = str(prior.get("complete_dialogue_sha256") or "").lower()
    prior_turns = prior.get("turns")
    turns = expanded.get("turns")
    usable_turns = (
        isinstance(prior_turns, list) and bool(prior_turns)
        and all(isinstance(row, Mapping) for row in prior_turns)
        and isinstance(turns, list) and all(isinstance(row, Mapping) for row in turns)
    )
    checks.update({
        "selected_prior_schema_valid": prior.get("expanded_dialogue_version") in {
            "simple_append_expanded_dialogue_v1",
            "simple_append_expanded_dialogue_multi_round_v1",
        },
        "selected_prior_root_matches": prior.get("root_id") == expected_root,
        "selected_prior_complete_hash_valid": (
            bool(re.fullmatch(r"[0-9a-f]{64}", prior_hash))
            and prior_hash == _expanded_dialogue_hash(prior)),
        "selected_prior_hash_chain_matches": (
            str(expanded.get("prior_complete_dialogue_sha256") or "").lower()
            == prior_hash
            == str(freeze.get("parent_dialogue_sha256") or "").lower()),
        "selected_original_parent_hash_matches": (
            bool(prior.get("parent_dialogue_sha256"))
            and expanded.get("parent_dialogue_sha256") == prior.get("parent_dialogue_sha256")),
        "selected_original_parent_count_matches": (
            expanded.get("parent_turn_count") == prior.get("parent_turn_count")
            and expanded.get("original_parent_turn_count") == prior.get("parent_turn_count")),
        "selected_prior_turns_valid": usable_turns,
    })
    if not usable_turns:
        return checks
    actual = validate_appended_round(expanded, prior_expanded=prior)
    checks.update({
        "selected_appended_round_valid": actual["valid"],
        "selected_validation_prior_count_matches": (
            validation.get("prior_turn_count") == actual["prior_turn_count"]
            == prior.get("turn_count")),
        "selected_validation_new_count_matches": (
            validation.get("new_turn_count") == actual["new_turn_count"]
            == expanded.get("latest_round_child_turn_count")),
    })
    return checks


def select_successful_expansion_source(
    expansion_root: Path,
    *,
    expected_root_id: str,
    expected_input_sha256: str,
    expected_parent_dialogue_sha256: str | None = None,
) -> dict[str, Any]:
    """Select one QA source from a completed expansion without writing files.

    The raw dialogue is produced before the live workflow reaches its final
    promotion gate.  Its mere presence, and even a locally valid structural
    validation file, therefore cannot authorize QA.  This selector first
    binds the run report and freeze manifest to the expected cleaned input,
    then prefers the promoted dialogue.  Raw fallback is allowed only for a
    completed, generation-ready run whose promotion artifact is absent.
    """

    root = expansion_root.resolve()
    expected_root = str(expected_root_id)
    expected_input_hash = str(expected_input_sha256).lower()
    expected_parent_hash = str(expected_parent_dialogue_sha256 or "").lower()
    if (not expected_root
            or not re.fullmatch(r"[0-9a-f]{64}", expected_input_hash)):
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=["expected_root_or_input_hash_invalid"])
    report_path = root / "reports" / "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
    freeze_path = root / "run_freeze_manifest.json"
    raw_path = root / "expanded_dialogue.json"
    raw_validation_path = root / "expanded_dialogue_validation.json"
    promotion_path = root / "promotion" / "promotable_expanded_dialogue.json"
    promotion_validation_path = (
        root / "promotion" / "promotable_expanded_dialogue_validation.json")
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    report, report_error = _read_selector_json(report_path)
    if report_error:
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=[f"report_{report_error}"], checks=checks)
    assert report is not None
    checks["report_status_completed"] = report.get("status") == "completed"
    checks["report_generation_ready"] = report.get("generation_ready") is True
    checks["report_root_matches"] = str(report.get("root_id") or "") == expected_root
    checks["report_input_hash_matches"] = (
        str(report.get("input_sha256") or "").lower() == expected_input_hash)
    try:
        reported_root = Path(str(report.get("run_root") or "")).resolve()
    except (OSError, RuntimeError, ValueError):
        reported_root = Path("/")
    checks["report_run_root_matches"] = reported_root == root
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    if reasons:
        # In particular, a failed report can never fall back to a raw file.
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=reasons, checks=checks)

    freeze, freeze_error = _read_selector_json(freeze_path)
    if freeze_error:
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=[f"freeze_manifest_{freeze_error}"], checks=checks)
    assert freeze is not None
    freeze_checks = {
        "freeze_decision_pass": freeze.get("decision") == "pass",
        "freeze_root_matches": str(freeze.get("root_id") or "") == expected_root,
        "freeze_input_hash_matches": (
            str(freeze.get("input_sha256") or "").lower() == expected_input_hash),
    }
    if expected_parent_hash:
        freeze_checks["freeze_parent_hash_matches"] = (
            str(freeze.get("parent_dialogue_sha256") or "").lower()
            == expected_parent_hash)
    checks.update(freeze_checks)
    reasons = [name for name, passed in freeze_checks.items() if not passed]
    if reasons:
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=reasons, checks=checks)

    # An existing but malformed promotion artifact is authoritative and must
    # quarantine the root; it must not be treated as an absent file so raw can
    # silently win.
    promotion_present = promotion_path.exists()
    checks["promotion_present"] = promotion_present
    if promotion_present:
        selected_path = promotion_path
        validation_path = promotion_validation_path
        source_kind = "promotion"
    else:
        selected_path = raw_path
        validation_path = raw_validation_path
        source_kind = "raw_success_fallback"

    expanded, expanded_error = _read_selector_json(selected_path)
    if expanded_error:
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=[f"selected_source_{expanded_error}"], checks=checks)
    validation, validation_error = _read_selector_json(validation_path)
    if validation_error:
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=[f"selected_validation_{validation_error}"], checks=checks)
    assert expanded is not None and validation is not None

    turns = expanded.get("turns")
    turn_count = expanded.get("turn_count")
    parent_count = expanded.get("parent_turn_count")
    child_count = expanded.get("child_turn_count")
    frozen_dialogue_hash = str(
        expanded.get("complete_dialogue_sha256") or "").lower()
    recomputed_dialogue_hash = _expanded_dialogue_hash(expanded)
    turn_refs = [
        str(row.get("turn_ref") or "") if isinstance(row, Mapping) else ""
        for row in turns
    ] if isinstance(turns, list) else []
    candidate_checks = {
        "selected_schema_version_valid": expanded.get(
            "expanded_dialogue_version") in {
                "simple_append_expanded_dialogue_v1",
                "simple_append_expanded_dialogue_multi_round_v1",
            },
        "selected_root_matches": str(expanded.get("root_id") or "") == expected_root,
        "selected_turns_nonempty": isinstance(turns, list) and bool(turns),
        "selected_turn_refs_valid": (
            isinstance(turns, list) and bool(turn_refs)
            and all(turn_refs) and len(turn_refs) == len(set(turn_refs))),
        "selected_turn_count_matches": (
            isinstance(turns, list) and isinstance(turn_count, int)
            and turn_count == len(turns)),
        "selected_parent_child_counts_match": (
            isinstance(parent_count, int) and isinstance(child_count, int)
            and isinstance(turn_count, int) and parent_count > 0
            and child_count > 0 and parent_count + child_count == turn_count),
        "selected_complete_hash_matches": (
            bool(re.fullmatch(r"[0-9a-f]{64}", frozen_dialogue_hash))
            and frozen_dialogue_hash == recomputed_dialogue_hash),
        "selected_validation_valid": validation.get("valid") is True,
        "selected_validation_errors_empty": not validation.get("errors"),
        "selected_validation_turn_count_matches": (
            isinstance(turn_count, int)
            and validation.get("turn_count") == turn_count),
    }
    if expanded.get("expanded_dialogue_version") == "simple_append_expanded_dialogue_multi_round_v1":
        candidate_checks.update(_multi_round_source_checks(
            expanded, validation, freeze, expected_root=expected_root,
            expected_input_hash=expected_input_hash,
        ))
    else:
        candidate_checks.update({
            "selected_validation_parent_count_matches": (
                isinstance(parent_count, int)
                and validation.get("parent_turn_count") == parent_count),
            "selected_validation_child_count_matches": (
                isinstance(child_count, int)
                and validation.get("child_turn_count") == child_count),
        })
        if expected_parent_hash:
            candidate_checks["selected_parent_hash_matches"] = (
                str(expanded.get("parent_dialogue_sha256") or "").lower()
                == expected_parent_hash)
    if source_kind == "promotion":
        promotion_validation = report.get("promotion_validation")
        candidate_checks["report_promotion_validation_valid"] = (
            isinstance(promotion_validation, Mapping)
            and promotion_validation.get("valid") is True
            and not promotion_validation.get("errors"))
        embedded = report.get("promotable_expanded_dialogue")
        candidate_checks["report_promotion_binding"] = (
            isinstance(embedded, Mapping)
            and str(embedded.get("root_id") or "") == expected_root
            and str(embedded.get("complete_dialogue_sha256") or "").lower()
            == frozen_dialogue_hash)
    checks.update(candidate_checks)
    reasons = [name for name, passed in candidate_checks.items() if not passed]
    if reasons:
        # Do not fall back to raw if an authoritative promotion exists but
        # fails identity, validation, or hash checks.
        return _selector_rejection(
            expansion_root=root, expected_root_id=expected_root,
            reasons=reasons, checks=checks)

    return {
        "selector_version": EXPANSION_SOURCE_SELECTOR_VERSION,
        "eligible": True,
        "status": "selected",
        "root_id": expected_root,
        "expansion_root": str(root),
        "report_path": str(report_path),
        "freeze_manifest_path": str(freeze_path),
        "selected_path": str(selected_path),
        "selected_sha256": sha256_file(selected_path),
        "complete_dialogue_sha256": frozen_dialogue_hash,
        "source_kind": source_kind,
        "validation_path": str(validation_path),
        "reason_codes": [],
        "checks": checks,
    }


def _namespace(prefix: str, root_id: str, *, attempt_index: int = 1) -> str:
    """Build a fresh Memorax namespace for exactly one root attempt.

    A shard is only a scheduling unit.  Sharing its namespace would make a
    failed or ambiguous root capable of contaminating a later root, so the
    namespace identity includes the root id.
    """
    # ``live.py`` appends the root id internally when it builds Memorax's
    # ``user_id``.  Use a stable per-root attempt token here instead of
    # embedding the root id twice in that final value.
    if int(attempt_index) <= 0:
        raise ValueError("attempt_index_must_be_positive")
    attempt = hashlib.sha256(
        f"{root_id}:attempt:{int(attempt_index)}".encode("utf-8")
    ).hexdigest()[:12]
    value = f"{prefix}:attempt-{attempt}"
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError("memorax_namespace_prefix_invalid")
    return value


def _expansion_command(
    *, row: Mapping[str, Any], expansion_root: Path, namespace: str,
    config: Path, tokenizer: Path, credentials: Path,
) -> list[str]:
    return [
        "python", "-m", EXPANSION_MODULE,
        "--live",
        "--input", str(row["input_path"]),
        "--run-root", str(expansion_root),
        "--config", str(config),
        "--tokenizer", str(tokenizer),
        "--memorax-credentials", str(credentials),
        "--memorax-user-namespace", namespace,
        "--allow-insecure-memorax-http",
        "--memory-only-fast",
        "--event-bucket-only",
        "--round-index", "1",
    ]


def _qa_inventory_command_template(*, qa_root: Path) -> list[str | int]:
    """Return a non-executable inventory template populated by the selector.

    A preflight plan is written before any expansion exists, so embedding raw
    paths here would make it too easy for an executor to bypass the success
    gate.  The executor must repeat ``{selected_path}`` for each eligible
    root, then replace ``{selected_root_count}`` with the resulting count.
    """

    return [
        "python", "-m", QA_MODULE, "--phase", "inventory",
        "--run-root", str(qa_root),
        "--root-source", "{selected_path}",
        "--max-roots", "{selected_root_count}",
    ]


def _qa_smoke_command(*, qa_root: Path, config: Path, tokenizer: Path,
                      solver_model: Path) -> list[str]:
    return [
        "python", "-m", QA_MODULE, "--phase", "smoke", "--live",
        "--run-root", str(qa_root), "--api-config", str(config),
        "--tokenizer", str(tokenizer),
        "--solver-model", str(solver_model),
    ]


def _qa_batch_command(*, qa_root: Path, config: Path, tokenizer: Path,
                      solver_model: Path) -> list[str]:
    return [
        "python", "-m", QA_MODULE, "--phase", "batch", "--live",
        "--run-root", str(qa_root), "--api-config", str(config),
        "--tokenizer", str(tokenizer),
        "--solver-model", str(solver_model),
    ]


def build_batch_plan(
    *,
    cleaned_batch: Path = DEFAULT_CLEANED_BATCH,
    run_root: Path = DEFAULT_RUN_ROOT,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    config: Path = DEFAULT_CONFIG,
    tokenizer: Path = DEFAULT_TOKENIZER,
    solver_model: Path = DEFAULT_SOLVER_MODEL,
    credentials: Path = DEFAULT_CREDENTIALS,
    namespace_prefix: str = DEFAULT_NAMESPACE_PREFIX,
) -> dict[str, Any]:
    """Build a deterministic, API-free execution plan."""

    if expected_count <= 0 or shard_size <= 0:
        raise ValueError("expected_count_and_shard_size_must_be_positive")
    if expected_count % shard_size:
        raise ValueError("expected_count_must_divide_evenly_into_shards")
    loaded = load_cleaned_batch(cleaned_batch, expected_count=expected_count)
    output = run_root.resolve()
    rows = loaded["rows"]
    missing_optional: dict[str, bool] = {
        "challenger_config": config.is_file(),
        "tokenizer": tokenizer.exists(),
        "solver_model": solver_model.exists(),
        "memorax_credentials": credentials.is_file(),
    }
    # The plan is still useful for a dry run when runtime assets are absent,
    # but it records the absence explicitly and never calls those assets.
    roots: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    shard_count = (len(rows) + shard_size - 1) // shard_size
    for shard_index in range(shard_count):
        shard_rows = rows[shard_index * shard_size:(shard_index + 1) * shard_size]
        shard_name = f"shard-{shard_index:02d}"
        shard_root = output / "shards" / shard_name
        expansion_root = shard_root / "expansion"
        qa_root = shard_root / "qa"
        shard_namespace = f"{namespace_prefix}:shard-{shard_index:02d}"
        # The frozen R1 runner always writes this raw, validated artifact.
        # Some recovery paths additionally emit a promotion JSON; the
        # executor may prefer that file only after recording its hash, but it
        # must not make the normal batch depend on an optional artifact.
        expanded_paths = [
            expansion_root / "roots" / _safe_id(str(row["root_id"])) /
            "expanded_dialogue.json"
            for row in shard_rows
        ]
        shard_root_rows: list[dict[str, Any]] = []
        for row, expanded_path in zip(shard_rows, expanded_paths):
            root_id = str(row["root_id"])
            root_expansion_root = expansion_root / "roots" / _safe_id(root_id)
            root_qa_root = qa_root / "roots" / _safe_id(root_id)
            root_namespace = _namespace(namespace_prefix, root_id)
            raw_expanded_path = root_expansion_root / "expanded_dialogue.json"
            promotion_path = root_expansion_root / "promotion" / "promotable_expanded_dialogue.json"
            root_plan = {
                **row,
                "shard": shard_name,
                "expansion": {
                    "run_root": str(root_expansion_root),
                    "input_path": str(row["input_path"]),
                    "expected_output": str(raw_expanded_path),
                    "preferred_output": str(promotion_path),
                    "expected_report": str(root_expansion_root / "reports" / "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"),
                    "command": _expansion_command(
                        row=row, expansion_root=root_expansion_root,
                        namespace=root_namespace, config=config.resolve(),
                        tokenizer=tokenizer.resolve(), credentials=credentials.resolve(),
                    ),
                    "route": {
                        "runner": EXPANSION_MODULE,
                        "memory_only_fast": True,
                        "event_bucket_only": EXPANSION_EVENT_BUCKET_ONLY,
                        "attempt_index": 1,
                        "round_index": 1,
                        "reuse_existing_namespace": False,
                        "namespace": root_namespace,
                        "user_id": f"{root_namespace}:{root_id}",
                        "namespace_scope": "one_root_attempt",
                    },
                },
                "qa": {
                    "run_root": str(root_qa_root),
                    # These are artifact locations only.  They are never a
                    # static authorization to send a file to the QA runner.
                    "input_expected": str(expanded_path),
                    "input_preferred": str(promotion_path),
                    "input_fallback": str(raw_expanded_path),
                    "input_expected_role": "unverified_artifact_location_only",
                    "source_selector": "select_successful_expansion_source",
                    "selected_input": None,
                    "input_selection_policy": (
                        "select only after the expansion report has status "
                        "completed, generation_ready=true, and validation/root "
                        "checks pass; prefer promotion when valid, otherwise "
                        "use the validated raw output; failed roots are "
                        "quarantined and never fall back into QA"
                    ),
                    "question_types": ["single_choice", "multi_select", "ordering"],
                    "rollouts_per_variant": 3,
                    "command_dependency": (
                        "run only after select_successful_expansion_source "
                        "returns eligible=true; quarantined roots are excluded"),
                },
            }
            shard_root_rows.append(root_plan)
            roots.append(root_plan)
        shard_plan = {
            "shard": shard_name,
            "shard_index": shard_index,
            "root_count": len(shard_root_rows),
            "root_ids": [str(row["root_id"]) for row in shard_root_rows],
                    "run_root": str(shard_root),
            "expansion_root": str(expansion_root),
            "qa_root": str(qa_root),
            "memorax_user_namespace": shard_namespace,
            "memorax_namespace_scope": "per_root_attempt",
            "roots": shard_root_rows,
            "commands": {
                "expansion": "execute each root command sequentially; checkpoint after each root",
                "source_selection": {
                    "selector": "select_successful_expansion_source",
                    "mode": "read_only_after_each_expansion",
                    "required_report_status": "completed",
                    "required_generation_ready": True,
                    "promotion_priority": True,
                    "failed_raw_fallback": False,
                    "eligible_root_ids": [],
                    "selected_sources": [],
                },
                # Deliberately no static root-source arguments: the template
                # becomes executable only after selector results are bound.
                "qa_inventory": None,
                "qa_inventory_command_template": _qa_inventory_command_template(
                    qa_root=qa_root),
                "qa_smoke": _qa_smoke_command(
                    qa_root=qa_root, config=config.resolve(),
                    tokenizer=tokenizer.resolve(),
                    solver_model=solver_model.resolve()),
                "qa_batch": _qa_batch_command(
                    qa_root=qa_root, config=config.resolve(),
                    tokenizer=tokenizer.resolve(),
                    solver_model=solver_model.resolve()),
            },
            "checkpoint_stages": [
                "ROOT_STARTED", "EXPANSION_READY", "QA_INVENTORY_READY",
                "QA_SMOKE_PASSED", "QA_BATCH_COMPLETED",
            ],
            "resume_policy": {
                "skip_only_if_success_artifacts_and_hashes_match": True,
                "never_overwrite_root_attempt": True,
                "namespace_reuse": False,
                "ambiguous_memorax_request": "quarantine_and_stop_root",
            },
        }
        shards.append(shard_plan)

    plan_payload: dict[str, Any] = {
        "plan_version": BATCH_PLAN_VERSION,
        "status": "offline_preflight_passed",
        "offline_only": True,
        "api_calls": 0,
        "solver_calls": 0,
        "memory_writes": 0,
        "run_root": str(output),
        "cleaned_batch": {
            "manifest_path": loaded["manifest_path"],
            "manifest_sha256": loaded["manifest_sha256"],
            "manifest_version": loaded["manifest"].get("manifest_version"),
            "record_count": len(rows),
            "cleaned_jsonl_path": loaded["cleaned_jsonl_path"],
            "cleaned_jsonl_sha256": loaded["cleaned_jsonl_sha256"],
            "collection_validation": loaded["collection_validation"],
        },
        "expected_count": expected_count,
        "shard_size": shard_size,
        "shard_count": len(shards),
        "root_ids": [str(row["root_id"]) for row in rows],
        "roots": roots,
        "shards": shards,
        "expansion_contract": {
            "module": EXPANSION_MODULE,
            "reference_success_run": str(EXPANSION_REFERENCE_RUN),
            "reference_prompt_contract": "event_bucket_only_planner_prompt_v2_bounded_scene_outcome",
            "input_contract": "one cleaned unified_root.json per invocation",
            "output_contract": (
                "run_root/expanded_dialogue.json (always emitted by the frozen "
                "R1 runner); promotion/promotable_expanded_dialogue.json is an "
                "optional preferred artifact when emitted; neither path is "
                "QA-eligible until select_successful_expansion_source passes"
            ),
            "source_selector": {
                "function": "select_successful_expansion_source",
                "requires": [
                    "report.status == completed",
                    "report.generation_ready == true",
                    "report/freeze root identity matches expected root",
                    "report/freeze input SHA-256 matches cleaned input",
                    "selected dialogue validation is valid",
                    "selected complete_dialogue_sha256 recomputes exactly",
                ],
                "promotion_priority": True,
                "raw_fallback_only_when": (
                    "promotion artifact is absent and the same successful "
                    "report/hash/validation gates pass"
                ),
                "failed_raw_fallback": False,
                "decision_on_failure": "quarantine_root_and_exclude_from_qa",
            },
            "parent_child_boundary": "expanded_dialogue.parent_turn_count + child_turn_count",
            "memorax_namespace_policy": "one fresh namespace per root attempt; shard namespace is metadata only",
            "fixed_flags": {
                "memory_only_fast": True,
                "event_bucket_only": EXPANSION_EVENT_BUCKET_ONLY,
                "round_index": 1,
                "reuse_existing_namespace": False,
            },
        },
        "qa_contract": {
            "module": QA_MODULE,
            "input_contract": (
                "one path returned by select_successful_expansion_source per "
                "--root-source; static raw paths are not accepted as proof"),
            "source_selector": "select_successful_expansion_source",
            "inventory_command_materialization": (
                "replace the deferred template only with eligible selector "
                "outputs; omit quarantined roots"),
            "candidate_source": "fresh_generated",
            "question_types": ["single_choice", "multi_select", "ordering"],
            "rollouts_per_variant": 3,
            "context_modes": ["no_context", "full_context"],
            "position_variants": {
                "single_choice": 4,
                "multi_select": 5,
                "ordering": 4,
            },
            "temperatures": {
                "single_choice": 0.7,
                "multi_select": 0.8,
                "ordering": 0.8,
            },
            "seed_inputs": ["root_id", "candidate_id", "slot_id"],
            "seed_excludes": ["context_mode", "display_position"],
            "semantic_grounding": "structural refs only; no semantic judgment in this plan",
        },
        "runtime_assets": {
            "challenger_config": str(config.resolve()),
            "tokenizer": str(tokenizer.resolve()),
            "solver_model": str(solver_model.resolve()),
            "memorax_credentials": str(credentials.resolve()),
            "existence_checks": missing_optional,
        },
        "source_integrity": {
            "source_manifest_sha256": loaded["manifest_sha256"],
            "root_input_sha256s": {str(row["root_id"]): row["input_sha256"] for row in rows},
            "plan_input_sha256": _sha256_json({
                "manifest": loaded["manifest_sha256"],
                "roots": [(str(row["root_id"]), row["input_sha256"]) for row in rows],
            }),
        },
        "mainline_untouched": True,
        "historical_runs_untouched": True,
    }
    return plan_payload


def write_batch_plan(plan: Mapping[str, Any]) -> Path:
    """Materialize only the new offline plan directory."""

    output = Path(str(plan["run_root"])).resolve()
    if output.exists():
        raise FileExistsError(f"plan_run_root_exists:{output}")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "batch_plan.json", dict(plan))
    atomic_json(output / "root_inventory.json", {
        "plan_version": plan["plan_version"],
        "root_count": plan["expected_count"],
        "root_ids": plan["root_ids"],
        "roots": plan["roots"],
        "api_calls": 0,
        "solver_calls": 0,
    })
    for shard in plan["shards"]:
        shard_root = Path(str(shard["run_root"]))
        shard_root.mkdir(parents=True, exist_ok=True)
        atomic_json(shard_root / "manifest.json", dict(shard))
        atomic_json(shard_root / "root_inventory.json", {
            "plan_version": plan["plan_version"],
            "shard": shard["shard"],
            "root_count": shard["root_count"],
            "root_ids": shard["root_ids"],
            "roots": shard["roots"],
            "api_calls": 0,
            "solver_calls": 0,
        })
    summary = [
        "# H4 Batch30 Memory-event + K=3 Offline Plan",
        "",
        "Status: `offline_preflight_passed`",
        "",
        f"- Cleaned roots: `{plan['expected_count']}`",
        f"- Shards: `{plan['shard_count']}` x `{plan['shard_size']}`",
        f"- Expansion route: `{EXPANSION_MODULE}`",
        f"- QA route: `{QA_MODULE}`",
        "- QA types: `single_choice`, `multi_select`, `ordering`",
        "- Solver: `K=3`, no/full context, cyclic display positions",
        "- API calls made by this planner: `0`",
        "- Solver calls made by this planner: `0`",
        "",
        "Expansion must complete and validate each root before the shard QA inventory is created.",
        "The commands in each shard manifest are a frozen execution contract; this package does not execute them.",
        "",
    ]
    atomic_text(output / "summary.md", "\n".join(summary))
    return output


__all__ = [
    "BATCH_PLAN_VERSION", "EXPANSION_MODULE", "QA_MODULE",
    "EXPANSION_SOURCE_SELECTOR_VERSION",
    "build_batch_plan", "load_cleaned_batch",
    "select_successful_expansion_source", "write_batch_plan",
]
