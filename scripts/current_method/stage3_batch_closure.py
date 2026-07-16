#!/usr/bin/env python3
"""Close one Stage3 generation batch with an independent cyclic-K5 verdict.

The generation/search run is immutable input.  This orchestrator optionally
hardens search-time easy/position-sensitive QA, softly repairs semantic-valid
hard0 QA while retaining the originals as diagnostics, combines accepted QA
repairs with *all* search-time robust-frontier rows, assigns stable namespaced
IDs, shuffles once, and launches one independent final cyclic K5 evaluation.

Every external boundary is checkpointed in an atomic state file.  ``--dry-run``
only validates inputs and publishes the exact command plan.  ``--execute`` may
call the challenger and the local solver through the existing scripts; this
module itself contains no API or model-loading code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import accumulate_stage64_reliable_frontier as accumulator  # noqa: E402
from scripts.current_method import stage3_solver_feedback_mining as feedback  # noqa: E402
from scripts.current_method import stage3_operation_controller as controller  # noqa: E402
from scripts.current_method import stage64_quality  # noqa: E402


SCHEMA_VERSION = "stage3_batch_closure_v1"
STATE_FILE = "closure_state.json"
MANIFEST_FILE = "closure_manifest.json"
FINAL_K = 5
FINAL_MAX_MODEL_LEN = 32768
FINAL_MAX_NEW_TOKENS = 80
FINAL_TEMPERATURE = 1.0
FINAL_TOP_P = 0.9
FINAL_FRONTIER_SHARE = 1.0
ELIGIBLE_HARDENING_STATUSES = frozenset(feedback.ELIGIBLE_STATUSES)
SEARCH_ROBUST_STATUS = "robust_frontier"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def artifact_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label: str
    if root is not None:
        try:
            label = str(resolved.relative_to(root.resolve()))
        except ValueError:
            label = str(resolved)
    else:
        label = str(resolved)
    return {
        "path": label,
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_artifact(record: dict[str, Any]) -> None:
    path = Path(str(record.get("resolved_path") or ""))
    if not path.is_file():
        raise RuntimeError(f"Recorded closure artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise RuntimeError(
            f"Recorded closure artifact changed: {path}; expected={record.get('sha256')} actual={actual}"
        )


def resolved_equal(left: Any, right: Path) -> bool:
    if not str(left or ""):
        return False
    return Path(str(left)).expanduser().resolve() == right.expanduser().resolve()


def status_of(sample: dict[str, Any]) -> str:
    audit = sample.get("stage64_permutation_eval")
    return str(audit.get("status") or "") if isinstance(audit, dict) else ""


def validate_robust_frontier_row(sample: dict[str, Any]) -> None:
    sample_id = str(sample.get("sample_id") or "")
    if not sample_id:
        raise RuntimeError("Search robust-frontier row has an empty sample_id")
    audit = sample.get("stage64_permutation_eval")
    if not isinstance(audit, dict) or audit.get("status") != SEARCH_ROBUST_STATUS:
        raise RuntimeError(f"Sample {sample_id!r} is not a robust frontier")
    if audit.get("k_eval_per_permutation") != FINAL_K:
        raise RuntimeError(f"Sample {sample_id!r} is not a cyclic K={FINAL_K} audit")
    if float(audit.get("required_frontier_permutation_share") or 0.0) != FINAL_FRONTIER_SHARE:
        raise RuntimeError(f"Sample {sample_id!r} did not require every cyclic position")
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    arity = len(qa.get("options") or []) if isinstance(qa.get("options"), list) else 0
    permutations = audit.get("permutations") if isinstance(audit.get("permutations"), list) else []
    if arity not in (2, 4) or len(permutations) != arity or audit.get("permutation_count") != arity:
        raise RuntimeError(f"Sample {sample_id!r} has an incomplete cyclic audit")
    labels = set("ABCD"[:arity])
    seen: set[str] = set()
    for permutation in permutations:
        if not isinstance(permutation, dict):
            raise RuntimeError(f"Sample {sample_id!r} has a malformed cyclic audit")
        gold = permutation.get("gold")
        if isinstance(gold, list):
            gold = gold[0] if len(gold) == 1 else ""
        seen.add(str(gold or ""))
        count = permutation.get("exact_correct_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count < FINAL_K:
            raise RuntimeError(f"Sample {sample_id!r} contains a non-frontier cyclic position")
    if seen != labels:
        raise RuntimeError(f"Sample {sample_id!r} does not cover every answer position")


def validate_search_inputs(
    *,
    batch_run_dir: Path,
    search_recalibration_dir: Path,
    generation_seed: int,
    search_eval_seed: int,
    final_eval_seed: int,
    accepted_solver: Path,
    base_model: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    batch = batch_run_dir.resolve()
    search = search_recalibration_dir.resolve()
    try:
        search.relative_to(batch)
    except ValueError as exc:
        raise RuntimeError("Search recalibration directory must belong to the declared batch run directory") from exc
    if isinstance(generation_seed, bool) or not isinstance(generation_seed, int):
        raise RuntimeError("generation_seed must be an integer")
    if search_eval_seed == final_eval_seed:
        raise RuntimeError("final_eval_seed must differ from search_eval_seed")
    if not accepted_solver.exists():
        raise RuntimeError(f"Accepted Stage2 solver does not exist: {accepted_solver}")
    if not base_model.exists():
        raise RuntimeError(f"Base model does not exist: {base_model}")
    summary_path = search / "recalibration_summary.json"
    clean_path = search / "clean_samples.jsonl"
    if not summary_path.is_file() or not clean_path.is_file():
        raise RuntimeError(f"Search recalibration is incomplete: {search}")
    summary = read_json(summary_path)
    config, _sources = accumulator.fixed_solver_eval_config(summary)
    expected_config = {
        "k_eval": FINAL_K,
        "max_model_len": FINAL_MAX_MODEL_LEN,
        "max_new_tokens": FINAL_MAX_NEW_TOKENS,
        "temperature": FINAL_TEMPERATURE,
        "top_p": FINAL_TOP_P,
        "seed": int(search_eval_seed),
    }
    gates = {
        "k5_cyclic": summary.get("k_eval") == FINAL_K and summary.get("permutation_mode") == "cyclic",
        "all_positions_required": float(summary.get("min_frontier_permutation_share") or 0.0)
        == FINAL_FRONTIER_SHARE,
        "semantic_gate": (summary.get("quality") or {}).get("semantic_quality_gate_enabled") is True,
        "decoding_healthy": (summary.get("decoding_health") or {}).get("healthy") is True,
        "accepted_solver_matches": resolved_equal(summary.get("accepted_stage2_solver_resolved"), accepted_solver),
        "loaded_solver_matches": resolved_equal(summary.get("loaded_solver_model_resolved"), accepted_solver),
        "adapter_absent": not summary.get("loaded_solver_adapter"),
        "base_model_matches": resolved_equal(summary.get("base_model"), base_model),
        "generation_seed_matches": summary.get("generation_seed") == generation_seed,
        "fixed_search_config_matches": config == expected_config,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Untrusted search recalibration {search}: {gates}; config={config}")
    rows = read_jsonl(clean_path)
    if len(rows) != int(summary.get("clean_count") or 0):
        raise RuntimeError("Search clean_samples count differs from recalibration_summary")
    ids: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise RuntimeError("Search clean_samples contains an empty sample_id")
        row_sha = sha256_json(row)
        if sample_id in ids:
            if ids[sample_id] != row_sha:
                raise RuntimeError(f"Conflicting duplicate search sample_id: {sample_id}")
            raise RuntimeError(f"Duplicate search sample_id: {sample_id}")
        ids[sample_id] = row_sha
        try:
            controller.validate_controller_sample(row)
        except ValueError as exc:
            raise RuntimeError(
                f"Search sample {sample_id!r} failed the current Stage3 generation protocol/semantic gate: {exc}"
            ) from exc
    original_robust = [copy.deepcopy(row) for row in rows if status_of(row) == SEARCH_ROBUST_STATUS]
    for row in original_robust:
        validate_robust_frontier_row(row)
    eligible = [copy.deepcopy(row) for row in rows if status_of(row) in ELIGIBLE_HARDENING_STATUSES]
    return summary, rows, original_robust, eligible


def _visible_candidate_key(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    return sha256_json(
        {
            "lineage": accumulator.parent_sample_id(sample),
            "question": stage64_quality.normalized_option_text(qa.get("question") or qa.get("question_text")),
            "options": sorted(
                stage64_quality.normalized_option_text(option.get("text"))
                for option in (qa.get("options") or [])
                if isinstance(option, dict)
            ),
        }
    )


def namespace_and_deduplicate_candidates(
    original_rows: Sequence[dict[str, Any]],
    hardened_rows: Sequence[dict[str, Any]],
    *,
    generation_seed: int,
    batch_namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    staged: list[tuple[str, str, dict[str, Any]]] = []
    for kind, rows in (("original", original_rows), ("hardened", hardened_rows)):
        source_ids: dict[str, str] = {}
        for source in rows:
            source_id = str(source.get("sample_id") or "")
            if not source_id:
                raise RuntimeError(f"{kind} candidate has an empty sample_id")
            row_sha = sha256_json(source)
            if source_id in source_ids:
                if source_ids[source_id] != row_sha:
                    raise RuntimeError(f"Conflicting duplicate {kind} candidate ID: {source_id}")
                continue
            source_ids[source_id] = row_sha
            staged.append((kind, source_id, copy.deepcopy(source)))

    output: list[dict[str, Any]] = []
    deduplicated: list[dict[str, Any]] = []
    visible: dict[str, tuple[str, str]] = {}
    namespaced: dict[str, str] = {}
    for kind, source_id, row in sorted(staged, key=lambda item: (item[0], item[1], sha256_json(item[2]))):
        semantic_sha = accumulator.content_fingerprint(row)
        visible_key = _visible_candidate_key(row)
        if visible_key in visible:
            prior_semantic_sha, prior_id = visible[visible_key]
            if prior_semantic_sha != semantic_sha:
                raise RuntimeError(
                    "Conflicting duplicate candidate content: identical lineage/question/options "
                    f"but different gold semantics ({prior_id!r}, {source_id!r})"
                )
            deduplicated.append(
                {
                    "source_kind": kind,
                    "source_sample_id": source_id,
                    "duplicate_of": prior_id,
                    "content_fingerprint": semantic_sha,
                }
            )
            continue
        source_sha = sha256_json(row)
        id_digest = sha256_json(
            {
                "schema": SCHEMA_VERSION,
                "batch_namespace": batch_namespace,
                "generation_seed": generation_seed,
                "source_kind": kind,
                "source_sample_id": source_id,
                "source_sha256": source_sha,
            }
        )[:24]
        new_id = f"stage3_g{generation_seed}_{kind}_{id_digest}"
        if new_id in namespaced and namespaced[new_id] != source_sha:
            raise RuntimeError(f"Stable namespaced sample_id collision: {new_id}")
        namespaced[new_id] = source_sha
        row["sample_id"] = new_id
        row["stage3_batch_closure"] = {
            "schema_version": SCHEMA_VERSION,
            "batch_namespace": batch_namespace,
            "generation_seed": int(generation_seed),
            "source_kind": kind,
            "source_sample_id": source_id,
            "source_sha256": source_sha,
            "content_fingerprint": semantic_sha,
            "stable_namespaced_sample_id": new_id,
        }
        visible[visible_key] = (semantic_sha, new_id)
        output.append(row)
    output.sort(key=lambda row: str(row.get("sample_id") or ""))
    return output, deduplicated


def command_text(argv: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def external_commands(
    *,
    search_dir: Path,
    output_dir: Path,
    config_path: Path | None,
    accepted_solver: Path,
    base_model: Path,
    generation_seed: int,
    final_eval_seed: int,
    hardening_needed: bool,
    max_hardening_retries: int,
    gpu_memory_utilization: float,
) -> list[dict[str, Any]]:
    python = sys.executable
    feedback_script = ROOT / "scripts/current_method/stage3_solver_feedback_mining.py"
    shuffle_script = ROOT / "scripts/current_method/shuffle_stage64_options.py"
    recal_script = ROOT / "scripts/current_method/recalibrate_stage64_shuffled_frontier.py"
    commands: list[dict[str, Any]] = []
    if hardening_needed:
        attempt = output_dir / "hardening" / "attempt_01"
        commands.append(
            {
                "stage": "qa_export",
                "conditional": False,
                "argv": [
                    python,
                    str(feedback_script),
                    "export",
                    "--recalibration-dir",
                    str(search_dir),
                    "--output-dir",
                    str(attempt),
                    "--accepted-stage2-solver-model",
                    str(accepted_solver),
                    "--final-eval-seed",
                    str(final_eval_seed),
                ],
            }
        )
        for index in range(1, max_hardening_retries + 2):
            attempt = output_dir / "hardening" / f"attempt_{index:02d}"
            commands.extend(
                [
                    {
                        "stage": f"qa_generate_attempt_{index:02d}",
                        "conditional": index > 1,
                        "condition": "previous attempt has rejected or missing queue IDs" if index > 1 else None,
                        "argv": [
                            python,
                            str(feedback_script),
                            "generate",
                            "--queue-dir",
                            str(attempt),
                            "--config",
                            str(config_path) if config_path else "<required-config>",
                            "--responses",
                            str(attempt / "challenger_qa_responses.jsonl"),
                        ],
                    },
                    {
                        "stage": f"qa_validate_attempt_{index:02d}",
                        "conditional": index > 1,
                        "condition": "the corresponding retry queue exists" if index > 1 else None,
                        "argv": [
                            python,
                            str(feedback_script),
                            "validate",
                            "--queue-dir",
                            str(attempt),
                            "--responses",
                            str(attempt / "challenger_qa_responses.jsonl"),
                            "--output-dir",
                            str(attempt / "validated"),
                            "--allow-partial",
                        ],
                    },
                ]
            )
    combined = output_dir / "combined" / "combined_candidates.jsonl"
    shuffled = output_dir / "shuffled"
    final = output_dir / "final_recalibration"
    commands.extend(
        [
            {
                "stage": "option_shuffle",
                "conditional": True,
                "condition": "combined candidate count is positive",
                "argv": [
                    python,
                    str(shuffle_script),
                    "--input",
                    str(combined),
                    "--output-dir",
                    str(shuffled),
                    "--seed",
                    str(final_eval_seed),
                ],
            },
            {
                "stage": "independent_final_cyclic_k5",
                "conditional": True,
                "condition": "shuffle output count is positive",
                "argv": [
                    python,
                    str(recal_script),
                    "--input",
                    str(shuffled / "clean_samples.jsonl"),
                    "--output-dir",
                    str(final),
                    "--report-dir",
                    str(final / "reports"),
                    "--base-model",
                    str(base_model),
                    "--solver-model",
                    str(accepted_solver),
                    "--accepted-stage2-solver-model",
                    str(accepted_solver),
                    "--solver-tokenizer",
                    str(base_model),
                    "--k-eval",
                    str(FINAL_K),
                    "--max-model-len",
                    str(FINAL_MAX_MODEL_LEN),
                    "--max-new-tokens",
                    str(FINAL_MAX_NEW_TOKENS),
                    "--temperature",
                    str(FINAL_TEMPERATURE),
                    "--top-p",
                    str(FINAL_TOP_P),
                    "--gpu-memory-utilization",
                    str(gpu_memory_utilization),
                    "--seed",
                    str(final_eval_seed),
                    "--generation-seed",
                    str(generation_seed),
                    "--permutation-mode",
                    "cyclic",
                    "--min-frontier-permutation-share",
                    str(FINAL_FRONTIER_SHARE),
                    "--heldout-ratio",
                    "0.0",
                ],
            },
        ]
    )
    for item in commands:
        item["command"] = command_text(item["argv"])
    return commands


class CommandRunner:
    def run(self, argv: Sequence[str], *, allowed_returncodes: set[int] | None = None) -> int:
        allowed = allowed_returncodes or {0}
        result = subprocess.run(list(argv), cwd=ROOT, check=False)
        if result.returncode not in allowed:
            raise RuntimeError(f"Closure command failed ({result.returncode}): {command_text(argv)}")
        return result.returncode


def _state_config(
    *,
    batch_run_dir: Path,
    search_dir: Path,
    output_dir: Path,
    generation_seed: int,
    search_eval_seed: int,
    final_eval_seed: int,
    accepted_solver: Path,
    base_model: Path,
    config_path: Path | None,
    max_hardening_retries: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    return {
        "batch_run_dir": str(batch_run_dir.resolve()),
        "search_recalibration_dir": str(search_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "generation_seed": int(generation_seed),
        "search_eval_seed": int(search_eval_seed),
        "final_eval_seed": int(final_eval_seed),
        "accepted_stage2_solver_model": str(accepted_solver.resolve()),
        "base_model": str(base_model.resolve()),
        "challenger_config": str(config_path.resolve()) if config_path else None,
        "max_hardening_retries": int(max_hardening_retries),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "final_eval_config": {
            "k_eval": FINAL_K,
            "max_model_len": FINAL_MAX_MODEL_LEN,
            "max_new_tokens": FINAL_MAX_NEW_TOKENS,
            "temperature": FINAL_TEMPERATURE,
            "top_p": FINAL_TOP_P,
            "permutation_mode": "cyclic",
            "min_frontier_permutation_share": FINAL_FRONTIER_SHARE,
        },
        "required_generation_protocol_fingerprint": controller.generation_protocol()["fingerprint"],
    }


def load_or_initialize_state(
    output_dir: Path,
    *,
    config: dict[str, Any],
    input_records: list[dict[str, Any]],
) -> dict[str, Any]:
    state_path = output_dir / STATE_FILE
    config_sha = sha256_json(config)
    if state_path.exists():
        state = read_json(state_path)
        if state.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Closure state schema mismatch")
        if state.get("config_sha256") != config_sha or state.get("config") != config:
            raise RuntimeError("Closure resume configuration differs from the existing state")
        expected_inputs = {record["resolved_path"]: record["sha256"] for record in state.get("inputs", [])}
        actual_inputs = {record["resolved_path"]: record["sha256"] for record in input_records}
        if expected_inputs != actual_inputs:
            raise RuntimeError("Closure input artifacts changed since the state was created")
        for record in state.get("outputs", {}).values():
            verify_artifact(record)
        for attempt in (state.get("hardening") or {}).get("attempts", []):
            if not isinstance(attempt, dict):
                raise RuntimeError("Closure state contains a malformed hardening attempt")
            for key in (
                "validation_summary",
                "responses",
                "accepted_revalidated_samples",
                "rejected_repairs",
            ):
                record = attempt.get(key)
                if isinstance(record, dict):
                    verify_artifact(record)
        return state
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to adopt a non-empty closure output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "search_ready",
        "config": config,
        "config_sha256": config_sha,
        "inputs": input_records,
        "outputs": {},
        "counts": {},
        "hardening": {"attempts": [], "accepted_queue_ids": [], "quarantined_queue_ids": []},
        "commands_executed": [],
        "history": [{"event": "search_ready"}],
    }
    write_json_atomic(state_path, state)
    return state


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    write_json_atomic(output_dir / STATE_FILE, state)


def record_output(output_dir: Path, state: dict[str, Any], key: str, path: Path) -> None:
    state.setdefault("outputs", {})[key] = artifact_record(path, root=output_dir)


def run_and_record(
    runner: CommandRunner,
    state: dict[str, Any],
    command: dict[str, Any],
    *,
    allowed_returncodes: set[int] | None = None,
) -> None:
    returncode = runner.run(command["argv"], allowed_returncodes=allowed_returncodes)
    state.setdefault("commands_executed", []).append(
        {"stage": command["stage"], "command": command["command"], "returncode": returncode}
    )


def _retry_queue(
    *,
    original_attempt_dir: Path,
    previous_attempt_dir: Path,
    retry_dir: Path,
    queue_ids: set[str],
    retry_index: int,
) -> None:
    source_manifest = read_json(original_attempt_dir / "queue_ready_manifest.json")
    queue_rows = read_jsonl(original_attempt_dir / "hardening_queue.jsonl")
    frozen_rows = read_jsonl(original_attempt_dir / "frozen_samples.jsonl")
    selected_queue = [row for row in queue_rows if str(row.get("queue_id") or "") in queue_ids]
    selected_frozen = [row for row in frozen_rows if str(row.get("queue_id") or "") in queue_ids]
    if {str(row.get("queue_id") or "") for row in selected_queue} != queue_ids:
        raise RuntimeError("Cannot build retry queue: one or more rejected queue IDs are missing")
    if {str(row.get("queue_id") or "") for row in selected_frozen} != queue_ids:
        raise RuntimeError("Cannot build retry queue: frozen retry bundle is incomplete")

    previous_validation_dir = previous_attempt_dir / "validated"
    previous_rejected = read_jsonl(previous_validation_dir / "rejected_repairs.jsonl")
    rejection_by_id: dict[str, dict[str, Any]] = {}
    for rejection in previous_rejected:
        queue_id = str(rejection.get("queue_id") or "")
        if not queue_id or queue_id in rejection_by_id:
            raise RuntimeError("Previous validation has empty or duplicate rejected queue IDs")
        rejection_by_id[queue_id] = rejection
    previous_summary = read_json(previous_validation_dir / "validation_summary.json")
    missing_ids = {str(item) for item in previous_summary.get("missing_queue_ids") or []}
    if "" in missing_ids or set(rejection_by_id) & missing_ids:
        raise RuntimeError("Previous validation has malformed rejected/missing partitions")

    retry_requirement = (
        "Fix previous_validation_failure.reason and return a complete self-contained "
        "qa_proposal object; include every required field even when unchanged."
    )
    retry_rows: list[dict[str, Any]] = []
    for source_row in selected_queue:
        row = copy.deepcopy(source_row)
        queue_id = str(row.get("queue_id") or "")
        original_contract = copy.deepcopy(row.get("frozen_contract"))
        rejection = rejection_by_id.get(queue_id)
        if rejection is not None:
            reason = str(rejection.get("reason") or "rejected_repair")
            failure_kind = "rejected_repair"
        elif queue_id in missing_ids:
            reason = "missing_response"
            failure_kind = "missing_response"
        else:
            raise RuntimeError(
                f"Retry queue ID {queue_id!r} has no previous validation failure"
            )
        request = row.get("challenger_request")
        if not isinstance(request, dict):
            raise RuntimeError(f"Retry queue ID {queue_id!r} lacks challenger_request")
        requirements = request.get("requirements")
        if not isinstance(requirements, list) or not all(
            isinstance(item, str) for item in requirements
        ):
            raise RuntimeError(f"Retry queue ID {queue_id!r} has malformed requirements")
        request["previous_validation_failure"] = {
            "attempt": int(retry_index) - 1,
            "queue_id": queue_id,
            "kind": failure_kind,
            "reason": reason,
        }
        if retry_requirement not in requirements:
            requirements.append(retry_requirement)
        if row.get("queue_id") != queue_id or row.get("frozen_contract") != original_contract:
            raise RuntimeError("Retry feedback mutation changed queue identity or frozen contract")
        retry_rows.append(row)

    retry_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(retry_dir / "hardening_queue.jsonl", retry_rows)
    write_jsonl_atomic(retry_dir / "frozen_samples.jsonl", selected_frozen)
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "queue_count": len(queue_ids),
            "retry_attempt": retry_index,
            "retry_queue_ids": sorted(queue_ids),
            "previous_validation_attempt": retry_index - 1,
            "previous_rejected_repairs_sha256": sha256_file(
                previous_validation_dir / "rejected_repairs.jsonl"
            ),
            "previous_validation_summary_sha256": sha256_file(
                previous_validation_dir / "validation_summary.json"
            ),
            "source_queue_manifest_sha256": sha256_file(original_attempt_dir / "queue_ready_manifest.json"),
            "response_file": str((retry_dir / "challenger_qa_responses.jsonl").resolve()),
            "closed_loop_commands": [],
        }
    )
    write_json_atomic(retry_dir / "queue_ready_manifest.json", manifest)


def _collect_attempt_result(attempt_dir: Path) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    validation_dir = attempt_dir / "validated"
    summary = read_json(validation_dir / "validation_summary.json")
    accepted = read_jsonl(validation_dir / "accepted_revalidated_samples.jsonl")
    rejected = read_jsonl(validation_dir / "rejected_repairs.jsonl")
    queue_ids = {
        str(item.get("queue_id") or "") for item in read_jsonl(attempt_dir / "hardening_queue.jsonl")
    }
    accepted_ids = [
        str((item.get("qa_hardening_provenance") or {}).get("queue_id") or "") for item in accepted
    ]
    rejected_ids = [str(item.get("queue_id") or "") for item in rejected]
    missing_ids = [str(item) for item in summary.get("missing_queue_ids") or []]
    if (
        "" in queue_ids
        or "" in accepted_ids
        or "" in rejected_ids
        or "" in missing_ids
        or len(queue_ids) != len(read_jsonl(attempt_dir / "hardening_queue.jsonl"))
        or len(accepted_ids) != len(set(accepted_ids))
        or len(rejected_ids) != len(set(rejected_ids))
        or len(missing_ids) != len(set(missing_ids))
    ):
        raise RuntimeError("Validation emitted empty or duplicate queue IDs")
    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)
    missing_set = set(missing_ids)
    if accepted_set & rejected_set or accepted_set & missing_set or rejected_set & missing_set:
        raise RuntimeError("Validation accepted/rejected/missing queue partitions overlap")
    if accepted_set | rejected_set | missing_set != queue_ids:
        raise RuntimeError("Validation accepted/rejected/missing queue partitions are incomplete")
    summary_contract = {
        "queue_count": len(queue_ids),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "missing_response_count": len(missing_ids),
    }
    if any(int(summary.get(key, -1)) != value for key, value in summary_contract.items()):
        raise RuntimeError(
            f"Validation summary disagrees with partial output files: expected={summary_contract}"
        )
    pending = rejected_set | missing_set
    if "" in pending:
        raise RuntimeError("Validation emitted an empty rejected/missing queue ID")
    return accepted, pending, rejected


def rebuild_completed_hardening(
    state: dict[str, Any],
    *,
    hardening_dir: Path,
    original_queue_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, list[dict[str, Any]]], int]:
    """Rebuild accepted/pending work solely from immutable completed attempts."""

    if not original_queue_ids or "" in original_queue_ids:
        raise RuntimeError("Cannot resume hardening from an empty/malformed original queue")
    accepted_hardened: dict[str, dict[str, Any]] = {}
    pending = set(original_queue_ids)
    quarantine_history: dict[str, list[dict[str, Any]]] = {}
    attempts = (state.get("hardening") or {}).get("attempts", [])
    if not isinstance(attempts, list):
        raise RuntimeError("Closure hardening attempts must be a list")
    ordered = sorted(attempts, key=lambda item: int(item.get("attempt") or 0))
    for expected_index, record in enumerate(ordered, start=1):
        if not isinstance(record, dict) or record.get("attempt") != expected_index:
            raise RuntimeError("Completed hardening attempts must be contiguous from attempt 1")
        attempt_dir = hardening_dir / f"attempt_{expected_index:02d}"
        queue_rows = read_jsonl(attempt_dir / "hardening_queue.jsonl")
        attempt_queue_ids = {str(row.get("queue_id") or "") for row in queue_rows}
        if "" in attempt_queue_ids or len(attempt_queue_ids) != len(queue_rows):
            raise RuntimeError(f"Malformed queue in completed hardening attempt {expected_index}")
        if attempt_queue_ids != pending:
            raise RuntimeError(
                f"Completed hardening attempt {expected_index} queue differs from prior retry set"
            )
        for key in (
            "validation_summary",
            "responses",
            "accepted_revalidated_samples",
            "rejected_repairs",
        ):
            artifact = record.get(key)
            if not isinstance(artifact, dict):
                raise RuntimeError(f"Completed hardening attempt {expected_index} lacks {key} hash")
            verify_artifact(artifact)
        accepted, next_pending, rejected = _collect_attempt_result(attempt_dir)
        accepted_ids: set[str] = set()
        for row in accepted:
            queue_id = str((row.get("qa_hardening_provenance") or {}).get("queue_id") or "")
            if queue_id in accepted_hardened:
                raise RuntimeError(f"Hardening queue ID was accepted more than once: {queue_id}")
            accepted_hardened[queue_id] = row
            accepted_ids.add(queue_id)
        if sorted(accepted_ids) != sorted(record.get("accepted_queue_ids") or []):
            raise RuntimeError(f"Completed hardening attempt {expected_index} accepted IDs changed")
        if sorted(next_pending) != sorted(record.get("retry_queue_ids") or []):
            raise RuntimeError(f"Completed hardening attempt {expected_index} retry IDs changed")
        rejection_by_id = {str(row.get("queue_id") or ""): row for row in rejected}
        source_by_queue = {
            str(row.get("queue_id") or ""): str((row.get("source") or {}).get("sample_id") or "")
            for row in queue_rows
        }
        validation_summary = read_json(attempt_dir / "validated" / "validation_summary.json")
        missing = {str(item) for item in validation_summary.get("missing_queue_ids") or []}
        for queue_id in sorted(next_pending):
            quarantine_history.setdefault(queue_id, []).append(
                {
                    "attempt": expected_index,
                    "source_sample_id": source_by_queue.get(queue_id),
                    "reason": (
                        "missing_response"
                        if queue_id in missing
                        else (rejection_by_id.get(queue_id) or {}).get("reason", "rejected_repair")
                    ),
                }
            )
        pending = next_pending
    return accepted_hardened, pending, quarantine_history, len(ordered)


def _publish_manifest(
    output_dir: Path,
    state: dict[str, Any],
    *,
    commands: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    status = str(state.get("status") or "")
    if status == "zero_candidate":
        next_step = "Generate the next Stage3 batch; this batch contains no candidate for final K5."
    elif status == "complete":
        next_step = "Pass final_reliable_batch to the reliable-frontier accumulator."
    elif mode == "dry_run":
        next_step = "Re-run this exact closure with --execute after reviewing the command plan."
    else:
        next_step = "Resume the same --execute command; state and hashes prevent duplicated completed stages."
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": mode,
        "config_sha256": state["config_sha256"],
        "batch_namespace": state.get("batch_namespace"),
        "counts": copy.deepcopy(state.get("counts") or {}),
        "hardening": copy.deepcopy(state.get("hardening") or {}),
        "inputs": copy.deepcopy(state.get("inputs") or []),
        "outputs": copy.deepcopy(state.get("outputs") or {}),
        "commands": commands,
        "commands_executed": copy.deepcopy(state.get("commands_executed") or []),
        "final_eval_contract": copy.deepcopy(state["config"]["final_eval_config"]),
        "independent_seeds": state["config"]["search_eval_seed"] != state["config"]["final_eval_seed"],
        "generation_seed": state["config"]["generation_seed"],
        "training_authorized": False,
        "next_step": next_step,
    }
    write_json_atomic(output_dir / MANIFEST_FILE, manifest)
    return manifest


def _materialize_uncapped_final_batch(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = output_dir / "final_recalibration"
    summary_path = source_dir / "recalibration_summary.json"
    clean_path = source_dir / "clean_samples.jsonl"
    summary = read_json(summary_path)
    rows = read_jsonl(clean_path)
    robust = [copy.deepcopy(row) for row in rows if status_of(row) == SEARCH_ROBUST_STATUS]
    for row in robust:
        validate_robust_frontier_row(row)
    published = output_dir / "final_reliable_batch"
    published.mkdir(parents=True, exist_ok=True)
    source_frontier_count = int(summary.get("frontier_count") or 0)
    closure_summary = copy.deepcopy(summary)
    closure_summary.update(
        {
            "frontier_count": len(robust),
            "closure_uncapped_reliable_frontier": True,
            "source_balanced_frontier_count": source_frontier_count,
            "source_recalibration_summary_sha256": sha256_file(summary_path),
            "source_clean_samples_sha256": sha256_file(clean_path),
        }
    )
    write_jsonl_atomic(published / "frontier.jsonl", robust)
    write_json_atomic(published / "recalibration_summary.json", closure_summary)
    return closure_summary, robust


def execute_closure(
    *,
    state: dict[str, Any],
    output_dir: Path,
    commands: list[dict[str, Any]],
    original_robust: list[dict[str, Any]],
    eligible_count: int,
    runner: CommandRunner,
) -> None:
    command_by_stage = {item["stage"]: item for item in commands}
    hardening = state.setdefault("hardening", {})
    accepted_hardened: dict[str, dict[str, Any]] = {}
    quarantine_history: dict[str, list[dict[str, Any]]] = {}
    hardening_dir = output_dir / "hardening"

    if eligible_count:
        attempt_one = hardening_dir / "attempt_01"
        if not (attempt_one / "queue_ready_manifest.json").is_file():
            run_and_record(runner, state, command_by_stage["qa_export"])
            state["status"] = "hardening_exported"
            record_output(output_dir, state, "qa_queue_manifest", attempt_one / "queue_ready_manifest.json")
            record_output(output_dir, state, "qa_queue", attempt_one / "hardening_queue.jsonl")
            record_output(output_dir, state, "qa_frozen", attempt_one / "frozen_samples.jsonl")
            state["history"].append({"event": "hardening_exported"})
            save_state(output_dir, state)
        original_queue_ids = {
            str(row.get("queue_id") or "") for row in read_jsonl(attempt_one / "hardening_queue.jsonl")
        }
        accepted_hardened, pending, quarantine_history, completed_count = rebuild_completed_hardening(
            state,
            hardening_dir=hardening_dir,
            original_queue_ids=original_queue_ids,
        )
        for attempt_index in range(completed_count + 1, int(state["config"]["max_hardening_retries"]) + 2):
            if not pending:
                break
            attempt_dir = hardening_dir / f"attempt_{attempt_index:02d}"
            if attempt_index > 1 and not (attempt_dir / "queue_ready_manifest.json").is_file():
                _retry_queue(
                    original_attempt_dir=attempt_one,
                    previous_attempt_dir=hardening_dir / f"attempt_{attempt_index - 1:02d}",
                    retry_dir=attempt_dir,
                    queue_ids=pending,
                    retry_index=attempt_index,
                )
            attempt_queue = read_jsonl(attempt_dir / "hardening_queue.jsonl")
            attempt_queue_ids = {str(row.get("queue_id") or "") for row in attempt_queue}
            if (
                "" in attempt_queue_ids
                or len(attempt_queue_ids) != len(attempt_queue)
                or attempt_queue_ids != pending
            ):
                raise RuntimeError(
                    f"Hardening attempt {attempt_index} queue does not match the pending retry set"
                )
            generate = command_by_stage[f"qa_generate_attempt_{attempt_index:02d}"]
            validate = command_by_stage[f"qa_validate_attempt_{attempt_index:02d}"]
            if not (attempt_dir / "qa_response_generation_summary.json").is_file():
                run_and_record(runner, state, generate)
            responses = attempt_dir / "challenger_qa_responses.jsonl"
            if not responses.exists():
                write_jsonl_atomic(responses, [])
            # The validator intentionally returns 2 for partial output, after
            # atomically publishing accepted/rejected/missing artifacts.
            run_and_record(runner, state, validate, allowed_returncodes={0, 2})
            accepted, next_pending, rejected = _collect_attempt_result(attempt_dir)
            queue_rows = read_jsonl(attempt_dir / "hardening_queue.jsonl")
            source_by_queue = {
                str(row.get("queue_id") or ""): str((row.get("source") or {}).get("sample_id") or "")
                for row in queue_rows
            }
            accepted_ids = {
                str((row.get("qa_hardening_provenance") or {}).get("queue_id") or "") for row in accepted
            }
            for row in accepted:
                queue_id = str((row.get("qa_hardening_provenance") or {}).get("queue_id") or "")
                if not queue_id or queue_id in accepted_hardened:
                    raise RuntimeError(f"Duplicate/empty accepted hardening queue ID: {queue_id!r}")
                accepted_hardened[queue_id] = row
            rejection_by_id = {str(row.get("queue_id") or ""): row for row in rejected}
            validation_summary = read_json(attempt_dir / "validated" / "validation_summary.json")
            missing_ids = {str(item) for item in validation_summary.get("missing_queue_ids") or []}
            for queue_id in sorted(next_pending):
                quarantine_history.setdefault(queue_id, []).append(
                    {
                        "attempt": attempt_index,
                        "source_sample_id": source_by_queue.get(queue_id),
                        "reason": (
                            "missing_response"
                            if queue_id in missing_ids
                            else (rejection_by_id.get(queue_id) or {}).get("reason", "rejected_repair")
                        ),
                    }
                )
            accepted_file = attempt_dir / "validated" / "accepted_revalidated_samples.jsonl"
            rejected_file = attempt_dir / "validated" / "rejected_repairs.jsonl"
            attempt_record = {
                "attempt": attempt_index,
                "queue_count": len(queue_rows),
                "accepted_queue_ids": sorted(accepted_ids),
                "retry_queue_ids": sorted(next_pending),
                "validation_summary": artifact_record(
                    attempt_dir / "validated" / "validation_summary.json", root=output_dir
                ),
                "responses": artifact_record(responses, root=output_dir),
                "accepted_revalidated_samples": artifact_record(accepted_file, root=output_dir),
                "rejected_repairs": artifact_record(rejected_file, root=output_dir),
            }
            hardening.setdefault("attempts", []).append(attempt_record)
            pending = next_pending
            state["status"] = "hardening_in_progress" if pending else "hardening_closed"
            state["history"].append(
                {
                    "event": "hardening_attempt_validated",
                    "attempt": attempt_index,
                    "accepted_count": len(accepted_ids),
                    "retry_count": len(pending),
                }
            )
            save_state(output_dir, state)
        hardening["accepted_queue_ids"] = sorted(accepted_hardened)
        hardening["quarantined_queue_ids"] = sorted(pending)
        accepted_path = hardening_dir / "accepted_hardened_samples.jsonl"
        quarantine_path = hardening_dir / "quarantine.jsonl"
        write_jsonl_atomic(accepted_path, [accepted_hardened[key] for key in sorted(accepted_hardened)])
        write_jsonl_atomic(
            quarantine_path,
            [
                {
                    "queue_id": queue_id,
                    "retry_count": int(state["config"]["max_hardening_retries"]),
                    "attempt_history": quarantine_history.get(queue_id, []),
                    "status": "quarantined_after_partial_rejection_or_missing_response",
                }
                for queue_id in sorted(pending)
            ],
        )
        record_output(output_dir, state, "accepted_hardened", accepted_path)
        record_output(output_dir, state, "hardening_quarantine", quarantine_path)
        state["status"] = "hardening_closed"
        save_state(output_dir, state)

    hardened_rows = [accepted_hardened[key] for key in sorted(accepted_hardened)]
    batch_namespace = str(state["batch_namespace"])
    combined, deduplicated = namespace_and_deduplicate_candidates(
        original_robust,
        hardened_rows,
        generation_seed=int(state["config"]["generation_seed"]),
        batch_namespace=batch_namespace,
    )
    combined_dir = output_dir / "combined"
    combined_path = combined_dir / "combined_candidates.jsonl"
    dedup_path = combined_dir / "deduplicated_candidates.jsonl"
    write_jsonl_atomic(combined_path, combined)
    write_jsonl_atomic(dedup_path, deduplicated)
    record_output(output_dir, state, "combined_candidates", combined_path)
    record_output(output_dir, state, "deduplicated_candidates", dedup_path)
    state["counts"].update(
        {
            "hardened_accepted": len(hardened_rows),
            "hardening_quarantined": len(hardening.get("quarantined_queue_ids") or []),
            "deduplicated": len(deduplicated),
            "combined_candidates": len(combined),
        }
    )
    state["status"] = "candidates_combined"
    state["history"].append({"event": "candidates_combined", "count": len(combined)})
    save_state(output_dir, state)
    if not combined:
        state["status"] = "zero_candidate"
        state["history"].append({"event": "zero_candidate"})
        save_state(output_dir, state)
        return

    shuffled_summary = output_dir / "shuffled" / "option_shuffle_summary.json"
    if not shuffled_summary.is_file():
        run_and_record(runner, state, command_by_stage["option_shuffle"])
    shuffled_clean = output_dir / "shuffled" / "clean_samples.jsonl"
    if not shuffled_clean.is_file():
        raise RuntimeError("Shuffle command did not publish clean_samples.jsonl")
    record_output(output_dir, state, "shuffle_summary", shuffled_summary)
    record_output(output_dir, state, "shuffled_candidates", shuffled_clean)
    state["status"] = "shuffled"
    state["history"].append({"event": "shuffled"})
    save_state(output_dir, state)

    final_summary = output_dir / "final_recalibration" / "recalibration_summary.json"
    state["status"] = "final_eval_pending"
    save_state(output_dir, state)
    if not final_summary.is_file():
        run_and_record(runner, state, command_by_stage["independent_final_cyclic_k5"])
    summary = read_json(final_summary)
    expected_final_config = {
        "k_eval": FINAL_K,
        "max_model_len": FINAL_MAX_MODEL_LEN,
        "max_new_tokens": FINAL_MAX_NEW_TOKENS,
        "temperature": FINAL_TEMPERATURE,
        "top_p": FINAL_TOP_P,
        "seed": int(state["config"]["final_eval_seed"]),
    }
    final_config, _ = accumulator.fixed_solver_eval_config(summary)
    final_gates = {
        "fixed_config": final_config == expected_final_config,
        "cyclic": summary.get("permutation_mode") == "cyclic",
        "share_one": float(summary.get("min_frontier_permutation_share") or 0.0) == FINAL_FRONTIER_SHARE,
        "generation_seed": summary.get("generation_seed") == state["config"]["generation_seed"],
        "accepted_solver": resolved_equal(
            summary.get("accepted_stage2_solver_resolved"),
            Path(state["config"]["accepted_stage2_solver_model"]),
        ),
        "loaded_solver": resolved_equal(
            summary.get("loaded_solver_model_resolved"),
            Path(state["config"]["accepted_stage2_solver_model"]),
        ),
        "adapter_absent": not summary.get("loaded_solver_adapter"),
        "decoding_healthy": (summary.get("decoding_health") or {}).get("healthy") is True,
        "semantic_gate": (summary.get("quality") or {}).get("semantic_quality_gate_enabled") is True,
    }
    if not all(final_gates.values()):
        raise RuntimeError(f"Final recalibration failed the independent fixed-config contract: {final_gates}")
    closure_summary, robust_rows = _materialize_uncapped_final_batch(output_dir)
    record_output(output_dir, state, "final_recalibration_summary", final_summary)
    record_output(
        output_dir,
        state,
        "final_reliable_summary",
        output_dir / "final_reliable_batch" / "recalibration_summary.json",
    )
    record_output(
        output_dir,
        state,
        "final_reliable_frontier",
        output_dir / "final_reliable_batch" / "frontier.jsonl",
    )
    state["counts"].update(
        {
            "final_clean": int(closure_summary.get("clean_count") or 0),
            "final_reliable_frontier": len(robust_rows),
        }
    )
    state["status"] = "complete"
    state["history"].append({"event": "complete", "reliable_frontier_count": len(robust_rows)})
    save_state(output_dir, state)


def run(args: argparse.Namespace, *, runner: CommandRunner | None = None) -> dict[str, Any]:
    if args.max_hardening_retries < 0 or args.max_hardening_retries > 2:
        raise ValueError("--max-hardening-retries must be between 0 and 2")
    summary, _search_rows, original_robust, eligible = validate_search_inputs(
        batch_run_dir=args.batch_run_dir,
        search_recalibration_dir=args.search_recalibration_dir,
        generation_seed=args.generation_seed,
        search_eval_seed=args.search_eval_seed,
        final_eval_seed=args.final_eval_seed,
        accepted_solver=args.accepted_stage2_solver_model,
        base_model=args.base_model,
    )
    config_path = args.config
    if eligible and (config_path is None or not config_path.is_file()):
        raise RuntimeError("QA hardening requires an existing --config for an exact dry-run/execute plan")
    input_paths = [
        args.search_recalibration_dir / "recalibration_summary.json",
        args.search_recalibration_dir / "clean_samples.jsonl",
    ]
    if eligible:
        marked = feedback.locate_marked_samples(args.search_recalibration_dir)
        input_paths.append(marked)
        if config_path is not None and config_path.is_file():
            input_paths.append(config_path)
    input_records = [artifact_record(path) for path in input_paths]
    config = _state_config(
        batch_run_dir=args.batch_run_dir,
        search_dir=args.search_recalibration_dir,
        output_dir=args.output_dir,
        generation_seed=args.generation_seed,
        search_eval_seed=args.search_eval_seed,
        final_eval_seed=args.final_eval_seed,
        accepted_solver=args.accepted_stage2_solver_model,
        base_model=args.base_model,
        config_path=config_path,
        max_hardening_retries=args.max_hardening_retries,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    state = load_or_initialize_state(args.output_dir, config=config, input_records=input_records)
    batch_namespace = sha256_json(
        {
            "schema": SCHEMA_VERSION,
            "generation_seed": args.generation_seed,
            "summary_sha256": sha256_file(args.search_recalibration_dir / "recalibration_summary.json"),
            "clean_samples_sha256": sha256_file(args.search_recalibration_dir / "clean_samples.jsonl"),
        }
    )[:20]
    if state.get("batch_namespace") not in (None, batch_namespace):
        raise RuntimeError("Batch namespace changed across closure resume")
    state["batch_namespace"] = batch_namespace
    state["counts"].update(
        {
            "search_clean": int(summary.get("clean_count") or 0),
            "search_original_robust_frontier": len(original_robust),
            "search_hardening_eligible": len(eligible),
        }
    )
    commands = external_commands(
        search_dir=args.search_recalibration_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        config_path=config_path.resolve() if config_path else None,
        accepted_solver=args.accepted_stage2_solver_model.resolve(),
        base_model=args.base_model.resolve(),
        generation_seed=args.generation_seed,
        final_eval_seed=args.final_eval_seed,
        hardening_needed=bool(eligible),
        max_hardening_retries=args.max_hardening_retries,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if not original_robust and not eligible:
        commands = []
        state["status"] = "zero_candidate"
        state["counts"].update(
            {
                "hardened_accepted": 0,
                "hardening_quarantined": 0,
                "deduplicated": 0,
                "combined_candidates": 0,
            }
        )
    if args.dry_run:
        state["history"].append({"event": "dry_run_plan_published"})
        save_state(args.output_dir, state)
        return _publish_manifest(args.output_dir, state, commands=commands, mode="dry_run")
    execute_closure(
        state=state,
        output_dir=args.output_dir,
        commands=commands,
        original_robust=original_robust,
        eligible_count=len(eligible),
        runner=runner or CommandRunner(),
    )
    return _publish_manifest(args.output_dir, state, commands=commands, mode="execute")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recoverably close one Stage3 self-play generation batch.")
    parser.add_argument("--batch-run-dir", type=Path, required=True)
    parser.add_argument("--search-recalibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--search-eval-seed", type=int, required=True)
    parser.add_argument("--final-eval-seed", type=int, required=True)
    parser.add_argument("--accepted-stage2-solver-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Required for an exact dry-run or execute plan when QA hardening is eligible.",
    )
    parser.add_argument(
        "--max-hardening-retries",
        type=int,
        default=2,
        help=(
            "Additional challenger-response/local-validation retries for the one-pass "
            "QA rewrite (0..2), not repeated post-evaluation difficulty rounds."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
