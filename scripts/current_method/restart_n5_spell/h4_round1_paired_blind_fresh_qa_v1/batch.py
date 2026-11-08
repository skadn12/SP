"""Round-1 paired blind Fresh QA batch.

This runner keeps the historical H4 root-level QA route intact.  It generates
Control and Feedback independently from the same frozen cumulative-dialogue
pair, with the only input difference being the R02 branch.  The first frozen
successful root is reused by immutable reference to the completed smoke run;
the remaining seven roots are generated serially.  This module does not call a
Solver, Memorax, semantic auditor, or training job.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa
from ..h4_qa_multiroot_k3_t08_probe_v1.permutation import audit_gold_rebinding, position_variants
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_batch
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config

from .run import (
    API_CONFIG,
    BRANCHES,
    COMPILER_RUN,
    CONTROL_PATHS,
    FEEDBACK_PATHS,
    GENERATOR_MAX_TOKENS,
    GENERATOR_TEMPERATURE,
    GENERATOR_VERSION,
    QUARANTINE_ROOTS,
    QUESTION_TYPES,
    RUN_BASE,
    SOURCES,
    SUCCESS_ROOTS,
    all_turns,
    candidate_pool_ids,
    close_pairs,
    evidence_span,
    load_card,
    load_json,
    load_obj,
    load_obj_from_line,
    ordering_risk,
    prompt_isolation,
    r02_turns,
    sidecar,
    write_json,
)


OUTPUT_RUN = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_4"
SMOKE_CONTROL = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2" / "control"
SMOKE_FEEDBACK = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_3" / "feedback"
SMOKE_ROOT = "cc-train-episode-254852"


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _component_fingerprint(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Fingerprint the exact source files that define a QA component."""
    files = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"qa_component_file_missing:{resolved}")
        files.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    return {
        "sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "files": files,
    }


def qa_component_fingerprints() -> dict[str, dict[str, Any]]:
    """Return auditable fingerprints for parser, repair, retry, and IDs."""
    root = _REPO_ROOT / "scripts/current_method/restart_n5_spell"
    return {
        "unique_parser": _component_fingerprint((
            root / "h4_qa_mining_hybrid_generation_v1/candidate_contract.py",
            root / "h4_qa_multiroot_k3_t08_probe_v1/validator.py",
            root / "simple_append_v1/live_runtime.py",
        )),
        "repair_prompt": _component_fingerprint((
            root / "h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py",
            root / "h4_qa_mining_hybrid_generation_v1/free_exploration_generator.py",
        )),
        "retry_policy": _component_fingerprint((
            root / "h4_qa_multiroot_k3_t08_probe_v1/generation.py",
            root / "h4_round1_paired_blind_fresh_qa_v1/reindex.py",
            root / "h4_round1_paired_blind_fresh_qa_v1/batch.py",
        )),
        "candidate_id_builder": _component_fingerprint((
            root / "h4_qa_multiroot_k3_t08_probe_v1/generation.py",
            root / "h4_round1_paired_blind_fresh_qa_v1/run.py",
            root / "h4_round1_paired_blind_fresh_qa_v1/batch.py",
        )),
    }


def _json(path: Path, value: Any) -> None:
    write_json(path, value)
    sidecar(path)


def _jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    sidecar(path)


def _request_states(*dirs: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in dirs:
        for path in sorted((directory / "ledger" / "requests").glob("*.json")):
            row = load_obj(path)
            request_id = str(row.get("request_id") or path.stem)
            if request_id in seen:
                continue
            seen.add(request_id)
            rows.append(row)
    return rows


def _scope_candidate(row: Mapping[str, Any], root_id: str, branch: str, fallback_index: int) -> dict[str, Any]:
    item = dict(row)
    old = str(item.get("source_candidate_id") or item.get("candidate_id") or "")
    prefix = f"{root_id}:R01-R02-{branch}:blind:"
    if str(item.get("candidate_id") or "").startswith(prefix):
        scoped = str(item["candidate_id"])
    else:
        source = str(item.get("generation_source") or "unknown")
        qtype = str(item.get("question_type") or "unknown")
        # `source_row_index` is local to each generator response and is
        # therefore not a candidate identity.  Use the stable root/branch
        # row ordinal supplied by the caller so candidates from different
        # intent/source responses cannot collide.
        ordinal = int(fallback_index)
        scoped = f"{root_id}:R01-R02-{branch}:blind:{source}:{qtype}:Q{ordinal:04d}"
    item["source_candidate_id"] = old
    item["candidate_id"] = scoped
    item.update({"root_id": root_id, "branch": branch, "round": "R01-R02", "qa_mode": "blind"})
    return item


def _scope_rows(rows: list[Mapping[str, Any]], root_id: str, branch: str) -> list[dict[str, Any]]:
    return [_scope_candidate(row, root_id, branch, index + 1) for index, row in enumerate(rows)]


def _build_gate_rows(raw_rows: list[dict[str, Any]], validity: Mapping[str, Any], root_id: str, branch: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits = validity.get("rows") if isinstance(validity.get("rows"), list) else []
    for index, audit in enumerate(audits):
        source = raw_rows[index] if index < len(raw_rows) else {}
        gate = dict(audit) if isinstance(audit, Mapping) else {"decision": "reject", "blocking_errors": ["invalid_audit_row"]}
        scoped = _scope_candidate(source, root_id, branch, index + 1)
        gate.update({
            "candidate_id": scoped["candidate_id"],
            "source_candidate_id": scoped.get("source_candidate_id"),
            "root_id": root_id,
            "branch": branch,
            "round": "R01-R02",
            "qa_mode": "blind",
            "generation_source": source.get("generation_source"),
            "question_type": source.get("question_type"),
        })
        rows.append(gate)
    return rows


def _offline_finalize_branch(*, branch_dir: Path, root_id: str, branch: str,
                             dialogue: Mapping[str, Any], raw_rows: list[dict[str, Any]],
                             valid_rows: list[dict[str, Any]], validity: Mapping[str, Any],
                             request_states: list[dict[str, Any]], source_run: str,
                             reused_by_reference: bool) -> dict[str, Any]:
    scoped_raw = _scope_rows(raw_rows, root_id, branch)
    scoped_valid = _scope_rows(valid_rows, root_id, branch)
    gates = _build_gate_rows(raw_rows, validity, root_id, branch)
    permutation_rows: list[dict[str, Any]] = []
    permutation_errors: list[str] = []
    for candidate in scoped_valid:
        try:
            variants = position_variants(candidate)
            audit = audit_gold_rebinding(candidate, variants)
            permutation_rows.append({"candidate_id": candidate["candidate_id"],
                                     "question_type": candidate.get("question_type"),
                                     "variants": variants, "gold_rebinding_audit": audit})
            if not audit.get("pass"):
                permutation_errors.extend(str(x) for x in audit.get("errors") or [])
        except Exception as exc:
            permutation_errors.append(f"{candidate.get('candidate_id')}:{type(exc).__name__}:{exc}")
    atomic_text(branch_dir / "candidate_pool.json", json.dumps(scoped_raw, ensure_ascii=False, indent=2) + "\n")
    atomic_text(branch_dir / "candidates_valid.json", json.dumps(scoped_valid, ensure_ascii=False, indent=2) + "\n")
    _json(branch_dir / "validity_audit.json", {
        **dict(validity),
        "candidate_ids_scoped": True,
        "candidate_count": len(scoped_raw),
        "valid_count": len(scoped_valid),
    })
    _jsonl(branch_dir / "program_gate_results.jsonl", gates)
    _json(branch_dir / "permutations.json", {
        "candidate_count": len(permutation_rows),
        "gold_rebinding_errors": len(permutation_errors),
        "errors": permutation_errors,
        "rows": permutation_rows,
    })
    _json(branch_dir / "request_summary.json", {
        "source_run": source_run,
        "reused_by_reference": reused_by_reference,
        "request_count": len(request_states),
        "completed_request_count": sum(x.get("status") == "completed" for x in request_states),
        "nonterminal_request_count": sum(x.get("status") != "completed" for x in request_states),
        "finish_reason_counts": {
            str((x.get("response_metadata") or {}).get("finish_reason")): sum(
                (y.get("response_metadata") or {}).get("finish_reason") == (x.get("response_metadata") or {}).get("finish_reason")
                for y in request_states)
            for x in request_states
        },
    })
    return {
        "root_id": root_id, "branch": branch, "source_run": source_run,
        "reused_by_reference": reused_by_reference,
        "candidate_count": len(scoped_raw), "program_pass_count": sum(x.get("decision") == "pass" for x in gates),
        "program_reject_count": sum(x.get("decision") != "pass" for x in gates),
        "permutation_count": len(permutation_rows), "gold_rebinding_errors": len(permutation_errors),
        "request_count": len(request_states),
        "completed_request_count": sum(x.get("status") == "completed" for x in request_states),
        "nonterminal_request_count": sum(x.get("status") != "completed" for x in request_states),
        "finish_reason_length_count": sum((x.get("response_metadata") or {}).get("finish_reason") == "length" for x in request_states),
        "candidate_pool_path": str(branch_dir / "candidate_pool.json"),
        "candidates_valid_path": str(branch_dir / "candidates_valid.json"),
        "program_gate_path": str(branch_dir / "program_gate_results.jsonl"),
        "permutation_path": str(branch_dir / "permutations.json"),
        "request_states": request_states,
        "candidate_rows": scoped_raw,
        "valid_rows": scoped_valid,
        "gates": gates,
        "permutation_errors": permutation_errors,
        "evidence_spans": {span: sum(evidence_span(c, dialogue) == span for c in scoped_raw)
                           for span in ("HISTORICAL_ONLY", "R02_ONLY", "CROSS_ROUND")},
        "ordering_risk": [ordering_risk(c, dialogue) for c in scoped_valid],
    }


def _copy_dialogue_source(path: Path, branch_dir: Path) -> dict[str, Any]:
    dialogue = load_obj(path)
    atomic_text(branch_dir / "dialogue_source.json", json.dumps(dialogue, ensure_ascii=False, indent=2) + "\n")
    sidecar(branch_dir / "dialogue_source.json")
    return dialogue


def _reused_branch(*, root_id: str, branch: str, output_root: Path,
                   dialogue_path: Path) -> dict[str, Any]:
    source_dir = SMOKE_CONTROL if branch == "control" else SMOKE_FEEDBACK
    output_root.mkdir(parents=True, exist_ok=True)
    dialogue = _copy_dialogue_source(dialogue_path, output_root)
    raw_rows = load_json(source_dir / "candidate_pool.json")
    valid_rows = load_json(source_dir / "candidates_valid.json")
    validity = load_obj(source_dir / "validity_audit.json")
    states = _request_states(
        RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2" / branch,
        RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_3" / branch if branch == "feedback" else source_dir,
    )
    # A reference record makes the no-repeat decision auditable without
    # copying or modifying the old run's ledger.
    _json(output_root / "source_reference.json", {
        "reused_by_reference": True,
        "source_candidate_pool": str(source_dir / "candidate_pool.json"),
        "source_candidate_pool_sha256": sha256_file(source_dir / "candidate_pool.json"),
        "source_validity_audit": str(source_dir / "validity_audit.json"),
        "source_dialogue": str(dialogue_path),
        "source_dialogue_sha256": sha256_file(dialogue_path),
        "source_request_dirs": [str(x) for x in (
            RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2" / branch,
            RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_3" / branch if branch == "feedback" else source_dir,
        )],
    })
    return _offline_finalize_branch(
        branch_dir=output_root, root_id=root_id, branch=branch, dialogue=dialogue,
        raw_rows=raw_rows, valid_rows=valid_rows, validity=validity,
        request_states=states, source_run=str(source_dir), reused_by_reference=True,
    )


def _generate_branch(*, root_id: str, branch: str, output_root: Path,
                     dialogue_path: Path, manifest_sha: str) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    dialogue = _copy_dialogue_source(dialogue_path, output_root)
    config = load_challenger_config(API_CONFIG)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=GENERATOR_TEMPERATURE, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))
    ledger = RequestLedger(output_root, manifest_sha, "live")
    generation = generate_root_qa(
        expanded=dialogue, root_dir=output_root, root_id=root_id,
        api_config=API_CONFIG, ledger=ledger, client=client,
    )
    raw_rows = [dict(x) for x in load_json(output_root / "candidate_pool.json") if isinstance(x, Mapping)]
    valid_rows = [dict(x) for x in load_json(output_root / "candidates_valid.json") if isinstance(x, Mapping)]
    validity = load_obj(output_root / "validity_audit.json")
    request_states = ledger.completed()
    # Preserve the generator's complete candidate files under a separate raw
    # name before replacing the public files with branch-scoped IDs.
    atomic_text(output_root / "candidate_pool_unscoped.json", json.dumps(raw_rows, ensure_ascii=False, indent=2) + "\n")
    atomic_text(output_root / "candidates_valid_unscoped.json", json.dumps(valid_rows, ensure_ascii=False, indent=2) + "\n")
    result = _offline_finalize_branch(
        branch_dir=output_root, root_id=root_id, branch=branch, dialogue=dialogue,
        raw_rows=raw_rows, valid_rows=valid_rows, validity=validity,
        request_states=request_states, source_run=str(output_root), reused_by_reference=False,
    )
    result["generation"] = generation
    result["generation_status"] = generation.get("status")
    return result


def _prompt_audit(branch_result: Mapping[str, Any], root_id: str) -> list[dict[str, Any]]:
    branch_dir = Path(str(branch_result["candidate_pool_path"])).parent
    paths: list[Path] = []
    paths.extend(sorted((branch_dir / "raw" / "prompts").glob("*.txt")))
    paths.extend(sorted((branch_dir / "raw_prompts").glob("*.txt")))
    card = load_card(root_id)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        audit = prompt_isolation(path.read_text(encoding="utf-8"), card)
        rows.append({"path": str(path), **audit})
    return rows


def _load_order() -> list[str]:
    selected = [load_obj_from_line(line) for line in (COMPILER_RUN / "selected_10_roots.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    order = [str(x.get("root_id") or "") for x in selected]
    if len(order) != 10 or len(set(order)) != 10 or set(order) != SUCCESS_ROOTS | QUARANTINE_ROOTS:
        raise RuntimeError(f"selected_root_order_mismatch:{order}")
    return [root for root in order if root in SUCCESS_ROOTS]


def _lock(run_root: Path):
    path = run_root / ".batch.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"batch_run_lock_busy:{path}") from exc
    return handle


def preflight(output_run: Path) -> dict[str, Any]:
    if output_run.exists() and not (output_run / "preflight_report.json").is_file():
        raise RuntimeError(f"unrecognized_existing_output:{output_run}")
    pair = close_pairs(output_run) if not (output_run / "paired_dialogue_manifest.jsonl").is_file() else load_obj(output_run / "paired_dialogue_summary.json")
    cfg = load_challenger_config(API_CONFIG)
    order = _load_order()
    checks = {
        "successful_pair_count": len(order) == 8,
        "quarantine_count": len(QUARANTINE_ROOTS) == 2,
        "effective_temperature": GENERATOR_TEMPERATURE == 0.1,
        "generator_version": GENERATOR_VERSION == "h4_qa_multiroot_k3_t08_fresh_generation_v1",
        "model": cfg.model == "Qwen/Qwen3-14B",
        "source_paths_exist": all(CONTROL_PATHS[x].is_file() and FEEDBACK_PATHS[x].is_file() for x in order),
        "smoke_reuse_artifacts": all((p / "candidate_pool.json").is_file() and (p / "candidates_valid.json").is_file() for p in (SMOKE_CONTROL, SMOKE_FEEDBACK)),
        "no_solver_or_memory_scope": True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "frozen_success_order": order,
        "quarantine_roots": sorted(QUARANTINE_ROOTS),
        "effective_temperature": GENERATOR_TEMPERATURE,
        "api_config_temperature_is_default_only": cfg.temperature,
        "model": cfg.model,
        "generator_version": GENERATOR_VERSION,
        "max_tokens": GENERATOR_MAX_TOKENS,
        "pair_closure": pair,
        "source_runs_read_only": True,
        "solver_calls": 0,
        "memorax_add_calls": 0,
        "semantic_auditor_calls": 0,
        "dev_test_accessed": False,
        "batch_reuse_rule": {"root_id": SMOKE_ROOT, "control": str(SMOKE_CONTROL), "feedback": str(SMOKE_FEEDBACK)},
    }
    _json(output_run / "preflight_report.json", report)
    component_fingerprints = qa_component_fingerprints()
    contract = {
        "generator_version": GENERATOR_VERSION,
        "model": cfg.model,
        "effective_temperature": GENERATOR_TEMPERATURE,
        "config_file_temperature": cfg.temperature,
        "max_tokens": GENERATOR_MAX_TOKENS,
        "intent_miner_max_tokens": 4096,
        "question_types": list(QUESTION_TYPES),
        "generation_sources": list(SOURCES),
        "route": "generate_root_qa_exact_historical_process",
        "repair_policy": "only_when_source_batch_below_minimum_counts",
        "feedback_guided_qa": False,
        "paired_difference": "cumulative_dialogue_branch_only",
        "solver_calls": 0,
        "memorax_add_calls": 0,
        "semantic_auditor_calls": 0,
        # These are component-level fingerprints, rather than one monolithic
        # runner hash, so a later preflight can verify the exact parser,
        # repair, retry, and candidate-ID behavior used by the batch.
        "unique_parser_sha256": component_fingerprints["unique_parser"]["sha256"],
        "repair_prompt_sha256": component_fingerprints["repair_prompt"]["sha256"],
        "retry_policy_sha256": component_fingerprints["retry_policy"]["sha256"],
        "candidate_id_builder_sha256": component_fingerprints["candidate_id_builder"]["sha256"],
        "component_fingerprints": component_fingerprints,
    }
    _json(output_run / "paired_qa_contract.json", contract)
    _json(output_run / "batch_request_plan.json", {
        "successful_roots": order,
        "branches": list(BRANCHES),
        "root_level_route": "one intent request plus two source requests per valid intent; repair only if required",
        "smoke_root_reused": SMOKE_ROOT,
        "new_api_roots": [x for x in order if x != SMOKE_ROOT],
        "expected_maximum_without_repairs": {"per_branch_per_root": "1 + 3*2 = 7", "both_branches_new_root": 14},
        "temperature": GENERATOR_TEMPERATURE,
    })
    return report


def run_batch(output_run: Path) -> dict[str, Any]:
    pre = load_obj(output_run / "preflight_report.json")
    if pre.get("status") != "PASS":
        raise RuntimeError("batch_preflight_failed")
    order = list(pre["frozen_success_order"])
    manifest_sha = sha256_file(output_run / "paired_dialogue_manifest.jsonl")
    results: list[dict[str, Any]] = []
    qa_isolation: list[dict[str, Any]] = []
    with _lock(output_run):
        for index, root_id in enumerate(order, 1):
            root_result: dict[str, Any] = {"root_id": root_id, "ordinal": index, "status": "RUNNING"}
            root_dir = output_run / "roots" / root_id
            try:
                for branch in BRANCHES:
                    branch_dir = output_run / branch / "roots" / root_id
                    done_path = branch_dir / "branch_result.json"
                    if done_path.is_file():
                        branch_result = load_obj(done_path)
                        # The stored result intentionally omits full candidate
                        # rows; reload them for aggregate collision/statistics.
                        branch_result["candidate_rows"] = load_json(branch_dir / "candidate_pool.json")
                        branch_result["valid_rows"] = load_json(branch_dir / "candidates_valid.json")
                        branch_result["gates"] = [json.loads(x) for x in (branch_dir / "program_gate_results.jsonl").read_text().splitlines() if x.strip()]
                    elif root_id == SMOKE_ROOT:
                        branch_result = _reused_branch(root_id=root_id, branch=branch, output_root=branch_dir,
                                                       dialogue_path=CONTROL_PATHS[root_id] if branch == "control" else FEEDBACK_PATHS[root_id])
                    else:
                        branch_result = _generate_branch(root_id=root_id, branch=branch, output_root=branch_dir,
                                                         dialogue_path=CONTROL_PATHS[root_id] if branch == "control" else FEEDBACK_PATHS[root_id],
                                                         manifest_sha=manifest_sha)
                    stored = {k: v for k, v in branch_result.items() if k not in {"request_states", "candidate_rows", "valid_rows", "gates"}}
                    _json(branch_dir / "branch_result.json", stored)
                    root_result[branch] = stored
                    qa_isolation.extend(_prompt_audit(branch_result, root_id))
                root_result["status"] = "PASS"
            except Exception as exc:
                root_result["status"] = "FAIL"
                root_result["error"] = f"{type(exc).__name__}:{exc}"
                # Continue to the next frozen root; no implicit retry is made.
            results.append(root_result)
            _jsonl(output_run / "batch_root_results.jsonl", results)

    candidate_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    branch_summaries: dict[str, dict[str, int]] = {b: {"candidate_count": 0, "program_pass_count": 0, "request_count": 0, "completed_request_count": 0, "nonterminal_request_count": 0, "finish_reason_length_count": 0, "gold_rebinding_errors": 0} for b in BRANCHES}
    evidence_summary: dict[str, dict[str, int]] = {b: {span: 0 for span in ("HISTORICAL_ONLY", "R02_ONLY", "CROSS_ROUND")} for b in BRANCHES}
    ordering_rows: dict[str, list[dict[str, Any]]] = {b: [] for b in BRANCHES}
    all_prompt_pass = True
    for root in results:
        for branch in BRANCHES:
            data = root.get(branch)
            if not isinstance(data, Mapping):
                continue
            bdir = output_run / branch / "roots" / str(root["root_id"])
            rows = load_json(bdir / "candidate_pool.json") if (bdir / "candidate_pool.json").is_file() else []
            vals = load_json(bdir / "candidates_valid.json") if (bdir / "candidates_valid.json").is_file() else []
            candidate_rows.extend(rows)
            valid_rows.extend(vals)
            for key in branch_summaries[branch]:
                if key in data:
                    branch_summaries[branch][key] += int(data[key] or 0)
            for span in evidence_summary[branch]:
                evidence_summary[branch][span] += int((data.get("evidence_spans") or {}).get(span) or 0)
            # Recompute ordering risk from the scoped valid rows and dialogue.
            dialogue = load_obj(bdir / "dialogue_source.json")
            ordering_rows[branch].extend(ordering_risk(c, dialogue) for c in vals)
            audit = _prompt_audit(data, str(root["root_id"]))
            all_prompt_pass = all_prompt_pass and all(x.get("pass") for x in audit)

    ids = [str(x.get("candidate_id") or "") for x in candidate_rows]
    old_ids = candidate_pool_ids()
    collision = {
        "candidate_count": len(ids),
        "global_unique": len(ids) == len(set(ids)),
        "cross_branch_overlap": sorted({x for x in ids if x.rsplit(":R01-R02-", 1)[0] in SUCCESS_ROOTS}),
        "round0_overlap": sorted(set(ids) & old_ids),
        "round0_candidate_count": len(old_ids),
        "candidate_id_schema_ok": all(":R01-R02-control:blind:" in x or ":R01-R02-feedback:blind:" in x for x in ids),
    }
    # The exact cross-branch overlap is computed by branch sets, not by a
    # string heuristic, and is kept separate for an auditable zero result.
    branch_id_sets = {b: {str(x.get("candidate_id")) for x in candidate_rows if x.get("branch") == b} for b in BRANCHES}
    collision["cross_branch_overlap"] = sorted(branch_id_sets["control"] & branch_id_sets["feedback"])
    collision["global_new_id_unique"] = len(ids) == len(set(ids))
    _json(output_run / "candidate_id_collision_audit.json", collision)
    _json(output_run / "qa_prompt_isolation_audit.json", {"status": "PASS" if all_prompt_pass else "FAIL", "rows": qa_isolation})
    _json(output_run / "evidence_span_summary.json", evidence_summary)
    _json(output_run / "ordering_static_risk_audit.json", ordering_rows)
    _json(output_run / "branch_summary.json", branch_summaries)
    _jsonl(output_run / "all_candidates.jsonl", candidate_rows)
    _jsonl(output_run / "all_candidates_valid.jsonl", valid_rows)

    successful = [x for x in results if x.get("status") == "PASS"]
    complete_pairs = [x for x in successful if all(isinstance(x.get(b), Mapping) and int(x[b].get("nonterminal_request_count", 1)) == 0 for b in BRANCHES)]
    all_gates = all(int(x.get("program_reject_count", 1)) == 0 and int(x.get("gold_rebinding_errors", 1)) == 0 for root in complete_pairs for x in (root.get("control"), root.get("feedback")))
    final_pass = len(successful) == 8 and len(complete_pairs) == 8 and all_gates and all_prompt_pass and collision["global_new_id_unique"] and not collision["round0_overlap"]
    final = {
        "status": "ROUND1_PAIRED_BLIND_QA_BATCH_PASS" if final_pass else "ROUND1_PAIRED_BLIND_QA_BATCH_PARTIAL",
        "successful_pair_count": len(successful),
        "complete_pair_count": len(complete_pairs),
        "planned_pair_count": 8,
        "quarantine_count": 2,
        "quarantine_roots": sorted(QUARANTINE_ROOTS),
        "reused_smoke_root": SMOKE_ROOT,
        "new_api_roots": [x for x in order if x != SMOKE_ROOT],
        "branches": branch_summaries,
        "candidate_count_total": len(candidate_rows),
        "valid_candidate_count_total": len(valid_rows),
        "candidate_id_collision_audit": collision,
        "qa_prompt_isolation_status": "PASS" if all_prompt_pass else "FAIL",
        "temperature": GENERATOR_TEMPERATURE,
        "model": load_challenger_config(API_CONFIG).model,
        "solver_calls": 0,
        "memorax_add_calls": 0,
        "semantic_auditor_calls": 0,
        "dev_test_accessed": False,
        "ready_for_8_root_paired_blind_qa_generation": final_pass,
        "next_step": "STOP; only proceed to Solver after this batch is PASS and separately approved" if final_pass else "inspect failed roots; do not silently replace or retry",
    }
    _json(output_run / "final_paired_qa_batch_report.json", final)
    atomic_text(output_run / "final_paired_qa_batch_report.md", "# Round-1 Paired Blind Fresh QA Batch\n\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    sidecar(output_run / "final_paired_qa_batch_report.md")
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "live", "all"), required=True)
    parser.add_argument("--output-run", type=Path, default=OUTPUT_RUN)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    output_run = args.output_run.resolve()
    if args.phase in {"preflight", "all"}:
        result = preflight(output_run)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"live", "all"}:
        if not args.live:
            raise RuntimeError("batch_live_requires_explicit_live")
        if not (output_run / "preflight_report.json").is_file():
            preflight(output_run)
        result = run_batch(output_run)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
