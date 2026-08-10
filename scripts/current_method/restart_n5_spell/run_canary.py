from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import PARSER_VERSION
from .core import (NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION, atomic_json, atomic_text,
                   canonical_json, classify_frontier_pre_audit, full_context_gate,
                   context_preflight, grounding_decision,
                   sha256_file, sha256_bytes)
from .input_adapter import load_lineage, read_json, verify_source_run_sessions
from .prompts import (QUESTIONER_FEEDBACK_PROMPT_VERSION,
                      QUESTIONER_FORMAT_RETRY_VERSION, QUESTIONER_PROMPT_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
                      QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
                      SEMANTIC_LEAKAGE_REPAIR_JSON_SCHEMA,
                      SEMANTIC_LEAKAGE_REPAIR_PROMPT_VERSION,
                      SEMANTIC_LEAKAGE_REPAIR_SCHEMA_VERSION,
                      SOLVER_NOTHINK_RESPONDER_PROMPT, build_questioner_feedback_prompt,
                      build_questioner_prompt,
                      build_semantic_leakage_repair_prompt,
                      build_questioner_span_ids_only_feedback_prompt,
                      feedback_questioner_template_hash, prompt_template_hashes,
                      questioner_format_retry_suffix_hash,
                      semantic_leakage_repair_template_hash,
                      semantic_repair_format_retry_suffix_hash,
                      span_ids_only_questioner_template_hash)
from .validation import (audit_continuity_hooks, materialize_questioner_item,
                         validate_config, validate_dual_admission_batch, validate_qa)
from .questioner_contract import (
    QUESTIONER_SCHEMA_VERSION, REASONING_CONTRACT_VERSION, SPAN_CATALOG_VERSION,
    build_span_catalog, default_reasoning_contracts, reasoning_contract_sha256,
    span_catalog_sha256,
)
from .minimal_admission import (
    LEGACY_OPTIONAL_SCHEMA_PROFILE, MINIMAL_MATERIALIZER_VERSION,
    MINIMAL_NORMALIZATION_VERSION, SPAN_IDS_ONLY_SCHEMA_PROFILE,
    SPELL_STYLE_MINIMAL_MODE, STRICT_ADMISSION_MODE, build_answer_evidence_contexts,
    materialize_spell_style_minimal,
)
from .minimal_replay import (
    MINIMAL_REPLAY_VERSION, render_minimal_replay_markdown,
    verify_bad_ground_permutation_source, verify_minimal_replay_source, verify_questioner_replay_source,
    verify_solver_feedback_source,
)
from .semantic_repair import (SEMANTIC_LEAKAGE_REPAIR_REASON,
                              SEMANTIC_LEAKAGE_REPAIR_VERSION,
                              build_semantic_repair_feedback_summary,
                              verify_semantic_repair_sources)

DEFAULTS = {
    "qa_format": "mcq", "max_total_paid_api_generations": 25,
    "max_realizer_generations": 20, "max_session_attempts_per_session": 2,
    "max_questioner_generations": 4, "max_post_freeze_memory_generations": 1,
    "max_verifier_generations": 0, "max_parse_retries_per_target": 1,
    "no_context_initial_valid_k": 3, "no_context_boundary_extra_valid_k": 2,
    "no_context_bad_ground_threshold_after_extension": 3, "full_context_valid_k": 5,
    "permutation_valid_k_per_order": 5, "questioner_generations": 4,
    "questioner_mcqs_per_generation": 3, "questioner_total_candidates": 12,
    "reserved_generation_tokens": 256, "max_sessions":10,
    # Frozen Stage2 evaluation protocol. The checkpoint advertises a larger
    # architectural limit, but runtime, preflight, and provenance all use 16K.
    "solver_effective_max_model_len": 16384,
    "minimum_solver_eligible_candidates": 1,
    "allow_local_difficulty_downgrade": False,
    "questioner_backend_context_limit": 32768,
}


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def project_diff_sha256(root: Path) -> str:
    material = bytearray(git_value(root, "diff", "--binary", "HEAD").encode())
    untracked = git_value(root, "ls-files", "--others", "--exclude-standard").splitlines()
    for relative in sorted(x for x in untracked if x):
        path = root / relative
        if path.is_file():
            material.extend(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return hashlib.sha256(material).hexdigest()


def initialize_run(run_root: Path, resume: bool) -> None:
    allowed = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs").resolve()
    target = run_root.resolve()
    if target != allowed and allowed not in target.parents:
        raise ValueError(f"run root must be below {allowed}")
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise FileExistsError("refusing to overwrite non-empty run root; use --resume")
    run_root.mkdir(parents=True, exist_ok=True)


def freeze_json(path: Path, value: object, resume: bool) -> None:
    if resume:
        if not path.exists() or canonical_json(read_json(path)) != canonical_json(value):
            raise RuntimeError(f"resume frozen JSON drift: {path}")
        return
    atomic_json(path, value)


def freeze_text(path: Path, value: str, resume: bool) -> None:
    if resume:
        if not path.exists() or path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"resume frozen text drift: {path}")
        return
    atomic_text(path, value)


def freeze_bytes(path: Path, value: bytes, resume: bool) -> None:
    if resume:
        if not path.exists() or path.read_bytes() != value:
            raise RuntimeError(f"resume frozen bytes drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp = Path(handle.name); handle.write(value); handle.flush()
        import os
        os.fsync(handle.fileno())
    temp.replace(path)


def create_source_snapshot(project: Path, run_root: Path, resume: bool = False) -> dict[str, object]:
    """Freeze recoverable bodies for the uncommitted isolated route without staging Git."""
    snapshot = run_root / "provenance/source_snapshot"
    route = project / "scripts/current_method/restart_n5_spell"
    source_paths = sorted(path for path in route.rglob("*") if path.is_file() and
                          "__pycache__" not in path.parts and path.suffix != ".pyc")
    source_paths.append(project / "tests/test_restart_n5_spell.py")
    records = []
    for path in sorted(set(source_paths)):
        relative = path.relative_to(project).as_posix()
        body = path.read_bytes()
        freeze_bytes(snapshot / "files" / relative, body, resume)
        records.append({"path":relative, "size":len(body),
                        "sha256":sha256_bytes(body)})
    status = subprocess.run(["git", "status", "--short"], cwd=project,
        text=True, capture_output=True, check=True).stdout
    tracked_diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=project,
        capture_output=True, check=True).stdout
    freeze_text(snapshot / "git_status.txt", status, resume)
    freeze_bytes(snapshot / "git_diff_tracked.patch", tracked_diff, resume)
    untracked_names = git_value(project, "ls-files", "--others", "--exclude-standard").splitlines()
    untracked = []
    for relative in sorted(x for x in untracked_names if x):
        path = project / relative
        if path.is_file():
            body = path.read_bytes()
            untracked.append({"path":relative, "size":len(body), "sha256":sha256_bytes(body)})
    freeze_json(snapshot / "untracked_file_manifest.json", untracked, resume)
    freeze_json(snapshot / "source_file_sha256.json", records, resume)
    manifest = {"schema_version":"restart_n5_spell_source_snapshot_v1",
        "git_commit":git_value(project, "rev-parse", "HEAD"), "git_commit_created":False,
        "source_files":records,
        "git_status_sha256":sha256_bytes(status.encode()),
        "git_diff_tracked_sha256":sha256_bytes(tracked_diff),
        "untracked_file_manifest_sha256":sha256_file(snapshot / "untracked_file_manifest.json"),
        "source_file_manifest_sha256":sha256_file(snapshot / "source_file_sha256.json")}
    manifest_path = snapshot / "snapshot_manifest.json"
    freeze_json(manifest_path, manifest, resume)
    return {"source_snapshot_root":str(snapshot),
        "source_snapshot_manifest_path":str(manifest_path),
        "source_snapshot_manifest_sha256":sha256_file(manifest_path),
        "source_snapshot_file_count":len(records)}


def write_post_run_provenance(run_root: Path, project: Path, manifest: dict[str, object]) -> dict[str, object]:
    checks = {"git_commit_match":git_value(project, "rev-parse", "HEAD") == manifest.get("project_git_commit"),
        "project_diff_sha256_match":project_diff_sha256(project) == manifest.get("project_diff_sha256"),
        "orchestrator_sha_match":sha256_file(Path(__file__).with_name("orchestrator.py")) == manifest.get("orchestrator_source_sha256"),
        "validation_sha_match":sha256_file(Path(__file__).with_name("validation.py")) == manifest.get("validation_source_sha256"),
        "ledger_sha_match":sha256_file(Path(__file__).with_name("ledger.py")) == manifest.get("ledger_source_sha256"),
        "run_canary_sha_match":sha256_file(Path(__file__)) == manifest.get("run_canary_source_sha256"),
        "parser_sha_match":sha256_file(Path(__file__).with_name("core.py")) == manifest.get("mcq_parser_source_sha256"),
        "questioner_prompt_source_sha_match":sha256_file(Path(__file__).with_name("prompts.py")) ==
            manifest.get("questioner_prompt_builder_source_sha256"),
        "questioner_contract_source_sha_match":sha256_file(Path(__file__).with_name("questioner_contract.py")) ==
            manifest.get("questioner_contract_source_sha256"),
        "source_snapshot_manifest_sha_match":sha256_file(Path(str(manifest["source_snapshot_manifest_path"]))) ==
            manifest.get("source_snapshot_manifest_sha256")}
    if manifest.get("admission_mode") == SPELL_STYLE_MINIMAL_MODE:
        checks.update({
            "minimal_validator_sha_match":sha256_file(Path(__file__).with_name(
                "minimal_admission.py")) == manifest.get("minimal_validator_source_sha256"),
            "normalization_source_sha_match":sha256_file(Path(__file__).with_name(
                "minimal_admission.py")) == manifest.get("normalization_function_source_sha256"),
            "minimal_replay_source_sha_match":sha256_file(Path(__file__).with_name(
                "minimal_replay.py")) == manifest.get("minimal_replay_source_sha256")})
    if manifest.get("vllm_backend_source_sha256"):
        checks["vllm_backend_sha_match"] = (sha256_file(Path(__file__).with_name(
            "vllm_backends.py")) == manifest.get("vllm_backend_source_sha256"))
    if manifest.get("live_backend_source_sha256"):
        checks["live_backend_sha_match"] = (sha256_file(Path(__file__).with_name(
            "live_backends.py")) == manifest.get("live_backend_source_sha256"))
    if manifest.get("semantic_repair_source_sha256"):
        checks["semantic_repair_source_sha_match"] = (sha256_file(Path(__file__).with_name(
            "semantic_repair.py")) == manifest.get("semantic_repair_source_sha256"))
    if manifest.get("cross_dialogue_calibration_source_sha256"):
        checks["cross_dialogue_calibration_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("cross_dialogue_calibration.py")) == manifest.get(
                "cross_dialogue_calibration_source_sha256"))
    if manifest.get("cross_dialogue_runner_source_sha256"):
        checks["cross_dialogue_runner_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("run_cross_dialogue_calibration.py")) == manifest.get(
                "cross_dialogue_runner_source_sha256"))
    if manifest.get("graph_expansion_source_sha256"):
        checks["graph_expansion_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("graph_expansion.py")) == manifest.get(
                "graph_expansion_source_sha256"))
    if manifest.get("graph_expansion_runner_source_sha256"):
        checks["graph_expansion_runner_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("run_graph_expansion.py")) == manifest.get(
                "graph_expansion_runner_source_sha256"))
    if manifest.get("full_graph_qa_pilot_source_sha256"):
        checks["full_graph_qa_pilot_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("run_full_graph_qa_pilot.py")) == manifest.get(
                "full_graph_qa_pilot_source_sha256"))
    if manifest.get("evidence_view_ablation_source_sha256"):
        checks["evidence_view_ablation_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("evidence_view_ablation.py")) == manifest.get(
                "evidence_view_ablation_source_sha256"))
    if manifest.get("evidence_view_ablation_runner_source_sha256"):
        checks["evidence_view_ablation_runner_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("run_evidence_view_ablation.py")) == manifest.get(
                "evidence_view_ablation_runner_source_sha256"))
    if manifest.get("three_scope_canary_source_sha256"):
        checks["three_scope_canary_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("three_scope_canary.py")) == manifest.get(
                "three_scope_canary_source_sha256"))
    if manifest.get("three_scope_canary_runner_source_sha256"):
        checks["three_scope_canary_runner_source_sha_match"] = (sha256_file(
            Path(__file__).with_name("run_three_scope_canary.py")) == manifest.get(
                "three_scope_canary_runner_source_sha256"))
    for field, filename in (
            ("lineage_relation_miner_source_sha256", "lineage_relation_miner.py"),
            ("lineage_qa_writer_source_sha256", "lineage_qa_writer.py"),
            ("lineage_distractor_writer_source_sha256", "lineage_distractor_writer.py"),
            ("lineage_full_dialogue_prompt_source_sha256",
             "lineage_full_dialogue_prompts.py"),
            ("lineage_three_stage_canary_source_sha256", "lineage_three_stage_canary.py"),
            ("lineage_three_stage_runner_source_sha256", "run_lineage_three_stage_canary.py")):
        if manifest.get(field):
            checks[field.replace("_sha256", "_sha_match")] = (sha256_file(
                Path(__file__).with_name(filename)) == manifest.get(field))
    report = {"run_id":manifest.get("run_id"),
              "run_manifest_sha256":sha256_file(run_root / "config/run_manifest.json"),
              **checks, "authoritative_release_provenance":all(checks.values()),
              "provenance_mixed":not all(checks.values())}
    atomic_json(run_root / "reports/post_run_provenance.json", report)
    return report


def item2_question_audit(qa: dict, frozen_context: dict) -> str:
    turn_map = {x.get("turn_id"): x.get("text", "") for x in frozen_context["combined_dialogue"]}
    evidence = {turn_id: turn_map.get(turn_id) for turn_id in qa.get("evidence_turn_ids") or []}
    evidence_exists = all(text is not None for text in evidence.values())
    return "\n".join(["# batch0_item2 offline question audit", "",
        "This is a read-only diagnostic and does not participate in admission.", "",
        f"- Question: {qa['question']}",
        f"- Reference: {qa['reference_answer_letter']} — {qa['reference_answer_text']}",
        f"- Evidence IDs exist: `{str(evidence_exists).lower()}`", f"- Evidence: `{json.dumps(evidence, ensure_ascii=False)}`",
        "- Answer type: all four options are causal person/event descriptions.",
        "- Wording/common-sense shortcut risk: moderate; all options repeat 'inspired ... volunteering', so surface form alone does not uniquely identify B.",
        "- Distractor quality: childhood, charity event, and job are plausible categories but unsupported and somewhat generic.",
        "- Dialogue relevance: the cited turn names Maria's aunt as the influence, while no option names the aunt.",
        "- Reference integrity diagnostic: the frozen reference selects friends (B), which conflicts with the cited aunt statement and the Questioner's own explanation.",
        "- Leakage: the question does not expose a letter or evidence ID.",
        "- Admission consequence: none; only real no-context rollouts determine grounding.", ""])


def select_solver_qa(raw_batch: list[dict], smoke_report: dict, qa_id: str, *,
                     frozen_turns: dict[str, str] | None = None,
                     turn_session_ids: dict[str, str] | None = None,
                     span_catalog: dict | None = None,
                     reasoning_contracts: list[dict] | None = None) -> tuple[dict, dict, dict]:
    items = [x for x in smoke_report.get("items", []) if x.get("qa_id") == qa_id]
    if len(items) != 1 or items[0].get("schema_decision") != "accept":
        raise RuntimeError("solver_qa_not_admitted")
    admission = items[0]
    materialized = admission.get("materialized_qa")
    if not isinstance(materialized, dict):
        raise RuntimeError("solver_qa_not_materialized")
    generated = [x for x in raw_batch if isinstance(x, dict) and x.get("qa_id") == qa_id]
    if len(generated) != 1 or materialized.get("qa_id") != qa_id:
        raise RuntimeError("solver_qa_integrity_mismatch")
    target = admission.get("target_letter")
    item_index = admission.get("item_index")
    if (not isinstance(item_index, int) or not isinstance(reasoning_contracts, list) or
            item_index < 0 or item_index >= len(reasoning_contracts) or
            not isinstance(span_catalog, dict)):
        raise RuntimeError("solver_qa_integrity_mismatch: local contract is required")
    rematerialized, materialize_errors = materialize_questioner_item(generated[0], target,
        span_catalog=span_catalog, reasoning_contract=reasoning_contracts[item_index])
    if (materialize_errors or rematerialized is None or
            canonical_json(rematerialized) != canonical_json(materialized)):
        raise RuntimeError("solver_qa_integrity_mismatch")
    if frozen_turns is None or turn_session_ids is None:
        raise RuntimeError("solver_qa_integrity_mismatch: frozen dialogue is required")
    errors = validate_qa(materialized, set(frozen_turns), frozen_turns, turn_session_ids)
    if errors or not materialized.get("local_option_binding") or not materialized.get("canonical_answer_text"):
        raise RuntimeError(f"solver_qa_integrity_mismatch: {errors}")
    return generated[0], dict(materialized), admission


def select_minimal_solver_qa(raw_batch: list[dict], admission_item: dict, qa_id: str, *,
                             materialized_qa: dict, span_catalog: dict,
                             reasoning_contracts: list[dict]) -> tuple[dict, dict, dict]:
    """Materialized-only selector for SPELL-style minimal admission."""
    minimal = admission_item.get(SPELL_STYLE_MINIMAL_MODE) or {}
    if (admission_item.get("qa_id") != qa_id or minimal.get("decision") != "accept" or
            admission_item.get("solver_eligible") is not True):
        raise RuntimeError("solver_qa_not_admitted")
    if not isinstance(materialized_qa, dict):
        raise RuntimeError("solver_qa_not_materialized")
    item_index = admission_item.get("item_index")
    if (not isinstance(item_index, int) or item_index not in (0, 1, 2) or
            item_index >= len(raw_batch) or item_index >= len(reasoning_contracts) or
            materialized_qa.get("qa_id") != qa_id):
        raise RuntimeError("solver_qa_integrity_mismatch")
    generated = raw_batch[item_index]
    if not isinstance(generated, dict) or generated.get("qa_id") != qa_id:
        raise RuntimeError("solver_qa_integrity_mismatch")
    strict_reasons = list((admission_item.get("strict_v7") or {}).get("reasons") or [])
    rematerialized, errors, _ = materialize_spell_style_minimal(generated,
        admission_item.get("target_letter"), span_catalog=span_catalog,
        strict_v7_reasons=strict_reasons, reasoning_contract=reasoning_contracts[item_index],
        schema_profile=str(materialized_qa.get("minimal_schema_profile") or
                           LEGACY_OPTIONAL_SCHEMA_PROFILE))
    if errors or rematerialized is None or canonical_json(rematerialized) != canonical_json(materialized_qa):
        raise RuntimeError(f"solver_qa_integrity_mismatch: {errors}")
    options = materialized_qa.get("options") or {}
    letter = materialized_qa.get("reference_answer_letter")
    if (materialized_qa.get("admission_mode") != SPELL_STYLE_MINIMAL_MODE or
            not materialized_qa.get("local_option_binding") or
            not isinstance(letter, str) or letter not in "ABCD" or
            options.get(letter) != materialized_qa.get("reference_answer_text") or
            materialized_qa.get("canonical_answer_text") != materialized_qa.get("reference_answer_text")):
        raise RuntimeError("solver_qa_integrity_mismatch")
    return generated, dict(materialized_qa), admission_item


def build_input_dry_run_audit(*, parent_turns: int, parent_tokens: int, lineage: str,
                              reserved_generation_tokens: int,
                              checkpoint_max_model_len: int, effective_max_model_len: int,
                              spell_commit: str, spell_dirty: bool) -> dict[str, object]:
    return {"mode":"dry_run", "paid_api_generations":0, "local_solver_generations":0,
        "verifier_api_generations":0, "formal_outputs_created":False, "lineage":lineage,
        "parent_turns":parent_turns, "parent_tokens":parent_tokens,
        "token_count_mode":"Qwen tokenizer exact", "outline_forbidden_qa_fields_present":[],
        "checkpoint_advertised_max_model_len":checkpoint_max_model_len,
        "context_preflight":context_preflight(64, parent_tokens, 128,
            reserved_generation_tokens, effective_max_model_len),
        "spell_repo_status":"available", "spell_commit":spell_commit,
        "spell_dirty":spell_dirty, "next_stage":"mock_e2e_not_started"}


def render_solver_prompt(tokenizer, qa: dict, context_turns: list[dict]) -> str:
    qa_text = qa["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in qa["options"].items())
    context_text = "\n".join(f"[{x.get('turn_id','')}] {x['speaker']}: {x['text']}" for x in context_turns)
    user_text = (f"Context:\n{context_text}\n\n" if context_text else "") + qa_text
    return tokenizer.apply_chat_template(
        [{"role":"system", "content":SOLVER_NOTHINK_RESPONDER_PROMPT},
         {"role":"user", "content":user_text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated restart-n5 SPELL-style canary")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lineage", default="conv-41")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend-mode", choices=("audit", "mock", "live", "local", "hybrid"), default="audit")
    parser.add_argument("--smoke-stage", choices=("realizer", "realizer_continuity", "questioner",
                                                   "questioner_feedback",
                                                   "questioner_span_ids_feedback",
                                                   "solver_no_context", "solver_grounding_extension",
                                                   "solver_adaptive_item", "solver_boxed_output",
                                                   "questioner_minimal_replay",
                                                   "solver_minimal_full_audit",
                                                   "solver_no_context_permutation_diagnostic",
                                                   "semantic_leakage_repair"))
    parser.add_argument("--admission-mode", choices=(STRICT_ADMISSION_MODE,
                                                       SPELL_STYLE_MINIMAL_MODE),
                        default=STRICT_ADMISSION_MODE)
    parser.add_argument("--api-config", type=Path,
                        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--mock-crash-before-session", type=int)
    parser.add_argument("--mock-questioner-shortfall-batch", type=int)
    parser.add_argument("--mock-questioner-all-invalid", action="store_true")
    parser.add_argument("--questioner-source-root", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_spell_live_smoke_realizer_continuity_v2_release"))
    parser.add_argument("--solver-qa-source-root", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_spell_live_smoke_questioner_v1"))
    parser.add_argument("--solver-prior-root", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_spell_solver_smoke_s1_v2_nothink"))
    parser.add_argument("--solver-qa-id", default="batch0_item1")
    parser.add_argument("--solver-diagnostic-source-root", type=Path)
    parser.add_argument("--bad-ground-feedback-root", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_spell_v7_minimal_solver_batch0_item1_v1"))
    parser.add_argument("--reject-easy-feedback-root", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conv-41_spell_v7_minimal_solver_batch0_item2_v1"))
    args = parser.parse_args(argv)
    if args.backend_mode == "live" and args.smoke_stage not in (
            "realizer", "realizer_continuity", "questioner", "questioner_feedback",
            "questioner_span_ids_feedback"):
        raise RuntimeError("live execution requires an explicit supported smoke stage")
    if args.backend_mode == "local" and args.smoke_stage not in ("solver_no_context", "solver_grounding_extension", "solver_adaptive_item", "solver_boxed_output", "solver_minimal_full_audit", "solver_no_context_permutation_diagnostic"):
        raise RuntimeError("local execution requires an explicit supported solver smoke")
    if args.backend_mode == "hybrid" and args.smoke_stage != "semantic_leakage_repair":
        raise RuntimeError("hybrid execution is restricted to semantic leakage repair")
    if args.smoke_stage == "semantic_leakage_repair" and (
            args.backend_mode != "hybrid" or args.solver_diagnostic_source_root is None):
        raise RuntimeError("semantic repair requires hybrid mode and a diagnostic source root")
    if args.smoke_stage in ("questioner_minimal_replay", "solver_minimal_full_audit",
                            "solver_no_context_permutation_diagnostic",
                            "semantic_leakage_repair") and \
            args.admission_mode != SPELL_STYLE_MINIMAL_MODE:
        raise RuntimeError("minimal replay/Solver requires spell_style_minimal_v1")
    if args.smoke_stage == "questioner_minimal_replay" and args.backend_mode != "audit":
        raise RuntimeError("minimal replay is offline audit-only")
    validate_config(DEFAULTS)
    initialize_run(args.run_root, args.resume)
    project = Path(__file__).resolve().parents[3]
    source_snapshot = create_source_snapshot(project, args.run_root, args.resume)
    legacy = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
    solver = legacy / "accepted_models/solver_base_restart_stage2_merged"
    tokenizer = Path("/root/autodl-tmp/Qwen3-4B")
    data = load_lineage(args.input_root, args.lineage, legacy)
    model_config = read_json(solver / "config.json")
    live_config = None
    if args.backend_mode in ("live", "hybrid"):
        from .live_backends import load_challenger_config
        live_config = load_challenger_config(args.api_config)
    spell_repo = Path("/root/autodl-tmp/third_party_readonly/Qwen-Doc")
    spell_commit = git_value(spell_repo, "rev-parse", "HEAD")
    spell_dirty = bool(git_value(spell_repo, "status", "--porcelain"))
    if spell_commit != "4e5aee32280febb7b0844237ba087f837e3c8905" or spell_dirty:
        raise RuntimeError("SPELL source provenance is not the frozen clean commit")
    from transformers import AutoTokenizer
    length_tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    parent_text = "\n".join(
        f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}"
        for turn in data["parent"]["dialogue"]["turns"]
    )
    parent_tokens = len(length_tokenizer.encode(parent_text, add_special_tokens=False))
    mode = ("live_questioner_span_ids_feedback_smoke" if args.backend_mode == "live" and args.smoke_stage == "questioner_span_ids_feedback"
            else "live_questioner_feedback_smoke" if args.backend_mode == "live" and args.smoke_stage == "questioner_feedback"
            else "live_questioner_smoke" if args.backend_mode == "live" and args.smoke_stage == "questioner"
            else "live_smoke_realizer_continuity" if args.backend_mode == "live" and args.smoke_stage == "realizer_continuity"
            else "live_smoke_realizer" if args.backend_mode == "live"
            else "local_solver_adaptive_item_smoke" if args.backend_mode == "local" and args.smoke_stage == "solver_adaptive_item"
            else "local_solver_minimal_full_audit" if args.backend_mode == "local" and args.smoke_stage == "solver_minimal_full_audit"
            else "local_solver_no_context_permutation_diagnostic" if args.backend_mode == "local" and args.smoke_stage == "solver_no_context_permutation_diagnostic"
            else "local_solver_boxed_output_smoke" if args.backend_mode == "local" and args.smoke_stage == "solver_boxed_output"
            else "local_solver_grounding_extension_smoke" if args.backend_mode == "local" and args.smoke_stage == "solver_grounding_extension"
            else "local_solver_no_context_smoke" if args.backend_mode == "local"
            else "hybrid_semantic_leakage_repair" if args.backend_mode == "hybrid"
            else "mock_e2e" if args.backend_mode == "mock"
            else "spell_style_minimal_replay" if args.smoke_stage == "questioner_minimal_replay"
            else "dry_run")
    resolved = {**DEFAULTS, "mode": mode,
                "backend_mode": args.backend_mode, "lineage": args.lineage,
                "input_root": str(args.input_root.resolve()), "run_root": str(args.run_root.resolve()),
                "admission_mode":args.admission_mode}
    if args.backend_mode == "mock":
        resolved["mock_questioner_shortfall_batch"] = args.mock_questioner_shortfall_batch
        resolved["mock_questioner_all_invalid"] = args.mock_questioner_all_invalid
    if args.backend_mode == "live":
        continuity = args.smoke_stage == "realizer_continuity"
        feedback_smoke = args.smoke_stage in ("questioner_feedback",
                                              "questioner_span_ids_feedback")
        questioner_smoke = args.smoke_stage == "questioner" or feedback_smoke
        paid_limit = 2 if continuity or feedback_smoke else 1
        resolved.update({"smoke_stage":args.smoke_stage, "smoke_max_paid_generations":paid_limit,
            "smoke_max_realizer_generations":0 if questioner_smoke else paid_limit,
            "max_session_attempts_per_session":1, "max_total_paid_api_generations":paid_limit,
            "max_realizer_generations":0 if questioner_smoke else paid_limit,
            "max_questioner_generations":paid_limit if questioner_smoke else 0,
            "questioner_generations":paid_limit if questioner_smoke else 0,
            "questioner_mcqs_per_generation":3, "questioner_total_candidates":3 if questioner_smoke else 0,
            "max_post_freeze_memory_generations":0,
            "challenger_temperature":live_config.temperature,
            "challenger_realizer_max_tokens":live_config.realizer_max_tokens,
            "challenger_questioner_max_tokens":live_config.questioner_max_tokens})
        if feedback_smoke:
            resolved.update({"max_questioner_format_retries":1,
                "questioner_physical_request_limit":2,
                "questioner_retry_trigger":"json_invalid_and_raw_object_count_zero"})
    if args.backend_mode == "local":
        extension = args.smoke_stage == "solver_grounding_extension"
        adaptive = args.smoke_stage == "solver_adaptive_item"
        minimal_audit = args.smoke_stage == "solver_minimal_full_audit"
        permutation_diagnostic = args.smoke_stage == "solver_no_context_permutation_diagnostic"
        resolved.update({"smoke_stage":args.smoke_stage, "max_total_paid_api_generations":0,
            "max_realizer_generations":0, "max_questioner_generations":0,
            "max_post_freeze_memory_generations":0, "max_verifier_generations":0,
            "max_session_attempts_per_session":1,
            "max_local_solver_generations":12 if permutation_diagnostic else 52 if minimal_audit else 20 if adaptive else 4 if extension else 6,
            "solver_requested_valid_k":2 if extension else 3, "solver_slot_start":3 if extension else 0,
            "solver_temperature":0.7, "solver_top_p":0.9,
            "solver_max_new_tokens":32, "solver_enable_thinking":False,
            "solver_add_generation_prompt":True, "solver_effective_max_model_len":16384,
            "solver_tokenizer_path":str(tokenizer), "solver_gpu_memory_utilization":0.80})
    if args.backend_mode == "hybrid":
        resolved.update({"smoke_stage":args.smoke_stage,
            "max_total_paid_api_generations":2, "max_semantic_repair_generations":2,
            "max_realizer_generations":0, "max_questioner_generations":0,
            "max_post_freeze_memory_generations":0, "max_verifier_generations":0,
            "max_session_attempts_per_session":1, "max_local_solver_generations":52,
            "semantic_repair_max_attempts":1, "semantic_repair_max_format_retries":1,
            "solver_temperature":0.7, "solver_top_p":0.9,
            "solver_max_new_tokens":32, "solver_enable_thinking":False,
            "solver_add_generation_prompt":True, "solver_effective_max_model_len":16384,
            "solver_tokenizer_path":str(tokenizer), "solver_gpu_memory_utilization":0.80,
            "challenger_temperature":live_config.temperature,
            "challenger_questioner_max_tokens":live_config.questioner_max_tokens,
            "challenger_semantic_repair_max_tokens":min(
                live_config.questioner_max_tokens, 2048)})
    if args.smoke_stage == "questioner_minimal_replay":
        resolved.update({"max_total_paid_api_generations":0, "max_realizer_generations":0,
            "max_questioner_generations":0, "max_post_freeze_memory_generations":0,
            "max_verifier_generations":0, "maximum_source_qa_items":3,
            "maximum_solver_qa_items":3})
    validate_config(resolved)
    config_dir = args.run_root / "config"
    config_path = config_dir / "resolved_canary_config.json"
    manifest_path = config_dir / "run_manifest.json"
    if args.resume and config_path.exists():
        frozen_config = read_json(config_path)
        if frozen_config != resolved:
            raise RuntimeError("resume config differs from frozen resolved config")
    else:
        atomic_json(config_path, resolved)
    outline_path = args.input_root / args.lineage / "expansion_outline.json"
    snapshot_path = args.input_root / args.lineage / "parent_anchor_snapshot.json"
    prompt_hashes = prompt_template_hashes()
    questioner_artifacts: dict[str, object] = {}
    solver_artifacts: dict[str, object] = {}
    minimal_replay_artifacts: dict[str, object] = {}
    repair_artifacts: dict[str, object] = {}
    if args.smoke_stage == "questioner_minimal_replay":
        source = args.questioner_source_root.resolve()
        attestation = verify_questioner_replay_source(source)
        source_paths = {key:Path(value) for key,value in attestation["paths"].items()}
        copied_paths = {
            "raw":args.run_root / "frozen/source_questioner_raw.txt",
            "context":args.run_root / "frozen/questioner_smoke_context.json",
            "span_catalog":args.run_root / "frozen/questioner_span_catalog.json",
            "reasoning_contracts":args.run_root / "frozen/questioner_reasoning_contracts.json",
        }
        for key, destination in copied_paths.items():
            freeze_bytes(destination, source_paths[key].read_bytes(), args.resume)
        raw_batch = attestation["raw_batch"]
        context = attestation["context"]
        catalog = attestation["span_catalog"]
        contracts = attestation["reasoning_contracts"]
        letters = list(attestation["target_gold_letters"])
        dialogue = context["combined_dialogue"]
        frozen_turns = {turn["turn_id"]:turn["text"] for turn in dialogue}
        turn_sessions = {turn["turn_id"]:str(turn.get("session_id") or
            turn["turn_id"].split(":", 1)[0]) for turn in dialogue}
        minimal_schema_profile = str(attestation.get("source_questioner_output_profile") or
                                     LEGACY_OPTIONAL_SCHEMA_PROFILE)
        dual = validate_dual_admission_batch(raw_batch, letters, span_catalog=catalog,
            reasoning_contracts=contracts, frozen_turns=frozen_turns,
            turn_session_ids=turn_sessions,
            minimal_schema_profile=minimal_schema_profile)
        report = json.loads(canonical_json(dual))
        for item in report["items"]:
            minimal = item[SPELL_STYLE_MINIMAL_MODE]
            materialized = minimal.pop("materialized_qa", None)
            if minimal["decision"] == "accept" and isinstance(materialized, dict):
                relative = Path("materialized") / f"item_{item['item_index']:02d}.json"
                materialized_path = args.run_root / relative
                freeze_json(materialized_path, materialized, args.resume)
                item["materialized_qa_path"] = relative.as_posix()
                item["materialized_qa_sha256"] = sha256_file(materialized_path)
                item["materialized_qa_canonical_sha256"] = sha256_bytes(
                    canonical_json(materialized).encode("utf-8"))
            else:
                item["materialized_qa_path"] = None
                item["materialized_qa_sha256"] = None
                item["materialized_qa_canonical_sha256"] = None
            item["answer_span_ids"] = list((raw_batch[item["item_index"]] or {}).get(
                "answer_support_span_ids") or [])
            item["derived_turn_ids"] = list((materialized or {}).get("evidence_turn_ids") or [])
            item["derived_session_ids"] = list((materialized or {}).get("evidence_session_ids") or [])
            item["optional_evidence_value_validity"] = minimal["diagnostics"].get(
                "answer_value_literal_checks", [])
            item["optional_field_presence"] = minimal["diagnostics"].get(
                "optional_field_status", {})
        attestation_report = {key:value for key,value in attestation.items()
            if key not in {"raw_batch", "context", "span_catalog", "reasoning_contracts", "paths"}}
        report.update({"replay_version":MINIMAL_REPLAY_VERSION,
            "admission_mode":SPELL_STYLE_MINIMAL_MODE,
            "strict_admission_mode":STRICT_ADMISSION_MODE,
            "source_attestation":attestation_report,
            "source_raw_sha256":sha256_file(copied_paths["raw"]),
            "frozen_dialogue_sha256":sha256_file(copied_paths["context"]),
            "span_catalog_sha256":span_catalog_sha256(catalog),
            "reasoning_contract_sha256":reasoning_contract_sha256(contracts),
            "normalization_version":MINIMAL_NORMALIZATION_VERSION,
            "minimal_materializer_version":MINIMAL_MATERIALIZER_VERSION,
            "minimal_schema_profile":minimal_schema_profile,
            "generation_performed":False, "paid_api_generations":0,
            "live_questioner_generations":0, "live_realizer_generations":0,
            "real_vllm_generations":0, "verifier_generations":0, "training_runs":0})
        replay_report_path = args.run_root / "reports/spell_style_minimal_replay.json"
        replay_markdown_path = args.run_root / "reports/spell_style_minimal_replay.md"
        freeze_json(replay_report_path, report, args.resume)
        freeze_text(replay_markdown_path, render_minimal_replay_markdown(report), args.resume)
        freeze_json(args.run_root / "frozen/source_attestation.json", attestation_report, args.resume)
        minimal_replay_artifacts = {"source":source, "attestation":attestation,
            "report":report, "report_path":replay_report_path,
            "markdown_path":replay_markdown_path, "copied_paths":copied_paths,
            "catalog":catalog, "contracts":contracts, "context":context}
    if args.backend_mode == "live" and args.smoke_stage in (
            "questioner", "questioner_feedback", "questioner_span_ids_feedback"):
        source = args.questioner_source_root.resolve()
        source_integrity = verify_source_run_sessions(source, ["session_01.json", "session_02.json"])
        session_paths = [source / "accepted/session_01.json", source / "accepted/session_02.json"]
        sessions = source_integrity["sessions"]
        parent_turns = data["parent"]["dialogue"]["turns"]
        child_turns = [turn for session in sessions for turn in session["dialogue_turns"]]
        frozen = {"schema_version":"questioner_smoke_frozen_context_v1",
            "questioner_smoke_only":True, "not_full_canary_context":True,
            "parent_turns":parent_turns, "accepted_child_sessions":sessions,
            "combined_dialogue":parent_turns + child_turns,
            "source_continuity_root":str(source)}
        frozen_path = args.run_root / "frozen/questioner_smoke_context.json"
        freeze_json(frozen_path, frozen, args.resume)
        valid_turn_ids = tuple(turn["turn_id"] for turn in frozen["combined_dialogue"])
        parent_session_ids = {str(turn["turn_id"]).split(":", 1)[0] for turn in parent_turns}
        valid_session_ids = tuple(sorted(parent_session_ids | {turn["session_id"] for turn in child_turns}))
        letters = ("A", "B", "C")
        span_catalog = build_span_catalog(frozen["combined_dialogue"])
        reasoning_contracts = default_reasoning_contracts(letters)
        span_catalog_path = args.run_root / "frozen/questioner_span_catalog.json"
        reasoning_contracts_path = args.run_root / "frozen/questioner_reasoning_contracts.json"
        freeze_json(span_catalog_path, span_catalog, args.resume)
        freeze_json(reasoning_contracts_path, reasoning_contracts, args.resume)
        feedback_attestations: list[dict[str, object]] = []
        feedback_path = None
        if args.smoke_stage in ("questioner_feedback", "questioner_span_ids_feedback"):
            feedback_attestations = [
                verify_solver_feedback_source(args.bad_ground_feedback_root,
                    expected_qa_id="batch0_item1", expected_decision="bad_ground"),
                verify_solver_feedback_source(args.reject_easy_feedback_root,
                    expected_qa_id="batch0_item2", expected_decision="reject_easy"),
            ]
            feedback_payload = {
                "feedback_version":"solver_failure_feedback_v1",
                "source_attestations":feedback_attestations,
                "feedback":[item["feedback"] for item in feedback_attestations],
                "factual_source":False,
            }
            feedback_path = args.run_root / "frozen/challenger_solver_feedback.json"
            freeze_json(feedback_path, feedback_payload, args.resume)
            if args.smoke_stage == "questioner_span_ids_feedback":
                rendered = build_questioner_span_ids_only_feedback_prompt(
                    frozen_dialogue=frozen["combined_dialogue"],
                    valid_turn_ids=valid_turn_ids, valid_session_ids=valid_session_ids,
                    assigned_gold_letters=letters, batch_index=0,
                    span_catalog=span_catalog, solver_feedback=feedback_payload["feedback"])
            else:
                rendered = build_questioner_feedback_prompt(
                    frozen_dialogue=frozen["combined_dialogue"],
                    valid_turn_ids=valid_turn_ids, valid_session_ids=valid_session_ids,
                    assigned_gold_letters=letters, recent_frontier_history=[], batch_index=0,
                    span_catalog=span_catalog, reasoning_contracts=reasoning_contracts,
                    solver_feedback=feedback_payload["feedback"])
        else:
            rendered = build_questioner_prompt(frozen_dialogue=frozen["combined_dialogue"],
                valid_turn_ids=valid_turn_ids, valid_session_ids=valid_session_ids,
                assigned_gold_letters=letters, recent_frontier_history=[], batch_index=0,
                span_catalog=span_catalog, reasoning_contracts=reasoning_contracts)
        rendered_tokens = len(length_tokenizer.encode(rendered, add_special_tokens=False))
        prompt_preflight = {"rendered_prompt_tokens":rendered_tokens,
            "max_generation_tokens":resolved["challenger_questioner_max_tokens"],
            "total_tokens":rendered_tokens + resolved["challenger_questioner_max_tokens"],
            "backend_context_limit":resolved["questioner_backend_context_limit"]}
        prompt_preflight["decision"] = ("pass" if prompt_preflight["total_tokens"] <=
            prompt_preflight["backend_context_limit"] else "reject_context_overflow")
        if prompt_preflight["decision"] != "pass":
            raise RuntimeError("questioner_prompt_context_overflow")
        prompt_preflight_path = args.run_root / "frozen/questioner_prompt_preflight.json"
        freeze_json(prompt_preflight_path, prompt_preflight, args.resume)
        questioner_artifacts = {"source":source, "session_paths":session_paths, "sessions":sessions,
            "frozen":frozen, "frozen_path":frozen_path, "valid_turn_ids":valid_turn_ids,
            "valid_session_ids":valid_session_ids, "letters":letters, "rendered":rendered,
            "source_integrity":source_integrity, "span_catalog":span_catalog,
            "span_catalog_path":span_catalog_path, "reasoning_contracts":reasoning_contracts,
            "reasoning_contracts_path":reasoning_contracts_path,
            "prompt_preflight":prompt_preflight, "prompt_preflight_path":prompt_preflight_path,
            "feedback_attestations":feedback_attestations, "feedback_path":feedback_path}
    if args.backend_mode == "hybrid":
        source = verify_semantic_repair_sources(
            solver_root=args.solver_qa_source_root.resolve(),
            diagnostic_root=args.solver_diagnostic_source_root.resolve(),
            qa_id=args.solver_qa_id)
        repaired_qa_id = f"{args.solver_qa_id}_repair1"
        admission = source["frozen_qa"]["schema_acceptance"]
        item_index = admission.get("item_index")
        if not isinstance(item_index, int) or item_index not in (0, 1, 2):
            raise RuntimeError("semantic_repair_source_qa_integrity_failed")
        reasoning_contract = source["reasoning_contracts"][item_index]
        frozen_dialogue = source["context"]["combined_dialogue"]
        rendered = build_semantic_leakage_repair_prompt(
            frozen_dialogue=frozen_dialogue, span_catalog=source["span_catalog"],
            original_item=source["original_raw_item"], repaired_qa_id=repaired_qa_id,
            target_answer_letter=source["target_letter"],
            original_no_context=source["original_no_context"],
            permutation_diagnostic=source["permutation_diagnostic"])
        rendered_tokens = len(length_tokenizer.encode(rendered, add_special_tokens=False))
        preflight = {"rendered_prompt_tokens":rendered_tokens,
            "max_generation_tokens":resolved["challenger_semantic_repair_max_tokens"],
            "total_tokens":rendered_tokens + resolved["challenger_semantic_repair_max_tokens"],
            "backend_context_limit":resolved["questioner_backend_context_limit"]}
        preflight["decision"] = ("pass" if preflight["total_tokens"] <=
            preflight["backend_context_limit"] else "reject_context_overflow")
        if preflight["decision"] != "pass":
            raise RuntimeError("semantic_repair_prompt_context_overflow")
        frozen_paths = {"context":args.run_root / "frozen/questioner_smoke_context.json",
            "span_catalog":args.run_root / "frozen/questioner_span_catalog.json",
            "reasoning_contracts":args.run_root / "frozen/questioner_reasoning_contracts.json",
            "original_item":args.run_root / "frozen/original_questioner_item.json",
            "diagnostic":args.run_root / "frozen/authoritative_permutation_diagnostic.json",
            "source_attestation":args.run_root / "frozen/semantic_repair_source_attestation.json",
            "prompt_preflight":args.run_root / "frozen/semantic_repair_prompt_preflight.json",
            "seed_schedule":args.run_root / "frozen/seed_schedule.json"}
        freeze_json(frozen_paths["context"], source["context"], args.resume)
        freeze_json(frozen_paths["span_catalog"], source["span_catalog"], args.resume)
        freeze_json(frozen_paths["reasoning_contracts"], source["reasoning_contracts"], args.resume)
        freeze_json(frozen_paths["original_item"], source["original_raw_item"], args.resume)
        freeze_json(frozen_paths["diagnostic"], source["permutation_diagnostic"], args.resume)
        source_attestation = {key:value for key,value in source.items() if key not in {
            "qa", "frozen_qa", "original_no_context", "context", "span_catalog",
            "reasoning_contracts", "original_raw_item", "permutation_diagnostic"}}
        freeze_json(frozen_paths["source_attestation"], source_attestation, args.resume)
        freeze_json(frozen_paths["prompt_preflight"], preflight, args.resume)
        seed_schedule = {}
        for evaluation_mode, order_ids, slots in (
                ("no_context", ("original",), range(5)),
                ("full_context", ("original",), range(5)),
                ("evidence_only", ("original",), range(3)),
                ("evidence_masked", ("original",), range(3)),
                ("option_permutation", ("perm1", "perm2"), range(5))):
            for order_id in order_ids:
                key = f"{evaluation_mode}:{order_id}"
                seed_schedule[key] = {str(slot):[int(hashlib.sha256(
                    f"{repaired_qa_id}:{evaluation_mode}:{order_id}:{slot}:{retry}".encode()
                ).hexdigest()[:8], 16) for retry in (0, 1)] for slot in slots}
        freeze_json(frozen_paths["seed_schedule"], seed_schedule, args.resume)
        repair_artifacts = {"source":source, "repaired_qa_id":repaired_qa_id,
            "reasoning_contract":reasoning_contract, "rendered":rendered,
            "preflight":preflight, "frozen_paths":frozen_paths}
    if args.backend_mode == "local":
        boxed_smoke = args.smoke_stage == "solver_boxed_output"
        minimal_audit = args.smoke_stage == "solver_minimal_full_audit"
        permutation_diagnostic = args.smoke_stage == "solver_no_context_permutation_diagnostic"
        source = args.solver_qa_source_root.resolve()
        raw_path = source / "raw/backend_responses/questioner-b00.txt"
        report_path = source / "reports/live_questioner_smoke.json"
        minimal_source_integrity = None
        permutation_source_integrity = None
        if permutation_diagnostic:
            permutation_source_integrity = verify_bad_ground_permutation_source(
                source, args.solver_qa_id)
            qa = dict(permutation_source_integrity["qa"])
            original_qa = qa
            frozen_source_qa = permutation_source_integrity["frozen_qa"]
            generated_qa = frozen_source_qa.get("source_generated_qa")
            accepted_item = frozen_source_qa.get("schema_acceptance")
            if not isinstance(generated_qa, dict) or not isinstance(accepted_item, dict):
                raise RuntimeError("bad_ground_permutation_source_integrity_failed")
            local_span_catalog = permutation_source_integrity["span_catalog"]
            local_reasoning_contracts = permutation_source_integrity["reasoning_contracts"]
            raw_path = Path(permutation_source_integrity["source_qa_path"])
            report_path = source / "reports/minimal_solver_full_audit.json"
        elif minimal_audit:
            minimal_source_integrity = verify_minimal_replay_source(source, args.solver_qa_id)
            raw_batch = minimal_source_integrity["raw_batch"]
            local_span_catalog = minimal_source_integrity["span_catalog"]
            local_reasoning_contracts = minimal_source_integrity["reasoning_contracts"]
            accepted_item = minimal_source_integrity["admission_item"]
            generated_qa, qa, accepted_item = select_minimal_solver_qa(raw_batch, accepted_item,
                args.solver_qa_id, materialized_qa=minimal_source_integrity["materialized_qa"],
                span_catalog=local_span_catalog, reasoning_contracts=local_reasoning_contracts)
            original_qa = qa
            raw_path = source / "frozen/source_questioner_raw.txt"
            report_path = source / "reports/spell_style_minimal_replay.json"
        elif boxed_smoke:
            generated_qa = {"qa_id":"synthetic_boxed_protocol_01",
                "question":"Which option exactly matches the target token 'blue' stated in this question?",
                "correct_answer_text":"blue target token", "distractors":["red","green","gold"],
                "answer_support_span_ids":["SYN1#S0001", "SYN2#S0001"],
                "answer_evidence_values":[
                    {"span_id":"SYN1#S0001", "value":"blue"},
                    {"span_id":"SYN2#S0001", "value":"target token"}],
                "distractor_support":[
                    {"distractor_index":0,"span_ids":["SYN1#S0001"],"evidence_values":["red"]},
                    {"distractor_index":1,"span_ids":["SYN1#S0001"],"evidence_values":["green"]},
                    {"distractor_index":2,"span_ids":["SYN2#S0001"],"evidence_values":["gold"]}],
                "correct_answer_explanation":"The synthetic protocol fixture names blue.",
                "distractor_explanations":["red is a different fixture value",
                    "green is a different fixture value","gold is a different fixture value"],
                "reasoning_steps":["Read the target token.","Select the matching value."],
            }
            synthetic_dialogue = [
                {"turn_id":"SYN1","session_id":"synthetic_a","speaker":"A",
                 "text":"blue red green"},
                {"turn_id":"SYN2","session_id":"synthetic_b","speaker":"B",
                 "text":"target token gold"}]
            local_span_catalog = build_span_catalog(synthetic_dialogue)
            local_reasoning_contracts = default_reasoning_contracts(("A", "B", "C"))
            qa, synthetic_errors = materialize_questioner_item(generated_qa, "B",
                span_catalog=local_span_catalog,
                reasoning_contract=local_reasoning_contracts[1])
            synthetic_turns = {"SYN1":"blue red green", "SYN2":"target token gold"}
            if synthetic_errors or qa is None or validate_qa(qa, set(synthetic_turns), synthetic_turns,
                    {"SYN1":"synthetic_a","SYN2":"synthetic_b"}):
                raise RuntimeError("synthetic boxed QA failed local materialization")
            original_qa = qa
            source = args.run_root / "frozen"
            raw_path = args.run_root / "frozen/synthetic_qa_source.json"
            report_path = raw_path
            freeze_json(raw_path, generated_qa, args.resume)
        else:
            raw_batch = json.loads(raw_path.read_text(encoding="utf-8"))
            smoke_report = read_json(report_path)
            source_context_path = source / "frozen/questioner_smoke_context.json"
            if not source_context_path.exists():
                raise RuntimeError("solver_qa_integrity_mismatch: frozen dialogue is missing")
            source_context = read_json(source_context_path)
            dialogue = source_context["combined_dialogue"]
            turn_text = {x["turn_id"]:x["text"] for x in dialogue}
            turn_sessions = {x["turn_id"]:str(x.get("session_id") or x["turn_id"].split(":", 1)[0])
                             for x in dialogue}
            local_span_catalog = build_span_catalog(dialogue)
            local_reasoning_contracts = default_reasoning_contracts(("A", "B", "C"))
            generated_qa, qa, accepted_item = select_solver_qa(raw_batch, smoke_report,
                args.solver_qa_id, frozen_turns=turn_text, turn_session_ids=turn_sessions,
                span_catalog=local_span_catalog, reasoning_contracts=local_reasoning_contracts)
            original_qa = qa
        qa_path = args.run_root / "frozen/solver_qa.json"
        frozen_qa_payload = {"source_root":str(source),
                              "source_generated_qa":generated_qa,
                              "source_generated_qa_sha256":sha256_bytes(canonical_json(generated_qa).encode()),
                              "original_qa":original_qa,
                              "original_qa_sha256":sha256_bytes(canonical_json(original_qa).encode()), "qa":qa,
                              "schema_acceptance":({"schema_decision":"synthetic_protocol_only",
                                  "materialized_qa":qa, "target_letter":"B"} if boxed_smoke else
                                  accepted_item if minimal_audit or permutation_diagnostic else
                                  next(x for x in smoke_report["items"] if x["qa_id"] == args.solver_qa_id)),
                              "source_raw_sha256":sha256_file(raw_path),
                              "source_report_sha256":sha256_file(report_path)}
        if minimal_audit:
            frozen_qa_payload["minimal_source_attestation"] = {key:value for key,value in
                minimal_source_integrity.items() if key not in {"raw_batch", "context",
                    "span_catalog", "reasoning_contracts", "materialized_qa", "admission_item"}}
        freeze_json(qa_path, frozen_qa_payload, args.resume)
        solver_artifacts = {"source":source, "raw_path":raw_path, "report_path":report_path,
                            "qa":qa, "original_qa":original_qa, "qa_path":qa_path,
                            "span_catalog":local_span_catalog,
                            "reasoning_contracts":local_reasoning_contracts}
        if minimal_audit:
            frozen_context = minimal_source_integrity["context"]
            context_source_path = source / "frozen/questioner_smoke_context.json"
            catalog_source_path = source / "frozen/questioner_span_catalog.json"
            contracts_source_path = source / "frozen/questioner_reasoning_contracts.json"
            context_path = args.run_root / "frozen/questioner_smoke_context.json"
            catalog_path = args.run_root / "frozen/questioner_span_catalog.json"
            contracts_path = args.run_root / "frozen/questioner_reasoning_contracts.json"
            freeze_bytes(context_path, context_source_path.read_bytes(), args.resume)
            freeze_bytes(catalog_path, catalog_source_path.read_bytes(), args.resume)
            freeze_bytes(contracts_path, contracts_source_path.read_bytes(), args.resume)
            child_turns = [turn for session in frozen_context["accepted_child_sessions"]
                           for turn in session["dialogue_turns"]]
            expected_combined = data["parent"]["dialogue"]["turns"] + child_turns
            if canonical_json(expected_combined) != canonical_json(frozen_context["combined_dialogue"]):
                raise RuntimeError("minimal Solver context differs from frozen Questioner context")
            freeze_json(args.run_root / "frozen_child_dialogue.json", {"turns":child_turns,
                "source_context_sha256":sha256_file(context_source_path)}, args.resume)
            evidence_contexts = build_answer_evidence_contexts(qa,
                span_catalog=local_span_catalog, frozen_dialogue=expected_combined)
            evidence_plan_path = args.run_root / "frozen/evidence_context_plan.json"
            freeze_json(evidence_plan_path, {"answer_span_ids":qa["evidence_span_ids"],
                "answer_turn_ids":evidence_contexts["answer_turn_ids"],
                "evidence_only_turn_ids":[turn["turn_id"] for turn in evidence_contexts["evidence_only"]],
                "evidence_masked_turn_ids":[turn["turn_id"] for turn in evidence_contexts["evidence_masked"]],
                "context_builder_sha256":evidence_contexts["context_builder_sha256"],
                "distractor_support_used":False}, args.resume)
            context_modes = {"no_context":[], "full_context":expected_combined,
                "evidence_only":evidence_contexts["evidence_only"],
                "evidence_masked":evidence_contexts["evidence_masked"]}
            prompt_meta = {}
            for evaluation_mode, turns in context_modes.items():
                rendered = render_solver_prompt(length_tokenizer, qa, turns)
                tokens = len(length_tokenizer.encode(rendered, add_special_tokens=False))
                prompt_meta[evaluation_mode] = {"rendered_prompt_sha256":sha256_bytes(rendered.encode()),
                    "rendered_prompt_tokens":tokens,
                    "reserved_generation_tokens":resolved["reserved_generation_tokens"],
                    "solver_max_model_len":resolved["solver_effective_max_model_len"],
                    "preflight_decision":"pass" if tokens + resolved["reserved_generation_tokens"] <=
                        resolved["solver_effective_max_model_len"] else "reject_context_overflow"}
            if any(value["preflight_decision"] != "pass" for value in prompt_meta.values()):
                raise RuntimeError("minimal Solver context preflight failed")
            prompt_meta_path = args.run_root / "frozen/rendered_prompt_preflight.json"
            freeze_json(prompt_meta_path, prompt_meta, args.resume)
            seed_schedule = {}
            for evaluation_mode, order_ids, slots in (
                    ("no_context", ("original",), range(5)),
                    ("full_context", ("original",), range(5)),
                    ("evidence_only", ("original",), range(3)),
                    ("evidence_masked", ("original",), range(3)),
                    ("option_permutation", ("perm1", "perm2"), range(5))):
                for order_id in order_ids:
                    key = f"{evaluation_mode}:{order_id}"
                    seed_schedule[key] = {str(slot):[
                        int(hashlib.sha256(
                            f"{qa['qa_id']}:{evaluation_mode}:{order_id}:{slot}:{retry}".encode()
                        ).hexdigest()[:8], 16) for retry in (0, 1)] for slot in slots}
            seed_path = args.run_root / "frozen/seed_schedule.json"
            freeze_json(seed_path, seed_schedule, args.resume)
            solver_artifacts.update({"minimal_source_integrity":minimal_source_integrity,
                "context_source_path":context_source_path, "context_path":context_path,
                "catalog_path":catalog_path, "contracts_path":contracts_path,
                "evidence_plan_path":evidence_plan_path, "prompt_meta":prompt_meta,
                "prompt_meta_path":prompt_meta_path, "seed_path":seed_path})
        if permutation_diagnostic:
            from .orchestrator import CanaryOrchestrator
            context_source_path = Path(permutation_source_integrity["source_context_path"])
            catalog_source_path = Path(permutation_source_integrity["source_catalog_path"])
            contracts_source_path = Path(permutation_source_integrity["source_contracts_path"])
            context_path = args.run_root / "frozen/questioner_smoke_context.json"
            catalog_path = args.run_root / "frozen/questioner_span_catalog.json"
            contracts_path = args.run_root / "frozen/questioner_reasoning_contracts.json"
            freeze_bytes(context_path, context_source_path.read_bytes(), args.resume)
            freeze_bytes(catalog_path, catalog_source_path.read_bytes(), args.resume)
            freeze_bytes(contracts_path, contracts_source_path.read_bytes(), args.resume)
            original_path = args.run_root / "frozen/original_no_context_k3.json"
            freeze_json(original_path, permutation_source_integrity["original_no_context"], args.resume)
            prompt_meta = {}
            for shift, order_id in ((1, "perm1"), (2, "perm2")):
                options = CanaryOrchestrator._permuted(qa["options"], shift)
                rendered = render_solver_prompt(length_tokenizer, {**qa, "options":options}, [])
                tokens = len(length_tokenizer.encode(rendered, add_special_tokens=False))
                prompt_meta[order_id] = {"rendered_prompt_sha256":sha256_bytes(rendered.encode()),
                    "rendered_prompt_tokens":tokens,
                    "reserved_generation_tokens":resolved["reserved_generation_tokens"],
                    "solver_max_model_len":resolved["solver_effective_max_model_len"],
                    "preflight_decision":"pass" if tokens + resolved["reserved_generation_tokens"] <=
                        resolved["solver_effective_max_model_len"] else "reject_context_overflow"}
            if any(value["preflight_decision"] != "pass" for value in prompt_meta.values()):
                raise RuntimeError("no-context permutation preflight failed")
            prompt_meta_path = args.run_root / "frozen/rendered_prompt_preflight.json"
            freeze_json(prompt_meta_path, prompt_meta, args.resume)
            seed_schedule = {f"no_context:{order_id}":{str(slot):[
                int(hashlib.sha256(
                    f"{qa['qa_id']}:no_context:{order_id}:{slot}:{retry}".encode()
                ).hexdigest()[:8], 16) for retry in (0, 1)] for slot in range(3)}
                for order_id in ("perm1", "perm2")}
            seed_path = args.run_root / "frozen/seed_schedule.json"
            freeze_json(seed_path, seed_schedule, args.resume)
            solver_artifacts.update({"permutation_source_integrity":permutation_source_integrity,
                "context_path":context_path, "catalog_path":catalog_path,
                "contracts_path":contracts_path, "original_path":original_path,
                "prompt_meta":prompt_meta, "prompt_meta_path":prompt_meta_path,
                "seed_path":seed_path})
        if boxed_smoke:
            rendered = render_solver_prompt(length_tokenizer, qa, [])
            prompt_meta_path = args.run_root / "frozen/rendered_prompt_preflight.json"
            freeze_json(prompt_meta_path, {"rendered_prompt_sha256":sha256_bytes(rendered.encode()),
                "rendered_prompt_tokens":len(length_tokenizer.encode(rendered, add_special_tokens=False)),
                "reserved_generation_tokens":resolved["reserved_generation_tokens"],
                "solver_max_model_len":resolved["solver_effective_max_model_len"],
                "contains_boxed_only_instruction":"<box>A</box>" in rendered}, args.resume)
            solver_artifacts["prompt_meta_path"] = prompt_meta_path
        if args.smoke_stage == "solver_adaptive_item":
            context_source_path = source / "frozen/questioner_smoke_context.json"
            frozen_context = read_json(context_source_path)
            context_path = args.run_root / "frozen/questioner_smoke_context.json"
            freeze_bytes(context_path, context_source_path.read_bytes(), args.resume)
            child_turns = [turn for session in frozen_context["accepted_child_sessions"] for turn in session["dialogue_turns"]]
            expected_combined = data["parent"]["dialogue"]["turns"] + child_turns
            if canonical_json(expected_combined) != canonical_json(frozen_context["combined_dialogue"]):
                raise RuntimeError("Questioner frozen context differs from exact parent plus accepted child sessions")
            freeze_json(args.run_root / "frozen_child_dialogue.json", {"turns":child_turns,
                "source_context_sha256":sha256_file(context_source_path)}, args.resume)
            audit_path = args.run_root / "reports/batch0_item2_question_audit.md"
            freeze_text(audit_path, item2_question_audit(original_qa, frozen_context), args.resume)
            no_prompt = render_solver_prompt(length_tokenizer, qa, [])
            full_prompt = render_solver_prompt(length_tokenizer, qa, expected_combined)
            prompt_meta = {"no_context_rendered_prompt_sha256":sha256_bytes(no_prompt.encode()),
                "full_context_rendered_prompt_sha256":sha256_bytes(full_prompt.encode()),
                "no_context_rendered_prompt_tokens":len(length_tokenizer.encode(no_prompt, add_special_tokens=False)),
                "full_context_rendered_prompt_tokens":len(length_tokenizer.encode(full_prompt, add_special_tokens=False)),
                "reserved_generation_tokens":resolved["reserved_generation_tokens"],
                "solver_max_model_len":resolved["solver_effective_max_model_len"]}
            prompt_meta["full_context_preflight_decision"] = ("pass" if
                prompt_meta["full_context_rendered_prompt_tokens"] + prompt_meta["reserved_generation_tokens"] <=
                prompt_meta["solver_max_model_len"] else "reject_context_overflow")
            prompt_meta_path = args.run_root / "frozen/rendered_prompt_preflight.json"
            freeze_json(prompt_meta_path, prompt_meta, args.resume)
            seed_schedule = {mode:{str(slot):[
                int(hashlib.sha256(f"{qa['qa_id']}:{mode}:original:{slot}:{retry}".encode()).hexdigest()[:8], 16)
                for retry in (0,1)] for slot in slots} for mode,slots in
                (("no_context", range(5)), ("full_context", range(5)))}
            seed_path = args.run_root / "frozen/seed_schedule.json"
            freeze_json(seed_path, seed_schedule, args.resume)
            solver_artifacts.update({"context_source_path":context_source_path, "context_path":context_path,
                "audit_path":audit_path, "prompt_meta":prompt_meta, "prompt_meta_path":prompt_meta_path,
                "seed_path":seed_path})
        if args.smoke_stage == "solver_grounding_extension":
            prior_root = args.solver_prior_root.resolve()
            prior_report_path = prior_root / "reports/local_solver_s1_smoke.json"
            prior = read_json(prior_report_path)
            slots = [x.get("target_valid_slot") for x in prior.get("rollouts") or []]
            if (prior.get("qa_id") != qa["qa_id"] or prior.get("evaluation_mode") != "no_context" or
                    prior.get("obtained_valid_rollouts") != 3 or prior.get("verified_rewards") != [1, 1, 0] or
                    slots != [0, 1, 2]):
                raise RuntimeError("prior S1-v2 grounding state does not match the frozen extension contract")
            prior_path = args.run_root / "frozen/prior_grounding_k3.json"
            freeze_json(prior_path, {"source_root":str(prior_root),
                "source_report_sha256":sha256_file(prior_report_path), "qa_id":prior["qa_id"],
                "verified_rewards":prior["verified_rewards"], "rollouts":prior["rollouts"]}, args.resume)
            solver_artifacts.update({"prior_root":prior_root, "prior_report_path":prior_report_path,
                                     "prior":prior, "prior_path":prior_path})
    manifest = {
        "run_id": args.run_root.name, "mode": resolved["mode"],
        "admission_mode":args.admission_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_value(project, "rev-parse", "HEAD"),
        "project_git_dirty": bool(git_value(project, "status", "--porcelain")),
        "project_diff_sha256": project_diff_sha256(project),
        "resolved_config_sha256": sha256_file(config_path),
        "realizer_prompt_sha256": prompt_hashes["realizer"],
        "questioner_prompt_sha256": prompt_hashes["questioner"],
        "questioner_prompt_version":QUESTIONER_PROMPT_VERSION,
        "questioner_schema_version":QUESTIONER_SCHEMA_VERSION,
        "span_catalog_version":SPAN_CATALOG_VERSION,
        "reasoning_contract_version":REASONING_CONTRACT_VERSION,
        "responder_prompt_sha256": prompt_hashes["responder"],
        "mcq_parser_version": PARSER_VERSION,
        "mcq_parser_source_sha256": sha256_file(Path(__file__).with_name("core.py")),
        "outline_sha256": sha256_file(outline_path),
        "parent_anchor_snapshot_sha256":sha256_file(snapshot_path),
        "parent_dialogue_sha256": data["anchors"]["dialogue_hash"],
        "parent_graph_sha256": data["anchors"].get("parent_graph_hash", ""),
        "solver_model_path": str(solver), "solver_model_identifier": solver.name,
        "solver_checkpoint_max_position_embeddings": model_config["max_position_embeddings"],
        "solver_max_model_len": resolved["solver_effective_max_model_len"],
        "tokenizer_path": str(tokenizer),
        "tokenizer_config_sha256": sha256_file(tokenizer / "tokenizer_config.json"),
        "challenger_model_identifier": live_config.model if live_config else "mock-scripted-v1" if args.backend_mode == "mock" else "NONE_LOCAL_ONLY" if args.backend_mode == "local" else "UNSET_DRY_RUN",
        "backend_mode": args.backend_mode,
        "sampling_parameters": ({"temperature":live_config.temperature,
            "max_tokens":resolved["challenger_semantic_repair_max_tokens"]
                if args.smoke_stage == "semantic_leakage_repair" else
                live_config.questioner_max_tokens if args.smoke_stage in
                ("questioner", "questioner_feedback", "questioner_span_ids_feedback",
                 )
                else live_config.realizer_max_tokens}
            if live_config else {"temperature":resolved["solver_temperature"], "top_p":resolved["solver_top_p"],
                "max_tokens":resolved["solver_max_new_tokens"]} if args.backend_mode == "local" else {}),
        "mcq_valid_rollout_policy": {"max_parse_retries_per_target": 1,
            "no_context_initial_valid_k": 3, "no_context_boundary_extra_valid_k": 2,
            "full_context_valid_k": 5, "permutation_valid_k_per_order": 5},
        "questioner_batch_policy": ({"generations":1, "mcqs_per_generation":3, "total_candidates":3,
            "model_output":"unlabeled_correct_answer_plus_three_distractors",
            "option_binding":"local_option_binding_v1",
            "evidence_binding":"local_span_catalog_v1",
            "reasoning_and_difficulty":"local_contract_derivation_v1"}
            if args.backend_mode == "live" and args.smoke_stage in
                ("questioner", "questioner_feedback", "questioner_span_ids_feedback") else
            {"task":"semantic_leakage_repair", "normal_requests":1,
             "maximum_format_retries":1, "items_per_request":1}
            if args.backend_mode == "hybrid" else
            {"generations": 4, "mcqs_per_generation": 3, "total_candidates": 12,
             "option_binding":"local_option_binding_v1",
             "evidence_binding":"local_span_catalog_v1",
             "reasoning_and_difficulty":"local_contract_derivation_v1"}),
        "paid_call_budgets": ({"max_total_paid_api_generations":resolved["smoke_max_paid_generations"],
            "max_realizer_generations":resolved["smoke_max_realizer_generations"],
            "max_questioner_generations":resolved["max_questioner_generations"],
            "max_post_freeze_memory_generations":0, "max_verifier_generations":0}
            if args.backend_mode == "live" else
            {"max_total_paid_api_generations":resolved["max_total_paid_api_generations"],
             "max_semantic_repair_generations":resolved["max_semantic_repair_generations"],
             "max_realizer_generations":0, "max_questioner_generations":0,
             "max_verifier_generations":0} if args.backend_mode == "hybrid" else
            {k: v for k, v in resolved.items() if k.startswith("max_") and "generations" in k}),
        "spell_repo_status": "available", "spell_repo_path": str(spell_repo),
        "spell_commit": spell_commit, "spell_dirty": spell_dirty,
        "input_provenance": data["provenance"],
        **source_snapshot,
        "orchestrator_source_sha256": sha256_file(Path(__file__).with_name("orchestrator.py")),
        "validation_source_sha256":sha256_file(Path(__file__).with_name("validation.py")),
        "materializer_source_sha256":sha256_file(Path(__file__).with_name("validation.py")),
        "questioner_prompt_builder_source_sha256":sha256_file(Path(__file__).with_name("prompts.py")),
        "questioner_contract_source_sha256":sha256_file(Path(__file__).with_name("questioner_contract.py")),
        "ledger_source_sha256":sha256_file(Path(__file__).with_name("ledger.py")),
        "run_canary_source_sha256":sha256_file(Path(__file__)),
        "backend_source_sha256": sha256_file(Path(__file__).with_name("mock_backends.py")) if args.backend_mode == "mock" else "",
        "live_backend_source_sha256": sha256_file(Path(__file__).with_name("live_backends.py")) if args.backend_mode in ("live", "hybrid") else "",
        "api_config_path": str(args.api_config.resolve()) if args.backend_mode in ("live", "hybrid") else "",
        "api_config_sha256": sha256_file(args.api_config) if args.backend_mode in ("live", "hybrid") else "",
    }
    if args.admission_mode == SPELL_STYLE_MINIMAL_MODE:
        manifest.update({"minimal_admission_mode":SPELL_STYLE_MINIMAL_MODE,
            "strict_admission_mode_preserved":STRICT_ADMISSION_MODE,
            "minimal_normalization_version":MINIMAL_NORMALIZATION_VERSION,
            "minimal_materializer_version":MINIMAL_MATERIALIZER_VERSION,
            "minimal_replay_version":MINIMAL_REPLAY_VERSION,
            "minimal_validator_source_sha256":sha256_file(
                Path(__file__).with_name("minimal_admission.py")),
            "normalization_function_source_sha256":sha256_file(
                Path(__file__).with_name("minimal_admission.py")),
            "minimal_replay_source_sha256":sha256_file(
                Path(__file__).with_name("minimal_replay.py"))})
    if questioner_artifacts:
        span_ids_only_smoke = args.smoke_stage == "questioner_span_ids_feedback"
        feedback_smoke = args.smoke_stage in ("questioner_feedback",
                                              "questioner_span_ids_feedback")
        manifest.update({"questioner_prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION
            if span_ids_only_smoke else QUESTIONER_FEEDBACK_PROMPT_VERSION
            if feedback_smoke else QUESTIONER_PROMPT_VERSION,
            "questioner_prompt_sha256":span_ids_only_questioner_template_hash()
            if span_ids_only_smoke else feedback_questioner_template_hash()
            if feedback_smoke else prompt_hashes["questioner"],
            "questioner_prompt_builder_source_sha256":sha256_file(Path(__file__).with_name("prompts.py")),
            "rendered_questioner_prompt_sha256":sha256_bytes(questioner_artifacts["rendered"].encode("utf-8")),
            "frozen_dialogue_sha256":sha256_file(questioner_artifacts["frozen_path"]),
            "questioner_source_root":str(questioner_artifacts["source"]),
            "accepted_session_01_sha256":sha256_file(questioner_artifacts["session_paths"][0]),
            "accepted_session_02_sha256":sha256_file(questioner_artifacts["session_paths"][1]),
            "source_run_id":questioner_artifacts["source_integrity"]["source_run_id"],
            "source_run_manifest_sha256":questioner_artifacts["source_integrity"]["source_manifest_sha256"],
            "source_accepted_index_sha256":questioner_artifacts["source_integrity"]["source_index_sha256"],
            "source_post_run_provenance_sha256":questioner_artifacts["source_integrity"]["source_provenance_sha256"],
            "source_accepted_session_sha256":questioner_artifacts["source_integrity"]["source_session_sha256"],
            "legal_turn_ids_sha256":sha256_bytes(canonical_json(list(questioner_artifacts["valid_turn_ids"])) .encode()),
            "legal_session_ids_sha256":sha256_bytes(canonical_json(list(questioner_artifacts["valid_session_ids"])) .encode()),
            "target_gold_letters":list(questioner_artifacts["letters"]),
            "span_catalog_version":SPAN_CATALOG_VERSION,
            "span_catalog_sha256":span_catalog_sha256(questioner_artifacts["span_catalog"]),
            "span_catalog_file_sha256":sha256_file(questioner_artifacts["span_catalog_path"]),
            "reasoning_contract_version":REASONING_CONTRACT_VERSION,
            "reasoning_contract_sha256":reasoning_contract_sha256(
                questioner_artifacts["reasoning_contracts"]),
            "reasoning_contract_file_sha256":sha256_file(
                questioner_artifacts["reasoning_contracts_path"]),
            "questioner_schema_version":QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION
                if span_ids_only_smoke else QUESTIONER_SCHEMA_VERSION,
            "questioner_prompt_preflight":questioner_artifacts["prompt_preflight"],
            "questioner_prompt_preflight_sha256":sha256_file(
                questioner_artifacts["prompt_preflight_path"])})
        if feedback_smoke:
            manifest.update({"solver_feedback_version":"solver_failure_feedback_v1",
                "solver_feedback_sha256":sha256_file(questioner_artifacts["feedback_path"]),
                "solver_feedback_sources":[{key:value for key,value in item.items()
                    if key != "feedback"} for item in
                    questioner_artifacts["feedback_attestations"]],
                "minimal_admission_policy":"spell_style_minimal_v1_unchanged",
                "questioner_output_profile":SPAN_IDS_ONLY_SCHEMA_PROFILE
                    if span_ids_only_smoke else LEGACY_OPTIONAL_SCHEMA_PROFILE,
                "strict_v7_admission_preserved":True,
                "questioner_format_retry_policy":{
                    "maximum_format_retries":1,
                    "trigger":"json_invalid_and_raw_object_count_zero",
                    "retry_mode":"full_batch_regeneration",
                    "format_retry_suffix_version":QUESTIONER_FORMAT_RETRY_VERSION,
                    "format_retry_suffix_sha256":questioner_format_retry_suffix_hash(),
                    "malformed_response_in_retry_context":False}})
    if minimal_replay_artifacts:
        source_attestation = minimal_replay_artifacts["attestation"]
        manifest.update({"source_questioner_run":str(minimal_replay_artifacts["source"]),
            "source_questioner_manifest_sha256":source_attestation["source_manifest_sha256"],
            "source_questioner_provenance_sha256":source_attestation["source_provenance_sha256"],
            "source_questioner_ledger_sha256":source_attestation["source_ledger_sha256"],
            "source_questioner_raw_sha256":source_attestation["source_raw_sha256"],
            "frozen_dialogue_sha256":sha256_file(
                minimal_replay_artifacts["copied_paths"]["context"]),
            "span_catalog_sha256":span_catalog_sha256(minimal_replay_artifacts["catalog"]),
            "span_catalog_file_sha256":sha256_file(
                minimal_replay_artifacts["copied_paths"]["span_catalog"]),
            "reasoning_contract_sha256":reasoning_contract_sha256(
                minimal_replay_artifacts["contracts"]),
            "reasoning_contract_file_sha256":sha256_file(
                minimal_replay_artifacts["copied_paths"]["reasoning_contracts"]),
            "dual_admission_replay_report_sha256":sha256_file(
                minimal_replay_artifacts["report_path"]),
            "dual_admission_replay_markdown_sha256":sha256_file(
                minimal_replay_artifacts["markdown_path"]),
            "strict_v7_accepted_count":minimal_replay_artifacts["report"][
                "strict_v7_accepted_count"],
            "minimal_accepted_count":minimal_replay_artifacts["report"][
                "minimal_accepted_count"]})
    if repair_artifacts:
        source = repair_artifacts["source"]
        paths = repair_artifacts["frozen_paths"]
        manifest.update({"semantic_repair_version":SEMANTIC_LEAKAGE_REPAIR_VERSION,
            "semantic_repair_reason":SEMANTIC_LEAKAGE_REPAIR_REASON,
            "semantic_repair_prompt_version":SEMANTIC_LEAKAGE_REPAIR_PROMPT_VERSION,
            "semantic_repair_prompt_sha256":semantic_leakage_repair_template_hash(),
            "semantic_repair_rendered_prompt_sha256":sha256_bytes(
                repair_artifacts["rendered"].encode()),
            "semantic_repair_schema_version":SEMANTIC_LEAKAGE_REPAIR_SCHEMA_VERSION,
            "semantic_repair_schema_sha256":sha256_bytes(canonical_json(
                SEMANTIC_LEAKAGE_REPAIR_JSON_SCHEMA).encode()),
            "semantic_repair_source_sha256":sha256_file(
                Path(__file__).with_name("semantic_repair.py")),
            "v8_generation_prompt_version":QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
            "v8_generation_prompt_sha256":span_ids_only_questioner_template_hash(),
            "format_retry_suffix_version":QUESTIONER_FORMAT_RETRY_VERSION,
            "format_retry_suffix_sha256":semantic_repair_format_retry_suffix_hash(),
            "repair_attempt_limit_per_item":1, "format_retry_limit":1,
            "parent_qa_id":args.solver_qa_id,
            "repaired_qa_id":repair_artifacts["repaired_qa_id"],
            "repair_index":1, "target_answer_letter":source["target_letter"],
            "source_solver_root":source["source_root"],
            "source_solver_manifest_sha256":source["source_manifest_sha256"],
            "source_solver_report_sha256":source["source_report_sha256"],
            "source_solver_provenance_sha256":source["source_provenance_sha256"],
            "source_solver_ledger_aggregate_sha256":source[
                "solver_ledger_aggregate_sha256"],
            "source_qa_wrapper_sha256":source["source_qa_wrapper_sha256"],
            "source_materialized_qa_sha256":source["source_materialized_qa_sha256"],
            "source_diagnostic_root":source["diagnostic_root"],
            "source_diagnostic_manifest_sha256":source["diagnostic_manifest_sha256"],
            "source_diagnostic_report_sha256":source["diagnostic_report_sha256"],
            "source_diagnostic_provenance_sha256":source["diagnostic_provenance_sha256"],
            "source_diagnostic_ledger_aggregate_sha256":source[
                "diagnostic_ledger_aggregate_sha256"],
            "frozen_dialogue_sha256":sha256_file(paths["context"]),
            "span_catalog_sha256":span_catalog_sha256(source["span_catalog"]),
            "span_catalog_file_sha256":sha256_file(paths["span_catalog"]),
            "reasoning_contract_sha256":reasoning_contract_sha256(
                source["reasoning_contracts"]),
            "source_attestation_sha256":sha256_file(paths["source_attestation"]),
            "semantic_repair_prompt_preflight":repair_artifacts["preflight"],
            "semantic_repair_max_tokens":resolved[
                "challenger_semantic_repair_max_tokens"],
            "semantic_repair_prompt_preflight_sha256":sha256_file(paths[
                "prompt_preflight"]),
            "seed_schedule_sha256":sha256_file(paths["seed_schedule"]),
            "vllm_backend_source_sha256":sha256_file(Path(__file__).with_name(
                "vllm_backends.py")),
            "solver_inference_protocol":{"enable_thinking":False,
                "add_generation_prompt":True, "tokenizer_path":str(tokenizer),
                "temperature":resolved["solver_temperature"],
                "top_p":resolved["solver_top_p"],
                "max_tokens":resolved["solver_max_new_tokens"],
                "max_model_len":resolved["solver_effective_max_model_len"],
                "responder_output_format":"<box>[A-D]</box>"}})
    if solver_artifacts:
        extension = args.smoke_stage == "solver_grounding_extension"
        adaptive = args.smoke_stage == "solver_adaptive_item"
        boxed_smoke = args.smoke_stage == "solver_boxed_output"
        minimal_audit = args.smoke_stage == "solver_minimal_full_audit"
        permutation_diagnostic = args.smoke_stage == "solver_no_context_permutation_diagnostic"
        manifest.update({"solver_smoke_stage":"bad_ground_no_context_permutation_diagnostic" if permutation_diagnostic else "minimal_grounding_full_evidence_permutation" if minimal_audit else "boxed_output_no_context_valid_k3" if boxed_smoke else "adaptive_item_grounding_then_conditional_full_context" if adaptive else "S1_grounding_extension_valid_k2" if extension else "S1_no_context_valid_k3",
            "frozen_solver_qa_sha256":sha256_file(solver_artifacts["qa_path"]),
            "source_questioner_raw_sha256":sha256_file(solver_artifacts["raw_path"]),
            "source_questioner_report_sha256":sha256_file(solver_artifacts["report_path"]),
            "vllm_backend_source_sha256":sha256_file(Path(__file__).with_name("vllm_backends.py")),
            "solver_model_resolved_path":str(solver.resolve()), "solver_model_config_sha256":sha256_file(solver / "config.json"),
            "solver_inference_protocol":{"enable_thinking":False, "add_generation_prompt":True,
                "tokenizer_path":str(tokenizer), "temperature":resolved["solver_temperature"],
                "top_p":resolved["solver_top_p"], "max_tokens":resolved["solver_max_new_tokens"],
                "max_model_len":resolved["solver_effective_max_model_len"],
                "stop_sequences":[], "responder_output_format":"<box>[A-D]</box>",
                "allow_plain_letter":False, "allow_option_text":False, "allow_free_text":False},
            "paid_call_budgets":{"max_total_paid_api_generations":0, "max_realizer_generations":0,
                "max_questioner_generations":0, "max_verifier_generations":0}})
        if boxed_smoke:
            manifest.update({"synthetic_protocol_smoke":True,
                "rendered_prompt_preflight_sha256":sha256_file(solver_artifacts["prompt_meta_path"])})
        manifest["solver_max_model_len"] = resolved["solver_effective_max_model_len"]
        if extension:
            manifest.update({"prior_grounding_state_sha256":sha256_file(solver_artifacts["prior_path"]),
                "prior_s1_report_sha256":sha256_file(solver_artifacts["prior_report_path"]),
                "prior_verified_rewards":solver_artifacts["prior"]["verified_rewards"],
                "extension_slot_ids":[3,4]})
        if adaptive:
            manifest.update({"frozen_context_sha256":sha256_file(solver_artifacts["context_path"]),
                "source_frozen_context_sha256":sha256_file(solver_artifacts["context_source_path"]),
                "offline_question_audit_sha256":sha256_file(solver_artifacts["audit_path"]),
                "rendered_prompt_preflight_sha256":sha256_file(solver_artifacts["prompt_meta_path"]),
                **solver_artifacts["prompt_meta"], "seed_schedule_sha256":sha256_file(solver_artifacts["seed_path"]),
                "adaptive_policy":{"initial_valid_k":3, "boundary_extension_valid_k":2,
                    "full_context_valid_k":5, "max_parse_retries_per_target":1}})
        if minimal_audit:
            source_integrity = solver_artifacts["minimal_source_integrity"]
            manifest.update({"source_minimal_replay_root":str(solver_artifacts["source"]),
                "source_minimal_replay_manifest_sha256":source_integrity["source_manifest_sha256"],
                "source_minimal_replay_provenance_sha256":source_integrity["source_provenance_sha256"],
                "source_minimal_replay_report_sha256":source_integrity["source_replay_report_sha256"],
                "source_questioner_raw_sha256":source_integrity["source_raw_sha256"],
                "materialized_qa_sha256":source_integrity["materialized_qa_sha256"],
                "frozen_dialogue_sha256":sha256_file(solver_artifacts["context_path"]),
                "span_catalog_sha256":span_catalog_sha256(solver_artifacts["span_catalog"]),
                "reasoning_contract_sha256":reasoning_contract_sha256(
                    solver_artifacts["reasoning_contracts"]),
                "evidence_context_plan_sha256":sha256_file(solver_artifacts["evidence_plan_path"]),
                "rendered_prompt_preflight_sha256":sha256_file(solver_artifacts["prompt_meta_path"]),
                "seed_schedule_sha256":sha256_file(solver_artifacts["seed_path"]),
                "final_decision_policy":"minimal_frontier_merge_v1",
                "original_full_context_k5_reuse_required":True})
        if permutation_diagnostic:
            source_integrity = solver_artifacts["permutation_source_integrity"]
            manifest.update({"diagnostic_version":NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION,
                "source_bad_ground_solver_root":str(solver_artifacts["source"]),
                "source_bad_ground_manifest_sha256":source_integrity["source_manifest_sha256"],
                "source_bad_ground_provenance_sha256":source_integrity["source_provenance_sha256"],
                "source_bad_ground_report_sha256":source_integrity["source_report_sha256"],
                "source_bad_ground_qa_wrapper_sha256":source_integrity[
                    "source_qa_wrapper_sha256"],
                "source_original_no_context_k3_sha256":sha256_file(
                    solver_artifacts["original_path"]),
                "materialized_qa_sha256":source_integrity["source_materialized_qa_sha256"],
                "frozen_dialogue_sha256":sha256_file(solver_artifacts["context_path"]),
                "span_catalog_sha256":span_catalog_sha256(solver_artifacts["span_catalog"]),
                "reasoning_contract_sha256":reasoning_contract_sha256(
                    solver_artifacts["reasoning_contracts"]),
                "rendered_prompt_preflight_sha256":sha256_file(
                    solver_artifacts["prompt_meta_path"]),
                "seed_schedule_sha256":sha256_file(solver_artifacts["seed_path"]),
                "original_no_context_k3_reuse_required":True,
                "new_permutation_orders":["perm1", "perm2"],
                "valid_k_per_new_order":3,
                "authoritative_bad_ground_decision_mutable":False})
    if args.resume and manifest_path.exists():
        frozen_manifest = read_json(manifest_path)
        old = {key:value for key,value in frozen_manifest.items() if key != "created_at"}
        new = {key:value for key,value in manifest.items() if key != "created_at"}
        if canonical_json(old) != canonical_json(new):
            changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
            raise RuntimeError(f"resume provenance differs from frozen manifest: {changed}")
        manifest = frozen_manifest
    else:
        atomic_json(manifest_path, manifest)
    audit = build_input_dry_run_audit(parent_turns=len(data["parent"]["dialogue"]["turns"]),
        parent_tokens=parent_tokens, lineage=args.lineage,
        reserved_generation_tokens=resolved["reserved_generation_tokens"],
        checkpoint_max_model_len=model_config["max_position_embeddings"],
        effective_max_model_len=resolved["solver_effective_max_model_len"],
        spell_commit=spell_commit, spell_dirty=spell_dirty)
    freeze_json(args.run_root / "reports/input_dry_run_audit.json", audit, args.resume)
    if args.smoke_stage == "questioner_minimal_replay":
        write_post_run_provenance(args.run_root, project, manifest)
        print(json.dumps({"manifest":str(manifest_path),
            "spell_style_minimal_replay":minimal_replay_artifacts["report"]}, indent=2))
        return 0
    if args.backend_mode == "mock":
        from .mock_backends import ScriptedMockQuestioner, ScriptedMockRealizer, ScriptedMockSolver
        from .orchestrator import CanaryOrchestrator
        orchestrator = CanaryOrchestrator(run_root=args.run_root, manifest=manifest, config=resolved,
            data=data, tokenizer=length_tokenizer, realizer=ScriptedMockRealizer(),
            questioner=ScriptedMockQuestioner(args.mock_questioner_shortfall_batch,
                                               reject_all=args.mock_questioner_all_invalid),
            solver=ScriptedMockSolver(), backend_mode="mock")
        report = orchestrator.run(args.mock_crash_before_session)
        write_post_run_provenance(args.run_root, project, manifest)
        print(json.dumps({"manifest": str(manifest_path), "mock_e2e": report}, indent=2))
    elif args.backend_mode == "live":
        from .live_backends import LiveQuestionerBackend, LiveRealizerBackend, OpenAICompatibleChallengerClient
        from .orchestrator import CanaryOrchestrator
        client = OpenAICompatibleChallengerClient(live_config)
        orchestrator = CanaryOrchestrator(run_root=args.run_root, manifest=manifest, config=resolved,
            data=data, tokenizer=length_tokenizer,
            realizer=None if args.smoke_stage in (
                "questioner", "questioner_feedback", "questioner_span_ids_feedback")
                else LiveRealizerBackend(client),
            questioner=LiveQuestionerBackend(client) if args.smoke_stage in
                ("questioner", "questioner_feedback", "questioner_span_ids_feedback") else None,
            solver=None, backend_mode="live")
        if args.smoke_stage in ("questioner", "questioner_feedback",
                                "questioner_span_ids_feedback"):
            result = orchestrator.question_smoke(questioner_artifacts["frozen"]["combined_dialogue"],
                questioner_artifacts["valid_turn_ids"], questioner_artifacts["valid_session_ids"],
                questioner_artifacts["letters"], span_catalog=questioner_artifacts["span_catalog"],
                reasoning_contracts=questioner_artifacts["reasoning_contracts"],
                rendered_prompt=questioner_artifacts["rendered"],
                prompt_protocol=QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION if
                    args.smoke_stage == "questioner_span_ids_feedback" else
                    QUESTIONER_FEEDBACK_PROMPT_VERSION if
                    args.smoke_stage == "questioner_feedback" else QUESTIONER_PROMPT_VERSION,
                max_format_retries=1 if args.smoke_stage in
                    ("questioner_feedback", "questioner_span_ids_feedback") else 0,
                solver_feedback_sha256=sha256_file(questioner_artifacts["feedback_path"])
                    if questioner_artifacts.get("feedback_path") else None,
                raw_schema_mode=SPAN_IDS_ONLY_SCHEMA_PROFILE if
                    args.smoke_stage == "questioner_span_ids_feedback" else
                    LEGACY_OPTIONAL_SCHEMA_PROFILE,
                questioner_schema_version=QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION if
                    args.smoke_stage == "questioner_span_ids_feedback" else
                    QUESTIONER_SCHEMA_VERSION)
            state = orchestrator.ledger.read(result["final_request_id"]) or {}
            result.update({"physical_paid_generations":orchestrator.ledger.summary()["paid_api_generation_count"],
                "request_status":state.get("status"), "usage":(state.get("response_metadata") or {}).get("usage"),
                "model_identifier":state.get("model_identifier"), "realizer_generations":0,
                "solver_generations":0, "vllm_generations":0, "verifier_generations":0, "training_runs":0,
                "feedback_guided":args.smoke_stage in
                    ("questioner_feedback", "questioner_span_ids_feedback"),
                "solver_feedback_sha256":sha256_file(questioner_artifacts["feedback_path"])
                    if questioner_artifacts.get("feedback_path") else None})
            atomic_json(args.run_root / "reports/live_questioner_smoke.json", result)
            write_post_run_provenance(args.run_root, project, manifest)
            print(json.dumps({"manifest":str(manifest_path), "live_questioner_smoke":result}, indent=2))
            return 0
        continuity = args.smoke_stage == "realizer_continuity"
        target_sessions = 2 if continuity else 1
        sessions = orchestrator._accepted_sessions()
        if len(sessions) < target_sessions:
            sessions = orchestrator.realize(request_limit=1)
        current_id = f"realizer-s{min(len(sessions), target_sessions):02d}-a01"
        state = orchestrator.ledger.read(current_id) or {}
        smoke = {"mode":mode, "physical_paid_generations":orchestrator.ledger.summary()["paid_api_generation_count"],
            "accepted_session_count":len(sessions), "request_id":current_id,
            "request_status":state.get("status"), "final_decision":state.get("final_decision"),
            "raw_response_path":state.get("raw_response_path"), "model_identifier":state.get("model_identifier"),
            "usage":(state.get("response_metadata") or {}).get("usage"),
            "questioner_generations":0, "solver_generations":0, "verifier_generations":0,
            "vllm_generations":0, "training_runs":0}
        if continuity:
            session1 = sessions[0] if sessions else {}
            session2 = sessions[1] if len(sessions) > 1 else {}
            prompt2_path = args.run_root / "raw/prompts/realizer-s02-a01.json"
            prompt2 = (read_json(prompt2_path).get("prompt", "") if prompt2_path.exists() else "")
            ids = [turn.get("turn_id") for session in sessions for turn in session.get("dialogue_turns") or []]
            refs2 = set((session2.get("continued_unresolved_hook_ids") or []) +
                        (session2.get("resolved_hook_ids") or []))
            active_before = {x.get("hook_id") for x in session1.get("new_unresolved_hooks") or [] if x.get("hook_id")}
            hook_audit = audit_continuity_hooks(active_before,
                set(session2.get("continued_unresolved_hook_ids") or []),
                set(session2.get("resolved_hook_ids") or []))
            checks = {"session_1_assigned_h0001": any(x.get("hook_id") == "H0001" for x in session1.get("new_unresolved_hooks") or []),
                "session_2_prompt_contains_h0001": "H0001" in prompt2,
                "session_2_references_h0001": "H0001" in refs2,
                "all_references_authorized":hook_audit["all_references_authorized"],
                "continues_or_resolves_existing_hook":hook_audit["continues_or_resolves_existing_hook"],
                "no_unknown_hook_reference":hook_audit["no_unknown_hook_reference"],
                "continued_resolved_disjoint":hook_audit["continued_resolved_disjoint"],
                "recent_session_full_text_passed": bool(session1.get("dialogue_turns")) and
                    all(turn.get("text", "") in prompt2 for turn in session1.get("dialogue_turns") or []),
                "continuity_summary_passed": 'accepted_session_count' in prompt2,
                "active_plans_passed": 'active_plans' in prompt2,
                "superseded_state_passed": 'superseded_information_do_not_treat_as_active' in prompt2,
                "global_turn_ids_monotonic": ids == [f"C{i:04d}" for i in range(1, len(ids) + 1)],
                "parent_unchanged": data["provenance"].get("dialogue_hash_verified") is True}
            smoke.update({"target_session_count":2, "continuity_checks":checks,
                          "dynamic_hook_audit":hook_audit,
                          "continuity_complete":len(sessions) == 2 and all(checks.values())})
        atomic_json(args.run_root / "reports/live_realizer_smoke.json", smoke)
        write_post_run_provenance(args.run_root, project, manifest)
        print(json.dumps({"manifest":str(manifest_path), "live_realizer_smoke":smoke}, indent=2))
    elif args.backend_mode == "hybrid":
        from .live_backends import (LiveSemanticRepairBackend,
                                    OpenAICompatibleChallengerClient)
        from .orchestrator import CanaryOrchestrator
        from .vllm_backends import VLLMSolverBackend

        source = repair_artifacts["source"]
        client = OpenAICompatibleChallengerClient(live_config)
        live_orchestrator = CanaryOrchestrator(run_root=args.run_root, manifest=manifest,
            config=resolved, data=data, tokenizer=length_tokenizer, realizer=None,
            questioner=LiveSemanticRepairBackend(client,
                resolved["challenger_semantic_repair_max_tokens"]),
            solver=None, backend_mode="live")
        source_contract = {"source_solver_manifest_sha256":source["source_manifest_sha256"],
            "source_solver_report_sha256":source["source_report_sha256"],
            "source_solver_ledger_aggregate_sha256":source["solver_ledger_aggregate_sha256"],
            "source_diagnostic_manifest_sha256":source["diagnostic_manifest_sha256"],
            "source_diagnostic_report_sha256":source["diagnostic_report_sha256"],
            "source_diagnostic_ledger_aggregate_sha256":source[
                "diagnostic_ledger_aggregate_sha256"],
            "source_frozen_dialogue_sha256":sha256_file(repair_artifacts[
                "frozen_paths"]["context"]),
            "source_span_catalog_sha256":span_catalog_sha256(source["span_catalog"])}
        admission = live_orchestrator.semantic_leakage_repair(
            parent_qa_id=args.solver_qa_id,
            repaired_qa_id=repair_artifacts["repaired_qa_id"],
            target_letter=source["target_letter"],
            semantic_prompt=repair_artifacts["rendered"],
            span_catalog=source["span_catalog"],
            reasoning_contract=repair_artifacts["reasoning_contract"],
            source_contract=source_contract)
        solver_result = None
        solver_report = None
        solver_preflight_path = None
        if admission.get("materialized_qa_path"):
            qa = admission["admission"]["materialized_qa"]
            frozen_dialogue = source["context"]["combined_dialogue"]
            evidence_contexts = build_answer_evidence_contexts(qa,
                span_catalog=source["span_catalog"], frozen_dialogue=frozen_dialogue)
            prompt_modes = {"no_context":[], "full_context":frozen_dialogue,
                "evidence_only":evidence_contexts["evidence_only"],
                "evidence_masked":evidence_contexts["evidence_masked"]}
            solver_preflight = {}
            for evaluation_mode, turns in prompt_modes.items():
                rendered = render_solver_prompt(length_tokenizer, qa, turns)
                tokens = len(length_tokenizer.encode(rendered, add_special_tokens=False))
                solver_preflight[evaluation_mode] = {
                    "rendered_prompt_sha256":sha256_bytes(rendered.encode()),
                    "rendered_prompt_tokens":tokens,
                    "reserved_generation_tokens":resolved["reserved_generation_tokens"],
                    "solver_max_model_len":resolved["solver_effective_max_model_len"],
                    "preflight_decision":"pass" if tokens + resolved[
                        "reserved_generation_tokens"] <= resolved[
                        "solver_effective_max_model_len"] else "reject_context_overflow"}
            if any(value["preflight_decision"] != "pass"
                   for value in solver_preflight.values()):
                raise RuntimeError("semantic repair Solver context overflow")
            solver_preflight_path = args.run_root / "frozen/repaired_solver_prompt_preflight.json"
            freeze_json(solver_preflight_path, solver_preflight, args.resume)
            backend = VLLMSolverBackend(model_path=solver, tokenizer_path=tokenizer,
                max_model_len=resolved["solver_effective_max_model_len"],
                temperature=resolved["solver_temperature"], top_p=resolved["solver_top_p"],
                max_tokens=resolved["solver_max_new_tokens"],
                gpu_memory_utilization=resolved["solver_gpu_memory_utilization"])
            solver_orchestrator = CanaryOrchestrator(run_root=args.run_root,
                manifest=manifest, config=resolved, data=data, tokenizer=length_tokenizer,
                realizer=None, questioner=None, solver=backend, backend_mode="local")
            solver_result = solver_orchestrator.audit_semantic_repair_candidate(
                qa, args.solver_qa_id)
            stages = [solver_result.get("initial_no_context"),
                solver_result.get("adaptive_extension"), solver_result.get("full_context"),
                solver_result.get("evidence_only"), solver_result.get("evidence_masked")]
            stages.extend(order.get("rollout") for order in
                          solver_result.get("option_orders") or [])
            stages = [stage for stage in stages if isinstance(stage, dict)]
            raw_rollouts = [row for stage in stages for row in stage.get("rollouts") or []]
            solver_report = {"result":solver_result,
                "vllm_generation_count":sum(stage.get("raw_generations", 0)
                                              for stage in stages),
                "model_identifier":backend.model_identifier,
                "solver_max_model_len":backend.max_model_len,
                "model_load_latency_seconds":backend.load_latency_seconds,
                "generation_latency_seconds":sum(float((row.get("backend_metadata") or {}).get(
                    "latency_seconds", 0.0)) for row in raw_rollouts)}
        final_decision = (solver_result.get("final_decision") if solver_result else
                          admission["final_decision"])
        ledger_summary = live_orchestrator.ledger.summary()
        final_state = live_orchestrator.ledger.read(admission["final_request_id"]) or {}
        report = {"mode":mode, "semantic_repair_version":SEMANTIC_LEAKAGE_REPAIR_VERSION,
            "parent_qa_id":args.solver_qa_id,
            "qa_id":repair_artifacts["repaired_qa_id"], "repair_index":1,
            "repair_attempt_count":1, "repair_reason":SEMANTIC_LEAKAGE_REPAIR_REASON,
            "repair":admission, "solver":solver_report,
            "repair_json_valid":admission["repair_json_valid"],
            "repair_minimal_accepted":bool(admission.get("materialized_qa_path")),
            "repair_passed_grounding":bool(solver_result and
                solver_result.get("grounding_decision") == "pass_grounding"),
            "final_decision":final_decision, **ledger_summary,
            "semantic_repair_normal_requests":1,
            "semantic_repair_format_retry_requests":max(0,
                admission["physical_api_request_count"] - 1),
            "paid_api_generations":ledger_summary["paid_api_generation_count"],
            "usage":(final_state.get("response_metadata") or {}).get("usage"),
            "realizer_generations":0, "ordinary_questioner_generations":0,
            "verifier_generations":0, "training_runs":0,
            "full_context_ran":bool(solver_result and solver_result.get("full_context")),
            "evidence_audit_ran":bool(solver_result and solver_result.get("evidence_audit")),
            "option_audit_ran":bool(solver_result and solver_result.get("option_audit"))}
        report_path = args.run_root / "reports/semantic_repair_full_audit.json"
        atomic_json(report_path, report)
        ledger_hashes = {path.stem:sha256_file(path) for path in sorted(
            (args.run_root / "ledger/requests").glob("*.json"))}
        output_provenance = {"run_manifest_sha256":sha256_file(manifest_path),
            "repair_report_sha256":sha256_file(report_path),
            "repair_raw_sha256":(admission["attempts"][-1].get("raw_response_sha256")
                                  if admission["attempts"] else None),
            "repair_materialized_qa_sha256":admission.get("materialized_qa_sha256"),
            "solver_prompt_preflight_sha256":sha256_file(solver_preflight_path)
                if solver_preflight_path else None,
            "ledger_sha256":ledger_hashes,
            "ledger_aggregate_sha256":sha256_bytes(canonical_json(ledger_hashes).encode()),
            "final_decision":final_decision}
        atomic_json(args.run_root / "reports/semantic_repair_output_provenance.json",
                    output_provenance)
        write_post_run_provenance(args.run_root, project, manifest)
        print(json.dumps({"manifest":str(manifest_path),
            "semantic_leakage_repair":report}, indent=2))
        return 0
    elif args.backend_mode == "local":
        from .orchestrator import CanaryOrchestrator
        from .vllm_backends import VLLMSolverBackend
        backend = VLLMSolverBackend(model_path=solver, tokenizer_path=tokenizer,
            max_model_len=resolved["solver_effective_max_model_len"],
            temperature=resolved["solver_temperature"], top_p=resolved["solver_top_p"],
            max_tokens=resolved["solver_max_new_tokens"],
            gpu_memory_utilization=resolved["solver_gpu_memory_utilization"])
        orchestrator = CanaryOrchestrator(run_root=args.run_root, manifest=manifest, config=resolved,
            data=data, tokenizer=length_tokenizer, realizer=None, questioner=None, solver=backend,
            backend_mode="local")
        qa = solver_artifacts["qa"]
        extension = args.smoke_stage == "solver_grounding_extension"
        adaptive = args.smoke_stage == "solver_adaptive_item"
        minimal_audit = args.smoke_stage == "solver_minimal_full_audit"
        permutation_diagnostic = args.smoke_stage == "solver_no_context_permutation_diagnostic"
        if permutation_diagnostic:
            result = orchestrator.diagnose_no_context_permutations(
                qa, solver_artifacts["permutation_source_integrity"]["original_no_context"])
            stages = [order["rollout"] for order in result["permutation_orders"]]
            raw_rollouts = [row for stage in stages for row in stage.get("rollouts") or []]
            ledger_summary = orchestrator.ledger.summary()
            report = {"mode":mode,
                "smoke_stage":"bad_ground_no_context_permutation_diagnostic",
                "diagnostic_version":NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION,
                "qa_id":qa["qa_id"], "admission_mode":SPELL_STYLE_MINIMAL_MODE,
                "result":result, **ledger_summary,
                "vllm_generation_count":sum(stage.get("raw_generations", 0) for stage in stages),
                "model_identifier":backend.model_identifier,
                "solver_max_model_len":backend.max_model_len,
                "model_load_latency_seconds":backend.load_latency_seconds,
                "generation_latency_seconds":sum(float((row.get("backend_metadata") or {}).get(
                    "latency_seconds", 0.0)) for row in raw_rollouts),
                "paid_api_generations":0, "realizer_generations":0,
                "questioner_generations":0, "verifier_generations":0,
                "training_runs":0, "full_context_generations":0,
                "evidence_generations":0, "frontier_permutation_generations":0}
            report_path = args.run_root / "reports/no_context_permutation_diagnostic.json"
            atomic_json(report_path, report)
            write_post_run_provenance(args.run_root, project, manifest)
            print(json.dumps({"manifest":str(manifest_path),
                "no_context_permutation_diagnostic":report}, indent=2))
            return 0
        if minimal_audit:
            result = orchestrator.audit_minimal_candidate(qa)
            stages = [result.get("initial_no_context"), result.get("adaptive_extension"),
                      result.get("full_context"), result.get("evidence_only"),
                      result.get("evidence_masked")]
            stages.extend(order.get("rollout") for order in result.get("option_orders") or [])
            stages = [stage for stage in stages if isinstance(stage, dict)]
            raw_rollouts = [row for stage in stages for row in stage.get("rollouts") or []]
            ledger_summary = orchestrator.ledger.summary()
            report = {"mode":mode, "smoke_stage":"minimal_full_solver_audit",
                "qa_id":qa["qa_id"], "admission_mode":SPELL_STYLE_MINIMAL_MODE,
                "result":result, **ledger_summary,
                "vllm_generation_count":sum(stage.get("raw_generations", 0) for stage in stages),
                "model_identifier":backend.model_identifier,
                "solver_max_model_len":backend.max_model_len,
                "model_load_latency_seconds":backend.load_latency_seconds,
                "generation_latency_seconds":sum(float((row.get("backend_metadata") or {}).get(
                    "latency_seconds", 0.0)) for row in raw_rollouts),
                "paid_api_generations":0, "realizer_generations":0,
                "questioner_generations":0, "verifier_generations":0, "training_runs":0,
                "stopped_after_final_retain_frontier":result.get("final_decision") == "retain_frontier"}
            report_path = args.run_root / "reports/minimal_solver_full_audit.json"
            atomic_json(report_path, report)
            write_post_run_provenance(args.run_root, project, manifest)
            print(json.dumps({"manifest":str(manifest_path),
                "minimal_solver_full_audit":report}, indent=2))
            return 0
        if adaptive:
            initial = orchestrator._rollouts(qa, "no_context", "original", 3, qa["options"], 0)
            extension_result = None
            if initial["obtained_valid_rollouts"] != 3:
                ground = {"decision":"bad_no_context_eval", "needs_extension":False}
            else:
                ground = grounding_decision(initial["verified_rewards"])
                if ground["needs_extension"]:
                    extension_result = orchestrator._rollouts(qa, "no_context", "original", 2, qa["options"], 3)
                    ground = grounding_decision(initial["verified_rewards"], extension_result["verified_rewards"])
            full = None
            preliminary = None
            gate = full_context_gate(ground)
            if gate["run_full_context"]:
                full = orchestrator._rollouts(qa, "full_context", "original", 5, qa["options"], 0)
                preliminary = full_context_gate(ground, full["verified_rewards"])
            all_stages = [initial] + ([extension_result] if extension_result else []) + ([full] if full else [])
            ledger_summary = orchestrator.ledger.summary()
            report = {"mode":mode, "smoke_stage":"adaptive_item_grounding_then_conditional_full_context",
                "qa_id":qa["qa_id"], "initial_no_context":initial,
                "initial_c3":sum(initial["verified_rewards"]) if len(initial["verified_rewards"]) == 3 else None,
                "extension_used":extension_result is not None, "extension_no_context":extension_result,
                "merged_no_context_rewards":initial["verified_rewards"] +
                    (extension_result["verified_rewards"] if extension_result else []),
                "final_grounding_decision":ground, "full_context_ran":full is not None,
                "full_context_gate":gate,
                "full_context":full, "full_context_preliminary_decision":preliminary,
                **ledger_summary, "vllm_generation_count":sum(x["raw_generations"] for x in all_stages),
                "model_identifier":backend.model_identifier, "solver_max_model_len":backend.max_model_len,
                "model_load_latency_seconds":backend.load_latency_seconds,
                "paid_api_generations":0, "realizer_generations":0, "questioner_generations":0,
                "verifier_generations":0, "training_runs":0, "evidence_audit_ran":False,
                "option_permutation_ran":False}
            atomic_json(args.run_root / "reports/local_solver_item2_adaptive.json", report)
            write_post_run_provenance(args.run_root, project, manifest)
            print(json.dumps({"manifest":str(manifest_path), "local_solver_item2_adaptive":report}, indent=2))
            return 0
        requested = 2 if extension else 3
        slot_start = 3 if extension else 0
        result = orchestrator._rollouts(qa, "no_context", "original", requested, qa["options"], slot_start)
        ledger_summary = orchestrator.ledger.summary()
        boxed_smoke = args.smoke_stage == "solver_boxed_output"
        report = {"mode":mode, "smoke_stage":"boxed_output_protocol" if boxed_smoke else "S1_grounding_extension" if extension else "S1",
            "qa_id":qa["qa_id"], "evaluation_mode":"no_context", "requested_valid_k":requested,
            **result, **ledger_summary, "model_identifier":backend.model_identifier,
            "solver_max_model_len":backend.max_model_len, "model_load_latency_seconds":backend.load_latency_seconds,
            "paid_api_generations":0, "realizer_generations":0, "questioner_generations":0,
            "vllm_generation_count":result["raw_generations"],
            "verifier_generations":0, "training_runs":0}
        if extension:
            report.update({"prior_verified_rewards":solver_artifacts["prior"]["verified_rewards"],
                "combined_grounding":grounding_decision(solver_artifacts["prior"]["verified_rewards"],
                                                         result["verified_rewards"]),
                "prior_rollouts_regenerated":False})
        report_name = "solver_boxed_output_smoke.json" if boxed_smoke else "local_solver_s1_grounding_extension.json" if extension else "local_solver_s1_smoke.json"
        atomic_json(args.run_root / f"reports/{report_name}", report)
        write_post_run_provenance(args.run_root, project, manifest)
        print(json.dumps({"manifest":str(manifest_path), report_name.removesuffix('.json'):report}, indent=2))
    else:
        write_post_run_provenance(args.run_root, project, manifest)
        print(json.dumps({"manifest": str(manifest_path), **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
