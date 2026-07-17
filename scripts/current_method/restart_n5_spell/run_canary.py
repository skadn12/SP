from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import PARSER_VERSION
from .core import (atomic_json, atomic_text, canonical_json, classify_frontier_pre_audit, full_context_gate,
                   context_preflight, grounding_decision,
                   sha256_file, sha256_bytes)
from .input_adapter import load_lineage, read_json, verify_source_run_sessions
from .prompts import (QUESTIONER_PROMPT_VERSION, SOLVER_NOTHINK_RESPONDER_PROMPT, build_questioner_prompt,
                      prompt_template_hashes)
from .validation import (audit_continuity_hooks, materialize_questioner_item,
                         validate_config, validate_qa)

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
        "source_snapshot_manifest_sha_match":sha256_file(Path(str(manifest["source_snapshot_manifest_path"]))) ==
            manifest.get("source_snapshot_manifest_sha256")}
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
                     turn_session_ids: dict[str, str] | None = None) -> tuple[dict, dict, dict]:
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
    rematerialized, materialize_errors = materialize_questioner_item(generated[0], target)
    if (materialize_errors or rematerialized is None or
            canonical_json(rematerialized) != canonical_json(materialized)):
        raise RuntimeError("solver_qa_integrity_mismatch")
    if frozen_turns is None or turn_session_ids is None:
        raise RuntimeError("solver_qa_integrity_mismatch: frozen dialogue is required")
    errors = validate_qa(materialized, set(frozen_turns), frozen_turns, turn_session_ids)
    if errors or not materialized.get("local_option_binding") or not materialized.get("canonical_answer_text"):
        raise RuntimeError(f"solver_qa_integrity_mismatch: {errors}")
    return generated[0], dict(materialized), admission


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
    parser.add_argument("--backend-mode", choices=("audit", "mock", "live", "local"), default="audit")
    parser.add_argument("--smoke-stage", choices=("realizer", "realizer_continuity", "questioner",
                                                   "solver_no_context", "solver_grounding_extension",
                                                   "solver_adaptive_item", "solver_boxed_output"))
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
    args = parser.parse_args(argv)
    if args.backend_mode == "live" and args.smoke_stage not in ("realizer", "realizer_continuity", "questioner"):
        raise RuntimeError("live execution requires an explicit supported smoke stage")
    if args.backend_mode == "local" and args.smoke_stage not in ("solver_no_context", "solver_grounding_extension", "solver_adaptive_item", "solver_boxed_output"):
        raise RuntimeError("local execution requires an explicit supported solver smoke")
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
    if args.backend_mode == "live":
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
    mode = ("live_questioner_smoke" if args.backend_mode == "live" and args.smoke_stage == "questioner"
            else "live_smoke_realizer_continuity" if args.backend_mode == "live" and args.smoke_stage == "realizer_continuity"
            else "live_smoke_realizer" if args.backend_mode == "live"
            else "local_solver_adaptive_item_smoke" if args.backend_mode == "local" and args.smoke_stage == "solver_adaptive_item"
            else "local_solver_boxed_output_smoke" if args.backend_mode == "local" and args.smoke_stage == "solver_boxed_output"
            else "local_solver_grounding_extension_smoke" if args.backend_mode == "local" and args.smoke_stage == "solver_grounding_extension"
            else "local_solver_no_context_smoke" if args.backend_mode == "local"
            else "mock_e2e" if args.backend_mode == "mock" else "dry_run")
    resolved = {**DEFAULTS, "mode": mode,
                "backend_mode": args.backend_mode, "lineage": args.lineage,
                "input_root": str(args.input_root.resolve()), "run_root": str(args.run_root.resolve())}
    if args.backend_mode == "mock":
        resolved["mock_questioner_shortfall_batch"] = args.mock_questioner_shortfall_batch
        resolved["mock_questioner_all_invalid"] = args.mock_questioner_all_invalid
    if args.backend_mode == "live":
        continuity = args.smoke_stage == "realizer_continuity"
        questioner_smoke = args.smoke_stage == "questioner"
        paid_limit = 2 if continuity else 1
        resolved.update({"smoke_stage":args.smoke_stage, "smoke_max_paid_generations":paid_limit,
            "smoke_max_realizer_generations":0 if questioner_smoke else paid_limit,
            "max_session_attempts_per_session":1, "max_total_paid_api_generations":paid_limit,
            "max_realizer_generations":0 if questioner_smoke else paid_limit,
            "max_questioner_generations":1 if questioner_smoke else 0,
            "questioner_generations":1 if questioner_smoke else 0,
            "questioner_mcqs_per_generation":3, "questioner_total_candidates":3 if questioner_smoke else 0,
            "max_post_freeze_memory_generations":0,
            "challenger_temperature":live_config.temperature,
            "challenger_realizer_max_tokens":live_config.realizer_max_tokens,
            "challenger_questioner_max_tokens":live_config.questioner_max_tokens})
    if args.backend_mode == "local":
        extension = args.smoke_stage == "solver_grounding_extension"
        adaptive = args.smoke_stage == "solver_adaptive_item"
        resolved.update({"smoke_stage":args.smoke_stage, "max_total_paid_api_generations":0,
            "max_realizer_generations":0, "max_questioner_generations":0,
            "max_post_freeze_memory_generations":0, "max_verifier_generations":0,
            "max_session_attempts_per_session":1, "max_local_solver_generations":20 if adaptive else 4 if extension else 6,
            "solver_requested_valid_k":2 if extension else 3, "solver_slot_start":3 if extension else 0,
            "solver_temperature":0.7, "solver_top_p":0.9,
            "solver_max_new_tokens":32, "solver_enable_thinking":False,
            "solver_add_generation_prompt":True, "solver_effective_max_model_len":16384,
            "solver_tokenizer_path":str(tokenizer), "solver_gpu_memory_utilization":0.80})
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
    if args.backend_mode == "live" and args.smoke_stage == "questioner":
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
        rendered = build_questioner_prompt(frozen_dialogue=frozen["combined_dialogue"],
            valid_turn_ids=valid_turn_ids, valid_session_ids=valid_session_ids,
            assigned_gold_letters=letters, recent_frontier_history=[], batch_index=0)
        questioner_artifacts = {"source":source, "session_paths":session_paths, "sessions":sessions,
            "frozen":frozen, "frozen_path":frozen_path, "valid_turn_ids":valid_turn_ids,
            "valid_session_ids":valid_session_ids, "letters":letters, "rendered":rendered,
            "source_integrity":source_integrity}
    if args.backend_mode == "local":
        boxed_smoke = args.smoke_stage == "solver_boxed_output"
        source = args.solver_qa_source_root.resolve()
        raw_path = source / "raw/backend_responses/questioner-b00.txt"
        report_path = source / "reports/live_questioner_smoke.json"
        if boxed_smoke:
            generated_qa = {"qa_id":"synthetic_boxed_protocol_01",
                "question":"Which option exactly matches the target token 'blue' stated in this question?",
                "correct_answer_text":"blue", "distractors":["red","green","gold"],
                "correct_answer_explanation":"The synthetic protocol fixture names blue.",
                "distractor_explanations":["red is a different fixture value",
                    "green is a different fixture value","gold is a different fixture value"],
                "distractor_values_from_evidence":["red","green","gold"],
                "distractor_supporting_quotes":[
                    [{"turn_id":"SYN1","exact_quote":"red"}],
                    [{"turn_id":"SYN1","exact_quote":"green"}],
                    [{"turn_id":"SYN2","exact_quote":"gold"}]],
                "evidence_turn_ids":["SYN1","SYN2"], "evidence_session_ids":["synthetic"],
                "answer_support_type":"multi_hop", "answer_evidence_values":[
                    {"turn_id":"SYN1","value":"blue","exact_quote":"blue","role":"answer_value"},
                    {"turn_id":"SYN2","value":"target token","exact_quote":"target token",
                     "role":"supporting_fact"}],
                "final_reasoning_conclusion":"blue",
                "reasoning_steps":["Read the target token.","Select the matching value."],
                "reasoning_type":"synthetic_exact_token_match", "difficulty_level":"L1"}
            qa, synthetic_errors = materialize_questioner_item(generated_qa, "B")
            synthetic_turns = {"SYN1":"blue red green", "SYN2":"target token gold"}
            if synthetic_errors or qa is None or validate_qa(qa, set(synthetic_turns), synthetic_turns,
                    {"SYN1":"synthetic","SYN2":"synthetic"}):
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
            generated_qa, qa, accepted_item = select_solver_qa(raw_batch, smoke_report,
                args.solver_qa_id, frozen_turns=turn_text, turn_session_ids=turn_sessions)
            original_qa = qa
        qa_path = args.run_root / "frozen/solver_qa.json"
        frozen_qa_payload = {"source_root":str(source),
                              "source_generated_qa":generated_qa,
                              "source_generated_qa_sha256":sha256_bytes(canonical_json(generated_qa).encode()),
                              "original_qa":original_qa,
                              "original_qa_sha256":sha256_bytes(canonical_json(original_qa).encode()), "qa":qa,
                              "schema_acceptance":({"schema_decision":"synthetic_protocol_only",
                                  "materialized_qa":qa, "target_letter":"B"} if boxed_smoke else
                                  next(x for x in smoke_report["items"] if x["qa_id"] == args.solver_qa_id)),
                              "source_raw_sha256":sha256_file(raw_path),
                              "source_report_sha256":sha256_file(report_path)}
        freeze_json(qa_path, frozen_qa_payload, args.resume)
        solver_artifacts = {"source":source, "raw_path":raw_path, "report_path":report_path,
                            "qa":qa, "original_qa":original_qa, "qa_path":qa_path}
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_value(project, "rev-parse", "HEAD"),
        "project_git_dirty": bool(git_value(project, "status", "--porcelain")),
        "project_diff_sha256": project_diff_sha256(project),
        "resolved_config_sha256": sha256_file(config_path),
        "realizer_prompt_sha256": prompt_hashes["realizer"],
        "questioner_prompt_sha256": prompt_hashes["questioner"],
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
            "max_tokens":live_config.questioner_max_tokens if args.smoke_stage == "questioner" else live_config.realizer_max_tokens}
            if live_config else {"temperature":resolved["solver_temperature"], "top_p":resolved["solver_top_p"],
                "max_tokens":resolved["solver_max_new_tokens"]} if args.backend_mode == "local" else {}),
        "mcq_valid_rollout_policy": {"max_parse_retries_per_target": 1,
            "no_context_initial_valid_k": 3, "no_context_boundary_extra_valid_k": 2,
            "full_context_valid_k": 5, "permutation_valid_k_per_order": 5},
        "questioner_batch_policy": ({"generations":1, "mcqs_per_generation":3, "total_candidates":3,
            "model_output":"unlabeled_correct_answer_plus_three_distractors",
            "option_binding":"local_option_binding_v1"}
            if args.backend_mode == "live" and args.smoke_stage == "questioner" else
            {"generations": 4, "mcqs_per_generation": 3, "total_candidates": 12,
             "option_binding":"local_option_binding_v1"}),
        "paid_call_budgets": ({"max_total_paid_api_generations":resolved["smoke_max_paid_generations"],
            "max_realizer_generations":resolved["smoke_max_realizer_generations"],
            "max_questioner_generations":resolved["max_questioner_generations"],
            "max_post_freeze_memory_generations":0, "max_verifier_generations":0}
            if args.backend_mode == "live" else
            {k: v for k, v in resolved.items() if k.startswith("max_") and "generations" in k}),
        "spell_repo_status": "available", "spell_repo_path": str(spell_repo),
        "spell_commit": spell_commit, "spell_dirty": spell_dirty,
        "input_provenance": data["provenance"],
        **source_snapshot,
        "orchestrator_source_sha256": sha256_file(Path(__file__).with_name("orchestrator.py")),
        "validation_source_sha256":sha256_file(Path(__file__).with_name("validation.py")),
        "ledger_source_sha256":sha256_file(Path(__file__).with_name("ledger.py")),
        "run_canary_source_sha256":sha256_file(Path(__file__)),
        "backend_source_sha256": sha256_file(Path(__file__).with_name("mock_backends.py")) if args.backend_mode == "mock" else "",
        "live_backend_source_sha256": sha256_file(Path(__file__).with_name("live_backends.py")) if args.backend_mode == "live" else "",
        "api_config_path": str(args.api_config.resolve()) if args.backend_mode == "live" else "",
        "api_config_sha256": sha256_file(args.api_config) if args.backend_mode == "live" else "",
    }
    if questioner_artifacts:
        manifest.update({"questioner_prompt_version":QUESTIONER_PROMPT_VERSION,
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
            "target_gold_letters":list(questioner_artifacts["letters"])})
    if solver_artifacts:
        extension = args.smoke_stage == "solver_grounding_extension"
        adaptive = args.smoke_stage == "solver_adaptive_item"
        boxed_smoke = args.smoke_stage == "solver_boxed_output"
        manifest.update({"solver_smoke_stage":"boxed_output_no_context_valid_k3" if boxed_smoke else "adaptive_item_grounding_then_conditional_full_context" if adaptive else "S1_grounding_extension_valid_k2" if extension else "S1_no_context_valid_k3",
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
            realizer=None if args.smoke_stage == "questioner" else LiveRealizerBackend(client),
            questioner=LiveQuestionerBackend(client) if args.smoke_stage == "questioner" else None,
            solver=None, backend_mode="live")
        if args.smoke_stage == "questioner":
            result = orchestrator.question_smoke(questioner_artifacts["frozen"]["combined_dialogue"],
                questioner_artifacts["valid_turn_ids"], questioner_artifacts["valid_session_ids"],
                questioner_artifacts["letters"])
            state = orchestrator.ledger.read("questioner-b00") or {}
            result.update({"physical_paid_generations":orchestrator.ledger.summary()["paid_api_generation_count"],
                "request_status":state.get("status"), "usage":(state.get("response_metadata") or {}).get("usage"),
                "model_identifier":state.get("model_identifier"), "realizer_generations":0,
                "solver_generations":0, "vllm_generations":0, "verifier_generations":0, "training_runs":0})
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
