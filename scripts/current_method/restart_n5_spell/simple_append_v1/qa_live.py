"""QA-only live pilot over one frozen simple-append expanded dialogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from ..backends import SolverRequest
from ..core import (
    atomic_json, atomic_text, canonical_json, local_reward,
    parse_boxed_mcq_answer, sha256_bytes, sha256_file,
)
from ..ledger import RequestLedger
from ..live_backends import (
    OpenAICompatibleChallengerClient, load_challenger_config,
)
from ..prompts import (
    SOLVER_POSITION_NEUTRAL_PARSE_RETRY_SUFFIX,
    SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT,
    SOLVER_TRAILING_FORMAT_INSTRUCTION,
)
from ..questioner_contract import build_span_catalog, span_catalog_sha256
from ..vllm_backends import VLLMSolverBackend
from .live_canary import _source_manifest
from .live_runtime import load_json_object_strict
from .prompts import build_mcq_batch_prompt
from .qa import (
    MIN_FOCUS_ROUND_JOINT_WITH_HISTORY,
    MIN_FOCUS_ROUND_SUPPORTED,
    build_evidence_alias_registry, retain_focus_writer_overflow,
    classify_balanced_no_context, classify_full_context_frontier,
    evaluate_focus_round_profile, evaluate_qa_quota_status,
    gold_letter_for_options,
    materialize_solver_candidate, partition_mcq_candidates, permuted_options,
    normalize_focus_writer_batch,
)


RUNNER_VERSION = "simple_append_qa_only_live_v16_spell_frontier_k5"
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_live_canary_4")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_qa_only_pilot_7")
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SOLVER_MODEL = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "05_stage2_repair_topup_from_stage1/training/models/merged")
QUESTION_COUNT = 8
MAX_QUESTION_COUNT = 12
MIN_CHILD_SUPPORTED = 4
MIN_JOINT_PARENT_CHILD = 2
TARGET_DISTINCT_QUESTION_TYPES = 4
WRITER_MAX_TOKENS = 8192
CONTEXT_LIMIT = 40_960
SOLVER_MAX_MODEL_LEN = 16_384
SOLVER_MAX_TOKENS = 32
SOLVER_GPU_MEMORY_UTILIZATION = 0.80


def _highest_round_index(expanded: Mapping[str, Any]) -> int:
    rounds = []
    for row in expanded.get("turns") or []:
        match = re.fullmatch(
            r"R(\d{2}):S\d{2}", str(row.get("scene_ref") or ""))
        if match:
            rounds.append(int(match.group(1)))
    return max(rounds, default=0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*")
                       if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(relative + b"\0" + str(len(payload)).encode() + b"\0" +
                      hashlib.sha256(payload).hexdigest().encode() + b"\0")
    return digest.hexdigest()


def build_solver_source(expanded: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the simple expanded artifact without importing v11 QA semantics."""
    turns = expanded.get("turns")
    if (expanded.get("qa_fact_source") != "complete_parent_plus_child" or
            expanded.get("memory_is_qa_fact_source") is not False or
            not isinstance(turns, list) or not turns):
        raise ValueError("simple_append_qa_source_invalid")
    dialogue: list[dict[str, str]] = []
    partition_by_turn: dict[str, str] = {}
    for row in turns:
        turn_id = str(row.get("turn_ref") or "")
        session_id = str(row.get("session_ref") or "")
        speaker = str(row.get("speaker") or "")
        text = str(row.get("text") or "")
        origin = str(row.get("origin") or "")
        partition = ({"source_grounded": "original_parent",
                      "generated_extension": "promoted_child"}.get(origin))
        if (not turn_id or turn_id in partition_by_turn or not session_id or
                not speaker or not text or partition is None):
            raise ValueError("simple_append_qa_source_turn_invalid")
        dialogue.append({"turn_id": turn_id, "session_id": session_id,
                         "speaker": speaker, "text": text})
        partition_by_turn[turn_id] = partition
    catalog = build_span_catalog(dialogue)
    return {"combined_dialogue": dialogue, "partition_by_turn": partition_by_turn,
            "span_catalog": catalog,
            "source_view_sha256": sha256_bytes(canonical_json({
                "combined_dialogue": dialogue,
                "partition_by_turn": partition_by_turn,
                "span_catalog": catalog,
            }).encode())}


def preflight(
    *, source_run: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, solver_model: Path,
    focus_round_index: int | None = None,
    writer_output_replay: Path | None = None,
) -> dict[str, Any]:
    expanded_path = source_run / "expanded_dialogue.json"
    live_summary_path = source_run / \
        "reports/simple_append_live_canary_summary.json"
    round_summary_path = source_run / "reports/ROUND_SUMMARY.json"
    summary_path = (live_summary_path if live_summary_path.is_file() else
                    round_summary_path)
    checks = {
        "run_root_absent": not run_root.exists(),
        "source_run_exists": source_run.is_dir(),
        "expanded_dialogue_exists": expanded_path.is_file(),
        "source_summary_exists": summary_path.is_file(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.is_dir(),
        "solver_model_exists": solver_model.is_dir(),
        "writer_replay_exists_if_requested": (
            writer_output_replay is None or writer_output_replay.is_file()),
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        try:
            expanded = _load(expanded_path)
            summary = _load(summary_path)
            effective_focus_round = (
                int(focus_round_index) if focus_round_index is not None else
                _highest_round_index(expanded))
            source = build_solver_source(expanded)
            prompt = build_mcq_batch_prompt(
                expanded, target_questions=QUESTION_COUNT,
                maximum_questions=MAX_QUESTION_COUNT,
                focus_round_index=effective_focus_round)
            evidence_aliases = build_evidence_alias_registry(expanded)
            partitions = list(source["partition_by_turn"].values())
            replay_envelope = None
            replay_partition = None
            if writer_output_replay is not None:
                replay_envelope = normalize_focus_writer_batch(
                    _load(writer_output_replay))
                if replay_envelope["valid"]:
                    replay_partition = partition_mcq_candidates(
                        replay_envelope["batch"],
                        expanded_dialogue=expanded,
                        minimum_count=QUESTION_COUNT,
                        maximum_count=None,
                        min_child_supported=MIN_CHILD_SUPPORTED,
                        min_joint_parent_child=MIN_JOINT_PARENT_CHILD)
            checks.update({
                "source_generation_ready": (
                    summary.get("generation_ready") is True if
                    summary_path == live_summary_path else
                    int(summary.get("turn_count") or 0) ==
                    int(expanded.get("turn_count") or 0)),
                "source_qa_ready": (
                    summary.get("qa_source_ready") is True if
                    summary_path == live_summary_path else True),
                "parent_present": "original_parent" in partitions,
                "child_present": "promoted_child" in partitions,
                "turn_count_matches": len(source["combined_dialogue"]) ==
                    int(expanded.get("turn_count") or 0),
                "writer_prompt_fits": prompt["token_estimate"] +
                    WRITER_MAX_TOKENS <= CONTEXT_LIMIT,
                "candidate_pool_minimum_is_eight": QUESTION_COUNT == 8,
                "candidate_pool_has_no_content_upper_gate": True,
                "batch_source_quotas_frozen": (
                    MIN_CHILD_SUPPORTED == 4 and
                    MIN_JOINT_PARENT_CHILD == 2),
                "focus_round_quota_is_four_plus_two": (
                    MIN_FOCUS_ROUND_SUPPORTED == 4 and
                    MIN_FOCUS_ROUND_JOINT_WITH_HISTORY == 2),
                "focus_round_exists": effective_focus_round >= 1 and any(
                    str(row.get("scene_ref") or "").startswith(
                        f"R{effective_focus_round:02d}:")
                    for row in expanded.get("turns") or []),
                "question_type_diversity_is_statistical":
                    TARGET_DISTINCT_QUESTION_TYPES == 4,
                "no_context_base_is_four_by_one": True,
                "planner_realizer_memory_calls_zero": True,
                "evidence_alias_registry_complete": (
                    evidence_aliases["turn_count"] ==
                    len(source["combined_dialogue"])),
                "writer_replay_envelope_valid_if_requested": (
                    writer_output_replay is None or bool(
                        replay_envelope and replay_envelope["valid"])),
                "writer_replay_has_solver_candidates_if_requested": (
                    writer_output_replay is None or bool(
                        replay_partition and
                        replay_partition["solver_ready_count"])),
            })
            details = {
                "root_id": expanded.get("root_id"),
                "source_summary_path": str(summary_path),
                "focus_round_index": effective_focus_round,
                "expanded_dialogue_sha256": sha256_file(expanded_path),
                "source_run_tree_sha256": _hash_tree(source_run),
                "source_view_sha256": source["source_view_sha256"],
                "span_catalog_sha256": span_catalog_sha256(source["span_catalog"]),
                "turn_count": len(source["combined_dialogue"]),
                "parent_turn_count": partitions.count("original_parent"),
                "child_turn_count": partitions.count("promoted_child"),
                "writer_prompt_sha256": prompt["prompt_sha256"],
                "writer_prompt_token_estimate": prompt["token_estimate"],
                "evidence_alias_schema_version":
                    evidence_aliases["evidence_alias_schema_version"],
                "evidence_alias_registry_sha256":
                    evidence_aliases["registry_sha256"],
                "evidence_alias_turn_count": evidence_aliases["turn_count"],
                "writer_output_replay": (
                    str(writer_output_replay) if writer_output_replay else None),
                "writer_output_replay_sha256": (
                    sha256_file(writer_output_replay)
                    if writer_output_replay else None),
                "writer_replay_solver_ready_count": (
                    int(replay_partition["solver_ready_count"])
                    if replay_partition else None),
            }
        except Exception as error:
            checks["source_adapter_valid"] = False
            details["preflight_error"] = f"{type(error).__name__}:{error}"
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {"preflight_version": RUNNER_VERSION,
            "decision": "pass" if not failed else "fail",
            "checks": checks, "failed_checks": failed, "details": details,
            "api_calls": 0, "run_root_created": run_root.exists()}


class LocalSolverRunner:
    """Minimal local boxed-letter runtime; no legacy QA orchestration semantics."""

    def __init__(
        self, *, run_root: Path, ledger: RequestLedger, tokenizer: Any,
        backend: VLLMSolverBackend, dialogue: Sequence[Mapping[str, Any]],
        physical_request_budget: int = 264,
    ):
        self.run_root = run_root
        self.ledger = ledger
        self.tokenizer = tokenizer
        self.backend = backend
        self.dialogue = [dict(row) for row in dialogue]
        self.physical_request_budget = int(physical_request_budget)
        if self.physical_request_budget < 1:
            raise ValueError("simple_append_solver_budget_invalid")

    def _prompt(self, qa: Mapping[str, Any], options: Mapping[str, str],
                mode: str, retry: int) -> str:
        context = self.dialogue if mode == "full_context" else []
        context_text = "\n".join(
            f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in context)
        qa_text = str(qa["question"]) + "\n" + "\n".join(
            f"{letter}. {options[letter]}" for letter in "ABCD")
        user = ((f"Context:\n{context_text}\n\n" if context_text else "") +
                qa_text + "\n\n" + SOLVER_TRAILING_FORMAT_INSTRUCTION)
        if retry:
            user += "\n\n" + SOLVER_POSITION_NEUTRAL_PARSE_RETRY_SUFFIX
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT},
            {"role": "user", "content": user},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def one(
        self, *, qa: Mapping[str, Any], options: Mapping[str, str],
        mode: str, order_id: str, slot: int,
    ) -> dict[str, Any]:
        for retry in range(2):
            prompt = self._prompt(qa, options, mode, retry)
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens + SOLVER_MAX_TOKENS > SOLVER_MAX_MODEL_LEN:
                raise RuntimeError("simple_append_solver_context_overflow")
            seed = int(hashlib.sha256(
                f"{qa['qa_id']}:{mode}:{order_id}:{slot}:{retry}".encode()
            ).hexdigest()[:8], 16)
            request_id = (
                f"solver-{qa['qa_id']}-{mode}-{order_id}-s{slot:02d}-r{retry}")
            request = SolverRequest(
                request_id, str(qa["qa_id"]), mode, order_id, slot, retry,
                seed, dict(options), str(qa["canonical_answer_text"]),
                sha256_bytes(prompt.encode()), prompt)
            atomic_json(self.run_root / f"raw/prompts/{request_id}.json", {
                "request_id": request_id, "prompt_sha256": request.prompt_sha256,
                "prompt": prompt, "enable_thinking": False,
            })
            response = self.ledger.invoke(
                request_id=request_id, intent=f"solver_{mode}", attempt=retry + 1,
                prompt_sha256=request.prompt_sha256,
                call=lambda request=request: self.backend.generate(request),
                budget_class="local_solver",
                budget_limit=self.physical_request_budget,
                total_budget_limit=self.physical_request_budget,
                request_metadata={"qa_id": qa["qa_id"], "mode": mode,
                    "order_id": order_id, "slot": slot, "retry": retry,
                    "seed": seed},
                backend_identity={"backend": "VLLMSolverBackend",
                    "model": self.backend.model_identifier,
                    "temperature": self.backend.temperature,
                    "top_p": self.backend.top_p})
            parsed = parse_boxed_mcq_answer(response.raw_text, dict(options))
            reward = local_reward(parsed, str(qa["canonical_answer_text"]))
            self.ledger.record_decision(
                request_id, "parse_invalid" if reward is None else
                f"valid_reward_{reward}")
            row = {"request_id": request_id, "raw_response": response.raw_text,
                   **parsed, "local_reward": reward, "sampling_seed": seed,
                   "retry_index": retry, "order_id": order_id,
                   "evaluation_mode": mode,
                   "backend_metadata": dict(response.metadata)}
            if reward is not None:
                return row
        return row


def build_local_solver_runtime(
    *, tokenizer_path: Path, solver_model: Path,
) -> tuple[Any, VLLMSolverBackend]:
    """Load the local Solver once so a batch can reuse the model process."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)
    solver = VLLMSolverBackend(
        model_path=solver_model, tokenizer_path=tokenizer_path,
        max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
        max_tokens=SOLVER_MAX_TOKENS,
        gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION)
    return tokenizer, solver


def _evaluate_grounding(
    runner: LocalSolverRunner, qa: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    for shift, order_id in enumerate(
            ("balanced_original", "balanced_perm1", "balanced_perm2",
             "balanced_perm3")):
        options = dict(qa["options"]) if shift == 0 else permuted_options(
            qa["options"], shift)
        gold = gold_letter_for_options(options, str(qa["canonical_answer_text"]))
        row = runner.one(qa=qa, options=options, mode="no_context",
                         order_id=order_id, slot=0)
        records.append({"order_id": order_id, "options": options,
                        "gold_letter": gold, "rollouts": [row]})
        orders.append({"order_id": order_id, "gold_letter": gold,
                       "valid_answer_letters": [row.get("parsed_letter")],
                       "valid_rewards": [row.get("local_reward")]})
    decision = classify_balanced_no_context(orders, samples_per_position=1)
    return decision, records


def _writer_call(
    *, run_root: Path, ledger: RequestLedger, prompt: Mapping[str, Any],
    api_config_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    api = replace(load_challenger_config(api_config_path), timeout_seconds=600,
                  temperature=0.25, enable_thinking=False)
    client = OpenAICompatibleChallengerClient(api)
    request_id = "simple-append-mcq-writer-a00"
    atomic_text(run_root / "raw/prompts" / f"{request_id}.txt", prompt["prompt"])
    response = ledger.invoke(
        request_id=request_id, intent="mcq_writer", attempt=1,
        prompt_sha256=prompt["prompt_sha256"],
        call=lambda: client.complete(
            prompt=prompt["prompt"], max_tokens=WRITER_MAX_TOKENS,
            json_object=True),
        budget_class="mcq_writer", budget_limit=1, total_budget_limit=265,
        request_metadata={"role": "simple_append_joint_mcq_writer_v13",
            "question_count": QUESTION_COUNT, "retry_allowed": False},
        backend_identity={"transport": "openai_compatible", "model": api.model,
            "base_url": api.base_url, "temperature": api.temperature})
    try:
        parsed = load_json_object_strict(response.raw_text)
        parse_error = None
    except (json.JSONDecodeError, ValueError) as error:
        parsed = None
        parse_error = f"{type(error).__name__}:{error}"
    usage = ((response.metadata or {}).get("usage") or {})
    return parsed, {"request_id": request_id, "parse_error": parse_error,
                    "model": response.model_identifier,
                    "usage": usage, "response_metadata": response.metadata}


def _evaluate_solver_candidates(
    *, raw: Mapping[str, Any], partition: Mapping[str, Any],
    expanded: Mapping[str, Any], source: Mapping[str, Any],
    run_root: Path, manifest_sha: str, tokenizer_path: Path,
    solver_model: Path,
    solver_runtime: tuple[Any, VLLMSolverBackend] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize and evaluate one Writer-admitted batch locally."""
    items: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    focus_by_candidate = {
        str(row.get("candidate_id")): row
        for row in (partition.get("focus_round") or {}).get("items") or []}
    for row, status in zip(raw["questions"], partition["items"]):
        if not status["solver_ready"]:
            items.append({
                "candidate_id": status["candidate_id"],
                "qa_id": None,
                "question": (row.get("question")
                             if isinstance(row, Mapping) else None),
                "solver_ready": False,
                "evidence_ready": False,
                "evidence_issues": status["evidence_issues"],
                "structural_errors": status["structural_errors"],
                "decision": "candidate_structural_quarantine",
                "admitted": False,
            })
            continue
        materialized.append(materialize_solver_candidate(
            row, index=int(status["question_index"]),
            expanded_dialogue=expanded,
            span_catalog=source["span_catalog"],
            candidate_status=status))
    atomic_json(run_root / "qa/final_mcq_batch.json", {
        "batch_version": RUNNER_VERSION, "root_id": expanded["root_id"],
        "items": materialized})
    for qa in materialized:
        atomic_json(run_root / f"qa/{qa['qa_id']}.json", qa)

    if solver_runtime is None:
        tokenizer, solver = build_local_solver_runtime(
            tokenizer_path=tokenizer_path, solver_model=solver_model)
    else:
        tokenizer, solver = solver_runtime
    local_ledger = RequestLedger(run_root, manifest_sha, "local")
    runner = LocalSolverRunner(
        run_root=run_root, ledger=local_ledger, tokenizer=tokenizer,
        backend=solver, dialogue=source["combined_dialogue"],
        physical_request_budget=max(264, len(materialized) * 22))
    for qa in materialized:
        grounding, ground_records = _evaluate_grounding(runner, qa)
        full_records: list[dict[str, Any]] = []
        if grounding["decision"] == "pass_grounding":
            for slot in range(5):
                full_records.append(runner.one(
                    qa=qa, options=qa["options"], mode="full_context",
                    order_id="original", slot=slot))
        full_rewards = [row.get("local_reward") for row in full_records]
        frontier = classify_full_context_frontier(full_rewards)
        full_correct = sum(int(value) for value in full_rewards
                           if value in (0, 1))
        admitted = (grounding["decision"] == "pass_grounding" and
                    frontier["admitted"] is True)
        decision = ("admit_frontier_k5" if admitted else
                    frontier["decision"] if full_records else
                    grounding["decision"])
        item = {"qa_id": qa["qa_id"], "question": qa["question"],
            "candidate_id": qa["candidate_id"],
            "solver_ready": True,
            "grounding": grounding, "no_context_orders": ground_records,
            "full_context_rollouts": full_records,
            "full_context_correct": full_correct,
            "full_context_total": len(full_records),
            "full_context_frontier": frontier,
            "full_context_bucket": frontier["bucket"],
            "evidence_ready": qa["evidence_ready"],
            "evidence_issues": qa["evidence_issues"],
            "invalid_evidence_turn_ids": qa["invalid_evidence_turn_ids"],
            "source_scope": qa["source_scope"],
            "semantic_evidence_audit_executed": False,
            "evidence_mismatch_policy": "mark_only_if_detected",
            **focus_by_candidate.get(str(qa["candidate_id"]), {}),
            "decision": decision, "admitted": admitted}
        items.append(item)
        atomic_json(run_root / f"filter/{qa['qa_id']}.json", item)
        atomic_json(run_root / f"reports/{qa['qa_id']}_evaluation.json", item)
    return items, materialized


def run(
    *, source_run: Path, run_root: Path, api_config_path: Path,
    tokenizer_path: Path, solver_model: Path,
    focus_round_index: int | None = None,
    writer_output_replay: Path | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    frozen = preflight(source_run=source_run, run_root=run_root,
                       api_config_path=api_config_path,
                       tokenizer_path=tokenizer_path,
                       solver_model=solver_model,
                       focus_round_index=focus_round_index,
                       writer_output_replay=writer_output_replay)
    if preflight_only:
        return frozen
    if frozen["decision"] != "pass":
        raise RuntimeError("PRE_API_PREFLIGHT_FAILED:" +
                           ",".join(frozen["failed_checks"]))
    expanded = _load(source_run / "expanded_dialogue.json")
    source = build_solver_source(expanded)
    evidence_aliases = build_evidence_alias_registry(expanded)
    effective_focus_round = int(frozen["details"]["focus_round_index"])
    prompt = build_mcq_batch_prompt(
        expanded, target_questions=QUESTION_COUNT,
        maximum_questions=MAX_QUESTION_COUNT,
        focus_round_index=effective_focus_round)
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("config", "frozen", "qa", "filter", "reports",
                     "raw/prompts"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {"freeze_version": RUNNER_VERSION, "created_at": _now(),
        "source_run": str(source_run), "source_run_tree_sha256":
            frozen["details"]["source_run_tree_sha256"],
        "expanded_dialogue_sha256": frozen["details"]["expanded_dialogue_sha256"],
        "source_manifest": _source_manifest(),
        "writer_prompt_sha256": prompt["prompt_sha256"],
        "evidence_alias_schema_version":
            evidence_aliases["evidence_alias_schema_version"],
        "evidence_alias_registry_sha256": evidence_aliases["registry_sha256"],
        "request_budget": {"mcq_writer": (
                0 if writer_output_replay is not None else 1),
            "minimum_candidate_questions": QUESTION_COUNT,
            "maximum_candidate_questions": None,
            "no_context_per_question_base": 4,
            "no_context_per_question_boundary_extension": 0,
            "no_context_acceptance_rule": (
                "any_wrong_answer_passes_unless_position_bias"),
            "full_context_per_grounded_question": 5,
            "full_context_acceptance_rule": "spell_frontier_1_to_4_of_5",
            "local_solver_physical_max_including_parse_retry": 264},
        "writer_batch_contract": {
            "min_child_supported": MIN_CHILD_SUPPORTED,
            "min_joint_parent_child": MIN_JOINT_PARENT_CHILD,
            "focus_round_index": effective_focus_round,
            "min_focus_round_supported": MIN_FOCUS_ROUND_SUPPORTED,
            "min_focus_round_joint_with_history":
                MIN_FOCUS_ROUND_JOINT_WITH_HISTORY,
            "writer_focus_failure_skips_solver": False,
            "quota_deficits_trigger_targeted_top_up": True,
            "quota_deficits_block_solver": False,
            "target_distinct_question_types": TARGET_DISTINCT_QUESTION_TYPES,
            "per_question_source_scope_binding": False,
            "question_count_upper_admission_gate": None,
            "evidence_count_preferred_minimum": 1,
            "evidence_count_preferred_maximum": 4,
            "evidence_defects_block_solver": False,
            "writer_evidence_reference_format": "D###",
            "canonical_turn_ids_program_resolved": True},
        "forbidden_calls": {"planner": 0, "realizer": 0, "memory": 0,
                            "semantic_auditor": 0},
        "writer_output_replay": (
            {"path": str(writer_output_replay),
             "sha256": sha256_file(writer_output_replay)}
            if writer_output_replay is not None else None),
        "preflight": frozen}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "frozen/expanded_dialogue.json", expanded)
    atomic_json(run_root / "frozen/questioner_smoke_context.json", {
        "combined_dialogue": source["combined_dialogue"]})
    atomic_json(run_root / "frozen/questioner_span_catalog.json",
                source["span_catalog"])
    atomic_json(run_root / "frozen/evidence_alias_registry.json",
                evidence_aliases)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "api_calls": 0})
    manifest_sha = sha256_file(run_root / "run_freeze_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "live")
    failure: str | None = None
    writer_meta: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    partition: dict[str, Any] = {}
    materialized: list[dict[str, Any]] = []
    try:
        if writer_output_replay is None:
            raw, writer_meta = _writer_call(
                run_root=run_root, ledger=ledger, prompt=prompt,
                api_config_path=api_config_path)
        else:
            raw = _load(writer_output_replay)
            writer_meta = {
                "replayed": True,
                "source_path": str(writer_output_replay),
                "source_sha256": sha256_file(writer_output_replay),
                "usage": {},
            }
        atomic_json(run_root / "qa/writer_output.json", raw)
        if raw is None:
            raise RuntimeError("simple_append_mcq_writer_parse_failed")
        overflow = retain_focus_writer_overflow(
            raw, target_count=QUESTION_COUNT,
            maximum_count=MAX_QUESTION_COUNT)
        atomic_json(run_root / "qa/writer_overflow_audit.json", overflow)
        writer_envelope = normalize_focus_writer_batch(overflow["value"])
        atomic_json(run_root / "qa/writer_focus_envelope.json",
                    writer_envelope)
        if not writer_envelope["valid"]:
            raise RuntimeError("simple_append_mcq_writer_focus_envelope_failed")
        normalized_raw = writer_envelope["batch"]
        atomic_json(run_root / "qa/writer_output_normalized.json",
                    normalized_raw)
        partition = partition_mcq_candidates(
            normalized_raw, expanded_dialogue=expanded,
            minimum_count=QUESTION_COUNT,
            maximum_count=None,
            min_child_supported=MIN_CHILD_SUPPORTED,
            min_joint_parent_child=MIN_JOINT_PARENT_CHILD)
        partition["parse_error"] = writer_meta.get("parse_error")
        partition["question_type_diversity_target"] = (
            TARGET_DISTINCT_QUESTION_TYPES)
        partition["question_type_diversity_target_met"] = (
            len(partition.get("distinct_question_types") or []) >=
            TARGET_DISTINCT_QUESTION_TYPES)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded,
            focus_round_index=effective_focus_round,
            pool_assignments=writer_envelope["pool_assignments"])
        partition["focus_round"] = focus
        partition["focus_round_profile_ready"] = focus[
            "focus_round_profile_ready"]
        quota = evaluate_qa_quota_status(partition, focus)
        partition.update(quota)
        partition["writer_admission_ready"] = bool(
            partition["envelope_valid"] and
            partition["solver_ready_count"])
        atomic_json(run_root / "qa/writer_validation.json", partition)
        if (not partition["envelope_valid"] or
                partition["solver_ready_count"] == 0):
            raise RuntimeError("simple_append_mcq_writer_envelope_failed")
        items, materialized = _evaluate_solver_candidates(
            raw=normalized_raw, partition=partition, expanded=expanded,
            source=source, run_root=run_root,
            manifest_sha=manifest_sha, tokenizer_path=tokenizer_path,
            solver_model=solver_model)
    except Exception as error:
        failure = f"{type(error).__name__}:{error}"

    admitted_ids = [row["qa_id"] for row in items if row.get("admitted")]
    if (run_root / "qa/final_mcq_batch.json").is_file():
        materialized = _load(run_root / "qa/final_mcq_batch.json")["items"]
        accepted = [qa for qa in materialized if qa["qa_id"] in admitted_ids]
        atomic_json(run_root / "qa/admitted_mcq_batch.json", {
            "batch_version": RUNNER_VERSION, "items": accepted})
    states = [_load(path) for path in (run_root / "ledger/requests").glob("*.json")]
    writer_states = [row for row in states if row.get("budget_class") == "mcq_writer"]
    solver_states = [row for row in states if row.get("budget_class") == "local_solver"]
    source_unchanged = _hash_tree(source_run) == frozen["details"]["source_run_tree_sha256"]
    admitted_scopes = [qa.get("source_scope") for qa in materialized
                       if qa.get("qa_id") in admitted_ids]
    admitted_child_supported = sum(
        scope in {"child_only", "parent_child"} for scope in admitted_scopes)
    batch_profile_ready = bool(partition.get("batch_profile_ready"))
    focus = partition.get("focus_round") or {}
    writer_admission_ready = bool(partition.get("writer_admission_ready"))
    admitted_current_round = sum(
        bool(row.get("admitted") and row.get("focus_round_supported"))
        for row in items)
    admitted_joint_history = sum(
        bool(row.get("admitted") and
             row.get("focus_round_joint_with_history"))
        for row in items)
    admitted_parent_child = admitted_scopes.count("parent_child")
    post_solver_quota_deficits: list[str] = []
    if admitted_child_supported < MIN_CHILD_SUPPORTED:
        post_solver_quota_deficits.append("admitted_child_supported_below_minimum")
    if admitted_parent_child < MIN_JOINT_PARENT_CHILD:
        post_solver_quota_deficits.append("admitted_parent_child_below_minimum")
    if admitted_current_round < MIN_FOCUS_ROUND_SUPPORTED:
        post_solver_quota_deficits.append("admitted_current_round_below_minimum")
    if admitted_joint_history < MIN_FOCUS_ROUND_JOINT_WITH_HISTORY:
        post_solver_quota_deficits.append(
            "admitted_current_round_joint_history_below_minimum")
    top_up_required = bool(post_solver_quota_deficits)
    qa_pipeline_ready = bool(
        writer_admission_ready and admitted_ids and not top_up_required)
    qa_stage_status = (
        "system_failed" if failure is not None else
        "completed_top_up_required" if top_up_required else
        "completed")
    candidate_count = len(partition.get("items") or [])
    report = {"report_version": RUNNER_VERSION,
        "status": "completed" if failure is None else "failed",
        "qa_stage_status": qa_stage_status,
        "failure": failure, "source_run": str(source_run),
        "run_root": str(run_root), "question_count": candidate_count,
        "solver_evaluated_count": len(items),
        "admitted_count": len(admitted_ids), "quarantined_count":
            candidate_count - len(admitted_ids), "items": items,
        "candidate_partition": partition,
        "batch_profile_ready": batch_profile_ready,
        "focus_round_index": effective_focus_round,
        "focus_round_supported_count":
            int(focus.get("focus_round_supported_count") or 0),
        "focus_round_joint_with_history_count": int(
            focus.get("focus_round_joint_with_history_count") or 0),
        "focus_round_profile_ready": bool(
            focus.get("focus_round_profile_ready")),
        "focus_round_profile_errors": list(
            focus.get("focus_round_profile_errors") or []),
        "writer_admission_ready": writer_admission_ready,
        "pre_solver_quota_profile_ready": bool(
            partition.get("quota_profile_ready")),
        "pre_solver_quota_deficits": list(
            partition.get("quota_deficits") or []),
        "coverage_role_warnings": list(
            focus.get("coverage_role_warnings") or []),
        "admitted_child_supported_count": admitted_child_supported,
        "admitted_current_round_count": admitted_current_round,
        "admitted_current_round_joint_with_history_count":
            admitted_joint_history,
        "admitted_parent_child_count": admitted_parent_child,
        "top_up_required": top_up_required,
        "post_solver_quota_deficits": post_solver_quota_deficits,
        "existing_valid_qa_preserved_when_quota_incomplete": True,
        "distinct_question_type_count": len(
            partition.get("distinct_question_types") or []),
        "question_type_diversity_target_met": bool(
            partition.get("question_type_diversity_target_met")),
        "api_statistics": {"mcq_writer_requests": len(writer_states),
            "writer_replay_count": int(writer_output_replay is not None),
            "local_solver_generations": len(solver_states),
            "planner_calls": 0, "realizer_calls": 0, "memory_calls": 0,
            "semantic_auditor_calls": 0,
            "writer_usage": writer_meta.get("usage") or {}},
        "source_unchanged": source_unchanged,
        "generation_ready": bool(candidate_count),
        "qa_pipeline_ready": qa_pipeline_ready, "training_ready": False,
        "completed_at": _now()}
    atomic_json(run_root / "reports/SIMPLE_APPEND_QA_ONLY_REPORT.json", report)
    lines = ["# Simple Append QA-only Pilot", "",
        f"- Status: `{report['status']}`",
        f"- Questions: `{report['question_count']}`",
        f"- Writer admission: `{writer_admission_ready}`",
        f"- Current-round candidates: "
        f"`{report['focus_round_supported_count']}`",
        f"- Current-round admitted: `{admitted_current_round}`",
        f"- Admitted: `{report['admitted_count']}`",
        f"- Writer requests: `{len(writer_states)}`",
        f"- Local Solver generations: `{len(solver_states)}`",
        f"- Source unchanged: `{source_unchanged}`", ""]
    for row in items:
        lines.extend([f"## {row['qa_id']}", "",
            f"- Grounding: `{row['grounding']['decision']}`",
            f"- Full context: `{row['full_context_correct']}/{row['full_context_total']}`",
            f"- Decision: `{row['decision']}`", ""])
    atomic_text(run_root / "reports/SIMPLE_APPEND_QA_ONLY_REPORT.md",
                "\n".join(lines))
    atomic_json(run_root / "run_status.json", {
        "status": report["status"], "failure": failure,
        "qa_stage_status": report["qa_stage_status"],
        "writer_admission_ready": report["writer_admission_ready"],
        "qa_pipeline_ready": report["qa_pipeline_ready"],
        "api_statistics": report["api_statistics"],
        "completed_at": report["completed_at"]})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simple append QA-only pilot")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--focus-round-index", type=int)
    parser.add_argument("--writer-output-replay", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(source_run=args.source_run.resolve(),
                 run_root=args.run_root.resolve(),
                 api_config_path=args.api_config.resolve(),
                 tokenizer_path=args.tokenizer.resolve(),
                 solver_model=args.solver_model.resolve(),
                 focus_round_index=args.focus_round_index,
                 writer_output_replay=(
                     args.writer_output_replay.resolve()
                     if args.writer_output_replay else None),
                 preflight_only=args.preflight_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report.get("decision") == "pass" if args.preflight_only else
                 report.get("status") == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
