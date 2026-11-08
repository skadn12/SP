"""B0 K=3 local Solver execution with canonical scoring and exact resume keys."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..backends import BackendResponse, SolverRequest
from ..core import canonical_json, sha256_bytes, sha256_file
from ..structured_qa_box_protocol import structured_box_choices
from ..vllm_backends import VLLMSolverBackend
from . import (
    BASE_SOLVER_MODEL,
    BASE_TOKENIZER,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    PARSER_SINGLE,
    PARSER_STRUCTURED,
    PROBE_VERSION,
    ROLLOUTS,
    SEED_SCHEDULE_VERSION,
    SINGLE_TEMPERATURE,
    STRUCTURED_OUTPUT_VERSION,
    STRUCTURED_TEMPERATURE,
    TOP_P,
)
from .contract import (
    CANONICAL_SCORING_VERSION,
    build_prompt,
    dialogue_rows,
    position_variants,
    score_output,
)
from .gate import (
    PROGRAM_PASS,
    assert_solver_paths,
    detect_structured_output_api,
    directory_identity,
)
from .io_utils import (
    load_json,
    load_jsonl,
    stable_seed,
    verify_sha_sidecar,
    write_once_json,
)
from .request_ledger import (
    RecordedSolverFailure,
    Round0RequestLedger,
    request_cache_key,
)


MAX_CONSECUTIVE_SOLVER_FAILURES = 3


def ensure_vllm_environment() -> None:
    current = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if current not in {None, "0"}:
        raise RuntimeError(f"VLLM_USE_FLASHINFER_SAMPLER_must_be_0:{current}")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def create_backend(
    *,
    model_path: Path = BASE_SOLVER_MODEL,
    tokenizer_path: Path = BASE_TOKENIZER,
    gpu_memory_utilization: float = 0.80,
) -> VLLMSolverBackend:
    model, tokenizer = assert_solver_paths(model_path, tokenizer_path)
    ensure_vllm_environment()
    return VLLMSolverBackend(
        model_path=model,
        tokenizer_path=tokenizer,
        max_model_len=MAX_MODEL_LEN,
        temperature=SINGLE_TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _structured_generate(
    backend: Any,
    request: SolverRequest,
    *,
    question_type: str,
) -> BackendResponse:
    hook = getattr(backend, "generate_structured", None)
    if callable(hook):
        return hook(request, question_type=question_type)
    from vllm import SamplingParams, sampling_params

    allowed = request.options.keys() if question_type in {"multi_select", "ordering"} else None
    choices = structured_box_choices(question_type, allowed_item_ids=allowed)
    if not choices:
        raise ValueError(f"structured_choices_empty:{question_type}")
    api_name, _ = detect_structured_output_api()
    common = {
        "temperature": STRUCTURED_TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": request.sampling_seed,
        "stop": ["</box>"],
        "include_stop_str_in_output": True,
    }
    if api_name.startswith("structured_outputs="):
        params = SamplingParams(
            **common,
            structured_outputs=sampling_params.StructuredOutputsParams(choice=choices),
        )
    else:
        params = SamplingParams(
            **common,
            guided_decoding=sampling_params.GuidedDecodingParams(choice=choices),
        )
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    output = outputs[0]
    completion = output.outputs[0]
    return BackendResponse(
        completion.text,
        backend.model_identifier,
        {
            "sampling_seed": request.sampling_seed,
            "prompt_token_count": len(output.prompt_token_ids),
            "completion_token_count": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "temperature": STRUCTURED_TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "structured_output_api": api_name,
            "structured_output_version": STRUCTURED_OUTPUT_VERSION,
        },
    )


def _parser_version(question_type: str) -> str:
    return PARSER_SINGLE if question_type == "single_choice" else PARSER_STRUCTURED


def _temperature(question_type: str) -> float:
    return SINGLE_TEMPERATURE if question_type == "single_choice" else STRUCTURED_TEMPERATURE


def _truncated_score(question_type: str) -> dict[str, Any]:
    return {
        "parse_status": "output_truncated",
        "parser_version": _parser_version(question_type),
        "scored": False,
        "correct": None,
        "display_prediction": [],
        "canonical_prediction": [],
        "failure_type": "solver_output_truncated",
        "parse_reason": "vllm_finish_reason_length",
        "metrics": {},
    }


def backend_identity(backend: Any, question_type: str) -> dict[str, Any]:
    return {
        "backend": type(backend).__name__,
        "model": str(backend.model_identifier),
        "tokenizer": str(getattr(backend, "tokenizer_path", BASE_TOKENIZER.resolve())),
        "max_model_len": int(getattr(backend, "max_model_len", MAX_MODEL_LEN)),
        "temperature": _temperature(question_type),
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }


def generation_config(question_type: str) -> dict[str, Any]:
    structured = question_type != "single_choice"
    structured_api, vllm_version = detect_structured_output_api()
    return {
        "temperature": _temperature(question_type),
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "structured_output_api": structured_api if structured else None,
        "structured_output_version": STRUCTURED_OUTPUT_VERSION if structured else None,
        "vllm_version": vllm_version,
        "stop": ["</box>"] if structured else [],
        "include_stop_str_in_output": structured,
        "enable_thinking": False,
    }


def _request_contract(
    *,
    solver_contract_sha256: str,
    root_id: str,
    candidate_id: str,
    question_type: str,
    variant_id: str,
    context_mode: str,
    slot_id: int,
    seed: int,
    prompt_sha256: str,
    backend: Any,
    candidate_pool_sha256: str,
    dialogue_sha256: str,
    display_to_canonical: Mapping[str, str],
    canonical_gold: Iterable[str],
) -> dict[str, Any]:
    return {
        "request_contract_version": "h4_round0_b0_request_key_v1",
        "probe_version": PROBE_VERSION,
        "solver_contract_sha256": solver_contract_sha256,
        "root_id": root_id,
        "candidate_id": candidate_id,
        "question_type": question_type,
        "variant_id": variant_id,
        "context_mode": context_mode,
        "slot_id": slot_id,
        "seed": seed,
        "seed_schedule_version": SEED_SCHEDULE_VERSION,
        "prompt_sha256": prompt_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
        "dialogue_sha256": dialogue_sha256,
        "display_to_canonical": dict(display_to_canonical),
        "canonical_gold": [str(value) for value in canonical_gold],
        "backend_identity": backend_identity(backend, question_type),
        "parser_version": _parser_version(question_type),
        "canonical_scoring_version": CANONICAL_SCORING_VERSION,
        "generation_config": generation_config(question_type),
    }


def _expected_prompt_audit(gate_row: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["variant_id"]), str(row["context_mode"])): row
        for row in gate_row.get("prompt_audit") or []
    }


def evaluate_candidate(
    *,
    candidate: Mapping[str, Any],
    expanded: Mapping[str, Any],
    gate_row: Mapping[str, Any],
    tokenizer: Any,
    backend: Any,
    ledger: Round0RequestLedger,
    solver_contract_sha256: str,
) -> dict[str, Any]:
    if gate_row.get("gate_status") != PROGRAM_PASS:
        raise RuntimeError("cannot_execute_non_program_pass_candidate")
    root_id = str(candidate["root_id"])
    candidate_id = str(candidate["candidate_id"])
    qtype = str(candidate["question_type"])
    if candidate_id != gate_row.get("candidate_id") or root_id != gate_row.get("root_id"):
        raise RuntimeError("candidate_gate_identity_mismatch")
    variants = position_variants(candidate)
    dialogue = dialogue_rows(expanded)
    expected_prompts = _expected_prompt_audit(gate_row)
    requests: list[dict[str, Any]] = []
    consecutive_solver_failures = 0
    for variant in variants:
        for context_mode in ("no_context", "full_context"):
            prompt = build_prompt(
                tokenizer, candidate, variant, dialogue, context_mode=context_mode,
            )
            prompt_sha = sha256_bytes(prompt.encode("utf-8"))
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            expected = expected_prompts.get((variant["variant_id"], context_mode))
            if not expected or expected.get("prompt_sha256") != prompt_sha:
                raise RuntimeError(
                    f"preflight_prompt_drift:{candidate_id}:{variant['variant_id']}:{context_mode}"
                )
            if int(expected["prompt_token_count"]) != prompt_tokens:
                raise RuntimeError("preflight_prompt_token_count_drift")
            if prompt_tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError("program_pass_became_context_overflow")
            display = dict(variant.get("options") or variant.get("items") or {})
            for slot_id in range(ROLLOUTS):
                seed = stable_seed(root_id, candidate_id, slot_id)
                request_id = (
                    f"{candidate_id}__{variant['variant_id']}__{context_mode}__s{slot_id:02d}"
                )
                request = SolverRequest(
                    request_id=request_id,
                    qa_id=candidate_id,
                    evaluation_mode=context_mode,
                    order_id=str(variant["variant_id"]),
                    target_slot=slot_id,
                    retry_index=0,
                    sampling_seed=seed,
                    options=display,
                    canonical_answer_text=",".join(variant["displayed_gold"]),
                    prompt_sha256=prompt_sha,
                    prompt=prompt,
                )
                contract = _request_contract(
                    solver_contract_sha256=solver_contract_sha256,
                    root_id=root_id,
                    candidate_id=candidate_id,
                    question_type=qtype,
                    variant_id=str(variant["variant_id"]),
                    context_mode=context_mode,
                    slot_id=slot_id,
                    seed=seed,
                    prompt_sha256=prompt_sha,
                    backend=backend,
                    candidate_pool_sha256=str(gate_row["candidate_pool_sha256"]),
                    dialogue_sha256=str(gate_row["dialogue_sha256"]),
                    display_to_canonical=variant["display_to_canonical"],
                    canonical_gold=variant["canonical_gold"],
                )
                cache_key = request_cache_key(contract)
                try:
                    response, _ = ledger.invoke(
                        contract=contract,
                        prompt=prompt,
                        call=(
                            (lambda request=request: backend.generate(request))
                            if qtype == "single_choice"
                            else (
                                lambda request=request, qtype=qtype: _structured_generate(
                                    backend, request, question_type=qtype,
                                )
                            )
                        ),
                    )
                except RecordedSolverFailure as error:
                    failed = ledger.read(root_id, cache_key) or {}
                    error_text = f"{failed.get('error_type')}:{failed.get('error')}"
                    if (
                        not error.cached
                        and (
                            "out of memory" in error_text.casefold()
                            or "oom" in error_text.casefold()
                        )
                    ):
                        raise RuntimeError(f"solver_oom_stop:{cache_key}:{error_text}")
                    if not error.cached:
                        consecutive_solver_failures += 1
                    requests.append({
                        "request_id": request_id,
                        "request_cache_key": cache_key,
                        "root_id": root_id,
                        "candidate_id": candidate_id,
                        "question_type": qtype,
                        "variant_id": variant["variant_id"],
                        "shift": variant["shift"],
                        "context_mode": context_mode,
                        "slot_id": slot_id,
                        "seed": seed,
                        "prompt_sha256": prompt_sha,
                        "prompt_token_count": prompt_tokens,
                        "display_to_canonical": variant["display_to_canonical"],
                        "displayed_gold": variant["displayed_gold"],
                        "canonical_gold": variant["canonical_gold"],
                        "raw_output": None,
                        "backend_metadata": {},
                        "parse_status": "not_run",
                        "parser_version": _parser_version(qtype),
                        "scored": False,
                        "correct": None,
                        "display_prediction": [],
                        "canonical_prediction": [],
                        "failure_type": "solver_error",
                        "error": error_text,
                        "metrics": {},
                    })
                    if (
                        not error.cached
                        and consecutive_solver_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES
                    ):
                        raise RuntimeError(
                            f"consecutive_solver_failure_stop:{MAX_CONSECUTIVE_SOLVER_FAILURES}"
                        )
                    continue
                consecutive_solver_failures = 0
                if str(response.model_identifier) != str(backend.model_identifier):
                    raise RuntimeError("response_model_identifier_drift")
                finish_reason = str(
                    (response.metadata or {}).get("finish_reason") or ""
                ).casefold()
                scored = (
                    _truncated_score(qtype)
                    if finish_reason == "length"
                    else score_output(response.raw_text, candidate, variant)
                )
                ledger.record_decision(
                    root_id,
                    cache_key,
                    {
                        "parser_version": scored["parser_version"],
                        "parse_status": scored["parse_status"],
                        "scored": scored["scored"],
                        "correct": scored["correct"],
                        "canonical_prediction": scored["canonical_prediction"],
                        "failure_type": scored["failure_type"],
                    },
                )
                requests.append({
                    "request_id": request_id,
                    "request_cache_key": cache_key,
                    "root_id": root_id,
                    "candidate_id": candidate_id,
                    "question_type": qtype,
                    "variant_id": variant["variant_id"],
                    "shift": variant["shift"],
                    "context_mode": context_mode,
                    "slot_id": slot_id,
                    "seed": seed,
                    "prompt_sha256": prompt_sha,
                    "prompt_token_count": prompt_tokens,
                    "display_to_canonical": variant["display_to_canonical"],
                    "displayed_gold": variant["displayed_gold"],
                    "canonical_gold": variant["canonical_gold"],
                    "raw_output": response.raw_text,
                    "backend_metadata": {
                        key: value for key, value in dict(response.metadata or {}).items()
                        if key != "reused"
                    },
                    **scored,
                })
    expected_count = len(variants) * 2 * ROLLOUTS
    if len(requests) != expected_count:
        raise RuntimeError(f"candidate_request_count_mismatch:{len(requests)}:{expected_count}")
    return {
        "candidate_result_version": "h4_round0_b0_k3_candidate_result_v1",
        "probe_version": PROBE_VERSION,
        "solver_contract_sha256": solver_contract_sha256,
        "root_id": root_id,
        "candidate_id": candidate_id,
        "question_type": qtype,
        "position_count": len(variants),
        "candidate_pool_sha256": gate_row["candidate_pool_sha256"],
        "dialogue_sha256": gate_row["dialogue_sha256"],
        "request_count": len(requests),
        "technical_failure_count": sum(not bool(row["scored"]) for row in requests),
        "requests": requests,
    }


def candidate_result_path(run_root: Path, candidate_id: str) -> Path:
    digest = sha256_bytes(candidate_id.encode("utf-8"))
    return run_root / "candidate_results" / f"{digest}.json"


def load_or_recover_candidate_result(
    *,
    run_root: Path,
    candidate: Mapping[str, Any],
    expanded: Mapping[str, Any],
    gate_row: Mapping[str, Any],
    tokenizer: Any,
    backend: Any,
    ledger: Round0RequestLedger,
    solver_contract_sha256: str,
) -> dict[str, Any] | None:
    candidate_id = str(candidate["candidate_id"])
    root_id = str(candidate["root_id"])
    result_path = candidate_result_path(run_root, candidate_id)
    if not result_path.is_file():
        return None
    result_sidecar = result_path.with_name(result_path.name + ".sha256")
    if result_sidecar.is_file():
        verify_sha_sidecar(result_path)
        return load_json(result_path)
    existing_result = load_json(result_path)
    for request_row in existing_result.get("requests") or []:
        request_state = ledger.read(root_id, str(request_row["request_cache_key"]))
        if not request_state or request_state.get("status") not in {"completed", "failed"}:
            raise RuntimeError(f"candidate_result_without_terminal_ledger:{candidate_id}")
    reconstructed = evaluate_candidate(
        candidate=candidate,
        expanded=expanded,
        gate_row=gate_row,
        tokenizer=tokenizer,
        backend=backend,
        ledger=ledger,
        solver_contract_sha256=solver_contract_sha256,
    )
    write_once_json(result_path, reconstructed)
    return reconstructed


def _load_sources_by_root(gate_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for row in gate_rows:
        root_id = str(row["root_id"])
        sources.setdefault(root_id, {
            "candidate_pool_path": row["candidate_pool_path"],
            "candidate_pool_sha256": row["candidate_pool_sha256"],
            "dialogue_path": row["dialogue_path"],
            "dialogue_sha256": row["dialogue_sha256"],
        })
    return sources


def _load_root_source(source: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pool_path = Path(str(source["candidate_pool_path"]))
    dialogue_path = Path(str(source["dialogue_path"]))
    if sha256_file(pool_path) != source["candidate_pool_sha256"]:
        raise RuntimeError(f"candidate_pool_source_drift:{pool_path}")
    if sha256_file(dialogue_path) != source["dialogue_sha256"]:
        raise RuntimeError(f"dialogue_source_drift:{dialogue_path}")
    pool = load_json(pool_path)
    return {str(row["candidate_id"]): row for row in pool}, load_json(dialogue_path)


def validate_solver_runtime(
    run_root: Path, tokenizer: Any,
) -> tuple[str, dict[str, Any]]:
    """Validate the frozen runtime before any vLLM engine is created."""
    contract_path = run_root / "solver_contract.json"
    contract_sha = verify_sha_sidecar(contract_path)
    contract = load_json(contract_path)
    runtime_api, runtime_vllm = detect_structured_output_api()
    frozen_generation = contract["generation_config"]
    if runtime_api != frozen_generation["structured_output_api"]:
        raise RuntimeError("structured_output_api_drift_since_preflight")
    if runtime_vllm != frozen_generation["vllm_version"]:
        raise RuntimeError("vllm_version_drift_since_preflight")
    sources = list(contract.get("contract_source_files") or [])
    if not sources:
        raise RuntimeError("solver_contract_source_identity_missing")
    for source in sources:
        path = Path(str(source["path"]))
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"solver_contract_source_drift:{path}")
    current_checkpoint_identity = directory_identity(Path(contract["solver_checkpoint"]))
    if current_checkpoint_identity != contract["checkpoint_identity"]:
        raise RuntimeError("solver_checkpoint_content_drift_since_preflight")
    if contract["tokenizer"] == contract["solver_checkpoint"]:
        current_tokenizer_identity = current_checkpoint_identity
    else:
        current_tokenizer_identity = directory_identity(Path(contract["tokenizer"]))
    if current_tokenizer_identity != contract["tokenizer_identity"]:
        raise RuntimeError("tokenizer_content_drift_since_preflight")
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    if sha256_bytes(chat_template.encode("utf-8")) != contract["chat_template_sha256"]:
        raise RuntimeError("tokenizer_chat_template_drift")
    return contract_sha, contract


def execute_scope(
    *,
    run_root: Path,
    scope: str,
    tokenizer: Any,
    backend: Any,
    validated_runtime: tuple[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if scope not in {"smoke", "batch"}:
        raise ValueError(f"unsupported_execution_scope:{scope}")
    if validated_runtime is None:
        contract_sha, contract = validate_solver_runtime(run_root, tokenizer)
    else:
        contract_sha, frozen_contract = validated_runtime
        contract = dict(frozen_contract)
        if verify_sha_sidecar(run_root / "solver_contract.json") != contract_sha:
            raise RuntimeError("solver_contract_drift_after_runtime_validation")
    if str(backend.model_identifier) != contract["solver_checkpoint"]:
        raise RuntimeError("backend_not_frozen_base_solver")
    if str(getattr(backend, "tokenizer_path", BASE_TOKENIZER.resolve())) != contract["tokenizer"]:
        raise RuntimeError("backend_tokenizer_not_frozen")
    gate_path = run_root / "program_gate_results.jsonl"
    plan_path = run_root / "solver_plan.json"
    verify_sha_sidecar(gate_path)
    verify_sha_sidecar(plan_path)
    gate_rows = load_jsonl(gate_path)
    plan = load_json(plan_path)
    passed = [row for row in gate_rows if row["gate_status"] == PROGRAM_PASS]
    if scope == "smoke":
        wanted = {row["candidate_id"] for row in plan["smoke_candidates"]}
        selected = [row for row in passed if row["candidate_id"] in wanted]
    else:
        smoke_path = run_root / "smoke_report.json"
        verify_sha_sidecar(smoke_path)
        smoke_report = load_json(smoke_path)
        if smoke_report.get("status") != "PASS":
            raise RuntimeError("passing_smoke_required_before_batch")
        selected = passed
    sources = _load_sources_by_root(selected)
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_root[str(row["root_id"])].append(row)
    ledger = Round0RequestLedger(run_root, solver_contract_sha256=contract_sha)
    completed: list[dict[str, Any]] = []
    for root_id in sorted(by_root):
        candidates, expanded = _load_root_source(sources[root_id])
        for gate_row in sorted(by_root[root_id], key=lambda row: row["candidate_id"]):
            candidate_id = str(gate_row["candidate_id"])
            result_path = candidate_result_path(run_root, candidate_id)
            result = load_or_recover_candidate_result(
                run_root=run_root,
                candidate=candidates[candidate_id],
                expanded=expanded,
                gate_row=gate_row,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256=contract_sha,
            )
            if result is not None:
                expected_identity = {
                    "candidate_id": candidate_id,
                    "root_id": root_id,
                    "solver_contract_sha256": contract_sha,
                    "candidate_pool_sha256": gate_row["candidate_pool_sha256"],
                    "dialogue_sha256": gate_row["dialogue_sha256"],
                }
                drift = {
                    key: (result.get(key), value)
                    for key, value in expected_identity.items()
                    if result.get(key) != value
                }
                if drift:
                    raise RuntimeError(f"candidate_result_contract_drift:{drift}")
            else:
                result = evaluate_candidate(
                    candidate=candidates[candidate_id],
                    expanded=expanded,
                    gate_row=gate_row,
                    tokenizer=tokenizer,
                    backend=backend,
                    ledger=ledger,
                    solver_contract_sha256=contract_sha,
                )
                write_once_json(result_path, result)
            completed.append(result)
    expected_candidates = len(plan["smoke_candidates"]) if scope == "smoke" else len(passed)
    expected_requests = (
        int(plan["smoke_request_count"])
        if scope == "smoke" else int(plan["planned_request_count"])
    )
    observed_requests = sum(int(row["request_count"]) for row in completed)
    paired_seed_errors = 0
    for result in completed:
        seeds: dict[int, set[int]] = defaultdict(set)
        for request in result["requests"]:
            seeds[int(request["slot_id"])].add(int(request["seed"]))
        paired_seed_errors += sum(len(values) != 1 for values in seeds.values())
    technical = sum(int(row["technical_failure_count"]) for row in completed)
    ledger_summary = ledger.summary()
    forbidden = str(contract["forbidden_solver_model"])
    forbidden_references = sum(
        str(((state.get("request_contract") or {}).get("backend_identity") or {}).get("model"))
        == forbidden
        for state in ledger.states()
    )
    report = {
        "execution_report_version": "h4_round0_b0_execution_report_v1",
        "scope": scope,
        "status": "PASS" if (
            len(completed) == expected_candidates
            and observed_requests == expected_requests
            and paired_seed_errors == 0
            and ledger_summary["request_count"] == expected_requests
            and (
                ledger_summary["completed_request_count"]
                + ledger_summary["failed_request_count"]
                == expected_requests
            )
            and ledger_summary["ambiguous_request_count"] == 0
            and forbidden_references == 0
            and (technical == 0 if scope == "smoke" else True)
        ) else "FAIL",
        "candidate_count": len(completed),
        "expected_candidate_count": expected_candidates,
        "request_count": observed_requests,
        "expected_request_count": expected_requests,
        "technical_failure_count": technical,
        "paired_seed_error_count": paired_seed_errors,
        "forbidden_solver_model_reference_count": forbidden_references,
        "ledger": ledger_summary,
        "solver_contract_sha256": contract_sha,
    }
    write_once_json(run_root / f"{scope}_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"{scope}_validation_failed:{canonical_json(report)}")
    return report


__all__ = [
    "backend_identity",
    "candidate_result_path",
    "create_backend",
    "ensure_vllm_environment",
    "evaluate_candidate",
    "execute_scope",
    "generation_config",
    "load_or_recover_candidate_result",
    "validate_solver_runtime",
]
