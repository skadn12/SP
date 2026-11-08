"""Fresh-candidate K=3 Solver evaluation for the isolated H4 probe.

This module deliberately has its own rollout loop.  The older H4 adapters are
K=5 and the production single-choice runner retries a slot, neither of which
matches this experiment's bookkeeping contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends import BackendResponse, SolverRequest
from ..core import (
    atomic_json, local_reward, parse_boxed_mcq_answer, sha256_bytes,
)
from ..ledger import RequestLedger
from ..prompts import SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT, SOLVER_TRAILING_FORMAT_INSTRUCTION
from ..structured_qa_box_protocol import (
    STRUCTURED_GUIDED_DECODING_VERSION,
    STRUCTURED_SOLVER_MAX_TOKENS,
    classify_structured_result,
    parse_structured_box,
    structured_box_choices,
)
from ..vllm_backends import VLLMSolverBackend

from .diagnostics import aggregate_position, classify_candidate, context_diagnostic
from .permutation import audit_gold_rebinding, position_variants
from .seed_schedule import SEED_SCHEDULE_VERSION, stable_seed


def _ordering_read_context(*, root_id: str, run_root: Path,
                           ordering_policy: Mapping[str, Any] | None,
                           ordering_context: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    from ..h4_r02_ordering_temporal_contract_v1.contract import resolve_runtime_context
    context = dict(ordering_context or {})
    if "root_id" in context and context["root_id"] != root_id:
        raise ValueError("SOLVER_ORDERING_ROOT_ID_MISMATCH")
    if "root_dir" in context and Path(context["root_dir"]).resolve() != Path(run_root).resolve():
        raise ValueError("SOLVER_ORDERING_ROOT_DIR_MISMATCH")
    context.update({"root_id": root_id, "root_dir": Path(run_root)})
    return resolve_runtime_context(Path(run_root), ordering_policy, context)


ROLLOUTS = 3
SINGLE_TEMPERATURE = 0.7
STRUCTURED_TEMPERATURE = 0.8
SOLVER_MAX_TOKENS = 32
SOLVER_MAX_MODEL_LEN = 16_384

STRUCTURED_SYSTEM_PROMPT = (
    "You are a strict structured dialogue QA solver. Use only the supplied "
    "dialogue when it is present. Return the final answer in exactly one "
    "<box>...</box> pair. Do not provide explanations, reasoning, or any "
    "text outside the box."
)


def _dialogue_rows(expanded: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "turn_id": str(row.get("turn_ref") or ""),
            "session_id": str(row.get("session_ref") or ""),
            "speaker": str(row.get("speaker") or ""),
            "text": str(row.get("text") or ""),
        }
        for row in expanded.get("turns") or []
    ]


def _single_prompt(tokenizer: Any, candidate: Mapping[str, Any], variant: Mapping[str, Any],
                   dialogue: Sequence[Mapping[str, Any]]) -> str:
    context = "\n".join(
        f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in dialogue
    )
    chunks = []
    if context:
        chunks.append("Context:\n" + context)
    chunks.append(str(candidate["question"]))
    chunks.append("\n".join(f"{key}. {value}" for key, value in variant["options"].items()))
    chunks.append(SOLVER_TRAILING_FORMAT_INSTRUCTION)
    user = "\n\n".join(chunks)
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _structured_prompt(tokenizer: Any, candidate: Mapping[str, Any], variant: Mapping[str, Any],
                       dialogue: Sequence[Mapping[str, Any]]) -> str:
    chunks = []
    if dialogue:
        chunks.append("Dialogue:\n" + "\n".join(
            f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in dialogue
        ))
    chunks.append(f"Question ({candidate['question_type']}):\n{candidate['question']}")
    if candidate["question_type"] == "multi_select":
        chunks.append("Options:\n" + "\n".join(
            f"{key}. {value}" for key, value in variant["options"].items()
        ))
        chunks.append("Output only selected letters separated by commas, e.g. <box>A,C</box>.")
    else:
        chunks.append("Items:\n" + "\n".join(
            f"{key}. {value}" for key, value in variant["items"].items()
        ))
        count = len(variant.get("items") or {})
        sample = ",".join(["I2", "I4", "I1", "I3"] + [f"I{x}" for x in range(5, count + 1)])
        chunks.append(f"Output only ordered item IDs separated by commas, e.g. <box>{sample}</box>.")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(chunks)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _structured_generate(backend: VLLMSolverBackend, request: SolverRequest,
                         question_type: str, sampling_temperature: float) -> BackendResponse:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    allowed = request.options.keys() if question_type in {"multi_select", "ordering"} else None
    choices = structured_box_choices(question_type, allowed_item_ids=allowed)
    if not choices:
        raise ValueError(f"structured_guided_choices_empty:{question_type}")
    params = SamplingParams(
        temperature=float(sampling_temperature),
        top_p=backend.top_p,
        max_tokens=STRUCTURED_SOLVER_MAX_TOKENS,
        seed=request.sampling_seed,
        stop=["</box>"],
        include_stop_str_in_output=True,
        guided_decoding=GuidedDecodingParams(choice=choices),
    )
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    metadata = {
        "sampling_seed": request.sampling_seed,
        "prompt_token_count": len(outputs[0].prompt_token_ids),
        "completion_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "sampling_temperature": float(sampling_temperature),
        "backend_temperature": backend.temperature,
        "top_p": backend.top_p,
        "max_tokens": STRUCTURED_SOLVER_MAX_TOKENS,
        "guided_decoding": True,
        "guided_decoding_version": STRUCTURED_GUIDED_DECODING_VERSION,
    }
    return BackendResponse(completion.text, backend.model_identifier, metadata)


def _failure_type(error: BaseException) -> str:
    text = f"{type(error).__name__}:{error}".casefold()
    if "timeout" in text:
        return "timeout"
    if "out of memory" in text or "oom" in text:
        return "OOM"
    if "guided" in text or "choice" in text:
        return "guided_decoding_error"
    return "runtime_error"


def _rollout_record(*, request_id: str, slot: int, seed: int, qtype: str,
                    temperature: float) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "slot_id": slot,
        "seed": seed,
        "question_type": qtype,
        "requested_temperature": temperature,
        "raw_output": None,
        "parsed_answer": [],
        "parse_status": "not_run",
        "failure_type": "runtime_error",
        "correct": False,
        "scored": False,
        "metrics": {},
    }


def _invoke_rollout(*, candidate: Mapping[str, Any], variant: Mapping[str, Any], mode: str,
                    slot: int, dialogue: Sequence[Mapping[str, Any]], tokenizer: Any,
                    backend: VLLMSolverBackend, ledger: RequestLedger, run_root: Path,
                    root_id: str) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    qtype = str(candidate["question_type"])
    variant_id = str(variant["variant"])
    temperature = SINGLE_TEMPERATURE if qtype == "single_choice" else STRUCTURED_TEMPERATURE
    seed = stable_seed(root_id, candidate_id, slot)
    request_id = f"solver-{candidate_id}-{variant_id}-{mode}-s{slot:02d}"
    row = _rollout_record(
        request_id=request_id, slot=slot, seed=seed, qtype=qtype,
        temperature=temperature,
    )
    context = list(dialogue) if mode == "full_context" else []
    prompt = (
        _single_prompt(tokenizer, candidate, variant, context)
        if qtype == "single_choice"
        else _structured_prompt(tokenizer, candidate, variant, context)
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    row["prompt_token_count"] = prompt_tokens
    row["prompt_sha256"] = sha256_bytes(prompt.encode())
    if prompt_tokens + SOLVER_MAX_TOKENS > SOLVER_MAX_MODEL_LEN:
        row["failure_type"] = "runtime_error"
        row["error"] = "context_overflow"
        return row
    options = dict(variant.get("options") or variant.get("items") or {})
    gold = [str(x) for x in variant.get("gold") or []]
    canonical = (
        options[gold[0]] if qtype == "single_choice" else ",".join(gold)
    )
    request = SolverRequest(
        request_id=request_id,
        qa_id=candidate_id,
        evaluation_mode=mode,
        order_id=variant_id,
        target_slot=slot,
        retry_index=0,
        sampling_seed=seed,
        options=options,
        canonical_answer_text=canonical,
        prompt_sha256=row["prompt_sha256"],
        prompt=prompt,
    )
    atomic_json(run_root / "raw" / "prompts" / f"{request_id}.json", {
        "request_id": request_id,
        "candidate_id": candidate_id,
        "variant": variant_id,
        "mode": mode,
        "slot_id": slot,
        "seed": seed,
        "prompt_sha256": row["prompt_sha256"],
        "prompt": prompt,
        "enable_thinking": False,
        "sampling_temperature": temperature,
    })
    try:
        response = ledger.invoke(
            request_id=request_id,
            intent=f"h4_k3_solver_{qtype}_{mode}",
            attempt=1,
            prompt_sha256=row["prompt_sha256"],
            call=(
                lambda: backend.generate(request)
                if qtype == "single_choice"
                else _structured_generate(backend, request, qtype, temperature)
            ),
            budget_class="k3_solver",
            budget_limit=100_000,
            total_budget_limit=100_000,
            request_metadata={
                "probe_version": "h4_qa_multiroot_k3_t08_probe_v1",
                "root_id": root_id,
                "candidate_id": candidate_id,
                "question_type": qtype,
                "variant": variant_id,
                "mode": mode,
                "slot_id": slot,
                "seed": seed,
                "seed_schedule_version": SEED_SCHEDULE_VERSION,
                "rollouts": ROLLOUTS,
                "sampling_temperature": temperature,
                "guided_decoding_version": (
                    STRUCTURED_GUIDED_DECODING_VERSION if qtype != "single_choice" else None
                ),
            },
            backend_identity={
                "backend": "VLLMSolverBackend",
                "model": backend.model_identifier,
                "temperature": temperature,
                "top_p": backend.top_p,
                "max_tokens": SOLVER_MAX_TOKENS,
            },
        )
    except Exception as error:
        row["failure_type"] = _failure_type(error)
        row["error"] = f"{type(error).__name__}:{error}"
        return row

    row["raw_output"] = response.raw_text
    row["backend_metadata"] = dict(response.metadata or {})
    if qtype == "single_choice":
        parsed = parse_boxed_mcq_answer(response.raw_text, options)
        reward = local_reward(parsed, canonical)
        row.update({
            "parsed_answer": [parsed["parsed_letter"]] if reward is not None else [],
            "parse_status": parsed.get("parse_status"),
            "parser_version": parsed.get("parser_version"),
            "scored": reward is not None,
            "correct": reward == 1,
            "failure_type": None if reward is not None and reward == 1 else (
                "reasoning_error" if reward is not None else "parse_failure"
            ),
            "metrics": {},
        })
    else:
        parsed = parse_structured_box(
            response.raw_text,
            question_type=qtype,
            allowed_item_ids=(variant.get("options") or variant.get("items") or {}).keys(),
        )
        result = classify_structured_result(parsed, gold=gold, question_type=qtype)
        metrics = dict(result.get("metrics") or {})
        if qtype == "multi_select":
            predicted = list(result.get("parsed_answer") or [])
            gold_set = set(gold)
            predicted_set = set(predicted)
            metrics.update({
                "gold_set_size": len(gold_set),
                "predicted_set_size": len(predicted_set),
                "missing_gold_count": len(gold_set - predicted_set),
                "extra_option_count": len(predicted_set - gold_set),
            })
        row.update({
            "parsed_answer": list(result.get("parsed_answer") or []),
            "parse_status": result.get("parse_status"),
            "parser_version": result.get("parser_version"),
            "scored": result.get("parse_status") == "success",
            "correct": bool(
                metrics.get("exact_match") if qtype == "multi_select"
                else metrics.get("exact_order_match")
            ),
            "failure_type": result.get("failure_type"),
            "metrics": metrics,
        })
    ledger.record_decision(request_id, row["failure_type"] or "correct")
    return row


def evaluate_candidates(*, candidates: list[Mapping[str, Any]], expanded: Mapping[str, Any],
                        root_id: str, run_root: Path, ledger: RequestLedger, tokenizer: Any,
                        backend: VLLMSolverBackend, candidate_limit: int | None = None,
                        ordering_policy: Mapping[str, Any] | None = None,
                        ordering_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    dialogue = _dialogue_rows(expanded)
    selected = candidates if candidate_limit is None else candidates[: int(candidate_limit)]
    if any(candidate.get("question_type") == "ordering" for candidate in selected):
        from ..h4_r02_ordering_temporal_contract_v1.contract import verify_candidate
        policy, context = _ordering_read_context(
            root_id=root_id, run_root=run_root, ordering_policy=ordering_policy,
            ordering_context=ordering_context,
        )
        # Check the entire selected batch before any backend call is possible.
        for candidate in selected:
            verify_candidate(candidate, expanded=expanded, policy=policy, context=context)
    all_results: list[dict[str, Any]] = []
    for candidate in selected:
        variants = position_variants(candidate)
        rebinding = audit_gold_rebinding(candidate, variants)
        if not rebinding.get("pass"):
            raise ValueError("SOLVER_GOLD_REBINDING_FAILED")
        result_rows: list[dict[str, Any]] = []
        for variant in variants:
            for mode in ("no_context", "full_context"):
                rollouts = [
                    _invoke_rollout(
                        candidate=candidate,
                        variant=variant,
                        mode=mode,
                        slot=slot,
                        dialogue=dialogue,
                        tokenizer=tokenizer,
                        backend=backend,
                        ledger=ledger,
                        run_root=run_root,
                        root_id=root_id,
                    )
                    for slot in range(ROLLOUTS)
                ]
                summary = aggregate_position(rollouts)
                result_rows.append({
                    "candidate_id": str(candidate["candidate_id"]),
                    "variant": variant["variant"],
                    "shift": variant["shift"],
                    "mode": mode,
                    "gold": list(variant["gold"]),
                    "summary": summary,
                    "rollouts": rollouts,
                })
        no_positions = [row for row in result_rows if row["mode"] == "no_context"]
        full_positions = [row for row in result_rows if row["mode"] == "full_context"]
        no_diag = classify_candidate([row["summary"] for row in no_positions])
        full_diag = classify_candidate([row["summary"] for row in full_positions])
        context = context_diagnostic(
            [dict(row["summary"], variant=row["variant"]) for row in no_positions],
            [dict(row["summary"], variant=row["variant"]) for row in full_positions],
        )
        all_results.append({
            "candidate_id": str(candidate["candidate_id"]),
            "question_type": str(candidate["question_type"]),
            "root_id": root_id,
            "variants": result_rows,
            "position_diagnostics": {
                "no_context": no_diag,
                "full_context": full_diag,
            },
            "context_diagnostics": context,
            "gold_rebinding_audit": rebinding,
        })
    return all_results


__all__ = [
    "ROLLOUTS", "SINGLE_TEMPERATURE", "STRUCTURED_TEMPERATURE",
    "evaluate_candidates",
]
