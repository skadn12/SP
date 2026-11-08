from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.backends import SolverRequest
from scripts.current_method.restart_n5_spell.core import sha256_bytes
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import LocalSolverRunner
from scripts.current_method.restart_n5_spell.structured_qa_box_protocol import (
    STRUCTURED_SOLVER_MAX_TOKENS,
    STRUCTURED_SOLVER_TEMPERATURE,
    classify_structured_result,
    parse_structured_box,
    structured_box_choices,
)
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend


STRUCTURED_SYSTEM_PROMPT = (
    "You are a strict structured dialogue QA solver. Use only the supplied "
    "dialogue when it is present. Return the final answer in exactly one "
    "<box>...</box> pair. Do not provide explanations, reasoning, or any "
    "text outside the box."
)


def _rotate(mapping: Mapping[str, str], shift: int) -> dict[str, str]:
    keys = list(mapping)
    values = [str(mapping[key]) for key in keys]
    return {key: values[(index - shift) % len(keys)] for index, key in enumerate(keys)}


def position_variants(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    qtype = str(candidate["question_type"])
    if qtype == "single_choice":
        original = {str(k): str(v) for k, v in candidate["options"].items()}
        correct_text = original[str(candidate["correct_option"])]
        result = []
        for shift in range(4):
            options = _rotate(original, shift)
            gold = next(key for key, value in options.items() if value == correct_text)
            result.append({"variant": f"shift_{shift}", "shift": shift,
                           "options": options, "gold": [gold]})
        return result
    if qtype == "multi_select":
        original = {str(k): str(v) for k, v in candidate["options"].items()}
        gold_texts = {original[str(key)] for key in candidate["gold"]}
        result = []
        for shift in range(5):
            options = _rotate(original, shift)
            gold = [key for key, value in options.items() if value in gold_texts]
            result.append({"variant": f"shift_{shift}", "shift": shift,
                           "options": options, "gold": gold})
        return result
    original = {str(k): str(v) for k, v in candidate["items"].items()}
    keys = list(original)
    result = []
    for shift in range(4):
        display = keys[shift:] + keys[:shift]
        result.append({"variant": f"shift_{shift}", "shift": shift,
                       "items": {key: original[key] for key in display},
                       "gold": list(candidate["gold"])})
    return result


def _structured_user_content(
    candidate: Mapping[str, Any], variant: Mapping[str, Any],
    dialogue: list[Mapping[str, Any]] | None,
) -> str:
    chunks: list[str] = []
    if dialogue is not None:
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
        chunks.append("Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.")
    return "\n\n".join(chunks)


def _structured_prompt(tokenizer: Any, candidate: Mapping[str, Any], variant: Mapping[str, Any],
                       dialogue: list[Mapping[str, Any]] | None) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
         {"role": "user", "content": _structured_user_content(candidate, variant, dialogue)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _structured_generate(
    backend: VLLMSolverBackend, request: SolverRequest, *, qtype: str,
) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    allowed = request.options.keys() if qtype in {"multi_select", "ordering"} else None
    choices = structured_box_choices(qtype, allowed_item_ids=allowed)
    params = SamplingParams(
        temperature=STRUCTURED_SOLVER_TEMPERATURE,
        top_p=backend.top_p,
        max_tokens=STRUCTURED_SOLVER_MAX_TOKENS,
        seed=request.sampling_seed,
        stop=["</box>"], include_stop_str_in_output=True,
        guided_decoding=GuidedDecodingParams(choice=choices),
    )
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    from scripts.current_method.restart_n5_spell.backends import BackendResponse
    return BackendResponse(completion.text, backend.model_identifier, {
        "sampling_seed": request.sampling_seed,
        "prompt_token_count": len(outputs[0].prompt_token_ids),
        "completion_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "temperature": STRUCTURED_SOLVER_TEMPERATURE,
        "max_tokens": STRUCTURED_SOLVER_MAX_TOKENS,
        "guided_decoding": True,
    })


def evaluate_candidates(
    *, candidates: list[Mapping[str, Any]], expanded: Mapping[str, Any],
    run_root: Path, ledger: RequestLedger, tokenizer: Any,
    backend: VLLMSolverBackend,
) -> list[dict[str, Any]]:
    dialogue = [
        {"turn_id": row["turn_ref"], "session_id": row.get("session_ref", ""),
         "speaker": row["speaker"], "text": row["text"]}
        for row in expanded.get("turns") or []
    ]
    single_runner = LocalSolverRunner(
        run_root=run_root, ledger=ledger, tokenizer=tokenizer,
        backend=backend, dialogue=dialogue, physical_request_budget=600,
    )
    all_results: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_results: list[dict[str, Any]] = []
        for variant in position_variants(candidate):
            for mode, context in (("no_context", []), ("full_context", dialogue)):
                if candidate["question_type"] == "single_choice":
                    single_runner.dialogue = context
                    qa = {
                        "qa_id": f"{candidate['candidate_id']}-{variant['variant']}",
                        "question": candidate["question"],
                        "canonical_answer_text": variant["options"][variant["gold"][0]],
                    }
                    rollouts = [single_runner.one(
                        qa=qa, options=variant["options"], mode=mode,
                        order_id=variant["variant"], slot=slot,
                    ) for slot in range(5)]
                    candidate_results.append({
                        "candidate_id": candidate["candidate_id"],
                        "variant": variant["variant"], "mode": mode,
                        "gold": variant["gold"],
                        "rollouts": rollouts,
                        "correct": sum(x.get("local_reward") == 1 for x in rollouts),
                        "total": 5,
                        "parse_failure": sum(x.get("local_reward") is None for x in rollouts),
                    })
                    continue
                prompt = _structured_prompt(tokenizer, candidate, variant, context)
                for slot in range(5):
                    request_id = f"structured-{candidate['candidate_id']}-{variant['variant']}-{mode}-s{slot:02d}"
                    seed = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16)
                    prompt_sha = sha256_bytes(prompt.encode())
                    request = SolverRequest(
                        request_id=request_id, qa_id=str(candidate["candidate_id"]),
                        evaluation_mode=mode, order_id=variant["variant"], target_slot=slot,
                        retry_index=0, sampling_seed=seed,
                        options=variant.get("options") or variant.get("items") or {},
                        canonical_answer_text=",".join(variant["gold"]),
                        prompt_sha256=prompt_sha, prompt=prompt,
                    )
                    response = ledger.invoke(
                        request_id=request_id, intent=f"structured_solver_{mode}", attempt=1,
                        prompt_sha256=prompt_sha,
                        call=lambda request=request: _structured_generate(
                            backend, request, qtype=candidate["question_type"]),
                        budget_class="structured_solver", budget_limit=1000,
                        total_budget_limit=1000,
                        request_metadata={"candidate_id": candidate["candidate_id"], "mode": mode, "variant": variant["variant"], "slot": slot},
                        backend_identity={"backend": "VLLMSolverBackend", "model": backend.model_identifier},
                    )
                    parsed = parse_structured_box(
                        response.raw_text, question_type=candidate["question_type"],
                        allowed_item_ids=(variant.get("options") or variant.get("items") or {}).keys(),
                    )
                    result = classify_structured_result(
                        parsed, gold=list(variant["gold"]), question_type=candidate["question_type"])
                    metrics = result.get("metrics") or {}
                    ledger.record_decision(request_id, result["failure_type"] or "exact_match")
                    candidate_results.append({
                        "candidate_id": candidate["candidate_id"],
                        "variant": variant["variant"], "mode": mode,
                        "gold": variant["gold"], "raw_output": response.raw_text,
                        "parsed_answer": result.get("parsed_answer"),
                        "parse_status": result.get("parse_status"),
                        "failure_type": result.get("failure_type"),
                        "metrics": metrics,
                        "correct": bool(metrics.get("exact_match" if candidate["question_type"] == "multi_select" else "exact_order_match")),
                        "total": 1,
                    })
        all_results.append({
            "candidate_id": candidate["candidate_id"],
            "question_type": candidate["question_type"],
            "variants": candidate_results,
        })
    return all_results
