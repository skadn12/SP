"""Hybrid-probe Solver adapter with structured temperature separation.

The production/earlier probe adapters remain untouched. This adapter reuses
their prompt and protocol code, but makes structured temperatures explicit
and aggregates all five structured rollouts per position.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from ..h4_qa_mining_probe_v1 import solver_adapter as _base
from ..h4_qa_mining_probe_v1.solver_adapter import position_variants


MULTI_SELECT_SOLVER_TEMPERATURE = 0.3
ORDERING_SOLVER_TEMPERATURE = 0.0


class _HybridLocalSolverRunner(_base.LocalSolverRunner):
    """Raise only this probe's local request ceiling for 36 candidates."""

    def __init__(self, *args: Any, physical_request_budget: int = 600, **kwargs: Any):
        super().__init__(*args, physical_request_budget=max(900, physical_request_budget), **kwargs)


def _structured_generate(backend: Any, request: Any, *, qtype: str) -> Any:
    """Use 0.3 for multi-select and retain deterministic ordering at 0.0."""
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from ..structured_qa_box_protocol import STRUCTURED_SOLVER_MAX_TOKENS, structured_box_choices
    from ..backends import BackendResponse

    temperature = (MULTI_SELECT_SOLVER_TEMPERATURE
                   if qtype == "multi_select" else ORDERING_SOLVER_TEMPERATURE)
    allowed = request.options.keys() if qtype in {"multi_select", "ordering"} else None
    choices = structured_box_choices(qtype, allowed_item_ids=allowed)
    params = SamplingParams(
        temperature=temperature, top_p=backend.top_p,
        max_tokens=STRUCTURED_SOLVER_MAX_TOKENS, seed=request.sampling_seed,
        stop=["</box>"], include_stop_str_in_output=True,
        guided_decoding=GuidedDecodingParams(choice=choices),
    )
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    return BackendResponse(completion.text, backend.model_identifier, {
        "sampling_seed": request.sampling_seed,
        "prompt_token_count": len(outputs[0].prompt_token_ids),
        "completion_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason, "temperature": temperature,
        "max_tokens": STRUCTURED_SOLVER_MAX_TOKENS, "guided_decoding": True,
    })


def _aggregate_structured(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep all five structured slots in one variant/mode summary row."""
    aggregated: list[dict[str, Any]] = []
    for result in results:
        if result.get("question_type") == "single_choice":
            aggregated.append(result)
            continue
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in result.get("variants") or []:
            groups[(str(row.get("variant")), str(row.get("mode")))].append(row)
        rows: list[dict[str, Any]] = []
        for (variant, mode), slot_rows in groups.items():
            first = dict(slot_rows[0])
            correct = sum(int(bool(row.get("correct"))) for row in slot_rows)
            parse_failures = sum(row.get("failure_type") == "parse_failure"
                                 for row in slot_rows)
            first.update({
                "correct": correct, "total": len(slot_rows),
                "parse_failure": parse_failures,
                "rollouts": [
                    {"local_reward": (1 if row.get("correct") else 0)
                     if row.get("failure_type") != "parse_failure" else None,
                     "raw_output": row.get("raw_output"),
                     "parsed_answer": row.get("parsed_answer"),
                     "failure_type": row.get("failure_type"), "slot": index}
                    for index, row in enumerate(slot_rows)
                ],
                "slot_results": slot_rows,
            })
            rows.append(first)
        aggregated.append({**result, "variants": rows})
    return aggregated


def evaluate_candidates(*, candidates: list[Mapping[str, Any]], expanded: Mapping[str, Any],
                        run_root: Any, ledger: Any, tokenizer: Any, backend: Any) -> list[dict[str, Any]]:
    # The base evaluator owns request construction and parsing. Temporarily
    # substitute only its local generation function within this isolated call.
    original = _base._structured_generate
    original_runner = _base.LocalSolverRunner
    _base._structured_generate = _structured_generate
    _base.LocalSolverRunner = _HybridLocalSolverRunner
    try:
        raw = _base.evaluate_candidates(
            candidates=candidates, expanded=expanded, run_root=run_root,
            ledger=ledger, tokenizer=tokenizer, backend=backend)
    finally:
        _base._structured_generate = original
        _base.LocalSolverRunner = original_runner
    return _aggregate_structured(raw)


__all__ = ["MULTI_SELECT_SOLVER_TEMPERATURE", "ORDERING_SOLVER_TEMPERATURE",
           "evaluate_candidates", "position_variants"]
