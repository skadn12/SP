"""Local Solver execution for the frozen position-permutation manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..backends import BackendResponse, SolverRequest
from ..core import atomic_json, atomic_text, parse_boxed_mcq_answer, sha256_bytes
from ..ledger import RequestLedger
from ..prompts import SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT, SOLVER_TRAILING_FORMAT_INSTRUCTION
from ..structured_qa_box_protocol import classify_structured_result, parse_structured_box, structured_box_choices
from ..vllm_backends import VLLMSolverBackend
from .core import MODEL_LEN, OUTPUT_TOKENS_LABEL
from .seed_schedule import stable_seed


CONTENT_SEPARATOR = " || "
CONTENT_PARSER_VERSION = "h4_position_ab_content_v1"
CONTENT_RESPONDER_PROMPT_VERSION = "h4_position_ab_content_responder_v2"
SOLVER_CONTENT_RESPONDER_PROMPT = f"""Answer the multiple-choice question using only the supplied context.
If no context is supplied, do not assume unstated facts.

Return only the complete option or item text(s) inside one <box>...</box>.
For multiple selections or ordering, separate complete texts with the literal delimiter
{CONTENT_SEPARATOR!r}. Do not output option labels, item IDs, reasoning, or any text outside
the box tags."""
# Kept as an explicit alias for the B2 repair provenance recorded in existing
# reports; B1 and B2 now intentionally share this same content contract.
SOLVER_B2_CONTENT_RESPONDER_PROMPT = SOLVER_CONTENT_RESPONDER_PROMPT
OUTPUT_MODE_LABEL = "label"
OUTPUT_MODE_CONTENT_VISIBLE = "content_visible"
OUTPUT_MODE_CONTENT_HIDDEN = "content_hidden"
OUTPUT_MODES = (OUTPUT_MODE_LABEL, OUTPUT_MODE_CONTENT_VISIBLE, OUTPUT_MODE_CONTENT_HIDDEN)
SINGLE_TEMPERATURE = 0.7
STRUCTURED_TEMPERATURE = 0.8


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(text: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).strip().split())


def _box_inner(raw: Any) -> tuple[str | None, str | None]:
    text = str(raw or "")
    if text.count("<box>") != 1 or text.count("</box>") != 1:
        return None, "exactly_one_box_required"
    match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", text, flags=re.DOTALL)
    return (match.group(1).strip(), None) if match else (None, "text_outside_box")


def _dialogue(expanded: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{
        "turn_id": str(row.get("turn_ref") or ""),
        "speaker": str(row.get("speaker") or ""),
        "text": str(row.get("text") or ""),
    } for row in expanded.get("turns") or []]


def _canonical(candidate: Mapping[str, Any]) -> tuple[list[str], dict[str, str], list[str]]:
    qtype = str(candidate.get("question_type") or "")
    if qtype == "single_choice":
        keys = ["A", "B", "C", "D"]
        values = candidate.get("options") or {}
        gold = [str(candidate.get("correct_option") or "")]
    elif qtype == "multi_select":
        keys = ["A", "B", "C", "D", "E"]
        values = candidate.get("options") or {}
        gold = [str(value) for value in candidate.get("gold") or []]
    elif qtype == "ordering":
        keys = ["I1", "I2", "I3", "I4"]
        values = candidate.get("items") or {}
        gold = [str(value) for value in candidate.get("gold") or []]
    else:
        raise ValueError(f"unsupported_question_type:{qtype}")
    texts = {key: str(values[key]) for key in keys}
    return keys, texts, gold


def _display(candidate: Mapping[str, Any], permutation: Mapping[str, Any], output_mode: str) -> tuple[dict[str, str], list[str], dict[str, str]]:
    keys, texts, gold = _canonical(candidate)
    display_to_canonical = {str(k): str(v) for k, v in (permutation.get("display_to_canonical_mapping") or {}).items()}
    if not display_to_canonical:
        display_to_canonical = {key: canonical for key, canonical in zip(keys, permutation["display_order"])}
    display = {label: texts[canonical] for label, canonical in display_to_canonical.items()}
    inverse = {canonical: label for label, canonical in display_to_canonical.items()}
    # Every displayed slot has a canonical item mapping, including ordering.
    # Therefore the expected label sequence must be rebound for all types.
    displayed_gold = [inverse[item] for item in gold]
    return display, displayed_gold, display_to_canonical


def _label_prompt(tokenizer: Any, candidate: Mapping[str, Any], display: Mapping[str, str], dialogue: Iterable[Mapping[str, str]]) -> str:
    qtype = str(candidate["question_type"])
    context = list(dialogue)
    chunks: list[str] = []
    if context:
        chunks.append("Context:\n" + "\n".join(f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in context))
    chunks.append(str(candidate["question"]))
    if qtype == "single_choice":
        chunks.append("\n".join(f"{key}. {value}" for key, value in display.items()))
        chunks.append(SOLVER_TRAILING_FORMAT_INSTRUCTION)
    elif qtype == "multi_select":
        chunks.append("Select all that apply. Return only selected labels in one box, e.g. <box>A,C</box>.")
        chunks.append("\n".join(f"{key}. {value}" for key, value in display.items()))
    else:
        chunks.append("Return only the chronological item IDs in one box, e.g. <box>I2,I4,I1,I3</box>.")
        chunks.append("\n".join(f"{key}. {value}" for key, value in display.items()))
    return tokenizer.apply_chat_template([
        {"role": "system", "content": SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT},
        {"role": "user", "content": "\n\n".join(chunks)},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _content_prompt(tokenizer: Any, candidate: Mapping[str, Any], display: Mapping[str, str], dialogue: Iterable[Mapping[str, str]], *, visible_labels: bool) -> str:
    context = list(dialogue)
    chunks: list[str] = []
    if context:
        chunks.append("Context:\n" + "\n".join(f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in context))
    chunks.append(str(candidate["question"]))
    if visible_labels:
        chunks.append("Choose by returning the complete option/item text, separated by ' || ' inside one <box>. Do not return labels.")
        chunks.append("\n".join(f"{key}. {value}" for key, value in display.items()))
    else:
        chunks.append("Choose by returning the complete option/item text, separated by ' || ' inside one <box>. No labels or item IDs are shown. Do not return labels.")
        chunks.append("\n".join(display.values()))
    return tokenizer.apply_chat_template([
        {"role": "system", "content": SOLVER_CONTENT_RESPONDER_PROMPT},
        {"role": "user", "content": "\n\n".join(chunks)},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _content_reserve(tokenizer: Any, candidate: Mapping[str, Any], display: Mapping[str, str]) -> int:
    qtype = str(candidate["question_type"])
    texts = list(display.values())
    if qtype == "single_choice":
        body = max(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts) + 24
    elif qtype == "multi_select":
        body = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts) + 32
    else:
        body = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts) + 32
    # Do not cap this by an arbitrary token count: B1/B2 must have enough
    # reserve to emit the longest legal complete-text answer.  The model-length
    # check in _invoke converts an unfit request into a technical failure.
    return max(32, body)


def _content_choices(candidate: Mapping[str, Any], display: Mapping[str, str], display_to_canonical: Mapping[str, str]) -> list[str]:
    """Enumerate legal complete-text answers for B1/B2 guided decoding."""
    qtype = str(candidate["question_type"])
    labels = list(display)
    if qtype == "single_choice":
        sequences = [(label,) for label in labels]
    elif qtype == "multi_select":
        sequences = [combo for size in range(1, len(labels) + 1) for combo in combinations(labels, size)]
    elif qtype == "ordering":
        sequences = []
        # The answer order is semantic canonical order, while display labels
        # are mapped back through display_to_canonical_mapping by the parser.
        inverse = {str(canonical): str(label) for label, canonical in display_to_canonical.items()}
        canonical_ids = sorted(inverse)
        # For the frozen four-item contract this is I1..I4.  Keeping the
        # construction generic avoids coupling the content mode to labels.
        for order in permutations(canonical_ids):
            sequences.append(tuple(inverse[item] for item in order))
    else:
        raise ValueError(f"unsupported_content_question_type:{qtype}")
    return [f"<box>{CONTENT_SEPARATOR.join(display[label] for label in sequence)}</box>" for sequence in sequences]


def _label_parse(raw: str, candidate: Mapping[str, Any], display: Mapping[str, str], gold: list[str]) -> tuple[list[str], dict[str, Any]]:
    qtype = str(candidate["question_type"])
    if qtype == "single_choice":
        parsed = parse_boxed_mcq_answer(raw, dict(display))
        answer = [str(parsed["parsed_letter"])] if parsed.get("parse_status", "").startswith("valid_") else []
        return answer, {"parse_status": parsed.get("parse_status"), "failure_type": None if answer else "parse_failure", "correct": bool(answer and answer == gold)}
    parsed = parse_structured_box(raw, question_type=qtype, allowed_item_ids=display.keys())
    result = classify_structured_result(parsed, gold=gold, question_type=qtype)
    metrics = dict(result.get("metrics") or {})
    correct = bool(metrics.get("exact_match") if qtype == "multi_select" else metrics.get("exact_order_match"))
    return list(result.get("parsed_answer") or []), {"parse_status": result.get("parse_status"), "failure_type": result.get("failure_type"), "correct": correct, "metrics": metrics}


def _content_parse(raw: str, candidate: Mapping[str, Any], display: Mapping[str, str], display_to_canonical: Mapping[str, str], gold: list[str]) -> tuple[list[str], dict[str, Any]]:
    inner, error = _box_inner(raw)
    base = {"parse_status": "parse_failure", "failure_type": "parse_failure", "correct": False}
    if error or inner is None:
        base["parse_reason"] = error
        return [], base
    pieces = [_norm(piece) for piece in inner.split(CONTENT_SEPARATOR)]
    if not pieces or any(not piece for piece in pieces):
        base["parse_reason"] = "empty_content_answer"
        return [], base
    by_text: dict[str, str] = {}
    for label, text in display.items():
        norm = _norm(text)
        if norm in by_text:
            base["parse_reason"] = "duplicate_display_text"
            return [], base
        by_text[norm] = label
    labels: list[str] = []
    for piece in pieces:
        if piece not in by_text:
            base["parse_reason"] = "content_not_exactly_mapped"
            return [], base
        labels.append(by_text[piece])
    qtype = str(candidate["question_type"])
    canonical = [display_to_canonical[label] for label in labels]
    if qtype == "single_choice":
        correct = len(canonical) == 1 and canonical == gold
    elif qtype == "multi_select":
        correct = set(canonical) == set(gold)
    else:
        correct = canonical == gold
    base.update({"parse_status": "success", "failure_type": None if correct else "reasoning_error", "correct": correct, "parsed_display_labels": labels, "parsed_canonical": canonical})
    return canonical, base


def _guided_generate(backend: VLLMSolverBackend, request: SolverRequest, candidate: Mapping[str, Any], temperature: float) -> BackendResponse:
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
        structured_key = "guided_decoding"
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams
        GuidedDecodingParams = StructuredOutputsParams
        structured_key = "structured_outputs"
    qtype = str(candidate["question_type"])
    choices = structured_box_choices(qtype, allowed_item_ids=request.options.keys())
    params = SamplingParams(temperature=temperature, top_p=backend.top_p, max_tokens=32, seed=request.sampling_seed, stop=["</box>"], include_stop_str_in_output=True, **{structured_key: GuidedDecodingParams(choice=choices)})
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    return BackendResponse(completion.text, backend.model_identifier, {"prompt_token_count": len(outputs[0].prompt_token_ids), "completion_token_count": len(completion.token_ids), "finish_reason": completion.finish_reason, "max_tokens": 32, "temperature": temperature, "guided_decoding": True})


def _content_generate(backend: VLLMSolverBackend, request: SolverRequest, candidate: Mapping[str, Any], display: Mapping[str, str], display_to_canonical: Mapping[str, str], reserve: int, temperature: float) -> BackendResponse:
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
        structured_key = "guided_decoding"
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams
        GuidedDecodingParams = StructuredOutputsParams
        structured_key = "structured_outputs"
    choices = _content_choices(candidate, display, display_to_canonical)
    if not choices:
        raise ValueError("content_guided_choices_empty")
    params = SamplingParams(temperature=temperature, top_p=backend.top_p, max_tokens=reserve, seed=request.sampling_seed, stop=["</box>"], include_stop_str_in_output=True, **{structured_key: GuidedDecodingParams(choice=choices)})
    outputs = backend.engine.generate([request.prompt], params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    return BackendResponse(completion.text, backend.model_identifier, {"prompt_token_count": len(outputs[0].prompt_token_ids), "completion_token_count": len(completion.token_ids), "finish_reason": completion.finish_reason, "max_tokens": reserve, "temperature": temperature, "guided_decoding": True, "structured_output_api": structured_key})


def _invoke(*, run_root: Path, ledger: RequestLedger, backend: VLLMSolverBackend, tokenizer: Any, candidate: Mapping[str, Any], permutation: Mapping[str, Any], mode: str, output_mode: str) -> dict[str, Any]:
    display, displayed_gold, mapping = _display(candidate, permutation, output_mode)
    qtype = str(candidate["question_type"])
    context = _dialogue(_load(Path(str(candidate["_dialogue_path"])))) if mode == "full_context" else []
    prompt = (_label_prompt(tokenizer, candidate, display, context) if output_mode == OUTPUT_MODE_LABEL else _content_prompt(tokenizer, candidate, display, context, visible_labels=output_mode == OUTPUT_MODE_CONTENT_VISIBLE))
    reserve = OUTPUT_TOKENS_LABEL if output_mode == OUTPUT_MODE_LABEL else _content_reserve(tokenizer, candidate, display)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    request_id = f"{candidate['candidate_id']}__{permutation['permutation_id']}__{mode}__{output_mode}__s00"
    seed = stable_seed(str(candidate["root_id"]), str(candidate["candidate_id"]), 0)
    row: dict[str, Any] = {"request_id": request_id, "candidate_id": candidate["candidate_id"], "root_id": candidate["root_id"], "question_type": qtype, "permutation_id": permutation["permutation_id"], "is_cyclic": bool(permutation["is_cyclic"]), "display_order": permutation["display_order"], "context_mode": mode, "output_mode": output_mode, "slot_id": 0, "seed": seed, "temperature": SINGLE_TEMPERATURE if qtype == "single_choice" else STRUCTURED_TEMPERATURE, "prompt_sha256": sha256_bytes(prompt.encode()), "prompt_token_count": prompt_tokens, "output_reserve": reserve, "canonical_gold": list(candidate.get("gold") or ([candidate.get("correct_option")] if qtype == "single_choice" else [])), "displayed_gold": displayed_gold, "content_responder_prompt_version": CONTENT_RESPONDER_PROMPT_VERSION if output_mode != OUTPUT_MODE_LABEL else None}
    if prompt_tokens + reserve > MODEL_LEN:
        row.update({"technical_status": "CONTEXT_OVERFLOW", "scored": False, "correct": None, "failure_type": "context_overflow", "raw_output": None, "parsed_answer": []})
        return row
    options = dict(display)
    canonical_answer = ",".join(displayed_gold)
    request = SolverRequest(request_id=request_id, qa_id=str(candidate["candidate_id"]), evaluation_mode=mode, order_id=str(permutation["permutation_id"]), target_slot=0, retry_index=0, sampling_seed=seed, options=options, canonical_answer_text=canonical_answer, prompt_sha256=row["prompt_sha256"], prompt=prompt)
    atomic_text(run_root / "raw_prompts" / f"{request_id}.txt", prompt)
    try:
        response = ledger.invoke(request_id=request_id, intent="h4_position_ab_solver", attempt=1, prompt_sha256=row["prompt_sha256"], call=lambda: _guided_generate(backend, request, candidate, row["temperature"]) if output_mode == OUTPUT_MODE_LABEL and qtype != "single_choice" else (_content_generate(backend, request, candidate, display, mapping, reserve, row["temperature"]) if output_mode != OUTPUT_MODE_LABEL else backend.generate(request)), budget_class="position_ab_solver", budget_limit=2232, total_budget_limit=2232, request_metadata={"probe_version": "h4_position_permutation_ab_probe_v1", "candidate_id": candidate["candidate_id"], "root_id": candidate["root_id"], "permutation_id": permutation["permutation_id"], "context_mode": mode, "output_mode": output_mode, "seed": seed, "slot_id": 0, "seed_schedule_version": "h4_position_ab_paired_seed_v1", "parser_version": ("boxed_letter_v1" if output_mode == OUTPUT_MODE_LABEL and qtype == "single_choice" else "structured_qa_box_v1" if output_mode == OUTPUT_MODE_LABEL else CONTENT_PARSER_VERSION), "content_responder_prompt_version": row["content_responder_prompt_version"], "guided_decoding_version": ("content_choice_set_v1" if output_mode != OUTPUT_MODE_LABEL else None), "generation_config": {"temperature": row["temperature"], "top_p": backend.top_p, "max_tokens": reserve}}, backend_identity={"backend": "VLLMSolverBackend", "model": backend.model_identifier, "temperature": row["temperature"], "top_p": backend.top_p, "max_tokens": reserve})
    except Exception as exc:
        row.update({"technical_status": "SOLVER_ERROR", "scored": False, "correct": None, "failure_type": type(exc).__name__, "error": str(exc), "raw_output": None, "parsed_answer": []})
        return row
    row["raw_output"] = response.raw_text
    row["backend_metadata"] = dict(response.metadata or {})
    parsed, audit = (_label_parse(response.raw_text, candidate, display, displayed_gold) if output_mode == OUTPUT_MODE_LABEL else _content_parse(response.raw_text, candidate, display, mapping, [str(x) for x in (candidate.get("gold") or ([candidate.get("correct_option")] if qtype == "single_choice" else []))]))
    row.update({"technical_status": None, "scored": audit.get("parse_status") in {"success", "valid_boxed_letter"}, "correct": audit.get("correct"), "failure_type": audit.get("failure_type"), "parsed_answer": parsed, "parse": audit})
    ledger.record_decision(request_id, row["failure_type"] or "correct")
    return row


def load_sample_candidates(run_root: Path) -> list[dict[str, Any]]:
    rows = _lines(run_root / "sample_manifest.jsonl")
    cache: dict[str, list[dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []
    for sample in rows:
        cid = str(sample["candidate_id"])
        pool_path = Path(str((sample.get("lineage") or {}).get("candidate_pool_path") or ""))
        if cid not in cache:
            pool = _load(pool_path)
            cache[cid] = [dict(row) for row in pool if str(row.get("candidate_id")) == cid]
        if len(cache[cid]) != 1:
            raise ValueError(f"candidate_lineage_not_unique:{cid}")
        candidate = dict(cache[cid][0])
        dialogue_path = Path(str((sample.get("lineage") or {}).get("dialogue_path") or ""))
        candidate["_dialogue_path"] = str(dialogue_path)
        candidate["root_id"] = str(sample["root_id"])
        output.append(candidate)
    return output


def load_permutations(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows = _lines(run_root / "permutation_manifest.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    return grouped


def load_result_index(run_root: Path) -> dict[str, dict[str, Any]]:
    """Read completed result records once so resume does not rescan per request."""
    index: dict[str, dict[str, Any]] = {}
    result_dir = run_root / "results"
    if not result_dir.is_dir():
        return index
    for path in sorted(result_dir.glob("*.json")):
        try:
            row = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        request_id = row.get("request_id")
        if request_id:
            index[str(request_id)] = row
    return index


def _write_result(run_root: Path, row: Mapping[str, Any]) -> None:
    digest = hashlib.sha256(str(row["request_id"]).encode()).hexdigest()[:24]
    atomic_json(run_root / "results" / f"{digest}.json", dict(row))


def run_requests(*, run_root: Path, candidates: list[dict[str, Any]], permutations: Mapping[str, list[dict[str, Any]]], backend: VLLMSolverBackend, tokenizer: Any, ledger: RequestLedger, output_mode: str, include_broader: bool, candidate_limit: int | None = None, cyclic_only: bool = False, result_index: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"unsupported_output_mode:{output_mode}")
    if cyclic_only and include_broader:
        raise ValueError("cyclic_only_conflicts_with_include_broader")
    selected = candidates if candidate_limit is None else candidates[:candidate_limit]
    results: list[dict[str, Any]] = []
    existing_index = result_index if result_index is not None else load_result_index(run_root)
    for candidate in selected:
        perms = list(permutations[str(candidate["candidate_id"])])
        if cyclic_only:
            perms = [perm for perm in perms if perm["is_cyclic"]]
        if not include_broader:
            perms = [perm for perm in perms if perm["is_cyclic"]]
        for permutation in perms:
            for mode in ("no_context", "full_context"):
                request_id = f"{candidate['candidate_id']}__{permutation['permutation_id']}__{mode}__{output_mode}__s00"
                existing = existing_index.get(request_id)
                row = existing or _invoke(run_root=run_root, ledger=ledger, backend=backend, tokenizer=tokenizer, candidate=candidate, permutation=permutation, mode=mode, output_mode=output_mode)
                if not existing:
                    _write_result(run_root, row)
                    existing_index[request_id] = row
                results.append(row)
    return results


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("scored")]
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("no_context", "full_context"):
        subset = [row for row in complete if row.get("context_mode") == mode]
        by_mode[mode] = {"count": len(subset), "accuracy": sum(bool(row.get("correct")) for row in subset) / len(subset) if subset else None, "technical_failures": len([row for row in rows if row.get("context_mode") == mode and not row.get("scored")])}
    correct = [row for row in complete if row.get("correct") is True]
    return {
        "row_count": len(rows), "scored_count": len(complete),
        "correct_count": len(correct),
        "accuracy": len(correct) / len(complete) if complete else None,
        "technical_failure_count": len(rows) - len(complete),
        "by_context": by_mode,
        "by_output_mode": dict(Counter(str(row.get("output_mode")) for row in rows)),
        "by_question_type": dict(Counter(str(row.get("question_type")) for row in rows)),
        "by_cyclic": {
            "cyclic": summarize_subset([row for row in rows if row.get("is_cyclic")]),
            "non_cyclic": summarize_subset([row for row in rows if not row.get("is_cyclic")]),
        },
    }


def summarize_subset(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("scored")]
    return {
        "count": len(rows),
        "scored_count": len(scored),
        "accuracy": sum(bool(row.get("correct")) for row in scored) / len(scored) if scored else None,
        "technical_failure_count": len(rows) - len(scored),
    }


__all__ = ["load_sample_candidates", "load_permutations", "load_result_index", "run_requests", "summarize", "summarize_subset", "OUTPUT_MODE_LABEL", "OUTPUT_MODE_CONTENT_VISIBLE", "OUTPUT_MODE_CONTENT_HIDDEN", "CONTENT_RESPONDER_PROMPT_VERSION", "SOLVER_CONTENT_RESPONDER_PROMPT", "SOLVER_B2_CONTENT_RESPONDER_PROMPT"]
