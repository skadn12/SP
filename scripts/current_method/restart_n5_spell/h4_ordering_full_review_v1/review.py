"""Offline semantic review of frozen ordering candidates.

This is an audit helper, not the project Solver. It reads frozen manifests and
dialogues, asks a local model to locate the four event statements, and writes
row-level review evidence. It has no network, Memorax, training, or evaluation
result writes.
"""

from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, sampling_params


BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_DATA = BASE / "h4_round0_b0_matched_training_v1_live_1"
DEFAULT_MODEL = Path("/root/autodl-tmp/Qwen3-4B")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dialogue(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(value.get("turns") or [])


def candidate_prompt(row: Mapping[str, Any], turns: list[Mapping[str, Any]]) -> str:
    c = row["candidate"]
    lines = [
        "You are an offline QA data auditor. Do not solve the question from memory.",
        "For each item, locate the earliest dialogue turn that directly states the item.",
        "Use turn_ref values exactly as printed. Ignore the order of supplied evidence_refs when judging chronology.",
        "A claim is unsupported if the dialogue does not directly state it; a claim is ambiguous if it combines events, has multiple materially different occurrences, or its scope is unclear.",
        "Set gold_status to valid only when all four claims are directly grounded, have one defensible occurrence, and their chronological order equals gold.",
        "Set gold_status to invalid_gold when all four claims are grounded and unambiguous but the chronological order differs from gold.",
        "Set gold_status to ambiguous when repetition or interpretation prevents one defensible order; set it to unsupported when any item is not directly grounded.",
        "Set evidence_status to clean only when every supplied evidence ref directly supports the corresponding item; otherwise mismatch.",
        "Return JSON only with keys: gold_status, evidence_status, actual_order, items. Do not output analysis, quotations, or any text before or after the JSON.",
        "items must be an array of four objects with keys item_id, earliest_turn_refs, support (direct|indirect|none|ambiguous), confidence (high|medium|low), note. Keep each note under 12 words.",
        "Do not infer an event merely because it is topically related. A retrospective mention is not an earlier occurrence.",
        "",
        f"candidate_id: {c.get('candidate_id')}",
        f"question: {c.get('question')}",
        f"items: {json.dumps(c.get('items') or {}, ensure_ascii=False, sort_keys=True)}",
        f"gold: {json.dumps(c.get('gold') or [], ensure_ascii=False)}",
        f"evidence_refs: {json.dumps(c.get('evidence_refs') or [], ensure_ascii=False)}",
        "",
        "DIALOGUE:",
    ]
    lines.extend(f"{t.get('turn_ref')} | {t.get('text')}" for t in turns)
    return "\n".join(lines)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"parse_status": "invalid_json", "raw_output": text}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"parse_status": "invalid_json", "raw_output": text}
    if not isinstance(value, dict):
        return {"parse_status": "invalid_json", "raw_output": text}
    value["parse_status"] = "ok"
    value["raw_output"] = text
    return value


def normalize(row: Mapping[str, Any], audit: Mapping[str, Any], turns: list[Mapping[str, Any]]) -> dict[str, Any]:
    c = row["candidate"]
    valid_refs = {str(t.get("turn_ref")) for t in turns}
    items = audit.get("items") if isinstance(audit.get("items"), list) else []
    cleaned = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        refs = [str(ref) for ref in item.get("earliest_turn_refs") or [] if str(ref) in valid_refs]
        cleaned.append({
            "item_id": str(item.get("item_id") or ""),
            "earliest_turn_refs": refs,
            "support": str(item.get("support") or ""),
            "confidence": str(item.get("confidence") or ""),
            "note": str(item.get("note") or ""),
        })
    positions = {str(t.get("turn_ref")): i for i, t in enumerate(turns)}
    by_item = {str(item.get("item_id")): item for item in cleaned}
    item_ids = list((c.get("items") or {}).keys())
    item_order = list(c.get("gold") or [])
    supports = [str(by_item.get(item_id, {}).get("support") or "") for item_id in item_ids]
    refs_by_item = [by_item.get(item_id, {}).get("earliest_turn_refs") or [] for item_id in item_ids]
    flat_refs = [ref for refs in refs_by_item for ref in refs]
    all_direct = len(cleaned) == len(item_ids) and all(s == "direct" for s in supports)
    any_ambiguous = any(s == "ambiguous" for s in supports)
    any_unsupported = any(s in {"none", "indirect", ""} for s in supports)
    unique_single_refs = all(len(refs) == 1 for refs in refs_by_item) and len(set(flat_refs)) == len(flat_refs)
    chronological_ids = (
        [item_id for _, item_id in sorted((positions[refs[0]], item_id) for item_id, refs in zip(item_ids, refs_by_item))]
        if unique_single_refs and all(refs[0] in positions for refs in refs_by_item)
        else []
    )
    if audit.get("parse_status") != "ok" or len(cleaned) != len(item_ids):
        derived_gold_status = "unparsed"
    elif any_ambiguous or not unique_single_refs:
        derived_gold_status = "ambiguous"
    elif any_unsupported:
        derived_gold_status = "unsupported"
    elif chronological_ids == item_order:
        derived_gold_status = "valid"
    else:
        derived_gold_status = "invalid_gold"
    evidence_refs = [str(ref) for ref in c.get("evidence_refs") or []]
    supplied_refs_valid = all(ref in positions for ref in evidence_refs)
    model_evidence_status = str(audit.get("evidence_status") or "unparsed")
    derived_evidence_status = "clean" if model_evidence_status == "clean" and supplied_refs_valid else "mismatch"
    actual_order = chronological_ids or [str(x) for x in audit.get("actual_order") or []]
    return {
        "candidate_id": str(c.get("candidate_id")),
        "root_id": str(c.get("root_id")),
        "question": c.get("question"),
        "items": c.get("items") or {},
        "gold": c.get("gold") or [],
        "evidence_refs": c.get("evidence_refs") or [],
        "generation_source": c.get("generation_source"),
        "dialogue_path": row.get("dialogue_path"),
        "gold_status": derived_gold_status,
        "model_gold_status": str(audit.get("gold_status") or "unparsed"),
        "evidence_status": derived_evidence_status,
        "model_evidence_status": model_evidence_status,
        "actual_order": actual_order,
        "items_review": cleaned,
        "parse_status": audit.get("parse_status"),
        "raw_output": audit.get("raw_output"),
        "audit_method": "local_qwen3_4b_offline_locator_v1",
    }


def compact_prompt(row: Mapping[str, Any], turns: list[Mapping[str, Any]]) -> str:
    c = row["candidate"]
    lines = [
        "Audit this ordering QA item offline. Return exactly five lines and nothing else.",
        "Lines 1-4: ITEM<TAB>EARLIEST_TURN_REF<TAB>SUPPORT<TAB>CONFIDENCE.",
        "Use item IDs I1,I2,I3,I4 exactly once. Use one earliest direct turn ref per item.",
        "SUPPORT is direct, indirect, none, or ambiguous. CONFIDENCE is high, medium, or low.",
        "Line 5: E<TAB>clean or E<TAB>mismatch. Evidence is clean only if every supplied evidence ref directly supports the corresponding item.",
        "If a claim is not directly stated or has no unique occurrence, use none or ambiguous.",
        f"ITEMS: {json.dumps(c.get('items') or {}, ensure_ascii=False, sort_keys=True)}",
        f"GOLD: {json.dumps(c.get('gold') or [], ensure_ascii=False)}",
        f"EVIDENCE_REFS: {json.dumps(c.get('evidence_refs') or [], ensure_ascii=False)}",
        "DIALOGUE:",
    ]
    lines.extend(f"{t.get('turn_ref')} | {t.get('text')}" for t in turns)
    return "\n".join(lines)


def parse_compact(text: str) -> dict[str, Any]:
    rows = []
    evidence_status = "unparsed"
    for raw in text.strip().splitlines():
        parts = raw.split("\t")
        if len(parts) == 2 and parts[0] == "E":
            evidence_status = parts[1].strip()
            continue
        if len(parts) == 4 and parts[0] in {"I1", "I2", "I3", "I4"}:
            rows.append({
                "item_id": parts[0], "earliest_turn_refs": [parts[1].strip()],
                "support": parts[2].strip(), "confidence": parts[3].strip(), "note": "compact_retry",
            })
    return {"parse_status": "ok" if len(rows) == 4 else "invalid_compact", "items": rows,
            "evidence_status": evidence_status, "raw_output": text}


def pairwise_prompt(item_id: str, item_text: str, anchor: str, turns: list[Mapping[str, Any]]) -> str:
    lines = [
        "Select the dialogue turn that directly states this event occurrence.",
        "The supplied evidence anchor is the intended occurrence hint, but it may be wrong or mismatched; verify its text against the item and correct it when needed.",
        "Return only the turn_ref. Choose AMBIGUOUS if materially different occurrences prevent one defensible occurrence. Choose NONE if the event is not directly stated.",
        "Do not choose a later sentence merely because it recalls an earlier event, unless it is the only direct statement of the item.",
        f"ITEM {item_id}: {item_text}",
        f"SUPPLIED_EVIDENCE_ANCHOR: {anchor or 'NONE'}",
        "DIALOGUE:",
    ]
    lines.extend(f"{t.get('turn_ref')} | {t.get('text')}" for t in turns)
    return "\n".join(lines)


def ref_scope(ref: str) -> str:
    parts = str(ref).split(":")
    if len(parts) >= 3 and parts[-3] == "R01" and parts[-2].startswith("S"):
        return ":".join(parts[-3:-1])
    if len(parts) >= 3 and parts[-2].startswith("P"):
        return parts[-2]
    return ""


def scoped_pairwise_prompt(item_id: str, item_text: str, anchor: str, turns: list[Mapping[str, Any]]) -> str:
    scope = ref_scope(anchor)
    lines = [
        "Select the event occurrence for this ordering QA item within the scoped scene.",
        "The anchor ref identifies the intended event-chain scene; do not select a retrospective mention from another scene.",
        "Return only the turn_ref. Choose AMBIGUOUS if the scoped scene has multiple materially different occurrences. Choose NONE if it is not directly stated in this scope.",
        f"ITEM {item_id}: {item_text}", f"ANCHOR_REF: {anchor}", f"ANCHOR_SCOPE: {scope}", "SCOPED DIALOGUE:",
    ]
    lines.extend(f"{t.get('turn_ref')} | {t.get('text')}" for t in turns)
    return "\n".join(lines)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "about", "after", "before", "over", "under", "that", "this", "these",
    "those", "was", "were", "is", "are", "be", "been", "being", "he", "she", "they",
    "them", "his", "her", "their", "there", "here", "it", "its", "who", "what", "which",
    "how", "why", "when", "where", "some", "someone", "something", "event", "events",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
        if token not in _STOPWORDS
    }


def retrieved_turns(item_text: str, anchor: str, turns: list[Mapping[str, Any]], limit: int = 14) -> list[Mapping[str, Any]]:
    """Return a small chronological candidate set for one item.

    Lexical retrieval is only a prompt-length/attention aid. Evidence anchors and
    their immediate neighbors are retained even when their wording is paraphrased.
    """
    query = _tokens(item_text)
    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, turn in enumerate(turns):
        text = str(turn.get("text") or "")
        words = _tokens(text)
        overlap = len(query & words)
        # Favor direct lexical matches, with a small length normalization so a
        # long recap does not dominate a short event statement.
        score = overlap / math.sqrt(max(len(words), 1))
        if overlap and query.issubset(words):
            score += 2.0
        scored.append((score, index, turn))

    keep: set[int] = set()
    ref_to_index = {str(turn.get("turn_ref")): index for index, turn in enumerate(turns)}
    if anchor in ref_to_index:
        index = ref_to_index[anchor]
        keep.update({index, max(0, index - 1), min(len(turns) - 1, index + 1)})
    for score, index, _turn in sorted(scored, key=lambda value: (-value[0], value[1]))[:limit]:
        if score > 0:
            keep.add(index)
    # Preserve dialogue chronology in the prompt; the model only chooses among
    # these turns and cannot accidentally emit a turn from another root.
    return [turns[index] for index in sorted(keep)]


def retrieval_pairwise_prompt(item_id: str, item_text: str, anchor: str, turns: list[Mapping[str, Any]]) -> str:
    lines = [
        "Choose the single dialogue turn that directly states this event.",
        "Return only one exact turn_ref from the candidates below.",
        "Choose NONE if no candidate directly states the event. Choose AMBIGUOUS only if two candidates are materially different occurrences and neither can be preferred as the event itself.",
        "A later sentence that merely recalls an earlier event is not the event occurrence.",
        f"ITEM {item_id}: {item_text}",
        f"SUPPLIED_EVIDENCE_ANCHOR: {anchor or 'NONE'}",
        "CANDIDATE_TURNS:",
    ]
    lines.extend(f"{t.get('turn_ref')} | {t.get('text')}" for t in turns)
    return "\n".join(lines)


def extract_choice(text: str, valid: set[str]) -> str:
    cleaned = text.strip()
    if cleaned in valid or cleaned in {"NONE", "AMBIGUOUS"}:
        return cleaned
    # Guided decoding can still return a short explanation. Pick an exact
    # candidate token only when it appears verbatim in the output.
    for ref in sorted(valid, key=len, reverse=True):
        if ref in cleaned:
            return ref
    upper = cleaned.upper()
    if "AMBIGUOUS" in upper:
        return "AMBIGUOUS"
    return "NONE"


def pairwise_normalize(
    row: Mapping[str, Any],
    choices_by_item: list[list[str]],
    outputs: list[Any],
    turns: list[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    c = row["candidate"]
    valid = {str(t.get("turn_ref")) for t in turns}
    item_ids = list((c.get("items") or {}).keys())
    selected: dict[str, str] = {}
    raw: list[str] = []
    for item_id, choices, output in zip(item_ids, choices_by_item, outputs):
        text = output.outputs[0].text.strip() if output.outputs else ""
        raw.append(text)
        selected[item_id] = extract_choice(text, set(choices) & valid)
    pos = {str(t.get("turn_ref")): i for i, t in enumerate(turns)}
    ordered = [item_id for _, item_id in sorted((pos[ref], item_id) for item_id, ref in selected.items() if ref in pos)]
    all_selected = len(selected) == len(item_ids) and all(ref in valid for ref in selected.values())
    has_ambiguous = any(ref == "AMBIGUOUS" for ref in selected.values())
    has_none = any(ref == "NONE" for ref in selected.values())
    if has_ambiguous:
        status = "ambiguous"
    elif has_none or not all_selected or len(set(ordered)) != len(item_ids):
        status = "unsupported"
    elif ordered == list(c.get("gold") or []):
        status = "valid"
    else:
        status = "invalid_gold"
    evidence_refs = [str(ref) for ref in c.get("evidence_refs") or []]
    evidence_status = "clean"
    for item_id, ref in zip(item_ids, evidence_refs):
        if selected.get(item_id) != ref:
            evidence_status = "mismatch"
    return {
        "candidate_id": str(c.get("candidate_id")),
        "root_id": str(c.get("root_id")),
        "question": c.get("question"),
        "items": c.get("items") or {},
        "gold": c.get("gold") or [],
        "evidence_refs": evidence_refs,
        "dialogue_path": row.get("dialogue_path"),
        "gold_status": status,
        "evidence_status": evidence_status,
        "actual_order": ordered,
        "selected_turn_refs": selected,
        "raw_outputs": raw,
        "parse_status": "ok",
        "audit_method": method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-run", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--scoped-pairwise", action="store_true")
    parser.add_argument("--retrieval-pairwise", action="store_true")
    parser.add_argument("--anchored-pairwise", action="store_true")
    args = parser.parse_args()

    splits = ("dev", "test") if args.split == "both" else (args.split,)
    rows: list[dict[str, Any]] = []
    for split in splits:
        manifest = load_jsonl(args.data_run / f"{split}_full_eval_manifest.jsonl")
        rows.extend(row for row in manifest if row.get("candidate", {}).get("question_type") == "ordering")
    rows.sort(key=lambda row: str(row["candidate"]["candidate_id"]))
    if args.ids_file is not None:
        wanted = {line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        rows = [row for row in rows if str(row["candidate"]["candidate_id"]) in wanted]
    if args.limit:
        rows = rows[:args.limit]

    dialogues: dict[str, list[dict[str, Any]]] = {}
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    prompts: list[str] = []
    pairwise_meta: list[tuple[dict[str, Any], list[str], list[str]]] = []
    for row in rows:
        path = str(row.get("dialogue_path"))
        if path not in dialogues:
            dialogues[path] = load_dialogue(Path(path))
        if args.pairwise or args.scoped_pairwise or args.retrieval_pairwise or args.anchored_pairwise:
            c = row["candidate"]
            for item_id, item_text in (c.get("items") or {}).items():
                all_turns = dialogues[path]
                evidence = [str(x) for x in c.get("evidence_refs") or []]
                anchor = evidence[list((c.get("items") or {}).keys()).index(item_id)] if len(evidence) == len(c.get("items") or {}) else ""
                if args.retrieval_pairwise:
                    turns_for_prompt = retrieved_turns(str(item_text), anchor, all_turns)
                    content = retrieval_pairwise_prompt(str(item_id), str(item_text), anchor, turns_for_prompt)
                    refs = [str(t.get("turn_ref")) for t in turns_for_prompt] + ["NONE", "AMBIGUOUS"]
                elif args.scoped_pairwise and anchor:
                    scope = ref_scope(anchor)
                    scoped = [t for t in all_turns if ref_scope(str(t.get("turn_ref"))) == scope]
                    turns_for_prompt = scoped or all_turns
                    content = scoped_pairwise_prompt(str(item_id), str(item_text), anchor, turns_for_prompt)
                    refs = [str(t.get("turn_ref")) for t in turns_for_prompt] + ["NONE", "AMBIGUOUS"]
                else:
                    turns_for_prompt = all_turns
                    content = pairwise_prompt(str(item_id), str(item_text), anchor if args.anchored_pairwise else "", turns_for_prompt)
                    refs = [str(t.get("turn_ref")) for t in turns_for_prompt] + ["NONE", "AMBIGUOUS"]
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ))
                pairwise_meta.append((row, refs, [str(item_id)]))
        else:
            content = compact_prompt(row, dialogues[path]) if args.compact else candidate_prompt(row, dialogues[path])
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ))

    llm = LLM(
        model=str(args.model), tokenizer=str(args.model), dtype="auto",
        max_model_len=32768, gpu_memory_utilization=0.85,
        enable_prefix_caching=True, trust_remote_code=True,
    )
    output_schema = {
        "type": "object",
        "properties": {
            "gold_status": {"enum": ["valid", "invalid_gold", "ambiguous", "unsupported"]},
            "evidence_status": {"enum": ["clean", "mismatch"]},
            "actual_order": {"type": "array", "items": {"enum": ["I1", "I2", "I3", "I4"]}},
            "items": {
                "minItems": 4, "maxItems": 4,
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"enum": ["I1", "I2", "I3", "I4"]},
                        "earliest_turn_refs": {"type": "array", "items": {"type": "string"}},
                        "support": {"enum": ["direct", "indirect", "none", "ambiguous"]},
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "note": {"type": "string"},
                    },
                    "required": ["item_id", "earliest_turn_refs", "support", "confidence", "note"],
                },
            },
        },
        "required": ["gold_status", "evidence_status", "actual_order", "items"],
    }
    if args.retrieval_pairwise or args.anchored_pairwise:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
            guided_decoding=sampling_params.GuidedDecodingParams(
                choice=list({ref for _, refs, _ in pairwise_meta for ref in refs})
            ),
        )
    elif args.pairwise or args.scoped_pairwise:
        params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=24,
            guided_decoding=sampling_params.GuidedDecodingParams(choice=list({ref for _, refs, _ in pairwise_meta for ref in refs})),
        )
    elif args.compact:
        params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=120,
            guided_decoding=sampling_params.GuidedDecodingParams(
                regex=r"(?:I[1-4]\t[A-Za-z0-9:._-]{1,180}\t(?:direct|indirect|none|ambiguous)\t(?:high|medium|low)\n){4}E\t(?:clean|mismatch)"
            ),
        )
    else:
        params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=420,
            guided_decoding=sampling_params.GuidedDecodingParams(json=output_schema),
        )
    outputs = llm.generate(
        prompts,
        params,
        use_tqdm=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        if args.pairwise or args.scoped_pairwise or args.retrieval_pairwise or args.anchored_pairwise:
            by_row: dict[str, list[tuple[Any, list[str]]]] = {}
            for meta, output in zip(pairwise_meta, outputs):
                by_row.setdefault(str(meta[0]["candidate"]["candidate_id"]), []).append((output, meta[1]))
            for row in rows:
                path = str(row.get("dialogue_path"))
                c = row["candidate"]
                grouped = by_row[str(c["candidate_id"])]
                method = (
                    "local_qwen3_4b_retrieval_pairwise_turn_choice_v1" if args.retrieval_pairwise
                    else "local_qwen3_4b_anchored_pairwise_turn_choice_v1" if args.anchored_pairwise
                    else "local_qwen3_4b_scoped_pairwise_turn_choice_v1" if args.scoped_pairwise
                    else "local_qwen3_4b_pairwise_turn_choice_v1"
                )
                result = pairwise_normalize(
                    row,
                    [choices for _output, choices in grouped],
                    [output for output, _choices in grouped],
                    dialogues[path],
                    method,
                )
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            for row, output in zip(rows, outputs):
                text = output.outputs[0].text if output.outputs else ""
                audit = parse_compact(text) if args.compact else parse_json(text)
                path = str(row.get("dialogue_path"))
                result = normalize(row, audit, dialogues[path])
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "audit_version": "h4_ordering_full_review_v1",
        "offline_only": True,
        "solver_calls": 0,
        "external_api_calls": 0,
        "memorax_calls": 0,
        "model": str(args.model),
        "split": args.split,
        "candidate_count": len(rows),
        "output": str(args.output),
    }
    (args.output.parent / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
