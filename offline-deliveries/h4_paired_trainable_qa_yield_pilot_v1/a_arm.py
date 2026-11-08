"""Frozen source adapter and shape-only audit for the paired pilot's Writer arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


V2_INSTRUCTIONS = Path(__file__).resolve().parent.parent / "h4_graph_guided_qa_pilot_v2" / "writer_instructions.txt"
V2_INSTRUCTIONS_SHA256 = "6d78bda6a4609e1af591266da3bf57ffb25f4076159aac3989a066686617fc66"
QUESTION_TYPES = {"multi_select", "ordering"}
CAP = {"multi_select": 2, "ordering": 1}


def _surface(surface: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(surface, dict):
        raise ValueError("surface_not_object")
    root_id, source_sha, turns = (surface.get(key) for key in ("root_id", "source_sha256", "turns"))
    if not isinstance(root_id, str) or not root_id.strip():
        raise ValueError("root_id_invalid")
    if not isinstance(source_sha, str) or len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("source_sha256_invalid")
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns_invalid")
    projected = []
    for row in turns:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("turn_ref", "speaker", "text")):
            raise ValueError("turn_invalid")
        projected.append({key: row[key] for key in ("turn_ref", "speaker", "text")})
    refs = [row["turn_ref"] for row in projected]
    if len(refs) != len(set(refs)):
        raise ValueError("duplicate_turn_refs")
    return {"root_id": root_id, "source_sha256": source_sha, "turns": projected}


def _adapt_instructions() -> str:
    source = V2_INSTRUCTIONS.read_bytes()
    if hashlib.sha256(source).hexdigest() != V2_INSTRUCTIONS_SHA256:
        raise RuntimeError("v2_writer_instructions_changed")
    instructions = source.decode("utf-8")
    replacements = (
        ("a frozen 188-turn history followed by new G02 scenes", "a complete promoted P+R01+R02 dialogue"),
        ("Write at most ONE multi_select and ONE ordering question.", "Write at most TWO multi_select questions or ONE ordering question, as specified for this attempt."),
        ("from the frozen R03 history with a fact from G02, or facts from TWO DISTINCT G02 scenes", "from the earlier P/R01 dialogue with a fact from R02, or facts from TWO DISTINCT R02 scenes"),
        ("from G02", "from R02"),
        ("span all three G02 scenes", "span at least two distinct R02 scenes"),
        ("G02:", "R02:"),
    )
    for old, new in replacements:
        if old not in instructions:
            raise RuntimeError(f"v2_writer_adapter_anchor_missing:{old}")
        instructions = instructions.replace(old, new)
    return instructions


def prompt(surface: dict[str, Any], question_type: str) -> str:
    """Build the single-attempt, type-scoped Writer prompt from Solver-visible turns."""
    if question_type not in QUESTION_TYPES:
        raise ValueError("question_type_invalid")
    visible = _surface(surface)
    instructions = _adapt_instructions()
    instruction = (
        f"THIS ATTEMPT: Submit only {question_type} questions, at most {CAP[question_type]}. "
        f"If there is no valid {question_type} candidate, return questions=[] and one "
        f"{question_type} abstention. Do not submit or abstain from the other type.\n\n"
    )
    marker = "VISIBLE DIALOGUE:\n"
    if not instructions.endswith(marker):
        raise RuntimeError("v2_writer_instruction_tail_changed")
    instructions = instructions[: -len(marker)] + instruction + marker
    return instructions + json.dumps({"turns": visible["turns"]}, ensure_ascii=False, indent=2) + "\n"


def _require(condition: bool, error: str) -> None:
    if not condition:
        raise ValueError(error)


def _refs(refs: Any, ref_map: dict[str, str], *, latest_only: bool = False) -> list[str]:
    _require(isinstance(refs, list) and bool(refs), "evidence_refs_invalid")
    _require(all(isinstance(ref, str) and ref in ref_map for ref in refs), "evidence_refs_invalid")
    if latest_only:
        _require(all(ref.startswith("R02:") for ref in refs), "ordering_requires_latest_round_refs")
    return refs


def _quote(row: dict[str, Any], refs: list[str], ref_map: dict[str, str]) -> None:
    quote = row.get("quote")
    _require(isinstance(quote, str) and bool(quote.strip()) and any(quote in ref_map[ref] for ref in refs), "quote_not_found_in_cited_speech")
    _require(isinstance(row.get("reason"), str) and bool(row["reason"].strip()), "evidence_reason_missing")


def _scene(ref: str) -> str:
    return ref.rsplit(":T", 1)[0]


def _candidate(row: dict[str, Any], question_type: str, ref_map: dict[str, str]) -> None:
    _require(row.get("question_type") == question_type, "candidate_question_type_invalid")
    for key in ("candidate_id", "intent", "question"):
        _require(isinstance(row.get(key), str) and bool(row[key].strip()), f"{key}_missing")
    field = "options" if question_type == "multi_select" else "items"
    labels = list("ABCDE") if question_type == "multi_select" else [f"I{i}" for i in range(1, 5)]
    values = row.get(field)
    _require(isinstance(values, dict) and list(values) == labels and all(isinstance(values[label], str) and values[label].strip() for label in labels), f"{field}_shape_invalid")
    _require(len(set(values.values())) == len(labels), f"{field}_duplicate_values")
    gold = row.get("gold")
    _require(isinstance(gold, list) and all(isinstance(label, str) and label in labels for label in gold) and len(gold) == len(set(gold)), "gold_shape_invalid")
    if question_type == "multi_select":
        _require(2 <= len(gold) <= 3, "multi_select_gold_size_invalid")
        evidence = row.get("option_evidence")
        _require(isinstance(evidence, dict) and set(evidence) == set(labels), "option_evidence_incomplete")
        for label in labels:
            item = evidence[label]
            _require(isinstance(item, dict) and item.get("verdict") == ("TRUE" if label in gold else "FALSE"), "option_verdict_mismatch")
            clauses = item.get("clauses")
            _require(isinstance(clauses, list) and len(clauses) == 2 and all(isinstance(clause, dict) for clause in clauses), "two_clauses_required")
            verdicts = []
            cited: list[str] = []
            for clause in clauses:
                _require(isinstance(clause.get("claim"), str) and bool(clause["claim"].strip()), "clause_claim_missing")
                _require(clause.get("verdict") in {"TRUE", "FALSE"}, "clause_verdict_invalid")
                verdicts.append(clause["verdict"])
                refs = _refs(clause.get("refs"), ref_map)
                cited.extend(refs)
                _quote(clause, refs, ref_map)
            _require(len(set(cited)) >= 2, "single_turn_option")
            _require(("TRUE" if all(v == "TRUE" for v in verdicts) else "FALSE") == item["verdict"], "clause_option_verdict_mismatch")
            if label in gold:
                first, second = clauses[0]["refs"], clauses[1]["refs"]
                crossed = any(ref.startswith("R02:") for ref in first) and any(not ref.startswith("R02:") for ref in second)
                crossed |= any(ref.startswith("R02:") for ref in second) and any(not ref.startswith("R02:") for ref in first)
                crossed |= any(a != b and a.startswith("R02:") and b.startswith("R02:") for a in map(_scene, first) for b in map(_scene, second))
                _require(crossed, "true_option_not_cross_scene_or_history")
    else:
        _require(len(gold) == 4, "ordering_gold_incomplete")
        evidence = row.get("item_evidence")
        _require(isinstance(evidence, dict) and set(evidence) == set(labels), "item_evidence_incomplete")
        tracks = []
        item_refs = {}
        for label in labels:
            item = evidence[label]
            _require(isinstance(item, dict) and isinstance(item.get("track"), str) and bool(item["track"].strip()), "item_track_missing")
            tracks.append(item["track"])
            refs = _refs(item.get("refs"), ref_map, latest_only=True)
            _require(len(refs) == 1, "ordering_item_must_cite_one_action_turn")
            item_refs[label] = refs
            _quote(item, refs, ref_map)
        _require(len(set(tracks)) == 2 and sum(evidence[a]["track"] != evidence[b]["track"] for a, b in zip(gold, gold[1:])) >= 2, "ordering_tracks_not_interleaved")
        _require(len({_scene(item_refs[label][0]) for label in gold}) >= 2, "ordering_does_not_span_two_scenes")
        _require(any(_scene(item_refs[a][0]) == _scene(item_refs[b][0]) and evidence[a]["track"] != evidence[b]["track"] for a, b in zip(gold, gold[1:])), "ordering_lacks_same_scene_cross_track_edge")
        _require(len({item_refs[label][0] for label in gold}) == 4, "ordering_actions_not_separate_turns")
        edges = row.get("adjacent_edges")
        _require(isinstance(edges, list) and len(edges) == 3, "adjacent_edges_incomplete")
        for index, edge in enumerate(edges):
            _require(isinstance(edge, dict) and (edge.get("before"), edge.get("after")) == (gold[index], gold[index + 1]), "adjacent_edge_order_invalid")
            before = _refs(edge.get("before_refs"), ref_map, latest_only=True)
            after = _refs(edge.get("after_refs"), ref_map, latest_only=True)
            _refs(edge.get("order_refs"), ref_map, latest_only=True)
            _require(bool(set(before) & set(item_refs[gold[index]])) and bool(set(after) & set(item_refs[gold[index + 1]])), "edge_not_grounded_in_item_turns")
            _require(not set(before) & set(after) and isinstance(edge.get("reason"), str) and bool(edge["reason"].strip()), "adjacent_edge_evidence_invalid")


def validate(raw: str, surface: dict[str, Any], question_type: str) -> dict[str, Any]:
    """Audit submitted shapes and citations; never certify semantics or chronology."""
    if question_type not in QUESTION_TYPES:
        raise ValueError("question_type_invalid")
    visible = _surface(surface)
    output: dict[str, Any] = {"status": "FAIL_SHAPE", "question_type": question_type, "source_sha256": visible["source_sha256"], "candidates": [], "errors": [], "abstentions": [], "semantic_verified": False, "chronology_verified": False, "training_promoted": False}
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        output["errors"].append("invalid_json")
        return output
    if not isinstance(result, dict) or not isinstance(result.get("questions"), list):
        output["errors"].append("questions_array_missing")
        return output
    questions = result["questions"]
    if len(questions) > CAP[question_type]:
        output["errors"].append("candidate_cap_exceeded")
    ids: set[str] = set()
    ref_map = {row["turn_ref"]: row["text"] for row in visible["turns"]}
    for index, row in enumerate(questions):
        record = {"index": index, "candidate_id": row.get("candidate_id") if isinstance(row, dict) else None, "status": "FAIL_SHAPE", "errors": [], "semantic_verified": False, "chronology_verified": False}
        if not isinstance(row, dict):
            record["errors"].append("candidate_not_object")
        else:
            try:
                _candidate(row, question_type, ref_map)
            except ValueError as exc:
                record["errors"].append(str(exc))
            candidate_id = row.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id in ids:
                record["errors"].append("duplicate_candidate_id")
            if isinstance(candidate_id, str):
                ids.add(candidate_id)
        if not record["errors"]:
            record["status"] = "PASS_SHAPE_ONLY_NOT_GOLD"
        output["candidates"].append(record)
    abstentions = result.get("abstentions")
    if not isinstance(abstentions, list):
        output["errors"].append("abstentions_array_missing")
    else:
        output["abstentions"] = abstentions
        if any(not isinstance(row, dict) or row.get("question_type") != question_type or not isinstance(row.get("reason"), str) or not row["reason"].strip() for row in abstentions):
            output["errors"].append("abstention_invalid")
        if bool(questions) == bool(abstentions) or len(abstentions) > 1:
            output["errors"].append("abstention_count_invalid")
    if not output["errors"] and all(row["status"] == "PASS_SHAPE_ONLY_NOT_GOLD" for row in output["candidates"]):
        output["status"] = "PASS_SHAPE_ONLY_NOT_GOLD" if questions else "ABSTAINED"
    return output
