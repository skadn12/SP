"""Offline ordering validation and Qwen-only review bridge for frozen R02 sources.

No function calls a model or writes artifacts. Question/item text and labels are
never changed. Gold is the only answer field computed here, from source order.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE.parent / "h4_groundless_random_feedback_v1/generation/prompts.py"
_spec = importlib.util.spec_from_file_location("r02_ordering_review_prompts", PROMPTS_PATH)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)
MODEL = _prompts.MODEL
LABELS = _prompts.LABELS
QUESTION_FIELDS = ("candidate_id", "root_id", "question_type", "chronology_kind",
                   "question", "items", "gold", "item_evidence")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source(source: Any) -> tuple[list[str], dict[str, dict], dict[str, int]]:
    if not isinstance(source, dict) or not isinstance(source.get("turns"), list):
        return ["source_schema"], {}, {}
    turns, positions = {}, {}
    for pos, turn in enumerate(source["turns"]):
        if (not isinstance(turn, dict) or not all(_text(turn.get(k)) for k in ("turn_ref", "speaker", "text"))
                or turn["turn_ref"] in turns):
            return ["source_turn_schema_or_duplicate_ref"], {}, {}
        turns[turn["turn_ref"]] = turn
        positions[turn["turn_ref"]] = pos
    return ([] if turns else ["source_empty"]), turns, positions


def validate_generated_ordering(row: Any, expected_id: str, root: str,
                                source: dict, source_path: str | Path) -> tuple[list[str], dict | None]:
    """Validate source bindings, retaining raw labels and mechanically fixing Gold.

    This is structural checking, not a human semantic judgment. No particular
    round prefix or non-identity Gold pattern is required.
    """
    errors, turns, positions = _source(source)
    if not isinstance(row, dict):
        return sorted(set(errors + ["candidate_not_object"])), None
    if (row.get("candidate_id") != expected_id or row.get("root_id") != root
            or not isinstance(source, dict) or source.get("root_id") != root or row.get("question_type") != "ordering"
            or row.get("chronology_kind") != "conversation_order"):
        errors.append("candidate_identity_or_chronology")
    question = row.get("question")
    if not _text(question):
        errors.append("question_missing")
    else:
        if "APPEAR IN THE SUPPLIED DIALOGUE" not in question.upper():
            errors.append("question_conversation_scope_missing")
        if re.search(r"\bI[1-6]\b|\b(?:R\d+:)?S\d+:T\d+\b", question, re.I):
            errors.append("question_contains_label_or_source_position_hint")
    items, evidence, gold = row.get("items"), row.get("item_evidence"), row.get("gold")
    if not isinstance(items, dict) or set(items) != set(LABELS):
        errors.append("items_schema")
    elif not all(_text(v) for v in items.values()):
        errors.append("item_text_missing")
    elif len({" ".join(v.split()).casefold() for v in items.values()}) != len(LABELS):
        errors.append("duplicate_item_text")
    elif any(re.search(r"\bI[1-6]\b|\b(?:R\d+:)?S\d+:T\d+\b", v, re.I) for v in items.values()):
        errors.append("item_contains_label_or_source_position_hint")
    if not isinstance(gold, list) or len(gold) != 6 or any(not isinstance(v, str) for v in gold) or set(gold) != set(LABELS):
        errors.append("gold_not_complete_permutation")
    anchors = {}
    if not isinstance(evidence, dict) or set(evidence) != set(LABELS):
        errors.append("item_evidence_schema")
    else:
        for label in LABELS:
            entry = evidence[label]
            if not isinstance(entry, dict):
                errors.append(label + ":evidence_schema")
                continue
            if not _text(entry.get("reason")):
                errors.append(label + ":evidence_reason_missing")
            if entry.get("status") not in {"statement", "report", "decision", "advice", "offer", "promise", "action"}:
                errors.append(label + ":evidence_status_invalid")
            ref = entry.get("turn_ref")
            if not isinstance(ref, str) or ref not in turns:
                errors.append(label + ":anchor_not_in_source")
                continue
            anchors[label] = ref
            if entry.get("speaker") != turns[ref]["speaker"]:
                errors.append(label + ":speaker_mismatch")
            if not _text(entry.get("quote")) or entry["quote"] not in turns[ref]["text"]:
                errors.append(label + ":quote_not_exact")
        if len(anchors) == 6 and len(set(anchors.values())) != 6:
            errors.append("order_anchors_not_distinct")
    source_file = Path(source_path)
    try:
        source_bytes = source_file.read_bytes()
        if json.loads(source_bytes) != source:
            errors.append("source_file_binding_mismatch")
    except (OSError, ValueError):
        errors.append("source_file_unreadable")
    if errors:
        return sorted(set(errors)), None
    candidate = deepcopy(row)
    # JSON key order is a protocol detail; labels retain their original content.
    candidate["items"] = {label: deepcopy(items[label]) for label in LABELS}
    candidate["item_evidence"] = {label: deepcopy(evidence[label]) for label in LABELS}
    candidate["original_gold"] = deepcopy(row.get("original_gold", gold))
    candidate["gold"] = sorted(LABELS, key=lambda label: positions[anchors[label]])
    candidate["gold_recomputed_from_generated_anchor"] = True
    candidate["source_path"] = str(source_file.resolve())
    candidate["source_sha256"] = _sha(source_bytes)
    return [], candidate


def _prompt_candidate(candidate: dict) -> dict:
    return {key: deepcopy(candidate[key]) for key in QUESTION_FIELDS}


def build_review_prompt(candidate: dict, source: dict, feedback_suffix: str = "") -> str:
    """Reuse the latest frozen review instructions, one candidate per request."""
    return _prompts.build_ordering_review_prompt(candidate["root_id"], [_prompt_candidate(candidate)],
                                                 source["turns"], feedback_suffix)


def _review_object(raw: str, expected_ids: list[str]) -> dict:
    """The existing review schema, with type checks before set/index operations."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("review_json_invalid") from error
    if not isinstance(obj, dict) or set(obj) != {"reviews"} or not isinstance(obj["reviews"], list):
        raise ValueError("review_schema_top_level")
    if any(not isinstance(v, dict) for v in obj["reviews"]) or [v.get("candidate_id") for v in obj["reviews"]] != expected_ids:
        raise ValueError("review_candidate_order_or_count")
    fields = {"candidate_id", "items", "coherent", "unique_order", "alternative_orders", "neutral_stem_ok", "reason"}
    item_fields = {"supported", "order_anchor", "support_refs", "alternative_matching_turn_refs",
                   "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok", "reason"}
    for row in obj["reviews"]:
        if set(row) != fields or not _text(row["reason"]):
            raise ValueError("review_question_schema")
        if not isinstance(row["items"], dict) or set(row["items"]) != set(LABELS):
            raise ValueError("review_items_schema")
        for label in LABELS:
            item = row["items"][label]
            if not isinstance(item, dict) or set(item) != item_fields or not _text(item["reason"]):
                raise ValueError("review_item_schema")
            anchor = item["order_anchor"]
            if not isinstance(anchor, dict) or set(anchor) != {"turn_ref", "reason"} or not all(_text(anchor[k]) for k in anchor):
                raise ValueError("review_anchor_schema")
            if any(not isinstance(item[k], bool) for k in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok")):
                raise ValueError("review_boolean_schema")
            for field in ("support_refs", "alternative_matching_turn_refs"):
                if not isinstance(item[field], list) or not all(_text(ref) for ref in item[field]):
                    raise ValueError("review_" + field + "_schema")
        if not isinstance(row["alternative_orders"], list):
            raise ValueError("review_alternative_order_schema")
        for order in row["alternative_orders"]:
            if not isinstance(order, list) or len(order) != 6 or not all(isinstance(v, str) for v in order) or set(order) != set(LABELS):
                raise ValueError("review_alternative_order_schema")
        if any(not isinstance(row[field], bool) for field in ("coherent", "unique_order", "neutral_stem_ok")):
            raise ValueError("review_question_boolean_schema")
    return obj


def _proof(raw: str, candidates: list[dict], source: dict, response_path: Path,
           request_path: Path, metadata: dict) -> tuple[list[str], dict]:
    errors, proof = [], {"requested_model": MODEL, "returned_model": metadata.get("returned_model"),
                        "request_path": str(request_path.resolve()), "response_path": str(response_path.resolve()),
                        "raw_response_sha256": _sha(raw.encode("utf-8"))}
    try:
        request_bytes, response_bytes = request_path.read_bytes(), response_path.read_bytes()
        request, response = json.loads(request_bytes), json.loads(response_bytes)
        proof.update(request_sha256=_sha(canonical(request)), request_file_sha256=_sha(request_bytes),
                     response_sha256=_sha(response_bytes))
        if request.get("model") != MODEL:
            errors.append("review_request_model_mismatch")
        messages = request.get("messages")
        if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
            errors.append("review_request_messages_invalid")
        else:
            prompt = messages[0]["content"]
            data_text = prompt.split("\nDATA:\n", 1)[1]
            data, end = json.JSONDecoder().raw_decode(data_text)
            suffix = data_text[end:].strip()
            if suffix:
                feedback = re.fullmatch(r"\[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4\]\s*(.*?)\s*\[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4\]", suffix, re.S)
                if feedback is None or not isinstance(json.loads(feedback.group(1)), dict):
                    errors.append("review_unrecognized_prompt_suffix")
            if (data != {"root_id": source["root_id"], "candidates": [_prompt_candidate(c) for c in candidates],
                         "source_turns": source["turns"]}
                    or not prompt.startswith(_prompts.ORDERING_REVIEW_INSTRUCTIONS)):
                errors.append("review_prompt_source_or_candidate_mismatch")
            proof["prompt_sha256"] = _sha(prompt.encode("utf-8"))
        if response.get("text", response.get("raw_text")) != raw:
            errors.append("review_response_raw_binding_mismatch")
        stored_metadata = response.get("metadata", response)
        for key in ("returned_model", "requested_model", "finish_reason"):
            if key in metadata and key in stored_metadata and metadata[key] != stored_metadata[key]:
                errors.append("review_metadata_binding_mismatch:" + key)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        errors.append("review_proof_unreadable_or_invalid")
    envelope = metadata.get("raw_response") or {}
    returned = metadata.get("returned_model") or envelope.get("model")
    proof["returned_model"] = returned
    proof["response_id"] = envelope.get("id", metadata.get("response_id"))
    if returned != MODEL or metadata.get("requested_model", MODEL) != MODEL:
        errors.append("review_response_model_mismatch")
    finish = metadata.get("finish_reason")
    choices = envelope.get("choices")
    if choices is not None:
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            errors.append("review_response_choices_invalid")
        else:
            finish = choices[0].get("finish_reason")
            if choices[0].get("message", {}).get("content") != raw:
                errors.append("review_envelope_content_mismatch")
        if envelope.get("model") != MODEL:
            errors.append("review_envelope_model_mismatch")
    if finish != "stop":
        errors.append("review_incomplete_response")
    if metadata.get("http_status", 200) != 200:
        errors.append("review_http_status_invalid")
    return sorted(set(errors)), proof


def review_response(raw: str, candidates: list[dict], source: dict,
                    response_path: str | Path, request_path: str | Path,
                    metadata: dict | None = None) -> dict:
    """Return accepted/rejected rows and a provenance artifact, without writes."""
    metadata = metadata or {}
    errors, proof = _proof(raw, candidates, source, Path(response_path), Path(request_path), metadata)
    source_errors, turns, positions = _source(source)
    errors += source_errors
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)) or not ids:
        errors.append("review_candidate_ids_empty_or_duplicate")
    parsed = {"reviews": []}
    try:
        parsed = _review_object(raw, ids)
    except ValueError as error:
        errors.append(str(error))
    accepted, rejected, checks = [], [], []
    for candidate, verdict in zip(candidates, parsed["reviews"] if not errors else [None] * len(candidates)):
        reasons = list(errors)
        anchors, alternatives = {}, {}
        if verdict is not None:
            candidate_errors, _ = validate_generated_ordering(candidate, candidate["candidate_id"], source["root_id"],
                                                               source, candidate.get("source_path", ""))
            reasons.extend(candidate_errors)
            if not candidate_errors:
                if candidate.get("source_sha256") != _sha(Path(candidate["source_path"]).read_bytes()):
                    reasons.append("candidate_source_sha256_mismatch")
            for label in LABELS:
                item = verdict["items"][label]
                ref = item["order_anchor"]["turn_ref"]
                anchors[label] = ref
                alternatives[label] = sorted(set([ref] + item["alternative_matching_turn_refs"]), key=lambda v: positions.get(v, -1))
                if ref not in positions:
                    reasons.append(label + ":anchor_not_in_source")
                if not all(item[k] for k in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok")):
                    reasons.append(label + ":qwen_item_check_failed")
                if any(ref2 not in turns for ref2 in item["support_refs"] + item["alternative_matching_turn_refs"]):
                    reasons.append(label + ":context_ref_not_in_source")
            if len(set(anchors.values())) != 6:
                reasons.append("order_anchors_not_distinct")
            if not all(verdict[k] for k in ("coherent", "unique_order", "neutral_stem_ok")):
                reasons.append("qwen_question_check_failed")
            all_refs_valid = all(ref in positions for refs in alternatives.values() for ref in refs)
            computed = sorted(LABELS, key=lambda label: positions[anchors[label]]) if all_refs_valid else []
            if computed:
                if any(order != computed for order in verdict["alternative_orders"]):
                    reasons.append("alternative_order_changes_gold")
                # Checking every adjacent pair bounds all simultaneous choices;
                # checking alternatives individually would miss mutual crossings.
                for before, after in zip(computed, computed[1:]):
                    if max(positions[r] for r in alternatives[before]) >= min(positions[r] for r in alternatives[after]):
                        reasons.append("alternative_anchor_changes_or_ties_order:" + before + ":" + after)
        checks.append({"candidate_id": candidate["candidate_id"], "reasons": sorted(set(reasons)),
                       "verified_order_anchors": anchors, "allowed_matching_turn_refs": alternatives})
        if reasons:
            rejected.append({"candidate_id": candidate["candidate_id"], "root_id": candidate["root_id"],
                             "reasons": sorted(set(reasons))})
            continue
        row = deepcopy(candidate)
        row["original_gold"] = deepcopy(candidate.get("original_gold", candidate["gold"]))
        row["gold"] = computed
        row["gold_recomputed_from_order_anchor"] = True
        row["order_anchor"] = anchors
        row["support_refs"] = {label: verdict["items"][label]["support_refs"] for label in LABELS}
        row["alternative_matching_turn_refs"] = {label: verdict["items"][label]["alternative_matching_turn_refs"] for label in LABELS}
        row["qwen_review"] = {"status": "PASS", **proof}
        row["provenance"] = {**row.get("provenance", {}), "codex_question_edits": False,
                              "codex_semantic_review_used": False, "review_owner_model": MODEL,
                              "source_sha256": row["source_sha256"]}
        row["status"] = "QWEN_REVIEWED_MECHANICALLY_VERIFIED_READY_FOR_SOLVER"
        accepted.append(row)
    artifact = {"status": "QWEN_REVIEW_SCHEMA_OR_BINDING_REJECTED" if errors else "QWEN_REVIEW_PARSED",
                "reviewer_model": MODEL, **proof, "candidate_ids": ids, "errors": sorted(set(errors)),
                "reviews": parsed["reviews"], "checks": checks, "accepted_count": len(accepted),
                "rejected_count": len(rejected), "codex_semantic_review_used": False,
                "historical_deepseek_glm_verdicts_used": False}
    return {"accepted": accepted, "rejected": rejected, "artifact": artifact}
