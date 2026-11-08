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

import prompt_layout
import reference_compat
from typing import Any


HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE.parent / "h4_feedback_training_pool_v15_qa/prompts.py"
_spec = importlib.util.spec_from_file_location("r02_ordering_review_prompts", PROMPTS_PATH)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)
MODEL = _prompts.MODEL
LABELS = _prompts.LABELS
ORDERING_GENERATION_INSTRUCTIONS = _prompts.ORDERING_GENERATION_INSTRUCTIONS
QUESTION_FIELDS = ("candidate_id", "root_id", "question_type", "chronology_kind",
                   "question", "items")
ANNOTATION_CHAR_BUDGET = 160
SCOPE_CONTRACT = "explicit_public_stem_scene_scope_v52"
SCOPE_START = "[ORDERING_MATCH_SCOPE_V52]"
SCOPE_END = "[/ORDERING_MATCH_SCOPE_V52]"


def ordering_match_scope(candidate: dict, source: dict) -> dict:
    """Resolve only an explicit terminal scene restriction in the public stem.

    No writer evidence, Gold, slot focus or topic is consulted. Unrecognized
    prose never causes implicit narrowing; every source turn remains available.
    Multi-scene questions use the union, in original transcript order.
    """
    turns = source["turns"]
    scenes = list(dict.fromkeys(turn["turn_ref"].rsplit(":", 1)[0] for turn in turns))
    question = candidate["question"]
    match = re.search(r"\bAPPEAR IN THE SUPPLIED DIALOGUE "
                      r"(?P<restriction>within the (?P<scenes>.+?) scenes?\.?)\s*$", question, re.I)
    selected, fragment = [], None
    if match:
        entries = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", match.group("scenes"))
        # Exact source IDs are a reproducible public boundary; guesses are not.
        if entries and all(entry in scenes for entry in entries):
            selected = [scene for scene in scenes if scene in entries]
            fragment = match.group("restriction")
    refs = [turn["turn_ref"] for turn in turns
            if not selected or turn["turn_ref"].rsplit(":", 1)[0] in selected]
    return {"contract": SCOPE_CONTRACT, "candidate_id": candidate["candidate_id"],
            "kind": "explicit_scene_union" if selected else "full_dialogue",
            "public_stem_restriction": fragment, "scene_refs": selected,
            "matching_turn_refs": refs, "context_turn_refs": [turn["turn_ref"] for turn in turns]}


def _scope_review_instruction(candidate: dict, source: dict) -> str:
    scope = ordering_match_scope(candidate, source)
    if scope["kind"] == "full_dialogue":
        return ""
    return ("\n" + SCOPE_START + "\n" + canonical(scope).decode("utf-8") + "\n" + SCOPE_END +
            "\nThe public stem explicitly restricts matching appearances to matching_turn_refs. "
            "Locate primary order_anchor and alternative_matching_anchors only in that union. "
            "Other scenes remain in the complete source for support_refs and referent resolution; "
            "a similar statement there is not an alternative occurrence for this question. "
            "Within the stated scope, list every plausible matching occurrence, including "
            "same-turn alternatives. Do not silently choose the first or preferred occurrence. "
            "Judge unique_order and alternative_orders using that same scope. "
            "Preserve the existing response schema and independently verify every item.\n")


def _annotation_diagnostic(value: Any, field: str) -> list[dict]:
    if not _text(value):
        return [{"field": field, "code": "annotation_missing_or_not_text", "value": value}]
    if len(value) > ANNOTATION_CHAR_BUDGET:
        return [{"field": field, "code": "annotation_longer_than_prompt_budget",
                 "characters": len(value), "value": value}]
    return []


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


def _resolve_anchor(entry: Any, turns: dict, positions: dict) -> tuple[list[str], dict | None]:
    """Locate an exact, unique Unicode string slice; never choose a repeated hit."""
    if not isinstance(entry, dict):
        return ["anchor_schema"], None
    ref, quote = entry.get("turn_ref"), entry.get("quote")
    if not isinstance(ref, str) or ref not in turns:
        return ["anchor_not_in_source"], None
    errors = []
    if "speaker" in entry and entry["speaker"] != turns[ref]["speaker"]:
        errors.append("speaker_mismatch")
    if not _text(quote):
        return errors + ["quote_not_exact"], None
    text = turns[ref]["text"]
    start = text.find(quote)
    if start < 0:
        return errors + ["quote_not_exact"], None
    # find(..., start + 1) catches overlapping occurrences as well.
    if text.find(quote, start + 1) >= 0:
        return errors + ["quote_not_unique_in_turn"], None
    resolved = {"turn_ref": ref, "speaker": turns[ref]["speaker"], "quote": quote,
                "turn_index": positions[ref], "char_start": start, "char_end": start + len(quote)}
    for key in ("turn_index", "char_start", "char_end"):
        if key in entry and (type(entry[key]) is not int or entry[key] != resolved[key]):
            errors.append("anchor_" + key + "_mismatch")
    return errors, resolved


def _anchor_key(anchor: dict) -> tuple[int, int, int]:
    return anchor["turn_index"], anchor["char_start"], anchor["char_end"]


def _strictly_before(before: dict, after: dict) -> bool:
    return (before["turn_index"] < after["turn_index"] or
            (before["turn_index"] == after["turn_index"] and before["char_end"] <= after["char_start"]))


def resolve_ordering_anchors(anchors: Any, source: dict) -> tuple[list[str], dict, list[str]]:
    """Shared structural contract for generation, review and Solver admission.

    Return resolved spans and a Gold permutation only when all six exact
    clauses are uniquely locatable and pairwise non-overlapping. No labels
    break ties and no semantic rewriting or inference occurs here.
    """
    errors, turns, positions = _source(source)
    resolved = {}
    if not isinstance(anchors, dict) or set(anchors) != set(LABELS):
        return sorted(set(errors + ["order_anchor_schema"])), resolved, []
    for label in LABELS:
        problems, anchor = _resolve_anchor(anchors[label], turns, positions)
        errors.extend(label + ":" + problem for problem in problems)
        if anchor is not None:
            resolved[label] = anchor
    if errors or len(resolved) != len(LABELS):
        return sorted(set(errors)), resolved, []
    ordered = sorted(LABELS, key=lambda label: _anchor_key(resolved[label]))
    for before, after in zip(ordered, ordered[1:]):
        if not _strictly_before(resolved[before], resolved[after]):
            errors.append("order_anchor_spans_overlap_or_tie:" + before + ":" + after)
    return sorted(set(errors)), resolved, ([] if errors else ordered)


def validate_generated_ordering(row: Any, expected_id: str, root: str,
                                source: dict, source_path: str | Path) -> tuple[list[str], dict | None]:
    """Validate source bindings, retaining raw labels and mechanically fixing Gold.

    This is structural checking, not a human semantic judgment. No particular
    round prefix or non-identity Gold pattern is required.
    """
    raw_row = deepcopy(row)
    row, ref_normalizations = reference_compat.normalize_question(row, root, source)
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
    annotation_diagnostics = _annotation_diagnostic(row.get("order_reason"), "order_reason")
    computed = []
    if not isinstance(evidence, dict) or set(evidence) != set(LABELS):
        errors.append("item_evidence_schema")
    else:
        for label in LABELS:
            entry = evidence[label]
            if not isinstance(entry, dict):
                errors.append(label + ":evidence_schema")
                continue
            annotation_diagnostics.extend(_annotation_diagnostic(entry.get("reason"), label + ".reason"))
            if (not isinstance(entry.get("status"), str)
                    or entry["status"] not in {"statement", "report", "decision", "advice", "offer", "promise", "action"}):
                annotation_diagnostics.append({"field": label + ".status", "code": "status_outside_suggested_vocabulary",
                                               "value": entry.get("status")})
            if not _text(entry.get("speaker")):
                errors.append(label + ":speaker_missing")
        anchor_errors, anchors, computed = resolve_ordering_anchors(evidence, source)
        errors.extend(anchor_errors)
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
    if ref_normalizations:
        candidate['raw_model_candidate'] = raw_row
        candidate['reference_normalizations'] = ref_normalizations
    # JSON key order is a protocol detail; labels retain their original content.
    candidate["items"] = {label: deepcopy(items[label]) for label in LABELS}
    candidate["item_evidence"] = {label: deepcopy(evidence[label]) for label in LABELS}
    candidate["original_gold"] = deepcopy(row.get("original_gold", gold))
    candidate["gold"] = computed
    candidate["generated_order_anchor_spans"] = anchors
    candidate["ordering_anchor_contract"] = "exact_clause_span_v1"
    candidate["generation_annotation_diagnostics"] = annotation_diagnostics
    candidate["gold_recomputed_from_generated_anchor"] = True
    candidate["source_path"] = str(source_file.resolve())
    candidate["source_sha256"] = _sha(source_bytes)
    return [], candidate


def _prompt_candidate(candidate: dict) -> dict:
    # Independent review sees only the public claims and source transcript.
    # The writer's Gold, evidence, offsets and prior judgments must not anchor it.
    return {key: deepcopy(candidate[key]) for key in QUESTION_FIELDS}


def build_review_prompt(candidate: dict, source: dict, feedback_suffix: str = "") -> str:
    """Retain full source and add a replayable public-stem matching boundary."""
    prompt = _prompts.build_ordering_review_prompt(candidate["root_id"], [_prompt_candidate(candidate)],
                                                  source["turns"], feedback_suffix)
    prefix, marker, data = prompt.partition("\nDATA:\n")
    return prefix + _scope_review_instruction(candidate, source) + marker + data


def _review_object(raw: str, expected_ids: list[str], *, normalizations: list | None = None,
                   annotation_diagnostics: list | None = None) -> dict:
    """The existing review schema, with type checks before set/index operations."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key:" + key)
            result[key] = value
        return result

    try:
        obj = json.loads(raw, object_pairs_hook=unique_keys)
    except (TypeError, ValueError) as error:
        raise ValueError("review_json_invalid") from error
    fields = {"candidate_id", "items", "coherent", "unique_order", "alternative_orders", "neutral_stem_ok"}
    if (isinstance(obj, dict) and len(expected_ids) == 1
            and obj.get("candidate_id") == expected_ids[0]
            and fields <= set(obj) <= fields | {"reason"}):
        obj = {"reviews": [obj]}
        if normalizations is not None:
            normalizations.append({"kind": "SINGLE_REVIEW_OBJECT_WRAPPED", "candidate_id": expected_ids[0],
                                   "semantic_fields_changed": False, "raw_response_unchanged": True})
    if (isinstance(obj, dict) and set(obj) == {'reviews'} and isinstance(obj['reviews'], dict)
            and len(expected_ids) == 1 and obj['reviews'].get('candidate_id') == expected_ids[0]
            and fields <= set(obj['reviews']) <= fields | {'reason'}):
        obj = {'reviews': [obj['reviews']]}
        if normalizations is not None:
            normalizations.append({'kind': 'NESTED_SINGLE_REVIEW_OBJECT_WRAPPED',
                'candidate_id': expected_ids[0], 'semantic_fields_changed': False, 'raw_response_unchanged': True})
    if not isinstance(obj, dict) or set(obj) != {"reviews"} or not isinstance(obj["reviews"], list):
        raise ValueError("review_schema_top_level")
    if any(not isinstance(v, dict) for v in obj["reviews"]) or [v.get("candidate_id") for v in obj["reviews"]] != expected_ids:
        raise ValueError("review_candidate_order_or_count")
    item_fields = {"supported", "order_anchor", "support_refs", "alternative_matching_anchors",
                   "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok"}
    annotation_diagnostics = annotation_diagnostics if annotation_diagnostics is not None else []
    for row in obj["reviews"]:
        if not fields <= set(row) <= fields | {"reason"}:
            raise ValueError("review_question_schema")
        annotation_diagnostics.extend(_annotation_diagnostic(row.get("reason"), row["candidate_id"] + ".reason"))
        if not isinstance(row["items"], dict) or set(row["items"]) != set(LABELS):
            raise ValueError("review_items_schema")
        for label in LABELS:
            item = row["items"][label]
            if not isinstance(item, dict) or not item_fields <= set(item) <= item_fields | {"reason"}:
                raise ValueError("review_item_schema")
            annotation_diagnostics.extend(_annotation_diagnostic(item.get("reason"), label + ".reason"))
            anchor = item["order_anchor"]
            if (not isinstance(anchor, dict) or not {"turn_ref", "quote"} <= set(anchor) <= {"turn_ref", "quote", "reason"}
                    or not all(_text(anchor[k]) for k in ("turn_ref", "quote"))):
                raise ValueError("review_anchor_schema")
            annotation_diagnostics.extend(_annotation_diagnostic(anchor.get("reason"), label + ".order_anchor.reason"))
            if any(not isinstance(item[k], bool) for k in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok")):
                raise ValueError("review_boolean_schema")
            if not isinstance(item["support_refs"], list) or not all(_text(ref) for ref in item["support_refs"]):
                raise ValueError("review_support_refs_schema")
            if (not isinstance(item["alternative_matching_anchors"], list)
                    or any(not isinstance(anchor, dict) or not {"turn_ref", "quote"} <= set(anchor) <= {"turn_ref", "quote", "reason"}
                           or not all(_text(anchor[key]) for key in ("turn_ref", "quote"))
                           for anchor in item["alternative_matching_anchors"])):
                raise ValueError("review_alternative_matching_anchors_schema")
            for index, anchor in enumerate(item["alternative_matching_anchors"]):
                annotation_diagnostics.extend(_annotation_diagnostic(anchor.get("reason"), label + ".alternative_matching_anchors." + str(index) + ".reason"))
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
            try:
                prompt_layout.review_feedback_suffix(suffix)
            except (ValueError, TypeError):
                errors.append("review_unrecognized_prompt_suffix")
            if (data != {"root_id": source["root_id"], "candidates": [_prompt_candidate(c) for c in candidates],
                         "source_turns": source["turns"]}
                    or not prompt.startswith(_prompts.ORDERING_REVIEW_INSTRUCTIONS)):
                errors.append("review_prompt_source_or_candidate_mismatch")
            if len(candidates) == 1:
                expected_prefix = build_review_prompt(candidates[0], source).partition("\nDATA:\n")[0]
                if prompt.partition("\nDATA:\n")[0] != expected_prefix:
                    errors.append("review_prompt_match_scope_changed")
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
    normalizations, annotation_diagnostics = [], []
    try:
        parsed = _review_object(raw, ids, normalizations=normalizations,
                                annotation_diagnostics=annotation_diagnostics)
        for verdict in parsed['reviews']:
            for label, item in verdict['items'].items():
                for field in ('order_anchor', 'alternative_matching_anchors'):
                    entries = [item[field]] if field == 'order_anchor' else item[field]
                    fixed = []
                    for index, entry in enumerate(entries):
                        normalized, changes = reference_compat.normalize_entry(entry, source['root_id'], source)
                        fixed.append(normalized)
                        normalizations.extend({**change, 'candidate_id': verdict['candidate_id'],
                            'field': f'items.{label}.{field}[{index}]'} for change in changes)
                    item[field] = fixed[0] if field == 'order_anchor' else fixed
    except ValueError as error:
        errors.append(str(error))
    accepted, rejected, checks = [], [], []
    for candidate, verdict in zip(candidates, parsed["reviews"] if not errors else [None] * len(candidates)):
        reasons = list(errors)
        anchors, alternatives, computed, outside_scope = {}, {}, [], {}
        match_scope = ordering_match_scope(candidate, source)
        matching_refs = set(match_scope["matching_turn_refs"])
        if verdict is not None:
            candidate_errors, _ = validate_generated_ordering(candidate, candidate["candidate_id"], source["root_id"],
                                                               source, candidate.get("source_path", ""))
            reasons.extend(candidate_errors)
            if not candidate_errors:
                if candidate.get("source_sha256") != _sha(Path(candidate["source_path"]).read_bytes()):
                    reasons.append("candidate_source_sha256_mismatch")
            anchor_errors, anchors, computed = resolve_ordering_anchors(
                {label: verdict["items"][label]["order_anchor"] for label in LABELS}, source)
            reasons.extend(anchor_errors)
            for label in LABELS:
                item = verdict["items"][label]
                outside_scope[label] = []
                if label in anchors and anchors[label]["turn_ref"] not in matching_refs:
                    reasons.append(label + ":primary_anchor_outside_public_stem_scope")
                alternatives[label] = [anchors[label]] if label in anchors else []
                for index, entry in enumerate(item["alternative_matching_anchors"]):
                    alt_errors, anchor = _resolve_anchor(entry, turns, positions)
                    reasons.extend(label + ":alternative_" + str(index) + ":" + error for error in alt_errors)
                    if anchor is not None and anchor["turn_ref"] not in matching_refs:
                        outside_scope[label].append(anchor)
                    elif anchor is not None and anchor not in alternatives[label]:
                        alternatives[label].append(anchor)
                alternatives[label].sort(key=_anchor_key)
                if not all(item[k] for k in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok")):
                    reasons.append(label + ":qwen_item_check_failed")
                if any(ref2 not in turns for ref2 in item["support_refs"]):
                    reasons.append(label + ":context_ref_not_in_source")
            if not all(verdict[k] for k in ("coherent", "unique_order", "neutral_stem_ok")):
                reasons.append("qwen_question_check_failed")
            if computed:
                if any(order != computed for order in verdict["alternative_orders"]):
                    reasons.append("alternative_order_changes_gold")
                # Checking every adjacent pair bounds all simultaneous choices;
                # checking alternatives individually would miss mutual crossings.
                for before, after in zip(computed, computed[1:]):
                    if any(not _strictly_before(left, right)
                           for left in alternatives[before] for right in alternatives[after]):
                        reasons.append("alternative_anchor_changes_or_ties_order:" + before + ":" + after)
        checks.append({"candidate_id": candidate["candidate_id"], "reasons": sorted(set(reasons)),
                       "ordering_match_scope": match_scope,
                       "out_of_scope_alternative_anchor_spans": outside_scope,
                       "verified_order_anchor_spans": anchors, "allowed_matching_anchor_spans": alternatives})
        if reasons:
            rejected.append({"candidate_id": candidate["candidate_id"], "root_id": candidate["root_id"],
                             "reasons": sorted(set(reasons))})
            continue
        row = deepcopy(candidate)
        row["original_gold"] = deepcopy(candidate.get("original_gold", candidate["gold"]))
        row["gold"] = computed
        row["gold_recomputed_from_order_anchor"] = True
        row["order_anchor"] = {label: anchor["turn_ref"] for label, anchor in anchors.items()}
        row["order_anchor_spans"] = anchors
        row["ordering_anchor_contract"] = "exact_clause_span_v1"
        row["support_refs"] = {label: verdict["items"][label]["support_refs"] for label in LABELS}
        # Keep the raw reviewer response and all findings in the artifact above.
        # Only valid matches to the public question enter downstream ordering;
        # an out-of-scope occurrence is context, not an ambiguity for this stem.
        # Reviewer booleans and alternative_orders are never overridden.
        row["alternative_matching_anchors"] = {
            label: [deepcopy(entry) for entry in verdict["items"][label]["alternative_matching_anchors"]
                    if entry["turn_ref"] in matching_refs] for label in LABELS}
        row["allowed_matching_anchor_spans"] = alternatives
        row["qwen_review"] = {"status": "PASS", **proof}
        row["provenance"] = {**row.get("provenance", {}), "codex_question_edits": False,
                              "codex_semantic_review_used": False, "review_owner_model": MODEL,
                              "source_sha256": row["source_sha256"]}
        row["status"] = "QWEN_REVIEWED_MECHANICALLY_VERIFIED_READY_FOR_SOLVER"
        accepted.append(row)
    artifact = {"status": "QWEN_REVIEW_SCHEMA_OR_BINDING_REJECTED" if errors else "QWEN_REVIEW_PARSED",
                "reviewer_model": MODEL, **proof, "candidate_ids": ids, "errors": sorted(set(errors)),
                "reviews": parsed["reviews"], "normalizations": normalizations,
                "annotation_diagnostics": annotation_diagnostics,
                "checks": checks, "accepted_count": len(accepted),
                "rejected_count": len(rejected), "codex_semantic_review_used": False,
                "historical_deepseek_glm_verdicts_used": False}
    return {"accepted": accepted, "rejected": rejected, "artifact": artifact}
