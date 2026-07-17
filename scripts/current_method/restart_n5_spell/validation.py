from __future__ import annotations

from collections import Counter
from typing import Any

from .core import LETTERS, normalize_option
from .prompts import QUESTIONER_GENERATED_ITEM_SCHEMA, REALIZER_OUTPUT_SCHEMA
from .questioner_contract import (
    QUESTIONER_SCHEMA_VERSION, REASONING_CONTRACT_VERSION, SPAN_CATALOG_VERSION,
    STRUCTURAL_SUPPORT_ROLES, contract_is_satisfied, derive_local_difficulty,
    single_reasoning_contract_sha256, span_catalog_index, span_catalog_sha256,
)


EVIDENCE_VALUE_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "be", "been", "being", "of", "to", "in",
    "on", "at", "for", "from", "with", "by", "and", "or", "during", "session", "person",
    "event", "this", "that", "it", "they", "them", "he", "she", "his", "her", "their",
})
DEFAULT_DURABLE_PREDICATES = frozenset({
    "status", "decision", "location", "commitment", "relationship", "ownership", "outcome",
})


def normalize_evidence_value(value: Any) -> str:
    import re
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE))


def validate_minimal_evidence_value(value: Any) -> list[str]:
    normalized = normalize_evidence_value(value)
    tokens = normalized.split()
    if (not normalized or len(normalized.replace(" ", "")) < 3 or
            not any(token not in EVIDENCE_VALUE_STOPWORDS and len(token) >= 3 for token in tokens)):
        return ["bad_generic_evidence_value"]
    return []


def _normalized_phrase_in_text(value: Any, text: Any) -> bool:
    needle = normalize_evidence_value(value)
    haystack = normalize_evidence_value(text)
    return bool(needle) and f" {needle} " in f" {haystack} "


def validate_answer_evidence_values(values: Any, evidence_turn_ids: list[str],
                                    frozen_turns: dict[str, str], correct_answer_text: str,
                                    support_type: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(values, list) or not values:
        return ["bad_answer_evidence_binding"]
    seen_values: list[str] = []
    covered_turns: set[str] = set()
    answer_value_count = 0
    answer_normalized = normalize_evidence_value(correct_answer_text)
    for item in values:
        if (not isinstance(item, dict) or
                set(item) != {"turn_id", "value", "exact_quote", "role"}):
            errors.append("bad_answer_evidence_binding"); continue
        turn_id, value, exact, role = (item.get("turn_id"), item.get("value"),
                                       item.get("exact_quote"), item.get("role"))
        normalized = normalize_evidence_value(value)
        errors.extend(validate_minimal_evidence_value(value))
        if (turn_id not in evidence_turn_ids or not isinstance(exact, str) or not exact or
                exact not in frozen_turns.get(str(turn_id), "") or
                not _normalized_phrase_in_text(value, exact)):
            errors.append("bad_answer_evidence_binding")
        if role not in STRUCTURAL_SUPPORT_ROLES:
            errors.append("bad_answer_evidence_binding")
        elif role == "answer_value":
            answer_value_count += 1
            if not _normalized_phrase_in_text(value, answer_normalized):
                errors.append("bad_answer_evidence_binding")
        seen_values.append(normalized); covered_turns.add(str(turn_id))
    if set(evidence_turn_ids) != covered_turns or answer_value_count < 1:
        errors.append("bad_answer_evidence_binding")
    if len(seen_values) != len(set(seen_values)):
        errors.append("bad_duplicate_information_point")
    if support_type == "multi_hop" and len(set(seen_values)) < 2:
        errors.append("bad_duplicate_information_point")
    return sorted(set(errors))


def validate_config(c: dict[str, Any]) -> None:
    expected = {"qa_format": "mcq", "max_verifier_generations": 0,
        "max_parse_retries_per_target": 1, "no_context_initial_valid_k": 3,
        "no_context_boundary_extra_valid_k": 2,
        "no_context_bad_ground_threshold_after_extension": 3,
        "full_context_valid_k": 5, "permutation_valid_k_per_order": 5,
        "solver_effective_max_model_len": 16384,
        "minimum_solver_eligible_candidates":1,
        "allow_local_difficulty_downgrade":False}
    mode = c.get("mode")
    if mode == "live_questioner_smoke":
        expected.update({"questioner_generations":1, "questioner_mcqs_per_generation":3,
            "questioner_total_candidates":3, "max_questioner_generations":1,
            "max_realizer_generations":0, "max_total_paid_api_generations":1})
    elif mode == "live_smoke_realizer":
        expected.update({"max_total_paid_api_generations":1, "max_realizer_generations":1,
                         "max_questioner_generations":0, "max_session_attempts_per_session":1})
    elif mode == "live_smoke_realizer_continuity":
        expected.update({"max_total_paid_api_generations":2, "max_realizer_generations":2,
                         "max_questioner_generations":0, "max_session_attempts_per_session":1})
    elif mode == "mock_e2e" or mode in (None, "dry_run"):
        expected.update({"questioner_generations":4, "questioner_mcqs_per_generation":3,
                         "questioner_total_candidates":12})
    if str(mode or "").startswith("local_solver"):
        expected.update({"max_total_paid_api_generations":0, "max_realizer_generations":0,
                         "max_questioner_generations":0, "solver_effective_max_model_len":16384})
    bad = {k: (c.get(k), v) for k, v in expected.items() if c.get(k) != v}
    if bad:
        raise ValueError(f"unsafe canary config: {bad}")


def validate_qa(qa: dict[str, Any], turn_ids: set[str],
                frozen_turns: dict[str, str] | None = None,
                turn_session_ids: dict[str, str] | None = None) -> list[str]:
    errors = []
    options = qa.get("options") or {}
    if tuple(options.keys()) != LETTERS or any(not str(options.get(x, "")).strip() for x in LETTERS):
        errors.append("bad_option_schema")
    elif len({normalize_option(v) for v in options.values()}) != 4:
        errors.append("bad_option_schema")
    letter = qa.get("reference_answer_letter")
    if letter not in LETTERS or options.get(letter) != qa.get("reference_answer_text"):
        errors.append("bad_reference_mapping")
        if qa.get("reference_answer_text") not in options.values():
            errors.append("bad_missing_correct_option")
    evidence = qa.get("evidence_turn_ids") or []
    if (not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence)) or
            any(x not in turn_ids for x in evidence)):
        errors.append("bad_evidence")
    declared_sessions = qa.get("evidence_session_ids") or []
    if turn_session_ids is not None and evidence and all(x in turn_session_ids for x in evidence):
        actual_sessions = {turn_session_ids[x] for x in evidence}
        if set(declared_sessions) != actual_sessions:
            errors.append("bad_evidence_session_mapping")
    distractors = qa.get("distractor_explanations") or {}
    if any(not str(distractors.get(x, "")).strip() for x in LETTERS if x != letter):
        errors.append("bad_format")
    for field in ("question", "correct_answer_explanation", "reasoning_steps", "reasoning_type", "difficulty_level"):
        if not qa.get(field):
            errors.append("bad_format")
            break
    support_type = qa.get("answer_support_type")
    structured_values = qa.get("answer_evidence_values")
    quotes = ([{"turn_id":entry.get("turn_id"), "exact_quote":entry.get("exact_quote")}
               for entry in structured_values if isinstance(entry, dict)]
              if isinstance(structured_values, list) else qa.get("answer_supporting_quotes"))
    value = str(qa.get("answer_value_from_evidence") or "")
    conclusion = qa.get("final_reasoning_conclusion")
    if support_type not in ("direct_span", "multi_hop") or not isinstance(quotes, list) or not quotes:
        errors.append("bad_supporting_quote")
    else:
        quote_texts: list[str] = []
        supported_turn_ids: set[str] = set()
        for quote in quotes:
            turn_id = quote.get("turn_id") if isinstance(quote, dict) else None
            exact = quote.get("exact_quote") if isinstance(quote, dict) else None
            if (turn_id not in evidence or not isinstance(exact, str) or not exact or
                    frozen_turns is not None and exact not in frozen_turns.get(str(turn_id), "")):
                errors.append("bad_supporting_quote")
            else:
                quote_texts.append(exact)
                supported_turn_ids.add(str(turn_id))
        if isinstance(evidence, list) and supported_turn_ids != set(evidence):
            errors.append("bad_supporting_quote")
        if structured_values is not None:
            evidence_source = frozen_turns or {
                str(entry.get("turn_id")):str(entry.get("exact_quote"))
                for entry in structured_values if isinstance(entry, dict)}
            errors.extend(validate_answer_evidence_values(structured_values, evidence,
                evidence_source,
                str(qa.get("reference_answer_text") or ""), str(support_type)))
        else:
            errors.extend(validate_minimal_evidence_value(value))
            if not value:
                errors.append("bad_answer_value_mapping")
            elif not any(_normalized_phrase_in_text(value, quote)
                         for quote in quote_texts):
                errors.append("bad_answer_value_mapping")
            if value and not _normalized_phrase_in_text(value, qa.get("reference_answer_text") or ""):
                errors.append("bad_reference_evidence_consistency")
    if conclusion != qa.get("reference_answer_text"):
        errors.append("bad_final_reasoning_conclusion")
    if letter in LETTERS and isinstance(options, dict):
        other_options = [text for key, text in options.items() if key != letter]
        steps = qa.get("reasoning_steps") or []
        last_step = str(steps[-1]) if isinstance(steps, list) and steps else ""
        explanation = str(qa.get("correct_answer_explanation") or "")
        if any(text and (text in last_step or f"{text} is correct" in explanation)
               for text in other_options):
            errors.append("bad_reference_evidence_consistency")
        distractor_values = qa.get("distractor_values_from_evidence")
        distractor_quote_groups = qa.get("distractor_supporting_quotes")
        distractor_letters = [key for key in LETTERS if key != letter]
        if (not isinstance(distractor_values, list) or len(distractor_values) != 3 or
                not isinstance(distractor_quote_groups, list) or len(distractor_quote_groups) != 3):
            errors.append("bad_distractor_support")
        else:
            for option_letter, candidate_value, quote_group in zip(
                    distractor_letters, distractor_values, distractor_quote_groups):
                valid_quote = False
                errors.extend(validate_minimal_evidence_value(candidate_value))
                if isinstance(candidate_value, str) and candidate_value and \
                        _normalized_phrase_in_text(candidate_value, options.get(option_letter) or "") and \
                        isinstance(quote_group, list):
                    for quote in quote_group:
                        turn_id = quote.get("turn_id") if isinstance(quote, dict) else None
                        exact = quote.get("exact_quote") if isinstance(quote, dict) else None
                        if (turn_id in turn_ids and isinstance(exact, str) and exact and
                                _normalized_phrase_in_text(candidate_value, exact) and
                                (frozen_turns is None or exact in frozen_turns.get(str(turn_id), ""))):
                            valid_quote = True
                            break
                if not valid_quote:
                    errors.append("bad_distractor_support")
    bad_distractor_markers = ("fabricated", "not mentioned", "unrelated", "not supported",
                              "outside knowledge", "not cited")
    if isinstance(distractors, dict) and any(
            marker in str(distractors.get(key, "")).casefold()
            for key in LETTERS if key != letter for marker in bad_distractor_markers):
        errors.append("bad_distractor_grounding")
    steps = qa.get("reasoning_steps") or []
    actual_sessions = ({turn_session_ids[x] for x in evidence if x in turn_session_ids}
                       if turn_session_ids is not None and isinstance(evidence, list) else set(declared_sessions))
    difficulty = qa.get("difficulty_level")
    if difficulty == "L1" and (len(evidence) < 2 or len(steps) < 2):
        errors.append("bad_difficulty_evidence_contract")
    elif difficulty == "L2" and (len(evidence) < 2 or len(actual_sessions) < 2 or len(steps) < 2):
        errors.append("bad_difficulty_evidence_contract")
    elif difficulty == "L3":
        if len(evidence) < 3 or len(actual_sessions) < 2 or len(steps) < 3:
            errors.append("bad_difficulty_evidence_contract")
    return sorted(set(errors))


def materialize_questioner_item(item: Any, target_letter: str, *,
                                span_catalog: dict[str, Any] | None = None,
                                reasoning_contract: dict[str, Any] | None = None,
                                allow_local_difficulty_downgrade: bool = False,
                                ) -> tuple[dict[str, Any] | None, list[str]]:
    """Restore every stable QA field from a catalog and an immutable local contract."""
    if not isinstance(item, dict) or target_letter not in LETTERS or not isinstance(span_catalog, dict):
        return None, ["questioner_generated_schema_invalid"]
    errors: list[str] = []
    forbidden = {"options", "reference_answer_letter", "reference_answer_text",
        "canonical_answer_text", "exact_quote", "answer_supporting_quotes",
        "distractor_supporting_quotes", "evidence_turn_ids", "evidence_session_ids",
        "reasoning_type", "difficulty_level", "target_letter", "final_reasoning_conclusion",
        "answer_support_type", "distractor_values_from_evidence"}
    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, dict):
            return bool(forbidden & set(value)) or any(contains_forbidden(x) for x in value.values())
        if isinstance(value, list):
            return any(contains_forbidden(x) for x in value)
        return False
    if contains_forbidden(item):
        errors.append("model_supplied_local_binding_fields")
    if set(item) != set(QUESTIONER_GENERATED_ITEM_SCHEMA):
        errors.append("questioner_generated_schema_invalid")
    try:
        catalog = span_catalog_index(span_catalog)
    except ValueError:
        return None, ["span_catalog_invalid"]
    if (not isinstance(reasoning_contract, dict) or
            reasoning_contract.get("reasoning_contract_version") != REASONING_CONTRACT_VERSION or
            reasoning_contract.get("target_letter") != target_letter or
            reasoning_contract.get("item_index") not in (0, 1, 2)):
        errors.append("bad_reasoning_contract")
    correct = item.get("correct_answer_text")
    distractors = item.get("distractors")
    explanations = item.get("distractor_explanations")
    if (not isinstance(correct, str) or not correct.strip() or
            not isinstance(distractors, list) or len(distractors) != 3 or
            any(not isinstance(x, str) or not x.strip() for x in distractors) or
            len({normalize_option(x) for x in [correct, *distractors] if isinstance(x, str)}) != 4):
        errors.append("bad_unlabeled_options")
    if (not isinstance(explanations, list) or len(explanations) != 3 or
            any(not isinstance(x, str) or not x.strip() for x in explanations)):
        errors.append("bad_distractor_explanations")
    required_strings = ("qa_id", "question", "correct_answer_explanation")
    if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required_strings):
        errors.append("questioner_generated_schema_invalid")
    steps = item.get("reasoning_steps")
    if (not isinstance(steps, list) or not steps or
            any(not isinstance(x, str) or not x.strip() for x in steps)):
        errors.append("questioner_generated_schema_invalid")

    answer_span_ids = item.get("answer_support_span_ids")
    if (not isinstance(answer_span_ids, list) or not answer_span_ids or
            any(not isinstance(x, str) or not x for x in answer_span_ids)):
        errors.append("questioner_generated_schema_invalid")
        answer_span_ids = []
    if len(answer_span_ids) != len(set(answer_span_ids)):
        errors.append("bad_duplicate_information_point")
    if any(span_id not in catalog for span_id in answer_span_ids):
        errors.append("bad_span_id")

    evidence_values = item.get("answer_evidence_values")
    if (not isinstance(evidence_values, list) or not evidence_values or any(
            not isinstance(entry, dict) or
            set(entry) != {"span_id", "value"} or
            any(not isinstance(entry.get(key), str) or not entry[key].strip()
                for key in ("span_id", "value"))
            for entry in evidence_values or [])):
        errors.append("questioner_generated_schema_invalid")
    normalized_values: list[str] = []
    covered_answer_spans: set[str] = set()
    for entry in evidence_values if isinstance(evidence_values, list) else []:
        if not isinstance(entry, dict):
            continue
        span_id, value = entry.get("span_id"), entry.get("value")
        normalized_values.append(normalize_evidence_value(value))
        errors.extend(validate_minimal_evidence_value(value))
        if span_id not in catalog:
            errors.append("bad_span_id"); continue
        covered_answer_spans.add(span_id)
        if (span_id not in answer_span_ids or
                not _normalized_phrase_in_text(value, catalog[span_id]["exact_text"]) or
                not _normalized_phrase_in_text(value, correct or "")):
            errors.append("bad_answer_evidence_binding")
    if set(answer_span_ids) != covered_answer_spans:
        errors.append("bad_answer_evidence_binding")
    if len(normalized_values) != len(set(normalized_values)):
        errors.append("bad_duplicate_information_point")

    distractor_support = item.get("distractor_support")
    local_distractor_quotes: list[list[dict[str, str]]] = []
    local_distractor_values: list[str] = []
    if not isinstance(distractor_support, list) or len(distractor_support) != 3:
        errors.append("bad_distractor_support")
    else:
        by_index: dict[int, dict[str, Any]] = {}
        invalid_or_duplicate_index = False
        for entry in distractor_support:
            index_value = entry.get("distractor_index") if isinstance(entry, dict) else None
            if (not isinstance(index_value, int) or isinstance(index_value, bool) or
                    index_value not in (0, 1, 2) or index_value in by_index):
                invalid_or_duplicate_index = True
                continue
            by_index[index_value] = entry
        if invalid_or_duplicate_index or set(by_index) != {0, 1, 2}:
            errors.append("bad_distractor_support")
        for index in range(3):
            entry = by_index.get(index, {})
            if set(entry) != {"distractor_index", "span_ids", "evidence_values"}:
                errors.append("bad_distractor_support"); local_distractor_quotes.append([])
                local_distractor_values.append(""); continue
            span_ids = entry.get("span_ids")
            values = entry.get("evidence_values")
            if (not isinstance(span_ids, list) or not span_ids or len(span_ids) != len(set(span_ids)) or
                    any(not isinstance(x, str) or not x for x in span_ids) or
                    not isinstance(values, list) or not values or
                    any(not isinstance(x, str) or not x for x in values)):
                errors.append("bad_distractor_support"); local_distractor_quotes.append([])
                local_distractor_values.append(""); continue
            if any(span_id not in catalog for span_id in span_ids):
                errors.extend(("bad_span_id", "bad_distractor_support"))
            for value in values:
                errors.extend(validate_minimal_evidence_value(value))
                if (not _normalized_phrase_in_text(value, distractors[index] if isinstance(distractors, list) else "") or
                        not any(span_id in catalog and
                                _normalized_phrase_in_text(value, catalog[span_id]["exact_text"])
                                for span_id in span_ids)):
                    errors.append("bad_distractor_support")
            if any(not any(_normalized_phrase_in_text(value, catalog[span_id]["exact_text"])
                           for value in values) for span_id in span_ids if span_id in catalog):
                errors.append("bad_distractor_support")
            local_distractor_quotes.append([
                {"turn_id":catalog[span_id]["turn_id"],
                 "exact_quote":catalog[span_id]["exact_text"]}
                for span_id in span_ids if span_id in catalog])
            local_distractor_values.append(values[0] if values else "")

    answer_spans = [catalog[span_id] for span_id in answer_span_ids if span_id in catalog]
    distinct_turns = len({span["turn_id"] for span in answer_spans})
    distinct_sessions = len({span["session_id"] for span in answer_spans})
    stats = {"distinct_spans":len(set(answer_span_ids)), "distinct_turns":distinct_turns,
        "distinct_sessions":distinct_sessions, "distinct_values":len(set(normalized_values)),
        "reasoning_steps":len(steps) if isinstance(steps, list) else 0}
    reasoning_type = str((reasoning_contract or {}).get("reasoning_type") or "")
    difficulty = derive_local_difficulty(**stats, reasoning_type=reasoning_type)
    try:
        contract_ok = bool(reasoning_contract) and contract_is_satisfied(reasoning_contract, **stats)
    except (KeyError, TypeError, ValueError):
        contract_ok = False
    if not contract_ok:
        errors.extend(("bad_reasoning_contract", "bad_difficulty_evidence_contract"))
    if difficulty is None:
        errors.append("bad_difficulty_evidence_contract")
    if allow_local_difficulty_downgrade:
        errors.append("unsafe_local_difficulty_downgrade_enabled")
    if errors:
        return None, sorted(set(errors))
    options: dict[str, str] = {}
    explanation_map: dict[str, str] = {}
    cursor = 0
    for letter in LETTERS:
        if letter == target_letter:
            options[letter] = correct
            explanation_map[letter] = str(item.get("correct_answer_explanation") or "")
        else:
            options[letter] = distractors[cursor]
            explanation_map[letter] = explanations[cursor]
            cursor += 1
    evidence_turn_ids = list(dict.fromkeys(span["turn_id"] for span in answer_spans))
    evidence_session_ids = sorted({span["session_id"] for span in answer_spans})
    local_answer_values = [{"turn_id":catalog[entry["span_id"]]["turn_id"],
        "value":entry["value"], "exact_quote":catalog[entry["span_id"]]["exact_text"],
        "role":"answer_value"} for entry in evidence_values]
    qa = {"qa_id":item["qa_id"], "question":item["question"],
        "correct_answer_explanation":item["correct_answer_explanation"],
        "reasoning_steps":list(steps), "answer_support_type":"multi_hop",
        "answer_support_span_ids":list(answer_span_ids),
        "answer_supporting_quotes":[{"turn_id":span["turn_id"],
            "exact_quote":span["exact_text"]} for span in answer_spans],
        "answer_evidence_values":local_answer_values,
        "distractor_supporting_quotes":local_distractor_quotes,
        "distractor_values_from_evidence":local_distractor_values,
        "evidence_turn_ids":evidence_turn_ids, "evidence_session_ids":evidence_session_ids,
        "reasoning_type":reasoning_type, "difficulty_level":difficulty,
        "final_reasoning_conclusion":correct,
        "questioner_schema_version":QUESTIONER_SCHEMA_VERSION,
        "span_catalog_version":SPAN_CATALOG_VERSION,
        "reasoning_contract_version":REASONING_CONTRACT_VERSION,
        "span_catalog_sha256":span_catalog_sha256(span_catalog),
        "reasoning_contract_sha256":single_reasoning_contract_sha256(reasoning_contract)}
    qa.update({"options":options, "reference_answer_letter":target_letter,
               "reference_answer_text":options[target_letter],
               "distractor_explanations":explanation_map,
               "canonical_answer_text":options[target_letter],
               "local_option_binding":True})
    return qa, []


def validate_dialogue(parent: list[dict[str, Any]], child: list[dict[str, Any]],
                      child_tokens: int, sessions: list[dict[str, Any]]) -> list[str]:
    errors = []
    ids = [t.get("turn_id") for t in parent + child]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_turn_id")
    if child_tokens < 2000:
        errors.append("child_token_delta_below_2000")
    for i, session in enumerate(sessions):
        if i == 0 and not (session.get("used_parent_anchor_ids") or session.get("opening_hook_used")):
            errors.append("session_1_missing_anchor")
        if i > 0 and not ((session.get("continued_unresolved_hook_ids") or []) +
                          (session.get("resolved_hook_ids") or [])):
            errors.append("session_missing_unresolved_hook")
        if set(session.get("continued_unresolved_hook_ids") or []) & set(session.get("resolved_hook_ids") or []):
            errors.append("continued_resolved_hook_overlap")
    return errors


def can_stop(child_tokens: int, event_arcs_closed: bool) -> bool:
    return child_tokens >= 2000 and event_arcs_closed


def validate_memory(assertion: dict[str, Any], frozen_turns: dict[str, str],
                    prior: list[dict[str, Any]] = (), *,
                    known_subjects: set[str] | None = None,
                    predicate_whitelist: set[str] | None = None) -> list[str]:
    errors = []
    ids = assertion.get("supporting_turn_ids") or []
    if not ids or any(i not in frozen_turns for i in ids):
        errors.append("supporting_turn_not_frozen")
    spans = assertion.get("supporting_text_spans") or []
    valid_exact_quotes: list[str] = []
    for span in spans:
        if span.get("turn_id") not in frozen_turns or span.get("exact_span", "") not in frozen_turns.get(span.get("turn_id"), ""):
            errors.append("exact_span_not_found")
        else:
            valid_exact_quotes.append(str(span.get("exact_span")))
    if any(not assertion.get(k) for k in ("subject", "predicate", "object_or_state")):
        errors.append("non_atomic_empty_field")
    normalized_subject = normalize_option(assertion.get("subject", ""))
    normalized_predicate = normalize_option(assertion.get("predicate", ""))
    normalized_value = normalize_option(assertion.get("object_or_state", ""))
    if known_subjects is not None:
        known = {normalize_option(x) for x in known_subjects}
        quote_material = normalize_option(" ".join(valid_exact_quotes))
        if normalized_subject not in known and not _normalized_phrase_in_text(normalized_subject, quote_material):
            errors.append("unsupported_assertion_subject")
    if predicate_whitelist is not None and normalized_predicate not in {
            normalize_option(x) for x in predicate_whitelist}:
        errors.append("unsupported_assertion_predicate")
    if valid_exact_quotes and not any(_normalized_phrase_in_text(normalized_value, x) for x in valid_exact_quotes):
        errors.append("unsupported_assertion_value")
    if set(ids) != {str(span.get("turn_id")) for span in spans if isinstance(span, dict)}:
        errors.append("supporting_turn_span_mismatch")
    conflicts = [p for p in prior if normalize_option(p.get("subject", "")) == normalized_subject and
                 normalize_option(p.get("predicate", "")) == normalized_predicate and
                 p.get("object_or_state") != assertion.get("object_or_state") and p.get("status") == "active"]
    supersedes = set(assertion.get("supersedes_assertion_ids") or [])
    if any(p.get("assertion_id") not in supersedes for p in conflicts):
        errors.append("missing_supersession")
    return sorted(set(errors))


def _evidence_stage_accounting(stage: Any, expected_k: int = 3) -> dict[str, Any]:
    if isinstance(stage, dict):
        return {key:stage.get(key, 0) for key in ("requested_valid_rollouts",
            "obtained_valid_rollouts", "raw_generations", "parse_invalid_generations",
            "retry_generations", "exhausted_target_slots")} | {
            "verified_rewards":list(stage.get("verified_rewards") or []), "expected_valid_k":expected_k}
    rewards = list(stage or []) if isinstance(stage, list) else []
    return {"requested_valid_rollouts":expected_k, "obtained_valid_rollouts":len(rewards),
        "raw_generations":len(rewards), "parse_invalid_generations":0,
        "retry_generations":0, "exhausted_target_slots":max(0, expected_k-len(rewards)),
        "verified_rewards":rewards, "expected_valid_k":expected_k}


def evidence_audit(qa: dict[str, Any], evidence_rewards: Any, masked_rewards: Any) -> dict[str, Any]:
    evidence_stage = _evidence_stage_accounting(evidence_rewards)
    masked_stage = _evidence_stage_accounting(masked_rewards)
    difficulty_before = qa.get("difficulty_level")
    complete = all(stage["requested_valid_rollouts"] == 3 and
                   stage["obtained_valid_rollouts"] == 3 and
                   len(stage["verified_rewards"]) == 3
                   for stage in (evidence_stage, masked_stage))
    if not complete:
        return {"decision":"evidence_audit_inconclusive", "status":"evidence_audit_inconclusive",
            "difficulty_level_before":difficulty_before, "difficulty_level_after":difficulty_before,
            "evidence_only_accounting":evidence_stage, "evidence_masked_accounting":masked_stage}
    evidence_values = evidence_stage["verified_rewards"]
    masked_values = masked_stage["verified_rewards"]
    l1 = len(qa.get("evidence_turn_ids") or []) >= 2 and len(qa.get("reasoning_steps") or []) >= 2
    certified = sum(evidence_values) >= 2 and sum(masked_values) <= 1
    if certified:
        decision = "certified"; after = difficulty_before
        status = "evidence_relevance_certified"
    else:
        decision = "downgrade_l1" if l1 else "bad_evidence_relevance"
        after = "L1" if decision == "downgrade_l1" else difficulty_before
        status = "evidence_relevance_uncertified"
    return {"decision":decision, "status":status, "difficulty_level_before":difficulty_before,
        "difficulty_level_after":after, "evidence_only_accounting":evidence_stage,
        "evidence_masked_accounting":masked_stage}


def validate_lineage_splits(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(str(row["conversation_id"]), set()).add(str(row["split"]))
    leaked = [k for k, values in seen.items() if len(values) > 1]
    if leaked:
        raise ValueError(f"lineage split leakage: {leaked}")


def parse_questioner_batches(batches: list[Any]) -> dict[str, Any]:
    counts, accepted, invalid = [], [], []
    for index, batch in enumerate(batches):
        count = len(batch) if isinstance(batch, list) else 0
        counts.append(count)
        if count != 3:
            invalid.append(index)
        else:
            accepted.extend(batch)
    shortfall = len(batches) != 4 or any(x != 3 for x in counts) or sum(counts) != 12
    return {"generation_counts": counts, "batch_count_invalid": invalid,
            "raw_candidate_count": sum(counts), "candidates_from_valid_batches": accepted,
            "raw_candidate_count_shortfall":shortfall,
            "questioner_candidate_shortfall": shortfall}


def assess_candidate_shortfall(generation_counts: list[int], solver_eligible_count: int,
                               *, minimum_solver_eligible_candidates: int) -> dict[str, Any]:
    raw_count = sum(generation_counts)
    raw_shortfall = len(generation_counts) != 4 or generation_counts != [3, 3, 3, 3]
    accepted_shortfall = solver_eligible_count < minimum_solver_eligible_candidates
    candidate_shortfall = raw_shortfall or accepted_shortfall
    return {"questioner_raw_candidate_count":raw_count,
        "questioner_solver_eligible_count":solver_eligible_count,
        "minimum_solver_eligible_candidates":minimum_solver_eligible_candidates,
        "raw_candidate_count_shortfall":raw_shortfall,
        "accepted_candidate_shortfall":accepted_shortfall,
        "candidate_shortfall":candidate_shortfall, "completed":not candidate_shortfall}


def validate_realizer_identity_contract(parsed: dict[str, Any], outline: dict[str, Any],
                                         session_index: int,
                                         active_hooks: list[dict[str, str]]) -> list[str]:
    """Validate people/location and disjoint P/E/H ID namespaces for one session."""
    errors: list[str] = []
    core = set(outline.get("source_parent", {}).get("core_speakers") or [])
    introductions = {x.get("name"): int(str(x.get("introduction_scene") or "S999").lstrip("S") or 999)
                     for x in outline.get("allowed_new_characters") or []}
    known_people = core | set(introductions)
    mentioned = {x.get("speaker") for x in parsed.get("dialogue_turns") or [] if isinstance(x, dict)}
    if any(name not in known_people for name in mentioned):
        errors.append("unauthorized_named_entity")
    if any(name in introductions and introductions[name] > session_index for name in mentioned):
        errors.append("unauthorized_early_entity_introduction")
    allowed_now = core | {name for name, first in introductions.items() if first <= session_index}
    if any(x.get("speaker") not in allowed_now for x in parsed.get("dialogue_turns") or [] if isinstance(x, dict)):
        errors.append("unauthorized_speaker")
    arc_ids = parsed.get("advanced_event_arc_ids") or []
    allowed_arcs = {x.get("event_id") for x in outline.get("planned_event_arcs") or []}
    if any(not isinstance(x, str) or not x.startswith("E") or x not in allowed_arcs for x in arc_ids):
        errors.append("unauthorized_event_arc")
    anchor_ids = parsed.get("used_parent_anchor_ids") or []
    allowed_anchors = {x.get("anchor_id") for x in outline.get("parent_anchors") or []}
    if any(not isinstance(x, str) or not x.startswith("P") or x not in allowed_anchors for x in anchor_ids):
        errors.append("unauthorized_parent_anchor_id")
    allowed_hook_ids = {x.get("hook_id") for x in active_hooks if isinstance(x, dict)}
    hook_ids = (parsed.get("continued_unresolved_hook_ids") or []) + (parsed.get("resolved_hook_ids") or [])
    if any(not isinstance(x, str) or not x.startswith("H") or x not in allowed_hook_ids for x in hook_ids):
        errors.append("unauthorized_unresolved_hook_id")
    if session_index == 1 and hook_ids:
        errors.append("session_1_existing_hook_ids_must_be_empty")
    if set(parsed.get("continued_unresolved_hook_ids") or []) & set(parsed.get("resolved_hook_ids") or []):
        errors.append("continued_resolved_hook_overlap")
    return sorted(set(errors))


def validate_realizer_output_schema(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return ["realizer_schema_invalid"]
    errors: list[str] = []
    required = set(REALIZER_OUTPUT_SCHEMA)
    actual = set(parsed)
    if "new_named_entities" in actual:
        errors.append("deprecated_new_named_entities")
    if required - actual:
        errors.append("realizer_missing_fields")
    if actual - required:
        errors.append("realizer_extra_fields")
    for key in required - {"dialogue_turns"}:
        if key in parsed and not isinstance(parsed[key], list):
            errors.append("realizer_schema_invalid")
    for key in ("used_parent_anchor_ids", "advanced_event_arc_ids",
                "continued_unresolved_hook_ids", "resolved_hook_ids"):
        values = parsed.get(key) or []
        if any(not isinstance(x, str) or not x for x in values) or len(values) != len(set(values)):
            errors.append("realizer_schema_invalid")
    turns = parsed.get("dialogue_turns")
    if not isinstance(turns, list) or not turns or any(
            not isinstance(x, dict) or set(x) != {"speaker", "text"} or
            not isinstance(x.get("speaker"), str) or not isinstance(x.get("text"), str) or
            not x["speaker"].strip() or not x["text"].strip() for x in turns or []):
        errors.append("bad_dialogue_turn_schema")
    for candidate in parsed.get("durable_assertion_candidates") or []:
        if (not isinstance(candidate, dict) or
                set(candidate) != {"subject", "predicate", "object_or_state", "supporting_turn_indices"} or
                any(not isinstance(candidate.get(key), str) or not candidate.get(key).strip()
                    for key in ("subject", "predicate", "object_or_state")) or
                not isinstance(candidate.get("supporting_turn_indices"), list) or
                not candidate.get("supporting_turn_indices")):
            errors.append("bad_durable_assertion_candidate_schema")
    for hook in parsed.get("new_unresolved_hooks") or []:
        if not isinstance(hook, dict) or set(hook) != {"description"} or not str(hook.get("description", "")).strip():
            errors.append("bad_new_unresolved_hook_schema")
    if any(not isinstance(x, str) or not x.strip() for x in parsed.get("local_scene_facts") or []):
        errors.append("realizer_schema_invalid")
    return sorted(set(errors))


def validate_durable_candidate_indices(candidates: Any, turn_count: int) -> list[str]:
    if not isinstance(candidates, list):
        return ["bad_supporting_turn_indices"]
    for candidate in candidates:
        indices = candidate.get("supporting_turn_indices") if isinstance(candidate, dict) else None
        if (not isinstance(indices, list) or not indices or
                any(not isinstance(index, int) or index < 0 or index >= turn_count for index in indices)):
            return ["bad_supporting_turn_indices"]
    return []


def audit_continuity_hooks(active_before: set[str], continued: set[str],
                           resolved: set[str]) -> dict[str, Any]:
    """Audit rolling hook continuity without assuming any fixed hook identifier."""
    referenced = continued | resolved
    checks = {
        "all_references_authorized": referenced <= active_before,
        "continues_or_resolves_existing_hook": bool(referenced),
        "no_unknown_hook_reference": not bool(referenced - active_before),
        "continued_resolved_disjoint": not bool(continued & resolved),
    }
    return {"active_before": sorted(active_before), "continued": sorted(continued),
            "resolved": sorted(resolved), "referenced": sorted(referenced),
            **checks, "continuity_complete": all(checks.values())}


def validate_new_hook_descriptions(new_hooks: list[Any], active_hooks: list[dict[str, str]]) -> list[str]:
    descriptions = [normalize_option(x.get("description", ""))
                    for x in new_hooks if isinstance(x, dict)]
    active = {normalize_option(x.get("description", "")) for x in active_hooks if isinstance(x, dict)}
    if (len(descriptions) != len(new_hooks) or len(descriptions) != len(set(descriptions)) or
            any(not value or value in active for value in descriptions)):
        return ["duplicate_unresolved_hook_description"]
    return []


def validate_hook_transition(active_before: set[str], continued: set[str], resolved: set[str],
                             new_hook_ids: list[str], *, can_stop_after: bool) -> dict[str, Any]:
    errors: list[str] = []
    referenced = continued | resolved
    if referenced - active_before:
        errors.append("unauthorized_unresolved_hook_id")
    if continued & resolved:
        errors.append("continued_resolved_hook_overlap")
    if len(new_hook_ids) != len(set(new_hook_ids)) or any(x in active_before for x in new_hook_ids):
        errors.append("duplicate_unresolved_hook_id")
    active_after = (active_before - resolved) | set(new_hook_ids)
    if not can_stop_after and not active_after:
        errors.append("no_unresolved_hook_for_required_continuation")
    return {"active_before":sorted(active_before), "continued":sorted(continued),
        "resolved":sorted(resolved), "new_hook_ids":list(new_hook_ids),
        "active_after":sorted(active_after), "can_stop_after":can_stop_after,
        "errors":sorted(set(errors))}


def validate_event_arc_advancement(advanced_ids: list[str], assigned_ids: set[str],
                                   closed_ids: set[str], dialogue_turns: list[dict[str, Any]],
                                   assigned_beat_support: dict[str, str]) -> list[str]:
    import re
    errors: list[str] = []
    advanced = set(advanced_ids)
    if advanced - assigned_ids:
        errors.append("unassigned_event_arc_advance")
    if advanced & closed_ids:
        errors.append("already_closed_event_arc")
    dialogue = normalize_option(" ".join(str(x.get("text") or "") for x in dialogue_turns
                                         if isinstance(x, dict)))
    stop = {"this", "that", "with", "from", "into", "whether", "current", "event", "stage"}
    for event_id in advanced & assigned_ids - closed_ids:
        material = normalize_option(assigned_beat_support.get(event_id, ""))
        keywords = {token for token in re.findall(r"[\w]+", material, flags=re.UNICODE)
                    if len(token) >= 4 and token not in stop}
        if not dialogue or not keywords or not any(token in dialogue for token in keywords):
            errors.append("missing_event_arc_dialogue_support")
    return sorted(set(errors))


def validate_questioner_smoke_batch(batch: Any, assigned_letters: list[str],
                                    valid_turn_ids: set[str], valid_session_ids: set[str],
                                    frozen_turns: dict[str, str] | None = None,
                                    turn_session_ids: dict[str, str] | None = None,
                                    span_catalog: dict[str, Any] | None = None,
                                    reasoning_contracts: list[dict[str, Any]] | None = None,
                                    allow_local_difficulty_downgrade: bool = False) -> dict[str, Any]:
    """Strict one-batch parser result with independent per-item acceptance."""
    if not isinstance(batch, list) or len(batch) != 3:
        count = len(batch) if isinstance(batch, list) else 0
        return {"json_top_level_array": isinstance(batch, list), "raw_object_count": count,
                "batch_decision": "questioner_batch_count_invalid",
                "questioner_candidate_shortfall": True, "items": []}
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    seen_option_sets: set[tuple[str, ...]] = set()
    if (not isinstance(reasoning_contracts, list) or len(reasoning_contracts) != 3 or
            not isinstance(span_catalog, dict)):
        return {"json_top_level_array":True, "raw_object_count":3,
            "batch_decision":"questioner_local_contract_invalid",
            "questioner_candidate_shortfall":True, "schema_accepted_count":0,
            "schema_rejected_count":3, "items":[]}
    for index, item in enumerate(batch):
        qa, errors = materialize_questioner_item(item, assigned_letters[index],
            span_catalog=span_catalog, reasoning_contract=reasoning_contracts[index],
            allow_local_difficulty_downgrade=allow_local_difficulty_downgrade)
        if qa is not None:
            errors.extend(validate_qa(qa, valid_turn_ids, frozen_turns, turn_session_ids))
        qa_id = str(item.get("qa_id", "")).strip() if isinstance(item, dict) else ""
        if not qa_id or qa_id in seen_ids:
            errors.append("bad_or_duplicate_qa_id")
        question = normalize_option(item.get("question", "")) if isinstance(item, dict) else ""
        options = qa.get("options") or {} if qa is not None else {}
        option_set = tuple(sorted(normalize_option(x) for x in options.values())) if isinstance(options, dict) else ()
        option_types = {_answer_type_signature(str(x)) for x in options.values()} if isinstance(options, dict) else set()
        if len(option_types) > 1:
            errors.append("option_answer_type_mismatch")
        if not question or question in seen_questions or (option_set and option_set in seen_option_sets):
            errors.append("duplicate_or_option_permutation")
        reasoning_type = str(reasoning_contracts[index].get("reasoning_type") or "")
        sessions = qa.get("evidence_session_ids", []) if qa is not None else []
        if not sessions or any(x not in valid_session_ids for x in sessions):
            errors.append("bad_evidence_session")
        if qa is not None and qa.get("difficulty_level") not in ("L1", "L2", "L3"):
            errors.append("bad_difficulty")
        material = canonical = normalize_option(json_like(item)) if isinstance(item, dict) else ""
        if any(token in material for token in ("placeholder", "insert answer", "external knowledge", "google")):
            errors.append("placeholder_or_external_knowledge")
        errors = sorted(set(errors))
        if qa_id:
            seen_ids.add(qa_id)
        if question:
            seen_questions.add(question)
        if option_set:
            seen_option_sets.add(option_set)
        items.append({"item_index": index, "qa_id": qa_id or None,
            "target_letter": assigned_letters[index],
            "parsed_reference_letter": qa.get("reference_answer_letter") if qa else None,
            "schema_decision": "accept" if not errors else "reject", "reject_reasons": errors,
            "evidence_turn_ids": qa.get("evidence_turn_ids", []) if qa else [],
            "evidence_session_ids":sessions,
            "reasoning_type":reasoning_type,
            "difficulty":qa.get("difficulty_level") if qa else None,
            "materialized_qa":qa})
    accepted = sum(x["schema_decision"] == "accept" for x in items)
    return {"json_top_level_array": True, "raw_object_count": 3,
            "batch_decision": "questioner_batch_count_valid",
            "questioner_candidate_shortfall": False, "schema_accepted_count": accepted,
            "schema_rejected_count": 3 - accepted, "items": items}


def json_like(value: Any) -> str:
    """Stable text material for local smoke-only hygiene checks."""
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _answer_type_signature(text: str) -> str:
    normalized = normalize_option(text)
    if normalized in {"yes", "no", "true", "false"}:
        return "boolean"
    if normalized and (normalized[0].isdigit() or normalized[0] in "$€£"):
        return "numeric_or_date"
    return "text_phrase"


def derive_local_character_ledger(parsed: dict[str, Any], outline: dict[str, Any],
                                  prior_sessions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Derive characters from legal speakers only; dialogue text is never treated as NER input."""
    prior_speakers = {turn.get("speaker") for session in prior_sessions
                      for turn in session.get("dialogue_turns") or []}
    current_speakers = {turn.get("speaker") for turn in parsed.get("dialogue_turns") or []}
    expandable = {x.get("name") for x in outline.get("allowed_new_characters") or []}
    introduced = sorted((current_speakers & expandable) - prior_speakers)
    return {"speaker_names": sorted(current_speakers),
            "newly_introduced_character_names": introduced,
            "all_seen_character_names": sorted(prior_speakers | current_speakers)}
