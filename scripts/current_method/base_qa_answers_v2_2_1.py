"""Typed semantic answers, task answers and stable option identities.

No presentation label is an answer identity.  JSON booleans, integers and other
scalar types remain distinct; unordered sets and partial-order edge sets are
canonicalized, while sequences retain their meaning-bearing order.
"""
from __future__ import annotations

import copy
import json
import math
import re

VERSION = "base_qa_answers_v2_2_1"
QUESTION_TYPES = frozenset({"single_choice", "multi_select", "ordering", "partial_order"})
SEMANTIC_KINDS = frozenset({"scalar", "object", "set", "sequence", "relation_set"})
_OPTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _json(value):
    """Return a typed JSON clone, rejecting coercible/nonstandard Python values."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("NONFINITE_JSON_NUMBER")
        return value
    if type(value) is list:
        return [_json(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("JSON_OBJECT_KEYS_MUST_BE_STRINGS")
        return {key: _json(value[key]) for key in sorted(value)}
    raise ValueError("NON_JSON_VALUE")


def typed_key(value):
    """Hashable canonical JSON bytes; unlike Python equality, true != 1."""
    return json.dumps(_json(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def typed_equal(left, right):
    return typed_key(left) == typed_key(right)


def _unique(values, field):
    keys = [typed_key(value) for value in values]
    if len(keys) != len(set(keys)):
        raise ValueError(field + ":DUPLICATE_VALUE")
    return keys


def _relation_closure(pairs, allowed=None):
    if type(pairs) is not list:
        raise ValueError("RELATION_PAIRS_MUST_BE_LIST")
    vertices, edges = {}, set()
    allowed_keys = None if allowed is None else {typed_key(item) for item in allowed}
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2:
            raise ValueError("RELATION_PAIR_MUST_HAVE_TWO_ITEMS")
        left, right = (_json(item) for item in pair)
        a, b = typed_key(left), typed_key(right)
        if a == b:
            raise ValueError("RELATION_SELF_PAIR")
        if allowed_keys is not None and (a not in allowed_keys or b not in allowed_keys):
            raise ValueError("RELATION_UNKNOWN_ITEM")
        vertices[a], vertices[b] = left, right
        if (a, b) in edges:
            raise ValueError("RELATION_DUPLICATE_PAIR")
        edges.add((a, b))
    # Reachability, not the author's chosen reduced/full edge serialization,
    # determines partial-order equality.  A cycle is never normalized away.
    reach = {node: set() for node in vertices}
    for a, b in edges:
        reach[a].add(b)
    for middle in sorted(vertices):
        for start in sorted(vertices):
            if middle in reach[start]:
                reach[start].update(reach[middle])
    if any(node in reach[node] for node in vertices):
        raise ValueError("RELATION_CYCLE")
    return [[vertices[a], vertices[b]] for a in sorted(reach)
            for b in sorted(reach[a])]


def semantic_answer(kind, value):
    """Build the canonical semantic value without binding it to a question form."""
    if type(kind) is not str or kind not in SEMANTIC_KINDS:
        raise ValueError("SEMANTIC_KIND_INVALID")
    value = _json(value)
    if kind == "scalar":
        if type(value) in (dict, list):
            raise ValueError("SCALAR_REQUIRED")
    elif kind == "object":
        if type(value) is not dict:
            raise ValueError("OBJECT_REQUIRED")
    elif kind in ("set", "sequence"):
        if type(value) is not list:
            raise ValueError("SEMANTIC_ARRAY_REQUIRED")
        if kind == "set":
            _unique(value, "SEMANTIC_SET")
            value.sort(key=typed_key)
    else:
        value = _relation_closure(value)
    return {"kind": kind, "value": value}


def validate_semantic(answer):
    if type(answer) is not dict or set(answer) != {"kind", "value"}:
        raise ValueError("SEMANTIC_ANSWER_EXACT_SCHEMA_REQUIRED")
    return semantic_answer(answer["kind"], answer["value"])


_build_semantic = semantic_answer


def _ids(option_ids):
    if not isinstance(option_ids, (dict, list, tuple, set, frozenset)):
        raise ValueError("OPTION_IDS_INVALID")
    ids = list(option_ids)
    if (not ids or any(type(item) is not str or not _OPTION_ID.fullmatch(item) for item in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("OPTION_IDS_INVALID")
    return set(ids)


def canonical_answer(answer, question_type, option_ids):
    """Validate a typed task answer over stable IDs (or frozen display labels)."""
    ids = _ids(option_ids)
    if type(question_type) is not str:
        raise ValueError("QUESTION_TYPE_INVALID")
    kind = {"single_choice": "single_option", "multi_select": "option_set",
            "ordering": "option_sequence", "partial_order": "partial_order"}.get(question_type)
    if kind is None or type(answer) is not dict:
        raise ValueError("TASK_ANSWER_TYPE_INVALID")
    field = "option_id" if question_type == "single_choice" else (
        "pairs" if question_type == "partial_order" else "option_ids")
    if set(answer) != {"kind", field} or answer.get("kind") != kind:
        raise ValueError("TASK_ANSWER_EXACT_SCHEMA_REQUIRED")
    value = answer[field]
    if question_type == "single_choice":
        if type(value) is not str or value not in ids:
            raise ValueError("ANSWER_OPTION_ID_UNKNOWN")
        return {"kind": kind, field: value}
    if question_type == "partial_order":
        if type(value) is not list or any(type(pair) is not list or len(pair) != 2
                                         or any(type(item) is not str for item in pair)
                                         for pair in value):
            raise ValueError("ANSWER_PARTIAL_PAIRS_INVALID")
        return {"kind": kind, field: _relation_closure(value, ids)}
    if (type(value) is not list or not value
            or any(type(item) is not str or item not in ids for item in value)
            or len(value) != len(set(value))):
        raise ValueError("ANSWER_OPTION_IDS_INVALID")
    if question_type == "ordering" and set(value) != ids:
        raise ValueError("ANSWER_COMPLETE_ORDER_COVERAGE_REQUIRED")
    return {"kind": kind, field: sorted(value) if question_type == "multi_select" else value[:]}


def map_answer(answer, question_type, mapping):
    """Map an already frozen namespace; no label is guessed from its position."""
    if type(mapping) is not dict or any(type(value) is not str for value in mapping.values()):
        raise ValueError("ANSWER_MAPPING_INVALID")
    canonical = canonical_answer(answer, question_type, mapping)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("ANSWER_MAPPING_NOT_BIJECTIVE")
    if question_type == "single_choice":
        result = {"kind": "single_option", "option_id": mapping[canonical["option_id"]]}
    elif question_type == "partial_order":
        result = {"kind": "partial_order", "pairs": [[mapping[a], mapping[b]]
                                                      for a, b in canonical["pairs"]]}
    else:
        result = {"kind": canonical["kind"],
                  "option_ids": [mapping[item] for item in canonical["option_ids"]]}
    return canonical_answer(result, question_type, list(mapping.values()))


def validate_candidate_answer(candidate, semantic_answer=None):
    """Require the selected stable option meanings to equal the frozen target.

    The optional argument is the pre-construction TARGET value.  The candidate's
    own private copy must agree with it; it never replaces a mismatching target.
    """
    if type(candidate) is not dict or "gold" in candidate:
        raise ValueError("LEGACY_GOLD_FORBIDDEN")
    options, analysis = candidate.get("options"), candidate.get("item_analysis")
    if type(options) is not dict or type(analysis) is not dict or set(analysis) != set(options):
        raise ValueError("OPTION_ANALYSIS_EXACT_COVERAGE_REQUIRED")
    ids = _ids(options)
    expected = validate_semantic(candidate.get("semantic_answer"))
    if semantic_answer is not None and not typed_equal(expected, validate_semantic(semantic_answer)):
        raise ValueError("CANDIDATE_TARGET_SEMANTIC_MISMATCH")
    values = {}
    for key in sorted(ids):
        if type(analysis[key]) is not dict or "semantic_value" not in analysis[key]:
            raise ValueError("OPTION_SEMANTIC_VALUE_REQUIRED")
        values[key] = _json(analysis[key]["semantic_value"])
    _unique(list(values.values()), "OPTION_SEMANTICS")
    qtype = candidate.get("question_type")
    gold = canonical_answer(candidate.get("gold_answer"), qtype, ids)
    if qtype == "single_choice":
        derived = _build_semantic(expected["kind"], values[gold["option_id"]])
        # Distinct raw edge/set serializations may have the same semantic value.
        equal_options = []
        for key, value in values.items():
            try:
                if typed_equal(_build_semantic(expected["kind"], value), expected):
                    equal_options.append(key)
            except ValueError:
                pass  # A distractor need not have the target answer's shape.
        if equal_options != [gold["option_id"]]:
            raise ValueError("SINGLE_SEMANTIC_ANSWER_NOT_UNIQUE")
    else:
        needed_kind = {"multi_select": "set", "ordering": "sequence",
                       "partial_order": "relation_set"}[qtype]
        if expected["kind"] != needed_kind:
            raise ValueError("QUESTION_SEMANTIC_KIND_MISMATCH")
        raw = ([[values[a], values[b]] for a, b in gold["pairs"]]
               if qtype == "partial_order" else [values[key] for key in gold["option_ids"]])
        derived = _build_semantic(needed_kind, raw)
    if not typed_equal(derived, expected):
        raise ValueError("GOLD_SEMANTIC_MISMATCH")
    return {"canonical_gold": gold, "derived_semantic": derived}


def normalize_candidate(candidate):
    """Canonical storage ignores insertion order and preserves meaningful order."""
    checked = validate_candidate_answer(candidate)
    normalized = _json(copy.deepcopy(candidate))
    options = normalized["options"]
    if (not 2 <= len(options) <= 26
            or any(type(text) is not str or not text.strip() for text in options.values())):
        raise ValueError("OPTIONS_REQUIRE_2_TO_26_NONEMPTY_TEXTS")
    if any(type(normalized.get(key)) is not str or not normalized[key].strip()
           for key in ("candidate_id", "author_id", "question")):
        raise ValueError("CANDIDATE_ID_AUTHOR_OR_QUESTION_INVALID")
    normalized["semantic_answer"] = validate_semantic(normalized["semantic_answer"])
    normalized["gold_answer"] = checked["canonical_gold"]
    normalized["options"] = {key: normalized["options"][key] for key in sorted(options)}
    normalized["item_analysis"] = {key: normalized["item_analysis"][key] for key in sorted(options)}
    return normalized
