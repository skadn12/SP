"""Lossless citation dereferencing, without generating evidence or verdicts.

Models select stable turn IDs. The program retrieves the complete corresponding
turn text from the supplied, frozen context. It never searches for a similar
sentence, repairs a model-written quote, or selects an alternative source.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _index(history, new_turns=None):
    records = {}
    for namespace, turns in (("history", history), ("new_scene", new_turns or [])):
        if not isinstance(turns, list):
            raise ValueError("citation_context_not_array:" + namespace)
        for position, turn in enumerate(turns, 1):
            if not isinstance(turn, dict) or not isinstance(turn.get("text"), str) or not turn["text"].strip():
                raise ValueError("citation_context_invalid_turn:" + namespace)
            ref = turn.get("turn_ref")
            if namespace == "new_scene" and ref is None:
                ref = f"NEW:T{position:03d}"
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError("citation_context_missing_reference:" + namespace)
            if ref in records:
                raise ValueError("citation_context_duplicate_reference:" + ref)
            records[ref] = {"namespace": namespace, "text": turn["text"],
                            "turn_sha256": digest(turn)}
    return records


def _walk(value, records, path, receipts):
    if isinstance(value, dict):
        if set(value) == {"turn_ref"}:
            ref = value["turn_ref"]
            record = records.get(ref) if isinstance(ref, str) else None
            if record is None:
                raise ValueError("citation_reference_not_in_context:" + str(ref))
            receipts.append({"path": path, "turn_ref": ref, "namespace": record["namespace"],
                             "turn_sha256": record["turn_sha256"],
                             "quote_sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
                             "selection": "model_selected_reference",
                             "text_origin": "program_dereferenced_complete_turn"})
            return {"turn_ref": ref, "quote": record["text"]}
        # Existing model-written quotes remain untouched. Callers must validate
        # them against their specific historical/new evidence namespace.
        return {key: _walk(item, records, f"{path}.{key}", receipts) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, records, f"{path}[{index}]", receipts) for index, item in enumerate(value)]
    return deepcopy(value)


def materialize_citations(raw, history, new_turns=None):
    """Return a new object; only exact {turn_ref: ID} citation leaves expand."""
    return _walk(raw, _index(history, new_turns), "$", [])


def citation_provenance(raw, history, new_turns=None):
    """Record model selection and program text lookup as distinct operations."""
    receipts = []
    _walk(raw, _index(history, new_turns), "$", receipts)
    return {"version": "v19_evidence_refs_1", "raw_object_sha256": digest(raw),
            "history_sha256": digest(history), "new_turns_sha256": digest(new_turns or []),
            "dereferenced_citations": receipts, "semantic_support_verified": False}
