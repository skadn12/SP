"""Check responsiveness to reported schema errors; never repair or certify meaning.

This is an additional retry gate. Every response still needs the original full
validator. A replacement quote/reference must be supplied for an invalid evidence
entry: deleting it, deleting its owning check, or erasing a negative cannot
silently turn a citation correction into acceptance. A formerly positive check
may become negative during complete substantive re-review.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re


VERSION = "v101_exact_comparison_wrapper_correction_responsiveness_v1"
IDENTITIES = {
    "segments": [("segment_id",)], "timeline": [("constraint_id",)],
    "plan_checks": [("development_index",)], "checks": [("development_index",)],
    "card_checks": [("card_id",), ("kind", "id")],
    "step_checks": [("step_id",)], "dependency_checks": [("dependency_index",)],
    "consistency_checks": [("category",), ("check_id",)],
    "state_checks": [("fact_id",)], "event_checks": [("constraint_id",)],
    "roles": [("actor_id",)],
}
EVIDENCE_LIST_KEYS = {"evidence", "plan_evidence", "history_evidence",
                      "request_evidence", "acceptance_evidence", "source_paths"}
MISSING = object()
CONTRACT_INVALID_NEGATIVE_CODES = {
    "event_state_history_absence_not_contradiction",
    "event_state_history_absence_not_card_failure",
    "event_state_history_absence_not_internal_conflict",
}
FUNCTION_WRAPPER_FIELDS = {"comparison", "plan_evidence", "reason", "verdict"}
BOUND_COMPARISON_FIELDS = {
    "purpose", "start_state", "end_state", "roles", "current_trigger",
    "nearest_old_event", "request_acceptance", "experience_state", "source_binding",
}


def _row_identity(collection, row):
    for keys in IDENTITIES.get(collection, []):
        if isinstance(row, dict) and all(key in row for key in keys):
            return {"identity": {key: deepcopy(row[key]) for key in keys}}
    return None


def _parse(raw):
    if isinstance(raw, dict):
        return deepcopy(raw)
    if not isinstance(raw, str):
        return None

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _tokens(path):
    if not isinstance(path, str) or path == "response":
        return [] if path == "response" else None
    parts = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\[[0-9]+\]", path)
    if not parts or ".".join(parts).replace(".[", "[") != path:
        return None
    return [int(part[1:-1]) if part.startswith("[") else part for part in parts]


def _stable_path(parsed, tokens):
    """Bind rows to declared identities so reordering cannot mimic a correction."""
    current, stable, collection = parsed, [], None
    for token in tokens:
        if isinstance(token, str):
            stable.append(token)
            current = current.get(token, MISSING) if isinstance(current, dict) else MISSING
            collection = token
        else:
            row = current[token] if isinstance(current, list) and token < len(current) else MISSING
            identity = _row_identity(collection, row)
            if identity:
                stable.append(identity)
            else:
                stable.append(token)
            current = row
    return stable


def _at(parsed, tokens):
    current = parsed
    for token in tokens:
        if isinstance(token, dict):
            matches = [row for row in current if isinstance(row, dict)
                       and all(row.get(key, MISSING) == value for key, value in token["identity"].items())] \
                if isinstance(current, list) else []
            current = matches[0] if len(matches) == 1 else MISSING
        elif isinstance(token, str):
            current = current.get(token, MISSING) if isinstance(current, dict) else MISSING
        else:
            current = current[token] if isinstance(current, list) and token < len(current) else MISSING
        if current is MISSING:
            break
    return current


def _protections(previous, errors):
    """Freeze explicit negatives/applicability and evidence duties within a retry.

    These are model-supplied values, not inferred semantic truth. Only an exact
    owner whose negative was itself rejected by a recognized contract-error
    code is exempt; no other negative in that document is released.
    """
    exempt = []
    for error in errors:
        if error.get("code") in CONTRACT_INVALID_NEGATIVE_CODES:
            tokens = _tokens(error.get("path"))
            if tokens and tokens[-1] == "reason":
                exempt.append(_stable_path(previous, tokens[:-1]))
    protected, evidence = [], []

    def walk(value, path, collection=None):
        if isinstance(value, dict):
            verdict = value.get("verdict")
            if verdict in ("revise", "contradicted") and path not in exempt:
                protected.append({"path": path + ["verdict"], "value": verdict, "kind": "explicit_negative"})
            for key, expected in (("realized", False), ("new_event_or_progress", False),
                                  ("consistent_with_history", False), ("duplicates_existing_event", True)):
                if key in value and value[key] is expected:
                    protected.append({"path": path + [key], "value": expected, "kind": "explicit_negative"})
            if path and path[-1] == "request_acceptance" and value.get("required") is True:
                protected.append({"path": path + ["required"], "value": True, "kind": "declared_applicability"})
            for key, child in value.items():
                if key in EVIDENCE_LIST_KEYS and isinstance(child, list) and child:
                    evidence.append({"path": path + [key], "minimum_count": len(child)})
                walk(child, path + [key], key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + [_row_identity(collection, child) or index])

    walk(previous, [])
    return protected, evidence, exempt


def _exact_fields(value, expected):
    return (isinstance(value, list) and all(isinstance(key, str) for key in value)
            and len(value) == len(expected) and set(value) == expected)


def _relocate_wrapper_obligations(previous, errors, protected, evidence):
    """Carry duties into one precisely reported, missing comparison wrapper.

    This recognizes only the exact pair of original validator errors for a
    flattened bound card function comparison. It neither edits the archived
    document nor searches other nodes for similar content. Error targets keep
    their original paths so the reported missing wrapper must still be fixed.
    """
    relocations, seen = [], set()
    for wrapper_error in errors:
        if (wrapper_error.get("code") != "event_state_fields_invalid"
                or not _exact_fields(wrapper_error.get("allowed"), FUNCTION_WRAPPER_FIELDS)):
            continue
        path = wrapper_error.get("path")
        tokens = _tokens(path)
        if (tokens is None or len(tokens) != 3 or tokens[0] != "card_checks"
                or not isinstance(tokens[1], int) or tokens[2] != "function_check"
                or path != f"card_checks[{tokens[1]}].function_check" or path in seen):
            continue
        stable = _stable_path(previous, tokens)
        node = _at(previous, tokens)
        # A unique declared card identity binds the relocation to this row even
        # when the retry reorders card_checks. Unidentified/ambiguous rows get
        # no authorization from this narrowly scoped exception.
        if (not isinstance(stable[1], dict) or _at(previous, stable) is not node
                or not isinstance(node, dict) or "comparison" in node
                or set(node) != BOUND_COMPARISON_FIELDS):
            continue
        missing_error = next((error for error in errors
                              if error.get("code") == "event_identity_fields_invalid"
                              and error.get("path") == path + ".comparison"
                              and error.get("actual", MISSING) is None
                              and _exact_fields(error.get("allowed"), BOUND_COMPARISON_FIELDS)
                              and set(error["allowed"]) == set(node)), None)
        if missing_error is None:
            continue
        destination = stable + ["comparison"]
        receipt = {
            "rule": "exact_flattened_bound_card_function_comparison",
            "original_path": path, "destination_path": path + ".comparison",
            "original_prefix": deepcopy(stable), "destination_prefix": deepcopy(destination),
            "trigger_errors": [deepcopy(wrapper_error), deepcopy(missing_error)],
            "obligation_mappings": [],
        }
        for kind, obligations in (("protected_judgment", protected), ("evidence_obligation", evidence)):
            for obligation in obligations:
                old_path = obligation["path"]
                if len(old_path) <= len(stable) or old_path[:len(stable)] != stable:
                    continue
                new_path = destination + old_path[len(stable):]
                mapping = {"obligation_type": kind, "old_path": deepcopy(old_path),
                           "new_path": deepcopy(new_path)}
                for key in ("value", "kind", "minimum_count"):
                    if key in obligation:
                        mapping[key] = deepcopy(obligation[key])
                obligation["path"] = new_path
                receipt["obligation_mappings"].append(mapping)
        relocations.append(receipt)
        seen.add(path)
    return relocations


def build_contract(errors, previous_raw):
    previous = _parse(previous_raw)
    targets = []
    for error in errors:
        path = error.get("path", "response")
        tokens = _tokens(path)
        target = {"code": error.get("code"), "path": path}
        if previous is not None and tokens is not None:
            stable = _stable_path(previous, tokens)
            value = _at(previous, stable)
            target.update(stable_path=stable, previous_present=value is not MISSING)
            if value is not MISSING:
                target["previous_value"] = deepcopy(value)
            evidence_position = next((i for i, token in enumerate(tokens)
                                      if token in EVIDENCE_LIST_KEYS), None)
            if evidence_position is not None:
                owner_path = stable[:evidence_position]
                owner = _at(previous, owner_path)
                key = tokens[evidence_position]
                entries = owner.get(key) if isinstance(owner, dict) else None
                after = tokens[evidence_position + 1:]
                if isinstance(entries, list) and after and isinstance(after[0], int) and after[0] < len(entries):
                    target["evidence_replacement"] = {
                        "owner_path": owner_path, "evidence_key": key,
                        "previous_count": len(entries), "previous_entry": deepcopy(entries[after[0]]),
                        "previous_verdict": owner.get("verdict"),
                    }
        targets.append(target)
    protected, evidence, exempt = _protections(previous, errors) if previous is not None else ([], [], [])
    relocations = _relocate_wrapper_obligations(previous, errors, protected, evidence) if previous is not None else []
    return {
        "version": VERSION, "previous_document_sha256": _digest(previous if previous is not None else previous_raw),
        "targets": targets, "full_validation_required": True, "semantic_certified": False,
        "protected_judgments": protected, "evidence_obligations": evidence,
        "wrapper_relocations": relocations,
        "contract_invalid_negative_owner_exemptions": exempt,
        "instruction": (
            "Change each reported invalid field or evidence reference using its actual allowed scope. "
            "Return the complete review again. Replace every invalid evidence entry with supported "
            "exact evidence; do not delete it, reduce that check's evidence count, remove the check, "
            "or erase a negative verdict to evade citation repair. A previous positive verdict "
            "may become revise after substantive review; a previous negative must not become "
            "positive or not_applicable during citation repair, unless the program precisely identified "
            "that same negative as an invalid reviewer-contract judgment. Preserve every other explicit "
            "negative and required=true applicability in the complete response, including nested checks. "
            "Do not reduce evidence counts or remove checks anywhere to evade format correction. "
            "Only an explicitly recorded wrapper_relocations mapping carries those same obligations "
            "into its exact missing comparison wrapper; all values and evidence counts still apply. "
            "A changed response is not proof of "
            "correctness; the complete original validator and semantic review still apply."
        ),
    }


def assess(raw, repair):
    result = {"version": VERSION, "status": "NOT_A_RETRY", "errors": [], "targets": [],
              "full_validation_required": True, "semantic_certified": False,
              "scope": "reported_error_responsiveness_only"}
    if repair is None:
        return result
    previous_raw = repair.get("previous_raw", repair.get("previous_response"))
    # Rebuild deterministically from the archived response, including older
    # repair envelopes made before this guard existed. Never trust a supplied
    # contract to omit a negative obligation.
    contract = build_contract(repair.get("errors", []), previous_raw)
    result["wrapper_relocations"] = deepcopy(contract["wrapper_relocations"])
    parsed = _parse(raw)
    if parsed is None:
        result.update(status="REJECT", errors=[{"code": "correction_response_not_parseable",
                                                 "path": "response", "actual": None}])
        return result
    result["previous_document_sha256"] = contract["previous_document_sha256"]
    result["response_document_sha256"] = _digest(parsed)
    if result["previous_document_sha256"] == result["response_document_sha256"]:
        result["errors"].append({"code": "correction_response_unchanged", "path": "response",
                                 "actual": result["response_document_sha256"]})
    result["protected_judgment_count"] = len(contract["protected_judgments"])
    result["evidence_obligation_count"] = len(contract["evidence_obligations"])
    result["contract_invalid_negative_owner_exemptions"] = deepcopy(contract["contract_invalid_negative_owner_exemptions"])
    for obligation in contract["protected_judgments"]:
        value = _at(parsed, obligation["path"])
        if type(value) is not type(obligation["value"]) or value != obligation["value"]:
            result["errors"].append({"code": "correction_protected_judgment_erased",
                                     "path": deepcopy(obligation["path"]),
                                     "actual": None if value is MISSING else deepcopy(value),
                                     "previous": deepcopy(obligation["value"]), "kind": obligation["kind"]})
    for obligation in contract["evidence_obligations"]:
        value = _at(parsed, obligation["path"])
        if not isinstance(value, list) or len(value) < obligation["minimum_count"]:
            result["errors"].append({"code": "correction_global_evidence_or_check_deleted",
                                     "path": deepcopy(obligation["path"]),
                                     "actual": None if value is MISSING else deepcopy(value),
                                     "minimum_count": obligation["minimum_count"]})
    for target in contract["targets"]:
        finding = {"code": target["code"], "path": target["path"], "changed": None}
        replacement = target.get("evidence_replacement")
        if replacement:
            owner = _at(parsed, replacement["owner_path"])
            entries = owner.get(replacement["evidence_key"]) if isinstance(owner, dict) else None
            if not isinstance(entries, list) or len(entries) < replacement["previous_count"]:
                result["errors"].append({"code": "correction_evidence_or_check_deleted",
                                         "path": target["path"], "actual": deepcopy(entries)})
            elif replacement["previous_entry"] in entries:
                result["errors"].append({"code": "correction_reported_evidence_unchanged",
                                         "path": target["path"], "actual": deepcopy(replacement["previous_entry"])})
            else:
                finding["changed"] = True
            old_verdict = replacement["previous_verdict"]
            new_verdict = owner.get("verdict") if isinstance(owner, dict) else None
            if old_verdict in ("compatible", "revise") and (
                    new_verdict not in ("compatible", "revise")
                    or (old_verdict == "revise" and new_verdict != "revise")):
                result["errors"].append({"code": "correction_evidence_owner_verdict_changed",
                                         "path": target["path"], "actual": new_verdict})
        elif "stable_path" in target:
            value = _at(parsed, target["stable_path"])
            finding["changed"] = (value is not MISSING) != target["previous_present"] or (
                value is not MISSING and value != target.get("previous_value"))
            if finding["changed"] is False:
                result["errors"].append({"code": "correction_reported_field_unchanged",
                                         "path": target["path"], "actual": None if value is MISSING else deepcopy(value)})
        result["targets"].append(finding)
    result["status"] = "REJECT" if result["errors"] else "PASS"
    return result
