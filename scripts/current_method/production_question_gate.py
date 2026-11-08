#!/usr/bin/env python3
"""Bind production questions to selected sources and prepare semantic review.

Exact quotation and graph checks are mechanical checks only. They do not prove
that an option is true, an edge is entailed, or a task requires its source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .question_presentation_gate import lint_candidate
except ImportError:  # Direct script execution.
    from question_presentation_gate import lint_candidate

SCHEMA_VERSION = "production_question_gate_v1"
OPERATION_CLASSES = {
    "clock_value", "occurrence_chronology", "discourse_reconstruction",
    "version_evolution", "source_defined_dependency",
}
OPERATION_HINTS = {
    "ORDER_EXPLICIT_TIME": "clock_value",
    "ORDER_OCCURRENCE": "occurrence_chronology",
    "ORDER_OCCURRENCE_TIME": "occurrence_chronology",
    "ORDER_EVENT_CHRONOLOGY": "occurrence_chronology",
    "ORDER_DISCOURSE": "discourse_reconstruction",
    "ORDER_DISCOURSE_RECONSTRUCTION": "discourse_reconstruction",
    "ORDER_VERSION_EVOLUTION": "version_evolution",
    "ORDER_PROCESS_DEPENDENCY": "source_defined_dependency",
}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _selection(selection_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(selection_path).resolve(strict=True)
    document = json.loads(path.read_text())
    roots = document.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("selection must contain nonempty roots")
    result = {}
    for entry in roots:
        root_id = entry.get("root_id")
        source_path = entry.get("path")
        if not isinstance(root_id, str) or not root_id or root_id in result:
            raise ValueError("selection root IDs must be nonempty and unique")
        if not isinstance(source_path, str) or not Path(source_path).is_absolute():
            raise ValueError("selected source paths must be absolute")
        binding = file_binding(source_path)
        if entry.get("sha256") != binding["sha256"]:
            raise ValueError(f"selected source hash drift: {root_id}")
        if entry.get("bytes", binding["bytes"]) != binding["bytes"]:
            raise ValueError(f"selected source byte count drift: {root_id}")
        source = json.loads(Path(binding["path"]).read_text())
        if not isinstance(source, dict) or source.get("root_id") != root_id:
            raise ValueError(f"selected source root mismatch: {root_id}")
        turns = source.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"selected source needs nonempty turns: {root_id}")
        refs = [turn.get("turn_ref") for turn in turns]
        if any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != len(refs):
            raise ValueError(f"source turn_ref missing or duplicated: {root_id}")
        if any(not isinstance(t.get("speaker"), str) or not isinstance(t.get("text"), str) for t in turns):
            raise ValueError(f"source turns need string speaker/text: {root_id}")
        result[root_id] = {"binding": binding, "turns": turns}
    return result, file_binding(path)


def bind_candidates(candidates: list[dict[str, Any]], selection_path: str | Path
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy candidates and derive absolute bindings solely from frozen selection.

    Authors may omit source_binding. Relative paths with a matching hash are
    normalized; a supplied different hash is an error, never silently repaired.
    """
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    sources, selection_binding = _selection(selection_path)
    bound, changes, seen = [], [], set()
    for original in candidates:
        candidate = copy.deepcopy(original)
        cid = candidate.get("candidate_id", candidate.get("id"))
        root_id = candidate.get("root_id", candidate.get("root"))
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("candidate IDs must be nonempty and unique")
        seen.add(cid)
        if root_id not in sources:
            raise ValueError(f"root not in frozen selection: {root_id}")
        binding = sources[root_id]["binding"]
        previous = candidate.get("source_binding")
        if isinstance(previous, dict) and previous.get("sha256", binding["sha256"]) != binding["sha256"]:
            raise ValueError(f"candidate source hash disagrees with selection: {cid}")
        candidate["source_binding"] = copy.deepcopy(binding)
        lint_candidate(candidate)  # Reuse strict presentation schema validation.
        if previous != binding:
            changes.append({"candidate_id": cid, "previous_binding": previous,
                            "derived_binding": copy.deepcopy(binding)})
        bound.append(candidate)
    return bound, {"schema_version": SCHEMA_VERSION, "selection_binding": selection_binding,
                   "candidate_count": len(bound), "binding_changes": changes,
                   "semantically_approved": False}


def _check_evidence(evidence: Any, by_ref: dict[str, Any], location: str,
                    errors: list[dict[str, str]]) -> list[int]:
    positions = []
    if not isinstance(evidence, list) or not evidence:
        errors.append({"location": location, "error": "nonempty evidence list required"})
        return positions
    for index, ev in enumerate(evidence):
        loc = f"{location}[{index}]"
        if not isinstance(ev, dict):
            errors.append({"location": loc, "error": "evidence must be an object"})
            continue
        turn = by_ref.get(ev.get("turn_ref"))
        if turn is None:
            errors.append({"location": loc, "error": "unknown turn_ref"})
            continue
        if ev.get("speaker") != turn["speaker"]:
            errors.append({"location": loc, "error": "speaker mismatch"})
        quote = ev.get("quote")
        if not isinstance(quote, str) or not quote or quote not in turn["text"]:
            errors.append({"location": loc, "error": "quote is not an exact nonempty source substring"})
        else:
            positions.append(turn["_position"])
    return positions


def _graph_audit(candidate: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, Any]:
    items = candidate["items"]
    successors = {key: set() for key in items}
    indegrees = {key: 0 for key in items}
    invalid = []
    for index, edge in enumerate(edges):
        before, after = edge.get("before"), edge.get("after")
        if before not in items or after not in items or before == after:
            invalid.append(index)
        elif after not in successors[before]:
            successors[before].add(after)
            indegrees[after] += 1
    order, unique = [], True
    while len(order) < len(items):
        ready = sorted(k for k in items if indegrees[k] == 0 and k not in order)
        if not ready:
            break
        if len(ready) != 1:
            unique = False
        selected = ready[0]
        order.append(selected)
        for after in successors[selected]:
            indegrees[after] -= 1
    acyclic = len(order) == len(items)
    return {"valid_edge_ids": not invalid, "invalid_edge_indices": invalid,
            "acyclic": acyclic, "unique_topological_order": acyclic and unique,
            "topological_order": order if acyclic else None,
            "graph_order_matches_gold": acyclic and unique and order == candidate["gold"],
            "entailed_edges_verified": False}


def audit_candidate(candidate: dict[str, Any], selection_path: str | Path) -> dict[str, Any]:
    """Audit exact references and proposed ordering graph without semantic claims."""
    sources, selection_binding = _selection(selection_path)
    lint = lint_candidate(candidate)
    cid = candidate.get("candidate_id", candidate.get("id"))
    root_id = candidate.get("root_id", candidate.get("root"))
    if root_id not in sources:
        raise ValueError(f"root not selected: {root_id}")
    source = sources[root_id]
    errors = []
    if candidate.get("source_binding") != source["binding"]:
        errors.append({"location": "source_binding", "error": "must equal absolute selection-derived binding"})
    by_ref = {turn["turn_ref"]: dict(turn, _position=i) for i, turn in enumerate(source["turns"])}
    kind = candidate.get("question_type", candidate.get("type"))
    choices = candidate["items" if kind == "ordering" else "options"]
    analyses = candidate.get("item_analysis")
    positions = {}
    if not isinstance(analyses, dict) or set(analyses) != set(choices):
        errors.append({"location": "item_analysis", "error": "one analysis required for every item and no extra IDs"})
        analyses = analyses if isinstance(analyses, dict) else {}
    for item_id in choices:
        analysis = analyses.get(item_id, {})
        found = _check_evidence(analysis.get("evidence"), by_ref, f"item_analysis.{item_id}.evidence", errors)
        positions[item_id] = min(found) if found else None
        if kind == "multi_select" and analysis.get("should_select") is not (item_id in candidate["gold"]):
            errors.append({"location": f"item_analysis.{item_id}.should_select", "error": "must be boolean matching Gold membership"})
    result = {"schema_version": SCHEMA_VERSION, "candidate_id": cid,
              "candidate_sha256": fingerprint(candidate), "source_binding": source["binding"],
              "selection_binding": selection_binding, "presentation_lint": lint,
              "mechanical_errors": errors, "semantically_approved": False,
              "source_truth_or_necessity_verified": False}
    if kind == "ordering":
        ordering = candidate.get("ordering_analysis", {})
        operation = candidate.get("task_operation", candidate.get("operation"))
        operation_class = ordering.get("operation_class", OPERATION_HINTS.get(operation))
        if operation_class not in OPERATION_CLASSES:
            errors.append({"location": "ordering_analysis.operation_class", "error": "explicit supported operation_class required for unknown operation"})
        edges = ordering.get("edges")
        if not isinstance(edges, list) or not edges:
            errors.append({"location": "ordering_analysis.edges", "error": "nonempty source-evidenced ordering edges required"})
            edges = []
        safe_edges = []
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append({"location": f"ordering_analysis.edges[{index}]", "error": "edge must be an object"})
                continue
            safe_edges.append(edge)
            _check_evidence(edge.get("evidence"), by_ref, f"ordering_analysis.edges[{index}].evidence", errors)
        graph = _graph_audit(candidate, safe_edges)
        if not graph["valid_edge_ids"] or not graph["graph_order_matches_gold"]:
            errors.append({"location": "ordering_analysis.edges", "error": "proposed graph must define unique total order equal to Gold"})
        values = [positions[item] for item in candidate["gold"]]
        known = all(p is not None for p in values)
        compatible = known and values == sorted(values)
        strict = compatible and len(set(values)) == len(values)
        result["ordering_checks"] = {
            "operation_class": operation_class, "graph": graph,
            "earliest_cited_turn_positions": positions,
            "gold_compatible_with_earliest_cited_turn_order": compatible,
            "gold_equals_strict_earliest_cited_turn_order": strict,
            "cited_position_ties": known and len(set(values)) != len(values),
            "first_relevant_mentions_verified": False,
            "mention_overlap_policy": ("May be the intended source-dependent operation; verify distinct discourse acts and reconstruction evidence."
                if operation_class == "discourse_reconstruction" else
                "Review possible shortcut; equality does not mechanically invalidate the question."),
            "common_sense_sufficiency": "REQUIRES_INDEPENDENT_SEMANTIC_REVIEW",
        }
    result["mechanical_pass"] = not errors
    return result


def required_checks(candidate: dict[str, Any]) -> dict[str, str]:
    checks = {
        "source_support": "Re-read every cited source turn; do the exact quotes entail each item's intended claim in its entity, scope and status?",
        "gold_truth": "Independently reconstruct the answer from the source, including all negatives; is the supplied Gold exactly correct?",
    }
    kind = candidate.get("question_type", candidate.get("type"))
    if kind == "multi_select":
        checks["option_distinctness"] = "Are the options distinct factual targets rather than paraphrases used to fill the Gold quota?"
        checks["six_gold_control" if len(candidate["gold"]) == 6 else "distractor_relation"] = (
            "Confirm all six are source-supported and mark this as an all-select control, never evidence of distractor difficulty."
            if len(candidate["gold"]) == 6 else
            "For every false option, identify the changed source relation and explain its falsity without unsupported negation or arbitrary outside facts."
        )
    else:
        checks.update({
            "operation_alignment": "Does the requested order mean clock values, event occurrence, discourse reconstruction, version evolution or source-defined dependencies? Do not substitute one for another.",
            "unique_total_order": "Do source-supported edges entail every necessary comparison and a unique total order? A mechanically valid authored graph is not evidence of entailment.",
            "mention_order": "Locate each item's FIRST RELEVANT mention, not merely its earliest author citation. If mention order matches Gold, explain whether this is intended discourse reconstruction or an unintended shortcut. Do not reject discourse reconstruction solely for this match.",
            "common_sense": "Could the displayed items be sorted by an ordinary workflow without this dialogue? Identify any pair requiring dialogue-specific facts, revisions or exceptions; do not invent a dependency.",
            "editorial_cues": "Inspect first/later/finally/最初/后来/总结 and overt ranks/times. Distinguish author-added answer-position cues from genuine evidence. Remove editorial cues, never silently delete meaningful source content.",
            "source_dependence": "Explain specifically which facts must be retrieved or bound from this dialogue. For discourse reconstruction, explain how distinct observed speech acts and their dialogue positions determine the answer; a common workflow alone is insufficient.",
        })
    return checks


def build_review_packet(candidate: dict[str, Any], selection_path: str | Path) -> dict[str, Any]:
    audit = audit_candidate(candidate, selection_path)
    return {"schema_version": SCHEMA_VERSION, "candidate_id": audit["candidate_id"],
            "candidate_sha256": audit["candidate_sha256"], "source_binding": audit["source_binding"],
            "mechanical_audit": audit, "required_checks": required_checks(candidate),
            "review_template": {"candidate_id": audit["candidate_id"],
                "candidate_sha256": audit["candidate_sha256"], "source_binding": audit["source_binding"],
                "author_id": candidate.get("author_id"), "reviewer_id": None,
                "reviewer_is_human": False, "gold_blind": False,
                "decision": "HOLD", "checks": {key: {"decision": "HOLD", "rationale": ""}
                    for key in required_checks(candidate)}},
            "semantic_review_required": True, "semantically_approved": False}


def validate_review(candidate: dict[str, Any], review: dict[str, Any], selection_path: str | Path
                    ) -> dict[str, Any]:
    audit = audit_candidate(candidate, selection_path)
    errors = []
    for key in ("candidate_id", "candidate_sha256", "source_binding"):
        if review.get(key) != audit[key]:
            errors.append(f"review {key} drift")
    author, reviewer = candidate.get("author_id"), review.get("reviewer_id")
    if not isinstance(author, str) or not author or not isinstance(reviewer, str) or not reviewer or author == reviewer:
        errors.append("nonempty independent author/reviewer IDs required")
    if review.get("author_id") != author:
        errors.append("review author_id mismatch")
    if review.get("decision") not in {"PASS", "HOLD", "FAIL"}:
        errors.append("review decision must be PASS/HOLD/FAIL")
    checks = review.get("checks", {})
    for key in required_checks(candidate):
        check = checks.get(key, {}) if isinstance(checks, dict) else {}
        if not isinstance(check, dict) or check.get("decision") not in {"PASS", "HOLD", "FAIL"} or not isinstance(check.get("rationale"), str) or not check["rationale"].strip():
            errors.append(f"explicit decision and nonempty rationale required: {key}")
        elif review.get("decision") == "PASS" and check["decision"] != "PASS":
            errors.append(f"PASS review contains unresolved check: {key}")
    admitted = audit["mechanical_pass"] and not errors and review.get("decision") == "PASS"
    return {"schema_version": SCHEMA_VERSION, "candidate_id": audit["candidate_id"],
            "candidate_sha256": audit["candidate_sha256"], "mechanical_pass": audit["mechanical_pass"],
            "review_record_complete": not errors, "review_errors": errors,
            "admitted": admitted, "semantic_admission_basis": "independent reviewer judgment, not a mechanical proof",
            "mechanical_audit": audit}


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reviews", type=Path, help="Independent reviews bound to the derived candidate hashes")
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    candidates = value.get("candidates") if isinstance(value, dict) else value
    bound, binding_audit = bind_candidates(candidates, args.selection)
    packets = [build_review_packet(c, args.selection) for c in bound]
    result = {"schema_version": SCHEMA_VERSION, "candidate_count": len(bound),
              "mechanical_pass_count": sum(p["mechanical_audit"]["mechanical_pass"] for p in packets),
              "binding_audit": binding_audit, "semantic_review_required": True,
              "admitted_count": 0, "training_launched": False}
    if args.reviews:
        reviews = json.loads(args.reviews.read_text())
        reviews = reviews.get("reviews") if isinstance(reviews, dict) else reviews
        if not isinstance(reviews, list) or len({r.get("candidate_id") for r in reviews}) != len(reviews):
            raise ValueError("reviews must be a list with unique candidate_id")
        by_id = {r["candidate_id"]: r for r in reviews}
        expected = {p["candidate_id"] for p in packets}
        if set(by_id) != expected:
            raise ValueError("reviews must cover exactly the submitted candidates")
        admissions = [validate_review(c, by_id[c.get("candidate_id", c.get("id"))], args.selection) for c in bound]
        result["admitted_count"] = sum(a["admitted"] for a in admissions)
        result["admissions"] = admissions
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write(args.output_dir / "CANDIDATES_BOUND_PRIVATE.json", bound)
    _write(args.output_dir / "REVIEW_PACKETS_PRIVATE.json", packets)
    _write(args.output_dir / "PRODUCTION_GATE_AUDIT.json", result)
    for p in args.output_dir.glob("*.json"):
        p.chmod(0o600)
    print(json.dumps({k: result[k] for k in ("candidate_count", "mechanical_pass_count", "admitted_count")}, ensure_ascii=False))
    return 0 if result["mechanical_pass_count"] == len(bound) and (not args.reviews or result["admitted_count"] == len(bound)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
