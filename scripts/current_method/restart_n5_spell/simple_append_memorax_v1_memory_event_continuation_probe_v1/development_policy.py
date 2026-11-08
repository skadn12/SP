"""Opt-in evidence and information-gain policy; no transports or model calls.

Exact evidence matching below proves provenance, not semantic entailment.
Semantic judgments remain attributable to the existing Gate/outline roles.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
from typing import Any, Mapping

POLICY = "development_gain_v2"
QUALITY = {"useful", "low", "unknown"}
COVERAGE = {"new_activity", "new_person", "new_setting", "relationship_change",
            "existing_event_progress", "new_information"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def memory_facts(view):
    return {str(fact["memory_ref"]): str(fact["text"])
            for rows in (view.get("buckets") or {}).values()
            for row in rows if isinstance(row, Mapping)
            for fact in row.get("memory_facts", [])
            if isinstance(fact, Mapping) and fact.get("memory_ref")
            and isinstance(fact.get("text"), str)}


def candidate_views(plan):
    views = []
    for scene in plan.get("candidate_pool") or plan.get("scenes") or []:
        for index, candidate in enumerate(scene.get("next_step_candidates") or [], 1):
            views.append({"candidate_ref": f"{scene['memory_event_id']}:C{index:02d}",
                          "development_mode": scene.get("development_mode"),
                          **{key: deepcopy(candidate.get(key)) for key in (
                              "before_state", "required_transition", "after_state",
                              "actor_bindings", "interaction_beats", "source_basis",
                              "new_people")},
                          "transition_from_previous": scene.get("transition_from_previous")})
    return views


def planner_prompt(prompt):
    result = deepcopy(prompt)
    result["development_policy"] = POLICY
    result["prompt_version"] += ":" + POLICY
    result["prompt"] += """

DEVELOPMENT QUALITY POLICY v2
Propose useful developments using the SAME output schema and candidate budget.
For each candidate, before_state, required_transition and after_state must make
the expected new information explicit: what concrete fact will become known or
what bounded decision, arrangement, action, outcome or relationship will change?
A new person/place/title alone is not information gain. Reaffirming trust,
retelling a known incident, exchanging encouragement, or saying one understands
better is insufficient by itself. A specific new fact, preference, constraint,
reason or discovered result CAN be valuable; not everything must be an action.
Existing-event progress and independent developments compete on useful gain.
Explore plausible new activities, settings and relationships as well as old
threads. Do not force new people or one development from every category.
Do not split one activity into synonymous scenes. Keep independent candidates
independent: do not claim another candidate has happened in before_state unless
it is a necessary dependency. State necessary dependencies clearly in the
existing transition_from_previous/interaction_beats fields; they may be omitted
from the selected set. Never make support/reflection the main development merely
to fill a scene quota. A plan is an EXPECTED change, never realized history.
Use the supplied frozen Parent transcript as authoritative history and Memory
as unverified navigation. Preserve intent/ongoing/complete distinctions; do not
manufacture negative before_state facts from missing Memory. This policy adds
no new historical input and removes none of the supplied Parent transcript.

"""
    result["additional_memory_fact_count"] = 0
    result["planner_input_boundary"] = "frozen_parent_transcript_plus_existing_memory_view"
    return result


def gate_prompt(context):
    if not context.get("memory_only_fast"):
        raise ValueError("development_gain_requires_memory_only_gate")
    text = """You are the existing Memory Continuation Gate, policy v2.
Use ONLY the frozen Memory facts and proposed candidates below. Planner states
are proposals/readings, not evidence that a prior event occurred. Do not infer
identity, completion or noncompletion from the name verified_track_ref.

For EVERY candidate classify relation as REALIZED, PLANNED_OR_INTENDED, ONGOING,
NEW_PROPOSAL, or UNKNOWN. REALIZED means the SAME actor, action, object, event
occurrence, and target state are already established by cited Memory. Every
material part of a compound transition/end state must already be established.
Evidence can span several sentences/facts, including explicit co-reference.
Related topics, earlier similar events, a prepared backup plan, confidence,
readiness, intention, or a plan to submit DO NOT prove submission or sharing.
PLANNED_OR_INTENDED/ONGOING need evidence of that same event's current status.
NEW_PROPOSAL means a plausible independent proposed development grounded in
Memory, without relying on a fabricated claim that a prior action did not occur.
It is NOT certification that the event has never occurred. Use UNKNOWN when
identity/state/preconditions are ambiguous or contradictory; silence cannot
prove noncompletion. UNKNOWN is deferred to the existing information-gain and
precondition review. It is neither an automatic rejection nor proof of novelty.

Return exact Memory quotes (multiple allowed) and explicit identity/state checks.
For REALIZED, all five identity_match fields must be true, every completion_part
must be completed, evidence_indices must identify supporting quotes, and all
material actions/results in required_transition AND after_state must be covered.
For plans/ongoing do not mark completion parts completed. Do not select or rewrite.
Program validates quote identity and decision consistency; these semantic checks
remain YOUR judgments, not mechanical proof. Keep reasons specific and concise.

OUTPUT {"candidate_reviews":[{"candidate_ref":"...",
"relation":"UNKNOWN", "identity_match":{"actor":null,"action":null,
"object":null,"occurrence":null,"target_state":null},
"completion_parts":[{"claim":"...","state":"unknown",
"evidence_indices":[]}], "evidence":[{"memory_ref":"...","quote":"..."}],
"reason":"..."}]}
identity_match values are true/false/null. completion_parts state is completed,
planned, ongoing, or unknown. Evidence indices are 1-based into this row's
evidence. Return every candidate exactly once. Empty evidence is allowed only
for UNKNOWN. NEW_PROPOSAL cites its grounding in existing people/circumstances,
not fabricated evidence that the proposed action has already occurred.

FROZEN GATE INPUT
"""
    return {"prompt_version": "memory_gate_evidence_v2", "development_policy": POLICY,
            "prompt": text + _json(context)}


def _validate_evidence(rows, facts, errors, prefix):
    if not isinstance(rows, list):
        errors.append(prefix + ":evidence_shape")
        return []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"memory_ref", "quote"}:
            errors.append(prefix + ":evidence_shape")
            continue
        quote = row.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in facts.get(str(row.get("memory_ref")), ""):
            errors.append(prefix + ":quote_not_in_frozen_memory")
    return rows


def validate_gate(value, *, plan, memory_view, minimum=1):
    refs = [row["candidate_ref"] for row in candidate_views(plan)]
    errors, passed, rejected, held, normalized = [], [], [], [], []
    reviews = value.get("candidate_reviews") if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping) or set(value) != {"candidate_reviews"} or not isinstance(reviews, list):
        reviews = []
        errors.append("gain_gate_shape")
    seen = []
    facts = memory_facts(memory_view)
    for row in reviews:
        if not isinstance(row, Mapping) or set(row) != {"candidate_ref", "relation", "identity_match", "completion_parts", "evidence", "reason"}:
            errors.append("gain_gate_review_shape")
            continue
        ref = row["candidate_ref"]
        if not isinstance(ref, str):
            errors.append("gain_gate_candidate_ref")
            continue
        seen.append(ref)
        relation = row["relation"]
        if not isinstance(relation, str) or relation not in {"REALIZED", "PLANNED_OR_INTENDED", "ONGOING", "NEW_PROPOSAL", "UNKNOWN"}:
            errors.append(f"{ref}:relation")
            relation = "UNKNOWN"
        evidence = _validate_evidence(row["evidence"], facts, errors, str(ref))
        identity = row["identity_match"]
        if not isinstance(identity, Mapping) or set(identity) != {"actor", "action", "object", "occurrence", "target_state"} or any(v is not None and not isinstance(v, bool) for v in identity.values()):
            errors.append(f"{ref}:identity_shape")
            identity = {}
        parts = row["completion_parts"]
        if not isinstance(parts, list) or not parts:
            errors.append(f"{ref}:completion_parts")
            parts = []
        for part in parts:
            if not isinstance(part, Mapping) or set(part) != {"claim", "state", "evidence_indices"} or not isinstance(part.get("claim"), str) or not part["claim"].strip():
                errors.append(f"{ref}:completion_part_shape")
                continue
            indices = part["evidence_indices"]
            if not isinstance(part["state"], str) or part["state"] not in {"completed", "planned", "ongoing", "unknown"}:
                errors.append(f"{ref}:completion_part_state")
            if not isinstance(indices, list) or any(type(i) is not int or not 1 <= i <= len(evidence) for i in indices):
                errors.append(f"{ref}:completion_part_evidence")
            if part["state"] != "unknown" and not indices:
                errors.append(f"{ref}:completion_part_evidence_required")
        if relation == "REALIZED" and (not evidence or not parts or len(identity) != 5 or not all(v is True for v in identity.values()) or any(not isinstance(p, Mapping) or p.get("state") != "completed" for p in parts)):
            errors.append(f"{ref}:realized_without_complete_identity_state_evidence")
        if relation in {"PLANNED_OR_INTENDED", "ONGOING"} and (not evidence or not all(identity.get(k) is True for k in ("actor", "action", "object", "occurrence")) or not any(isinstance(p, Mapping) and p.get("state") in ("planned", "ongoing") for p in parts)):
            errors.append(f"{ref}:noncompletion_without_state_evidence")
        if relation == "NEW_PROPOSAL" and not evidence:
            errors.append(f"{ref}:new_proposal_grounding_evidence_required")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            errors.append(f"{ref}:reason_required")
        decision = "reject" if relation == "REALIZED" else "defer" if relation == "UNKNOWN" else "pass"
        if decision == "reject":
            rejected.append(ref)
        else:
            passed.append(ref)  # Eligible for the remaining review, not accepted.
            if decision == "defer":
                held.append(ref)
        normalized.append({**deepcopy(row), "parent_relation": relation,
                           "decision": decision, "duplicate_with_parent": relation == "REALIZED",
                           "supporting_memory_refs": [e.get("memory_ref") for e in evidence if isinstance(e, Mapping)]})
    if sorted(map(str, seen)) != sorted(refs):
        errors.append("gain_gate_candidate_coverage")
    return {"valid": not errors, "errors": errors, "development_policy": POLICY,
            "overall_decision": "invalid" if errors else "pass" if len(passed) >= minimum else "reject",
            "passed_candidate_refs": passed if not errors else [],
            "rejected_candidate_refs": rejected, "deferred_candidate_refs": held,
            "passed_candidate_count": len(passed) if not errors else 0,
            "minimum_pass_candidates": minimum, "normalized_candidate_reviews": normalized,
            "semantic_entailment": "model_judgment_not_mechanically_verified"}


def outline_prompt(*, plan, memory_view, gate_validation=None):
    from .contracts import build_planner_outline_pairs
    views = candidate_views(plan)
    pairs = [{"pair_ref": p["pair_ref"],
              "left_candidate_ref": p["left"]["candidate_ref"],
              "right_candidate_ref": p["right"]["candidate_ref"]}
             for p in build_planner_outline_pairs(plan)]
    text = """You are the existing Planner outline auditor, development quality v2.
In THIS SAME request review each candidate's expected information gain and all
pairs' duplication. Use the frozen Memory only; do not assume Memory silence
proves an event has not occurred. Do not write QA, rewrite plans, or output
realized facts: all candidates describe EXPECTED additions pending Realizer.

For each candidate: quality useful/low/unknown. Useful means a specific
new arrangement, responsibility, result, action, relationship change, constraint,
reason or information that can be clearly established within the scene.
This is a discrete admissibility judgment, not a novelty score. Reaffirmed trust,
retelling known events, generic reassurance and vague increased confidence are
low alone. A concrete informative discovery can be useful without an action.
New people, locations or activities alone do not earn quality. Do not force them.
Assess old-event continuations and independent developments on equal terms.

Copy target_after_state EXACTLY from the candidate. expected_information states
the concrete predicted addition. coverage uses only new_activity, new_person,
new_setting, relationship_change, existing_event_progress, new_information.
These are audit tags, not quotas. If nothing fits, use [].
precondition_status is supported/conditional/unknown. supported means Memory
supports factual prerequisites or the scene establishes them without inventing
history. conditional means prerequisite actions must occur in other candidates;
list those candidate refs in depends_on. unknown means unclear support and holds
the candidate out. Read before_state and transition_from_previous for necessary
dependencies. Mere topic or narrative proximity is NOT a dependency. A candidate
depending on rejected/absent work must not be silently relabelled independent.
Explain support with exact Memory evidence when relevant; evidence may be [].
The prior Gate's UNKNOWN means completion was not determined. It alone neither
establishes a missing prerequisite nor prevents a useful independent proposal.
Evaluate factual prerequisites separately; do not infer noncompletion from it.

For each pair return distinct/duplicate/uncertain. Similar topics can have
independent developments; conversely review of a topic followed by discussion
of the same understanding is not automatically a distinct new development.
Never mark stages distinct solely because verbs differ. Report missing/uncertain
facts conservatively. Output every candidate and pair exactly once.

OUTPUT {"candidate_reviews":[{"candidate_ref":"...","quality":"useful",
"expected_information":"...","target_after_state":"...","coverage":[],
"precondition_status":"supported","depends_on":[],"evidence":[],
"reason":"..."}],"pair_reviews":[{"pair_ref":"P01",
"decision":"distinct","reason":"..."}]}
Evidence objects are {"memory_ref":"...","quote":"exact text"}.

FROZEN MEMORY
"""
    return {"prompt_version": "planner_gain_outline_v2", "development_policy": POLICY,
            "candidate_count": len(views), "pair_count": len(pairs),
            "prompt": text + _json(memory_view) + "\nCANDIDATES\n" + _json(views) + "\nPAIRS\n" + _json(pairs) +
            "\nPRIOR GATE (semantic judgments, not facts)\n" + _json([
                {k: r.get(k) for k in ("candidate_ref", "relation", "decision", "reason")}
                for r in (gate_validation or {}).get("normalized_candidate_reviews", [])])}


def validate_outline(value, *, plan, memory_view):
    from .contracts import validate_planner_outline_semantic_output
    errors = []
    if not isinstance(value, Mapping) or set(value) != {"candidate_reviews", "pair_reviews"}:
        return {"valid": False, "errors": ["gain_outline_shape"]}
    audit = validate_planner_outline_semantic_output({"pair_reviews": value["pair_reviews"]}, plan=plan)
    if not audit["valid"]:
        errors.extend(audit["errors"])
    candidates = {row["candidate_ref"]: row for row in candidate_views(plan)}
    reviews = value["candidate_reviews"]
    seen, normalized = [], {}
    if not isinstance(reviews, list):
        errors.append("gain_candidate_reviews_shape")
        reviews = []
    for row in reviews:
        required = {"candidate_ref", "quality", "expected_information", "target_after_state", "coverage", "precondition_status", "depends_on", "evidence", "reason"}
        if not isinstance(row, Mapping) or set(row) != required:
            errors.append("gain_candidate_review_shape")
            continue
        ref = row["candidate_ref"]
        if not isinstance(ref, str):
            errors.append("gain_candidate_ref")
            continue
        seen.append(ref)
        if ref not in candidates:
            errors.append(f"{ref}:unknown_candidate")
        elif row["target_after_state"] != candidates[ref]["after_state"]:
            errors.append(f"{ref}:target_state_changed")
        if not isinstance(row["quality"], str) or row["quality"] not in QUALITY:
            errors.append(f"{ref}:quality")
        for key in ("expected_information", "reason"):
            if not isinstance(row[key], str) or not row[key].strip():
                errors.append(f"{ref}:{key}_required")
        if not isinstance(row["coverage"], list) or any(not isinstance(t, str) or t not in COVERAGE for t in row["coverage"]):
            errors.append(f"{ref}:coverage")
        deps = row["depends_on"]
        if not isinstance(deps, list) or any(not isinstance(d, str) or d not in candidates or d == ref for d in deps):
            errors.append(f"{ref}:dependencies")
            deps = []
        if not isinstance(row["precondition_status"], str) or row["precondition_status"] not in {"supported", "conditional", "unknown"}:
            errors.append(f"{ref}:precondition_status")
        if (row["precondition_status"] == "conditional") != bool(deps) and row["precondition_status"] != "unknown":
            errors.append(f"{ref}:dependency_status_conflict")
        _validate_evidence(row["evidence"], memory_facts(memory_view), errors, ref)
        normalized[ref] = deepcopy(row)
    if sorted(seen) != sorted(candidates):
        errors.append("gain_outline_candidate_coverage")
    exact_conflicts = []
    for left, right in combinations(candidates.values(), 2):
        a = " ".join(str(left["required_transition"] or "").casefold().split())
        b = " ".join(str(right["required_transition"] or "").casefold().split())
        if a and a == b and left.get("actor_bindings") == right.get("actor_bindings"):
            exact_conflicts.append(sorted([left["candidate_ref"], right["candidate_ref"]]))
    audit["conflict_edges"] = [list(pair) for pair in sorted({tuple(sorted(e)) for e in audit.get("conflict_edges", []) + exact_conflicts})]
    audit["exact_duplicate_conflicts"] = exact_conflicts
    return {**audit, "valid": not errors, "errors": errors,
            "development_policy": POLICY, "candidate_reviews": normalized,
            "semantic_entailment": "model_judgment_not_mechanically_verified"}


def select_candidates(passed_refs, *, audit, limit):
    """Filter low gain, then maximize useful scene count and coverage."""
    if not audit.get("valid"):
        return {"valid": False, "selected_candidate_refs": [], "errors": ["invalid_gain_audit"]}
    reviews = audit["candidate_reviews"]
    eligible = [r for r in passed_refs if reviews[r]["quality"] == "useful" and reviews[r]["precondition_status"] != "unknown"]
    conflicts = {frozenset(edge) for edge in audit.get("conflict_edges", [])}

    def ordered(combo):
        remaining, result = list(combo), []
        while remaining:
            ready = [r for r in remaining if set(reviews[r]["depends_on"]) <= set(result)]
            if not ready:
                return []  # missing prerequisites or cycle
            ref = ready[0]
            result.append(ref)
            remaining.remove(ref)
        return result

    best, best_coverage = [], -1
    for count in range(1, min(limit, len(eligible)) + 1):
        for combo in combinations(eligible, count):
            if any(frozenset(pair) in conflicts for pair in combinations(combo, 2)):
                continue
            order = ordered(combo)
            if not order:
                continue
            # No model novelty score. All remaining candidates are useful;
            # retain the largest legal set, use coverage and original rank
            # only to resolve equal-size sets.
            coverage_count = len({tag for r in combo for tag in reviews[r]["coverage"]})
            if (count, coverage_count) > (len(best), best_coverage):
                best, best_coverage = order, coverage_count
    reasons = {}
    for ref in passed_refs:
        if ref not in best:
            r = reviews[ref]
            reasons[ref] = ("low_or_unknown_information_gain" if r["quality"] != "useful" else
                            "uncertain_preconditions" if r["precondition_status"] == "unknown" else
                            "dependency_conflict_or_coverage_tiebreak")
    return {"valid": bool(best), "development_policy": POLICY,
            "selected_candidate_refs": best, "blocked_candidate_refs": [r for r in passed_refs if r not in best],
            "required_count": limit, "allow_partial": True,
            "unfilled_scene_count": limit - len(best), "selected_coverage_count": best_coverage,
            "selection_policy": "useful_admission_then_count_coverage_original_rank",
            "blocked_reasons": reasons, "candidate_reviews": deepcopy(reviews)}


def materialization_input(plan, selected):
    """Order selected candidates by validated dependencies before legacy compile."""
    result = deepcopy(plan)
    rank = {ref: i for i, ref in enumerate(selected)}
    pool = result.get("candidate_pool") or result.get("scenes") or []
    result["candidate_pool"] = sorted(pool, key=lambda s: rank.get(f"{s['memory_event_id']}:C01", len(rank)))
    return result


def annotate_selected(plan, audit, *, round_index):
    result = deepcopy(plan)
    ref_to_scene = {s["selected_candidate_ref"]: f"R{round_index:02d}:S{i:02d}"
                    for i, s in enumerate(result["scenes"], 1)}
    result["development_policy"] = POLICY
    for scene in result["scenes"]:
        review = audit["candidate_reviews"][scene["selected_candidate_ref"]]
        scene["development_policy"] = POLICY
        scene["expected_information"] = review["expected_information"]
        scene["depends_on_scene_refs"] = [ref_to_scene[r] for r in review["depends_on"]]
        scene["information_status"] = "expected_not_realized"
    return result
