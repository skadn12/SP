"""Offline Codex authoring contracts for complex relationship discovery.

Inventory -> typed sweep -> composition/counterpath search -> difficulty ranking
-> answer freeze -> construction -> query-conditioned dependency audit.

No model, network, Solver, training, or source-approval implementation exists
here. A structural L3/L4 is a proposal for semantic review, not a proof that a
model will find a question difficult. Old pipelines and artifacts are untouched.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

try:
    from . import base_qa_pipeline_v1 as v1
    from . import base_qa_pipeline_v2 as v2
    from . import base_qa_pipeline_v2_1 as v21
except ImportError:
    import base_qa_pipeline_v1 as v1
    import base_qa_pipeline_v2 as v2
    import base_qa_pipeline_v2_1 as v21

VERSION = "base_qa_generation_v2_2"
PROMPT_DIR = Path(__file__).with_name("base_qa_prompts_v2_2")
SCHEMA_PATH = Path(__file__).with_name("base_qa_schema_v2_2.json")
FAMILIES = (
    "cross_scene_event_identity", "actor_action_binding", "event_state_update",
    "temporal_plan_revision", "multi_actor_roles", "object_attribute_binding",
    "quantity_unit_scope_constraint", "causal_relation", "joint_event_constraint",
    "composed_state_chain",
)
PASSES = ("inventory", "typed_sweep", "composition_counterpath")
LOW_OPS = {"extract", "retrieve_bind", "compare", "arithmetic", "explicit_order"}
REL_OPS = {"state_update", "plan_revision", "constraint_apply", "constraint_chain",
           "role_resolution", "object_resolution", "causal_chain", "composition"}
LEVELS = ("L0", "L1", "L2", "L3", "L4")
fp = v21.fingerprint
POLICY = dict(version=VERSION, challenger="Codex direct staged semantic work",
              model_calls_enabled=False, solver_enabled=False, training_enabled=False,
              formal_generation_requires_material_authorization=True,
              exhaustive_semantic_discovery_claim=False,
              structural_difficulty_is_empirical_difficulty=False)


class ValidationError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise ValidationError(message)


def _list(value, label, nonempty=False):
    _require(isinstance(value, list) and (not nonempty or bool(value)), label + ": list required")
    return value


def _same(a, b):
    """JSON-typed equality: booleans are not integers."""
    return fp(a) == fp(b)


def instruction(stage):
    _require(stage in {"inventory", "typed_sweep", "composition_counterpath", "ranking",
                       "target", "construction", "postconstruction_audit"}, "unknown authoring stage")
    filename = {"typed_sweep": "typed_relation_sweep",
                "composition_counterpath": "composition_and_counterpath"}.get(stage, stage)
    return (PROMPT_DIR / (filename + ".txt")).read_text(encoding="utf-8")


def source_index(source, source_scope=None):
    """Actual visible turns, including context-only turns, without inventing lineage."""
    _require(isinstance(source, dict) and v1.text(source.get("root_id")), "source root required")
    turns = _list(source.get("turns"), "source turns", True)
    scope = {r["turn_id"]: r for r in (source_scope or {}).get("turns", [])}
    result = {}
    for turn in turns:
        ref = turn.get("turn_ref")
        _require(v1.text(ref) and ref not in result and v1.text(turn.get("text")), "invalid or duplicate source turn")
        scene = turn.get("scene_ref") or turn.get("session_ref")
        # Missing scene identity is unknown, never manufacture a distinct scene per turn.
        row = scope.get(ref, {})
        result[ref] = dict(turn_id=ref, scene_id=scene or "UNKNOWN_SCENE",
                           source_id=row.get("source_id", turn.get("source_id", source.get("source_id", "UNKNOWN"))),
                           source_version=row.get("source_version", turn.get("source_version", "UNKNOWN")),
                           provenance_type=row.get("provenance_type", turn.get("provenance_type", "UNKNOWN")),
                           source_file_hash=row.get("source_file_hash", "UNKNOWN"),
                           formal_qa_allowed=row.get("formal_qa_allowed") is True,
                           text=turn["text"], speaker=turn.get("speaker"))
    return result


def _refs(refs, index, label):
    _list(refs, label, True)
    _require(all(isinstance(r, str) and r in index for r in refs), label + ": unknown source ref")
    _require(len(refs) == len(set(refs)), label + ": duplicate ref")
    return refs


def _quotes(evidence, source, label):
    errors = []
    refs = v1._evidence(evidence, {t["turn_ref"]: t for t in source["turns"]}, errors, label)
    _require(not errors, "; ".join(errors))
    return refs


def _full_scan(scan, index, label):
    _list(scan, label, True)
    refs = []
    for row in scan:
        _require(isinstance(row, dict) and v1.text(row.get("rationale")), label + ": rationale required")
        refs += _refs(row.get("turn_refs"), index, label)
    _require(len(refs) == len(set(refs)) and set(refs) == set(index),
             label + ": actual visible source must be covered exactly once")


def discovery_workplan(source, source_scope=None, inventory=None):
    """Generate work requests, not relationships. Codex executes each semantic pass.

    Inventory and explicit links narrow composition search; there is deliberately
    no all-scene Cartesian product or numeric-first question generator.
    """
    index = source_index(source, source_scope)
    scenes = list(dict.fromkeys(r["scene_id"] for r in index.values()))
    events = (inventory or {}).get("events", [])
    return dict(version=VERSION, root_id=source["root_id"], source_sha256=fp(source),
                source_scope_sha256=fp(source_scope or {}), policy=copy.deepcopy(POLICY),
                visible_turn_index=index, stages=[
        dict(stage="inventory", instruction=instruction("inventory"), scene_ids=scenes),
        dict(stage="typed_sweep", instruction=instruction("typed_sweep"),
             tasks=[dict(family=f, scene_ids=scenes, event_ids=[e["event_id"] for e in events]) for f in FAMILIES]),
        dict(stage="composition_counterpath", instruction=instruction("composition_counterpath"),
             search_basis="explicit identity/constraint links from inventory; preserve rejected and unchecked hypotheses"),
        dict(stage="ranking", instruction=instruction("ranking"))],
                source_is_not_formally_approved_by_this_request=True)


def _relational_rule(node, values, *, probe=False):
    """Execute bounded typed relations, not labels or natural-language guessing.

    The symbolic inputs are Codex extractions and still need semantic review.
    Unsupported reasoning is unresolved, never silently converted to arithmetic.
    """
    rule = node.get("rule")
    _require(isinstance(rule, dict) and "expected_result" in rule, "high-order tag without executable relational rule")
    typ = rule.get("type")
    _require(all(isinstance(v, dict) for v in values), "relational rule requires typed symbolic inputs")
    if typ == "precedence_closure":
        _require(node["kind"] in ("constraint_chain", "composition", "constraint_apply"), "rule/kind mismatch")
        direct = set()
        for value in values:
            for edge in _list(value.get("precedence_edges"), "precedence edges", True):
                _require(isinstance(edge, list) and len(edge) == 2 and all(v1.text(x) for x in edge) and edge[0] != edge[1], "invalid precedence edge")
                direct.add(tuple(edge))
        closure = set(direct)
        while True:
            expanded = closure | {(a, d) for a, b in closure for c, d in closure if b == c}
            _require(not any(a == b for a, b in expanded), "cyclic precedence constraints")
            if expanded == closure:
                break
            _require(len(expanded) <= 4096, "precedence closure bound exceeded")
            closure = expanded
        vertices = sorted({x for edge in closure for x in edge})
        order, remaining = [], set(vertices)
        while remaining:
            ready = sorted(v for v in remaining if not any(b == v and a in remaining for a, b in closure))
            _require(len(ready) == 1, "precedence constraints do not define a unique order")
            order.append(ready[0]); remaining.remove(ready[0])
        _require(probe or bool(closure - direct), "no new relational conclusion; explicit order retrieval is not L3")
        result = dict(precedence_edges=[list(e) for e in sorted(closure)], order=order)
    elif typ == "guarded_transition":
        _require(node["kind"] in ("state_update", "plan_revision", "causal_chain", "composition"), "rule/kind mismatch")
        states = [v["state"] for v in values if "state" in v]
        transitions = [v["transition"] for v in values if "transition" in v]
        _require(len(values) == 2 and len(states) == len(transitions) == 1, "guarded transition requires one initial state and one conditional update")
        state, transition = states[0], transitions[0]
        _require(v1.text(state.get("event_id")) and state["event_id"] == transition.get("event_id"), "transition cannot merge independent events")
        before, guard, update = state.get("values"), transition.get("when"), transition.get("set")
        _require(all(isinstance(x, dict) and x for x in (before, guard, update)), "nonempty state, guard and update required")
        _require(all(k in before and _same(before[k], v) for k, v in guard.items()), "state does not satisfy explicit transition guard")
        _require(any(k not in before or not _same(before[k], v) for k, v in update.items()), "no state change")
        result = dict(state=dict(event_id=state["event_id"], values={**before, **update}))
    elif typ == "constraint_filter":
        _require(node["kind"] in ("constraint_apply", "role_resolution", "object_resolution", "composition"), "rule/kind mismatch")
        domains, constraints = {}, []
        for value in values:
            for key, items in value.get("domains", {}).items():
                _require(v1.text(key), "constraint variable required")
                _list(items, "variable domain", True)
                _require(all(isinstance(i, (str, int, bool)) for i in items), "finite scalar domains required")
                _require(len({fp(x) for x in items}) == len(items), "duplicate typed domain value")
                domains[key] = [x for x in domains.get(key, items) if any(_same(x,y) for y in items)]
            constraints += value.get("constraints", [])
        _require(len(domains) >= 2 and constraints, "scalar arithmetic/range lookup is not a coupled constraint")
        coupled = False
        for constraint in constraints:
            _require(isinstance(constraint, dict) and set(constraint) in ({"if", "then"}, {"not"}), "constraint must be if/then or not")
            for mapping in constraint.values():
                _require(isinstance(mapping, dict) and mapping and all(k in domains for k in mapping), "constraint references unknown variable")
                _require(all(any(_same(v,x) for x in domains[k]) for k, v in mapping.items()), "constraint value outside declared domain")
            if "if" in constraint:
                coupled |= bool(set(constraint["if"]) != set(constraint["then"]))
        _require(coupled, "independent unary constraints are not higher-order coupling")
        count = 1
        keys = sorted(domains)
        for k in keys:
            count *= len(domains[k])
        _require(1 < count <= 256, "unconstrained assignment already unique, empty domain or search bound exceeded")
        solutions = []
        def matches(assignment, condition):
            return all(_same(assignment[k], v) for k, v in condition.items())
        unconstrained = []
        for items in itertools.product(*(domains[k] for k in keys)):
            assignment = dict(zip(keys, items))
            unconstrained.append(assignment)
            if all(not matches(assignment, c["not"]) if "not" in c else
                   not matches(assignment, c["if"]) or matches(assignment, c["then"]) for c in constraints):
                solutions.append(assignment)
        _require(len(solutions) == 1, "constraints must imply a unique assignment")
        unary_only = [c for c in constraints if "not" in c or set(c["if"]) == set(c["then"])]
        unary_solutions = [a for a in unconstrained if all(not matches(a, c["not"]) if "not" in c else
                           not matches(a, c["if"]) or matches(a, c["then"]) for c in unary_only)]
        _require(len(unary_solutions) > 1, "cross-variable coupling is redundant; unary lookup is not L3")
        result = dict(domains=domains, constraints=constraints, solutions=solutions)
    else:
        raise ValidationError("unsupported relational rule; retain candidate UNRESOLVED")
    if not probe:
        _require(_same(result, rule["expected_result"]), "computed relational result disagrees with declared conclusion")
        def conclusion(value):
            return value.get("order", value.get("state", value.get("solutions")))
        for i in range(len(values)):
            try:
                reduced = _relational_rule(node, values[:i] + values[i + 1:], probe=True)
            except (ValidationError, KeyError, TypeError, ValueError):
                continue
            _require(not _same(conclusion(reduced), conclusion(result)),
                     "redundant relational input; record alternative sufficient path instead of inflating joint dependency")
    return result


def _graph_analysis(graph, source):
    index = source_index(source)
    _require(isinstance(graph, dict), "dependency_graph required")
    nodes = _list(graph.get("nodes"), "graph nodes", True)
    paths, parents, by_id, values = {}, {}, {}, {}
    for node in nodes:
        _require(isinstance(node, dict), "graph node object required")
        nid = node.get("node_id")
        _require(v1.text(nid) and nid not in by_id and v1.text(node.get("claim")), "unique node_id and claim required")
        typ = node.get("type")
        if typ == "fact":
            _require(v1.text(node.get("event_id")), "fact event_id required")
            supports = _list(node.get("supports"), "fact supports", True)
            routes = []
            for support in supports:
                refs = _refs(support.get("turn_refs"), index, "fact support")
                _require(refs == _quotes(support.get("evidence"), source, "fact support"), "fact support quote/ref drift")
                _require(v1.text(support.get("entailment_rationale")), "support entailment rationale required")
                routes.append((frozenset(refs), frozenset(), frozenset(refs)))
            paths[nid], parents[nid] = routes, set()
            values[nid] = copy.deepcopy(node.get("symbolic_value"))
            if isinstance(values[nid], dict):
                for state_type in ("state", "transition"):
                    if state_type in values[nid]:
                        _require(isinstance(values[nid][state_type], dict) and
                                 values[nid][state_type].get("event_id") == node["event_id"],
                                 "symbolic state/transition event differs from source-bound fact event")
        elif typ == "operation":
            kind = node.get("kind")
            _require(kind in LOW_OPS | REL_OPS and v1.text(node.get("rationale")), "typed operation and rationale required")
            inputs = _list(node.get("inputs"), "operation inputs", True)
            alternatives = node.get("input_alternatives", [inputs])
            _list(alternatives, "input alternatives", True)
            _require(all(isinstance(i, str) and i in by_id for i in inputs), "inputs must be preceding nodes")
            _require(inputs in alternatives, "primary inputs must be one declared alternative")
            par, routes = set(), []
            for branch in alternatives:
                _list(branch, "input branch", True)
                _require(all(isinstance(i, str) and i in by_id for i in branch) and len(set(branch)) == len(branch), "invalid input alternative")
                par.update(branch)
                count = 1
                for key in branch:
                    count *= len(paths[key])
                _require(count <= 4096, "support product limit; HOLD without truncation")
                for combination in itertools.product(*(paths[key] for key in branch)):
                    refs = frozenset().union(*(r[0] for r in combination))
                    ops = frozenset({nid}).union(*(r[1] for r in combination))
                    fact_refs = frozenset().union(*(r[2] for r in combination))
                    routes.append((refs, ops, fact_refs))
            if kind in REL_OPS:
                bridge = node.get("bridge", {})
                _require(isinstance(bridge, dict), "relational bridge required")
                _require(bridge.get("identity_basis") in ("explicit_reference", "explicit_constraint"), "similar actors/time are not an identity bridge")
                _require(v1.text(bridge.get("rationale")) and v1.text(bridge.get("change_or_constraint")), "bridge must explain state change or coupled constraint")
                _list(bridge.get("event_ids"), "bridge events", True)
                bridge_refs = _quotes(bridge.get("identity_evidence"), source, "identity bridge")
                routes = [(r | frozenset(bridge_refs), o, f) for r, o, f in routes]
                _require(all(len(branch) >= 2 for branch in alternatives), "relational operation needs jointly used inputs")
                _require(all(by_id[i]["type"] == "fact" or by_id[i].get("kind") in REL_OPS
                             for branch in alternatives for i in branch),
                         "unexecuted low-level output cannot manufacture a typed relational premise")
                results = [_relational_rule(node, [values[i] for i in branch]) for branch in alternatives]
                _require(all(_same(v,results[0]) for v in results), "alternative relations disagree on result")
                values[nid] = results[0]
            else:
                values[nid] = copy.deepcopy(node.get("symbolic_value"))
            paths[nid], parents[nid] = list(set(routes)), par
        else:
            raise ValidationError("unknown graph node type")
        _require(len(paths[nid]) <= 512, "support path limit; HOLD without truncation")
        by_id[nid] = node
    answer = graph.get("answer_node")
    _require(isinstance(answer, str) and answer in paths, "answer_node unknown")
    ancestors = set()
    def visit(nid):
        if nid not in ancestors:
            ancestors.add(nid)
            for i in parents[nid]:
                visit(i)
    visit(answer)
    _require(ancestors == set(by_id), "unused graph facts/operations cannot inflate difficulty")
    if any(n.get("kind") in REL_OPS for n in nodes):
        _require(by_id[answer].get("kind") in REL_OPS,
                 "hard graph must end at an executed relational conclusion; use TARGET answer_projection")
    def preceding(nid):
        found = set()
        todo = list(parents[nid])
        while todo:
            x = todo.pop()
            if x not in found:
                found.add(x)
                todo += list(parents[x])
        return found
    route_rows = []
    for refs, operations, fact_refs in paths[answer]:
        scenes = {index[r]["scene_id"] for r in refs}
        complex_ops = []
        for oid in operations:
            op = by_id[oid]
            if op.get("kind") not in REL_OPS:
                continue
            # Check the specific route, not the union of all support alternatives.
            op_routes = [r for r in paths[oid] if r[0] <= refs and r[1] <= operations and r[2] <= fact_refs]
            if not op_routes or any(len({index[x]["scene_id"] for x in r[2]}) < 2 or
                                    "UNKNOWN_SCENE" in {index[x]["scene_id"] for x in r[2]} for r in op_routes):
                continue
            event_ids = {by_id[x]["event_id"] for x in preceding(oid) if by_id[x]["type"] == "fact"}
            _require(set(op["bridge"]["event_ids"]) == event_ids, "bridge event binding drift")
            complex_ops.append(oid)
        coupled = any(a in preceding(b) for a in complex_ops for b in complex_ops if a != b)
        kinds = {by_id[o].get("kind") for o in operations}
        if complex_ops and len(scenes) >= 2:
            level = "L4" if len(complex_ops) >= 2 and coupled and len(scenes) >= 3 else "L3"
        elif "arithmetic" in kinds:
            level = "L1"  # cross-scene numbers alone are still simple arithmetic
        elif len(refs) == 1:
            level = "L0"
        else:
            level = "L2"
        route_rows.append(dict(turn_refs=sorted(refs), operation_ids=sorted(operations),
                               scene_ids=sorted(scenes), relational_operations=sorted(complex_ops),
                               coupled_relations=coupled, level=level))
    route_rows.sort(key=lambda r: (LEVELS.index(r["level"]), len(r["turn_refs"]), r["turn_refs"], r["operation_ids"]))
    minimal = v2._minimal([r["turn_refs"] for r in route_rows])
    return dict(routes=route_rows, minimal_support_sets=[sorted(s) for s in minimal],
                symbolic_answer=values[answer], symbolic_node_values=values,
                level=route_rows[0]["level"],
                turn_deletions=[dict(removed_turn_ref=r, remaining_paths=[sorted(p) for p in minimal if r not in p]) for r in sorted(set().union(*minimal))],
                scene_deletions=[dict(removed_scene=s, remaining_paths=[sorted(p) for p in minimal if all(index[r]["scene_id"] != s for r in p)]) for s in sorted({index[r]["scene_id"] for p in minimal for r in p})],
                semantic_completeness_proved=False)


def _path_audit(audit, source):
    index = source_index(source)
    _require(isinstance(audit, dict), "path_audit required")
    _require(audit.get("semantic_completeness_proved") is False, "must not claim exhaustive semantic proof")
    _full_scan(audit.get("source_scan"), index, "counterpath visible-source scan")
    _require(v1.text(audit.get("natural_value_rationale")), "natural value rationale required")
    rows = _list(audit.get("counterpaths"), "counterpaths")
    for row in rows:
        _require(isinstance(row, dict) and row.get("status") in ("sufficient", "excluded", "unresolved") and v1.text(row.get("rationale")), "invalid counterpath")
        _refs(row.get("turn_refs"), index, "counterpath")
        if row["status"] == "sufficient":
            _require(row.get("level") in LEVELS, "sufficient counterpath needs difficulty level")
        elif row["status"] == "excluded":
            _quotes(row.get("exclusion_evidence"), source, "counterpath exclusion")
    return rows


def _formal_scope_gate(source, source_scope, refs, author_id=None, require_receipt=False):
    """Reuse admission validation; never create or approve material records."""
    gate = v21.formal_evidence_gate(source_scope or {}, refs)
    errors = list(gate["errors"])
    scope = source_scope or {}
    binding = scope.get("source_binding")
    if binding:
        valid, error = v21._verify_source_binding(source, binding)
        if not valid:
            errors.append(error)
        expected_sha = binding.get("sha256")
    else:
        expected_sha = fp(source)
    if scope.get("source_sha256") != expected_sha:
        errors.append("SOURCE_SCOPE_VERSION_MISMATCH")
    if scope.get("root_id", source["root_id"]) != source["root_id"]:
        errors.append("SOURCE_SCOPE_ROOT_MISMATCH")
    rows = scope.get("turns", [])
    ids = [row.get("turn_id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(set(ids)) or set(ids) != {t["turn_ref"] for t in source["turns"]}:
        errors.append("SOURCE_SCOPE_COVERAGE_MISMATCH")
    if require_receipt:
        admission, error = v21._read_verified_json_binding(scope.get("admission_binding"))
        if error:
            errors.append(error)
        else:
            rebuilt = v21.source_manifest(source, binding, admission,
                                           scope.get("admission_binding"), author_id)
            if rebuilt != scope:
                errors.append("SOURCE_SCOPE_REPLAY_MISMATCH")
    return dict(eligible=not errors, errors=sorted(set(errors)),
                ineligible_turn_refs=gate["ineligible_turn_refs"])


def classify_relation(card, source, source_scope=None):
    """Conservative structural classification of a submitted semantic ledger.

    Classification never verifies that prose claims follow from quotations.
    Counterpaths include actual visible context, irrespective of evidence license.
    """
    try:
        _require(isinstance(card, dict), "relation card required")
        if card.get("representation_status") == "UNSUPPORTED":
            _require(card.get("decision") in ("HOLD", "REJECT") and v1.text(card.get("representation_reason")),
                     "unsupported relationship requires HOLD/REJECT and diagnostic reason")
            return dict(level="UNRESOLVED", eligible_for_hard=False, formal_qa_allowed=False,
                        status="UNSUPPORTED_RELATION_RETAINED", errors=[],
                        reason=card["representation_reason"], semantic_quality_proved=False,
                        empirical_difficulty_measured=False)
        _require(card.get("representation_status", "MODELED") == "MODELED", "unknown representation_status")
        graph = _graph_analysis(card.get("dependency_graph"), source)
        alternatives = _path_audit(card.get("path_audit"), source)
        levels = [graph["level"]] + [x["level"] for x in alternatives if x["status"] == "sufficient"]
        level = min(levels, key=LEVELS.index)
        unresolved = [x for x in alternatives if x["status"] == "unresolved"]
        unmodeled = [x for x in alternatives if x["status"] == "sufficient" and
                     not any(set(p) <= set(x["turn_refs"]) for p in graph["minimal_support_sets"])]
        natural = card.get("natural_value") == "PASS" and v1.text(card.get("information_need"))
        eligible = level in ("L3", "L4") and natural and card.get("uncertainties") == [] and not unresolved and not unmodeled
        used = {r for path in graph["routes"] for r in path["turn_refs"]}
        used.update(e["turn_ref"] for n in card["dependency_graph"]["nodes"] if n.get("type") == "operation" for e in n.get("bridge", {}).get("identity_evidence", []))
        formal = _formal_scope_gate(source, source_scope, sorted(used))
        return dict(level=level, eligible_for_hard=eligible, status="HARD_TARGET_PROPOSAL" if eligible else "NOT_A_SAFE_HARD_TARGET",
                    errors=[], graph_audit=graph, sufficient_counterpaths=[x for x in alternatives if x["status"] == "sufficient"],
                    unresolved_counterpaths=unresolved, formal_evidence_gate=formal,
                    unmodeled_sufficient_counterpaths=unmodeled,
                    formal_qa_allowed=False, formal_scope_flags_eligible=eligible and formal["eligible"],
                    semantic_quality_proved=False, empirical_difficulty_measured=False,
                    limitations=["source entailment/identity/naturalness are supplied semantic judgments", "all-visible path search is not proved exhaustive", "structural difficulty is not measured Solver difficulty"])
    except (ValidationError, KeyError, TypeError, ValueError, IndexError) as exc:
        return dict(level="UNRESOLVED", eligible_for_hard=False, formal_qa_allowed=False,
                    status="INVALID_RELATION_LEDGER", errors=[str(exc)], semantic_quality_proved=False,
                    empirical_difficulty_measured=False)


def classify_legacy_target(target, source, candidate=None):
    """Read-only regression annotation. No automatic old PASS -> L3 promotion."""
    if "dependency_graph" in target and "path_audit" in target:
        return classify_relation(target, source)
    arithmetic = bool(target.get("requires_deterministic_check")) or bool((candidate or target).get("deterministic_checks"))
    refs = {e.get("turn_ref") for f in target.get("facts", []) for e in f.get("evidence", [])}
    index = source_index(source)
    scenes = sorted({index[r]["scene_id"] for r in refs if r in index})
    return dict(level="L1" if arithmetic else "UNRESOLVED", eligible_for_hard=False,
                semantic_quality_proved=False, formal_qa_allowed=False, evidence_scenes=scenes,
                reason="historical local deterministic calculation" if arithmetic else "legacy artifact has no V2.2 typed graph/counterpath assessment; representation retained, difficulty not certified")


def coverage_report(discovery, source):
    errors = []
    try:
        index = source_index(source)
        coverage = discovery.get("coverage", {})
        _require(coverage.get("claims_exhaustive") is False, "coverage must disclaim exhaustive relationship discovery")
        _require(coverage.get("passes") == list(PASSES), "all three ordered semantic passes must be recorded")
        _full_scan(discovery.get("source_scan"), index, "discovery scan")
        scenes = {r["scene_id"] for r in index.values()}
        event_ids = {e["event_id"] for e in discovery.get("events", [])}
        relation_ids = {r["relation_id"] for r in discovery.get("relations", [])}
        checks = _list(coverage.get("checks"), "coverage checks", True)
        families = set()
        represented_scenes, represented_events = set(), set()
        family_coverage = {f: dict(scenes=set(), events=set()) for f in FAMILIES}
        unchecked = []
        for row in checks:
            _require(row.get("family") in FAMILIES and row.get("status") in ("checked", "no_candidate", "not_checked"), "unknown coverage family/status")
            _require(v1.text(row.get("rationale")), "coverage rationale required")
            for key, allowed in (("scene_ids", scenes), ("event_ids", event_ids), ("relation_ids", relation_ids)):
                vals = _list(row.get(key), "coverage " + key)
                _require(all(isinstance(x, str) and x in allowed for x in vals), "coverage unknown " + key)
            refs = _list(row.get("turn_refs"), "coverage turn_refs")
            if refs:
                _refs(refs, index, "coverage")
                _require({index[r]["scene_id"] for r in refs} <= set(row["scene_ids"]), "coverage turn/scene drift")
            families.add(row["family"])
            if row["status"] == "not_checked":
                unchecked.append(row)
            else:
                represented_scenes.update(row["scene_ids"])
                represented_events.update(row["event_ids"])
                family_coverage[row["family"]]["scenes"].update(row["scene_ids"])
                family_coverage[row["family"]]["events"].update(row["event_ids"])
        _require(families == set(FAMILIES), "coverage must account for all ten relation families, including unchecked")
        missing_scenes = sorted(scenes - represented_scenes)
        missing_events = sorted(event_ids - represented_events)
        missing_by_family = [dict(family=f, scene_ids=sorted(scenes - checked["scenes"]),
                                  event_ids=sorted(event_ids - checked["events"]))
                             for f, checked in family_coverage.items()
                             if scenes - checked["scenes"] or event_ids - checked["events"]]
        uncovered = _list(coverage.get("uncovered_ranges"), "uncovered ranges")
        _require(not (missing_scenes or missing_events or missing_by_family or unchecked) or bool(uncovered), "missing/unchecked ranges must be disclosed")
        return dict(errors=[], recorded=True, operationally_complete=not (missing_scenes or missing_events or missing_by_family or unchecked or uncovered),
                    scene_count=len(scenes), event_count=len(event_ids), family_count=len(families),
                    candidate_count=len(relation_ids), selected=sum(r.get("decision") == "SELECT" for r in discovery.get("relations", [])),
                    rejected=sum(r.get("decision") == "REJECT" for r in discovery.get("relations", [])),
                    missing_scenes=missing_scenes, missing_events=missing_events, missing_by_family=missing_by_family, unchecked=unchecked,
                    uncovered_ranges=uncovered, semantic_exhaustiveness_proved=False)
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return dict(errors=errors, recorded=False, operationally_complete=False, semantic_exhaustiveness_proved=False)


def validate_discovery(discovery, source, source_scope=None):
    errors = []
    try:
        _require(discovery.get("schema_version") == VERSION + ":discovery", "discovery schema version")
        _require(discovery.get("status") in ("READY", "NO_SAFE_L3_L4"), "discovery status required")
        _require(discovery.get("root_id") == source["root_id"], "discovery source root drift")
        _require(discovery.get("source_sha256") == fp(source), "discovery source binding drift")
        _require(discovery.get("source_scope_sha256") == fp(source_scope or {}), "discovery source-scope binding drift")
        # Do not let the legacy NO_SAFE shortcut skip inventory or low-level diagnostics.
        legacy = copy.deepcopy(discovery)
        legacy["status"] = "DISCOVERED"
        errors += v2.validate_discovery(legacy, source)
        coverage = coverage_report(discovery, source)
        errors += coverage["errors"]
        ids = {e["event_id"] for e in discovery.get("events", [])}
        for row in discovery.get("relations", []):
            result = classify_relation(row, source, source_scope)
            errors += [row["relation_id"] + ": " + e for e in result["errors"]]
            for n in (row.get("dependency_graph", {}).get("nodes", []) if row.get("representation_status") != "UNSUPPORTED" else []):
                if n.get("type") == "fact" and n.get("event_id") not in ids:
                    errors.append("graph fact unknown event")
            if row.get("decision") == "SELECT" and not result["eligible_for_hard"]:
                errors.append(row["relation_id"] + ": SELECT requires a safe structural L3/L4")
        if discovery.get("status") == "NO_SAFE_L3_L4":
            _require(v1.text(discovery.get("reason")), "NO_SAFE_L3_L4 reason required")
            _require(not any(classify_relation(r, source, source_scope)["eligible_for_hard"] and r.get("decision") == "SELECT" for r in discovery.get("relations", [])), "NO_SAFE conflicts with an available hard proposal")
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return dict(errors=errors, mechanical_pass=not errors, coverage=coverage_report(discovery, source),
                semantic_quality_proved=False)


def rank_relations(discovery, source, source_scope=None):
    validation = validate_discovery(discovery, source, source_scope)
    rows = []
    for card in discovery.get("relations", []):
        assessment = classify_relation(card, source, source_scope)
        available = assessment["eligible_for_hard"] and card.get("decision") == "SELECT"
        rows.append(dict(relation_id=card["relation_id"], author_decision=card.get("decision"),
                         eligible=available, assessment=assessment,
                         selection_reason="natural source-bound hard relationship proposal" if available else
                         (card.get("decision_rationale") or assessment["status"])))
    # No forced preference of L4 over a more reliable/natural L3. Stable ids break ties.
    rows.sort(key=lambda r: (not r["eligible"], r["relation_id"]))
    operational = validation["mechanical_pass"] and validation["coverage"]["operationally_complete"]
    selected = [r["relation_id"] for r in rows if r["eligible"]] if operational else []
    return dict(schema_version=VERSION + ":selection", discovery_sha256=v2.fp(discovery),
                source_sha256=fp(source), source_scope_sha256=fp(source_scope or {}),
                status="HARD_RELATION_POOL" if selected else ("NO_SAFE_L3_L4" if operational else "INCOMPLETE_DISCOVERY_HOLD"),
                eligible_relation_ids=selected, relations=rows, validation=validation,
                semantic_quality_proved=False, formal_generation_authorized=False)


def _answer_projection(symbolic_answer, projection):
    _require(isinstance(projection, dict) and "expected_answer" in projection,
             "TARGET answer_projection with expected_answer required")
    path = _list(projection.get("path"), "answer projection path")
    value = symbolic_answer
    for key in path:
        if isinstance(value, dict):
            _require(isinstance(key, str) and key in value, "answer projection unknown object key")
        elif isinstance(value, list):
            _require(type(key) is int and 0 <= key < len(value), "answer projection invalid array index")
        else:
            raise ValidationError("answer projection traverses a scalar")
        value = value[key]
    _require(fp(value) == fp(projection["expected_answer"]), "computed answer projection disagrees with Gold")
    return value


def _projected_dependency(graph, audit, projection):
    """Reject typed operations irrelevant to the frozen query within this subset.

    Natural-language query equivalence is still an independent semantic task.
    Full order/assignment projections and backwards-live guarded-state fields
    are supported; other projections are retained as unresolved proposals.
    """
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    answer = nodes[graph["answer_node"]]
    path = projection["path"]
    typ = answer["rule"]["type"]
    if typ == "precedence_closure":
        _require(path == ["order"], "order difficulty requires complete order projection")
    elif typ == "constraint_filter":
        _require(path == ["solutions"], "constraint difficulty requires complete assignment projection")
    elif typ == "guarded_transition":
        _require(path[:2] == ["state", "values"], "state query must project state values")
        final_values = audit["symbolic_answer"]["state"]["values"]
        live = {path[2]} if len(path) > 2 else set(final_values)
        values = audit["symbolic_node_values"]
        def walk(nid, needed):
            node = nodes[nid]
            if node["type"] == "fact":
                return
            _require(node.get("rule", {}).get("type") == "guarded_transition", "unsupported mixed state projection")
            for branch in node.get("input_alternatives", [node["inputs"]]):
                state_ids = [i for i in branch if "state" in values[i]]
                transition_ids = [i for i in branch if "transition" in values[i]]
                _require(len(state_ids) == len(transition_ids) == 1, "state projection branch mismatch")
                before = values[state_ids[0]]["state"]["values"]
                transition = values[transition_ids[0]]["transition"]
                changed = {k for k, v in transition["set"].items() if k not in before or not _same(before[k],v)}
                _require(bool(needed & changed), "state operation does not affect the frozen answer projection")
                prior_needed = (needed - set(transition["set"])) | set(transition["when"])
                walk(state_ids[0], prior_needed)
        walk(graph["answer_node"], live)
    else:
        raise ValidationError("unsupported difficulty projection")


def validate_target(target, discovery, source, source_scope=None, selection=None):
    errors = []
    selection = selection or rank_relations(discovery, source, source_scope)
    try:
        current = rank_relations(discovery, source, source_scope)
        _require(selection == current, "selection receipt drift")
        rid = target.get("relation_id")
        _require(rid in current["eligible_relation_ids"], "TARGET_GATE: no safe L3/L4 before answer freeze")
        card = next(c for c in discovery["relations"] if c["relation_id"] == rid)
        _require(target.get("dependency_graph") == card["dependency_graph"], "target graph drift from ranked relationship")
        _require(target.get("path_audit") == card["path_audit"], "target counterpath audit drift")
        _require(target.get("selection_sha256") == fp(selection), "target selection binding drift")
        graph_audit = classify_relation(card, source, source_scope)["graph_audit"]
        projected = _answer_projection(graph_audit["symbolic_answer"], target.get("answer_projection"))
        _require(fp(projected) == fp(target.get("expected_answer")), "TARGET Gold differs from executed relation answer")
        _projected_dependency(card["dependency_graph"], graph_audit, target["answer_projection"])
        audit = v2.validate_target(target, discovery, source, source_scope or v21.source_manifest(source))
        errors += audit["errors"]
        graph_facts = {n["node_id"]: n for n in card["dependency_graph"]["nodes"] if n["type"] == "fact"}
        _require(set(graph_facts) == {f["fact_id"] for f in target.get("facts", [])}, "target fact/graph drift")
        for fact in target["facts"]:
            node = graph_facts[fact["fact_id"]]
            _require(node["claim"] == fact["claim"] and node["supports"] == fact["supports"] and fact["event_ids"] == [node["event_id"]], "target fact semantics drift")
        graph_ops = [n for n in card["dependency_graph"]["nodes"] if n["type"] == "operation"]
        _require(len(graph_ops) == len(target["inference_steps"]), "target operation count drift")
        for node, step in zip(graph_ops, target["inference_steps"]):
            _require(node["node_id"] == step["step_id"] and node["inputs"] == step["inputs"] and node["claim"] == step["conclusion"] and node.get("input_alternatives") == step.get("input_alternatives"), "target inference graph drift")
        _require({frozenset(p) for p in target["evidence_certificate"]["all_minimal_support_sets"]} ==
                 {frozenset(p) for p in graph_audit["minimal_support_sets"]},
                 "TARGET certificate must include relational bridge and all graph support paths")
    except (ValidationError, KeyError, TypeError, ValueError, StopIteration) as exc:
        errors.append(str(exc))
    return dict(errors=errors, mechanical_pass=not errors, selection=selection,
                semantic_quality_proved=False, formal_qa_allowed=False)


def freeze_target(path, target, discovery, source, source_scope=None):
    audit = validate_target(target, discovery, source, source_scope)
    _require(audit["mechanical_pass"], "; ".join(audit["errors"]))
    payload = dict(version=VERSION, target=copy.deepcopy(target), target_sha256=fp(target),
                   selection_sha256=target["selection_sha256"], source_sha256=fp(source),
                   source_scope_sha256=fp(source_scope or {}), semantic_quality_proved=False)
    v1.write_new(path, payload)
    _require(json.loads(Path(path).read_text()) == payload, "target freeze readback drift")
    return payload


def validate_candidate(candidate, target, discovery, source, source_scope=None):
    audit = validate_target(target, discovery, source, source_scope)
    errors = list(audit["errors"])
    try:
        legacy = v2.validate_candidate(candidate, target, source)
        errors += legacy["errors"]
        _require(candidate.get("target_sha256") == v2.fp(target), "candidate target binding drift")
        query_audit = candidate.get("query_path_audit")
        _require(isinstance(query_audit, dict), "post-construction query_path_audit required")
        _require(query_audit.get("question_sha256") == fp(candidate["question"]) and
                 query_audit.get("options_sha256") == fp(candidate["options"]), "query audit must bind actual question and options")
        _require(query_audit.get("critical_bridges_preserved") is True and
                 query_audit.get("supplied_bridge_ids") == [], "stem/options supply or bypass critical bridge")
        rows = _path_audit(query_audit, source)
        _require(not any(r["status"] == "unresolved" or (r["status"] == "sufficient" and LEVELS.index(r["level"]) < 3) for r in rows), "constructed question has unresolved/simple sufficient path")
        _require(query_audit.get("minimum_route_level") in ("L3", "L4"), "construction difficulty not preserved")
        _require(query_audit["minimum_route_level"] == classify_relation(target, source, source_scope)["level"], "construction difficulty assertion drift")
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    formal = classify_relation(target, source, source_scope).get("formal_evidence_gate", {})
    style = v21.style_audit(candidate) if isinstance(candidate.get("options"), dict) and candidate.get("options") else {}
    if candidate.get("question_type") != "ordering" and style.get("flags"):
        errors += ["option shortcut review required: " + flag for flag in style["flags"]]
    return dict(errors=errors, mechanical_pass=not errors, option_audit=style,
                formal_evidence_gate=formal, formal_qa_allowed=False,
                formal_scope_flags_eligible=not errors and formal.get("eligible") is True,
                semantic_quality_proved=False, empirical_difficulty_measured=False)


def prepare_frozen_review(run_dir, candidate, target_freeze_path, discovery, source, source_scope, **reviewer_context):
    """Only formal authoring facade; refuses pending material and unfrozen targets."""
    from importlib import import_module
    options = import_module((__package__ + "." if __package__ else "") + "base_qa_option_freeze_v2_2")
    frozen = options._read(target_freeze_path)
    _require(isinstance(frozen, dict) and set(frozen) == {"version", "target", "target_sha256",
                 "selection_sha256", "source_sha256", "source_scope_sha256", "semantic_quality_proved"},
             "TARGET freeze envelope schema mismatch")
    target = frozen["target"]
    _require(frozen.get("version") == VERSION and frozen.get("target_sha256") == fp(target) and frozen.get("source_sha256") == fp(source) and frozen.get("source_scope_sha256") == fp(source_scope), "target freeze binding drift")
    _require(frozen["selection_sha256"] == target.get("selection_sha256") and
             frozen["semantic_quality_proved"] is False, "TARGET freeze selection/claim drift")
    audit = validate_candidate(candidate, target, discovery, source, source_scope)
    used = {r for p in classify_relation(target, source, source_scope).get("graph_audit", {}).get("minimal_support_sets", []) for r in p}
    def collect_evidence(value):
        if isinstance(value, dict):
            if "turn_ref" in value and "quote" in value:
                used.add(value["turn_ref"])
            for child in value.values():
                collect_evidence(child)
        elif isinstance(value, list):
            for child in value:
                collect_evidence(child)
    collect_evidence(target)
    collect_evidence(candidate)
    for row in candidate.get("item_analysis", {}).values():
        used.update(row.get("error_path_turn_refs", []))
    used = sorted(used)
    formal = _formal_scope_gate(source, source_scope, used, candidate.get("author_id"), require_receipt=True)
    _require(audit["mechanical_pass"] and formal["eligible"], "formal generation blocked: " + "; ".join(audit["errors"] + formal["errors"]))
    return options.freeze_bundle(run_dir, candidate, source, source_scope,
                                 extra_paths=(target_freeze_path, SCHEMA_PATH), **reviewer_context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("workplan", "discovery", "rank", "target", "candidate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = v1.read(args.source)
    scope = v1.read(args.scope) if args.scope else {}
    data = v1.read(args.input) if args.input else {}
    if args.stage == "workplan":
        result = discovery_workplan(source, scope, data)
    elif args.stage == "discovery":
        result = validate_discovery(data, source, scope)
    elif args.stage == "rank":
        result = rank_relations(data, source, scope)
    elif args.stage == "target":
        _require(args.discovery is not None, "--discovery required")
        result = freeze_target(args.output, data, v1.read(args.discovery), source, scope)
        print(json.dumps(dict(output=str(args.output), target_sha256=result["target_sha256"])))
        return 0
    else:
        _require(args.discovery is not None and args.target is not None, "--discovery and --target required")
        result = validate_candidate(data, v1.read(args.target), v1.read(args.discovery), source, scope)
    v1.write_new(args.output, result)
    print(json.dumps(dict(output=str(args.output), status=result.get("status"), errors=result.get("errors", []))))
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
