"""V2.2.1 typed answers and query-conditioned complex relation contracts.

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
import hashlib
from fractions import Fraction
from pathlib import Path

try:
    from . import base_qa_answers_v2_2_1 as answers
    from . import base_qa_pipeline_v1 as v1
    from . import base_qa_pipeline_v2 as v2
    from . import base_qa_pipeline_v2_1 as v21
except ImportError:
    import base_qa_answers_v2_2_1 as answers
    import base_qa_pipeline_v1 as v1
    import base_qa_pipeline_v2 as v2
    import base_qa_pipeline_v2_1 as v21

VERSION = "base_qa_generation_v2_2_1"
EXECUTION_PROFILE = "acceptance_difficulty_label_only_v1"
PROMPT_DIR = Path(__file__).with_name("base_qa_prompts_v2_2_1")
SCHEMA_PATH = Path(__file__).with_name("base_qa_schema_v2_2_1.json")
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
def fp(value):
    return hashlib.sha256(answers.typed_key(value).encode("utf-8")).hexdigest()
POLICY = dict(version=VERSION, challenger="Codex direct staged semantic work",
              model_calls_enabled=False, solver_enabled=False, training_enabled=False,
              formal_generation_requires_material_authorization=True,
              execution_profile=EXECUTION_PROFILE,
              difficulty_is_label_not_generation_gate=True,
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
    """Execute bounded possible-world relations; queries, not graphs, need uniqueness.

    probe is used only for deleted-premise counterfactuals. It never certifies a
    missing premise, changes the declared result or approves a natural-language
    extraction. Non-probe keeps identity, coupling and redundant-input checks.
    """
    rule = node.get("rule")
    _require(isinstance(rule, dict) and "expected_result" in rule, "high-order tag without executable relational rule")
    _require(all(isinstance(v, dict) for v in values), "relational rule requires typed symbolic inputs")
    typ = rule.get("type")
    if typ == "precedence_closure":
        _require(node["kind"] in ("constraint_chain", "composition", "constraint_apply"), "rule/kind mismatch")
        direct, explicit_vertices = set(), set()
        supplied_vertices = any("vertices" in v for v in values)
        for value in values:
            if "vertices" in value:
                vertices = _list(value["vertices"], "precedence vertices", True)
                _require(all(v1.text(v) for v in vertices) and len(set(vertices)) == len(vertices), "invalid precedence vertices")
                explicit_vertices.update(vertices)
            for edge in _list(value.get("precedence_edges", []), "precedence edges"):
                _require(isinstance(edge, list) and len(edge) == 2 and all(v1.text(x) for x in edge) and edge[0] != edge[1], "invalid precedence edge")
                direct.add(tuple(edge))
        closure = set(direct)
        while True:
            expanded = closure | {(a, d) for a, b in closure for c, d in closure if b == c}
            _require(not any(a == b for a, b in expanded), "cyclic precedence constraints")
            _require(len(expanded) <= 4096, "precedence closure bound exceeded")
            if expanded == closure:
                break
            closure = expanded
        vertices = sorted(explicit_vertices | {x for e in closure for x in e})
        _require(vertices, "precedence has no known vertices")
        order, remaining, unique = [], set(vertices), True
        while remaining:
            ready = sorted(v for v in remaining if not any(b == v and a in remaining for a, b in closure))
            _require(ready, "cyclic precedence constraints")
            unique &= len(ready) == 1
            order.append(ready[0]); remaining.remove(ready[0])
        _require(probe or bool(closure - direct), "no new relational conclusion; explicit order retrieval is not L3")
        result = dict(precedence_edges=[list(e) for e in sorted(closure)], order=order if unique else None)
        if supplied_vertices:
            result["vertices"] = vertices
    elif typ == "guarded_transition":
        _require(node["kind"] in ("state_update", "plan_revision", "causal_chain", "composition"), "rule/kind mismatch")
        states = [v["state"] for v in values if "state" in v]
        changes = [v["transition"] for v in values if "transition" in v]
        if probe and len(states) == 1 and not changes:
            return dict(state=copy.deepcopy(states[0]))
        _require(len(values) == 2 and len(states) == len(changes) == 1, "guarded transition requires one initial state and one conditional update")
        state, change = states[0], changes[0]
        _require(v1.text(state.get("event_id")) and state["event_id"] == change.get("event_id"), "transition cannot merge independent events")
        before, guard, update = state.get("values"), change.get("when"), change.get("set")
        _require(all(isinstance(x, dict) and x for x in (before, guard, update)), "nonempty state, guard and update required")
        guard_holds = all(k in before and _same(before[k], v) for k, v in guard.items())
        if probe and not guard_holds:
            return dict(state=copy.deepcopy(state))
        _require(guard_holds, "state does not satisfy explicit transition guard")
        _require(probe or any(k not in before or not _same(before[k], v) for k, v in update.items()), "no state change")
        result = dict(state=dict(event_id=state["event_id"], values={**before, **update}))
    elif typ == "constraint_filter":
        _require(node["kind"] in ("constraint_apply", "role_resolution", "object_resolution", "composition"), "rule/kind mismatch")
        domains, constraints = {}, []
        for value in values:
            for key, items in value.get("domains", {}).items():
                _require(v1.text(key), "constraint variable required")
                _list(items, "variable domain", True)
                _require(all(type(i) in (str, int, bool) for i in items), "finite scalar domains required")
                _require(len({fp(x) for x in items}) == len(items), "duplicate typed domain value")
                domains[key] = [x for x in domains.get(key, items) if any(_same(x, y) for y in items)]
            constraints += value.get("constraints", [])
        # A deleted-premise probe must still solve a single remaining variable.
        # Otherwise an unrelated coupled subproblem can make a local lookup
        # look necessary merely because the probe refuses its simpler shape.
        _require(len(domains) >= (1 if probe else 2), "scalar arithmetic/range lookup is not a coupled constraint")
        coupled = False
        for c in constraints:
            _require(isinstance(c, dict) and set(c) in ({"if", "then"}, {"not"}), "constraint must be if/then or not")
            for mapping in c.values():
                _require(isinstance(mapping, dict) and mapping and all(k in domains for k in mapping), "constraint references unknown variable")
                _require(all(any(_same(v, x) for x in domains[k]) for k, v in mapping.items()), "constraint value outside declared domain")
            if "if" in c:
                coupled |= set(c["if"]) != set(c["then"])
        _require(probe or coupled, "independent unary constraints are not higher-order coupling")
        count, keys = 1, sorted(domains)
        for k in keys:
            count *= len(domains[k])
        _require(0 < count <= 256 and (probe or count > 1), "unconstrained assignment already unique, empty domain or search bound exceeded")
        def matches(a, condition):
            return all(_same(a[k], v) for k, v in condition.items())
        def allowed(a, cs):
            return all(not matches(a, c["not"]) if "not" in c else not matches(a, c["if"]) or matches(a, c["then"]) for c in cs)
        universe = [dict(zip(keys, items)) for items in itertools.product(*(domains[k] for k in keys))]
        solutions = [a for a in universe if allowed(a, constraints)]
        _require(solutions, "inconsistent constraints: no possible assignment")
        unary = [c for c in constraints if "not" in c or set(c["if"]) == set(c["then"])]
        _require(probe or len([a for a in universe if allowed(a, unary)]) > len(solutions), "cross-variable coupling is redundant; unary lookup is not L3")
        result = dict(domains=domains, constraints=constraints, solutions=solutions)
    else:
        raise ValidationError("unsupported relational rule; retain candidate UNRESOLVED")
    if not probe:
        _require(_same(result, rule["expected_result"]), "computed relational result disagrees with declared conclusion")
        for i in range(len(values)):
            try:
                reduced = _relational_rule(node, values[:i] + values[i+1:], probe=True)
            except (ValidationError, KeyError, TypeError, ValueError):
                continue
            # Compare executable conclusions, excluding diagnostic domains.
            def conclusion(v):
                return {k: v[k] for k in ("order", "precedence_edges", "state", "solutions") if k in v}
            _require(not _same(conclusion(reduced), conclusion(result)), "redundant relational input; record alternative sufficient path instead of inflating joint dependency")
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
            return dict(level="UNRESOLVED", eligible_for_hard=False, eligible_for_candidate=False,
                        formal_qa_allowed=False,
                        status="UNSUPPORTED_RELATION_RETAINED", errors=[],
                        reason=card["representation_reason"], semantic_quality_proved=False,
                        empirical_difficulty_measured=False)
        _require(card.get("representation_status", "MODELED") == "MODELED", "unknown representation_status")
        graph = _graph_analysis(card.get("dependency_graph"), source)
        has_relations = any(n.get("kind") in REL_OPS for n in card["dependency_graph"]["nodes"])
        query_dependency = _query_dependency(card["dependency_graph"], graph, card.get("query")) if has_relations else {}
        if "semantic_answer" in card and query_dependency:
            _require(_same(answers.validate_semantic(card["semantic_answer"]), query_dependency["semantic_answer"]), "relation/query semantic answer drift")
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
        candidate_eligible = (level in LEVELS and natural and card.get("uncertainties") == []
                              and not unresolved and not unmodeled and formal.get("eligible") is True)
        return dict(level=level, eligible_for_hard=eligible, status="HARD_TARGET_PROPOSAL" if eligible else "NOT_A_SAFE_HARD_TARGET",
                    eligible_for_candidate=candidate_eligible,
                    difficulty_gate_removed=True,
                    errors=[], graph_audit=graph, query_dependency_audit=query_dependency,
                    semantic_answer=query_dependency.get("semantic_answer"), sufficient_counterpaths=[x for x in alternatives if x["status"] == "sufficient"],
                    unresolved_counterpaths=unresolved, formal_evidence_gate=formal,
                    unmodeled_sufficient_counterpaths=unmodeled,
                    formal_qa_allowed=False, formal_scope_flags_eligible=eligible and formal["eligible"],
                    semantic_quality_proved=False, empirical_difficulty_measured=False,
                    limitations=["source entailment/identity/naturalness are supplied semantic judgments", "all-visible path search is not proved exhaustive", "structural difficulty is not measured Solver difficulty"])
    except (ValidationError, KeyError, TypeError, ValueError, IndexError) as exc:
        return dict(level="UNRESOLVED", eligible_for_hard=False, eligible_for_candidate=False,
                    formal_qa_allowed=False,
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
        _require(discovery.get("status") in ("READY", "NO_SAFE_RELATION", "NO_SAFE_L3_L4"), "discovery status required")
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
            if row.get("decision") == "SELECT" and not result["eligible_for_candidate"]:
                errors.append(row["relation_id"] + ": SELECT requires a safe, source-bound candidate at its recorded difficulty")
        if discovery.get("status") == "NO_SAFE_RELATION":
            _require(v1.text(discovery.get("reason")), "NO_SAFE_RELATION reason required")
            _require(not any(classify_relation(r, source, source_scope)["eligible_for_candidate"] and r.get("decision") == "SELECT" for r in discovery.get("relations", [])), "NO_SAFE_RELATION conflicts with an available proposal")
        if discovery.get("status") == "NO_SAFE_L3_L4":
            _require(v1.text(discovery.get("reason")), "NO_SAFE_L3_L4 reason required")
            _require(not any(classify_relation(r, source, source_scope)["eligible_for_hard"] and r.get("decision") == "SELECT" for r in discovery.get("relations", [])), "NO_SAFE_L3_L4 conflicts with an available hard proposal")
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return dict(errors=errors, mechanical_pass=not errors, coverage=coverage_report(discovery, source),
                semantic_quality_proved=False)


def rank_relations(discovery, source, source_scope=None):
    validation = validate_discovery(discovery, source, source_scope)
    rows = []
    for card in discovery.get("relations", []):
        assessment = classify_relation(card, source, source_scope)
        available = assessment["eligible_for_candidate"] and card.get("decision") == "SELECT"
        rows.append(dict(relation_id=card["relation_id"], author_decision=card.get("decision"),
                         eligible=available, hard_target_eligible=assessment["eligible_for_hard"], assessment=assessment,
                         selection_reason="natural source-bound relation proposal; difficulty retained as a label" if available else
                         (card.get("decision_rationale") or assessment["status"])))
    # No forced preference of L4 over a more reliable/natural L3. Stable ids break ties.
    rows.sort(key=lambda r: (not r["eligible"], r["relation_id"]))
    operational = validation["mechanical_pass"] and validation["coverage"]["operationally_complete"]
    selected = [r["relation_id"] for r in rows if r["eligible"]] if operational else []
    hard_selected = [r["relation_id"] for r in rows if r["eligible"] and r["hard_target_eligible"]] if operational else []
    return dict(schema_version=VERSION + ":selection_difficulty_label_only_v1", discovery_sha256=v2.fp(discovery),
                source_sha256=fp(source), source_scope_sha256=fp(source_scope or {}),
                execution_profile=EXECUTION_PROFILE,
                status="GENERATION_RELATION_POOL" if selected else ("NO_SAFE_RELATION" if operational else "INCOMPLETE_DISCOVERY_HOLD"),
                eligible_relation_ids=selected, hard_relation_ids=hard_selected, relations=rows, validation=validation,
                semantic_quality_proved=False, formal_generation_authorized=False)






def validate_target(target, discovery, source, source_scope=None, selection=None):
    errors = []
    selection = selection or rank_relations(discovery, source, source_scope)
    try:
        _require(not any(k in target for k in ("expected_answer", "answer_projection", "gold", "gold_answer")), "legacy/option answer fields forbidden in semantic TARGET")
        semantic = answers.validate_semantic(target.get("semantic_answer"))
        current = rank_relations(discovery, source, source_scope)
        _require(selection == current, "selection receipt drift")
        rid = target.get("relation_id")
        _require(rid in current["eligible_relation_ids"], "TARGET_GATE: no safe L3/L4 before answer freeze")
        card = next(c for c in discovery["relations"] if c["relation_id"] == rid)
        for field in ("dependency_graph", "path_audit", "query"):
            _require(target.get(field) == card.get(field), "target " + field + " drift from ranked relationship")
        _require(target.get("selection_sha256") == fp(selection), "target selection binding drift")
        assessment = classify_relation(card, source, source_scope)
        graph_audit = assessment["graph_audit"]
        _require(_same(_query_answer(graph_audit["symbolic_answer"], target["query"]), semantic), "TARGET semantic Gold differs from executed query")
        # Reuse original event/identity/proof and necessity checks, without its
        # candidate answer-list comparison or numeric answer-index contract.
        audit = v2.validate_target(_structural_proof_view(target), discovery, source,
                                   source_scope or v21.source_manifest(source))
        errors += audit["errors"]
        graph_facts = {n["node_id"]:n for n in card["dependency_graph"]["nodes"] if n["type"] == "fact"}
        _require(set(graph_facts) == {f["fact_id"] for f in target.get("facts", [])}, "target fact/graph drift")
        for fact in target["facts"]:
            n = graph_facts[fact["fact_id"]]
            _require(n["claim"] == fact["claim"] and n["supports"] == fact["supports"] and fact["event_ids"] == [n["event_id"]], "target fact semantics drift")
        ops = [n for n in card["dependency_graph"]["nodes"] if n["type"] == "operation"]
        _require(len(ops) == len(target["inference_steps"]), "target operation count drift")
        for n,step in zip(ops,target["inference_steps"]):
            _require(n["node_id"] == step["step_id"] and n["inputs"] == step["inputs"] and n["claim"] == step["conclusion"] and n.get("input_alternatives") == step.get("input_alternatives"), "target inference graph drift")
        _require({frozenset(p) for p in target["evidence_certificate"]["all_minimal_support_sets"]} ==
                 {frozenset(p) for p in graph_audit["minimal_support_sets"]}, "TARGET certificate must include relational bridge and all graph support paths")
    except (ValidationError, KeyError, TypeError, ValueError, StopIteration, IndexError) as exc:
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
    errors, checks, style = list(audit["errors"]), [], {}
    try:
        normalized = answers.normalize_candidate(candidate)
        _require(candidate.get("root_id") == source["root_id"], "candidate/source root mismatch")
        _require(candidate.get("author_id") == target.get("author_id"), "target/candidate author drift")
        _require(candidate.get("target_sha256") == fp(target) and candidate.get("relation_sha256") == fp(target), "candidate target binding drift")
        answer = answers.validate_candidate_answer(candidate, target["semantic_answer"])
        _require(len(set(candidate["options"].values())) == len(candidate["options"]), "duplicate option text")
        kind, gold = candidate["question_type"], answer["canonical_gold"]
        selected = ({gold["option_id"]} if kind == "single_choice" else set(gold["option_ids"]) if kind in ("multi_select","ordering") else set(candidate["options"]))
        if kind == "multi_select":
            _require(len(selected) < len(candidate["options"]), "multi-select needs supported and unsupported alternatives; no quota")
        by_ref = {t["turn_ref"]:t for t in source["turns"]}
        for key,row in candidate["item_analysis"].items():
            evidence_errors = []
            v1._evidence(row.get("evidence"), by_ref, evidence_errors, key)
            _require(not evidence_errors, "; ".join(evidence_errors))
            _require(v1.text(row.get("rationale")), "choice rationale required")
            if kind in ("single_choice","multi_select"):
                _require(row.get("status") in ("supported","contradicted","insufficient") and
                         (row["status"] == "supported") == (key in selected), "choice truth/Gold conflict")
                if key not in selected:
                    _require(v1.text(row.get("error_reason")) and v1.text(row.get("error_path_type")), "concrete distractor error path required")
                    _refs(row.get("error_path_turn_refs"), by_ref, "distractor error path")
        if kind in ("ordering","partial_order"):
            edges = _list(candidate.get("ordering_edges"), "source-bound ordering edges", True)
            actual_edges = set()
            for e in edges:
                _require(isinstance(e,dict) and e.get("before") in candidate["options"] and e.get("after") in candidate["options"], "ordering edge has unknown option IDs")
                _quotes(e.get("evidence"), source, "ordering edge")
                actual_edges.add((e["before"],e["after"]))
            closed = answers.semantic_answer("relation_set", [list(e) for e in sorted(actual_edges)])["value"]
            if kind == "partial_order":
                _require(_same(closed, gold["pairs"]), "partial-order evidence differs from Gold closure")
            else:
                ids = gold["option_ids"]
                expected = [[a,b] for i,a in enumerate(ids) for b in ids[i+1:]]
                _require(_same(closed, answers.semantic_answer("relation_set",expected)["value"]), "source edges must define unique full order equal to Gold")
        checks = _deterministic_audit(candidate, target)
        query_audit = candidate.get("query_path_audit")
        _require(isinstance(query_audit,dict), "post-construction query_path_audit required")
        _require(query_audit.get("question_sha256") == fp(candidate["question"]) and query_audit.get("options_sha256") == fp(candidate["options"]) and
                 query_audit.get("form_sha256") == candidate_form_fingerprint(candidate), "query audit must bind actual type/question/options/semantic values")
        _require(query_audit.get("critical_bridges_preserved") is True and query_audit.get("supplied_bridge_ids") == [], "stem/options supply or bypass critical bridge")
        rows = _path_audit(query_audit,source)
        _require(not any(r["status"] == "unresolved" for r in rows), "constructed question has unresolved sufficient path")
        assessment = classify_relation(target,source,source_scope)
        _require(query_audit.get("minimum_route_level") in LEVELS and query_audit["minimum_route_level"] == assessment["level"], "construction difficulty label drift")
        _require(not any(r["status"] == "sufficient" and not any(set(p) <= set(r["turn_refs"]) for p in assessment["graph_audit"]["minimal_support_sets"]) for r in rows), "constructed sufficient path must be modeled")
        view = copy.deepcopy(normalized)
        view["question_type"] = "ordering" if kind == "partial_order" else kind
        view["gold"] = list(selected)
        style = v21.style_audit(view)
        if kind not in ("ordering","partial_order"):
            errors += ["option shortcut review required: " + flag for flag in style.get("flags",[])]
    except (ValidationError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        errors.append(str(exc))
    formal = classify_relation(target,source,source_scope).get("formal_evidence_gate",{})
    return dict(errors=errors, mechanical_pass=not errors, deterministic_checks=checks,
                option_audit=style, formal_evidence_gate=formal, formal_qa_allowed=False,
                formal_scope_flags_eligible=not errors and formal.get("eligible") is True,
                semantic_quality_proved=False, empirical_difficulty_measured=False)


def prepare_frozen_review(run_dir, candidate, target_freeze_path, discovery, source, source_scope, **reviewer_context):
    """Only formal authoring facade; refuses pending material and unfrozen targets."""
    from importlib import import_module
    options = import_module((__package__ + "." if __package__ else "") + "base_qa_option_freeze_v2_2_1")
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




def _query_answer(symbolic_answer, query):
    """Evaluate an explicit question over all represented possible worlds.

    Absence of a precedence proof is unknown, not false. Existential/universal
    plan queries use explicit quantifiers; a fact-valued local query must have
    exactly one value across every feasible assignment.
    """
    _require(isinstance(query, dict) and v1.text(query.get("type")), "explicit query required")
    _require(isinstance(symbolic_answer, dict), "query needs typed symbolic result")
    typ = query["type"]
    shapes = {
        "full_order": ({"type"}, set()),
        "precedence": ({"type", "before", "after"}, set()),
        "partial_order": ({"type", "items"}, set()),
        "assignment": ({"type"}, set()),
        "assignment_field": ({"type", "field"}, {"answer_kind"}),
        "state_field": ({"type", "path"}, {"answer_kind"}),
        "entailed_choices": ({"type", "judgments"}, set()),
        "feasible_plans": ({"type", "plans", "quantifier", "require_unique"}, set()),
    }
    _require(typ in shapes, "unsupported query type")
    required, optional = shapes[typ]
    _require(required <= set(query) and set(query) <= required | optional, "query exact fields required")
    def typed(value, kind=None):
        if kind is None:
            kind = "object" if isinstance(value, dict) else "sequence" if isinstance(value, list) else "scalar"
        try:
            return answers.semantic_answer(kind, value)
        except ValueError as e:
            raise ValidationError(str(e)) from e
    def only(values):
        unique = {fp(v): v for v in values}
        _require(len(unique) == 1, "query answer ambiguous across possible worlds")
        return next(iter(unique.values()))
    if typ == "full_order":
        order = symbolic_answer.get("order")
        _require(isinstance(order, list) and order, "full order is not uniquely determined")
        return typed(order, "sequence")
    if typ in ("precedence", "partial_order"):
        pairs = symbolic_answer.get("precedence_edges")
        _list(pairs, "query precedence edges")
        vertices = set(symbolic_answer.get("vertices", [])) | {v for e in pairs for v in e}
        closure = {tuple(e) for e in pairs}
        if typ == "precedence":
            a, b = query.get("before"), query.get("after")
            _require(isinstance(a, str) and isinstance(b, str) and a != b and {a,b} <= vertices, "precedence query references unknown/same items")
            _require((a,b) in closure or (b,a) in closure, "precedence unknown: neither direction is entailed")
            return typed((a,b) in closure, "scalar")
        items = _list(query.get("items"), "partial-order query items", True)
        _require(len(items) >= 2 and all(isinstance(x,str) for x in items) and len(set(items)) == len(items) and set(items) <= vertices, "partial-order query has invalid/unknown items")
        return typed([list(e) for e in sorted(closure) if set(e) <= set(items)], "relation_set")
    if typ == "state_field":
        path = _list(query.get("path"), "state query path", True)
        _require(path[:2] == ["state", "values"], "state query must address state values")
        value = symbolic_answer
        for key in path:
            _require(isinstance(value, dict) and isinstance(key,str) and key in value, "unknown state query field")
            value = value[key]
        return typed(value, query.get("answer_kind"))
    _require(typ in ("assignment", "assignment_field", "entailed_choices", "feasible_plans"), "unsupported query type")
    solutions = _list(symbolic_answer.get("solutions"), "query possible assignments", True)
    _require(all(isinstance(a, dict) and a for a in solutions), "invalid possible assignments")
    if typ == "assignment":
        return typed(only(solutions), "object")
    if typ == "assignment_field":
        field = query.get("field")
        _require(v1.text(field) and all(field in a for a in solutions), "unknown assignment query field")
        return typed(only([a[field] for a in solutions]), query.get("answer_kind"))
    rows = _list(query.get("judgments" if typ == "entailed_choices" else "plans"), "query judgments/plans", True)
    ids, satisfied = [], []
    for row in rows:
        _require(isinstance(row, dict) and v1.text(row.get("id")) and row["id"] not in ids, "unique judgment/plan ID required")
        fields = {"id", "where", "quantifier"} if typ == "entailed_choices" else {"id", "requirements"}
        _require(set(row) == fields, "query judgment/plan exact fields required")
        ids.append(row["id"])
        conditions = row.get("where" if typ == "entailed_choices" else "requirements")
        _require(isinstance(conditions, dict) and conditions and all(all(k in a for k in conditions) for a in solutions), "query has unknown/empty conditions")
        quantifier = row.get("quantifier") if typ == "entailed_choices" else query.get("quantifier")
        _require(quantifier in ("all", "any"), "explicit all/any query quantifier required")
        truths = [all(_same(a[k], v) for k,v in conditions.items()) for a in solutions]
        if (all(truths) if quantifier == "all" else any(truths)):
            satisfied.append(row["id"])
    if typ == "feasible_plans" and query.get("require_unique") is True:
        _require(len(satisfied) == 1, "plan selection is not unique")
        return typed(satisfied[0], "scalar")
    if typ == "feasible_plans":
        _require(type(query.get("require_unique")) is bool, "explicit plan uniqueness policy required")
    return typed(satisfied, "set")

def _query_dependency(graph, graph_audit, query):
    """Check each declared AND/OR branch against the actual query, not full output.

    This is a bounded symbolic deletion test, not proof that natural-language
    counterpath search is exhaustive. Alternative source support sets remain in
    the preserved evidence/path audits rather than being silently dropped.
    """
    by_id = {n["node_id"]: n for n in graph["nodes"]}
    expected = _query_answer(graph_audit["symbolic_answer"], query)
    cache = {}
    def branches(nid):
        if nid in cache:
            return cache[nid]
        n = by_id[nid]
        if n["type"] == "fact":
            result = [({nid}, {})]
        else:
            result = []
            for branch in n.get("input_alternatives", [n["inputs"]]):
                choices = [branches(i) for i in branch]
                count = 1
                for items in choices:
                    count *= len(items)
                _require(count <= 512, "query branch product limit; HOLD without truncation")
                for combo in itertools.product(*choices):
                    facts, selected, consistent = set(), {}, True
                    for f, selections in combo:
                        facts.update(f)
                        for key, value in selections.items():
                            if key in selected and selected[key] != value:
                                consistent = False
                            selected[key] = value
                    if consistent:
                        selected[nid] = branch
                        result.append((facts, selected))
        _require(len(result) <= 512 and bool(result), "query branch limit or inconsistent shared branches")
        cache[nid] = result
        return result
    def replay(facts, selected, removed=None, ablated=None):
        values = {}
        for n in graph["nodes"]:
            nid = n["node_id"]
            if n["type"] == "fact":
                if nid in facts and nid != removed:
                    values[nid] = n.get("symbolic_value")
            elif nid in selected:
                available = [values[i] for i in selected[nid] if i in values]
                try:
                    if n.get("kind") not in REL_OPS:
                        continue
                    if nid != ablated:
                        values[nid] = _relational_rule(n, available, probe=True)
                    elif n["rule"]["type"] == "guarded_transition":
                        states = [v["state"] for v in available if "state" in v]
                        _require(len(states) == 1, "state ablation needs one initial state")
                        values[nid] = dict(state=copy.deepcopy(states[0]))
                    elif n["rule"]["type"] == "constraint_filter":
                        full = _relational_rule(n, available, probe=True)
                        # Keep domains and unary restrictions. Remove actual
                        # cross-variable constraints, not unrelated premises.
                        unary = [c for c in full["constraints"]
                                 if len(set().union(*(set(m) for m in c.values()))) == 1]
                        values[nid] = _relational_rule(n, [dict(domains=full["domains"], constraints=unary)], probe=True)
                    else:
                        direct = {tuple(e) for v in available for e in v.get("precedence_edges", [])}
                        vertices = set().union(*(set(v.get("vertices", [])) for v in available))
                        vertices.update(x for e in direct for x in e)
                        values[nid] = dict(precedence_edges=[list(e) for e in sorted(direct)],
                                           vertices=sorted(vertices), order=None)
                except (ValidationError, ValueError, TypeError, KeyError):
                    pass
        return values.get(graph["answer_node"])

    rows, operations = [], []
    for facts, selected in branches(graph["answer_node"]):
        for removed in sorted(facts):
            same, answer = False, None
            try:
                answer = _query_answer(replay(facts,selected,removed=removed), query)
                same = _same(answer, expected)
            except (ValidationError, ValueError, TypeError, KeyError):
                pass
            row = dict(removed_fact_id=removed, branch_fact_ids=sorted(facts), remaining_answer=answer,
                       answer_unchanged=same, necessary_in_branch=not same)
            rows.append(row)
            _require(not same, "query-irrelevant premise: " + removed + "; local/simple path cannot inherit global difficulty")
        for nid in selected:
            n = by_id[nid]
            if n.get("kind") not in REL_OPS:
                continue
            if n["rule"]["type"] == "precedence_closure" and nid == graph["answer_node"]:
                direct = {tuple(e) for i in selected[nid]
                          for e in graph_audit["symbolic_node_values"][i].get("precedence_edges", [])}
                inferred = {tuple(e) for e in graph_audit["symbolic_node_values"][nid]["precedence_edges"]} - direct
                if query["type"] == "partial_order":
                    contributes = any(set(e) <= set(query["items"]) for e in inferred)
                elif query["type"] == "precedence":
                    a,b = query["before"],query["after"]
                    contributes = (a,b) in inferred or (b,a) in inferred
                else:
                    contributes = query["type"] == "full_order" and bool(inferred)
                _require(contributes, "relational operation adds no new query conclusion; independent/direct relation bundle is not L3")
                operations.append(dict(operation_id=nid, branch_fact_ids=sorted(facts),
                                       method="new_query_precedence_conclusion", contributes=True))
                continue
            after, unchanged = None, False
            try:
                after = _query_answer(replay(facts,selected,ablated=nid),query)
                unchanged = _same(after,expected)
            except (ValidationError, ValueError, TypeError, KeyError):
                pass
            _require(not unchanged, "query-irrelevant relational operation: " + nid + "; removing coupling/update preserves answer")
            operations.append(dict(operation_id=nid, branch_fact_ids=sorted(facts),
                                   method="query_conditioned_relation_ablation", ablated_answer=after,
                                   answer_unchanged=unchanged, contributes=True))
    return dict(query=copy.deepcopy(query), semantic_answer=expected, fact_deletions=rows, operation_contributions=operations,
                all_branch_facts_necessary=True, semantic_completeness_proved=False)

def candidate_form_fingerprint(candidate):
    """Bind cognitive form and option meaning; ignore dictionary insertion order."""
    return fp(dict(question_type=candidate.get("question_type"), question=candidate.get("question"),
                   options=candidate.get("options"), option_semantics={
                       k: row.get("semantic_value") for k,row in candidate.get("item_analysis", {}).items()}))

def _structural_proof_view(target):
    """Private adapter ONLY for unchanged legacy proof/evidence checks.

    Legacy validators expect a list but never decide the V2.2.1 answer. This
    opaque one-element proof view is not persisted, compared to option answers
    or used for numeric checks. All answer semantics use the typed contract.
    """
    value = copy.deepcopy(target)
    value["expected_answer"] = [answers.validate_semantic(target["semantic_answer"])]
    return value

def _deterministic_audit(candidate, target):
    """Exact rational checks address typed semantic values by path, not list slots."""
    value = target["semantic_answer"]["value"]
    numeric_paths = []
    def collect(v, path):
        if isinstance(v, dict):
            if "unit" in v and "value" in v:
                numeric_paths.append(path)
            else:
                for k,x in v.items(): collect(x,path+[k])
        elif isinstance(v, list):
            for i,x in enumerate(v): collect(x,path+[i])
    collect(value, [])
    proposed = candidate.get("deterministic_checks", [])
    _list(proposed, "deterministic_checks")
    _require(not (target.get("requires_deterministic_check") or numeric_paths) or proposed, "numeric/time target requires an exact deterministic check")
    covered, checks = set(), []
    for check in proposed:
        _require("answer_index" not in check, "numeric check uses semantic_path, not legacy answer_index")
        path = _list(check.get("semantic_path"), "numeric semantic_path")
        selected = value
        for key in path:
            if isinstance(selected, list):
                _require(type(key) is int and 0 <= key < len(selected), "invalid numeric semantic_path index")
            else:
                _require(isinstance(selected,dict) and isinstance(key,str) and key in selected, "invalid numeric semantic_path key")
            selected = selected[key]
        computed = v1.evaluate_check(check, {f["fact_id"] for f in target["facts"]})
        _require(isinstance(selected,dict) and selected.get("unit") == check["unit"] and
                 Fraction(str(selected["value"])) == Fraction(computed["result"]), "computation does not match frozen semantic answer/unit")
        checks.append(computed);covered.add(fp(path))
    _require(all(fp(path) in covered for path in numeric_paths), "every typed numeric answer requires an exact check")
    return checks

def adapt_v22_fixture(source, discovery, target, candidate, source_scope=None):
    """Explicit mechanical fixture migration; never new QA or source approval."""
    source, discovery, target, candidate = copy.deepcopy((source,discovery,target,candidate))
    kind = candidate.get("question_type")
    old = target.pop("expected_answer")
    projection = target.pop("answer_projection")
    if projection["path"] == ["order"]:
        query = dict(type="full_order"); semantic = answers.semantic_answer("sequence",old)
    elif projection["path"] == ["solutions"]:
        _require(len(old) == 1, "old complete assignment fixture must be unique")
        query = dict(type="assignment");semantic = answers.semantic_answer("object",old[0])
    else:
        query = dict(type="state_field",path=projection["path"])
        semantic = _query_answer(_graph_analysis(target["dependency_graph"],source)["symbolic_answer"],query)
    discovery["schema_version"] = VERSION+":discovery"
    for card in discovery["relations"]:
        if card["relation_id"] == target["relation_id"]:
            card["query"] = copy.deepcopy(query)
    target.update(query=query,semantic_answer=semantic,discovery_sha256=fp(discovery))
    target["selection_sha256"] = fp(rank_relations(discovery,source,source_scope))
    old_gold = candidate.pop("gold")
    candidate["gold_answer"] = (dict(kind="single_option",option_id=old_gold[0]) if kind == "single_choice" else
                                dict(kind="option_set" if kind == "multi_select" else "option_sequence",option_ids=old_gold))
    candidate["semantic_answer"] = semantic
    for row in candidate["item_analysis"].values():
        row["semantic_value"] = row.pop("answer_value")
    candidate.update(target_sha256=fp(target),relation_sha256=fp(target))
    candidate["query_path_audit"].update(form_sha256=candidate_form_fingerprint(candidate))
    return source,discovery,target,candidate,source_scope or {}

if __name__ == "__main__":
    raise SystemExit(main())
