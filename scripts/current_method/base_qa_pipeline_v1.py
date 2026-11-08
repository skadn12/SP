"""Offline, versioned relation-first QA authoring and review boundary.

No model/network/training entry point. Old generators and scorers are untouched.
Mechanical success never proves entailment, necessity, naturalness, or quality.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import random
import re

try:
    from . import production_question_gate as legacy
except ImportError:
    import production_question_gate as legacy

VERSION = "base_qa_p0_v1"
POLICY = {
    "version": VERSION, "feedback_enabled": False, "language": "en",
    "arms": ["F", "G"], "type_quotas": None, "gold_count_quotas": None,
    "model_calls_enabled": False, "training_enabled": False,
    "historical_scores_reusable": False,
}
CHECKS = ("gold_truth", "source_support", "connected_relation", "natural_question",
          "joint_necessity", "redundancy_bypass", "no_context_shortcut",
          "option_style", "type_validity")
PROMPT_DIR = Path(__file__).with_name("base_qa_prompts_v1")


def instruction(stage):
    return (PROMPT_DIR / (stage + ".txt")).read_text()


def source_view(source):
    """Entire dialogue, without expansion/generator provenance or latent targets."""
    return {"root_id": source["root_id"], "turns": [
        {k: t[k] for k in ("turn_ref", "scene_ref", "session_ref", "speaker", "text") if k in t}
        for t in source["turns"]]}


def text(value):
    return isinstance(value, str) and bool(value.strip())


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def source_from(binding, root_id):
    if legacy.file_binding(binding["path"]) != binding:
        raise ValueError("source binding drift")
    source = read(binding["path"])
    if source.get("root_id") != root_id:
        raise ValueError("source root mismatch")
    turns = source.get("turns", [])
    refs = [t["turn_ref"] for t in turns]
    if not turns or len(refs) != len(set(refs)):
        raise ValueError("empty source or duplicated turn refs")
    return source


def _evidence(evidence, by_ref, errors, label):
    refs = []
    if not isinstance(evidence, list) or not evidence:
        errors.append(label + ": nonempty evidence required")
        return refs
    for ev in evidence:
        if not isinstance(ev, dict):
            errors.append(label + ": evidence must be an object")
            continue
        turn = by_ref.get(ev.get("turn_ref"))
        if not (turn and turn["speaker"] == ev.get("speaker") and text(ev.get("quote"))
                and ev["quote"] in turn["text"]):
            errors.append(label + ": evidence is not an exact source quote")
        else:
            refs.append(ev["turn_ref"])
    return refs


def validate_relation(relation, source):
    """Verify a connected proposed proof, not the truth of its propositions."""
    errors = []
    if not isinstance(relation, dict):
        return ["relation must be an object"]
    for field in ("relation_id", "author_id", "target", "relation_type", "why_ask"):
        if not text(relation.get(field)):
            errors.append("missing relation field: " + field)
    if not isinstance(relation.get("expected_answer"), list) or not relation["expected_answer"]:
        errors.append("position-independent expected_answer must be a nonempty list")
    if not isinstance(relation.get("requires_deterministic_check"), bool):
        errors.append("requires_deterministic_check must be an explicit boolean")
    by_ref = {t["turn_ref"]: t for t in source["turns"]}
    facts = relation.get("facts", [])
    facts = facts if isinstance(facts, list) else []
    nodes, refs_by_fact = {}, {}
    for fact in facts:
        if not isinstance(fact, dict):
            errors.append("fact must be an object")
            continue
        fid = fact.get("fact_id")
        if not text(fid) or fid in nodes or not text(fact.get("claim")):
            errors.append("fact needs unique ID and claim")
            continue
        nodes[fid] = set()
        refs_by_fact[fid] = _evidence(fact.get("evidence"), by_ref, errors, fid)
        if fact.get("evidence_turn_ids") != refs_by_fact[fid]:
            errors.append(fid + ": evidence_turn_ids must equal exact citation IDs")
    if len(refs_by_fact) < 2:
        errors.append("main reasoning cohort requires at least two proposed necessary facts")
    steps = relation.get("inference_steps", [])
    steps = steps if isinstance(steps, list) else []
    last = None
    for step in steps:
        if not isinstance(step, dict):
            errors.append("step must be an object")
            continue
        sid, inputs = step.get("step_id"), step.get("inputs")
        if (not text(sid) or sid in nodes or not isinstance(inputs, list) or not inputs
                or any(not isinstance(i, str) or i not in nodes for i in inputs)
                or not text(step.get("conclusion"))):
            errors.append("step needs unique ID, preceding inputs, and conclusion")
            continue
        nodes[sid] = set(inputs)
        last = sid
    ancestors = set()
    def visit(nid):
        if nid not in ancestors:
            ancestors.add(nid)
            for parent in nodes[nid]:
                visit(parent)
    if last:
        visit(last)
    if not last or not set(refs_by_fact) <= ancestors:
        errors.append("all facts must feed the final inference; independent fact bundles rejected")
    groups = relation.get("necessity_groups", [])
    groups = groups if isinstance(groups, list) else []
    if len(groups) < 2:
        errors.append("at least two removal groups required")
    group_ids, group_facts, grouped_refs = set(), set(), set()
    positions = {t["turn_ref"]: i for i, t in enumerate(source["turns"])}
    for group in groups:
        if not isinstance(group, dict):
            errors.append("necessity group must be an object")
            continue
        gid, fids, refs = group.get("group_id"), group.get("fact_ids"), group.get("turn_refs")
        if not text(gid) or gid in group_ids:
            errors.append("unique group IDs required")
        else:
            group_ids.add(gid)
        if not isinstance(fids, list) or not fids or any(f not in refs_by_fact for f in fids):
            errors.append("group must bind known facts")
            fids = []
        group_facts.update(fids)
        if (not isinstance(refs, list) or not refs or any(r not in by_ref for r in refs)
                or len(refs) != len(set(refs))):
            errors.append("group needs distinct known source refs")
            refs = []
        if refs != sorted(refs, key=positions.get):
            errors.append("removal refs must remain in source order")
        if grouped_refs.intersection(refs):
            errors.append("overlapping removal groups must be merged")
        grouped_refs.update(refs)
        if any(not set(refs_by_fact[f]) <= set(refs) for f in fids):
            errors.append("group omits cited evidence for its facts")
        if not all(text(group.get(k)) for k in ("why_necessary", "removal_effect")):
            errors.append("group needs necessity and removal rationale")
    if group_facts != set(refs_by_fact):
        errors.append("removal groups must cover all facts")
    return errors


def evaluate_check(check, fact_ids):
    """Exact rational arithmetic / ISO time checks; no eval and no text guessing.

    Values are proposed source extractions. Their grounding still needs review.
    """
    dimensions = {"ratio": {}, "count": {"count": 1}, "person": {"person": 1},
                  "minute": {"minute": 1}, "person_minute": {"person": 1, "minute": 1},
                  "GBP": {"GBP": 1}, "iso_datetime": {"datetime": 1}}
    values, units, used = {}, {}, set()
    for name, row in check["inputs"].items():
        if row["fact_id"] not in fact_ids:
            raise ValueError("deterministic input refers to an unknown fact")
        used.add(row["fact_id"])
        values[name] = row["value"]
        units[name] = dimensions[row["unit"]]
    op = check["operation"]
    operands = [values[k] for k in check["operands"]]
    operand_units = [units[k] for k in check["operands"]]
    if op == "elapsed_minutes":
        if len(operands) != 2 or any(u != {"datetime": 1} for u in operand_units):
            raise ValueError("elapsed_minutes requires two ISO datetime inputs")
        a, b = [datetime.fromisoformat(v) for v in operands]
        result = Fraction(str((b - a).total_seconds())) / 60
        result_unit = {"minute": 1}
    else:
        nums = [Fraction(str(v)) for v in operands]
        if not nums:
            raise ValueError("empty arithmetic operands")
        if op in ("sum", "difference"):
            if any(u != operand_units[0] for u in operand_units):
                raise ValueError("cannot add/subtract different units")
            result_unit = operand_units[0]
        else:
            result_unit = Counter()
            for i, unit in enumerate(operand_units):
                for dimension, power in unit.items():
                    result_unit[dimension] += -power if op == "ratio" and i == 1 else power
            result_unit = {d: p for d, p in result_unit.items() if p}
        if op == "sum": result = sum(nums, Fraction(0))
        elif op == "product":
            result = Fraction(1)
            for n in nums: result *= n
        elif op == "difference" and len(nums) == 2: result = nums[0] - nums[1]
        elif op == "ratio" and len(nums) == 2: result = nums[0] / nums[1]
        else: raise ValueError("unsupported deterministic operation")
    if result != Fraction(str(check["expected_value"])):
        raise ValueError("deterministic result disagrees with expected value")
    if result_unit != dimensions[check["unit"]]:
        raise ValueError("output unit disagrees with exact operation dimensions")
    return {"result": str(result), "unit": check["unit"], "fact_ids": sorted(used),
            "source_value_grounding_verified": False}


def same_answer(kind, left, right):
    if kind == "multi_select":
        return Counter(map(legacy.fingerprint, left)) == Counter(map(legacy.fingerprint, right))
    return left == right


def validate_candidate(candidate, relation, source):
    errors = validate_relation(relation, source)
    if not isinstance(candidate, dict):
        return {"mechanical_pass": False, "errors": ["candidate must be an object"]}
    for key in ("candidate_id", "root_id", "author_id", "question"):
        if not text(candidate.get(key)): errors.append("missing candidate field: " + key)
    if candidate.get("relation_sha256") != legacy.fingerprint(relation):
        errors.append("candidate does not bind frozen relation")
    if candidate.get("author_id") != relation.get("author_id"):
        errors.append("relation and question author identity drift")
    if candidate.get("root_id") != source.get("root_id"):
        errors.append("candidate/source root mismatch")
    kind = candidate.get("question_type")
    if kind not in ("single_choice", "multi_select", "ordering"):
        errors.append("unsupported question_type")
    options = candidate.get("options", {})
    if (not isinstance(options, dict) or not 2 <= len(options) <= 26
            or not all(text(k) and text(v) for k, v in options.items())
            or len(set(options.values())) != len(options)):
        errors.append("2..26 distinct nonempty choices with stable IDs required")
        options = {}
    gold = candidate.get("gold", [])
    valid_gold = (isinstance(gold, list) and all(isinstance(k, str) for k in gold)
                  and len(gold) == len(set(gold)) and set(gold) <= set(options))
    if not valid_gold:
        errors.append("invalid Gold IDs")
        gold = []
    if kind == "single_choice" and len(gold) != 1: errors.append("single choice must have one Gold")
    if kind == "multi_select" and not 1 <= len(gold) < len(options):
        errors.append("main multi-select must have supported and unsupported alternatives; no quota")
    if kind == "ordering" and set(gold) != set(options): errors.append("ordering Gold is incomplete")
    analysis = candidate.get("item_analysis", {})
    if not isinstance(analysis, dict) or set(analysis) != set(options):
        errors.append("one analysis per choice required")
        analysis = {}
    by_ref = {t["turn_ref"]: t for t in source["turns"]}
    for key, row in analysis.items():
        if not isinstance(row, dict):
            errors.append("choice analysis must be an object")
            continue
        _evidence(row.get("evidence"), by_ref, errors, key)
        if not text(row.get("rationale")) or "answer_value" not in row:
            errors.append(key + ": rationale and position-independent answer_value required")
        if kind != "ordering":
            status = row.get("status")
            if status not in ("supported", "contradicted", "insufficient"):
                errors.append(key + ": explicit supported/contradicted/insufficient status required")
            if (status == "supported") != (key in gold): errors.append(key + ": truth/Gold conflict")
            if key not in gold and not text(row.get("error_reason")):
                errors.append(key + ": concrete distractor error required")
    selected = [analysis.get(k, {}).get("answer_value") for k in gold]
    if not same_answer(kind, selected, relation.get("expected_answer", [])):
        errors.append("Gold semantics disagree with answer frozen before options")
    if kind == "ordering":
        edges = candidate.get("ordering_edges", [])
        if not isinstance(edges, list): edges = []
        for edge in edges:
            if not isinstance(edge, dict):
                errors.append("ordering edge must be an object")
                continue
            _evidence(edge.get("evidence"), by_ref, errors, "ordering edge")
        safe = [e for e in edges if isinstance(e, dict)]
        graph = legacy._graph_audit({"items": options, "gold": gold}, safe)
        if not graph["valid_edge_ids"] or not graph["graph_order_matches_gold"]:
            errors.append("source-proposed edges must define a unique full order equal to Gold")
    checks, covered = [], set()
    numeric_targets = {i for i, answer in enumerate(relation.get("expected_answer", []))
                       if isinstance(answer, dict) and "value" in answer}
    proposed = candidate.get("deterministic_checks", [])
    if not isinstance(proposed, list):
        errors.append("deterministic_checks must be a list")
        proposed = []
    if (relation.get("requires_deterministic_check") is True or numeric_targets) and not proposed:
        errors.append("numeric/time target requires an exact deterministic check")
    for check in proposed:
        try:
            computed = evaluate_check(check, {f["fact_id"] for f in relation["facts"]})
            # Typed numeric answer binds the computation to the frozen answer.
            target = check["answer_index"]
            if isinstance(target, bool) or not isinstance(target, int) or target < 0:
                raise ValueError("answer_index must be a nonnegative integer")
            answer = relation["expected_answer"][target]
            if (not isinstance(answer, dict) or answer.get("unit") != check["unit"]
                    or Fraction(str(answer["value"])) != Fraction(computed["result"])):
                raise ValueError("computation does not match frozen semantic answer/unit")
            checks.append(computed)
            covered.add(target)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError) as exc:
            errors.append("invalid deterministic check: " + str(exc))
    if not numeric_targets <= covered:
        errors.append("every typed numeric answer must have a successful exact check")
    return {"candidate_id": candidate.get("candidate_id"), "mechanical_pass": not errors,
            "errors": errors, "deterministic_checks": checks,
            "semantic_quality_proved_by_code": False}


def prepare_display(candidate, seed):
    """Gold-independent shuffle; include no slot/operation, proof, or author data."""
    ids = list(candidate["options"])
    rng = random.Random(int(legacy.fingerprint([VERSION, seed, candidate["candidate_id"], ids]), 16))
    rng.shuffle(ids)
    mapping = {chr(65 + i): key for i, key in enumerate(ids)}
    reverse = {v: k for k, v in mapping.items()}
    public = {k: candidate[k] for k in ("question_type", "question")}
    public["options"] = {label: candidate["options"][key] for label, key in mapping.items()}
    gold = [reverse[k] for k in candidate["gold"]]
    if candidate["question_type"] == "multi_select": gold.sort()
    private = {"candidate_id": candidate["candidate_id"], "question_type": candidate["question_type"],
               "display_to_canonical": mapping, "canonical_gold": candidate["gold"][:],
               "display_gold": gold, "public_sha256": legacy.fingerprint(public), "seed": seed}
    if set(mapping.values()) != set(candidate["options"]) or len(mapping) != len(set(mapping.values())):
        raise ValueError("display mapping is not a bijection")
    if not same_answer(candidate["question_type"], [mapping[k] for k in gold], candidate["gold"]):
        raise ValueError("display Gold roundtrip failed")
    return public, private


def diagnose_response(private, answer, reported_value=None, expected_value=None):
    """Separate response defects from QA defects; does not replace old scoring."""
    mapping = private["display_to_canonical"]
    if (not isinstance(answer, list) or any(not isinstance(k, str) for k in answer)
            or len(set(answer)) != len(answer) or not set(answer) <= set(mapping)):
        return {"response_valid": False, "diagnosis": "INVALID_LABELS_OR_DUPLICATES"}
    kind = private["question_type"]
    if kind == "ordering" and set(answer) != set(mapping):
        return {"response_valid": False, "diagnosis": "ORDERING_MISSING_ITEMS",
                "missing_labels": sorted(set(mapping) - set(answer))}
    if kind == "single_choice" and len(answer) != 1:
        return {"response_valid": False, "diagnosis": "SINGLE_CHOICE_CARDINALITY"}
    canonical = [mapping[k] for k in answer]
    correct = same_answer(kind, canonical, private["canonical_gold"])
    label_error = (not correct and expected_value is not None and reported_value == expected_value)
    return {"response_valid": True, "answer_correct": correct,
            "diagnosis": "VALUE_CORRECT_OPTION_WRONG" if label_error else "CORRECT" if correct else "WRONG_ANSWER",
            "legacy_score_unchanged": True}


def dataset_audit(candidates):
    """Report patterns, not enforce balanced types/counts or rewrite Gold."""
    count_hist, type_hist, stem_hist, by_relation = Counter(), Counter(), Counter(), defaultdict(set)
    lengths = {"supported": [], "unsupported": []}
    true_first, quoted = [], []
    for c in candidates:
        kind, gold, options = c["question_type"], c["gold"], c["options"]
        type_hist[kind] += 1
        if kind != "ordering":
            count_hist[len(gold)] += 1
            by_relation[c.get("relation_type", "unrecorded")].add(len(gold))
            if set(list(options)[:len(gold)]) == set(gold): true_first.append(c["candidate_id"])
        stem_hist[" ".join(c["question"].lower().split()[:3])] += 1
        for key, value in options.items():
            if kind != "ordering":
                lengths["supported" if key in gold else "unsupported"].append(len(value.split()))
            if value.lstrip().startswith(('"', '“', "‘")): quoted.append(c["candidate_id"])
    return {"type_counts": dict(type_hist), "gold_count_histogram": dict(count_hist),
            "canonical_first_k_candidates": true_first, "stem_prefix_counts": dict(stem_hist),
            "quoted_option_candidates": sorted(set(quoted)),
            "mean_option_words": {k: sum(v)/len(v) if v else None for k, v in lengths.items()},
            "gold_counts_by_relation": {k: sorted(v) for k, v in by_relation.items()},
            "semantic_shortcut_exploitability": "NOT_ESTABLISHED_REQUIRES_BLIND_REVIEW_AND_EXPERIMENT",
            "quota_policy": "diagnostic_only; do not alter true answers to balance counts"}


def blind_packet(candidate, source):
    # Whitelist: reviewers cannot see author Gold, certificates, or targeted errors.
    public, private = prepare_display(candidate, seed="independent-review-v1")
    shuffled = {private["display_to_canonical"][label]: value
                for label, value in public["options"].items()}
    return {"stage": "INDEPENDENT_SOLVE", "language": POLICY["language"],
            "instruction": instruction("blind"),
            "candidate_id": candidate["candidate_id"], "source": source_view(source),
            **{k: copy.deepcopy(candidate[k]) for k in ("question_type", "question")},
            "options": shuffled}


def validate_blind(candidate, source, packet, blind):
    errors = []
    if not isinstance(blind, dict):
        return ["blind solution must be an object"]
    if blind.get("packet_sha256") != legacy.fingerprint(packet): errors.append("blind packet hash drift")
    if not text(blind.get("reviewer_id")) or blind.get("reviewer_id") == candidate["author_id"]:
        errors.append("independent reviewer identity required")
    if not text(blind.get("model_id")): errors.append("actual reviewer model or human identity required")
    if blind.get("decision") != "SOLVED": errors.append("blind result unresolved")
    if blind.get("ambiguities") != []:
        errors.append("blind ambiguities must be explicitly empty; unresolved ambiguity means HOLD")
    gold = blind.get("independent_gold")
    if (not isinstance(gold, list) or not gold or any(not isinstance(k, str) for k in gold)
            or len(set(gold)) != len(gold) or not set(gold) <= set(candidate["options"])):
        errors.append("invalid independent answer")
    elif candidate["question_type"] == "ordering" and set(gold) != set(candidate["options"]):
        errors.append("independent ordering answer omits items")
    elif candidate["question_type"] == "single_choice" and len(gold) != 1:
        errors.append("independent single choice answer is not unique")
    _evidence(blind.get("evidence"), {t["turn_ref"]: t for t in source["turns"]}, errors, "blind")
    if not isinstance(blind.get("reasoning_steps"), list) or not all(text(s) for s in blind["reasoning_steps"]) or not blind["reasoning_steps"]:
        errors.append("independent source reasoning required")
    return errors


def validate_review(candidate, relation, source, packet, blind, comparison):
    audit = validate_candidate(candidate, relation, source)
    errors = audit["errors"] + validate_blind(candidate, source, packet, blind)
    if not isinstance(blind, dict): blind = {}
    if not isinstance(comparison, dict):
        errors.append("comparison must be an object")
        comparison = {}
    if not same_answer(candidate["question_type"], blind.get("independent_gold", []), candidate["gold"]):
        errors.append("Gold disagreement: HOLD, no automatic winner")
    if comparison.get("blind_sha256") != legacy.fingerprint(blind): errors.append("blind freeze binding drift")
    if comparison.get("candidate_sha256") != legacy.fingerprint(candidate): errors.append("candidate freeze binding drift")
    if comparison.get("reviewer_id") != blind.get("reviewer_id"): errors.append("comparison reviewer identity drift")
    if comparison.get("decision") != "PASS": errors.append("comparison not PASS")
    semantic_checks = comparison.get("checks", {})
    if not isinstance(semantic_checks, dict):
        errors.append("comparison checks must be an object")
        semantic_checks = {}
    for check in CHECKS:
        row = semantic_checks.get(check, {})
        if not isinstance(row, dict): row = {}
        if row.get("decision") != "PASS" or not text(row.get("rationale")):
            errors.append("unresolved semantic check: " + check)
    groups = relation.get("necessity_groups", [])
    rows = comparison.get("necessity_review", [])
    if not isinstance(rows, list): rows = []
    if any(not isinstance(row, dict) for row in rows):
        errors.append("necessity review rows must be objects")
        rows = [row for row in rows if isinstance(row, dict)]
    if len(rows) != len(groups) or {r.get("group_id") for r in rows} != {g["group_id"] for g in groups}:
        errors.append("independent necessity review must cover all groups exactly once")
    for g in groups:
        matching = [r for r in rows if r.get("group_id") == g["group_id"]]
        if len(matching) != 1: continue
        r = matching[0]
        if (r.get("removed_turn_refs") != g["turn_refs"]
                or r.get("target_underdetermined") is not True
                or r.get("single_group_insufficient") is not True
                or r.get("no_remaining_bypass") is not True
                or not text(r.get("rationale"))):
            errors.append("necessity or redundancy unresolved: " + g["group_id"])
    return {**audit, "errors": errors, "decision": "HOLD" if errors else "PASS_STATIC_REVIEW",
            "admitted": not errors, "empirical_quality_established": False,
            "basis": "bound independent semantic judgments plus deterministic checks; not a proof by schema"}


def _context(run):
    p = _frozen(run, "INPUT.json", "INPUT_FREEZE.json")
    if p["policy"] != POLICY:
        raise ValueError("P0 policy drift")
    if legacy.file_binding(p["selection_binding"]["path"]) != p["selection_binding"]:
        raise ValueError("selection binding drift")
    if p["warnings_binding"] and legacy.file_binding(p["warnings_binding"]["path"]) != p["warnings_binding"]:
        raise ValueError("source warning binding drift")
    if p["implementation_bindings"] != implementation_bindings():
        raise ValueError("implementation or prompt drift; start a new versioned run")
    return p, source_from(p["source_binding"], p["root_id"])


def implementation_bindings():
    return [legacy.file_binding(p) for p in [Path(__file__), Path(legacy.__file__),
            *sorted(PROMPT_DIR.glob("*.txt"))]]


def _frozen(run, filename, receipt):
    data = read(run / filename)
    if read(run / receipt)["sha256"] != legacy.fingerprint(data):
        raise ValueError("frozen artifact drift: " + filename)
    return data


def _freeze(run, filename, receipt, data):
    write_new(run / filename, data)
    write_new(run / receipt, {"sha256": legacy.fingerprint(data), "version": VERSION})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--arm", choices=POLICY["arms"], default="G")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--feedback", type=Path, help="rejected in P0")
    p = sub.add_parser("audit-batch")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-display")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    for name in ("freeze-relation", "freeze-candidate", "freeze-blind", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit-batch":
        value = read(args.input)
        candidates = value.get("candidates") if isinstance(value, dict) else value
        if not isinstance(candidates, list): raise ValueError("candidate list required")
        write_new(args.output, {"input_binding": legacy.file_binding(args.input),
                               "audit": dataset_audit(candidates), "model_calls": 0})
        return 0
    run = args.run_dir
    if args.command == "prepare":
        if args.feedback is not None: raise ValueError("QA weakness feedback is disabled in P0")
        sources, selection_binding = legacy._selection(args.selection)
        entry = sources[args.root]
        source = source_from(entry["binding"], args.root)
        # Source warnings are part of reviewed input, not silently discarded.
        selected = next(r for r in read(args.selection)["roots"] if r["root_id"] == args.root)
        warnings_path = selected.get("warnings_path")
        warnings = read(warnings_path) if warnings_path else []
        run.mkdir(parents=True, exist_ok=False)
        _freeze(run, "INPUT.json", "INPUT_FREEZE.json", {"policy": POLICY, "arm": args.arm, "root_id": args.root,
                  "source_binding": entry["binding"], "selection_binding": selection_binding,
                  "source_warnings": warnings,
                  "warnings_binding": legacy.file_binding(warnings_path) if warnings_path else None,
                  "implementation_bindings": implementation_bindings()})
        write_new(run / "DISCOVERY_PACKET.json", {"stage": "DISCOVER_RELATION", "policy": POLICY,
                  "instruction": instruction("discovery"),
                  "source": source_view(source), "source_warnings": warnings})
    else:
        context, source = _context(run)
        if args.command == "prepare-display":
            frozen = _frozen(run, "FROZEN_QA.json", "FROZEN_QA_FREEZE.json")
            if frozen["decision"]["decision"] != "PASS_STATIC_REVIEW":
                raise ValueError("only statically admitted frozen QA may be displayed")
            public, private = prepare_display(frozen["candidate"], args.seed)
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write_new(args.output_dir / "PUBLIC_QUESTION.json", public)
            write_new(args.output_dir / "PRIVATE_MAPPING.json", private)
            print(json.dumps({"stage": args.command, "output_dir": str(args.output_dir), "model_calls": 0}))
            return 0
        value = read(args.input)
        if args.command == "freeze-relation":
            if value.get("status") == "NO_SAFE_QA":
                if not text(value.get("reason")): raise ValueError("NO_SAFE_QA requires a reason")
                _freeze(run, "RELATION.json", "RELATION_FREEZE.json", value)
                print(json.dumps({"status": "NO_SAFE_QA", "model_calls": 0}))
                return 0
            errors = validate_relation(value, source)
            if errors: raise ValueError(errors)
            _freeze(run, "RELATION.json", "RELATION_FREEZE.json", value)
            write_new(run / "AUTHOR_PACKET.json", {"stage": "WRITE_NATURAL_QUESTION", "policy": POLICY,
                      "instruction": instruction("author"),
                      "source": source_view(source), "source_warnings": context["source_warnings"], "relation": value,
                      "relation_sha256": legacy.fingerprint(value)})
        elif args.command == "freeze-candidate":
            relation = _frozen(run, "RELATION.json", "RELATION_FREEZE.json")
            if relation.get("status") == "NO_SAFE_QA": raise ValueError("NO_SAFE_QA has no question")
            audit = validate_candidate(value, relation, source)
            if not audit["mechanical_pass"]: raise ValueError(audit["errors"])
            _freeze(run, "CANDIDATE.json", "CANDIDATE_FREEZE.json", value)
            packet = blind_packet(value, source)
            packet["source_warnings"] = context["source_warnings"]
            _freeze(run, "BLIND_PACKET.json", "BLIND_PACKET_FREEZE.json", packet)
            write_new(run / "MECHANICAL_AUDIT.json", audit)
        elif args.command == "freeze-blind":
            candidate = _frozen(run, "CANDIDATE.json", "CANDIDATE_FREEZE.json")
            packet = _frozen(run, "BLIND_PACKET.json", "BLIND_PACKET_FREEZE.json")
            errors = validate_blind(candidate, source, packet, value)
            # Freeze uncertainty as well; it cannot be overwritten by a Gold-aware answer.
            _freeze(run, "BLIND_SOLUTION.json", "BLIND_SOLUTION_FREEZE.json", value)
            relation = _frozen(run, "RELATION.json", "RELATION_FREEZE.json")
            write_new(run / "COMPARISON_PACKET.json", {"stage": "COMPARE_AFTER_BLIND_FREEZE",
                      "instruction": instruction("comparison"),
                      "source": source_view(source), "source_warnings": context["source_warnings"],
                      "candidate": candidate, "relation": relation, "blind_solution": value,
                      "blind_errors": errors, "required_checks": CHECKS,
                      "removal_contexts": [{"group_id": g["group_id"], "removed_turn_refs": g["turn_refs"],
                          "remaining_turn_refs": [t["turn_ref"] for t in source["turns"] if t["turn_ref"] not in g["turn_refs"]],
                          "single_group_turn_refs": [t["turn_ref"] for t in source["turns"] if t["turn_ref"] in g["turn_refs"]]}
                          for g in relation["necessity_groups"]],
                      "necessity_check_mode": "STATIC_COUNTERFACTUAL_REVIEW_NOT_BLIND_ABLATION"})
        else:
            candidate = _frozen(run, "CANDIDATE.json", "CANDIDATE_FREEZE.json")
            relation = _frozen(run, "RELATION.json", "RELATION_FREEZE.json")
            packet = _frozen(run, "BLIND_PACKET.json", "BLIND_PACKET_FREEZE.json")
            blind = _frozen(run, "BLIND_SOLUTION.json", "BLIND_SOLUTION_FREEZE.json")
            result = validate_review(candidate, relation, source, packet, blind, value)
            _freeze(run, "COMPARISON_REVIEW.json", "COMPARISON_REVIEW_FREEZE.json", value)
            write_new(run / "DECISION.json", result)
            if result["admitted"]:
                _freeze(run, "FROZEN_QA.json", "FROZEN_QA_FREEZE.json", {"candidate": candidate, "relation": relation,
                          "blind_solution_sha256": legacy.fingerprint(blind), "decision": result})
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["admitted"] else 2
    print(json.dumps({"stage": args.command, "run_dir": str(run), "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
