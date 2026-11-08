"""OET1 first-attempt question controls. No model calls or solver-score reads.

All content files and freezes use exclusive creation. This module checks and
packages authored/reviewed content; it never synthesizes or repairs content.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
QFP2 = BASE.parent / "postweek_question_firstpass_v2"
RULES = BASE.parent / "postweek_question_presentation_repair_v1/QUESTION_AUTHOR_RULES_V2.txt"
PROTOCOL = BASE / "PROTOCOL_FREEZE_V1.json"
SLOTS = {"MC2": "MC_ENTITY_BINDING", "MC3": "MC_STATE_UPDATE_CONJUNCTION",
         "MC4": "MC_CONCLUSION_EVIDENCE", "MC5": "MC_JOINT_CONSTRAINT",
         "MC6": "MC_RECORD_AUDIT_CONTROL", "ORDER1": "ORDER_EXPLICIT_CLOCK_VALUE",
         "ORDER2": "ORDER_ENQUIRY_PROGRESSION"}
ARMS = ("GENERIC", "TARGETED")
MAIN_SLOTS = tuple(s for s in SLOTS if s != "MC6")
_spec = importlib.util.spec_from_file_location("oet1_public_gate", REPO / "scripts/current_method/question_presentation_gate_v2.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
CUE_WORDS = re.compile(r"\b(?:first|second|third|fourth|next|later|finally|summary|before|after|earlier|lastly|subsequently|then)\b", re.I)
REVIEW_PUBLIC_RULES = """Selection stems must be count-neutral and must not reveal
any private quota or control label. Preserve factual numbers and original source
quotations; do not globally remove numerals. Verify the stated operation: entity
binding, updated-state conjunction, conclusion evidence, linked constraints,
independent factual record verification, numeric clock comparison, or recorded
disclosure progression. Check every clause and its role using exact source
evidence. Repeated identity fields do not establish joint-constraint reasoning.
Assess the available foil discrimination honestly; do not require impossible
properties from a fixed set. Ordering display must be shuffled, distinct from
insertion and independently derived Gold order. Items omit chronology/disclosure
cue words. Numeric clock comparison and recorded disclosure progression remain
distinct operations. Independently derive the unique answer before comparing to
any authored answer. Preserve failures; no solver-score filtering, content repair
or semantic retry. Mechanical checking alone never establishes semantic validity.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def binding_fields(value):
    return {k: value[k] for k in ("path", "sha256", "bytes")}


def verify(value):
    require(bind(value["path"]) == binding_fields(value), "BINDING_CHANGED", value["path"])
    return read(value["path"])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return bind(path)


def require(condition, code, detail=None):
    if not condition:
        raise ValueError(f"{code}: {detail}" if detail is not None else code)


def source_path(root, arm):
    return BASE / "sources" / root / arm / "FINAL_SOURCE.json"


def check_unit(root, arm):
    p = read(PROTOCOL)
    require(root in p["roots"] and arm in ARMS, "UNPLANNED_UNIT", [root, arm])


def check_source(path):
    source = read(path)
    turns = source["turns"]
    require(isinstance(turns, list) and bool(turns), "EMPTY_SOURCE")
    refs = [t["turn_ref"] for t in turns]
    require(len(set(refs)) == len(refs), "DUPLICATE_TURN_REF")
    for t in turns:
        require(all(isinstance(t[k], str) and t[k].strip() for k in ("turn_ref", "speaker", "text")), "INVALID_TURN")
    return source, refs, {t["turn_ref"]: t for t in turns}


def check_evidence(evidence, turns, context):
    require(isinstance(evidence, list) and evidence, "MISSING_EVIDENCE", context)
    for e in evidence:
        require(e.get("turn_ref") in turns, "UNKNOWN_TURN", [context, e])
        require(isinstance(e.get("quote"), str) and e["quote"] and e["quote"] in turns[e["turn_ref"]]["text"], "QUOTE_NOT_CONTIGUOUS", [context, e])
        if "speaker" in e:
            require(e["speaker"] == turns[e["turn_ref"]]["speaker"], "SPEAKER_BINDING_MISMATCH", context)


def source_admission(root, arm):
    """Require the six actual sequential acceptance receipts, not only a source."""
    receipts = []
    last_source = None
    for scene in read(PROTOCOL)["scene_sequence"]:
        path = source_path(root, arm).parent / scene.replace(":", "_") / "ACCEPTED.json"
        require(path.exists(), "SOURCE_NOT_FULLY_REVIEWED", str(path))
        receipt = read(path)
        require(receipt.get("status") == "READY", "SOURCE_SCENE_NOT_ACCEPTED", str(path))
        # FBC1 acceptance receipts include bindings; validate every direct binding.
        for value in receipt.values():
            if isinstance(value, dict) and {"path", "sha256", "bytes"} <= value.keys():
                verify(value)
        receipts.append(bind(path))
        last_source = receipt.get("source")
    check_source(source_path(root, arm))
    if last_source:
        final = bind(source_path(root, arm))
        require(final["sha256"] == last_source["sha256"] and final["bytes"] == last_source["bytes"], "FINAL_SOURCE_DIFFERS_FROM_ACCEPTED_SOURCE")
    return receipts


def prepare(root, arm):
    check_unit(root, arm)
    receipts = source_admission(root, arm)
    path = source_path(root, arm)
    source, refs, _ = check_source(path)
    packet = {"schema_version": "oet1_question_author_input_v1", "root_id": root,
              "condition": arm, "source_binding": bind(path), "full_source": source,
              "slots": SLOTS, "mc_options": 6, "ordering_items": 4,
              "required_gold_counts": {f"MC{i}": i for i in range(2, 7)},
              "drafts_per_slot": 1, "semantic_revisions": 0,
              "source_turn_refs": refs, "source_acceptance_bindings": receipts,
              "rules_binding": bind(RULES), "protocol_binding": bind(PROTOCOL),
              "question_author_feedback": "Full accepted source only; no solver, old questions, or private evaluation data.",
              "formal_weakness_admission": False, "causal_feedback_benefit_claim": False}
    return save(BASE / "question_handoffs" / root / arm / "QUESTION_AUTHOR_INPUT.json", packet)


def mechanical(path, *, historical=False):
    q = read(path)
    slot = q["slot_id"]
    require(slot in SLOTS and q["task_operation"] == SLOTS[slot], "WRONG_OPERATION")
    if not historical:
        check_unit(q["root_id"], q["condition"])
        require(q["source_binding"] == bind(source_path(q["root_id"], q["condition"])), "WRONG_FINAL_SOURCE")
    source = verify(q["source_binding"])
    _, refs, turns = check_source(q["source_binding"]["path"])
    scope = q["source_read_scope"]
    require(scope.get("all_turns_read") is True and scope.get("turn_count") == len(refs), "INCOMPLETE_AUTHOR_READ_SCOPE")
    multi = slot.startswith("MC")
    require(q["question_type"] == ("multi_select" if multi else "ordering"), "WRONG_QUESTION_TYPE")
    texts = q["options"] if multi else q["items"]
    keys = list(texts)
    require(set(keys) == (set("ABCDEF") if multi else {"I1", "I2", "I3", "I4"}), "WRONG_CHOICES")
    require(all(isinstance(t, str) and t.strip() for t in texts.values()) and len(set(texts.values())) == len(keys), "EMPTY_OR_DUPLICATE_CHOICES")
    gold = q["gold"]
    require(len(gold) == (int(slot[2:]) if multi else 4) and len(set(gold)) == len(gold) and set(gold) <= set(keys), "INVALID_GOLD_CARDINALITY")
    require(len(q["display_order"]) == len(keys) and set(q["display_order"]) == set(keys), "DISPLAY_NOT_BIJECTION")
    require(isinstance(q["display_seed"], int), "MISSING_DISPLAY_SEED")
    require(isinstance(q["question"], str) and q["question"].strip(), "EMPTY_STEM")
    if not multi:
        require(q["display_order"] != keys and q["display_order"] != gold, "ORDER_DISPLAY_NOT_SHUFFLED")
        require(not any(CUE_WORDS.search(text) for text in texts.values()), "ORDER_ITEM_CHRONOLOGY_CUE")
    require(set(q["evidence"]) == set(q["private_option_notes"]) == set(keys), "INCOMPLETE_OPTION_RECORDS")
    for key in keys:
        require(isinstance(q["private_option_notes"][key], str) and q["private_option_notes"][key].strip(), "EMPTY_OPTION_NOTE", key)
        check_evidence(q["evidence"][key], turns, key)
    if slot in ("MC3", "MC5"):
        clauses = q.get("clause_checks", {})
        require(set(clauses) == set(keys), "MISSING_CLAUSE_CHECKS")
        for key in keys:
            require(isinstance(clauses[key], list) and len(clauses[key]) >= 2, "CONJUNCTION_NEEDS_MULTIPLE_CLAUSES", key)
            for clause in clauses[key]:
                require(bool(clause.get("clause")) and bool(clause.get("verdict")), "INVALID_CLAUSE", key)
                require(bool(clause.get("turn_refs")) and set(clause["turn_refs"]) <= set(turns), "INVALID_CLAUSE_TURNS", key)
                if clause.get("evidence"):
                    check_evidence(clause["evidence"], turns, key)
    if slot == "ORDER1":
        clocks = q.get("canonical_clock_values", {})
        require(set(clocks) == set(keys), "MISSING_CLOCK_VALUES")
        values = {}
        for key, clock in clocks.items():
            require(isinstance(clock, str) and bool(re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", clock)), "INVALID_CLOCK", key)
            # A source may explicitly say "2:15 that afternoon" while the
            # private normalized comparison value is 14:15. Semantic review
            # checks that normalization; mechanics requires a cited reading.
            hour, minute = clock.split(":")
            forms = {clock, f"{int(hour)}:{minute}", f"{(int(hour)-1)%12+1}:{minute}"}
            require(any(any(value in e["quote"] for value in forms) for e in q["evidence"][key]), "CLOCK_NOT_EXPLICITLY_CITED", key)
            require(not re.search(r"\b\d{1,2}:\d{2}\b", texts[key]), "CLOCK_LEAKED_IN_ITEM", key)
            values[key] = int(hour) * 60 + int(minute)
        require(len(set(values.values())) == 4 and gold == sorted(keys, key=values.get), "CLOCK_GOLD_INCORRECT")
    if slot == "ORDER2":
        edges = q.get("relation_edges", [])
        require(len(edges) == 3, "MISSING_ADJACENT_RELATIONS")
        for edge, left, right in zip(edges, gold, gold[1:]):
            require(edge.get("from", edge.get("before")) == left and edge.get("to", edge.get("after")) == right, "WRONG_RELATION_EDGE")
            require(bool(edge.get("turn_refs")) and set(edge["turn_refs"]) <= set(turns), "INVALID_RELATION_TURNS")
    gate.require_public_selection(q["question"], gate.public_operation(q["task_operation"]), q["question_type"])
    return {"candidate_id": q["candidate_id"], "status": "PASS_MECHANICAL_NOT_SEMANTIC", "binding": bind(path), "turn_count": len(source["turns"])}


def review_packet(root, arm):
    """Hide submitted Gold and all private notes until independent derivation."""
    check_unit(root, arm)
    source_admission(root, arm)
    path = source_path(root, arm)
    source, refs, _ = check_source(path)
    candidates = []
    for slot in SLOTS:
        review_key = "QR_" + hashlib.sha256(f"{root}:{arm}:{slot}".encode()).hexdigest()[:18]
        candidate_path = BASE / "questions" / root / arm / f"{slot}.json"
        if candidate_path.exists():
            q = read(candidate_path)
            keep = ("root_id", "condition", "question_type", "question", "options", "items", "display_order", "evidence")
            candidates.append({k: q.get(k) for k in keep} |
                              {"review_key": review_key, "question_sha256": bind(candidate_path)["sha256"],
                               "task_operation": gate.public_operation(q["task_operation"])})
        else:
            short = candidate_path.with_name(f"{slot}_SHORTFALL.json")
            candidates.append({"review_key": review_key, "status": "SHORTFALL" if short.exists() else "MISSING",
                               "task_operation": gate.public_operation(SLOTS[slot]),
                               "shortfall_reason": read(short).get("reason", read(short).get("notes")) if short.exists() else None})
    return {"schema_version": "oet1_gold_hidden_derivation_input_v1", "root_id": root,
            "condition": arm, "source_binding": bind(path), "full_source": source,
            "source_turn_refs": refs, "candidates": sorted(candidates, key=lambda row: row["review_key"]),
            "rules": REVIEW_PUBLIC_RULES, "rules_binding": bind(RULES),
            "review_rule_projection": "Presentation and operation requirements; private quotas/control identities withheld.",
            "submitted_gold_visible": False, "private_author_notes_visible": False,
            "slot_quotas_and_control_labels_visible": False,
            "independent_ai_review_condition_visible": True}


def seal_review(root, arm, derivation_path, reviewer_id, input_path=None):
    """Master compares frozen independent derivation with submitted private Gold.

    Any controller downgrade is explicit; no independently derived answer changes.
    The raw derivation and its CLI provenance remain separately immutable.
    """
    d = read(derivation_path)
    source, refs, _ = check_source(source_path(root, arm))
    require(d.get("full_source_read") is True and d.get("turn_refs_read") == refs, "INCOMPLETE_REVIEW_SOURCE_READ")
    require(d.get("solver_scores_read") is False, "SOLVER_ISOLATION_UNCONFIRMED")
    require(d.get("independent_of_question_author") is True and d.get("independent_of_source_author") is True, "REVIEW_NOT_INDEPENDENT")
    rows = {row["slot_id"]: row for row in d["questions"]}
    require(len(d["questions"]) == 7 and set(rows) == set(SLOTS), "REVIEW_SLOT_DENOMINATOR_CHANGED")
    for slot, row in rows.items():
        row["reviewer_id"] = reviewer_id
        path = BASE / "questions" / root / arm / f"{slot}.json"
        if not path.exists():
            require(row["decision"] in ("FAIL", "SHORTFALL"), "MISSING_CANDIDATE_ADMITTED")
            continue
        q = read(path)
        require(reviewer_id != q["author_id"], "SELF_REVIEW")
        row["question_binding"] = bind(path)
        derived = row.get("independently_derived_gold", [])
        same = (set(derived) == set(q["gold"]) and len(derived) == len(q["gold"])) if slot.startswith("MC") else derived == q["gold"]
        row["submitted_gold_matches"] = same
        row["reviewer_decision"] = row["decision"]
        if not same and row["decision"] == "ADMIT":
            row["decision"] = "FAIL"
            row["controller_rejection_codes"] = ["INDEPENDENT_GOLD_MISMATCH"]
        row["source_or_gold_ambiguity_failure"] = bool(row.get("source_or_gold_ambiguity_failure")) or not same or row.get("checks", {}).get("unique_key") == "FAIL"
    result = {**d, "schema_version": "oet1_independent_question_review_v1",
              "reviewer_id": reviewer_id, "reviewer_type": "codex_cli_ai",
              "root_id": root, "condition": arm, "created_at_utc": now(),
              "source_binding": bind(source_path(root, arm)),
              "questions": [rows[s] for s in SLOTS],
              "blinding": "condition_visible_author_gold_hidden_until_derivation",
              "independent_derivation_binding": bind(derivation_path),
              "controller_metadata": {"gold_comparison_only": True, "semantic_changes": False}}
    if input_path:
        result["derivation_input_binding"] = bind(input_path)
    path = BASE / "question_reviews" / root / arm / "REVIEW.json"
    # A malformed per-slot review fails that slot, preserving valid independent
    # slots. Unit-wide source/read/identity errors still block the whole unit.
    # No missing semantic judgment is filled in by this controller.
    for row in result["questions"]:
        try:
            validate_review(root, arm, review=result, check_receipt=True, selected_slots={row["slot_id"]})
        except Exception as error:
            row["controller_contract_errors"] = [str(error)]
            row.setdefault("reviewer_decision", row.get("decision"))
            row["decision"] = "FAIL"
            row["notes"] = (row.get("notes", "") + "\nController contract failure: " + str(error)).strip()
    validate_review(root, arm, review=result, check_receipt=True)
    return save(path, result)


def validate_receipt(root, arm):
    path = BASE / "questions" / root / arm / "AUTHOR_RECEIPT.json"
    receipt = read(path)
    require(receipt["root_id"] == root and receipt["condition"] == arm, "WRONG_AUTHOR_RECEIPT_UNIT")
    require(receipt["source_binding"] == bind(source_path(root, arm)), "AUTHOR_RECEIPT_SOURCE_CHANGED")
    _, refs, _ = check_source(source_path(root, arm))
    scope = receipt["source_read_scope"]
    require(scope.get("all_turns_read") is True and scope.get("turn_count") == len(refs), "INCOMPLETE_AUTHOR_RECEIPT_SCOPE")
    read_refs = receipt.get("full_turn_refs_read", scope.get("turn_refs", scope.get("all_turn_refs_read")))
    require(read_refs == refs, "AUTHOR_FULL_TURN_REFS_REQUIRED")
    candidates = {v["path"]: binding_fields(v) for v in receipt["candidate_bindings"]}
    require(len(candidates) == 7, "AUTHOR_RECEIPT_SLOT_DENOMINATOR_CHANGED")
    for value in candidates.values():
        verify(value)
    for value in receipt.get("input_bindings", []):
        verify(value)
    return receipt, candidates, bind(path)


def validate_review(root, arm, review=None, *, check_receipt=True, historical=False, selected_slots=None):
    rp = (QFP2 if historical else BASE) / "question_reviews" / root / arm / "REVIEW.json"
    review = read(rp) if review is None else review
    source_binding = review["source_binding"]
    verify(source_binding)
    _, refs, turns = check_source(source_binding["path"])
    require(review.get("independent_of_question_author") is True and review.get("independent_of_source_author") is True, "REVIEW_NOT_INDEPENDENT")
    require(review.get("full_source_read") is True and review.get("turn_refs_read") == refs, "REVIEW_INCOMPLETE_SOURCE_READ")
    require(review.get("solver_scores_read") is False, "REVIEW_SOLVER_ISOLATION_UNCONFIRMED")
    require(review["root_id"] == root and review["condition"] == arm, "REVIEW_WRONG_UNIT")
    if not historical:
        require(source_binding == bind(source_path(root, arm)), "REVIEW_SOURCE_CHANGED")
    rows = {r["slot_id"]: r for r in review["questions"]}
    require(len(review["questions"]) == 7 and set(rows) == set(SLOTS), "REVIEW_SLOT_DENOMINATOR_CHANGED")
    receipt = candidates = None
    if check_receipt and not historical:
        receipt, candidates, _ = validate_receipt(root, arm)
    for slot, row in rows.items():
        if selected_slots is not None and slot not in selected_slots:
            continue
        require(row["decision"] in ("ADMIT", "FAIL", "SHORTFALL"), "INVALID_REVIEW_DECISION")
        path = (QFP2 if historical else BASE) / "questions" / root / arm / f"{slot}.json"
        if not path.exists():
            require(row["decision"] != "ADMIT", "MISSING_CANDIDATE_ADMITTED")
            short = path.with_name(f"{slot}_SHORTFALL.json")
            require(short.exists() or row["decision"] == "FAIL", "UNEXPLAINED_SHORTFALL")
            if candidates is not None and short.exists():
                require(candidates.get(str(short.resolve())) == bind(short), "SHORTFALL_RECEIPT_DRIFT")
            continue
        q = read(path)
        require(row["question_binding"] == bind(path), "REVIEW_QUESTION_CHANGED")
        require(row["reviewer_id"] != q["author_id"], "QUESTION_SELF_REVIEW")
        if receipt is not None:
            require(receipt["author_id"] == q["author_id"] and candidates.get(str(path.resolve())) == bind(path), "AUTHOR_RECEIPT_DRIFT")
        if row["decision"] != "ADMIT":
            require(bool(row.get("notes")), "FAILED_REVIEW_NEEDS_REASON", slot)
            continue
        require(q["source_binding"] == source_binding, "QUESTION_REVIEW_SOURCE_MISMATCH")
        require(row.get("submitted_gold_matches") is True and row.get("operation_alignment") == "PASS", "ADMISSION_GOLD_OR_OPERATION_FAILED")
        derived = row["independently_derived_gold"]
        require((set(derived) == set(q["gold"]) and len(derived) == len(q["gold"])) if slot.startswith("MC") else derived == q["gold"], "INDEPENDENT_GOLD_MISMATCH")
        for dimension in ("grounding", "unique_key", "actual_operation", "presentation"):
            require(row["checks"][dimension] == "PASS", "ADMISSION_CHECK_FAILED", [slot, dimension])
        require(row["checks"]["foils"] in ("PASS", "WARNING", "NOT_APPLICABLE"), "FOIL_REVIEW_FAILED")
        keys = set(q["options"] if slot.startswith("MC") else q["items"])
        findings = row["item_findings"]
        require(set(findings) == keys, "REVIEW_ITEM_DENOMINATOR_CHANGED")
        for key, finding in findings.items():
            require(finding["verdict"] in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED") and bool(finding.get("notes")), "MISSING_ITEM_JUDGMENT", key)
            check_evidence(finding["evidence"], turns, [slot, key])
            if slot.startswith("MC") and slot != "MC4":
                require((finding["verdict"] == "SUPPORTED") == (key in derived), "ITEM_GOLD_INCONSISTENT", [slot, key])
            if slot == "MC4" and not historical:
                require(isinstance(finding.get("relevant_to_conclusion"), bool), "MC4_RELEVANCE_JUDGMENT_MISSING", key)
                require((finding["verdict"] == "SUPPORTED" and finding["relevant_to_conclusion"]) == (key in derived), "MC4_EVIDENCE_GOLD_INCONSISTENT", key)
            if not historical and slot in ("MC3", "MC5"):
                clauses = finding.get("clause_findings", [])
                require(len(clauses) >= 2, "REVIEW_ALL_CONJUNCTION_CLAUSES", [slot, key])
                for clause in clauses:
                    require(bool(clause.get("clause")) and bool(clause.get("verdict")), "EMPTY_REVIEW_CLAUSE")
                    check_evidence(clause["evidence"], turns, [slot, key, clause.get("clause")])
        if not slot.startswith("MC"):
            adjacent = row.get("ordering_adjacent_evidence", [])
            require(len(adjacent) == 3, "REVIEW_ADJACENT_RELATIONS_MISSING")
    return {"status": "PASS_REVIEW_CONTRACT", "root_id": root, "condition": arm,
            "decision_counts": dict(Counter(r["decision"] for r in rows.values()))}


def collect(*, roots=None):
    protocol = read(PROTOCOL)
    roots = protocol["roots"] if roots is None else roots
    decisions = []
    for root in roots:
        for arm in ARMS:
            rp = BASE / "question_reviews" / root / arm / "REVIEW.json"
            unit_errors = []
            try:
                source_admission(root, arm)
                validate_review(root, arm)
            except Exception as error:
                unit_errors.append(str(error))
            review = read(rp) if rp.exists() else {}
            rows = {r["slot_id"]: r for r in review.get("questions", [])}
            for slot in SLOTS:
                path = BASE / "questions" / root / arm / f"{slot}.json"
                short = path.with_name(f"{slot}_SHORTFALL.json")
                row = {"root_id": root, "condition": arm, "slot_id": slot,
                       "attempted": path.exists() or short.exists(), "status": "SHORTFALL",
                       "source_or_gold_ambiguity_failure": False}
                rr = rows.get(slot, {})
                if rp.exists():
                    row["review_binding"] = bind(rp)
                if path.exists():
                    row["question_binding"] = bind(path)
                    row["status"] = rr.get("decision", "FAIL")
                    row["source_or_gold_ambiguity_failure"] = bool(rr.get("source_or_gold_ambiguity_failure")) or rr.get("submitted_gold_matches") is False or rr.get("checks", {}).get("unique_key") == "FAIL"
                    # If the reviewer reports only a batch-level ambiguity, do
                    # not lose it merely because item metadata is incomplete.
                    if slot == next(iter(SLOTS)) and review.get("source_or_gold_ambiguity_failure") is True:
                        row["source_or_gold_ambiguity_failure"] = True
                    try:
                        mechanical(path)
                        row["mechanical_status"] = "PASS"
                    except Exception as error:
                        row.update(status="FAIL", mechanical_status="FAIL", mechanical_error=str(error))
                    if unit_errors and row["status"] == "ADMIT":
                        row["status"] = "FAIL"
                    row["draft_candidate_id"] = read(path).get("candidate_id")
                elif short.exists():
                    row["shortfall_binding"] = bind(short)
                    row["reason"] = read(short).get("reason", read(short).get("notes", "AUTHOR_SOURCE_SHORTFALL"))
                else:
                    row["reason"] = "SOURCE_FAILED_OR_QUESTION_NOT_PRODUCED; planned slot retained"
                if unit_errors:
                    row["unit_contract_errors"] = unit_errors
                decisions.append(row)
    return decisions


def engineering_gate(decisions=None):
    p = read(PROTOCOL)
    roots = p["first_wave_roots"]
    decisions = collect(roots=roots) if decisions is None else [r for r in decisions if r["root_id"] in roots]
    require(len(decisions) == len(roots) * 2 * len(SLOTS), "GATE_PLANNED_DENOMINATOR_CHANGED")
    def rate(rows):
        admitted = sum(r["status"] == "ADMIT" for r in rows)
        return {"admitted": admitted, "planned": len(rows), "fraction": admitted / len(rows)}
    overall = rate(decisions)
    by_root = {root: rate([r for r in decisions if r["root_id"] == root]) for root in roots}
    by_slot = {slot: rate([r for r in decisions if r["slot_id"] == slot]) for slot in SLOTS}
    ambiguity = sum(bool(r.get("source_or_gold_ambiguity_failure")) for r in decisions)
    # Source ambiguity is counted from original reviewer flags, separately from
    # question quality. Missing/unreviewed source already fails the yield gate.
    source_ambiguities = []
    for root in roots:
        for arm in ARMS:
            for path in source_path(root, arm).parent.glob("*/REVIEW.json"):
                r = read(path)
                if r.get("source_or_gold_ambiguity_failure") is True or "SOURCE_AMBIGUITY" in r.get("failure_codes", []):
                    source_ambiguities.append(bind(path))
    checks = {"overall_at_least_90pct": overall["fraction"] >= 0.90,
              "every_root_at_least_85pct": all(v["fraction"] >= 0.85 for v in by_root.values()),
              "every_slot_at_least_75pct": all(v["fraction"] >= 0.75 for v in by_slot.values()),
              "no_source_or_gold_ambiguity_failure": ambiguity + len(source_ambiguities) == 0}
    return {"schema_version": "oet1_engineering_gate_v1", "created_at_utc": now(),
            "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "overall": overall, "by_root": by_root, "by_slot": by_slot,
            "by_arm": {arm: rate([r for r in decisions if r["condition"] == arm]) for arm in ARMS},
            "question_ambiguity_failures": ambiguity, "source_ambiguity_bindings": source_ambiguities,
            "decisions": decisions, "planned_slots": 56, "solver_scores_read": False,
            "on_fail": "Stop automatic roots 5-8; QFP2-only pilot allowed if separately frozen/audited."}


def identity(q):
    return json.dumps({"source": q["source_binding"]["sha256"], "question": q["question"],
                       "choices": q.get("options") or q["items"]}, sort_keys=True, ensure_ascii=False)


def export():
    decisions = collect()
    public, private, seen = [], [], set()
    bindings = [bind(PROTOCOL), bind(RULES), bind(__file__), bind(gate.__file__)]
    for row in decisions:
        if row["status"] != "ADMIT":
            continue
        q = read(row["question_binding"]["path"])
        unique = identity(q)
        if unique in seen:
            row["status"] = "EXACT_WHOLE_DUPLICATE"
            continue
        seen.add(unique)
        opaque = "OET1_" + hashlib.sha256(q["candidate_id"].encode()).hexdigest()[:20]
        labels = list("ABCDEF" if row["slot_id"].startswith("MC") else "ABCD")
        choices = q.get("options") or q["items"]
        mapping = dict(zip(labels, q["display_order"]))
        inverse = {value: key for key, value in mapping.items()}
        operation = gate.public_operation(q["task_operation"])
        gate.require_public_selection(q["question"], operation, q["question_type"])
        public.append({k: q[k] for k in ("root_id", "condition", "question_type", "question", "source_binding")} |
                      {"candidate_id": opaque, "task_operation": operation,
                       "display": {key: choices[value] for key, value in mapping.items()}})
        private.append({"candidate_id": opaque, "draft_candidate_id": q["candidate_id"], "slot_id": row["slot_id"],
                        "question_type": q["question_type"], "canonical_gold": q["gold"],
                        "display_gold": [inverse[key] for key in q["gold"]],
                        "display_to_canonical": mapping, "question_binding": row["question_binding"]})
        row["candidate_id"] = opaque
        bindings.extend([row["question_binding"], row["review_binding"], q["source_binding"]])
    require(len(decisions) == 112, "FULL_PLANNED_DENOMINATOR_CHANGED")
    freeze = {"schema_version": "oet1_question_freeze_v1", "status": "FROZEN_BEFORE_SOLVER",
              "created_at_utc": now(), "planned_slots": 112, "admitted_questions": len(public),
              "decisions": decisions, "first_attempts_only": True, "semantic_revisions": 0,
              "solver_scores_read": False, "mc6_primary_training_excluded": True}
    qb = save(BASE / "QUESTION_FREEZE.json", freeze)
    pb = save(BASE / "inputs/PUBLIC_QUESTIONS.json", {"questions": public})
    gb = save(BASE / "inputs/PRIVATE_SCORING.json", {"questions": private})
    eb = save(BASE / "inputs/EXPORT_FREEZE.json", {"schema_version": "oet1_solver_export_v1",
         "status": "FROZEN_BEFORE_SOLVER", "created_at_utc": now(), "planned_slots": 112,
         "planned_root_ids": read(PROTOCOL)["roots"], "content_review_status": "PASS_ADMITTED_ONLY",
         "solver_score_filtering": False, "full_source_no_truncation": True,
         "public_binding": pb, "private_binding": gb, "question_freeze_binding": qb,
         "bindings": list({v["path"]: v for v in bindings}.values())})
    return {"planned": 112, "admitted": len(public), "failed_or_shortfall": 112-len(public), "export_binding": eb}


def training_record(question_path, review_path, layer):
    q = read(question_path)
    slot = q["slot_id"]
    require(slot in MAIN_SLOTS, "CONTROL_CANNOT_ENTER_TRAINING")
    return {"candidate_id": q["candidate_id"], "root_id": q["root_id"], "condition": q["condition"],
            "split": "train", "question_type": q["question_type"], "slot": slot, "slot_id": slot,
            "gold_cardinality": len(q["gold"]) if slot.startswith("MC") else None,
            "question_binding": bind(question_path), "source_binding": q["source_binding"],
            "content_review_binding": bind(review_path), "review_binding": bind(review_path),
            "paired_slot_key": f"{layer}:{q['root_id']}:{slot}", "historical_exposure": layer,
            "question": q, "source_full_no_truncation": True, "solver_score_selection": False}


def freeze_training(*, include_qfp2=True, gate_path=None):
    gate_path = Path(gate_path) if gate_path else BASE / "FIRST_WAVE_ENGINEERING_GATE.json"
    g = read(gate_path)
    require(g["status"] in ("PASS", "FAIL"), "GATE_NOT_TERMINAL")
    # Recompute against immutable content before relying on the gate.
    current = engineering_gate()
    require(current["status"] == g["status"] and current["checks"] == g["checks"] and current["decisions"] == g["decisions"], "ENGINEERING_GATE_DRIFT")
    records, excluded, matched = [], [], []
    if g["status"] == "PASS":
        frozen = read(BASE / "QUESTION_FREEZE.json")
        require(len(frozen["decisions"]) == 112, "NEW_POOL_PLANNED_DENOMINATOR_CHANGED")
        accepted = {(r["root_id"], r["condition"], r["slot_id"]): r for r in frozen["decisions"] if r["status"] == "ADMIT"}
        for root in read(PROTOCOL)["roots"]:
            for slot in MAIN_SLOTS:
                if not all((root, arm, slot) in accepted for arm in ARMS):
                    excluded.append({"root_id": root, "slot": slot, "reason": "BOTH_ARMS_NOT_INDEPENDENTLY_ADMITTED"})
                    continue
                matched.append(f"NEW_PROSPECTIVE:{root}:{slot}")
                for arm in ARMS:
                    validate_review(root, arm)
                    path = BASE / "questions" / root / arm / f"{slot}.json"
                    mechanical(path)
                    records.append(training_record(path, BASE / "question_reviews" / root / arm / "REVIEW.json", "NEW_PROSPECTIVE"))
    if include_qfp2:
        qf = read(QFP2 / "QUESTION_FREEZE.json")
        require(qf.get("first_attempts_only") is True and qf.get("solver_scores_read") is False, "QFP2_NOT_PRESOLVER_ADMISSION")
        accepted = {(r["root_id"], r["condition"], r["slot_id"]): r for r in qf["decisions"] if r["status"] == "ADMIT"}
        roots = read(QFP2 / "PROTOCOL_FREEZE.json")["roots"]
        for root in roots:
            for slot in MAIN_SLOTS:
                require(all((root, arm, slot) in accepted for arm in ARMS), "PREDECLARED_QFP2_PILOT_SHORTFALL")
                matched.append(f"PILOT_QFP2_HISTORICALLY_OBSERVED:{root}:{slot}")
                for arm in ARMS:
                    validate_review(root, arm, check_receipt=False, historical=True)
                    path = QFP2 / "questions" / root / arm / f"{slot}.json"
                    mechanical(path, historical=True)
                    records.append(training_record(path, QFP2 / "question_reviews" / root / arm / "REVIEW.json", "PILOT_QFP2_HISTORICALLY_OBSERVED"))
    counts = {arm: sum(r["condition"] == arm for r in records) for arm in ARMS}
    require(counts["GENERIC"] == counts["TARGETED"] and counts["GENERIC"] > 0, "EMPTY_OR_UNMATCHED_TRAINING_POOL")
    for key in matched:
        require(sum(r["paired_slot_key"] == key for r in records) == 2, "PAIRED_SLOT_MULTIPLICITY")
    require(len({r["candidate_id"] for r in records}) == len(records), "DUPLICATE_TRAINING_ID")
    per_arm = {arm: dict(Counter(r["slot"] for r in records if r["condition"] == arm)) for arm in ARMS}
    require(per_arm["GENERIC"] == per_arm["TARGETED"], "SLOT_BUDGET_MISMATCH")
    value = {"schema_version": "oet1_matched_training_pool_v1", "status": "FROZEN_BEFORE_TRAINING",
             "created_at_utc": now(), "mode": "NEW_PLUS_HISTORICAL_PILOT" if g["status"] == "PASS" else "QFP2_ONLY_PILOT_NEW_ROOT_GATE_FAILED",
             "new_root_main_objective_completed": False, "engineering_gate_status": g["status"],
             "engineering_gate_binding": bind(gate_path), "protocol_binding": bind(PROTOCOL),
             "records": records, "question_counts": counts, "slot_counts": per_arm,
             "matched_paired_slot_keys": matched, "excluded_new_slots": excluded,
             "planned_new_slots": 112, "new_controls_excluded": True,
             "foundation_included": False, "old_133_pool_included": False,
             "permutations_per_question": 2, "epochs": 2, "semantic_revisions": 0,
             "solver_scores_read_for_selection": False, "input_token_difference": "Must be measured by training schedule tokenizer; no content trimming allowed.",
             "bindings": [bind(__file__), bind(gate_path), bind(QFP2 / "QUESTION_FREEZE.json")] if include_qfp2 else [bind(__file__), bind(gate_path)]}
    return save(BASE / "training_pool/TRAINING_POOL_FREEZE.json", value)


AUTHOR_TASK = """
Author exactly one immutable first-attempt question or explicit source SHORTFALL
for every private slot MC2, MC3, MC4, MC5, MC6, ORDER1, ORDER2. Do not read or
request old questions, solver outputs, another condition, evaluation data, or
private weakness cards. Do not edit source facts. The complete source and rules
are embedded. Think and check exact quotations before final output; successful
semantic responses are never regenerated. Produce no files or tool calls.

Each multiple-select question has six canonical options A..F. The PRIVATE true
counts are MC2=2, MC3=3, MC4=4, MC5=5, MC6=6. Never disclose these counts or the
control label in stems, options, public operation names, or public rationale.
Use count-neutral stems such as "Select all supported statements". Preserve all
factual numbers in source constraints. MC6 uses an ordinary factual stem.

Required operations: MC2 person/entity-to-statement binding; MC3 updated-state
conjunctions (check every clause and request/confirmation/completion state);
MC4 evidence bearing on an explicitly named conclusion (true irrelevant options
can be excluded only when relevance is explicit); MC5 linked joint constraints,
not five duplicate rows with one changed ID; MC6 independently verified supported
records. MC5's one-foil limitation is inherent, so do not assert impossible
no-single-feature discrimination. Foils must be plausible, grounded changes of
actor/state/constraint, not unsupported external knowledge or lexical giveaways.

ORDER1: four unique reported explicit clock readings/constraints, smallest to
largest numeric clock-of-day. State that requests remain requests and separate
enquiries are not a chronology of completed events. Four distinct explicit times
must be in source. Items omit clocks and direct ordering cue words. Provide
canonical_clock_values:{I1:"HH:MM",I2:...,I3:...,I4:...} and source evidence.
ORDER2: four distinct disclosures within one identified recorded enquiry, by
source disclosure turns, not task procedure or event time. Provide three adjacent
relation_edges:[{from:"I1",to:"I3",relation:"disclosed_before",turn_refs:[...]}].
Items omit first/next/later/summary/final/before/after/then and similar cues.
Both orderings use canonical keys I1..I4; Gold is a unique canonical-key order.
For every question set a deterministic integer display_seed and display_order
that bijects all canonical keys. Ordering display differs from both insertion
order I1,I2,I3,I4 and Gold. Preserve semantic task distinctions.

Every candidate returns slot_id, question_type (multi_select/ordering), exact
task_operation from packet, question, options:{A:text,...F:text} or items:{I1:text,
...I4:text}, gold:[canonical keys], display_seed integer, display_order:[keys],
evidence:{every key:[{turn_ref,quote:EXACT continuous source substring}]},
private_option_notes:{every key:precise supported/refuted/unsupported reason},
construction_rationale. MC3 and MC5 additionally require clause_checks for EVERY
option, at least two individually checked clauses per option:
{key:[{clause,verdict,turn_refs:[...],evidence:[{turn_ref,quote}],role}]}.
Do not manufacture a source relation to fill quota. Instead return
{slot_id,status:"SHORTFALL",reason:specific missing relation,evidence:[...]}
for that slot. Shortfall is allowed and consumes its sole attempt.

Return ONLY {all_source_turns_read:true,source_turn_count:N,
questions:[all seven candidates/shortfalls],notes:brief truthful scope notes}.
Do not provide made-up model IDs, usage, timestamps, hashes, author/reviewer IDs;
the controller attaches observed metadata and exact bindings without changing
your text, options, Gold, evidence or decisions.
"""

REVIEW_TASK = """
Independently review the complete accepted source and the supplied first-attempt
candidates. You did not author the source or questions. Submitted Gold, private
author notes, quota labels, and counterpart candidates are hidden. Do not infer
admission from production yield. FIRST independently derive the supported keys
or unique ordering from source; your frozen response will then be compared to
the author's private Gold by the controller. Never rewrite a source, option,
stem, Gold, or evidence; no semantic retries. No files or tool calls.

Check all speaker/entity/status bindings and every clause of every option,
including false options. Quotes must be exact continuous substrings of the
specified source turn. Distinguish CONTRADICTED from merely UNSUPPORTED. For
MC_CONCLUSION_EVIDENCE assess both truth and relevance to its named conclusion;
a true but irrelevant item can be excluded only under an explicit relevance
stem. MC_STATE_UPDATE_CONJUNCTION and MC_JOINT_CONSTRAINT need at least two
independently checked clauses per option, not repeated identity rows. A single
foil can inherently limit discrimination; this alone is not a failure.

Clock ordering must have four distinct explicit readings and unique scope,
correct AM/PM normalization, no invented time or claims of actual travel. Check
requested vs confirmed vs completed. Recorded enquiry ordering must be one
specific enquiry's disclosure progression supported by three adjacent relations,
not latent event chronology or inferred procedure. Display must be a bijection,
actually shuffled and different from both insertion and inferred Gold order.
Check item clock/chronology cue leakage, count-neutral stems and neutral public
operation labels, actual semantic operation, diversity and reliable unique Gold.
Mechanical validity cannot substitute for semantic review. Do not score a model.

Return {full_source_read:true,source_turn_count:N,
independent_of_question_author:true,independent_of_source_author:true,
solver_scores_read:false,source_or_gold_ambiguity_failure:boolean,
questions:[exactly one per supplied review_key]}. Each row includes review_key,
decision:ADMIT/FAIL/SHORTFALL, independently_derived_gold:[canonical keys],
operation_alignment:PASS/FAIL,
checks:{grounding:PASS/FAIL,unique_key:PASS/FAIL,actual_operation:PASS/FAIL,
presentation:PASS/FAIL,foils:PASS/WARNING/NOT_APPLICABLE/FAIL},
item_findings:{EVERY canonical key:{verdict:SUPPORTED/CONTRADICTED/UNSUPPORTED,
notes:concrete source reasoning,evidence:[{turn_ref,quote}],
relevant_to_conclusion:boolean (required for conclusion-evidence items),
clause_findings:[{clause,verdict,evidence:[{turn_ref,quote}]}]
(>=2 clauses for every conjunction/joint-constraint option)}},
ordering_adjacent_evidence:[three relations with exact source evidence] for
ordering, source_or_gold_ambiguity_failure:boolean, failure_codes:[...],
notes:concrete decision reason. For a supplied SHORTFALL/MISSING candidate retain
SHORTFALL or FAIL, explain its missing support and do not invent replacement.
Report source/Gold ambiguity explicitly, including any nonunique ordering.
"""


def save_identical(path, value):
    """Crash recovery only: identical content is reusable, never replace text."""
    if Path(path).exists():
        require(read(path) == value, "IMMUTABLE_OUTPUT_MISMATCH", str(path))
        return bind(path)
    return save(path, value)


def run_cli_unit(root, arm, call, prompt_prefix):
    """Called by production_driver; only that caller owns actual CLI scheduling.

    The return JSON is untrusted content. Controller metadata is marked, raw
    responses are bound, and a complete semantic response is never requested
    again. Author/reviewer sessions must have different observed thread IDs.
    """
    check_unit(root, arm)
    source_admission(root, arm)
    u = BASE / "questions" / root / arm
    rp = BASE / "question_reviews" / root / arm / "REVIEW.json"
    if rp.exists():
        validate_review(root, arm)
        return {"status": "REUSED_HASH_VERIFIED_REVIEW", "review_binding": bind(rp)}
    source, refs, _ = check_source(source_path(root, arm))
    packet_path = BASE / "question_handoffs" / root / arm / "QUESTION_AUTHOR_INPUT.json"
    if not packet_path.exists():
        prepare(root, arm)
    packet = read(packet_path)
    require(packet["source_binding"] == bind(source_path(root, arm)), "QUESTION_PACKET_SOURCE_CHANGED")
    receipt_path = u / "AUTHOR_RECEIPT.json"
    if not receipt_path.exists():
        prompt = prompt_prefix("question author") + AUTHOR_TASK + "\n" + json.dumps({"packet": packet, "rules": RULES.read_text()}, ensure_ascii=False)
        response, result = call(f"{root}__{arm}__QUESTIONS_AUTHOR", prompt)
        require(response.get("all_source_turns_read") is True and response.get("source_turn_count") == len(refs), "QUESTION_AUTHOR_INCOMPLETE_READ")
        rows = {r["slot_id"]: r for r in response["questions"]}
        require(len(response["questions"]) == 7 and set(rows) == set(SLOTS), "AUTHOR_SLOT_DENOMINATOR_CHANGED")
        author_id = result["backend"]["thread_id"]
        require(bool(author_id), "OBSERVED_AUTHOR_SESSION_REQUIRED")
        candidate_bindings = []
        mechanical_results = []
        for slot in SLOTS:
            raw = rows[slot]
            if raw.get("status") == "SHORTFALL":
                require(bool(raw.get("reason")), "SHORTFALL_MISSING_REASON", slot)
                value = {**raw, "schema_version": "oet1_first_attempt_shortfall_v1",
                         "root_id": root, "condition": arm, "author_id": author_id,
                         "source_binding": bind(source_path(root, arm)),
                         "created_at_utc": result["finished_at_utc"],
                         "cli_raw_binding": result["raw_response"], "controller_metadata_only": True}
                candidate_bindings.append(save_identical(u / f"{slot}_SHORTFALL.json", value))
                continue
            # Only provenance/identity metadata are added; semantic fields are
            # copied verbatim from the one observed author response.
            value = {**raw, "schema_version": "oet1_private_question_v1",
                     "candidate_id": f"OET1__{root.split('-')[-1]}__{arm}__{slot}",
                     "root_id": root, "condition": arm, "author_id": author_id,
                     "author_type": "codex_cli_ai", "created_at_utc": result["finished_at_utc"],
                     "source_binding": bind(source_path(root, arm)),
                     "source_read_scope": {"all_turns_read": True, "turn_count": len(refs),
                                           "controller_expanded_from_author_attestation": True},
                     "formal_weakness_admission": False, "causal_feedback_benefit_claim": False,
                     "semantic_admission": "PENDING_INDEPENDENT_REVIEW",
                     "cli_raw_binding": result["raw_response"], "controller_metadata_only": True}
            path = u / f"{slot}.json"
            candidate_bindings.append(save_identical(path, value))
            try:
                mechanical_results.append(mechanical(path))
            except Exception as error:
                mechanical_results.append({"slot_id": slot, "status": "FAIL", "error": str(error), "question_binding": bind(path)})
        receipt = {"schema_version": "oet1_first_attempt_author_receipt_v1", "root_id": root, "condition": arm,
                   "author_id": author_id, "created_at_utc": result["finished_at_utc"],
                   "source_binding": bind(source_path(root, arm)),
                   "source_read_scope": {"all_turns_read": True, "turn_count": len(refs)},
                   "full_turn_refs_read": refs, "controller_expanded_turn_refs": True,
                   "author_read_attestation": {k: response[k] for k in ("all_source_turns_read", "source_turn_count")},
                   "candidate_bindings": candidate_bindings, "input_bindings": [bind(packet_path), bind(RULES)],
                   "cli_result": result, "mechanical_results": mechanical_results,
                   "first_attempts_only": True, "semantic_revisions": 0}
        save(receipt_path, receipt)
    receipt, _, _ = validate_receipt(root, arm)
    review_input = BASE / "question_reviews" / root / arm / "GOLD_HIDDEN_REVIEW_INPUT.json"
    packet = review_packet(root, arm)
    save_identical(review_input, packet)
    prompt = prompt_prefix("independent question reviewer") + REVIEW_TASK + "\n" + json.dumps(packet, ensure_ascii=False)
    response, result = call(f"{root}__{arm}__QUESTIONS_REVIEW", prompt)
    reviewer_id = result["backend"]["thread_id"]
    require(bool(reviewer_id) and reviewer_id != receipt["author_id"], "FRESH_INDEPENDENT_REVIEW_SESSION_REQUIRED")
    source_authors = {read(path).get("author_id") for path in source_path(root, arm).parent.glob("*/DRAFT.json")}
    require(reviewer_id not in source_authors, "SOURCE_AUTHOR_CANNOT_REVIEW_QUESTIONS")
    require(response.get("full_source_read") is True and response.get("source_turn_count") == len(refs), "QUESTION_REVIEW_INCOMPLETE_READ")
    mapping = {"QR_" + hashlib.sha256(f"{root}:{arm}:{slot}".encode()).hexdigest()[:18]: slot for slot in SLOTS}
    require(len(response["questions"]) == 7 and {r["review_key"] for r in response["questions"]} == set(mapping), "REVIEW_KEY_DENOMINATOR_CHANGED")
    derivation = {**response, "questions": [{**r, "slot_id": mapping[r["review_key"]]} for r in response["questions"]],
                  "turn_refs_read": refs, "controller_expanded_turn_refs": True,
                  "cli_raw_binding": result["raw_response"], "cli_result": result,
                  "independently_derived_before_gold_comparison": True,
                  "source_binding": bind(source_path(root, arm)),
                  "created_at_utc": result["finished_at_utc"]}
    derivation_path = rp.with_name("INDEPENDENT_DERIVATION.json")
    save_identical(derivation_path, derivation)
    sealed = seal_review(root, arm, derivation_path, reviewer_id, review_input)
    return {"status": "REVIEWED", "review_binding": sealed,
            "decisions": dict(Counter(r["decision"] for r in read(rp)["questions"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "mechanical", "review-packet", "seal-review", "validate-review", "collect", "gate", "export", "freeze-training"))
    parser.add_argument("--root")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--path")
    parser.add_argument("--output")
    parser.add_argument("--reviewer-id")
    parser.add_argument("--review-input")
    parser.add_argument("--gate-path")
    parser.add_argument("--no-qfp2", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(args.root, args.arm)
    elif args.action == "mechanical":
        value = mechanical(args.path)
    elif args.action == "review-packet":
        value = review_packet(args.root, args.arm)
    elif args.action == "seal-review":
        value = seal_review(args.root, args.arm, args.path, args.reviewer_id, args.review_input)
    elif args.action == "validate-review":
        value = validate_review(args.root, args.arm)
    elif args.action == "collect":
        value = {"planned_slots": 112, "decisions": collect()}
    elif args.action == "gate":
        value = engineering_gate()
    elif args.action == "export":
        value = export()
    else:
        value = freeze_training(include_qfp2=not args.no_qfp2, gate_path=args.gate_path)
    if args.output:
        print(json.dumps(save(args.output, value), ensure_ascii=False))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
