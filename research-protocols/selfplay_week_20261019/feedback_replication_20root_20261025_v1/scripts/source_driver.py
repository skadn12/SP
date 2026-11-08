"""Bounded shared-source production for the frozen additional sixteen Train roots.

Run from any directory with ``python scripts/source_driver.py``.  Each root has
one complete six-scene author call, a fresh independent review, and at most one
revision/recheck pair.  Model calls are delegated to codex_calls; no F cards or
Solver outputs are read.  Immutable per-root outcomes make restart explicit.
"""
from __future__ import annotations

import concurrent.futures
import copy
import fcntl
import json
from pathlib import Path
import traceback

import codex_calls as cc

BASE = Path(__file__).resolve().parents[1]
CHECKS = ["time", "location", "actor_identity", "plan_vs_completion",
          "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity"]
SCENES = [f"R03:{route}:S{number:02}" for route in "AB" for number in range(1, 4)]
RECEIPT_SHA = "62203a0b34c9ea8d742cefa70e0424d5deebb3333ad6d189a65872890c6bcec9"


def immutable(path, value):
    """Exclusive publication; restart can reuse only byte-independent equal JSON."""
    path = Path(path)
    if path.exists():
        if cc.read(path) != value:
            raise RuntimeError(f"Immutable artifact drift: {path}")
    else:
        cc.write(path, value)
    return cc.bind(path)


def verify(binding):
    if cc.bind(binding["path"]) != binding:
        raise RuntimeError(f"Input/cache binding drift: {binding['path']}")


def checked(binding):
    verify(binding)
    return cc.read(binding["path"])


def load_inputs(row):
    packet = checked(row["material_packet_binding"])
    parent = checked(row["frozen_initial_source_binding"])
    checked(row["source_binding"])
    if packet["parent_source_binding"] != row["frozen_initial_source_binding"]:
        raise ValueError("Packet/selection parent mismatch")
    if packet["root_id"] != row["root_id"] or parent["root_id"] != row["root_id"]:
        raise ValueError("Root mismatch")
    if packet["split"] != "train" or row["split"] != "train":
        raise ValueError("Non-Train root")
    if packet["scene_sequence"] != SCENES or len(packet["native_events"]) != 3:
        raise ValueError("Frozen scene/native-event contract mismatch")
    natives = [{"reservation": event, "complete_record": checked(event["binding"])}
               for event in packet["native_events"]]
    # The prompt contains only neutral task contents, not their historical origins.
    safe_packet = {k: copy.deepcopy(v) for k, v in packet.items()
                   if k not in ("replication_provenance", "output_contract")}
    return packet, parent, natives, safe_packet


def normalize(raw, packet):
    """Normalize reference/origin metadata only; never alter utterances/speakers."""
    draft = copy.deepcopy(raw)
    warnings = []
    if not isinstance(draft.get("scenes"), list):
        return draft, warnings
    for scene in draft["scenes"]:
        if not isinstance(scene, dict) or scene.get("scene_ref") not in SCENES:
            continue
        scene_ref = scene["scene_ref"]
        expected_route = scene_ref.split(":")[1]
        if scene.get("route") != expected_route:
            warnings.append({"field": f"{scene_ref}.route", "from": scene.get("route"), "to": expected_route})
        scene["route"] = expected_route
        if not isinstance(scene.get("turns"), list):
            continue
        for number, turn in enumerate(scene["turns"], 1):
            if not isinstance(turn, dict):
                continue
            metadata = {"turn_ref": f"{scene_ref}:T{number:03}", "scene_ref": scene_ref,
                        "session_ref": f"{packet['root_id']}:{scene_ref}",
                        "origin": "codex_synthetic_extension"}
            for key, value in metadata.items():
                if key in turn and turn[key] != value:
                    warnings.append({"field": f"{scene_ref}.turn{number}.{key}", "from": turn[key], "to": value})
                turn[key] = value
    return draft, warnings


def turns_for(draft):
    return [turn for scene in draft.get("scenes", []) if isinstance(scene, dict)
            for turn in (scene.get("turns") or []) if isinstance(turn, dict)]


def source_index(parent, draft):
    return {t.get("turn_ref"): t.get("text", "")
            for t in parent["turns"] + turns_for(draft) if t.get("turn_ref")}


def exact_anchor(anchor, index):
    return (isinstance(anchor, dict) and isinstance(anchor.get("quote"), str)
            and bool(anchor["quote"].strip()) and anchor.get("turn_ref") in index
            and anchor["quote"] in index[anchor["turn_ref"]])


def validate_draft(draft, packet, parent, natives):
    errors, stats = [], []
    if draft.get("root_id") != packet["root_id"]:
        errors.append("root_id mismatch")
    if draft.get("source_status") != packet["source_status"]:
        errors.append("source_status mismatch")
    scenes = draft.get("scenes")
    if not isinstance(scenes, list) or [s.get("scene_ref") if isinstance(s, dict) else None for s in scenes] != SCENES:
        errors.append("Exactly six scenes in frozen A1-A3/B1-B3 sequence required")
        scenes = scenes if isinstance(scenes, list) else []
    speakers = {t["speaker"] for t in parent["turns"]}
    for scene in scenes:
        if not isinstance(scene, dict):
            errors.append("Scene must be object")
            continue
        scene_turns = scene.get("turns")
        if not isinstance(scene_turns, list) or len(scene_turns) != 8:
            errors.append(f"{scene.get('scene_ref')}: exactly 8 turns required")
            continue
        words = 0
        for turn in scene_turns:
            if not isinstance(turn, dict) or turn.get("speaker") not in speakers or not isinstance(turn.get("text"), str) or not turn["text"].strip():
                errors.append(f"{scene.get('scene_ref')}: valid text/existing speaker required")
                continue
            words += len(turn["text"].split())
        stats.append({"scene_ref": scene.get("scene_ref"), "turn_count": len(scene_turns),
                      "average_english_words_per_turn": words / 8})
        if words > 8 * 18:
            errors.append(f"{scene.get('scene_ref')}: average words exceeds 18 ({words / 8})")
    new_turns = turns_for(draft)
    if len(new_turns) != 48:
        errors.append("Exactly 48 new turns required")
    refs = [t.get("turn_ref") for t in parent["turns"] + new_turns]
    if None in refs or len(refs) != len(set(refs)):
        errors.append("Turn references must be nonempty and globally unique")
    receipt = draft.get("read_receipt", {})
    expected = {"parent_turn_count": len(parent["turns"]), "parent_first_turn_ref": parent["turns"][0]["turn_ref"],
                "parent_last_turn_ref": parent["turns"][-1]["turn_ref"], "native_record_count": 3,
                "native_turn_counts": {n["reservation"]["source_id"]: len(n["complete_record"]["turns"]) for n in natives}}
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        errors.append("Complete-input read_receipt missing/mismatched")
    # Every native utterance is either anchored in its B scene or explicitly
    # classified as nonmaterial by the author, then independently reviewed.
    mapping = draft.get("native_evidence")
    mapping = mapping if isinstance(mapping, list) else []
    expected_keys = {(n["reservation"]["source_id"], str(t["turn_id"])): (n, t)
                     for n in natives for t in n["complete_record"]["turns"]}
    seen = set()
    index = source_index(parent, draft)
    for entry in mapping:
        if not isinstance(entry, dict):
            errors.append("Native mapping row must be object")
            continue
        key = (entry.get("source_id"), str(entry.get("native_turn_id")))
        if key not in expected_keys or key in seen:
            errors.append(f"Unknown/duplicate native mapping {key}")
            continue
        seen.add(key)
        native, turn = expected_keys[key]
        quote = entry.get("native_quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in turn["utterance"]:
            errors.append(f"Invalid exact native quote {key}")
        if entry.get("disposition") == "OMITTED_NONMATERIAL":
            if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip():
                errors.append(f"Omission needs rationale {key}")
        elif entry.get("disposition") == "PRESERVED":
            anchors = entry.get("source_evidence")
            if not isinstance(anchors, list) or not anchors:
                errors.append(f"Preserved native turn needs source evidence {key}")
            else:
                for anchor in anchors:
                    if not exact_anchor(anchor, index) or not str(anchor.get("turn_ref", "")).startswith(native["reservation"]["scene_ref"] + ":T"):
                        errors.append(f"Invalid exact destination quote {key}")
        else:
            errors.append(f"Unknown native mapping disposition {key}")
    if seen != set(expected_keys):
        errors.append(f"Every native turn needs mapping; missing {len(set(expected_keys) - seen)}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "scene_statistics": stats,
            "parent_turn_count": len(parent["turns"]), "new_turn_count": len(new_turns),
            "full_solver_prompt_budget_checked_here": False,
            "budget_note": "Final question-plus-source retokenization is mandatory in downstream Solver preflight; no source truncation."}


def validate_review(review, packet, parent, draft, natives):
    errors = []
    if review.get("root_id") != packet["root_id"]:
        errors.append("Review root mismatch")
    if review.get("decision") not in ("PASS", "REVISE", "FAIL"):
        errors.append("Review decision must be PASS/REVISE/FAIL")
    if review.get("complete_inputs_read") is not True:
        errors.append("Reviewer did not acknowledge complete inputs")
    index = source_index(parent, draft)
    dimensions = review.get("dimensions", {})
    if not isinstance(dimensions, dict) or set(dimensions) != set(CHECKS):
        errors.append("Exactly eight review dimensions required")
        dimensions = dimensions if isinstance(dimensions, dict) else {}
    for name in CHECKS:
        dim = dimensions.get(name, {})
        if not isinstance(dim, dict) or dim.get("status") not in ("PASS", "WARNING", "FAIL"):
            errors.append(f"Invalid dimension: {name}")
            continue
        if not isinstance(dim.get("finding"), str) or not dim["finding"].strip():
            errors.append(f"Missing specific finding: {name}")
        anchors = dim.get("source_evidence")
        if not isinstance(anchors, list) or not anchors or not all(exact_anchor(a, index) for a in anchors):
            errors.append(f"Invalid exact source evidence: {name}")
    coverage = review.get("native_coverage", [])
    if not isinstance(coverage, list):
        coverage = []
    expected = {n["reservation"]["source_id"]: n for n in natives}
    seen = set()
    for row in coverage:
        if not isinstance(row, dict) or row.get("source_id") not in expected or row.get("source_id") in seen:
            errors.append("Invalid/duplicate native review coverage")
            continue
        seen.add(row["source_id"])
        native = expected[row["source_id"]]
        native_index = {str(t["turn_id"]): t["utterance"] for t in native["complete_record"]["turns"]}
        native_anchor = {"turn_ref": str(row.get("native_turn_id")), "quote": row.get("native_quote")}
        if not exact_anchor(native_anchor, native_index):
            errors.append(f"Invalid reviewer native quote {row['source_id']}")
        anchors = row.get("source_evidence")
        if not isinstance(anchors, list) or not anchors or not all(exact_anchor(a, index) and a["turn_ref"].startswith(native["reservation"]["scene_ref"] + ":T") for a in anchors):
            errors.append(f"Invalid reviewer native destination evidence {row['source_id']}")
        if row.get("all_native_turns_and_omissions_checked") is not True or row.get("status") not in ("PASS", "FAIL"):
            errors.append(f"Incomplete native review {row['source_id']}")
    if seen != set(expected):
        errors.append("Review must cover all three native records")
    for key in ("blocking_issues", "warnings", "revision_instructions"):
        if not isinstance(review.get(key), list):
            errors.append(f"Review {key} must be a list")
    semantic_pass = (review.get("decision") == "PASS" and not review.get("blocking_issues")
                     and not any(d.get("status") == "FAIL" for d in dimensions.values() if isinstance(d, dict))
                     and all(r.get("status") == "PASS" for r in coverage if isinstance(r, dict)))
    return {"status": "PASS" if not errors and semantic_pass else "FAIL", "contract_errors": errors,
            "semantic_pass": semantic_pass, "review_is_independent_ai_not_human": True}


AUTHOR_RULES = """
Create ONE complete JSON draft with all six scenes, never separate scene calls.
Append 48 turns, exactly eight turns per scene; average English whitespace words
per turn <=18 separately in each scene (<=144 words each). Use only the existing
speakers. A1-A3 continue existing events in a fresh planning cycle. B1-B3 report,
respectively, the three complete reserved service records between those same
speakers. Service/place names present in native records are allowed; new named
people, invented gender, unsupported service facts or implied completed trips
are forbidden. Preserve explicit clocks, numbers, corrections, refusals, unknowns,
and proposal/booking/completion states. A source record can itself contradict an
earlier statement: report its correction/uncertainty without silently repairing
the record. Native boilerplate greetings/closings may be omitted with rationale;
material facts must not be mislabeled as nonmaterial. Do not copy database frames
as dialogue. Keep dialogue natural, coherent across scenes, not quota explanations.
Do not generate questions, answers, expected errors, feedback cards, or task labels.
Do not reproduce/alter the parent; it is preserved by the assembler.
Return keys root_id, source_status, scenes, native_evidence, read_receipt, warnings.
scenes: [{scene_ref:'R03:A:S01',route:'A',turns:[{turn_ref:'R03:A:S01:T001',
speaker:existing_name,text:utterance,origin:'codex_synthetic_extension'}, ...]}, ...].
native_evidence contains EXACTLY one row for EVERY native turn across all records:
{source_id, native_turn_id: string, native_quote: exact nonempty substring of that
native utterance, disposition:'PRESERVED'|'OMITTED_NONMATERIAL', rationale:string,
source_evidence:[{turn_ref,quote:exact nonempty substring of new dialogue}]}. For
PRESERVED use >=1 source_evidence anchors in that record's assigned B scene; for
OMITTED_NONMATERIAL provide the reason, with source_evidence allowed empty.
read_receipt: {parent_turn_count, parent_first_turn_ref, parent_last_turn_ref,
native_record_count:3,native_turn_counts:{source_id:integer,...}}.
warnings is a list. Return only one JSON object, no tool calls.
"""

REVIEW_RULES = """
Independently audit the COMPLETE parent, COMPLETE three native records and COMPLETE
new draft. Do not trust author read receipts/self-checks; check material omissions,
native mapping dispositions and unsupported inferences. This is source quality,
never Solver difficulty or question authoring. Eight dimensions only:
time (clock/calendar consistency), location, actor_identity (no invented people or
gender), plan_vs_completion (requests/proposals/confirmation/completion distinct),
cross_scene_continuity (parent and all six scenes), semantic_repetition,
naturalness, event_fidelity (including every native turn and all author omissions).
For EVERY dimension give status PASS/WARNING/FAIL, a specific finding, and >=1
source_evidence exact quote anchors {turn_ref,quote} against parent or draft.
Warnings are allowed only when genuinely nonblocking; FAIL prevents admission.
Return {root_id,complete_inputs_read:true,decision:'PASS'|'REVISE'|'FAIL',
dimensions:{time:{status,finding,source_evidence:[{turn_ref,quote}]},...eight...},
native_coverage:[{source_id,status:'PASS'|'FAIL',all_native_turns_and_omissions_checked:true,
native_turn_id:string,native_quote:exact nonempty substring,source_evidence:[{turn_ref,quote}],
finding:string},...exactly three...],blocking_issues:[],warnings:[],revision_instructions:[]}.
Each native_coverage row anchors at least one material native statement and its
corresponding exact new dialogue quote in the assigned B scene, while the finding
must assess the full record. Source and native evidence are checked by program.
Do not rewrite dialogue. PASS only if no blocking issue and all eight dimensions
plus all three records are acceptable. Return exactly one JSON object, no tools.
"""


def root_outcome(row):
    root_id = row["root_id"]
    work = BASE / "source_authoring" / root_id
    done = work / "OUTCOME.json"
    if done.exists():
        outcome = cc.read(done)
        if outcome.get("material_packet_binding") != row["material_packet_binding"]:
            raise RuntimeError("Cached source outcome input drift")
        for binding in outcome.get("artifacts", []):
            checked(binding)
        if outcome.get("source_binding"):
            checked(outcome["source_binding"])
        return outcome
    artifacts = []
    outcome = {"schema_version": "fgr20_source_root_outcome_v1", "root_id": root_id,
               "material_packet_binding": row["material_packet_binding"], "status": "FAILED",
               "max_revisions": 1, "substitution_allowed": False}
    try:
        packet, parent, natives, safe_packet = load_inputs(row)
        embedded = {"neutral_packet": safe_packet, "complete_parent_source": parent,
                    "complete_native_records": natives}
        common = json.dumps(embedded, ensure_ascii=False, separators=(",", ":"))
        draft = review = mechanical = audit = None
        for attempt in range(2):
            if cc.stop_generation():
                raise RuntimeError("Generation cutoff/request reached; root retained as missing")
            if attempt == 0:
                prompt = cc.prefix("neutral shared-source author") + AUTHOR_RULES + "\nCOMPLETE_INPUTS:\n" + common
            else:
                correction = {"previous_normalized_draft": draft, "mechanical_validation": mechanical,
                              "independent_review": review, "review_contract_validation": audit}
                prompt = (cc.prefix("neutral shared-source author performing the one allowed revision") + AUTHOR_RULES
                          + "\nThis is the only revision. Resolve all blocking findings; preserve all valid material.\nCOMPLETE_INPUTS:\n"
                          + common + "\nPREVIOUS_DRAFT_AND_FEEDBACK:\n" + json.dumps(correction, ensure_ascii=False, separators=(",", ":")))
            label = f"{root_id.replace('-', '_')}_source_{'draft' if attempt == 0 else 'revision'}"
            raw, call_result = cc.call(label, prompt)
            artifacts.append(immutable(work / f"AUTHOR_{attempt}.json", raw))
            artifacts.append(immutable(work / f"AUTHOR_CALL_{attempt}.json", call_result))
            draft, metadata_warnings = normalize(raw, packet)
            artifacts.append(immutable(work / f"DRAFT_{attempt}.json", draft))
            artifacts.append(immutable(work / f"METADATA_NORMALIZATION_{attempt}.json", metadata_warnings))
            mechanical = validate_draft(draft, packet, parent, natives)
            artifacts.append(immutable(work / f"MECHANICAL_{attempt}.json", mechanical))
            if cc.stop_generation():
                raise RuntimeError("Source draft returned after generation cutoff; no unreviewed admission")
            review_input = {"complete_inputs": embedded, "complete_normalized_draft": draft,
                            "mechanical_validation": mechanical}
            review_prompt = (cc.prefix("independent shared-source reviewer") + REVIEW_RULES + "\nCOMPLETE_REVIEW_INPUTS:\n"
                             + json.dumps(review_input, ensure_ascii=False, separators=(",", ":")))
            review, review_call = cc.call(label + "_independent_review", review_prompt)
            artifacts.append(immutable(work / f"REVIEW_{attempt}.json", review))
            artifacts.append(immutable(work / f"REVIEW_CALL_{attempt}.json", review_call))
            audit = validate_review(review, packet, parent, draft, natives)
            artifacts.append(immutable(work / f"REVIEW_VALIDATION_{attempt}.json", audit))
            if mechanical["status"] == audit["status"] == "PASS":
                final = copy.deepcopy(parent)
                new_turns = turns_for(draft)
                final.update(schema_version="feedback_closed_loop_r03_source_v1", source_status=packet["source_status"],
                             scene_ref=SCENES[-1], input_source_sha256=packet["parent_source_binding"]["sha256"],
                             draft_sha256=cc.bind(work / f"DRAFT_{attempt}.json")["sha256"],
                             previous_turn_count=len(parent["turns"]), new_turn_count=48,
                             cumulative_new_turn_count=parent["cumulative_new_turn_count"] + 48,
                             turns=copy.deepcopy(parent["turns"]) + new_turns)
                final["closed_loop_provenance"] = {
                    "round": "R03", "mode": "A_then_B_sequential_cumulative",
                    "parent_source_binding": packet["parent_source_binding"],
                    "draft_binding": cc.bind(work / f"DRAFT_{attempt}.json"),
                    "review_binding": cc.bind(work / f"REVIEW_{attempt}.json"),
                    "native_events": packet["native_events"], "warnings": review["warnings"],
                    "metadata_normalization_warnings": metadata_warnings,
                    "status": "SOURCE_REVIEWED_BEFORE_QUESTION_AUTHORING",
                    "extension_purpose": "EXPLORATORY_NEUTRAL_TRANSFER_NOT_ROOT_SPECIFIC_DIAGNOSIS",
                    "historical_parent_seen_by_M_clean": True, "unseen_root_claim_permitted": False,
                    "same_source_for_F_and_G": True, "revision_count": attempt}
                if final["turns"][:len(parent["turns"])] != parent["turns"]:
                    raise AssertionError("Parent prefix modified")
                if len(final["turns"]) != final["baseline_turn_count"] + final["cumulative_new_turn_count"]:
                    raise AssertionError("Cumulative turn-count mismatch")
                source_binding = immutable(BASE / "sources" / f"{root_id}.json", final)
                outcome.update(status="PASS", revision_count=attempt, source_binding=source_binding,
                               final_draft_binding=cc.bind(work / f"DRAFT_{attempt}.json"),
                               final_review_binding=cc.bind(work / f"REVIEW_{attempt}.json"),
                               parent_prefix_preserved=True, warnings=review["warnings"])
                break
        else:
            outcome.update(failure_reason="Source rejected after one revision/recheck", revision_count=1,
                           mechanical_errors=mechanical["errors"], review_validation=audit)
    except Exception as exc:
        outcome.update(failure_reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    outcome["artifacts"] = artifacts
    immutable(done, outcome)
    cc.progress("SOURCE_ROOT_FINISHED", root_id=root_id, status=outcome["status"],
                reason=outcome.get("failure_reason"))
    return outcome


def validate_preparation():
    receipt_path = BASE / "SOURCE_PREPARATION_RECEIPT.json"
    if cc.bind(receipt_path)["sha256"] != RECEIPT_SHA:
        raise RuntimeError("Source preparation receipt differs from frozen receipt")
    receipt = cc.read(receipt_path)
    for binding in receipt["artifacts"]:
        verify(binding)
    selection = cc.read(BASE / "PARENT_SOURCE_SELECTION.json")
    roots = selection["roots"]
    config = cc.read(BASE / "CONFIG.json")
    if [r["root_id"].rsplit("-", 1)[1] for r in roots] != config["roots"] or len(roots) != 16:
        raise RuntimeError("Frozen additional-sixteen selection/config mismatch")
    for row in roots:
        load_inputs(row)
    return roots


def finish(roots, outcomes):
    ordered = {r["root_id"]: r for r in outcomes}
    if set(ordered) != {r["root_id"] for r in roots}:
        raise ValueError("Every frozen root must retain an explicit outcome")
    rows = [ordered[r["root_id"]] for r in roots]
    passing = [r for r in rows if r["status"] == "PASS"]
    selection = {"schema_version": "closed_loop_source_selection_v1", "split": "train",
                 "roots": [{"root_id": r["root_id"], **r["source_binding"]} for r in passing]}
    selection_binding = immutable(BASE / "SOURCE_SELECTION_FINAL.json", selection)
    failures = {"schema_version": "fgr20_source_failures_v1", "planned_roots": 16,
                "failed_roots": [r for r in rows if r["status"] != "PASS"],
                "substitutions": [], "retries_after_terminal_outcome_permitted": False,
                "native_reservations_remain_consumed": True}
    failure_binding = immutable(BASE / "SOURCE_FAILURES.json", failures)
    freeze = {"schema_version": "fgr20_source_freeze_v1", "status": "PASS" if len(passing) == 16 else "PARTIAL",
              "planned_root_count": 16, "passed_root_count": len(passing), "failed_root_count": 16 - len(passing),
              "source_selection_binding": selection_binding, "failures_binding": failure_binding,
              "parent_selection_binding": cc.bind(BASE / "PARENT_SOURCE_SELECTION.json"),
              "root_outcomes": [cc.bind(BASE / "source_authoring" / r["root_id"] / "OUTCOME.json") for r in rows],
              "training": False, "old_sources_modified": False, "original_four_rerun": False,
              "same_shared_source_for_paired_F_G": True, "source_prefix_preservation_checked": True,
              "full_solver_prompt_budget_deferred_to_mandatory_preflight": True}
    immutable(BASE / "SOURCE_FREEZE.json", freeze)
    cc.progress("SOURCE_STAGE_FROZEN", passed=len(passing), failed=16 - len(passing))
    return freeze


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / ".source_driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roots = validate_preparation()
        if (BASE / "SOURCE_FREEZE.json").exists():
            freeze = cc.read(BASE / "SOURCE_FREEZE.json")
            for key in ("source_selection_binding", "failures_binding", "parent_selection_binding"):
                checked(freeze[key])
            for binding in freeze["root_outcomes"]:
                checked(binding)
            # Validate saved source and all intermediate bindings as well.
            for row in roots:
                root_outcome(row)
            cc.progress("SOURCE_STAGE_REUSED", status=freeze["status"])
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(root_outcome, roots))
        finish(roots, outcomes)


if __name__ == "__main__":
    main()
