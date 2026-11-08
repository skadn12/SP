#!/usr/bin/env python3
"""Deterministically assemble reviewed F/G candidates for runner.prepare."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = __import__("importlib.util").util.spec_from_file_location("nq15_runner", HERE / "runner.py")
runner = __import__("importlib.util").util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
legacy = runner.legacy

BASE = HERE / "review/CANDIDATES_NORMALIZED_PRE_REVIEW.json"
SLOTS = tuple(runner.SLOTS)
ROOTS = runner.ROOTS
HOLD_IDS = {
    "NQ15-F-179186-MC5",
    "NQ15-F-25652-MC5",
    "NQ15-G-109164-MC3",
    "NQ15-G-25652-MC4",
    "NQ15-G-45849-MC4",
    "NQ15-G-55788-MC4",
}
HOLD_REASONS = {
    "NQ15-F-179186-MC5": "Strict blind review found only three source-supported causal options for the five-answer slot.",
    "NQ15-F-25652-MC5": "Strict blind review found the fifth causal link is not reliably explicit in the authorized source scope.",
    "NQ15-G-109164-MC3": "Revision 1 has no independent blind review bound to the revised candidate; retain HOLD pending that review.",
    "NQ15-G-25652-MC4": "Independent Gold BCDF does not match candidate Gold ABCD.",
    "NQ15-G-45849-MC4": "Independent review found five reasonable answers for the four-answer slot.",
    "NQ15-G-55788-MC4": "Independent review/adjudication found Gold ABC, which does not match candidate Gold ABCD.",
}
CONTROL_ADJUDICATION_IDS = {
    "NQ15-F-25652-MC6",
    "NQ15-G-194901-MC6",
    "NQ15-G-238359-MC6",
    "NQ15-G-246750-MC6",
    "NQ15-G-55788-MC6",
}
HASH_RECONCILIATION_PATH = HERE / "review/F_DISCREPANCIES_HASH_RECONCILIATION.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_id(value: str) -> str:
    if value.startswith("NQ15-"):
        return value
    for arm in ("F", "G"):
        prefix = f"{arm}-batch2-cc-train-episode-"
        if value.startswith(prefix):
            tail = value[len(prefix):]
            root, slot = tail.rsplit("-", 1)
            return f"NQ15-{arm}-{root}-{slot}"
        prefix = f"{arm}-batch2-"
        if value.startswith(prefix):
            tail = value[len(prefix):]
            root, slot = tail.rsplit("-", 1)
            return f"NQ15-{arm}-{root}-{slot}"
    raise ValueError(f"Unrecognized candidate ID: {value}")


def normalized_options(value):
    if isinstance(value, dict):
        return {label: text for label, text in value.items()}
    if isinstance(value, list):
        return {item["label"]: item["text"] for item in value}
    raise ValueError("Options must be a mapping or labeled list")


def load_review_sources():
    reviews = {}
    paths = []
    groups = (
        ("F_batch1", "F_batch1_REVIEWER.json", "F_batch1_REVIEW_MAP_PRIVATE.json"),
        ("F_batch2", "F_batch2_REVIEWER.json", "F_batch2_REVIEW_MAP_PRIVATE.json"),
        ("F_batch3", "F_batch3_REVIEWER.json", "F_batch3_REVIEW_MAP_PRIVATE.json"),
        ("G_batch1_A", "G_batch1_REVIEWER_A.json", "G_batch1_REVIEW_MAP_PRIVATE.json"),
        ("G_batch1_B", "G_batch1_REVIEWER_B.json", "G_batch1_REVIEW_MAP_PRIVATE.json"),
        ("G_batch2", "G_batch2_REVIEWER.json", "G_batch2_REVIEW_MAP_PRIVATE.json"),
    )
    for label, review_name, map_name in groups:
        review_path = HERE / "review" / review_name
        map_path = HERE / "review" / map_name
        raw = read(review_path)
        mapping = {item["review_id"]: item for item in read(map_path)["mapping"]}
        for record in raw["reviews"]:
            mapped = mapping[record["review_id"]]
            cid = canonical_id(mapped["candidate_id"])
            if cid in reviews:
                raise ValueError(f"More than one base review mapped to {cid}")
            reviews[cid] = {
                "record": record,
                "reviewer_id": record.get("reviewer_id") or mapped.get("reviewer_id"),
                "review_id": record["review_id"],
                "source": str(review_path.relative_to(HERE)),
                "review_family": label,
            }
        paths.extend((review_path, map_path))
    return reviews, paths


def load_count_hint_reviews():
    out, paths = {}, []
    for name in ("COUNT_HINT_REVIEW_A.json", "COUNT_HINT_REVIEW_B.json"):
        path = HERE / "review" / name
        doc = read(path)
        for ordinal, record in enumerate(doc["reviews"]):
            cid = canonical_id(record["candidate_id"])
            if cid in out:
                raise ValueError(f"Duplicate count-hint review: {cid}")
            out[cid] = {
                "record": record,
                "reviewer_id": doc["reviewer_id"],
                "review_id": f"{name}:{record.get('index', ordinal)}",
                "source": str(path.relative_to(HERE)),
                "review_family": "QUESTION_STEM_COUNT_HINT_REVISION1",
            }
        paths.append(path)
    return out, paths


def load_content_revision_reviews(hash_audit):
    out, paths = {}, []
    path = HERE / "review/G_batch1_REVISION1_STRICT_BLIND_REVIEW.json"
    doc = read(path)
    for record in doc["reviews"]:
        cid = canonical_id(record["candidate_id"])
        out[cid] = {
            "record": record,
            "reviewer_id": record["reviewer_id"],
            "review_id": f"G1_REVISION1:{cid}",
            "source": str(path.relative_to(HERE)),
            "review_family": "POST_CONTENT_STRICT_BLIND",
        }
    paths.append(path)
    path = HERE / "review/REVISION_POST_CONTENT_REVIEW.json"
    doc = read(path)
    for record in doc["results"]:
        cid = canonical_id(record["candidate_id"])
        if cid in out:
            raise ValueError(f"More than one post-content review: {cid}")
        out[cid] = {
            "record": record,
            "reviewer_id": doc["reviewer_id"],
            "review_id": f"POST_CONTENT:{cid}",
            "source": str(path.relative_to(HERE)),
            "review_family": "POST_CONTENT_BLIND",
        }
        if cid == "NQ15-F-45849-MC4":
            out[cid]["extra_provenance"] = {
                "candidate_hash_reconciliation": hash_audit,
            }
    paths.append(path)
    return out, paths


def hash_reconciliation():
    candidate_id = "NQ15-F-45849-MC4"
    revisions, _ = revision_candidates()
    revised = revisions[candidate_id]
    blind_doc = read(HERE / "review/REVISION_POST_CONTENT_BLIND_INPUT.json")
    blind_rows = {row["candidate_id"]: row for row in blind_doc["candidates"]}
    review_doc = read(HERE / "review/REVISION_POST_CONTENT_REVIEW.json")
    review_rows = {row["candidate_id"]: row for row in review_doc["results"]}
    raw_id = next(key for key in revisions if key == candidate_id)
    revised_raw = next(
        item["revised_candidate"]
        for item in read(HERE / "question_authoring/F/batch2/CANDIDATES_REVISION_1.json")["revisions"]
        if canonical_id(item["candidate_id"]) == raw_id
    )
    blind_id = next(key for key in blind_rows if canonical_id(key) == candidate_id)
    review_id = next(key for key in review_rows if canonical_id(key) == candidate_id)
    blind, result = blind_rows[blind_id], review_rows[review_id]
    corrected_hash = runner.orig.fingerprint(revised_raw)
    if blind.get("candidate_sha256") != corrected_hash:
        raise ValueError("F-45849 blind-input hash does not match the revised candidate")
    if any(revised_raw.get(key) != value for key, value in blind.items()
           if key != "candidate_sha256"):
        raise ValueError("F-45849 blind input is not an exact view of its revised candidate")
    reported_hash = result.get("candidate_sha256")
    if len(corrected_hash) != 64 or not isinstance(reported_hash, str) or len(reported_hash) != 63:
        raise ValueError("Unexpected F-45849 review hash discrepancy")
    if not any(corrected_hash[:index] + corrected_hash[index + 1:] == reported_hash
               for index in range(len(corrected_hash))):
        raise ValueError("F-45849 review hash is not a single-character deletion")
    if result.get("revision_status") != "PASS" or result.get("independent_gold") != revised_raw["gold"]:
        raise ValueError("F-45849 post-revision review does not support PASS with the revised Gold")
    return {
        "schema_version": "candidate_review_hash_reconciliation_v1",
        "append_only": True,
        "candidate_id": candidate_id,
        "reviewer_reported_sha256": reported_hash,
        "verified_sha256": corrected_hash,
        "verification_basis": {
            "review_input": "review/REVISION_POST_CONTENT_BLIND_INPUT.json",
            "review_result": "review/REVISION_POST_CONTENT_REVIEW.json",
            "revision_artifact": "question_authoring/F/batch2/CANDIDATES_REVISION_1.json",
            "blind_input_matches_revision_candidate": True,
            "blind_input_hash_matches_fingerprint": True,
            "review_hash_is_verified_single_character_deletion": True,
        },
        "decision": "ACCEPT_PASS_WITH_HASH_TYPO_RECONCILED",
        "raw_review_artifacts_modified": False,
    }


def load_control_adjudications():
    docs = (
        ("review/F_DISCREPANCIES_ADJUDICATION.json", "items"),
        ("review/G_batch1_ADJUDICATION.json", "items"),
        ("review/G_batch2_ADJUDICATION.json", "cases"),
    )
    out, paths = {}, []
    for name, key in docs:
        path = HERE / name
        doc = read(path)
        paths.append(path)
        for item in doc[key]:
            cid = canonical_id(item["candidate_id"])
            if cid not in CONTROL_ADJUDICATION_IDS:
                continue
            if cid in out:
                raise ValueError(f"Duplicate control-role adjudication for {cid}")
            disposition = item.get("adjudicated_disposition", item.get("adjudicated_decision"))
            gold = item.get("adjudicated_gold", item.get("strict_blind_gold"))
            if cid == "NQ15-G-55788-MC6" and not isinstance(gold, list):
                map_doc = read(HERE / "review/G_batch2_REVIEW_MAP_PRIVATE.json")
                mapped = {row["review_id"]: row for row in map_doc["mapping"]}[item["review_id"]]
                if canonical_id(mapped["candidate_id"]) != cid:
                    raise ValueError(f"G-55788-MC6 adjudication mapping drift: {mapped}")
                review_doc = read(HERE / "review/G_batch2_REVIEWER.json")
                raw = {row["review_id"]: row for row in review_doc["reviews"]}[item["review_id"]]
                gold = raw.get("independent_gold")
            if disposition not in {"PASS_CONTROL_ROLE", "PASS"} or not isinstance(gold, list):
                raise ValueError(f"Missing passing control adjudication for {cid}")
            reviewer_id = doc.get("created_by") or doc.get("adjudicator") or "control_role_adjudication"
            out[cid] = {
                "record": {
                    "decision": "PASS",
                    "independent_gold": gold,
                    "checks": {
                        key: {"decision": "PASS", "rationale": item["rationale"]}
                        for key in ("source_support", "gold_truth", "option_distinctness",
                                    "six_gold_control")
                    },
                },
                "reviewer_id": reviewer_id,
                "review_id": item.get("review_reference", item.get("review_id", f"{name}:{cid}")),
                "source": str(path.relative_to(HERE)),
                "review_family": "PREDECLARED_CONTROL_ROLE_ADJUDICATION",
                "extra_provenance": {
                    "adjudicated_disposition": disposition,
                    "raw_review_decision": item.get("raw_review_decision", item.get("original_decision")),
                    "raw_review_gold": item.get("raw_independent_gold"),
                    "adjudication_rationale": item["rationale"],
                },
            }
    if set(out) != CONTROL_ADJUDICATION_IDS:
        raise ValueError(f"Incomplete control-role adjudication set: {sorted(set(out) ^ CONTROL_ADJUDICATION_IDS)}")
    return out, paths


def load_missing_revision_hold():
    path = HERE / "review/G_batch1_ADJUDICATION.json"
    doc = read(path)
    matches = [item for item in doc["items"]
               if item["candidate_id"] == "NQ15-G-109164-MC3"]
    if len(matches) != 1 or matches[0].get("adjudicated_gold") != ["A", "B", "C"]:
        raise ValueError("Missing source adjudication for G-109164-MC3")
    item = matches[0]
    return {
        "record": {"independent_gold": item["adjudicated_gold"]},
        "reviewer_id": doc.get("created_by") or "G1_source_adjudication",
        "review_id": item["review_reference"],
        "source": str(path.relative_to(HERE)),
        "review_family": "SOURCE_ADJUDICATION_HELD_FOR_MISSING_REVISION_REVIEW",
        "extra_provenance": {
            "adjudicated_disposition": item["adjudicated_disposition"],
            "adjudication_rationale": item["rationale"],
        },
    }


def load_f_discrepancy_holds():
    path = HERE / "review/F_DISCREPANCIES_ADJUDICATION.json"
    doc = read(path)
    out = {}
    for item in doc["items"]:
        cid = canonical_id(item["candidate_id"])
        if cid not in {"NQ15-F-179186-MC5", "NQ15-F-25652-MC5"}:
            continue
        if item.get("adjudicated_disposition") != "HOLD" or not isinstance(item.get("strict_blind_gold"), list):
            raise ValueError(f"Missing strict-blind HOLD basis for {cid}")
        out[cid] = {
            "record": {"independent_gold": item["strict_blind_gold"]},
            "reviewer_id": "strict_blind_discrepancy_adjudicator",
            "review_id": item["review_reference"],
            "source": str(path.relative_to(HERE)),
            "review_family": "STRICT_BLIND_DISCREPANCY_ADJUDICATION",
            "extra_provenance": {
                "adjudicated_disposition": item["adjudicated_disposition"],
                "adjudication_rationale": item["rationale"],
            },
        }
    if set(out) != {"NQ15-F-179186-MC5", "NQ15-F-25652-MC5"}:
        raise ValueError("Incomplete F strict-blind discrepancy HOLD set")
    return out


def revision_candidates():
    revisions = {}
    paths = []
    files = (
        ("question_authoring/F/batch2/CANDIDATES_REVISION_1.json", "revisions"),
        ("question_authoring/G/batch1/CANDIDATES_REVISION_1.json", "revisions"),
        ("question_authoring/G/batch2/CANDIDATES_REVISION_1.json", "candidates"),
    )
    for name, list_key in files:
        path = HERE / name
        doc = read(path)
        for item in doc[list_key]:
            candidate = item.get("revised_candidate", item)
            cid = canonical_id(candidate["candidate_id"])
            if cid in revisions:
                raise ValueError(f"Duplicate content revision: {cid}")
            revisions[cid] = candidate
        paths.append(path)
    return revisions, paths


def rationale_text(value, fallback):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("rationale", "basis", "note", "assessment", "judgment"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return fallback


def review_checks(candidate, selected, decision, disposition_reason=None):
    source = selected["record"]
    slot = candidate["slot_id"]
    control = len(candidate["gold"]) == 6
    relation_key = "six_gold_control" if control else "distractor_relation"
    source_checks = source.get("checks", {})
    alias = source_checks.get("distractor_relation_or_six_gold_control")
    if alias is not None:
        source_checks = dict(source_checks)
        source_checks.setdefault(relation_key, alias)

    if decision == "HOLD":
        reason = disposition_reason or "The candidate remains held by the frozen disposition audit."
        return {
            key: {"decision": "HOLD", "rationale": reason}
            for key in ("source_support", "gold_truth", "option_distinctness", relation_key)
        }

    if source_checks:
        checks = {}
        for key in ("source_support", "gold_truth", "option_distinctness", relation_key):
            raw = source_checks.get(key)
            if not isinstance(raw, dict):
                raw = source_checks.get("distractor_relation_or_six_gold_control") if key == relation_key else None
            if isinstance(raw, dict):
                status = str(raw.get("decision", raw.get("status", "PASS"))).upper()
                if status not in {"PASS", "HOLD", "FAIL"}:
                    raise ValueError(f"Unknown review check status for {candidate['candidate_id']}: {status}")
                checks[key] = {
                    "decision": status,
                    "rationale": rationale_text(raw.get("rationale") or raw.get("basis"),
                                                 "The independent review records this check as passed."),
                }
        if len(checks) == 4:
            return checks

    evidence = source.get("evidence_basis", source.get("source_support", {}))
    if isinstance(evidence, list):
        evidence_note = f"The independent review records source evidence for {len(evidence)} option(s)."
    elif isinstance(evidence, dict):
        evidence_note = "The independent review records source evidence and findings for the candidate options."
    else:
        evidence_note = "The independent review decision and its linked evidence record are retained."
    issue_list = source.get("issues", [])
    no_issue_note = "No unresolved option issue was listed in the source review." if not issue_list else \
        "The source review lists issues; the candidate is held if any remain unresolved."
    distinction = source.get("distractor_discrimination") or source.get("distractor_quality")
    if not distinction:
        distinction = no_issue_note + " The source review did not store a separate option-distinctness field."
    relation = source.get("distractor_discrimination") or source.get("distractor_quality") \
        or source.get("operation_alignment") or source.get("operation_result")
    if not relation:
        relation = "Option-level truth judgments are recorded in the linked independent review."
    gold = source.get("independent_gold", source.get("independently_determined_gold", candidate["gold"]))
    checks = {
        "source_support": {"decision": "PASS", "rationale": evidence_note},
        "gold_truth": {"decision": "PASS", "rationale": f"The independent review derived Gold {gold}; final disposition confirms the frozen Gold."},
        "option_distinctness": {"decision": "PASS", "rationale": str(distinction)},
        relation_key: {"decision": "PASS", "rationale": str(relation)},
    }
    if control:
        checks[relation_key]["rationale"] = (
            "The six supported options are retained as the predeclared all-true control. "
            + str(relation)
        )
    return checks


def make_review(candidate, selected, decision, *, reason=None, final_gold=None, extra=None):
    candidate_id = candidate["candidate_id"]
    source = selected["record"]
    raw_gold = source.get("independent_gold", source.get("independently_determined_gold"))
    if decision == "PASS":
        if not isinstance(raw_gold, list) or final_gold is None or set(raw_gold) != set(final_gold):
            raise ValueError(f"PASS review Gold does not match candidate Gold for {candidate_id}")
        gold = list(final_gold)
    else:
        gold = list(raw_gold if raw_gold is not None else candidate["gold"])
    reviewer_id = selected["reviewer_id"]
    if not reviewer_id or reviewer_id == candidate["author_id"]:
        raise ValueError(f"Invalid reviewer identity for {candidate_id}: {reviewer_id}")
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": runner.orig.fingerprint(candidate),
        "source_binding": candidate["source_binding"],
        "author_id": candidate["author_id"],
        "reviewer_id": reviewer_id,
        "decision": decision,
        "independent_gold": gold,
        "checks": review_checks(candidate, selected, decision, reason),
        "review_provenance": {
            "source_file": selected["source"],
            "review_id": selected["review_id"],
            "review_family": selected["review_family"],
            "source_review_candidate_sha256": source.get("candidate_sha256"),
            "source_review_independent_gold": raw_gold,
            "frozen_disposition_reason": reason,
            "normalization_note": "The original review artifacts remain unchanged; this row normalizes their disposition and required check schema for runner.prepare.",
    } | (selected.get("extra_provenance", {})) | (extra or {}),
}


def assemble():
    plan = read(BASE)
    records = {row["candidate_id"]: copy.deepcopy(row) for row in plan["records"]}
    if len(records) != 150 or plan.get("complete") is not True:
        raise ValueError("Expected a complete 150-slot normalized plan")

    source_selection = read(HERE / "SOURCE_SELECTION_FINAL.json")
    sources = {row["root_id"]: {key: row[key] for key in ("path", "sha256", "bytes")}
               for row in source_selection["roots"]}

    revisions, revision_paths = revision_candidates()
    hash_audit = hash_reconciliation()
    stem_path = HERE / "question_authoring/QUESTION_STEM_COUNT_HINT_REVISION_1.json"
    stem_doc = read(stem_path)
    stem_revisions = {item["candidate_id"]: item for item in stem_doc["revisions"]}
    if len(stem_revisions) != 48:
        raise ValueError("Expected exactly 48 count-hint question-stem revisions")
    if set(stem_revisions) & set(revisions):
        raise ValueError("Question-stem and content revision scopes unexpectedly overlap")

    for cid, revision in stem_revisions.items():
        original = records.get(cid)
        revised = revision.get("revised_candidate", {})
        if original is None or revision.get("original_question") != original.get("question"):
            raise ValueError(f"Question-stem revision does not bind to the original candidate: {cid}")
        expected = copy.deepcopy(original)
        expected.pop("status", None)
        expected.pop("execution_status", None)
        expected["question"] = revised.get("question")
        projected = copy.deepcopy(revised)
        projected.pop("status", None)
        projected.pop("execution_status", None)
        if expected != projected:
            raise ValueError(f"Question-stem revision changed fields beyond question: {cid}")
        if runner.orig.fingerprint(revised) != next(
                row["candidate_sha256"] for row in
                read(HERE / "review" / ("COUNT_HINT_REVIEW_A.json" if cid in {
                    item["candidate_id"] for item in read(HERE / "review/COUNT_HINT_REVIEW_A.json")["reviews"]
                } else "COUNT_HINT_REVIEW_B.json"))["reviews"] if row["candidate_id"] == cid):
            raise ValueError(f"Question-stem review hash does not bind to its revision: {cid}")

    for cid, raw in revisions.items():
        if cid not in records or "question" not in records[cid]:
            raise ValueError(f"Content revision has no authored candidate: {cid}")
        base = records[cid]
        if "options" in raw:
            base["options"] = normalized_options(raw["options"])
        for key in ("question", "gold", "item_analysis", "r03_necessity"):
            if key in raw:
                base[key] = copy.deepcopy(raw[key])
        base["question_type"] = "multi_select"
        base["task_operation"] = runner.SLOTS[base["slot_id"]][0]
    for cid, revision in stem_revisions.items():
        if cid not in records or "question" not in records[cid]:
            raise ValueError(f"Question-stem revision has no authored candidate: {cid}")
        records[cid]["question"] = revision["revised_candidate"]["question"]

    candidates = []
    for cid in sorted(records):
        candidate = records[cid]
        candidate.pop("status", None)
        candidate.pop("execution_status", None)
        root = candidate["root_id"]
        candidate["source_binding"] = sources[root]
        candidate["candidate_id"] = cid
        candidate["question_type"] = "multi_select"
        candidate["task_operation"] = runner.SLOTS[candidate["slot_id"]][0]
        if "question" not in candidate:
            reason = str(candidate.get("r03_necessity") or candidate.get("failure_reason") or
                         "No candidate was authored for this planned slot; retained without replacement.")
            candidate["source_status"] = "SLOT_FAILURE"
            candidate["execution_status"] = "HOLD"
            candidate["failure_record"] = {
                "candidate_id": cid,
                "root_id": root,
                "slot_id": candidate["slot_id"],
                "reason": reason,
            }
        candidates.append(candidate)

    base_reviews, base_paths = load_review_sources()
    count_reviews, count_paths = load_count_hint_reviews()
    post_reviews, post_paths = load_content_revision_reviews(hash_audit)
    control_reviews, control_paths = load_control_adjudications()
    missing_revision_hold = load_missing_revision_hold()
    f_discrepancy_holds = load_f_discrepancy_holds()
    reviews = []
    for candidate in candidates:
        if "question" not in candidate:
            continue
        cid = candidate["candidate_id"]
        selected = (control_reviews.get(cid) or f_discrepancy_holds.get(cid)
                    or post_reviews.get(cid)
                    or count_reviews.get(cid) or base_reviews.get(cid))
        if cid == "NQ15-G-109164-MC3":
            selected = missing_revision_hold
        if selected is None:
            raise ValueError(f"No independent review record found for {cid}")
        source_decision = str(selected["record"].get("decision", selected["record"].get("revision_status", ""))).upper()
        decision = "HOLD" if cid in HOLD_IDS else ("PASS" if source_decision == "PASS" else "")
        if not decision:
            raise ValueError(f"No explicit passing review/adjudication for {cid}: {source_decision or 'missing'}")
        reason = HOLD_REASONS.get(cid)
        final_gold = list(candidate["gold"]) if decision == "PASS" else None
        review = make_review(candidate, selected, decision, reason=reason,
                             final_gold=final_gold)
        if decision == "PASS" and any(item["decision"] != "PASS" for item in review["checks"].values()):
            raise ValueError(f"Passing review contains a non-passing check for {cid}")
        reviews.append(review)

    expected_ids = {
        f"NQ15-{arm}-{root.split('-')[-1]}-{slot}"
        for root in ROOTS for arm in runner.ARMS for slot in SLOTS
    }
    if {item["candidate_id"] for item in candidates} != expected_ids:
        raise ValueError("Candidate set does not exactly cover the 150 planned slots")
    question_ids = {item["candidate_id"] for item in candidates if "question" in item}
    if len(candidates) != 150 or len(question_ids) != 141 or len(reviews) != 141:
        raise ValueError("Expected 150 slots, 141 authored candidates, and one review per candidate")
    if len(HOLD_IDS) != 6 or len(question_ids - HOLD_IDS) != 135:
        raise ValueError("Unexpected final review disposition count")

    review_by_id = {row["candidate_id"]: row for row in reviews}
    matched = []
    for root in ROOTS:
        for slot in SLOTS:
            arms = [f"NQ15-{arm}-{root.split('-')[-1]}-{slot}" for arm in runner.ARMS]
            if all(cid in review_by_id and review_by_id[cid]["decision"] == "PASS" for cid in arms):
                matched.append({"root_id": root, "slot_id": slot})
    if len(matched) != 61:
        raise ValueError(f"Expected 61 complete F/G slot pairs after HOLDs, found {len(matched)}")

    return (candidates, reviews, matched, hash_audit,
            sorted(set(revision_paths + base_paths + count_paths + post_paths + control_paths + [
                BASE, stem_path, HERE / "review/G_batch1_ADJUDICATION.json",
                HERE / "review/G_batch2_ADJUDICATION.json",
                HERE / "review/F_DISCREPANCIES_ADJUDICATION.json",
                HERE / "review/REVISION_POST_CONTENT_BLIND_INPUT.json",
                HERE / "review/F_DISCREPANCIES_STRICT_BLIND_REVIEW.json",
            ])))


def build_bindings(paths, candidates):
    all_paths = set(paths)
    for relative in ("AUTHORING_INPUT_FREEZE.json", "PROTOCOL_FREEZE.json",
                     "SOURCE_SELECTION_FINAL.json", "SCOPE_ZH.md",
                     "COMMON_AUTHOR_CONTRACT_ZH.md", "AUTHORING_ADDENDUM_ZH.md"):
        path = HERE / relative
        if path.exists():
            all_paths.add(path)
    for folder in (HERE / "question_authoring", HERE / "review", HERE / "source_warnings"):
        all_paths.update(p for p in folder.rglob("*") if p.is_file())
    all_paths.add(HERE / "SOURCE_SELECTION_FINAL.json")
    all_paths.add(HERE / "PROTOCOL_FREEZE.json")
    all_paths.update(Path(item["source_binding"]["path"]) for item in candidates if "question" in item)
    bindings = {}
    for path in sorted(all_paths, key=lambda item: str(item)):
        bindings[str(path.resolve())] = legacy.bind(path)
    return list(bindings.values())


def write_once(path: Path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to overwrite existing frozen file: {path}")
        return
    path.write_bytes(encoded)
    path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the three frozen JSON inputs")
    args = parser.parse_args()
    candidates, reviews, matched, hash_audit, audit_paths = assemble()
    candidate_doc = candidates
    review_doc = {
        "schema_version": "natural_question_forms_fg_content_review_v1",
        "basis": "Independent source review, revision review, and frozen disposition audit; raw review artifacts are retained unchanged.",
        "reviews": reviews,
    }
    if args.write:
        write_once(HASH_RECONCILIATION_PATH, hash_audit)
        write_once(HERE / "CANDIDATES_PRIVATE.json", candidate_doc)
        write_once(HERE / "CONTENT_REVIEW_AI.json", review_doc)
        bindings = build_bindings(audit_paths + [HERE / "CANDIDATES_PRIVATE.json",
                                                 HERE / "CONTENT_REVIEW_AI.json"], candidates)
        freeze = {
            "schema_version": "natural_question_forms_fg_content_freeze_v1",
            "frozen_before_solver": True,
            "bindings": bindings,
            "planned_slots": 150,
            "authored_candidates": 141,
            "pass_candidates": 135,
            "hold_candidates": 6,
            "empty_or_failed_slots": 9,
            "matched_fg_slot_pairs": matched,
            "matched_fg_pair_count": len(matched),
            "admitted_questions": len(matched) * 2,
            "requests_after_prepare": len(matched) * len(runner.ARMS) * 2 * 2,
            "no_replacement_sampling": True,
            "hold_reasons": HOLD_REASONS,
            "control_role_adjudications": sorted(CONTROL_ADJUDICATION_IDS),
            "hold_without_post_revision_blind_review": "NQ15-G-109164-MC3",
        }
        write_once(HERE / "CONTENT_FREEZE.json", freeze)
    print(json.dumps({
        "planned_slots": 150,
        "authored_candidates": len([c for c in candidates if "question" in c]),
        "pass_candidates": 135,
        "hold_candidates": 6,
        "empty_or_failed_slots": 9,
        "matched_fg_slot_pairs": len(matched),
        "requests_after_prepare": len(matched) * len(runner.ARMS) * 2 * 2,
        "write": args.write,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
