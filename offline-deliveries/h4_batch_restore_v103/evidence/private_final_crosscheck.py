"""Crosscheck saved final source judgments against unchanged private artifacts.

This helper never generates or repairs content. The shared v79 source-only
validator must accept every configured group before this process opens or hashes
the current private pool. A saved unknown, invalid scope, ambiguous fact, Gold
disagreement, or invalid private citation is a substantive exclusion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


SCHEMA_PATH = (Path(__file__).resolve().parents[2]
               / "h4_output_contract_multiselect_v79/review_schema.py")
IDENTITY_FIELDS = ("root_id", "question_type", "question", "options",
                   "source_sha256", "source_path")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def need(condition, reason):
    if not condition:
        raise ValueError("private_final_crosscheck:" + reason)


def load_schema():
    spec = importlib.util.spec_from_file_location(
        "_v103_private_final_crosscheck_review_schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_evidence_findings(question, judgment, turns):
    """Apply the original v79 exact citation and saved relation requirements."""
    findings = []
    options = judgment["options"]
    evidence = question.get("option_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(options):
        return ["option_evidence:coverage"]
    for label, saved in options.items():
        entry = evidence[label]
        if not isinstance(entry, dict):
            findings.append(label + ":option_evidence:not_object")
            continue
        turn = turns.get(entry.get("turn_ref"))
        quote = entry.get("quote")
        if entry.get("relation") != saved["relation"]:
            findings.append(label + ":option_evidence:relation_disagrees")
        if (turn is None or entry.get("speaker") != turn["speaker"]
                or not isinstance(quote, str) or not quote
                or quote not in turn["text"]):
            findings.append(label + ":option_evidence:not_exact_source")
    return findings


def question_check(private, public, judgment, turns):
    need(all(private.get(key) == public.get(key) for key in IDENTITY_FIELDS),
         "public_private_changed:" + public["candidate_id"])
    options = judgment["options"]
    findings = []
    if not judgment["stem_valid"]:
        findings.append("stem_invalid")
    for label, option in options.items():
        if option["relation"] == "unknown":
            findings.append(label + ":saved_relation_unknown")
        if not option["scope_valid"]:
            findings.append(label + ":saved_scope_invalid")
        if not option["fact_distinct"]:
            findings.append(label + ":saved_fact_not_distinct")
    supported = {label for label, option in options.items()
                 if option["relation"] == "supports"}
    gold = private.get("gold")
    valid_gold = (isinstance(gold, list)
                  and all(isinstance(label, str) for label in gold)
                  and len(gold) == len(set(gold))
                  and set(gold) <= set(options))
    if not valid_gold:
        findings.append("private_gold_invalid")
    elif supported != set(gold):
        findings.append("private_gold_disagrees_with_saved_source_judgment")
    findings.extend(option_evidence_findings(private, judgment, turns))
    findings = sorted(set(findings))
    return {
        "candidate_id": public["candidate_id"],
        "root_id": public["root_id"],
        "source_sha256": public["source_sha256"],
        "core_status": "KNOWN_SUBSTANTIVE_ISSUE" if findings else "CORE_USABLE",
        "reason": ("; ".join(findings) if findings else
                   "Saved whole-question source relations match unchanged private "
                   "Gold labels and exact source evidence. Stem, scope and "
                   "distinctness pass; no edits."),
        "findings": findings,
        "saved_supported_labels": sorted(supported),
    }


def crosscheck(home):
    home = Path(home).resolve()
    schema = load_schema()

    # Keep this call before every access (including hashes) to POOL.json.
    context = schema.validate_final_sources(home)
    groups = context["groups"]
    audit_paths = {group["group"]: home / "final_review"
                   / ("group_" + group["group"] + "_AUDIT.json")
                   for group in groups}
    for path in audit_paths.values():
        need(not path.exists(), "audit_already_exists:" + str(path))
    receipt_path = home / "final_review/PRIVATE_CROSSCHECK.COMPLETE.json"
    need(not receipt_path.exists(), "receipt_already_exists")
    saved = [
        {"group": group["group"], **group["binding"],
         "source_judgments_saved_at_utc": group["judgment"]["created_at_utc"]}
        for group in groups
    ]
    source_bindings = [entry for group in groups
                       for entry in group["judgment"]["source"]]
    frozen = [context["public_binding"], bind(home / "CONFIG.json"),
              *[group["binding"] for group in groups],
              *[{"path": entry["path"], "sha256": entry["sha256"]}
                for entry in source_bindings]]
    source_turns = {
        entry["root_id"]: {turn["turn_ref"]: turn
                            for turn in read(entry["path"])["turns"]}
        for entry in source_bindings
    }

    # All configured source reviews are saved and validated at this point.
    first_private_at = now()
    pool_path = home / "POOL.json"
    pool = read(pool_path)
    pool_binding = bind(pool_path)
    private = schema.index(pool.get("questions"), "candidate_id", pool_path,
                           "final", "questions")
    need(set(private) == set(context["questions"]), "public_pool_coverage")
    audits = []
    for group in groups:
        rows = [
            question_check(private[qid], context["questions"][qid], judgment,
                           source_turns[context["questions"][qid]["root_id"]])
            for qid, judgment in group["rows"].items()
        ]
        audits.append({
            "group": group["group"],
            "created_at_utc": now(),
            "pool_sha256": pool_binding["sha256"],
            "source_judgments_sha256": group["binding"]["sha256"],
            "current_solver_scores_read": False,
            "solver_scores_used": False,
            "source_first_for_current_children": True,
            "child_gold_read_before_save": False,
            "source_judgments_bytes_changed_after_private_read": False,
            "current_pool_read_after_both_group_source_judgments_save": True,
            "first_private_pool_read_at_utc": first_private_at,
            "both_source_judgments_saved_bindings": saved,
            "question_or_gold_edits": [],
            "training_allowed": False,
            "questions": rows,
        })
    for binding in [*frozen, pool_binding]:
        need(bind(binding["path"]) == binding,
             "input_changed_after_private_read:" + binding["path"])
    for audit in audits:
        write_new(audit_paths[audit["group"]], audit)

    # Replay the unchanged original audit contract over the actual saved files.
    validated = schema.validate_final_reviews(home, pool=pool)
    for binding in [*frozen, pool_binding]:
        need(bind(binding["path"]) == binding,
             "input_changed_during_validation:" + binding["path"])
    rows = validated["questions"]
    result = {
        "version": "v103_saved_source_private_crosscheck_v1",
        "created_at_utc": now(),
        "status": "AUDIT_CONTRACT_VALIDATED",
        "source_validation_completed_before_private_read": True,
        "first_private_pool_read_at_utc": first_private_at,
        "script": bind(__file__),
        "review_schema": bind(SCHEMA_PATH),
        "pool": pool_binding,
        "source_judgments": saved,
        "audits": [bind(audit_paths[group["group"]]) for group in groups],
        "question_count": len(rows),
        "usable_count": sum(row["core_status"] == "CORE_USABLE" for row in rows),
        "substantive_issue_count": sum(
            row["core_status"] == "KNOWN_SUBSTANTIVE_ISSUE" for row in rows),
        "question_or_gold_edits": [],
        "current_solver_scores_read": False,
        "training_allowed": False,
        "limitations": "Programmatic crosscheck of saved semantic judgments; "
                       "does not constitute an additional independent semantic "
                       "review or inspect solver scores.",
    }
    write_new(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", type=Path, help="Installed multiselect shard directory")
    args = parser.parse_args()
    print(json.dumps(crosscheck(args.home), ensure_ascii=False))


if __name__ == "__main__":
    main()
