"""Candidate retention is independent of review completeness and training admission.

The module routes mechanically valid artifacts for later review. It never emits
a semantic PASS, repairs a model verdict, or admits anything to training.
Explicit model-reported core problems cause quarantine, not a claim that the
problem was independently proved. All input artifacts remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


VERSION = "candidate_retention_v41"
CORE_CHECKS = {
    "source_core": "source_fidelity",
    "actor_actions": "actor_identity",
    "history_consistency": "history_consistency",
    "actual_progress": "substantive_progress",
}
CORE_CATEGORIES = set(CORE_CHECKS.values()) | {"duplication"}
AUXILIARY_CATEGORIES = {"plan_coverage", "summary", "auxiliary_fields"}


def _digest(value):
    data = value.encode("utf-8") if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_review_key:" + key)
        result[key] = value
    return result


def _parse(raw):
    if isinstance(raw, dict):
        return deepcopy(raw)
    if not isinstance(raw, str):
        raise ValueError("review_not_json_object")
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    value = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("review_not_json_object")
    return value


def _one_of(value, choices):
    # Invalid reviewer leaves (including arrays/objects) are audit defects.
    return isinstance(value, str) and value in choices


def decide_candidate(*, mechanical_failures=(), source_failures=(),
                     duplicate_result=None, review_raw=None, review_error=None,
                     review_validation=None, expected_identity=None):
    """Return a retention decision without mutating inputs or claiming quality.

    Callers first save the raw generation and mechanically validate the dialogue
    and source lineage. ``source_failures`` means a provable integrity failure
    (wrong source ID/hash/card mapping), not a semantic source-fidelity opinion.
    Reviewer transport/identity failures belong in ``review_error``. Such output
    is retained for audit but its semantic fields are not applied to this scene.

    Supports compact ``core_checks`` reports, compact ``findings`` reports, and
    legacy reports for offline replay. Review summary prose is never a gate.
    """
    hard_failures = [{"stage": "mechanical", "reason": str(item)}
                     for item in mechanical_failures]
    hard_failures += [{"stage": "source_integrity", "reason": str(item)}
                      for item in source_failures]
    audit_issues, risks, observations = [], [], []
    raw_verdict, parsed, schema = None, None, "unavailable"
    if duplicate_result is not None:
        if not isinstance(duplicate_result, dict) or duplicate_result.get("status") != "PASS":
            hard_failures.append({"stage": "duplicate", "reason": "duplicate_check_not_passed",
                                  "result": deepcopy(duplicate_result)})
    else:
        audit_issues.append({"code": "duplicate_check_not_supplied"})

    def issue(code, **details):
        audit_issues.append({"code": code, **deepcopy(details)})

    def finding(category, status, value, path):
        entry = {"category": category, "reported_verdict": status,
                 "review_path": path, "finding": deepcopy(value),
                 "provenance": "model_reported_not_independently_verified",
                 "semantic_certified": False}
        if category in CORE_CATEGORIES and status == "problem":
            risks.append(entry)
        elif status in {"problem", "uncertain"}:
            observations.append(entry)

    if review_error is not None:
        issue("review_unavailable_or_unusable", detail=str(review_error))
    if review_raw is None:
        issue("review_missing")
    else:
        try:
            parsed = _parse(review_raw)
            raw_verdict = deepcopy(parsed.get("verdict", parsed.get("status")))
            if review_error is not None:
                # Preserve the raw aggregate even when transport/context makes
                # its findings unusable for this scene's decision.
                parsed = None
            elif expected_identity is not None:
                mismatches = [key for key, value in expected_identity.items()
                              if parsed.get(key) != value]
                if mismatches:
                    issue("review_identity_mismatch", fields=mismatches)
                    parsed = None
        except (ValueError, TypeError) as error:
            issue("review_parse_error", detail=str(error))

    if isinstance(review_validation, dict):
        if review_validation.get("schema_valid") is False:
            issue("legacy_review_schema_incomplete",
                  failures=review_validation.get("schema_failures", []))
        # Failure lists may mix semantic and report-format errors. Preserve them
        # as audit data; never convert arbitrary failure text into a content fact.
        if review_validation.get("failures"):
            observations.append({"category": "legacy_validator_findings",
                                 "findings": deepcopy(review_validation["failures"])})

    if parsed is not None:
        if "core_checks" in parsed:
            schema = "compact_core_checks"
            if not _one_of(parsed.get("verdict"), {"acceptable", "problem", "uncertain"}):
                issue("review_verdict_missing_or_invalid")
            checks = parsed.get("core_checks")
            if not isinstance(checks, dict):
                issue("core_checks_not_object")
                checks = {}
            for name, category in CORE_CHECKS.items():
                value = checks.get(name)
                if not isinstance(value, dict) or not _one_of(value.get("status"), {"ok", "problem", "uncertain"}):
                    issue("core_check_missing_or_invalid", check=name)
                    continue
                if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                    issue("core_check_reason_missing", check=name)
                if not isinstance(value.get("evidence_refs", value.get("evidence")), list):
                    issue("core_check_evidence_missing", check=name)
                finding(category, value["status"], value, "core_checks." + name)
            if parsed.get("plan_observations"):
                observations.append({"category": "plan_coverage",
                                     "findings": deepcopy(parsed["plan_observations"])})
        elif "findings" in parsed:
            schema = "compact_findings"
            if not _one_of(parsed.get("verdict"), {"PASS", "REJECT", "UNCERTAIN"}):
                issue("review_verdict_missing_or_invalid")
            findings = parsed.get("findings")
            if not isinstance(findings, list):
                issue("review_findings_not_array")
                findings = []
            for index, value in enumerate(findings):
                if not isinstance(value, dict):
                    issue("review_finding_not_object", index=index)
                    continue
                category, status = value.get("category"), value.get("verdict")
                if not _one_of(category, CORE_CATEGORIES | AUXILIARY_CATEGORIES) or not _one_of(status, {"ok", "problem", "uncertain"}):
                    issue("review_finding_invalid", index=index)
                    continue
                if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                    issue("review_finding_reason_missing", index=index)
                if not isinstance(value.get("evidence"), list):
                    issue("review_finding_evidence_missing", index=index)
                finding(category, status, value, f"findings[{index}]")
        elif any(key in parsed for key in ("checks", "card_checks", "consistent_with_history", "new_event_or_progress")):
            schema = "legacy_review"
            for name, category, negative in (
                ("consistent_with_history", "history_consistency", False),
                ("new_event_or_progress", "substantive_progress", False),
                ("duplicates_existing_event", "duplication", True),
            ):
                if type(parsed.get(name)) is not bool:
                    issue("legacy_core_flag_missing", field=name)
                elif parsed[name] is negative:
                    is_soft = (name == "consistent_with_history" and isinstance(review_validation, dict)
                               and review_validation.get("history_consistency_status") == "UNRESOLVED_SOFT_FINDINGS")
                    finding(category, "uncertain" if is_soft else "problem", {name: parsed[name]}, name)
            soft = review_validation.get("soft_findings", []) if isinstance(review_validation, dict) else []
            if not isinstance(soft, list):
                soft = []
            soft_ids = {item.get("fact_id", item.get("constraint_id"))
                        for item in soft if isinstance(item, dict)
                        and isinstance(item.get("fact_id", item.get("constraint_id")), str)}
            for group in ("state_checks", "event_checks"):
                rows = parsed.get(group, [])
                if not isinstance(rows, list):
                    issue("legacy_check_group_invalid", group=group)
                    continue
                for index, row in enumerate(rows):
                    if isinstance(row, dict) and row.get("verdict") == "contradicted":
                        identifier = row.get("fact_id", row.get("constraint_id"))
                        finding("history_consistency", "uncertain" if isinstance(identifier, str) and identifier in soft_ids else "problem",
                                row, f"{group}[{index}]")
            for group in ("checks", "card_checks"):
                rows = parsed.get(group, [])
                if not isinstance(rows, list):
                    issue("legacy_check_group_invalid", group=group)
                    continue
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    if row.get("realized") is False:
                        finding("plan_coverage", "problem", row, f"{group}[{index}]")
                    actors = row.get("actor_checks", [])
                    if not isinstance(actors, list):
                        issue("legacy_actor_checks_invalid", group=group, index=index)
                        continue
                    for actor_index, actor in enumerate(actors):
                        if not isinstance(actor, dict) or actor.get("satisfied") is not False:
                            continue
                        expected, actual = actor.get("expected_actor"), actor.get("actor")
                        # An unperformed optional planned action is plan drift.
                        # A named wrong actor is an explicit identity risk.
                        wrong_actor = bool(isinstance(expected, str) and expected.strip()
                                           and isinstance(actual, str) and actual.strip() and expected != actual)
                        finding("actor_identity" if wrong_actor else "plan_coverage", "problem", actor,
                                f"{group}[{index}].actor_checks[{actor_index}]")
            if not isinstance(review_validation, dict):
                issue("legacy_review_completeness_not_checked")
        else:
            issue("review_schema_unrecognized")
        if _one_of(raw_verdict, {"REJECT", "problem"}) and not risks:
            issue("negative_aggregate_without_explicit_core_problem")

    disposition = "hard_reject" if hard_failures else "quarantined" if risks else "retained_pending_review"
    return {
        "version": VERSION,
        "disposition": disposition,
        "artifact_retained": True,
        "candidate_pool_eligible": disposition == "retained_pending_review",
        "continue_candidate_history": disposition == "retained_pending_review",
        "training_admitted": False,
        "training_pool_eligible": False,
        "semantic_certified": False,
        "needs_independent_review": True,
        "raw_review_verdict": raw_verdict,
        "raw_review_sha256": _digest(review_raw) if review_raw is not None else None,
        "review_schema": schema,
        "audit_complete": parsed is not None and not audit_issues,
        "audit_issues": audit_issues,
        "content_risks": risks,
        "observations": observations,
        "hard_failures": hard_failures,
        "decision_basis": "mechanical_integrity_and_explicit_model_risks_not_semantic_certification",
    }
