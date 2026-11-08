#!/usr/bin/env python3
"""Build a controller-only field inventory without changing historical inputs."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PLAN = Path(__file__).resolve().parent.parent
AUDIT = PLAN / "execution_20261106_v1/review/HISTORICAL_SEALED135_OPERATION_AUDIT.json"
HISTORY = PLAN / "execution_20261106_v1/manifests/HISTORICAL_COVERAGE.json"
V6_PROJECTION = PLAN / "execution_20261107_v6/inputs/anonymous_history/HISTORICAL_COVERAGE_ANON.json"
OUTPUT = Path(__file__).with_name("SEALED135_FIELD_AUDIT_CONTROLLER.json")
FIXED_ROOTS = {"138343", "14304", "162896", "191749", "225575", "25652", "75241", "77882"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def field(value, status, basis):
    return {"value": value, "status": status, "basis": basis}


def main():
    audit = read(AUDIT)
    history = read(HISTORY)
    v6 = read(V6_PROJECTION)
    projection_meta = audit["inputs"]["sealed_projection_addendum"]
    projection_path = Path(projection_meta["path"])
    projection = read(projection_path)["items"]
    assert digest(AUDIT) == "8281e2bacf2fc7dd1f76f5df4684b0bd906ef0c701f449371d1dc3f9eb887b05"
    assert digest(HISTORY) == audit["inputs"]["historical_coverage"]["sha256"]
    assert digest(projection_path) == projection_meta["sha256"]
    assert len(audit["records"]) == len(projection) == 135
    assert len(history["items"]) == len(v6["records"]) == 966

    records = []
    for i, row in enumerate(audit["records"]):
        candidate = row["candidate_id"]
        idx = row["historical_coverage_record_index_zero_based"]
        pidx = row["projection_record_index_zero_based"]
        assert i == pidx and history["items"][idx]["candidate_id"] == candidate
        assert projection[pidx]["candidate_id"] == candidate
        assert projection[pidx]["question_and_displayed_options"] == row["question_and_displayed_options"]
        question_hash = hashlib.sha256(row["question_and_displayed_options"].encode()).hexdigest()
        assert question_hash == row["source_hashes"]["question_content_sha256"]
        assert row["coding_status"] == "CONFIRMED_OPERATION_FROM_QUESTION_AND_QUOTATIONS"
        assert row["unknowns"] == [] and row["evidence_quotations"]
        ref = row["evidence_sources_read"][0]
        assert digest(ref["path"]) == ref["sha256"]
        original = read(ref["path"])[ref["record_index_zero_based"]]
        assert original["candidate_id"] == candidate
        history_id = v6["records"][idx]["history_item_id"]
        assert history_id == f"H{idx + 1:04d}"
        root = row["root_id"].rsplit("-", 1)[-1]
        reference_scope = original.get("relation_plan")
        reference_scope = reference_scope.get("scope") if isinstance(reference_scope, dict) else None

        event = original.get("event_identity")
        need = original.get("natural_information_need")
        if event and need:
            event_field = field(event, "EXPLICIT_SAME_ID_REFERENCE", "reference.event_identity")
            need_field = field(need, "EXPLICIT_SAME_ID_REFERENCE", "reference.natural_information_need")
        else:
            assert not event and not need and 72 <= i <= 120
            event_field = field(row["natural_information_need_and_query_scope"],
                                "MULTI_SCENE_CONTEXT_FROM_OPERATION_AUDIT",
                                "audit.natural_information_need_and_query_scope; no single common event asserted")
            need_field = field(None, "UNKNOWN_NOT_STATED", "The question provides no unified natural business purpose.")

        if reference_scope:
            scope_field = field(reference_scope, "EXPLICIT_SAME_ID_REFERENCE", "reference.relation_plan.scope")
        elif 72 <= i <= 120:
            is_appearance = i == 106
            scope = ("The listed statements' appearance in the six specified dialogue scenes, including within-turn clause order."
                     if is_appearance else
                     "The listed statements and whether each is supported in the six specified dialogue scenes.")
            scope_field = field(scope, "ANNOTATED_FROM_QUESTION_AND_AUDIT", "question projection and same-ID operation audit")
        else:
            scope_field = field(None, "UNKNOWN_PENDING_SCOPE_SPLIT",
                                "Operation audit combines natural need and scope; no separate scope field is explicit.")

        records.append({
            "history_item_id": history_id,
            "candidate_id_controller_only": candidate,
            "root_id_controller_only": row["root_id"],
            "fixed_eight_root": root in FIXED_ROOTS,
            "audit_record_index_zero_based": i,
            "projection_record_index_zero_based": pidx,
            "reference": {"path": ref["path"], "sha256": ref["sha256"],
                          "record_index_zero_based": ref["record_index_zero_based"]},
            "source_snapshot_sha256": row["source_hashes"]["associated_source_snapshot_sha256"],
            "question_content_sha256": question_hash,
            "evidence_turn_refs": [e.get("turn_ref") for e in row["evidence_quotations"]],
            "combined_need_and_scope_audit_text_controller_only": row["natural_information_need_and_query_scope"],
            "event": event_field,
            "natural_information_need": need_field,
            "query_scope": scope_field,
            "answer_operation": field(row["actual_sufficient_answer_program"],
                                      "INDEPENDENT_OPERATION_AUDIT", "audit.actual_sufficient_answer_program"),
            "evidence_family": field(row["event_or_evidence_families"],
                                     "INDEPENDENT_OPERATION_AUDIT", "audit.event_or_evidence_families"),
            "operation_codes": row["operation_codes"],
            "selector_exposure": "FORBIDDEN_PENDING_LEAKAGE_SAFE_NORMALIZATION_AND_SEMANTIC_REVIEW",
        })

    assert len({r["candidate_id_controller_only"] for r in records}) == 135
    counts = {name: dict(Counter(r[name]["status"] for r in records)) for name in
              ("event", "natural_information_need", "query_scope", "answer_operation", "evidence_family")}
    assert counts["natural_information_need"] == {"EXPLICIT_SAME_ID_REFERENCE": 86, "UNKNOWN_NOT_STATED": 49}
    assert counts["query_scope"] == {"EXPLICIT_SAME_ID_REFERENCE": 12,
                                     "UNKNOWN_PENDING_SCOPE_SPLIT": 74,
                                     "ANNOTATED_FROM_QUESTION_AND_AUDIT": 49}
    result = {
        "schema_version": "sealed135_field_inventory_controller_v1",
        "status": "INCOMPLETE_NOT_SELECTOR_READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": "CONTROLLER_ONLY",
        "scope": "135 sealed training candidates; field inventory, not quality/Gold/training review.",
        "source_bindings": {
            "operation_audit": {"path": str(AUDIT), "sha256": digest(AUDIT)},
            "historical_coverage": {"path": str(HISTORY), "sha256": digest(HISTORY)},
            "question_projection": {"path": str(projection_path), "sha256": digest(projection_path)},
            "v6_anonymous_id_mapping": {"path": str(V6_PROJECTION), "sha256": digest(V6_PROJECTION)},
        },
        "limitations": [
            "No selector may read this artifact: controller fields can disclose historical question and answer content.",
            "49 cross-scene tasks state no unified natural business purpose; UNKNOWN is not evidence of novelty.",
            "74 query scopes remain UNKNOWN pending independent extraction from exact questions and evidence.",
            "Reference event and need values are historical annotations, not newly independently reviewed judgments.",
            "Existing operation audit observed sufficient paths, not exhaustive proof of all possible shortcuts.",
            "Hash verification of a source snapshot is not semantic reading of the full source narrative.",
        ],
        "field_status_counts": counts,
        "records": records,
    }
    assert not OUTPUT.exists(), f"refusing to overwrite {OUTPUT}"
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": digest(OUTPUT), "field_status_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
