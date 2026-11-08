"""Aggregate immutable rejected source reviews without changing production.

All question admission/export/training-pool rules remain those of frozen v2.
The engineering gate also counts raw source-review ambiguity flags when the
review could not be materialized because the AI full-read attestation failed.
No CLI generation or semantic recovery is available in this module.
"""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oet1_aggregate_frozen_questions_v2", BASE / "scripts/questions_v2.py")
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
_original_engineering_gate = q.engineering_gate


def engineering_gate(decisions=None):
    value = _original_engineering_gate(decisions)
    roots = q.read(q.PROTOCOL)["first_wave_roots"]
    rejected = []
    extra_ambiguities = []
    for root in roots:
        for arm in q.ARMS:
            unit = BASE / "sources" / root / arm
            for path in sorted(unit.glob("*/*REJECTION.json")):
                row = q.read(path)
                rejection_binding = q.bind(path)
                rejected.append(rejection_binding)
                if path.name != "REVIEW_REJECTION.json":
                    continue
                raw = q.verify(row["raw_parsed_response_binding"])
                assert raw.get("source_or_gold_ambiguity_failure") == row.get("source_or_gold_ambiguity_failure")
                assert raw.get("decision") == row.get("review_decision")
                # A source review already materialized is counted by frozen v2.
                if (path.parent / "REVIEW.json").exists():
                    continue
                if raw.get("source_or_gold_ambiguity_failure") is True or "SOURCE_AMBIGUITY" in raw.get("failure_codes", []):
                    extra_ambiguities.append(row["raw_parsed_response_binding"])
    value["source_review_rejection_bindings"] = rejected
    value["unmaterialized_raw_source_ambiguity_bindings"] = extra_ambiguities
    value["source_ambiguity_bindings"].extend(extra_ambiguities)
    value["source_ambiguity_failure_count"] = len(value["source_ambiguity_bindings"])
    value["checks"]["no_source_or_gold_ambiguity_failure"] = value["question_ambiguity_failures"] + len(value["source_ambiguity_bindings"]) == 0
    value["status"] = "PASS" if all(value["checks"].values()) else "FAIL"
    value["aggregation_code_binding"] = q.bind(__file__)
    value["admission_rules_unchanged"] = True
    return value


# freeze_training recomputes its gate through the frozen v2 module global.
q.engineering_gate = engineering_gate
for _name in ("read", "save", "bind", "collect", "export", "freeze_training"):
    globals()[_name] = getattr(q, _name)
