"""Diagnose bracketed citation formatting without changing the raw response."""

from __future__ import annotations

import json
from pathlib import Path

from citation_normalization import normalize_event
from run import HERE, SOURCE, SOURCE_SHA256, audit, write_json
import hashlib


def main() -> None:
    out = HERE / "normalized_diagnostic_v2.json"
    if out.exists():
        raise RuntimeError("refusing overwrite")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    turns = json.loads(SOURCE.read_text(encoding="utf-8"))["turns"]
    valid_refs = {turn["turn_ref"] for turn in turns}
    events = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))["events"]
    normalized_events, changes = [], []
    for index, event in enumerate(events):
        normalized, event_changes = normalize_event(event, valid_refs, index)
        normalized_events.append(normalized)
        changes.extend(event_changes)
    checks = audit(normalized_events, turns)
    write_json(out, {
        "source_sha256": SOURCE_SHA256,
        "normalization": "one outer bracket pair around an existing source turn ref",
        "raw_audit_path": str(HERE / "audit.json"),
        "changes": changes, "normalized_events": normalized_events,
        "mechanical_checks": checks,
        "all_mechanical_pass": bool(checks) and all(row["mechanical_pass"] for row in checks),
        "semantic_entailment_certified": False, "training_promotion": False,
    })
    print(json.dumps({"count": len(checks), "mechanical_pass": [row["mechanical_pass"] for row in checks]}))


if __name__ == "__main__":
    main()
