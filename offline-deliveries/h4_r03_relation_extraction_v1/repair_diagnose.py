"""Secondary citation-format diagnosis for the untouched repair output."""

from __future__ import annotations

import hashlib
import json

from citation_normalization import normalize_event
from run import HERE, SOURCE, SOURCE_SHA256, audit, write_json


def main() -> None:
    out = HERE / "repair_normalized_diagnostic_v2.json"
    if out.exists():
        raise RuntimeError("refusing overwrite")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    turns = json.loads(SOURCE.read_text(encoding="utf-8"))["turns"]
    valid_refs = {turn["turn_ref"] for turn in turns}
    original = json.loads((HERE / "repair_audit.json").read_text(encoding="utf-8"))["results"]
    events, changes = [], []
    for row in original:
        if row.get("status") != "FOUND":
            continue
        event, event_changes = normalize_event(row["event"], valid_refs, row["target_index"])
        events.append(event)
        changes.extend(event_changes)
    checks = audit(events, turns)
    write_json(out, {
        "source_sha256": SOURCE_SHA256,
        "normalization": "one outer bracket pair around an existing source turn ref",
        "raw_audit_path": str(HERE / "repair_audit.json"), "changes": changes,
        "normalized_events": events, "mechanical_checks": checks,
        "all_mechanical_pass": bool(checks) and all(row["mechanical_pass"] for row in checks),
        "semantic_entailment_certified": False, "training_promotion": False,
    })
    print(json.dumps({"count": len(events), "mechanical_pass": [row["mechanical_pass"] for row in checks]}))


if __name__ == "__main__":
    main()
