"""Mechanical checks only; semantic judgments are documented in REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

from run import HERE, SOURCES, write_new


def main() -> None:
    output = {}
    for root, (path, _) in SOURCES.items():
        turns = json.loads(path.read_text(encoding="utf-8"))["turns"]
        lookup = {turn["turn_ref"]: (i, turn) for i, turn in enumerate(turns)}
        discovery = json.loads(json.loads((HERE / f"{root}_discover.response.json").read_text())["raw_output"])["outcomes"]
        backtrack = json.loads(json.loads((HERE / f"{root}_backtrack.response.json").read_text())["raw_output"])["relations"]
        discoveries = []
        for row in discovery:
            refs = row.get("outcome_refs")
            valid = isinstance(refs, list) and bool(refs) and all(isinstance(r, str) and r in lookup for r in refs)
            discoveries.append({
                "id": row.get("id"), "refs_exist_exactly": valid,
                "all_refs_in_R02": valid and all(lookup[r][1].get("scene_ref", "").startswith("R02:") for r in refs),
                "speaker_in_refs": valid and any(lookup[r][1]["speaker"] == row.get("actor") for r in refs),
                "semantic_review": "SEE_REPORT",
            })
        relations = []
        for row in backtrack:
            earlier = row.get("earlier_ref")
            later = row.get("later_refs")
            bridge = row.get("bridge_refs")
            exact = isinstance(earlier, str) and earlier in lookup and isinstance(later, list) and bool(later) and all(isinstance(r, str) and r in lookup for r in later) and isinstance(bridge, list) and all(isinstance(r, str) and r in lookup for r in bridge)
            relations.append({
                "id": row.get("id"), "status": row.get("status"),
                "reference_schema_and_exact_membership": exact,
                "earlier_precedes_all_later": exact and all(lookup[earlier][0] < lookup[r][0] for r in later),
                "earlier_distinct_from_later": exact and earlier not in later,
                "semantic_review": "SEE_REPORT",
            })
        output[root] = {"source_turns": len(turns), "discovered_count": len(discovery),
                        "relations_count": len(backtrack), "discovery_checks": discoveries,
                        "relation_checks": relations}
    write_new(HERE / "mechanical_audit.json", output)


if __name__ == "__main__":
    main()
