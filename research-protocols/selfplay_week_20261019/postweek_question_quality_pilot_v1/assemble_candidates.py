"""Collect authored candidates and mechanical findings, before content review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from scripts.current_method.question_presentation_gate import coverage_report, prepare_display


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def fingerprint(candidate):
    field = "options" if candidate["question_type"] == "multi_select" else "items"
    # Exact question and exact choice texts, independent of presentation order.
    return candidate["question_type"], candidate["question"], tuple(sorted(candidate[field].values()))


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    author_paths = [HERE / name / "CANDIDATES_PRIVATE.json" for name in ("author_mc", "author_order")]
    historical_path = HERE.parent / "postweek_question_quality_patch_v1/audit/HISTORICAL_CANDIDATES_PRIVATE.json"
    historical = {c["candidate_id"]: c for c in read(historical_path)["candidates"]}
    candidates = [candidate for path in author_paths for candidate in read(path)]
    assert len({c["candidate_id"] for c in candidates}) == len(candidates), "DUPLICATE_ID"
    assert not (set(historical) & {c["candidate_id"] for c in candidates}), "NEW_IDS_REQUIRED"
    seen = {fingerprint(c) for c in historical.values()}
    accepted, excluded, findings, sources = [], [], [], {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        assert candidate.get("author_id"), cid + ":AUTHOR_REQUIRED"
        _, private = prepare_display(candidate, 2026102301)
        source_binding = candidate["source_binding"]
        assert bind(source_binding["path"]) == source_binding, cid + ":SOURCE_BINDING_DRIFT"
        source = read(source_binding["path"])
        assert source["root_id"] == candidate["root_id"], cid + ":ROOT_DRIFT"
        refs = [t["turn_ref"] for t in source["turns"]]
        assert len(refs) == len(set(refs)), cid + ":DUPLICATE_SOURCE_REF"
        sources[source_binding["path"]] = {"binding": source_binding, "turns": len(refs)}
        parent_id = candidate.get("parent_candidate_id")
        if parent_id:
            parent = historical[parent_id]
            assert parent["source_binding"] == source_binding, cid + ":PARENT_SOURCE_CHANGED"
            if candidate.get("paired_item_identity_preserved"):
                field = "options" if candidate["question_type"] == "multi_select" else "items"
                assert list(candidate[field]) == list(parent[field]), cid + ":PAIRED_KEY_ORDER_CHANGED"
                assert candidate["question_type"] == parent["question_type"], cid + ":PAIRED_TYPE_CHANGED"
        key = fingerprint(candidate)
        if key in seen:
            excluded.append({"candidate_id": cid, "reason": "EXACT_WHOLE_QUESTION_DUPLICATE"})
            continue
        seen.add(key)
        accepted.append(candidate)
        findings.append({"candidate_id": cid, "lint": private["lint"],
                         "candidate_sha256": private["candidate_sha256"],
                         "parent_candidate_id": parent_id})
    report = {"status": "MECHANICAL_ONLY_PENDING_INDEPENDENT_CONTENT_REVIEW",
              "authored": len(candidates), "collected": len(accepted), "excluded": excluded,
              "roots": len({c["root_id"] for c in accepted}), "source_versions": list(sources.values()),
              "coverage": coverage_report(accepted), "findings": findings,
              "author_collection_bindings": [bind(p) for p in author_paths],
              "historical_collection_binding": bind(historical_path)}
    save(HERE / "CANDIDATES_PRIVATE.json", accepted)
    report["candidate_collection_binding"] = bind(HERE / "CANDIDATES_PRIVATE.json")
    save(HERE / "COLLECTION_AUDIT.json", report)
    print(json.dumps({"collected": len(accepted), "excluded": excluded,
                      "coverage": report["coverage"]["gold_count_histogram"],
                      "missing": report["coverage"]["missing_gold_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
