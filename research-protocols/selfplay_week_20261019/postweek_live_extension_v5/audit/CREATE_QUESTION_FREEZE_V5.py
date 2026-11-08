#!/usr/bin/env python3
"""Create and verify the v5 question freeze before any solver request.

The freeze binds the exact reviewed question bytes, source bytes, manifests, and
review receipt. It intentionally does not copy private gold values into the
freeze; the question-file hashes and gold digests bind them without exposing
them to execution code before raw responses are frozen.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
QDIR = BASE / "questions"
OUT = BASE / "QUESTION_FREEZE.json"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def binding(path: Path, *, root: Path = BASE) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        display_path = str(path.relative_to(root))
    except ValueError:
        display_path = str(path)
    return {
        "path": str(path),
        "relative_path": display_path,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }

def json_binding(path: Path) -> dict[str, Any]:
    return binding(path)

def digest_json(value: Any) -> str:
    # Compact, sorted representation gives a stable digest independent of file
    # indentation while preserving list order (canonical gold/display order).
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(data)

def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)

def main() -> None:
    require(not OUT.exists(), f"refusing to overwrite existing freeze: {OUT}")
    manifest_path = QDIR / "QUESTION_BATCH_MANIFEST.json"
    review_path = QDIR / "CONTENT_REVIEW.json"
    receipt_path = QDIR / "QUESTION_AUTHOR_READ_RECEIPT.json"
    readme_path = QDIR / "README_ZH.md"
    source_ready_path = BASE / "QUESTION_READY_SOURCE.json"
    source_ready_manifest_path = BASE / "QUESTION_READY_SOURCE_MANIFEST.json"
    protocol_path = BASE / "PROTOCOL_FREEZE.json"
    config_path = BASE / "CONFIG.json"
    review_md_path = QDIR / "CONTENT_REVIEW_ZH.md"
    for p in [manifest_path, review_path, receipt_path, source_ready_path,
              source_ready_manifest_path, protocol_path, config_path,
              review_md_path, readme_path]:
        require(p.exists(), f"missing required artifact: {p}")

    manifest = json.loads(manifest_path.read_text())
    review = json.loads(review_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    source_ready_manifest = json.loads(source_ready_manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    config = json.loads(config_path.read_text())

    require(manifest.get("schema_version") == "live_extension_v5_question_batch_manifest_v1", "unexpected manifest schema")
    require(review.get("status") == "PASS", "content review is not PASS")
    require(review.get("solver_started") is False, "content review says solver already started")
    require(review.get("training_admitted") is False, "content review says training already admitted")
    require(review.get("question_count") == 14, "content review question_count is not 14")
    require(manifest.get("question_count") == 14, "manifest question_count is not 14")
    require(source_ready_manifest.get("solver_started") is False, "source-ready manifest says solver started")
    require(source_ready_manifest.get("training_admitted") is False, "source-ready manifest says training admitted")
    require(protocol.get("generation_started") is False, "protocol says generation already started")
    require(config.get("generation_started") is False, "config says generation already started")
    require(protocol.get("question_plan", {}).get("solver_score_filtering") is False, "solver score filtering not disabled")
    require(protocol.get("question_plan", {}).get("exact_whole_question_duplicate_only") is True, "duplicate policy changed")
    require(protocol.get("full_source_no_truncation") is True, "source no-truncation requirement missing")

    # Source-ready binding must match both manifest and the actual artifact.
    ready_ref = manifest.get("source_ready_binding", {})
    require(ready_ref.get("sha256") == sha256_bytes(source_ready_path.read_bytes()), "QUESTION_READY_SOURCE hash mismatch")
    require(ready_ref.get("bytes") == source_ready_path.stat().st_size, "QUESTION_READY_SOURCE byte count mismatch")
    require(source_ready_manifest.get("question_ready_source_binding", {}).get("sha256") == ready_ref.get("sha256"), "source-ready manifest mismatch")

    review_entries = {x["candidate_id"]: x for x in review.get("questions", [])}
    require(len(review_entries) == 14, "review does not have 14 unique candidate IDs")
    seen_ids: set[str] = set()
    seen_file_hashes: set[str] = set()
    questions: list[dict[str, Any]] = []
    source_bindings: dict[str, dict[str, Any]] = {}
    by_type: dict[str, int] = {}
    by_gold_count: dict[str, int] = {}
    root_counts: dict[str, int] = {}
    failures: list[str] = []

    for root_id, root_rec in manifest.get("roots", {}).items():
        for slot in root_rec.get("slots", []):
            cid = slot["candidate_id"]
            require(cid not in seen_ids, f"duplicate candidate ID: {cid}")
            seen_ids.add(cid)
            fb = slot.get("candidate_binding", {})
            path = Path(fb["path"])
            require(path.exists(), f"missing question file: {path}")
            q = json.loads(path.read_text())
            actual = binding(path)
            require(actual["sha256"] == fb.get("sha256"), f"manifest question hash mismatch: {cid}")
            require(actual["bytes"] == fb.get("bytes"), f"manifest question byte mismatch: {cid}")
            require(q.get("candidate_id") == cid, f"candidate_id mismatch in {path}")
            require(q.get("root_id") == root_id, f"root_id mismatch in {path}")
            require(q.get("question_type") == slot.get("type"), f"type mismatch in {path}")
            require(len(q.get("gold", [])) == slot.get("gold_count"), f"gold count mismatch in {path}")
            require(cid in review_entries, f"candidate missing from content review: {cid}")
            rr = review_entries[cid]
            require(rr.get("file") == path.name, f"review file mismatch: {cid}")
            require(rr.get("root_id") == root_id, f"review root mismatch: {cid}")
            require(rr.get("type") == q.get("question_type"), f"review type mismatch: {cid}")
            require(rr.get("gold") == q.get("gold"), f"review gold mismatch: {cid}")
            require(rr.get("issues") == [], f"content review issues present: {cid}")
            require(q.get("semantic_admission") == "PENDING_INDEPENDENT_REVIEW", f"unexpected semantic admission: {cid}")
            require(q.get("source_read_scope", {}).get("all_turns_read") is True, f"source not fully read: {cid}")
            require(q.get("source_binding", {}).get("path"), f"missing question source binding: {cid}")
            source_path = Path(q["source_binding"]["path"])
            require(source_path.exists(), f"missing bound source: {source_path}")
            source_actual = binding(source_path)
            require(source_actual["sha256"] == q["source_binding"].get("sha256"), f"question source hash mismatch: {cid}")
            require(source_actual["bytes"] == q["source_binding"].get("bytes"), f"question source byte mismatch: {cid}")
            source_bindings[root_id] = source_actual

            typ = q.get("question_type")
            by_type[typ] = by_type.get(typ, 0) + 1
            gold = q.get("gold")
            display = q.get("display_order")
            if typ == "multi_select":
                options = q.get("options")
                require(isinstance(options, dict) and len(options) == 6, f"MC options not six: {cid}")
                labels = list(options)
                require(len(set(labels)) == 6, f"duplicate MC labels: {cid}")
                require(isinstance(gold, list) and 2 <= len(gold) <= 6, f"MC gold arity invalid: {cid}")
                require(len(set(gold)) == len(gold) and set(gold) <= set(labels), f"MC gold invalid: {cid}")
                require(isinstance(display, list) and display != labels and sorted(display) == sorted(labels), f"MC display permutation invalid: {cid}")
            elif typ == "ordering":
                items = q.get("items")
                require(isinstance(items, dict) and len(items) >= 3, f"ordering items invalid: {cid}")
                labels = list(items)
                require(isinstance(gold, list) and len(gold) == len(labels) and sorted(gold) == sorted(labels), f"ordering canonical gold invalid: {cid}")
                require(isinstance(display, list) and display != gold and sorted(display) == sorted(labels), f"ordering display permutation invalid: {cid}")
            else:
                raise RuntimeError(f"unknown question type {typ}: {cid}")
            by_gold_count[str(len(gold))] = by_gold_count.get(str(len(gold)), 0) + 1
            root_counts[root_id] = root_counts.get(root_id, 0) + 1
            if actual["sha256"] in seen_file_hashes:
                failures.append(f"duplicate question file hash: {cid}")
            seen_file_hashes.add(actual["sha256"])
            questions.append({
                "candidate_id": cid,
                "root_id": root_id,
                "slot": slot["slot"],
                "question_type": typ,
                "task_operation": q.get("task_operation"),
                "question_file": actual,
                "question_schema": q.get("schema_version"),
                "gold_count": len(gold),
                "gold_digest": digest_json(gold),
                "display_order_digest": digest_json(display),
                "display_seed": q.get("display_seed"),
                "source_read_scope": q.get("source_read_scope"),
                "source_binding": source_actual,
            })

    require(not failures, "; ".join(failures))
    require(len(questions) == 14, f"expected 14 questions, got {len(questions)}")
    require(root_counts == {"cc-train-episode-19708": 7, "cc-train-episode-20432": 7}, f"root counts mismatch: {root_counts}")
    require(by_type == {"multi_select": 10, "ordering": 4}, f"type counts mismatch: {by_type}")
    require(all(str(n) in by_gold_count for n in [2, 3, 4, 5, 6]), f"gold coverage incomplete: {by_gold_count}")

    # Bind reviewer-declared source hashes and the actual source file hashes.
    review_sources = review.get("source_bindings", {})
    require(set(review_sources) == set(source_bindings), "review source roots differ from question sources")
    for rid, actual in source_bindings.items():
        rs = review_sources[rid]
        require(rs.get("sha256") == actual["sha256"], f"review source hash mismatch: {rid}")
        require(rs.get("bytes") == actual["bytes"], f"review source bytes mismatch: {rid}")

    # Keep the freeze self-contained enough for a later preflight to check every
    # input without re-reading the review files' content.
    freeze = {
        "schema_version": "live_extension_v5_question_freeze_v1",
        "status": "FROZEN_BEFORE_SOLVER",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "batch_id": manifest.get("batch_id"),
        "namespace": protocol.get("namespace", "LEX1"),
        "phase": "content_admitted_before_solver",
        "policy": {
            "solver_score_filtering": False,
            "exact_whole_question_duplicate_only": True,
            "full_source_no_truncation": True,
            "question_content_review_required": True,
            "raw_responses_freeze_before_gold_read": True,
            "training_admitted": False,
            "private_gold_not_copied_into_freeze": True,
            "ordering_canonical_gold_retained": True,
            "ordering_display_order_required": True,
        },
        "gates": {
            "source_ready_verified": True,
            "content_review_status": review.get("status"),
            "content_review_solver_started": review.get("solver_started"),
            "content_review_training_admitted": review.get("training_admitted"),
            "manifest_question_hashes_refreshed": manifest.get("gates", {}).get("question_hashes_refreshed"),
            "solver_started": False,
            "training_admitted": False,
        },
        "counts": {
            "question_count": len(questions),
            "root_count": len(root_counts),
            "by_root": root_counts,
            "by_question_type": by_type,
            "by_gold_count": by_gold_count,
        },
        "bindings": {
            "question_batch_manifest": json_binding(manifest_path),
            "content_review": json_binding(review_path),
            "content_review_markdown": json_binding(review_md_path),
            "question_author_read_receipt": json_binding(receipt_path),
            "question_ready_source": json_binding(source_ready_path),
            "question_ready_source_manifest": json_binding(source_ready_manifest_path),
            "protocol_freeze": json_binding(protocol_path),
            "config": json_binding(config_path),
            "questions_readme": json_binding(readme_path),
        },
        "source_bindings": source_bindings,
        "questions": sorted(questions, key=lambda x: x["candidate_id"]),
        "integrity": {
            "checks_passed": [
                "manifest file hashes match all 14 question files",
                "content review is PASS with no per-question issues",
                "all 14 candidate IDs and root/slot/type bindings match review and manifest",
                "all bound cumulative source files exist and hashes match review and question files",
                "multi-select options are six-way unique permutations with gold arity 2/3/4/5/6 covered",
                "ordering items have canonical gold and independently shuffled display_order",
                "question file hashes are unique; no whole-question file duplicate",
                "all source_read_scope records declare complete source reads",
            ],
            "duplicate_question_policy": "exact_whole_question_duplicate_only",
            "review_limitations": review.get("limitations", []),
        },
        "next_gate": {
            "required_before_solver": "independent CPU preflight must validate this freeze and request plan",
            "solver_must_write": "raw response freeze before reading gold",
            "solver_must_not": ["automatic retry", "source/context truncation", "solver-based question admission"],
        },
        "generator": {
            "script": str(Path(__file__)),
            "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(OUT), "sha256": sha256_bytes(OUT.read_bytes()), "counts": freeze["counts"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
