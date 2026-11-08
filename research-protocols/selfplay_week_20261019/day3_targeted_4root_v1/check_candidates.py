"""Validate fixed-slot coverage, source bindings and exact evidence before review."""
from pathlib import Path
from collections import Counter
import hashlib
import json

HERE = Path(__file__).resolve().parent


def walk_quotes(value):
    if isinstance(value, dict):
        if "turn_ref" in value and "quote" in value:
            yield value
        for child in value.values():
            yield from walk_quotes(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_quotes(child)


def main():
    freeze = json.loads((HERE / "GENERATION_FREEZE.json").read_text())
    slots = {s["slot_id"]: s for s in freeze["slots"]}
    source_bindings = json.loads((HERE / "SOURCE_FREEZE.json").read_text())["sources"]
    for slot in slots.values():
        slot["source_sha256"] = source_bindings[slot["root_id"]]["source"]["sha256"]
    rows = []
    errors = []
    checks = 0

    def check(ok, kind, qid):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append({"candidate_id": qid, "kind": kind})

    for path in sorted((HERE / "authors").glob("author_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["author_artifact"] = str(path)
                rows.append(row)
    check(Counter(r["slot_id"] for r in rows) == Counter(slots.keys()), "ALL24_SLOTS_ONCE", "batch")
    quote_count = 0
    for row in rows:
        qid = row["slot_id"]
        slot = slots.get(qid)
        check(slot is not None, "REGISTERED_SLOT", qid)
        if slot is None:
            continue
        check(row["candidate_id"] == qid, "CANDIDATE_ID_EQUALS_SLOT", qid)
        for key in ("root_id", "template_id", "source_path", "source_sha256", "task", "answer_format"):
            check(row.get(key) == slot[key], "FROZEN_" + key, qid)
        path = Path(slot["source_path"])
        raw = path.read_bytes()
        check(hashlib.sha256(raw).hexdigest() == slot["source_sha256"], "SOURCE_SHA256", qid)
        source = json.loads(raw)
        turns = {t["turn_ref"]: t for t in source["turns"]}
        check(row["status"] in ("DRAFT", "SHORTFALL"), "AUTHOR_STATUS", qid)
        if row["status"] == "SHORTFALL":
            check(bool(row.get("reason")) and bool(row.get("missing_material_feedback")), "SHORTFALL_HAS_REASON_AND_FEEDBACK", qid)
            continue
        check(bool(row.get("question")), "QUESTION_PRESENT", qid)
        check(bool(row.get("primary_relation")) and bool(row.get("fact_family_id")), "RELATION_AND_FAMILY_PRESENT", qid)
        gold = row.get("gold", [])
        check(isinstance(gold, list) and len(gold) == len(set(gold)), "GOLD_UNIQUE_LIST", qid)
        field = "options" if row["task"] == "multi_select" else "items"
        entries = row.get(field, {})
        check(isinstance(entries, dict) and all(isinstance(v, str) and v.strip() for v in entries.values()), "TEXT_ENTRY_DICTIONARY", qid)
        if row["task"] == "multi_select":
            check(set(entries) == set("ABCDEF"), "SIX_OPTIONS", qid)
            check(set(gold) <= set(entries), "GOLD_MEMBERSHIP", qid)
            check(len(gold) == slot["target_gold_count"], "FROZEN_GOLD_COUNT", qid)
            check(bool(row.get("option_audit")), "OPTION_AUDIT_PRESENT", qid)
        else:
            check(len(entries) in slot["allowed_items"], "ALLOWED_ORDER_ARITY", qid)
            check(set(entries) == {"I" + str(i + 1) for i in range(len(entries))}, "CANONICAL_ITEM_IDS", qid)
            check(len(gold) == len(entries) and set(gold) == set(entries), "ORDER_COVERS_ALL_ITEMS", qid)
        check(bool(row.get("uniqueness_proof")), "UNIQUENESS_EXPLANATION", qid)
        quotes = list(walk_quotes(row))
        check(bool(quotes), "SOURCE_EVIDENCE_PRESENT", qid)
        for quote in quotes:
            quote_count += 1
            turn = turns.get(quote["turn_ref"])
            check(turn is not None and quote["quote"] in turn["text"], "EXACT_SOURCE_QUOTE:" + quote["turn_ref"], qid)
    output = {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "candidate_records": len(rows),
        "author_status_counts": dict(Counter(row["status"] for row in rows)),
        "exact_quote_occurrences_checked": quote_count,
        "errors": errors,
        "semantic_review_established": False,
        "solver_calls": 0,
    }
    (HERE / "review" / "AUTHOR_STRUCTURAL_VALIDATION.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
