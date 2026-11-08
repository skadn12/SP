"""Source-bound, deterministic option-position binding for six-item ordering questions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re


LABELS = tuple(f"I{i}" for i in range(1, 7))
LABEL_SET = set(LABELS)
VERSION = "ordering-position-v1"
LABEL_TOKEN = re.compile(r"\bI[1-6]\b")


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def source_order(row: dict, source: dict) -> list[str]:
    qid = row.get("candidate_id", "unknown")
    require(isinstance(row.get("candidate_id"), str) and bool(row["candidate_id"]), f"{qid}:candidate_id")
    require(row.get("question_type") == "ordering" and row.get("chronology_kind") == "conversation_order",
            f"{qid}:question_type_or_kind")
    require(row.get("root_id") == source.get("root_id"), f"{qid}:root_id")
    require(isinstance(row.get("items"), dict) and set(row["items"]) == LABEL_SET,
            f"{qid}:items")
    require(isinstance(row.get("item_evidence"), dict) and set(row["item_evidence"]) == LABEL_SET,
            f"{qid}:item_evidence")
    require(isinstance(row.get("gold"), list) and len(row["gold"]) == 6
            and set(row["gold"]) == LABEL_SET, f"{qid}:gold_permutation")
    require(isinstance(row.get("question"), str) and "APPEAR IN THE SUPPLIED DIALOGUE" in row["question"],
            f"{qid}:question_scope")
    require(not LABEL_TOKEN.search(row["question"]), f"{qid}:question_contains_label_reference")
    require(not LABEL_TOKEN.search(str(row.get("topic", ""))), f"{qid}:topic_contains_label_reference")
    turns = source.get("turns")
    require(isinstance(turns, list), f"{qid}:source_turns")
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    require(len(by_ref) == len(turns), f"{qid}:duplicate_source_refs")
    positions = {ref: index for index, ref in enumerate(by_ref)}
    refs = {}
    for label in LABELS:
        item, evidence = row["items"][label], row["item_evidence"][label]
        require(isinstance(item, str) and bool(item.strip()) and not LABEL_TOKEN.search(item),
                f"{qid}:{label}:item")
        require(isinstance(evidence, dict), f"{qid}:{label}:evidence")
        ref, quote = evidence.get("turn_ref"), evidence.get("quote")
        require(isinstance(ref, str) and ref in by_ref, f"{qid}:{label}:turn_ref")
        turn = by_ref[ref]
        require(evidence.get("speaker") == turn.get("speaker") and isinstance(quote, str)
                and bool(quote.strip()) and quote in turn.get("text", ""),
                f"{qid}:{label}:quote_or_speaker")
        require(not LABEL_TOKEN.search(str(evidence.get("reason", ""))),
                f"{qid}:{label}:evidence_reason_contains_label_reference")
        refs[label] = ref
    require(len(set(refs.values())) == 6, f"{qid}:duplicate_anchor_turns")
    require(any(ref.startswith("R01:") for ref in refs.values()), f"{qid}:r01_anchor_required")
    expected = sorted(LABELS, key=lambda label: positions[refs[label]])
    require(row["gold"] == expected, f"{qid}:gold_conflicts_with_source_order")
    return expected


def target_gold(candidate_id: str, source_hash: str, old_gold: list[str]) -> tuple[list[str], str]:
    key = sha(canonical([VERSION, candidate_id, source_hash]))
    eligible = [order for order in permutations(LABELS)
                if list(order) != old_gold and order[0] != "I1" and order[-1] != "I6"
                and sum(label != f"I{index}" for index, label in enumerate(order, 1)) >= 4]
    require(bool(eligible), "no_target_permutations")
    return list(eligible[int(key, 16) % len(eligible)]), key


def content_pairs(row: dict) -> list[bytes]:
    return sorted(canonical([row["items"][label], row["item_evidence"][label]]) for label in LABELS)


def bind(row: dict, source: dict, source_hash: str) -> tuple[dict, dict]:
    old_gold = source_order(row, source)
    new_gold, key = target_gold(row["candidate_id"], source_hash, old_gold)
    old_to_new = dict(zip(old_gold, new_gold))
    new_to_old = {new: old for old, new in old_to_new.items()}
    result = deepcopy(row)
    result["items"] = {new: deepcopy(row["items"][new_to_old[new]]) for new in LABELS}
    result["item_evidence"] = {new: deepcopy(row["item_evidence"][new_to_old[new]]) for new in LABELS}
    result["gold"] = new_gold
    result["order_reason"] = "Source turn appearance order: " + ", ".join(
        f"{label} ({result['item_evidence'][label]['turn_ref']})" for label in new_gold
    ) + "."
    require(source_order(result, source) == new_gold, f"{row['candidate_id']}:rebound_gold_invalid")
    require(content_pairs(row) == content_pairs(result), f"{row['candidate_id']}:content_changed")
    audit = {
        "candidate_id": row["candidate_id"], "root_id": row["root_id"],
        "source_sha256": source_hash, "seed_sha256": key,
        "old_gold": old_gold, "new_gold": new_gold,
        "old_to_new_label": old_to_new,
        "old_item_evidence_multiset_sha256": sha(canonical([pair.decode("utf-8") for pair in content_pairs(row)])),
        "new_item_evidence_multiset_sha256": sha(canonical([pair.decode("utf-8") for pair in content_pairs(result)])),
        "old_question_sha256": sha(canonical(row)), "new_question_sha256": sha(canonical(result)),
    }
    return result, audit


def input_rows(payload: object) -> list[dict]:
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    require(isinstance(rows, list), "input_questions_array_missing")
    return rows


def build_result(input_pool: Path, source_root: Path) -> tuple[dict, dict]:
    raw = input_pool.read_bytes()
    rows = input_rows(json.loads(raw))
    outputs, audits, rejects = [], [], []
    seen, seen_event_sets = set(), set()
    for index, row in enumerate(rows):
        qid = row.get("candidate_id") if isinstance(row, dict) else None
        try:
            require(isinstance(row, dict), "candidate_not_object")
            require(isinstance(qid, str) and qid not in seen, "candidate_id_missing_or_duplicate")
            seen.add(qid)
            root = row.get("root_id")
            require(isinstance(root, str) and re.fullmatch(r"cc-train-episode-[0-9]+", root) is not None,
                    f"{qid}:root_id_path")
            source_path = source_root / root / "source.json"
            source_raw = source_path.read_bytes()
            source_hash = sha(source_raw)
            require(row.get("source_sha256", source_hash) == source_hash, f"{qid}:source_hash_mismatch")
            if "source_path" in row:
                require(Path(row["source_path"]).resolve() == source_path.resolve(),
                        f"{qid}:source_path_mismatch")
            result, audit = bind(row, json.loads(source_raw), source_hash)
            event_set = (root, tuple(sorted(entry["turn_ref"] for entry in result["item_evidence"].values())))
            require(event_set not in seen_event_sets, f"{qid}:duplicate_event_set")
            seen_event_sets.add(event_set)
            result["source_path"] = str(source_path.resolve())
            result["source_sha256"] = source_hash
            audit["new_question_sha256"] = sha(canonical(result))
            outputs.append(result)
            audits.append(audit)
        except (KeyError, TypeError, ValueError, OSError) as error:
            rejects.append({"index": index, "candidate_id": qid, "reason": str(error)})
    report = {
        "status": "ORDERING_POSITION_BOUND_MECHANICAL_ONLY", "version": VERSION,
        "input_pool": str(input_pool.resolve()), "input_pool_sha256": sha(raw),
        "source_root": str(source_root.resolve()), "input_count": len(rows),
        "bound_count": len(outputs), "rejected_count": len(rejects),
        "audits": audits, "rejects": rejects,
        "semantic_gold_verified": False, "solver_calls": 0, "training_started": False,
    }
    return {"status": report["status"], "questions": outputs}, report


def run(input_pool: Path, source_root: Path, output_dir: Path) -> dict:
    pool, report = build_result(input_pool, source_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("POOL.json", pool), ("REPORT.json", report)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def verify(input_pool: Path, source_root: Path, output_dir: Path) -> dict:
    expected_pool, expected_report = build_result(input_pool, source_root)
    saved_pool = json.loads((output_dir / "POOL.json").read_text(encoding="utf-8"))
    saved_report = json.loads((output_dir / "REPORT.json").read_text(encoding="utf-8"))
    require(saved_pool == expected_pool, "saved_pool_changed")
    require(saved_report == expected_report, "saved_report_changed")
    return {"status": "VERIFIED", "bound_count": expected_report["bound_count"],
            "rejected_count": expected_report["rejected_count"],
            "pool_sha256": sha((output_dir / "POOL.json").read_bytes()),
            "report_sha256": sha((output_dir / "REPORT.json").read_bytes())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-dir", type=Path)
    args = parser.parse_args()
    if args.verify_dir:
        print(json.dumps(verify(args.input_pool, args.source_root, args.verify_dir), ensure_ascii=False))
    else:
        print(json.dumps(run(args.input_pool, args.source_root, args.output_dir), ensure_ascii=False))
