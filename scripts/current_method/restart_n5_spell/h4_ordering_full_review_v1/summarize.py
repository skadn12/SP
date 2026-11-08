"""Summarize the frozen Dev/Test ordering audit without changing source data."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_matched_training_v1_live_1")
OUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1")
HUMAN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_ordering_candidate_review_v1_live_1/review_decisions_v2.tsv")
PAIRWISE = OUT / "ordering_review_pairwise.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    human = {row["candidate_id"]: row for row in csv.DictReader(HUMAN.open(encoding="utf-8"), delimiter="\t")}
    machine = {row["candidate_id"]: row for row in load_jsonl(PAIRWISE)} if PAIRWISE.is_file() else {}
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    root_counts: dict[str, Counter[str]] = defaultdict(Counter)
    structural = Counter()
    evidence_order = Counter()
    candidates: list[dict[str, Any]] = []

    for split in ("dev", "test"):
        manifest_path = DATA / f"{split}_full_eval_manifest.jsonl"
        for row in load_jsonl(manifest_path):
            candidate = row.get("candidate") or {}
            if candidate.get("question_type") != "ordering":
                continue
            turns = load_jsonl(Path(row["dialogue_path"])) if False else json.loads(Path(row["dialogue_path"]).read_text(encoding="utf-8")).get("turns", [])
            refs = [str(ref) for ref in candidate.get("evidence_refs") or []]
            items = candidate.get("items") or {}
            gold = [str(item) for item in candidate.get("gold") or []]
            item_ids = list(items)
            valid_refs = {str(turn.get("turn_ref")) for turn in turns}
            positions = {str(turn.get("turn_ref")): index for index, turn in enumerate(turns)}
            structural_status = "pass"
            if item_ids != ["I1", "I2", "I3", "I4"]:
                structural["item_id_error"] += 1
                structural_status = "fail"
            if sorted(gold) != ["I1", "I2", "I3", "I4"]:
                structural["gold_permutation_error"] += 1
                structural_status = "fail"
            if len(set(refs)) != len(refs):
                structural["duplicate_evidence_ref"] += 1
                structural_status = "fail"
            if any(ref not in valid_refs for ref in refs):
                structural["missing_evidence_ref"] += 1
                structural_status = "fail"
            if len(refs) != len(item_ids):
                structural["evidence_count_mismatch"] += 1
                structural_status = "fail"
            chronology = "unresolved"
            chronology_order: list[str] = []
            if len(refs) == len(item_ids) and all(ref in positions for ref in refs):
                chronology_order = [item_ids[index] for index, _ in sorted(enumerate(refs), key=lambda pair: positions[pair[1]])]
                chronology = "matches_gold" if chronology_order == gold else "differs_from_gold"
                evidence_order[chronology] += 1
            else:
                evidence_order["unresolved"] += 1
            cid = str(candidate.get("candidate_id"))
            human_row = human.get(cid)
            machine_row = machine.get(cid)
            record = {
                "candidate_id": cid,
                "split": split,
                "root_id": candidate.get("root_id"),
                "gold": gold,
                "evidence_refs": refs,
                "item_count": len(item_ids),
                "turn_count": len(turns),
                "structural_status": structural_status,
                "evidence_chronology": chronology,
                "evidence_chronology_order": chronology_order,
                "human_review_status": human_row.get("gold_status") if human_row else None,
                "human_evidence_status": human_row.get("evidence_status") if human_row else None,
                "machine_triage_status": machine_row.get("gold_status") if machine_row else None,
                "machine_triage_evidence_status": machine_row.get("evidence_status") if machine_row else None,
                "semantic_final_status": "CONFIRMED_HUMAN" if human_row else "PENDING_SEMANTIC_REVIEW",
            }
            candidates.append(record)
            split_counts[split]["total"] += 1
            split_counts[split][f"structural_{structural_status}"] += 1
            split_counts[split][f"evidence_{chronology}"] += 1
            root_counts[str(candidate.get("root_id"))][f"{split}_total"] += 1
            if human_row:
                split_counts[split]["human_confirmed"] += 1

    output = {
        "audit_version": "h4_ordering_full_review_v1_summary",
        "offline_only": True,
        "semantic_final_status": "PENDING_SEMANTIC_REVIEW",
        "source_data_run": str(DATA),
        "source_manifests": {
            split: {
                "path": str(DATA / f"{split}_full_eval_manifest.jsonl"),
                "sha256": sha256(DATA / f"{split}_full_eval_manifest.jsonl"),
            }
            for split in ("dev", "test")
        },
        "candidate_count": len(candidates),
        "split_counts": {split: dict(sorted(counts.items())) for split, counts in sorted(split_counts.items())},
        "structural_counts": dict(sorted(structural.items())),
        "evidence_chronology_counts": dict(sorted(evidence_order.items())),
        "human_review": {
            "source": str(HUMAN),
            "source_sha256": sha256(HUMAN),
            "confirmed_count": len(human),
            "not_yet_human_confirmed": len(candidates) - len(human),
            "gold_status_counts": dict(Counter(row["gold_status"] for row in human.values())),
            "evidence_status_counts": dict(Counter(row["evidence_status"] for row in human.values())),
        },
        "machine_triage": {
            "source": str(PAIRWISE),
            "source_sha256": sha256(PAIRWISE) if PAIRWISE.is_file() else None,
            "status": "TRIAGE_ONLY_NOT_FINAL",
            "candidate_count": len(machine),
            "gold_status_counts": dict(Counter(row.get("gold_status") for row in machine.values())),
            "evidence_status_counts": dict(Counter(row.get("evidence_status") for row in machine.values())),
        },
        "root_counts": {root: dict(sorted(counts.items())) for root, counts in sorted(root_counts.items())},
        "candidates": sorted(candidates, key=lambda record: record["candidate_id"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ordering_full_audit_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
