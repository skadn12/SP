"""Read-only capacity inventory for the fresh-root ordering pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_unexpanded122_memory_event_k3_v1/shards")
POOL_BASE = Path("/root/autodl-tmp/gxy-selfplay/SP-820-publish/first_training_pipeline/data/qa_generation_only/shards")
ROOTS = (("210046", "24"), ("231899", "31"), ("194259", "19"), ("96544", "58"))
SPLIT_MANIFEST = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    split_rows = {
        row["root_id"]: row
        for line in SPLIT_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    rows = []
    for suffix, shard in ROOTS:
        root_id = f"cc-train-episode-{suffix}"
        source_path = SOURCE_BASE / f"shard-{shard}" / "expansion" / "roots" / root_id / "promotion" / "promotable_expanded_dialogue.json"
        pool_path = POOL_BASE / f"shard-{shard}" / "roots" / root_id / "candidate_pool.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        turns = source["turns"]
        split = split_rows.get(root_id)
        if split is None or split.get("candidate_pool_sha256") != digest(pool_path):
            raise ValueError(f"split_record_or_pool_hash_mismatch:{root_id}")
        rows.append({
            "root_id": root_id,
            "source_path": str(source_path),
            "source_sha256": digest(source_path),
            "prior_candidates_path": str(pool_path),
            "prior_candidates_sha256": digest(pool_path),
            "turn_count": len(turns),
            "r01_turn_count": sum(turn["turn_ref"].startswith("R01:") for turn in turns),
            "historical_candidate_count": len(pool),
            "split": split["split"],
            "train_eligible": split["split"] == "train",
            "capacity_approved": False,
            "reason": "Manual evidence/topic capacity review pending." if split["split"] == "train" else "Dev split; excluded from generation.",
        })
    print(json.dumps({"status": "INVENTORY_ONLY_NOT_APPROVED", "split_manifest_sha256": digest(SPLIT_MANIFEST), "roots": rows}, indent=2))


if __name__ == "__main__":
    main()
