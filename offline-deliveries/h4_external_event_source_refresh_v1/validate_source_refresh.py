"""Read-only source validation; writes only its independent validation artifacts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / "h4_sequential_ab_v34"
OLD = HERE.parent / "h4_external_event_seed_selection_v1"
sys.path.insert(0, str(RUNTIME))
import source_inventory as inventory
import source_pool as pool


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def extract_hashes(value, target):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in target and isinstance(item, str):
                target[key].add(item)
            extract_hashes(item, target)
    elif isinstance(value, list):
        for item in value:
            extract_hashes(item, target)


def main():
    prior = inventory.collect_prior(RUNTIME / "runs")
    occupied_rows = prior["accepted_uses"] + prior["reserved_uses"]
    consumed = {inventory.event_key(row) for row in occupied_rows}
    old = pool.freeze_unused_source_subset(set(), count=10000, source_dir=OLD)
    new = pool.freeze_unused_source_subset(consumed, count=10000, source_dir=HERE)
    short = pool.freeze_unused_source_subset(consumed, count=3, source_dir=HERE)
    soda = read(HERE / "soda_train_sample.json")
    multiwoz = read(HERE / "multiwoz_train_sample.json")
    assert len(soda["rows"]) == 100 and soda["partial"] is False
    assert [row["row_idx"] for row in soda["rows"]] == list(range(20, 120))
    assert all(row["row"]["split"] == "train" and not row.get("truncated_cells") for row in soda["rows"])
    assert len(multiwoz) == 512 and all(row["turns"] for row in multiwoz)
    assert all([int(turn["turn_id"]) for turn in row["turns"]] == list(range(len(row["turns"]))) for row in multiwoz)
    files = [HERE / name for name in ("soda_train_sample.json", "soda_train_sample_source.json",
             "multiwoz_train_sample.json", "multiwoz_train_source.json")]
    for raw_name, receipt_name in (("soda_train_sample.json", "soda_train_sample_source.json"),
                                   ("multiwoz_train_sample.json", "multiwoz_train_source.json")):
        raw, receipt = binding(HERE / raw_name), read(HERE / receipt_name)
        assert raw["sha256"] == receipt["sha256"] == receipt["download_sha256"]
        assert raw["bytes"] == receipt["bytes"] == receipt["download_bytes"]
        assert receipt["local_is_exact_download_bytes"] and receipt["transformations"] == []
    old_keys = {inventory.event_key(row) for row in old["items"]}
    new_keys = {inventory.event_key(row) for row in new["items"]}
    assert len(old_keys) == len(consumed) == 70 and old_keys == consumed
    assert len(new_keys) == len(new["items"]) == 612
    assert not new_keys & consumed and not new_keys & old_keys
    prior_hashes = {"source_item_sha256": set(), "excerpt_sha256": set()}
    extract_hashes(old, prior_hashes)
    extract_hashes(occupied_rows, prior_hashes)
    evidence_paths = {Path(row["path"]) for row in prior["prior_evidence"]}
    for run in prior["prior_runs"]:
        evidence_paths.update(Path(run).glob("*source*.json"))
        evidence_paths.update(Path(run).glob("*card*.json"))
        evidence_paths.update(Path(run).glob("UNIQUE_EVENT_ALLOCATION.json"))
    for path in sorted(evidence_paths):
        extract_hashes(read(path), prior_hashes)
    overlap, duplicate = {}, {}
    for key in prior_hashes:
        counts = Counter(row[key] for row in new["items"])
        overlap[key] = sorted(set(counts) & prior_hashes[key])
        duplicate[key] = sorted(value for value, count in counts.items() if count > 1)
        assert not overlap[key], ("prior hash overlap", key, overlap[key])
        assert not duplicate[key], ("new duplicate hash", key, duplicate[key])
    assert short["items"] == new["items"][:3]
    report = {
        "status": "PASS", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(HERE), "model_api_calls": 0, "source_reservations_created": 0,
        "download_files": [binding(path) for path in files],
        "runtime_files": [binding(RUNTIME / name) for name in ("source_inventory.py", "source_pool.py")],
        "old_source_files": [binding(OLD / path.name) for path in files],
        "native_item_counts": {"allenai/soda": 100, "MultiWOZ_2.2": 512},
        "available_item_counts": new["selection_provenance"]["available_counts_by_dataset"],
        "consumed_event_count": len(consumed), "consumed_includes_failed_reservations": True,
        "consumed_event_keys_sha256": pool._sha(pool._canonical(sorted(consumed))),
        "consumed_events": [list(key) for key in sorted(consumed)],
        "prior_hash_counts": {key: len(value) for key, value in prior_hashes.items()},
        "prior_event_overlap": [], "prior_hash_overlap": overlap, "internal_hash_duplicates": duplicate,
        "prior_evidence_files": [binding(path) for path in sorted(evidence_paths)],
        "three_item_selector_check": {"selected_count": len(short["items"]),
            "source_ids": [row["source_id"] for row in short["items"]],
            "prefix_matches_full_inventory": True, "reservation_created": False},
        "full_read_only_manifest_sha256": new["manifest_sha256"],
        "limitations": [
            "Source URLs are not pinned to an immutable revision. Exact original response bytes, timestamp, URL and SHA256 are retained.",
            "GitHub API commit lookup returned HTTP 403 rate limit; download used the official repository master raw URL.",
            "The frozen legacy reader labels MultiWOZ as a download subset; in this refresh its local bytes equal the complete upstream shard exactly, verified by download SHA256 and byte count.",
            "This is a read-only availability snapshot. A future live run must recollect global consumption and reserve chosen events under the existing lock.",
            "Dataset metadata stratification and byte/identity uniqueness do not certify semantic novelty or training quality."
        ]
    }
    for name, value in (("SOURCE_AVAILABILITY_MANIFEST.json", new), ("VALIDATION.json", report)):
        with (HERE / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "source_dir", "native_item_counts", "available_item_counts",
                      "consumed_event_count", "prior_hash_counts", "prior_hash_overlap", "internal_hash_duplicates",
                      "three_item_selector_check", "model_api_calls", "source_reservations_created")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
