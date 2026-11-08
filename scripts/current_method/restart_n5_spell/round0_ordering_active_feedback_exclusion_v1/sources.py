"""Hash-bound local sources; no implicit latest, service config, or credentials."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import VERSION

HASH_ONLY_PATHS = set()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    HASH_ONLY_PATHS.add(path)
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    finally:
        HASH_ONLY_PATHS.discard(path)
    return h.hexdigest()


def read(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return json.load(stream)


class Sources:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = read(self.manifest_path)
        if self.manifest.get("filter_version") != VERSION:
            raise ValueError("source_manifest_version_mismatch")
        self.identities = {str(self.manifest_path): file_sha(self.manifest_path)}

    def path(self, name):
        spec = self.manifest["inputs"][name]
        path = Path(spec["path"]).resolve()
        self.verify(path, spec["sha256"])
        return path

    def verify(self, path, expected):
        path = Path(path).resolve()
        observed = file_sha(path)
        if observed != expected:
            raise ValueError(f"source_sha_mismatch:{path}")
        self.identities[str(path)] = observed
        return path

    def get(self, name):
        return read(self.path(name))

    def unchanged(self):
        return [{"path": p, "before_sha256": sha, "after_sha256": file_sha(p),
                 "unchanged": file_sha(p) == sha} for p, sha in sorted(self.identities.items())]


def authority(sources):
    inventory = sources.get("round0_inventory")
    splits = {row["root_id"]: row for row in sources.get("round0_split")}
    if len(splits) != len(sources.get("round0_split")):
        raise ValueError("duplicate_split_root")
    corpus = next(row for row in sources.get("corpus_scope") if row["corpus_id"] == "ROUND0_TRAIN")
    if corpus["source_round"] != "ROUND0_PARENT_PLUS_R01":
        raise ValueError("bootstrap_lineage_mismatch")
    rows, original = {}, {}
    for root in inventory["roots"]:
        split = splits.get(root["root_id"])
        if split is None:
            if root["candidate_count"]:
                raise ValueError("candidate_bearing_root_missing_split")
            continue
        if split["split"] != "train":
            continue
        for name in ("candidate_pool", "dialogue", "source_metadata"):
            sources.verify(root[name + "_path"], root[name + "_sha256"])
        if any(root[k] != split[k] for k in ("candidate_pool_sha256", "dialogue_sha256")):
            raise ValueError("inventory_split_snapshot_mismatch")
        pool = read(root["candidate_pool_path"])
        if len(pool) != root["candidate_count"]:
            raise ValueError("candidate_pool_count_mismatch")
        for qa in pool:
            cid = qa["candidate_id"]
            if cid in rows or qa["root_id"] != root["root_id"]:
                raise ValueError("candidate_identity_conflict")
            original[cid] = qa
            rows[cid] = {
                "candidate_id": cid, "root_id": root["root_id"], "split": "train",
                "corpus": "ROUND0_TRAIN", "source_round": corpus["source_round"],
                "question_type": qa["question_type"], "candidate_record_sha256": value_sha(qa),
                "source_snapshot": {k: root[k] for k in (
                    "candidate_pool_path", "candidate_pool_sha256", "dialogue_path",
                    "dialogue_sha256", "source_metadata_path", "source_metadata_sha256")},
            }
    counts = {}
    for row in rows.values():
        counts[row["root_id"]] = counts.get(row["root_id"], 0) + 1
    if counts != corpus["root_candidate_counts"] or len(rows) != corpus["candidate_count"]:
        raise ValueError("bootstrap_train_candidate_scope_mismatch")
    return rows, original, splits


def scope_filter(bank, frozen_scope, authoritative, originals, splits):
    scope = {row["candidate_id"]: row for row in frozen_scope}
    candidates = {row["candidate_id"]: row for row in bank}
    if len(candidates) != len(bank) or len(scope) != len(frozen_scope) or set(scope) != set(candidates):
        raise ValueError("bank_scope_identity_mismatch")
    if not set(authoritative) <= set(candidates):
        raise ValueError("round0_candidates_missing_from_bank")
    excluded, kept = [], []
    for cid, entry in sorted(candidates.items()):
        qa, item = entry["candidate"], scope[cid]
        root = qa["root_id"]
        if qa["candidate_id"] != cid or item["corpus_id"] != entry["corpus_id"] or item["normalized_experiment_arm"] != entry["normalized_experiment_arm"]:
            raise ValueError("bank_nested_identity_or_lineage_mismatch")
        if entry["split"] != "train" or item["split"] != "train" or splits.get(root, {}).get("split") != "train":
            raise ValueError(f"non_train_candidate:{cid}")
        if item["root_id"] != root or item["question_type"] != qa["question_type"]:
            raise ValueError("bank_scope_fields_mismatch")
        if cid in authoritative:
            if qa != originals[cid] or entry["context_signature"] != authoritative[cid]["source_snapshot"]["dialogue_sha256"]:
                raise ValueError(f"round0_bank_snapshot_mismatch:{cid}")
            if entry["normalized_experiment_arm"] != "ROUND0_PARENT_PLUS_R01":
                raise ValueError("round0_arm_mismatch")
            identity = dict(authoritative[cid])
        else:
            if entry["normalized_experiment_arm"] not in {"E0Q0_CONTROL_BLIND", "E1Q0_FEEDBACK_BLIND", "E1Q1_FEEDBACK_GUIDED"}:
                raise ValueError("later_candidate_lineage_unknown")
            identity = {"candidate_id": cid, "root_id": root, "split": "train",
                        "corpus": entry["corpus_id"], "source_round": entry["normalized_experiment_arm"],
                        "question_type": qa["question_type"], "candidate_record_sha256": value_sha(qa),
                        "source_snapshot": {"context_signature": entry["context_signature"], "source_path": item["source_path"]}}
        identity.update({"filter_version": VERSION, "bank_record_sha256": value_sha(entry),
                         "normalized_experiment_arm": entry["normalized_experiment_arm"],
                         "solver_prompt_fingerprint": item["solver_prompt_fingerprint"]})
        remove = cid in authoritative and authoritative[cid]["question_type"] == "ordering"
        identity["active_feedback_scope_allowed"] = not remove
        if remove:
            identity.update({"active_feedback_eligible": False,
                             "reason": "ROUND0 ordering active feedback is temporarily out of scope for this iteration",
                             "semantic_judgment": "NONE", "historical_training_data_deletion": False})
            excluded.append(identity)
        else:
            kept.append(identity)
    return excluded, kept
