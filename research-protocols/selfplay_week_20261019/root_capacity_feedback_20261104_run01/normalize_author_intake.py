#!/usr/bin/env python3
"""Create an auditable intake copy when an author draft uses a bare JSON array."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_hash_for(target_path, target_doc):
    definition = target_doc.get("target_hash_definition")
    if definition:
        expected = (
            "SHA-256 of the targets array encoded as UTF-8 JSON with sorted keys, "
            "compact separators, and ensure_ascii=false."
        )
        if definition != expected:
            raise SystemExit("unsupported frozen-target hash definition")
        targets = json.dumps(
            target_doc["targets"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        computed = hashlib.sha256(targets).hexdigest()
        if target_doc.get("targets_sha256") != computed:
            raise SystemExit("frozen target array hash mismatch")
        return computed, "canonical_targets_array"
    return sha256(target_path), "target_file_bytes"


parser = argparse.ArgumentParser()
parser.add_argument("--author", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()

author = args.author.resolve()
out = args.out.resolve()
if out.exists():
    raise SystemExit(f"intake destination already exists: {out}")

draft_path = author / "CANDIDATES_FIRST_DRAFT.json"
targets_path = author / "TARGETS_FROZEN.json"
draft = json.loads(draft_path.read_text(encoding="utf-8"))
target_doc = json.loads(targets_path.read_text(encoding="utf-8"))
target_hash, target_hash_mode = target_hash_for(targets_path, target_doc)

if isinstance(draft, list):
    candidates = draft
    normalized = {"targets_sha256": target_hash, "candidates": candidates}
    normalization = "wrapped_bare_candidate_array"
elif isinstance(draft, dict) and isinstance(draft.get("candidates"), list):
    candidates = draft["candidates"]
    normalized = dict(draft)
    normalized.setdefault("targets_sha256", target_hash)
    normalization = "copied_candidate_object"
elif isinstance(draft, dict) and draft.get("candidate_id"):
    candidates = [draft]
    normalized = {"targets_sha256": target_hash, "candidates": candidates}
    normalization = "wrapped_single_candidate_object"
else:
    raise SystemExit("unsupported first-draft JSON shape")

if normalized.get("targets_sha256") != target_hash:
    raise SystemExit("draft target hash does not match frozen targets")
if any(c.get("targets_sha256", target_hash) != target_hash for c in candidates):
    raise SystemExit("a candidate target hash does not match frozen targets")

out.mkdir(parents=True)
for path in sorted(author.iterdir()):
    if path.is_file() and path.name != "CANDIDATES_FIRST_DRAFT.json":
        shutil.copy2(path, out / path.name)
shutil.copy2(draft_path, out / "CANDIDATES_FIRST_DRAFT_ORIGINAL.json")
normalized_path = out / "CANDIDATES_FIRST_DRAFT.json"
with normalized_path.open("x", encoding="utf-8") as handle:
    json.dump(normalized, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

copied = json.loads(normalized_path.read_text(encoding="utf-8"))
if copied["candidates"] != candidates:
    raise SystemExit("candidate array changed during normalization")

reconciliation = {
    "schema_version": "author_intake_reconciliation_v1",
    "normalization": normalization,
    "author_original_path": str(draft_path),
    "author_original_sha256": sha256(draft_path),
    "intake_original_copy_path": str(out / "CANDIDATES_FIRST_DRAFT_ORIGINAL.json"),
    "intake_original_copy_sha256": sha256(out / "CANDIDATES_FIRST_DRAFT_ORIGINAL.json"),
    "normalized_intake_path": str(normalized_path),
    "normalized_intake_sha256": sha256(normalized_path),
    "candidate_count": len(candidates),
    "candidate_array_equal": copied["candidates"] == candidates,
    "target_hash_mode": target_hash_mode,
    "frozen_targets_sha256": target_hash,
    "target_file_sha256": sha256(targets_path),
    "author_original_unchanged": sha256(draft_path) == sha256(out / "CANDIDATES_FIRST_DRAFT_ORIGINAL.json"),
}
with (out / "TARGET_HASH_RECONCILIATION.json").open("x", encoding="utf-8") as handle:
    json.dump(reconciliation, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(reconciliation, ensure_ascii=False))
