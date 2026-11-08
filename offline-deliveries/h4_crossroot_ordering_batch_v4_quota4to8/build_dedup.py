"""Freeze all known candidate drafts for cross-type novelty constraints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE / "h4_crossroot_ordering_batch_v3_pplusr01_quota3to5"
MULTI = BASE / "h4_crossroot_multiselect_batch_v25_quota_trial"
REVISION = BASE / "h4_crossroot_multiselect_codex_revision_v26_191749"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    old_manifest = load(OLD / "manifest.json")
    sources = []
    for source in old_manifest["sources"]:
        root = source["root_id"]
        pool_path = Path(source["prior_candidates_path"])
        if digest(pool_path) != source["prior_candidates_sha256"]:
            raise ValueError(f"old pool changed: {root}")
        records = []
        inputs = []

        def include(path: Path, origin: str, questions: list, review: dict | None = None) -> None:
            inputs.append({"path": str(path), "sha256": digest(path), "origin": origin,
                           "candidate_count": len(questions)})
            decisions = {row.get("candidate_id"): row for row in review.get("question_decisions", [])} if review else {}
            for question in questions:
                if not isinstance(question, dict):
                    raise ValueError(f"invalid candidate row: {path}")
                row = dict(question)
                row.setdefault("root_id", root)
                if row["root_id"] != root:
                    raise ValueError(f"candidate root mismatch: {path}")
                row["dedup_source"] = origin
                decision = decisions.get(row.get("candidate_id"))
                row["audit_status"] = ("content_approved" if decision["approved"] else "content_rejected") if decision else "unverified_or_historical"
                if origin == "v25_multiselect_draft":
                    row["question_type"] = "multi_select"
                records.append(row)

        include(pool_path, "historical_candidate_pool_all_types", load(pool_path))
        order_path = OLD / root / "initial" / "PARSED_RESPONSE.json"
        include(order_path, "v3_ordering_raw_draft", load(order_path).get("questions", []))
        multi_path = MULTI / root / "batch-01" / "MODEL_DRAFT.json"
        review_path = MULTI / root / "batch-01" / "CONTENT_REVIEW.json"
        review = load(review_path) if review_path.exists() else None
        include(multi_path, "v25_multiselect_draft", load(multi_path).get("questions", []), review)
        if review_path.exists():
            inputs.append({"path": str(review_path), "sha256": digest(review_path), "origin": "v25_content_review"})
        if root.endswith("191749"):
            revision_path = REVISION / "READY_FOR_EVALUATION.json"
            include(revision_path, "v26_codex_revision_diagnostic", load(revision_path).get("questions", []))

        combined_path = HERE / "dedup_inputs" / root / "combined_candidates.json"
        provenance_path = HERE / "dedup_inputs" / root / "PROVENANCE.json"
        save_new(combined_path, {"root_id": root, "questions": records})
        save_new(provenance_path, {"root_id": root, "inputs": inputs,
                                   "combined_path": str(combined_path),
                                   "combined_sha256": digest(combined_path),
                                   "candidate_count": len(records),
                                   "note": "Candidate records constrain novelty; they do not assert training admission."})
        sources.append({"root_id": root, "source_path": source["source_path"],
                        "source_sha256": source["source_sha256"],
                        "prior_candidates_path": str(combined_path),
                        "prior_candidates_sha256": digest(combined_path)})
    save_new(HERE / "manifest.json", {"sources": sources})
    print(json.dumps({"status": "DEDUP_INPUTS_FROZEN", "roots": len(sources),
                      "counts": {row["root_id"]: load(Path(row["prior_candidates_path"]))["questions"].__len__()
                                 for row in sources}}))


if __name__ == "__main__":
    main()
