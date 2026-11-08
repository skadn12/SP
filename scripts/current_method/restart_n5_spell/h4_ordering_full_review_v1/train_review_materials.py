"""Build frozen, offline review materials for ordering examples used in training.

This script does not adjudicate examples.  It resolves the exact candidate and
dialogue frozen by each policy manifest, verifies provenance, and renders one
readable file per root for a subsequent full-dialogue manual review.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round0_b0_matched_training_v1_live_2"
)
OUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round0_train_ordering_full_manual_review_v1_live_1"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(path: Path, candidate_id: str) -> dict[str, Any]:
    pool = json.loads(path.read_text(encoding="utf-8"))
    candidates = pool if isinstance(pool, list) else pool.get("candidates", [])
    matches = [candidate for candidate in candidates if candidate.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {candidate_id} in {path}, found {len(matches)}")
    return matches[0]


def fmt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    roots_dir = OUT / "roots"
    roots_dir.mkdir()

    policy_rows: dict[str, list[dict[str, Any]]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    membership: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    failures: list[str] = []

    for policy in ("clean", "inclusive"):
        manifest = RUN / f"m_{policy}_policy_training_manifest.jsonl"
        rows = [row for row in load_jsonl(manifest) if row.get("question_type") == "ordering"]
        policy_rows[policy] = rows
        source_manifests[policy] = {
            "path": str(manifest),
            "sha256": sha256(manifest),
            "ordering_count": len(rows),
        }
        for row in rows:
            cid = str(row["candidate_id"])
            pool_path = Path(row["candidate_pool_path"])
            dialogue_path = Path(row["dialogue_path"])
            if sha256(pool_path) != row["candidate_pool_sha256"]:
                failures.append(f"candidate_pool_sha256:{policy}:{cid}")
            if sha256(dialogue_path) != row["dialogue_sha256"]:
                failures.append(f"dialogue_sha256:{policy}:{cid}")
            candidate = load_candidate(pool_path, cid)
            if candidate.get("question_type") != "ordering":
                failures.append(f"candidate_question_type:{policy}:{cid}")
            if list(candidate.get("gold") or []) != list(row.get("canonical_gold") or []):
                failures.append(f"canonical_gold:{policy}:{cid}")

            reverse = {canonical: displayed for displayed, canonical in row["display_to_canonical"].items()}
            expected_displayed = [reverse[item_id] for item_id in row["canonical_gold"]]
            if expected_displayed != row["displayed_target"]:
                failures.append(f"displayed_target:{policy}:{cid}")
            if row["target_text"] != f"<box>{','.join(expected_displayed)}</box>":
                failures.append(f"target_text:{policy}:{cid}")

            membership[cid][policy] = {
                "selection_role": row["selection_role"],
                "evidence_status": row["evidence_status"],
                "display_to_canonical": row["display_to_canonical"],
                "displayed_target": row["displayed_target"],
                "target_text": row["target_text"],
                "messages_sha256": row["messages_sha256"],
            }
            if cid in by_id:
                existing = by_id[cid]
                if existing["candidate"] != candidate or existing["dialogue_path"] != str(dialogue_path):
                    failures.append(f"cross_policy_candidate_mismatch:{cid}")
            else:
                dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
                by_id[cid] = {
                    "candidate": candidate,
                    "dialogue_path": str(dialogue_path),
                    "candidate_pool_path": str(pool_path),
                    "turns": dialogue.get("turns", []),
                }

    if failures:
        raise RuntimeError("provenance validation failed: " + ", ".join(failures))

    root_candidates: dict[str, list[str]] = defaultdict(list)
    bundle_path = OUT / "train_ordering_review_bundle.jsonl"
    worklist_path = OUT / "train_ordering_review_worklist.tsv"
    fields = [
        "review_index", "candidate_id", "root_id", "policies", "clean_role",
        "inclusive_role", "manifest_evidence_strata", "generation_source", "question",
        "canonical_gold", "items", "evidence_refs", "evidence_text",
        "evidence_ref_order", "dialogue_path", "candidate_pool_path", "gold_status",
        "evidence_status", "primary_status", "corrected_order", "item_turn_refs", "note",
        "review_basis",
    ]
    work_rows: list[dict[str, str]] = []
    bundle_rows: list[dict[str, Any]] = []

    for index, cid in enumerate(sorted(by_id), 1):
        data = by_id[cid]
        candidate = data["candidate"]
        turns = data["turns"]
        root_id = str(candidate["root_id"])
        root_candidates[root_id].append(cid)
        positions = {str(turn["turn_ref"]): i for i, turn in enumerate(turns)}
        turn_text = {str(turn["turn_ref"]): str(turn.get("text", "")) for turn in turns}
        refs = [str(ref) for ref in candidate.get("evidence_refs") or []]
        item_ids = list((candidate.get("items") or {}).keys())
        ref_order = []
        if len(refs) == len(item_ids) and all(ref in positions for ref in refs):
            ref_order = [item_ids[i] for i, _ref in sorted(enumerate(refs), key=lambda pair: positions[pair[1]])]
        policies = sorted(membership[cid])
        strata = sorted({membership[cid][policy]["evidence_status"] for policy in policies})
        row = {
            "review_index": f"T{index:03d}",
            "candidate_id": cid,
            "root_id": root_id,
            "policies": ",".join(policies),
            "clean_role": membership[cid].get("clean", {}).get("selection_role", ""),
            "inclusive_role": membership[cid].get("inclusive", {}).get("selection_role", ""),
            "manifest_evidence_strata": ",".join(strata),
            "generation_source": str(candidate.get("generation_source") or ""),
            "question": str(candidate.get("question") or ""),
            "canonical_gold": ",".join(candidate.get("gold") or []),
            "items": fmt_json(candidate.get("items") or {}),
            "evidence_refs": fmt_json(refs),
            "evidence_text": fmt_json({ref: turn_text.get(ref, "<MISSING>") for ref in refs}),
            "evidence_ref_order": ",".join(ref_order),
            "dialogue_path": data["dialogue_path"],
            "candidate_pool_path": data["candidate_pool_path"],
            "gold_status": "",
            "evidence_status": "",
            "primary_status": "",
            "corrected_order": "",
            "item_turn_refs": "",
            "note": "",
            "review_basis": "",
        }
        work_rows.append(row)
        bundle_rows.append({
            **{key: row[key] for key in fields[:16]},
            "candidate": candidate,
            "policy_membership": membership[cid],
            "turns": turns,
        })

    with worklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(work_rows)
    with bundle_path.open("w", encoding="utf-8") as handle:
        for row in bundle_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for root_id in sorted(root_candidates):
        first = by_id[root_candidates[root_id][0]]
        turns = first["turns"]
        lines = [f"ROOT {root_id}", f"CANDIDATES {len(root_candidates[root_id])}", ""]
        for cid in root_candidates[root_id]:
            candidate = by_id[cid]["candidate"]
            lines.extend([
                f"### {cid} [{','.join(sorted(membership[cid]))}]",
                f"QUESTION: {candidate.get('question')}",
                f"GOLD: {','.join(candidate.get('gold') or [])}",
                f"EVIDENCE_REFS: {fmt_json(candidate.get('evidence_refs') or [])}",
                "ITEMS:",
            ])
            lines.extend(f"  {item_id}: {text}" for item_id, text in (candidate.get("items") or {}).items())
            lines.append("EVIDENCE TEXT:")
            lookup = {str(turn["turn_ref"]): turn for turn in turns}
            for ref in candidate.get("evidence_refs") or []:
                turn = lookup.get(str(ref), {})
                lines.append(f"  {ref} | {turn.get('speaker', '<MISSING>')}: {turn.get('text', '<MISSING>')}")
            lines.append("")
        lines.extend(["=== FULL DIALOGUE ===", ""])
        lines.extend(
            f"{turn.get('turn_ref')} | {turn.get('speaker')}: {turn.get('text')}"
            for turn in turns
        )
        (roots_dir / f"{root_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    policy_id_sets = {policy: {row["candidate_id"] for row in rows} for policy, rows in policy_rows.items()}
    report = {
        "audit_version": "h4_round0_train_ordering_full_manual_review_materials_v1",
        "status": "MATERIALS_READY_REVIEW_PENDING",
        "offline_only": True,
        "solver_calls": 0,
        "external_api_calls": 0,
        "memorax_calls": 0,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
        "source_run": str(RUN),
        "source_manifests": source_manifests,
        "policy_ordering_counts": {policy: len(rows) for policy, rows in policy_rows.items()},
        "unique_candidate_count": len(by_id),
        "shared_candidate_count": len(policy_id_sets["clean"] & policy_id_sets["inclusive"]),
        "clean_only_candidate_count": len(policy_id_sets["clean"] - policy_id_sets["inclusive"]),
        "inclusive_only_candidate_count": len(policy_id_sets["inclusive"] - policy_id_sets["clean"]),
        "unique_root_count": len(root_candidates),
        "policy_role_counts": {
            policy: dict(sorted(Counter(row["selection_role"] for row in rows).items()))
            for policy, rows in policy_rows.items()
        },
        "provenance_validation_failures": failures,
        "worklist": str(worklist_path),
        "bundle": str(bundle_path),
        "root_materials_dir": str(roots_dir),
    }
    (OUT / "materials_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
