"""Offline rescore frozen Solver outputs with the completed ordering audit.

This never calls a model. It changes only the scoring gold in memory:
valid/evidence_mismatch rows keep their original gold, invalid_gold rows use
the manually corrected order, and ambiguous ordering rows are excluded.
Non-ordering candidates retain their original gold. Existing raw Solver
outputs and source manifests are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DATA = BASE / "h4_round0_b0_matched_training_v1_live_1"
DIFFICULTY = BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
DEV_RUN = BASE / "h4_round0_common_dev_eval_v1_live_3"
REVIEW = BASE / "h4_round0_dev_test_ordering_review_v1_live_1/final_manual_review.tsv"
DEFAULT_OUT = BASE / "h4_round0_ordering_corrected_offline_rescore_v1_live_1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_review(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["corrected_order"] = [x for x in row["corrected_order"].split(",") if x]
            rows[row["candidate_id"]] = row
    return rows


def corrected_gold(candidate: dict[str, Any], review: dict[str, dict[str, Any]]) -> tuple[list[str] | None, str, str]:
    cid = str(candidate["candidate_id"])
    # The frozen split manifests retain the full candidate object under the
    # ``candidate`` key; top-level fields only contain indexing metadata.
    payload = candidate.get("candidate") or candidate
    if str(payload.get("question_type")) != "ordering":
        return ([str(x) for x in payload.get("gold") or []], "not_ordering", "include")
    row = review.get(cid)
    if not row:
        raise RuntimeError(f"ordering_review_missing:{cid}")
    status = str(row["gold_status"])
    if status in {"ambiguous", "unsupported"}:
        return (None, "ambiguous", "exclude")
    if status == "invalid_gold":
        order = list(row["corrected_order"])
        if order != ["I1", "I2", "I3", "I4"] and sorted(order) != ["I1", "I2", "I3", "I4"]:
            raise RuntimeError(f"invalid_corrected_order:{cid}:{order}")
        return (order, "manual_corrected", "include")
    if status == "valid":
        return ([str(x) for x in payload.get("gold") or []], "original_valid", "include")
    raise RuntimeError(f"unknown_review_status:{cid}:{status}")


def request_correct(request: dict[str, Any], gold: list[str], qtype: str) -> bool | None:
    if not request.get("scored"):
        return None
    prediction = [str(x) for x in request.get("canonical_prediction") or []]
    if qtype == "single_choice":
        return prediction == gold
    if qtype == "multi_select":
        return sorted(set(prediction)) == sorted(set(gold))
    if qtype == "ordering":
        return prediction == gold
    raise RuntimeError(f"unknown_qtype:{qtype}")


def rescore_result(result: dict[str, Any], candidate: dict[str, Any], gold: list[str]) -> dict[str, Any]:
    qtype = str((candidate.get("candidate") or candidate)["question_type"])
    modes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for req in result.get("requests") or []:
        modes[str(req.get("context_mode"))].append(req)
    score: dict[str, Any] = {}
    for mode in ("no_context", "full_context"):
        reqs = modes.get(mode, [])
        values = [request_correct(req, gold, qtype) for req in reqs]
        scored = [x for x in values if x is not None]
        score[mode] = {
            "correct_count": sum(bool(x) for x in scored),
            "total_count": len(scored),
            "technical_or_unscored_count": len(values) - len(scored),
            "accuracy": (sum(bool(x) for x in scored) / len(scored)) if scored else None,
        }
    return score


def aggregate(rows: list[dict[str, Any]], *, breakdowns: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"candidate_count": len(rows), "question_type_counts": dict(Counter(r["question_type"] for r in rows))}
    for mode in ("no_context", "full_context"):
        correct = sum(r["score"][mode]["correct_count"] for r in rows)
        total = sum(r["score"][mode]["total_count"] for r in rows)
        out[mode] = {"correct_count": correct, "total_count": total, "accuracy": correct / total if total else None}
    n = out["no_context"]; f = out["full_context"]
    out["context_gain"] = (f["accuracy"] - n["accuracy"]) if n["accuracy"] is not None and f["accuracy"] is not None else None
    if breakdowns:
        # Child summaries must not recursively build their own breakdowns.
        out["by_question_type"] = {
            qtype: aggregate(
                [r for r in rows if r["question_type"] == qtype], breakdowns=False
            )
            for qtype in sorted(set(r["question_type"] for r in rows))
        }
        by_root = {
            root: aggregate(
                [r for r in rows if r["root_id"] == root], breakdowns=False
            )
            for root in sorted(set(r["root_id"] for r in rows))
        }
        root_values = {
            mode: [v[mode]["accuracy"] for v in by_root.values() if v[mode]["accuracy"] is not None]
            for mode in ("no_context", "full_context")
        }
        out["root_equal"] = {
            mode: (sum(values) / len(values) if values else None)
            for mode, values in root_values.items()
        }
        out["root_equal"]["context_gain"] = (
            out["root_equal"]["full_context"] - out["root_equal"]["no_context"]
            if out["root_equal"]["full_context"] is not None
            and out["root_equal"]["no_context"] is not None
            else None
        )
        out["by_root"] = by_root
    return out


def load_candidates(split: str) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): dict(row) for row in read_jsonl(DATA / f"{split}_full_eval_manifest.jsonl")}


def result_map(scope: Path) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in read_jsonl(scope / "candidate_results.jsonl")}


def m0_result_map(split: str) -> dict[str, dict[str, Any]]:
    return {str(read_json(p)["candidate_id"]): read_json(p) for p in (DIFFICULTY / "candidate_results").glob("*.json") if read_json(p).get("split") in {None, split}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"output_exists:{out}")
    out.mkdir(parents=True)
    review = load_review(REVIEW)
    all_reviewed = set(review)
    candidates_by_split = {split: load_candidates(split) for split in ("dev", "test")}
    expected_ordering = {cid for cands in candidates_by_split.values() for cid, c in cands.items() if c.get("question_type") == "ordering"}
    if expected_ordering != all_reviewed:
        raise RuntimeError(f"review_identity_mismatch:missing={sorted(expected_ordering-all_reviewed)}:extra={sorted(all_reviewed-expected_ordering)}")

    scopes: dict[str, dict[str, dict[str, Any]]] = {
        "dev/M0": m0_result_map("dev"),
        "dev/M_clean_policy": result_map(DEV_RUN / "clean_dev"),
        "dev/M_inclusive_policy": result_map(DEV_RUN / "inclusive_dev"),
        "test/M0": m0_result_map("test"),
    }
    outputs: dict[str, Any] = {}
    inventory: dict[str, Any] = {"review": str(REVIEW), "review_sha256": sha256(REVIEW), "splits": {}, "scopes": {}}
    for key, results in scopes.items():
        split, model = key.split("/", 1)
        rows: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for cid, candidate in candidates_by_split[split].items():
            if cid not in results:
                raise RuntimeError(f"result_missing:{key}:{cid}")
            gold, gold_source, inclusion = corrected_gold(candidate, review)
            if inclusion == "exclude":
                excluded.append({"candidate_id": cid, "root_id": candidate["root_id"], "reason": gold_source})
                continue
            result = results[cid]
            payload = candidate.get("candidate") or candidate
            original_gold = [str(x) for x in payload.get("gold") or []]
            row = {
                "candidate_id": cid,
                "root_id": candidate["root_id"],
                "question_type": (candidate.get("candidate") or candidate)["question_type"],
                "generation_source": (candidate.get("candidate") or candidate).get("generation_source"),
                "original_gold": original_gold,
                "rescored_gold": gold,
                "gold_source": gold_source,
                "review_gold_status": review.get(cid, {}).get("gold_status"),
                "review_evidence_status": review.get(cid, {}).get("evidence_status"),
                "score": rescore_result(result, candidate, gold),
                "original_score": rescore_result(result, candidate, original_gold),
            }
            rows.append(row)
        summary = aggregate(rows)
        original_summary = aggregate(
            [{**row, "score": row["original_score"]} for row in rows]
        )
        summary["original_gold_same_subset"] = original_summary
        summary["delta_corrected_minus_original_same_subset"] = {
            "no_context_accuracy": (
                summary["no_context"]["accuracy"] - original_summary["no_context"]["accuracy"]
                if summary["no_context"]["accuracy"] is not None and original_summary["no_context"]["accuracy"] is not None else None
            ),
            "full_context_accuracy": (
                summary["full_context"]["accuracy"] - original_summary["full_context"]["accuracy"]
                if summary["full_context"]["accuracy"] is not None and original_summary["full_context"]["accuracy"] is not None else None
            ),
            "context_gain": (
                summary["context_gain"] - original_summary["context_gain"]
                if summary["context_gain"] is not None and original_summary["context_gain"] is not None else None
            ),
        }
        scope_name = key.replace("/", "__")
        (out / f"{scope_name}_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        (out / f"{scope_name}_excluded.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs[key] = summary
        inventory["scopes"][key] = {"candidate_count_included": len(rows), "candidate_count_excluded": len(excluded), "rows": str(out / f"{scope_name}_rows.jsonl"), "excluded": str(out / f"{scope_name}_excluded.json")}
    inventory["splits"] = {}
    for split, candidates in candidates_by_split.items():
        excluded_status = Counter(
            review[cid]["gold_status"]
            for cid, candidate in candidates.items()
            if candidate.get("question_type") == "ordering"
            and review[cid]["gold_status"] in {"ambiguous", "unsupported"}
        )
        inventory["splits"][split] = {
            "candidate_count": len(candidates),
            "ordering_review_count": sum(c.get("question_type") == "ordering" for c in candidates.values()),
            "excluded_by_gold_status": dict(sorted(excluded_status.items())),
            "excluded_total": sum(excluded_status.values()),
        }
    report = {"version": "h4_round0_ordering_corrected_offline_rescore_v1", "offline_only": True, "solver_calls": 0, "external_api_calls": 0, "memorax_calls": 0, "training_started": False, "semantic_auditor_calls": 0, "status": "COMPLETE", "outputs": outputs, "inventory": inventory}
    (out / "final_offline_rescore_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out), "scopes": {k: {"included": inventory["scopes"][k]["candidate_count_included"], "excluded": inventory["scopes"][k]["candidate_count_excluded"], "full": v["full_context"]["accuracy"], "no": v["no_context"]["accuracy"], "gain": v["context_gain"]} for k, v in outputs.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
