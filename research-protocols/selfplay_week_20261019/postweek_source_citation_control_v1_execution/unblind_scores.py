from __future__ import annotations

import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORE_DIR = HERE / "outputs" / "scoring"
REVIEW_DIR = HERE / "outputs" / "blind_review"


def read(path: Path):
    return json.loads(path.read_text())


def bind(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def check(row: dict) -> None:
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != row["bytes"] or bind(path)["sha256"] != row["sha256"]:
        raise RuntimeError("REVIEW_BINDING_MISMATCH:" + str(path))


def main() -> None:
    out = SCORE_DIR / "UNBLINDED_MECHANICAL_REPORT.json"
    if out.exists():
        raise RuntimeError("UNBLINDED_REPORT_EXISTS_NO_OVERWRITE")
    freeze_path = SCORE_DIR / "BLIND_SCORE_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("BLIND_SCORE_FREEZE_REQUIRED")
    freeze = read(freeze_path)
    for key in ("raw_response_freeze", "score_file", "private_unblind_key"):
        row = freeze[key]
        p = Path(row["path"])
        if not p.is_file():
            raise RuntimeError("BLIND_SCORE_BINDING_MISMATCH:" + key)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if p.stat().st_size != row["bytes"] or h != row["sha256"]:
            raise RuntimeError("BLIND_SCORE_BINDING_MISMATCH:" + key)
    blind = read(Path(freeze["score_file"]["path"]))
    review_freeze = read(REVIEW_DIR / "PACKET_FREEZE.json")
    for row in review_freeze["packet_bindings"] + [review_freeze["private_member_binding"], review_freeze["blind_score_freeze"]]:
        check(row)
    member_map = read(Path(review_freeze["private_member_binding"]["path"]))["review_id_to_blind_items"]
    review_results = {}
    review_files = []
    for binding in review_freeze["packet_bindings"]:
        packet_path = Path(binding["path"])
        packet = read(packet_path)
        expected_ids = {r["review_id"] for r in packet["review_rows"]}
        result_path = packet_path.with_name(packet_path.stem + ".review.json")
        result = read(result_path)
        if result.get("status") != "COMPLETE_BLINDED_REVIEW" or result.get("condition_labels_accessed") is not False or result.get("final_answers_accessed") is not False:
            raise RuntimeError("BLIND_SEMANTIC_REVIEW_NOT_COMPLETE")
        result_rows = result.get("reviews", [])
        if len(result_rows) != len(expected_ids) or {r["review_id"] for r in result_rows} != expected_ids:
            raise RuntimeError("BLIND_SEMANTIC_REVIEW_COVERAGE_MISMATCH")
        for row in result_rows:
            if not isinstance(row.get("sufficient"), bool) or not isinstance(row.get("semantic_judgment_correct"), bool):
                raise RuntimeError("SEMANTIC_REVIEW_VALUE_TYPE_INVALID")
            review_results[row["review_id"]] = row
        review_files.append(bind(result_path))
    if set(review_results) != set(member_map):
        raise RuntimeError("REVIEW_BUNDLE_SET_MISMATCH")
    item_review_by_blind = defaultdict(dict)
    for review_id, members in member_map.items():
        for member in members:
            item_review_by_blind[member["blind_id"]][member["displayed_label"]] = review_results[review_id]
    if len(item_review_by_blind) != 24 or any(len(v) != 6 for v in item_review_by_blind.values()):
        raise RuntimeError("BLIND_REVIEW_ITEM_DENOMINATOR_MISMATCH")
    semantic_freeze_path = REVIEW_DIR / "SEMANTIC_REVIEW_FREEZE.json"
    with semantic_freeze_path.open("x") as f:
        json.dump({"status": "BLIND_SEMANTIC_REVIEW_COMMITTED_BEFORE_UNBLIND",
            "packet_freeze": bind(REVIEW_DIR / "PACKET_FREEZE.json"),
            "review_bindings": review_files, "review_bundles": len(review_results),
            "item_denominator": 144, "request_denominator": 24}, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    # Condition mappings are opened only after the semantic reviews are committed.
    key = read(Path(freeze["private_unblind_key"]["path"]))
    rows = []
    for row in blind["rows"]:
        meta = key["mapping"][row["blind_id"]]
        item_reviews = item_review_by_blind[row["blind_id"]]
        semantic = {
            "sufficient_citation_items": sum(r["sufficient"] for r in item_reviews.values()),
            "semantically_correct_judgment_items": sum(r["semantic_judgment_correct"] for r in item_reviews.values()),
            "citation_sufficiency_rate": sum(r["sufficient"] for r in item_reviews.values()) / 6,
            "all_citations_sufficient": all(r["sufficient"] for r in item_reviews.values()),
            "semantic_item_reviews": item_reviews,
        }
        rows.append({**meta, **row, **semantic})
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["condition"]] = row
    if len(by_pair) != 12 or any(set(v) != {"C0", "C1"} for v in by_pair.values()):
        raise RuntimeError("MATCHED_PAIR_SCOPE_INVALID")
    pair_results = []
    for pair_id, pair in by_pair.items():
        c0, c1 = pair["C0"], pair["C1"]
        pair_results.append({
            "pair_id": pair_id, "root_id": c0["root_id"],
            "display_variant": c0["display_variant"], "replicate": c0["replicate"],
            "C0_exact": c0["exact_order_correct"], "C1_exact": c1["exact_order_correct"],
            "exact_delta_C1_minus_C0": int(c1["exact_order_correct"]) - int(c0["exact_order_correct"]),
            "C0_anchor_hit_rate": c0["citation_anchor_hit_rate"],
            "C1_anchor_hit_rate": c1["citation_anchor_hit_rate"],
            "anchor_hit_delta_C1_minus_C0": c1["citation_anchor_hit_rate"] - c0["citation_anchor_hit_rate"],
            "C0_pairwise_accuracy": c0["pairwise_accuracy"],
            "C1_pairwise_accuracy": c1["pairwise_accuracy"],
            "C0_all_supported_judgments": c0["all_judgments_supported"],
            "C1_all_supported_judgments": c1["all_judgments_supported"],
            "C0_format_valid": c0["json_schema_valid"], "C1_format_valid": c1["json_schema_valid"],
        })
    root_rows = []
    roots = sorted({r["root_id"] for r in rows})
    for root in roots:
        root_pairs = [r for r in pair_results if r["root_id"] == root]
        cond_rows = {}
        for cond in ("C0", "C1"):
            subset = [r for r in rows if r["root_id"] == root and r["condition"] == cond]
            cond_rows[cond] = {
                "requests": len(subset),
                "exact_correct": sum(r["exact_order_correct"] for r in subset),
                "exact_accuracy": sum(r["exact_order_correct"] for r in subset) / len(subset),
                "pairwise_correct": sum(r["pairwise_correct"] for r in subset),
                "pairwise_opportunities": sum(r["pairwise_denominator"] for r in subset),
                "pairwise_accuracy": sum(r["pairwise_correct"] for r in subset) / sum(r["pairwise_denominator"] for r in subset),
                "schema_valid": sum(r["json_schema_valid"] for r in subset),
                "sufficient_citation_items": sum(r["sufficient_citation_items"] for r in subset),
                "citation_sufficiency_rate": sum(r["sufficient_citation_items"] for r in subset) / (len(subset) * 6),
                "all_supported_judgments": sum(r["all_judgments_supported"] for r in subset),
                "anchor_hits": sum(r["first_sufficient_anchor_hits"] for r in subset),
                "anchor_opportunities": sum(r["item_denominator"] for r in subset),
                "anchor_hit_rate": sum(r["first_sufficient_anchor_hits"] for r in subset) / sum(r["item_denominator"] for r in subset),
            }
        root_rows.append({
            "root_id": root, "C0": cond_rows["C0"], "C1": cond_rows["C1"],
            "exact_accuracy_delta_C1_minus_C0": cond_rows["C1"]["exact_accuracy"] - cond_rows["C0"]["exact_accuracy"],
            "anchor_hit_rate_delta_C1_minus_C0": cond_rows["C1"]["anchor_hit_rate"] - cond_rows["C0"]["anchor_hit_rate"],
            "paired_exact_deltas": [r["exact_delta_C1_minus_C0"] for r in root_pairs],
        })
    overall = {}
    for cond in ("C0", "C1"):
        subset = [r for r in rows if r["condition"] == cond]
        overall[cond] = {
            "requests": len(subset),
            "exact_correct": sum(r["exact_order_correct"] for r in subset),
            "exact_accuracy": sum(r["exact_order_correct"] for r in subset) / len(subset),
            "pairwise_correct": sum(r["pairwise_correct"] for r in subset),
            "pairwise_opportunities": sum(r["pairwise_denominator"] for r in subset),
            "pairwise_accuracy": sum(r["pairwise_correct"] for r in subset) / sum(r["pairwise_denominator"] for r in subset),
            "schema_valid": sum(r["json_schema_valid"] for r in subset),
            "sufficient_citation_items": sum(r["sufficient_citation_items"] for r in subset),
            "citation_sufficiency_rate": sum(r["sufficient_citation_items"] for r in subset) / (len(subset) * 6),
            "all_supported_judgments": sum(r["all_judgments_supported"] for r in subset),
            "anchor_hits": sum(r["first_sufficient_anchor_hits"] for r in subset),
            "anchor_opportunities": sum(r["item_denominator"] for r in subset),
            "anchor_hit_rate": sum(r["first_sufficient_anchor_hits"] for r in subset) / sum(r["item_denominator"] for r in subset),
        }
    report = {
        "schema_version": "source_citation_control_unblinded_mechanical_report_v1",
        "status": "MECHANICAL_RESULTS_UNBLINDED_AFTER_BLIND_SCORE_COMMIT",
        "scope": "Exploratory descriptive results for three previously exposed Train questions only.",
        "overall": overall,
        "overall_deltas_C1_minus_C0": {
            "exact_accuracy": overall["C1"]["exact_accuracy"] - overall["C0"]["exact_accuracy"],
            "pairwise_accuracy": overall["C1"]["pairwise_accuracy"] - overall["C0"]["pairwise_accuracy"],
            "citation_sufficiency_rate": overall["C1"]["citation_sufficiency_rate"] - overall["C0"]["citation_sufficiency_rate"],
            "first_sufficient_anchor_hit_rate": overall["C1"]["anchor_hit_rate"] - overall["C0"]["anchor_hit_rate"],
        },
        "by_root": root_rows,
        "matched_pairs": sorted(pair_results, key=lambda x: x["pair_id"]),
        "individual_responses": sorted(rows, key=lambda x: x["request_id"]),
        "interpretation_limits": [
            "Only three roots/questions were sampled; do not infer unseen-item generalization or statistical significance.",
            "C1 is a bundled evidence/context package; the 147997/A supplemental context contains a cue relevant to the ordering relation.",
            "Mechanical first-anchor hit rate is a strict citation-reliability proxy; other semantically sufficient citations may not count as hits.",
            "This comparison does not establish a causal benefit from feedback training or self-play.",
        ],
    }
    with out.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": report["status"], "overall": overall,
        "deltas": report["overall_deltas_C1_minus_C0"], "by_root": root_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
