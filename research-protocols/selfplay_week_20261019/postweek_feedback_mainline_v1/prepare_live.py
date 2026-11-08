"""Feed the real reviewed reasons and reviewed material gap to the main entry."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from scripts.current_method import reason_material_feedback_pipeline as pipeline


def main():
    reason_packet = WEEK / "postweek_reason_feedback_repair_v1/challenger_live_v1/INPUT_PACKET.json"
    ledger = WEEK / "day3_diversity_pilot_v3/frozen_inputs/ADMISSION_LEDGER_FROZEN.json"
    roots = []
    for suffix in ("136202", "147997", "156136"):
        root_id = "cc-train-episode-" + suffix
        roots.append({
            "root_id": root_id,
            "source_binding": pipeline.binding(WEEK / "day3_production_v1/sources" / (root_id + ".json")),
            "task_operations": [{"operation_id": "ORDER_FIRST_APPEARANCE", "question_type": "ordering",
                "criterion": "First semantically sufficient appearance of the exact event state in the complete dialogue."}],
            "material_targets": []})
    root_id = "cc-train-episode-104814"
    roots.append({
        "root_id": root_id,
        "source_binding": pipeline.binding(WEEK / "day3_production_v1/sources" / (root_id + ".json")),
        "task_operations": [{"operation_id": "ORDER_EXPLICIT_TIME", "question_type": "ordering",
            "criterion": "Unique event-time order within an explicit shared date/time frame, distinct from dialogue mention order."}],
        "material_targets": [{"slot_id": "DIVV3-R2-ORD2", "material_goal":
            "Provide a natural shared calendar or activity sequence with explicit common dates and distinct confirmed event times. Distinguish proposals, requests, confirmation and completion. Do not retroactively equate the two historical Tuesdays or treat a Friday leave request as an approved event. An already reviewed append-only descendant may be reused if independent review verifies these relationships."}]})
    config = {"schema_version": pipeline.VERSION + "_config", "batch_id": "postweek_feedback_mainline_v1",
        "namespace": "RMF1", "scope": "train_only_exploratory",
        "reason_packet": pipeline.binding(reason_packet), "shortfall_ledger": pipeline.binding(ledger),
        "roots": roots, "source_policy": "Prefer an audited existing descendant before authoring a new extension; never claim historical text was generated from current feedback.",
        "no_model_calls_by_preparation": True, "formal_weakness_admission": False}
    config_path = HERE / "CONFIG.json"
    pipeline.write(config_path, config)
    result = pipeline.prepare(config_path, HERE / "live_run_v2")
    print({"roots": len(result["roots"]), "routes": {row["root_id"]: row["mode"] for row in result["roots"]},
           "llm_consumption_proven": result["llm_consumption_proven"]})


if __name__ == "__main__":
    main()
