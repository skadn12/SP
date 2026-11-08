"""Freeze the finite supplementary evaluation attempt before question authoring."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WEEK = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
REPO = WEEK.parents[1]


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    target = OUT / "EXECUTION_FREEZE.json"
    if target.exists():
        raise SystemExit("Freeze already exists; do not overwrite it.")
    roots = []
    for short, scenes, expected in [
        ("101421", ["R01:S03"], "eba44f35d19f73945056e0f4474a3c5e2a667c15f7f84004a0753d528669bdba"),
        ("35167", ["R01:S01", "R01:S03"], "3f69c0ecbd75f7710e5badc67081e957d995992371aa4be4e4dbc152551ce5c3"),
    ]:
        source = REPO / "offline-deliveries/h4_codex_dev_test_ab_v124/units" / ("root" + short) / "inputs/BASELINE.json"
        binding = bind(source)
        assert binding["sha256"] == expected
        roots.append({"root_id": "cc-train-episode-" + short, "original_split": "test", "allowed_target_scenes": scenes, "source": binding,
                      "exposure": "HISTORICALLY_EXPOSED_HELDOUT_ROOT_NOT_PRISTINE", "public_context": "entire_exact_source_including_other_scenes"})
    slots = []
    for root in roots:
        for index, (kind, design, size) in enumerate([
            ("multi_select", "ENTITY_OBJECT_BINDING", 2),
            ("multi_select", "QUALIFIED_RELATION_OR_PLAN_FINAL_STATE", 3),
            ("multi_select", "CROSS_TURN_COMPOSITE_SUPPORT", 4),
            ("ordering", "FIRST_SEMANTIC_REPORT_FOUR", 4),
            ("ordering", "FIRST_SEMANTIC_REPORT_FIVE", 5),
            ("ordering", "FIRST_SEMANTIC_REPORT_SIX", 6),
        ], 1):
            slots.append({"candidate_id": "d3s2-" + root["root_id"].split("-")[-1] + "-" + str(index).zfill(2),
                          "root_id": root["root_id"], "question_type": kind, "design": design,
                          "gold_count" if kind == "multi_select" else "item_count": size,
                          "global_slot_index": 186 + len(slots), "status": "PLANNED_NOT_AUTHORED"})
    obj = {
        "schema_version": "day3_finite_new_family_supplement_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_QUESTION_AUTHORING",
        "reason": "The existing prospective supplement contains four MC questions and no ordering question. Complete a bounded source-first multitype preparation attempt without Solver-based selection.",
        "upstream": [bind(WEEK / p) for p in ["day3_mainline_readiness_v1/READINESS_DECISION.json", "day3_production_v1/DAY3_SCOPE_AMENDMENT.json", "day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json", "day1/evaluation/ROOT_BOUNDARY.json", "day4_common_set_admission_v1/DECISION.json"]],
        "roots": roots, "slots": slots,
        "authoring": {"initial_drafts_per_slot": 1, "maximum_wording_clarification_and_rereview": 1,
                      "new_dialogue_turns": 0, "new_roots": 0, "solver_calls": 0, "training_runs": 0,
                      "source_first": True, "use_failure_feedback_or_scores": False,
                      "backend": "interactive Codex source-based authoring; actual provider token cap not enforceable",
                      "MC": "six options; all truths and falsehoods source-verifiable; no unsupported-not-mentioned distractors",
                      "ordering": "first semantic report in complete provided source; all earlier turns searched; unique first-turn anchors; report order is not real-world chronological order",
                      "scope": "Only allowed target scenes supply answer facts, but audit and public input always include the complete source. Do not silently reuse R02 facts or narrow the first-mention search."},
        "admission": {"content": "Independent reviewer reconstructs every proposition and its polarity or every ordering event and first semantic anchor from full source; audit all failed drafts too.",
                      "dedup": "Exclude only whole-task semantic equivalence; partial overlap retained and linked. This user rule does not permit training/evaluation answer leakage.",
                      "family": "Audit fixed current training/diagnosis and historical same-root task inventory before any new-family label. Whole-task historical equivalents may only be regression; unresolved required-fact leakage is HOLD.",
                      "score_independence": "No Solver, item outcomes or question difficulty scores used for admission.",
                      "minimum_for_scoped_preparation_GO": {"new_roots_with_both_types": 2, "MC_per_new_root": 2, "ordering_per_new_root": 2, "MC_relation_classes": 2, "ordering_arities": 2},
                      "interpretation": "Prospective finite feasibility floor, frozen before authoring; not a power calculation or proof of generalization. Existing four MC remain a separate sealed cohort.",
                      "stopping": "Attempt only these twelve slots. One wording clarification at most. Unsupported source, unresolved truth/order, equivalent old tasks or leakage remain SHORTFALL/HOLD/EXCLUDE. If the minimum fails, close with NO_GO; do not lower the floor, search more roots or fabricate missing material."},
        "requests": {"inherit_policy": "day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json", "sampling_and_display": "unchanged; new immutable slot indices 186..197", "freeze_actual_bytes_after_admission": True,
                     "tokenize_full_chat": True, "max_model_len": 16384, "reserved_answer_tokens": 32,
                     "execute": False, "future_checkpoint_and_runtime_bindings": "DAY4_OR_LATER_BEFORE_EXECUTION"},
        "completion": "Day3 content/source/family/request preparation may be GO only after independent content and technical audits plus scoped coverage pass. Day4 mixed pools/runtime and Day5 training remain separate. Original sealed decisions are preserved.",
        "claims_forbidden": ["unseen_or_pristine_root_generalization", "feedback_causal_benefit", "memory_benefit", "training_benefit", "statistically_reliable_multitype_effect_from_small_supplement"],
    }
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(bind(target)))


if __name__ == "__main__":
    main()
