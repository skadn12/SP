"""Create a reviewable source manifest from the final handoff, without compiling."""

import argparse
from pathlib import Path

from . import VERSION
from .sources import file_sha, read
from ..h4_round1_question_bank_feedback_compiler_v1.compiler import write_artifact


def create_manifest(handoff, output):
    handoff, output = Path(handoff).resolve(), Path(output).resolve()
    inventory = read(handoff / "FINAL_FILE_INVENTORY.json")
    inputs = {}

    def add(name, suffix):
        matches = {row["path"]: row for row in inventory if row["path"].endswith(suffix)}
        if len(matches) != 1:
            raise ValueError(f"handoff_source_not_unique:{name}:{suffix}")
        row = next(iter(matches.values()))
        if file_sha(row["path"]) != row["sha256"]:
            raise ValueError(f"handoff_source_drift:{name}")
        inputs[name] = {"path": row["path"], "sha256": row["sha256"]}

    add("round0_inventory", "h4_round0_b0_k3_difficulty_v1_live_1/source_inventory.json")
    add("round0_split", "h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl")
    add("corpus_scope", "h4_round1_question_bank_bootstrap_v1_live_1/corpus_scope_manifest.jsonl")
    add("bank", "h4_round1_qa_feedback_multilabel_rerouting_v1_live_9/provisional_question_bank_entries.jsonl")
    freeze = "h4_round1_question_bank_lifecycle_weight_freeze_v1_live_1/"
    for name, filename in {
        "scope": "active_train_scope_manifest.jsonl", "clusters": "cluster_lifecycle_manifest.jsonl",
        "members": "candidate_to_cluster_map.jsonl", "origins": "cluster_observation_origin_manifest.jsonl",
        "eligibility": "operation_eligibility_manifest.jsonl", "cooldown": "cooldown_state_initialization.jsonl",
        "saturation": "normalized_stem_saturation_ledger.jsonl",
        "target_weights": "questioner_targeted_sampling_weights.jsonl",
        "composition_weights": "questioner_composition_sampling_weights.jsonl",
        "weight_contract": "weighting_contract.json",
    }.items():
        # The final inventory omits cooldown, but the frozen compiler explicitly binds it.
        if name == "cooldown":
            directory = Path(inputs["scope"]["path"]).parent
            path = directory / filename
            expected = path.with_name(path.name + ".sha256").read_text().split()[0]
            if file_sha(path) != expected:
                raise ValueError("cooldown_sha_mismatch")
            inputs[name] = {"path": str(path), "sha256": expected,
                            "authority": "HANDOFF compiler FREEZE_FILES + frozen SHA sidecar"}
        else:
            add(name, freeze + filename)
    add("dialogue_identity", "h4_round1_r03_memory_readonly_trial_preflight_v2_live_3/r03_input_dialogue_identity_manifest.jsonl")
    add("live13_cards", "h4_round1_question_bank_feedback_compiler_v1_live_13/r03_expansion_feedback_cards_full.jsonl")
    add("live13_fresh", "h4_round1_question_bank_feedback_compiler_v1_live_13/fresh_exploration_negative_signature_view.jsonl")
    add("dev_global", "h4_round1_feedback_compiler_v1_live_1/dev_global_feedback.json")
    add("round0_features", "h4_round1_feedback_compiler_v1_live_1/candidate_feedback_features.jsonl")
    add("round0_routes", "h4_round1_feedback_compiler_v1_live_1/feedback_routing_manifest.jsonl")
    add("round0_transitions", "h4_round0_trained_solver_train_reeval_v1_live_1/candidate_transition_matrix.jsonl")
    for filename in ("FINAL_DO_NOT_OVERWRITE.json", "FINAL_FILE_INVENTORY.json", "counts.json",
                     "three_question_type_feedback_examples_COMPLETE.json"):
        path = handoff / filename
        inputs[filename] = {"path": str(path), "sha256": file_sha(path)}
    # Exact schema sources are named by HANDOFF's shared renderer package.
    runs = handoff.parent
    schema_run = runs / "h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight_live_7"
    for mode in ("targeted", "composition", "fresh"):
        path = schema_run / f"qa_output_schema_v1_1_revision_1_{mode}.json"
        sha = path.with_name(path.name + ".sha256").read_text().split()[0]
        if file_sha(path) != sha:
            raise ValueError("schema_sha_mismatch")
        inputs[f"schema_{mode}"] = {"path": str(path), "sha256": sha}
    path = runs / "h4_round1_r03_qa_feedback_template_v1_preflight_v1_live_4/qa_evolution_strategy_schema_v1.json"
    inputs["strategy_schema"] = {"path": str(path), "sha256": file_sha(path)}
    manifest = {"filter_version": VERSION, "source_chain": "FINAL_HANDOFF_EXPLICIT_SNAPSHOTS",
                "inputs": inputs, "dev_global_policy": "PRESERVE_HISTORICAL_AGGREGATE_AUDIT_ONLY",
                "automatic_default_switch": False, "real_calls_authorized": False,
                "current_execution_identity": "EXTRA_VAL_122711_ALIGNED_PROMPTS_V1_01"}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(output.parent, output.name, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_manifest(args.handoff, args.output)


if __name__ == "__main__":
    main()
