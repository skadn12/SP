"""Freeze the paired pilot before any model request."""

from __future__ import annotations

import importlib
from importlib import metadata
import json
from pathlib import Path
import random
import sys

from common import (BACKUP_ROOT, HERE, MODEL, OUTPUT_LIMIT, ROOTS, SEED, SOURCES,
                    SOURCE_VALIDATOR, TEMPERATURE, load_surface, sha_file, sha_bytes, write_once)


REPO = HERE.parent.parent
RUN = HERE / "run"
CODE_FILES = ("common.py", "a_arm.py", "b_arm.py", "prepare.py", "run.py",
              "review_gate.py", "m0_runner.py")
M0_SCRIPT = HERE.parent / "h4_graph_guided_qa_pilot_v1" / "solver_probe.py"
M0_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
M0_CODE = (
    "__init__.py", "backends.py", "core.py", "prompts.py",
    "structured_qa_box_protocol.py", "vllm_backends.py",
    "h4_ordering_display_v2/__init__.py", "h4_ordering_display_v2/contract.py",
    "h4_round0_b0_k3_difficulty_v1/__init__.py",
    "h4_round0_b0_k3_difficulty_v1/contract.py",
    "h4_round0_b0_k3_difficulty_v1/solver.py",
    "h4_round0_b0_k3_difficulty_v1/run.py",
    "h4_round0_b0_k3_difficulty_v1/gate.py",
    "h4_round0_b0_k3_difficulty_v1/difficulty.py",
    "h4_round0_b0_k3_difficulty_v1/inventory.py",
    "h4_round0_b0_k3_difficulty_v1/io_utils.py",
    "h4_round0_b0_k3_difficulty_v1/request_ledger.py",
)
M0_CODE_BASE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1/scripts/current_method/restart_n5_spell")
TRANSPORT_CODE = (
    "h4_memory_light_gate_siliconflow_v1/__init__.py",
    "h4_memory_light_gate_siliconflow_v1/run.py",
    "h4_writer_model_comparison_v1/transport.py",
)
M0_MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json",
                  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                  "model-00003-of-00003.safetensors")


def m0_identity() -> dict[str, object]:
    return {
        "model_path": str(M0_MODEL.resolve()),
        "model_files_sha256": {name: sha_file(M0_MODEL / name) for name in M0_MODEL_FILES},
        "code_files_sha256": {name: sha_file(M0_CODE_BASE / name) for name in M0_CODE},
        "package_versions": {name: metadata.version(name) for name in ("transformers", "vllm", "torch")},
    }


def transport_identity() -> dict[str, str]:
    return {name: sha_file(M0_CODE_BASE / name) for name in TRANSPORT_CODE}
REVIEW_FORM = {
    "version": "paired_trainable_qa_review_v1",
    "raw_relation_decisions": ["supported", "contradicted", "ambiguous"],
    "candidate_checks": ["shape", "source_supported", "two_independent_turns", "two_distinct_relations",
                         "same_actor_object_version_checkpoint", "direct_false_contradiction",
                         "visible_occurrence_order", "nontrivial", "no_answer_cue", "deduplicated"],
    "candidate_decisions": ["pass_to_m0", "reject"],
    "rejection_reasons": ["shape", "unsupported_true", "unsupported_false", "status_upgrade",
                          "actor_object_swap", "checkpoint_ambiguous", "one_turn_lookup",
                          "redundant_clause", "answer_cue", "chronology_unproved", "duplicate", "other"],
    "training_promotion": False,
}


QUESTION_ARTIFACT_NAMES = frozenset({
    "candidates.json", "candidates_valid.json", "candidates_valid_scoped.json",
    "candidates_valid_unscoped.json", "candidate_pool.json", "candidate_pool_scoped.json",
    "candidate_pool_unscoped.json", "candidate_generation.json", "hard_candidates.json",
    "questions.json", "accepted_question.json", "historical_v3_questions.json",
    "candidates_quality_accepted.json", "training_ready_hard_candidates.json",
    "semantically_valid_hard_candidates.json", "rejected_or_uncertain_hard_candidates.json",
    "strict_content_pass_candidates.json", "corrected_candidates.json",
    "original_candidate.json", "corrected_candidate.json",
})


def _question_roots(value: object, root_ids: frozenset[str], inherited_root: str | None = None) -> set[str]:
    """Find root provenance in a cross-root question artifact's JSON fields."""
    if isinstance(value, list):
        found = set()
        for item in value:
            found.update(_question_roots(item, root_ids, inherited_root))
        return found
    if not isinstance(value, dict):
        return set()
    found = set()
    root_id = value.get("root_id")
    owner = root_id if isinstance(root_id, str) and root_id in root_ids else inherited_root
    candidate_id = value.get("candidate_id")
    if isinstance(value.get("question"), str) and value["question"].strip():
        if owner:
            found.add(owner)
        if isinstance(candidate_id, str):
            found.update(root for root in root_ids if candidate_id.startswith(root + ":"))
    for item in value.values():
        if isinstance(item, (dict, list)):
            found.update(_question_roots(item, root_ids, owner))
    return found


def question_bank() -> list[dict[str, object]]:
    root_ids = frozenset((*ROOTS, BACKUP_ROOT))
    rows = []
    for directory in (REPO / "offline-deliveries", REPO / "restart-n5-runs"):
        for path in directory.rglob("*.json"):
            if path.name not in QUESTION_ARTIFACT_NAMES:
                continue
            roots = root_ids.intersection(path.parts)
            if not roots:
                payload = json.loads(path.read_text(encoding="utf-8"))
                roots = _question_roots(payload, root_ids)
            if roots:
                rows.append({"path": str(path.resolve()), "sha256": sha_file(path),
                             "root_ids": sorted(roots)})
    return sorted(rows, key=lambda row: row["path"])


def arm_order(slot_index: int) -> list[str]:
    choices = ["A", "B"]
    random.Random(SEED).shuffle(choices)
    return choices if slot_index == 0 else list(reversed(choices))


def select_roots() -> list[dict]:
    """Preflight both primary slots before writing the immutable run directory."""
    selected = []
    backup_used = False
    for slot_index, primary_id in enumerate(ROOTS):
        try:
            view = load_surface(primary_id)
            root_id, substitution = primary_id, None
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            if backup_used:
                raise RuntimeError("both_primary_roots_invalid_one_backup_only") from error
            root_id = BACKUP_ROOT
            try:
                view = load_surface(root_id)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as backup_error:
                raise RuntimeError("backup_source_integrity_failed") from backup_error
            backup_used = True
            substitution = {"replaced_root_id": primary_id,
                            "reason_type": type(error).__name__, "reason": str(error)}
        selected.append({"slot_index": slot_index, "root_id": root_id,
                         "view": view, "substitution": substitution})
    return selected


def prepare() -> dict:
    if RUN.exists():
        raise RuntimeError("run_directory_exists_no_overwrite")
    selected = select_roots()
    a_arm = importlib.import_module("a_arm")
    b_arm = importlib.import_module("b_arm")
    code = {name: sha_file(HERE / name) for name in CODE_FILES}
    m0_hash = sha_file(M0_SCRIPT)
    bank = question_bank()
    roots = {}
    for selection in selected:
        root_id = selection["root_id"]
        spec = SOURCES[root_id]
        view = selection["view"]
        projected_path = RUN / "frozen" / root_id / "surface.json"
        write_once(projected_path, view)
        parent_count = spec["parent_turn_count"]
        write_once(RUN / "frozen" / root_id / "m0_parent.json", {"turns": view["turns"][:parent_count]})
        write_once(RUN / "frozen" / root_id / "m0_child.json", {"turns": view["turns"][parent_count:]})
        prompts = {}
        for question_type in ("multi_select", "ordering"):
            content = a_arm.prompt(view, question_type)
            path = RUN / "frozen" / root_id / f"a_{question_type}_prompt.txt"
            write_once(path, content, raw=True)
            prompts[f"A_{question_type}"] = {"path": str(path), "sha256": sha_bytes(content.encode("utf-8"))}
        content = b_arm.extraction_prompt(view)
        path = RUN / "frozen" / root_id / "b_extraction_prompt.txt"
        write_once(path, content, raw=True)
        prompts["B_extraction"] = {"path": str(path), "sha256": sha_bytes(content.encode("utf-8"))}
        roots[root_id] = {
            "source_path": str(spec["path"]), "source_sha256": spec["sha256"],
            "prior_path": str(spec["prior_path"]), "prior_sha256": spec["prior_sha256"],
            "promotion_validation_path": str(spec["path"].with_name("promotable_expanded_dialogue_validation.json")),
            "promotion_validation_sha256": sha_file(spec["path"].with_name("promotable_expanded_dialogue_validation.json")),
            "surface_path": str(projected_path), "surface_sha256": sha_file(projected_path),
            "turn_count": spec["turn_count"], "parent_turn_count": parent_count,
            "source_slot": selection["slot_index"], "substitution": selection["substitution"],
            "arm_order": arm_order(selection["slot_index"]), "prompts": prompts,
        }
    write_once(RUN / "frozen" / "question_bank.json", bank)
    write_once(RUN / "frozen" / "review_form.json", REVIEW_FORM)
    manifest = {
        "version": "h4_paired_trainable_qa_yield_pilot_v1",
        "status": "PREPARED_BEFORE_FIRST_MODEL_REQUEST", "seed": SEED,
        "model": MODEL, "decoding": {"temperature": TEMPERATURE, "max_tokens": OUTPUT_LIMIT,
                                     "response_format": "json_object", "enable_thinking": False, "retries": 0},
        "candidate_cap_per_arm_root": {"multi_select": 2, "ordering": 1},
        "source_validator_sha256": sha_file(SOURCE_VALIDATOR),
        "roots": roots, "code_sha256": code, "m0_scorer_sha256": m0_hash,
        "m0_identity": m0_identity(),
        "transport_code_sha256": transport_identity(),
        "question_bank_sha256": sha_file(RUN / "frozen" / "question_bank.json"),
        "review_form_sha256": sha_file(RUN / "frozen" / "review_form.json"),
        "protocol_sha256": sha_file(HERE / "PRE_REGISTERED_EVAL.md"),
        "m0": {"displays": 6, "modes": ["no_context", "full_context"],
               "seed": 20261004, "no_context_max_correct": 1,
               "full_context_min_correct": 2, "full_context_max_correct": 4,
               "minimum_gain": 2},
        "training_promoted": 0,
    }
    write_once(RUN / "manifest.json", manifest)
    return manifest


def verify() -> dict:
    manifest_path = RUN / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(sha_file(HERE / name) != expected for name, expected in manifest["code_sha256"].items()):
        raise RuntimeError("frozen_code_changed")
    if sha_file(SOURCE_VALIDATOR) != manifest["source_validator_sha256"]:
        raise RuntimeError("frozen_source_validator_changed")
    if sha_file(M0_SCRIPT) != manifest["m0_scorer_sha256"] or sha_file(HERE / "PRE_REGISTERED_EVAL.md") != manifest["protocol_sha256"]:
        raise RuntimeError("frozen_protocol_or_scorer_changed")
    if m0_identity() != manifest["m0_identity"]:
        raise RuntimeError("frozen_m0_model_or_dependency_changed")
    if transport_identity() != manifest["transport_code_sha256"]:
        raise RuntimeError("frozen_transport_code_changed")
    for name in ("question_bank", "review_form"):
        if sha_file(RUN / "frozen" / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise RuntimeError(f"frozen_{name}_changed")
    frozen_bank = json.loads((RUN / "frozen" / "question_bank.json").read_text(encoding="utf-8"))
    if question_bank() != frozen_bank:
        raise RuntimeError("historical_question_bank_inventory_changed")
    for row in frozen_bank:
        if sha_file(Path(row["path"])) != row["sha256"]:
            raise RuntimeError("historical_question_bank_changed")
    for root_id, row in manifest["roots"].items():
        spec = SOURCES[root_id]
        if (row["source_path"] != str(spec["path"]) or row["source_sha256"] != spec["sha256"]
                or row["prior_path"] != str(spec["prior_path"]) or row["prior_sha256"] != spec["prior_sha256"]
                or row["promotion_validation_path"] != str(spec["path"].with_name("promotable_expanded_dialogue_validation.json"))):
            raise RuntimeError("frozen_source_spec_changed")
        if sha_file(Path(row["promotion_validation_path"])) != row["promotion_validation_sha256"]:
            raise RuntimeError("frozen_promotion_validation_changed")
        view = load_surface(root_id)
        if sha_file(Path(row["surface_path"])) != row["surface_sha256"]:
            raise RuntimeError("frozen_surface_changed")
        if view != json.loads(Path(row["surface_path"]).read_text(encoding="utf-8")):
            raise RuntimeError("source_projection_changed")
        for prompt in row["prompts"].values():
            if sha_file(Path(prompt["path"])) != prompt["sha256"]:
                raise RuntimeError("frozen_prompt_changed")
    return manifest


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
        result = {"status": "PREPARED", "manifest": str(RUN / "manifest.json")}
    elif sys.argv[1:] == ["verify"]:
        result = {"status": "VERIFIED", "root_count": len(verify()["roots"])}
    else:
        raise SystemExit("usage: python prepare.py prepare|verify")
    print(json.dumps(result, ensure_ascii=False))
