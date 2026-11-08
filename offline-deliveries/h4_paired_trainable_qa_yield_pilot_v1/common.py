"""Shared frozen inputs and auditable SiliconFlow transport for the paired pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
MODEL = "Qwen/Qwen3.6-35B-A3B"
TEMPERATURE = 0
OUTPUT_LIMIT = 6144
SEED = 20261004
ROOTS = {
    "cc-train-episode-136202": {
        "path": Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_generation_corrected_live_1/cc-train-episode-136202/attempt-01/promotion/promotable_expanded_dialogue.json"),
        "sha256": "61a0127ac31e3837d3475e566c062a1b95e221be66f113b37b978a33ce4b7627",
        "turn_count": 149,
        "parent_turn_count": 57,
        "child_turn_count": 92,
        "round_count": 2,
        "prior_path": Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_full_pipeline_newroots_v1_inputs/cc-train-episode-136202/input_R1.json"),
        "prior_sha256": "dd553f6d2e95eaedba4fd5853855b9c4e49ff3fab32ea3aa53a1efbb4b48fe00",
    },
    "cc-train-episode-122711": {
        "path": Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_generation_corrected_live_1/cc-train-episode-122711/attempt-02/promotion/promotable_expanded_dialogue.json"),
        "sha256": "776553c9887068ee9d969fd7fe4dcfafef0f2cc5df66cd387399fbe4540277ec",
        "turn_count": 133,
        "parent_turn_count": 59,
        "child_turn_count": 74,
        "round_count": 2,
        "prior_path": Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_development_gain_v2_prepare_1/cc-train-episode-122711/input_R1.json"),
        "prior_sha256": "af6c9714ea6966be2dc6f610078b3eafc5787354a6b159f598659c3e2e90436d",
    },
}
BACKUP_ROOT = "cc-train-episode-197635"
SOURCES = {
    **ROOTS,
    BACKUP_ROOT: {
        "path": Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_generation_corrected_live_1/cc-train-episode-197635/attempt-03/promotion/promotable_expanded_dialogue.json"),
        "sha256": "7b291fcf058d81520af49077c292e990b2e91af32e2356509928b6820e922e74",
        "turn_count": 148,
        "parent_turn_count": 54,
        "child_turn_count": 94,
        "round_count": 2,
        "prior_path": Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_full_pipeline_newroots_v1_inputs/cc-train-episode-197635/input_R1.json"),
        "prior_sha256": "ffc2314cc98e4bae49a25d37816d6c0167bdca8dd5044397674ea41475d0af9c",
    },
}
SOURCE_VALIDATOR = WORKTREE / "scripts/current_method/restart_n5_spell/simple_append_v1/assembly.py"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_once(path: Path, value: Any, *, raw: bool = False) -> None:
    data = value.encode("utf-8") if raw else json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def load_surface(root_id: str) -> dict[str, Any]:
    spec = SOURCES[root_id]
    path = spec["path"]
    actual = sha_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"source_hash_mismatch:{root_id}")
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or any(source.get(key) != spec[key] for key in ("turn_count", "parent_turn_count", "child_turn_count", "round_count")):
        raise ValueError(f"source_counts_mismatch:{root_id}")
    if source.get("root_id") != root_id or not isinstance(source.get("turns"), list):
        raise ValueError(f"source_shape_invalid:{root_id}")
    source_turns = source["turns"]
    if len(source_turns) != spec["turn_count"]:
        raise ValueError(f"turn_count_mismatch:{root_id}")
    if any(not isinstance(row, dict) for row in source_turns):
        raise ValueError(f"turn_not_object:{root_id}")
    prior_path = spec["prior_path"]
    if sha_file(prior_path) != spec["prior_sha256"]:
        raise RuntimeError(f"prior_hash_mismatch:{root_id}")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("root_id") != root_id:
        raise ValueError(f"prior_root_mismatch:{root_id}")
    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))
    from scripts.current_method.restart_n5_spell.simple_append_v1.assembly import validate_appended_round

    validation = validate_appended_round(source, prior_expanded=prior)
    recorded_path = path.with_name("promotable_expanded_dialogue_validation.json")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    if not validation["valid"] or validation != recorded:
        raise ValueError(f"promoted_source_integrity_failed:{root_id}:{validation['errors']}")
    parent_count = spec["parent_turn_count"]
    if any(not row.get("turn_ref", "").startswith(f"{root_id}:P") for row in source_turns[:parent_count]):
        raise ValueError(f"parent_boundary_invalid:{root_id}")
    if any(not row.get("turn_ref", "").startswith(("R01:", "R02:")) for row in source_turns[parent_count:]):
        raise ValueError(f"child_boundary_invalid:{root_id}")
    projected = []
    seen = set()
    for row in source_turns:
        if not isinstance(row, dict):
            raise ValueError("turn_not_object")
        turn = {key: row.get(key) for key in ("turn_ref", "speaker", "text")}
        if any(not isinstance(value, str) or not value.strip() for value in turn.values()):
            raise ValueError("turn_surface_invalid")
        if turn["turn_ref"] in seen:
            raise ValueError("duplicate_turn_ref")
        seen.add(turn["turn_ref"])
        projected.append(turn)
    boundaries = source.get("round_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != spec["round_count"]:
        raise ValueError("round_boundaries_invalid")
    for boundary in boundaries:
        first, last = boundary["first_turn_index_1_based"] - 1, boundary["last_turn_index_1_based"] - 1
        if projected[first]["turn_ref"] != boundary["first_turn_ref"] or projected[last]["turn_ref"] != boundary["last_turn_ref"]:
            raise ValueError("round_boundary_ref_mismatch")
    return {"root_id": root_id, "source_sha256": actual, "turns": projected}


def call_qwen(run_dir: Path, prompt: str, *, frozen_prompt_sha256: str) -> dict[str, Any]:
    """One attempt; reserve and write raw output before any parsing."""
    if sha_bytes(prompt.encode("utf-8")) != frozen_prompt_sha256:
        raise RuntimeError("prompt_differs_from_freeze")
    if any(run_dir.glob("*")):
        raise RuntimeError(f"stage_already_has_artifacts:{run_dir}")
    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))
    from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (
        credential_client, provider_profile,
    )

    profile = provider_profile()
    if profile["model"] != MODEL:
        raise RuntimeError("unexpected_model")
    profile["profile_id"] = "h4_paired_trainable_qa_yield_pilot_v1"
    profile["request_parameters"].update(temperature=TEMPERATURE, max_tokens=OUTPUT_LIMIT, enable_thinking=False)
    profile["output_limit"] = OUTPUT_LIMIT
    profile["http_retry_limit"] = 0
    write_once(run_dir / "attempt.reserved.json", {
        "status": "RESERVED_ONE_ATTEMPT", "prompt_sha256": frozen_prompt_sha256,
        "profile": profile, "maximum_http_attempts": 1, "training_promotion": False,
    })
    write_once(run_dir / "prompt.txt", prompt, raw=True)

    def on_attempt(payload: dict[str, Any]) -> None:
        write_once(run_dir / "http_attempt.json", {
            "attempt_number": 1, "payload_sha256": sha_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")),
            "payload": payload,
            "model": payload.get("model"), "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"), "enable_thinking": payload.get("enable_thinking"),
        })

    try:
        response = credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True,
        )
        write_once(run_dir / "response_raw.txt", response.raw_text, raw=True)
        metadata = {"status": "RESPONSE_RECEIVED", "raw_sha256": sha_bytes(response.raw_text.encode("utf-8")),
                    "metadata": response.metadata, "training_promotion": False}
        write_once(run_dir / "response_meta.json", metadata)
        return {"raw": response.raw_text, "metadata": metadata}
    except Exception as error:
        failure = {"status": "FAILED_NO_RETRY", "error_type": type(error).__name__,
                   "training_promotion": False}
        metadata = getattr(error, "metadata", None)
        if isinstance(metadata, dict):
            # The transport has already redacted credentials from this metadata.
            failure["transport_metadata"] = metadata
        transport_status = getattr(error, "status", None)
        if isinstance(transport_status, str):
            failure["transport_status"] = transport_status
        write_once(run_dir / "failure.json", failure)
        raise
