"""Program-owned scene-local dialogue references for Realizer v3."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping


SCENE_LOCAL_REF_VERSION = "scene_local_dialogue_refs_v1"
_RAW_TURN_KEY = re.compile(r"^t[0-9]{3}$")


def dialogue_sha256(dialogue_turns: Any) -> str:
    payload = json.dumps(dialogue_turns, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_full_realizer_output(
    output: Any, *, scene_ref: str,
) -> dict[str, Any]:
    """Map model-local turn keys to immutable program-owned scene refs."""
    errors: list[str] = []
    if not isinstance(output, Mapping) or set(output) != {"dialogue_turns", "proposal_claims"}:
        return {"valid": False, "errors": ["realizer_full_top_level_schema"]}
    raw_turns = output.get("dialogue_turns")
    raw_claims = output.get("proposal_claims")
    if not isinstance(raw_turns, list) or not raw_turns:
        errors.append("realizer_dialogue_turns_missing")
        raw_turns = []
    if not isinstance(raw_claims, list):
        errors.append("realizer_proposal_claims_not_list")
        raw_claims = []
    key_map: dict[str, str] = {}
    normalized_turns: list[dict[str, str]] = []
    for index, turn in enumerate(raw_turns, 1):
        if not isinstance(turn, Mapping) or set(turn) != {"turn_key", "speaker_ref", "text"}:
            errors.append(f"realizer_raw_turn_schema:{index}")
            continue
        key = str(turn.get("turn_key") or "")
        if not _RAW_TURN_KEY.fullmatch(key):
            errors.append(f"realizer_raw_turn_key_format:{index}")
        if key in key_map:
            errors.append(f"realizer_raw_turn_key_duplicate:{key}")
        ref = f"{scene_ref}:T{index:03d}"
        key_map[key] = ref
        normalized_turns.append({"turn_ref": ref,
                                 "speaker_ref": str(turn.get("speaker_ref") or ""),
                                 "text": str(turn.get("text") or "")})
    normalized_claims: list[dict[str, Any]] = []
    for index, claim in enumerate(raw_claims):
        if (not isinstance(claim, Mapping) or
                set(claim) != {"proposal_ref", "realization_status",
                               "candidate_evidence_turn_keys"}):
            errors.append(f"realizer_raw_claim_schema:{index}")
            continue
        raw_keys = claim.get("candidate_evidence_turn_keys")
        if not isinstance(raw_keys, list) or not all(isinstance(key, str) for key in raw_keys):
            errors.append(f"realizer_raw_claim_evidence_schema:{index}")
            raw_keys = []
        missing = [key for key in raw_keys if key not in key_map]
        if missing:
            errors.extend(f"realizer_raw_claim_unknown_turn_key:{key}" for key in missing)
        normalized_claims.append({
            "proposal_ref": str(claim.get("proposal_ref") or ""),
            "realization_status": str(claim.get("realization_status") or ""),
            "candidate_evidence_turn_refs": [key_map[key] for key in raw_keys if key in key_map],
        })
    normalized = {"dialogue_turns": normalized_turns,
                  "proposal_claims": normalized_claims}
    return {
        "valid": not errors, "errors": errors,
        "scene_local_ref_version": SCENE_LOCAL_REF_VERSION,
        "turn_key_to_scene_ref": key_map,
        "normalized_output": normalized,
        "dialogue_sha256": dialogue_sha256(normalized_turns),
    }


def reconstruct_claim_only_output(
    *, frozen_dialogue_turns: list[Mapping[str, Any]], claim_only_output: Any,
) -> dict[str, Any]:
    if not isinstance(claim_only_output, Mapping) or set(claim_only_output) != {"proposal_claims"}:
        raise ValueError("realizer_claim_only_top_level_schema")
    if not isinstance(claim_only_output["proposal_claims"], list):
        raise ValueError("realizer_claim_only_claims_not_list")
    before = dialogue_sha256(frozen_dialogue_turns)
    reconstructed = {
        "dialogue_turns": deepcopy(list(frozen_dialogue_turns)),
        "proposal_claims": deepcopy(list(claim_only_output["proposal_claims"])),
    }
    after = dialogue_sha256(reconstructed["dialogue_turns"])
    if before != after:
        raise RuntimeError("claim_only_retry_mutated_dialogue")
    return {"normalized_output": reconstructed, "dialogue_sha256_before": before,
            "dialogue_sha256_after": after, "dialogue_sha_unchanged": True}
