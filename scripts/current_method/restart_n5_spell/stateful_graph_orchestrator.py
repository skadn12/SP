"""Crash-safe scene executor for the stateful Graph v2 protocol."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backends import RealizerBackend, RealizerRequest
from .core import atomic_json, sha256_file
from .ledger import RequestLedger
from .stateful_graph_expansion import (
    build_stateful_graph_realizer_prompt, evidence_phrase_present,
    initialize_stateful_ledger, stateful_completion_report,
    terminal_evidence_phrase_present,
    validate_and_apply_stateful_scene,
)


STATEFUL_REALIZER_DIRECTED_RETRY_VERSION = (
    "stateful_realizer_stage_specific_retry_v2")


_STAGE_ERROR_CODES = {
    "assigned_stage_update_set_mismatch",
    "carry_forward_must_not_claim_evidence",
    "nonterminal_stage_cannot_change_state",
    "stage_evidence_speaker_mismatch",
    "stage_evidence_turn_invalid",
    "stage_result_not_allowed",
    "stage_result_speaker_unauthorized",
    "stage_result_state_mismatch",
    "terminal_evidence_is_future_or_conditional",
    "terminal_evidence_marker_missing",
}

_METADATA_ONLY_HANDOFF_ERROR_CODES = {"unknown_handoff_id"}


def stateful_realizer_stage_diagnostics(
    *, contract: Mapping[str, Any], parsed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compute precise per-stage binding diagnostics without changing output."""
    turns = parsed.get("dialogue_turns") or []
    updates = {
        str(row.get("stage_id")): row
        for row in parsed.get("candidate_stage_updates") or []
        if isinstance(row, Mapping)
    }
    diagnostics = []
    for stage in contract.get("assigned_event_stages") or []:
        stage_id = str(stage["stage_id"])
        update = updates.get(stage_id)
        if update is None:
            diagnostics.append({
                "stage_id": stage_id, "valid": False,
                "stage_errors": ["stage_update_missing"],
                "reported_evidence_turn_index": None,
                "reported_result_speaker": None,
                "actual_indexed_speaker": None,
                "required_speakers": list(
                    stage["authorized_result_speakers"]),
                "required_terminal_markers": list(
                    stage["completion_evidence_phrases"]),
                "marker_turn_candidates": [],
                "recommended_evidence_turn_indices": [],
                "metadata_only_repair_available": False,
                "candidate_stage_update": None,
            })
            continue
        status = update.get("status")
        evidence_index = update.get("evidence_turn_index")
        evidence = None
        if (isinstance(evidence_index, int) and
                0 <= evidence_index < len(turns) and
                isinstance(turns[evidence_index], Mapping)):
            evidence = turns[evidence_index]
        required_speakers = list(stage["authorized_result_speakers"])
        phrases = list(stage["completion_evidence_phrases"])
        marker_candidates = []
        for index, turn in enumerate(turns):
            if (isinstance(turn, Mapping) and
                    terminal_evidence_phrase_present(
                        str(turn.get("text") or ""), phrases)):
                marker_candidates.append({
                    "turn_index": index,
                    "speaker": turn.get("speaker"),
                    "speaker_authorized":
                        turn.get("speaker") in required_speakers,
                })
        stage_errors: list[str] = []
        terminal = status in {"completed", "failed"}
        if status == "carry_forward":
            if (update.get("result_state") is not None or
                    evidence_index is not None or
                    update.get("result_speaker") is not None):
                stage_errors.append("carry_forward_must_not_claim_evidence")
        else:
            if evidence is None:
                stage_errors.append("stage_evidence_turn_invalid")
            else:
                if update.get("result_speaker") != evidence.get("speaker"):
                    stage_errors.append("stage_evidence_speaker_mismatch")
            if update.get("result_speaker") not in required_speakers:
                stage_errors.append("stage_result_speaker_unauthorized")
        if terminal:
            if update.get("result_state") != stage["target_post_state"]:
                stage_errors.append("stage_result_state_mismatch")
            if evidence is not None:
                evidence_text = str(evidence.get("text") or "")
                if not evidence_phrase_present(evidence_text, phrases):
                    stage_errors.append("terminal_evidence_marker_missing")
                elif not terminal_evidence_phrase_present(
                        evidence_text, phrases):
                    stage_errors.append(
                        "terminal_evidence_is_future_or_conditional")
            if update.get("result_state") not in stage["allowed_post_states"]:
                stage_errors.append("stage_result_not_allowed")
        elif update.get("result_state") is not None:
            stage_errors.append("nonterminal_stage_cannot_change_state")
        recommended = [
            row["turn_index"] for row in marker_candidates
            if row["speaker_authorized"]
        ]
        metadata_binding_errors = {
            "stage_evidence_speaker_mismatch",
            "stage_evidence_turn_invalid",
            "stage_result_speaker_unauthorized",
            "terminal_evidence_marker_missing",
        }
        diagnostics.append({
            "stage_id": stage_id,
            "valid": not stage_errors,
            "stage_errors": sorted(set(stage_errors)),
            "status": status,
            "reported_evidence_turn_index": evidence_index,
            "reported_result_speaker": update.get("result_speaker"),
            "actual_indexed_speaker": evidence.get("speaker")
                if evidence is not None else None,
            "required_speakers": required_speakers,
            "required_terminal_markers": phrases,
            "marker_turn_candidates": marker_candidates,
            "recommended_evidence_turn_indices": recommended,
            "metadata_only_repair_available": bool(
                stage_errors and recommended and
                set(stage_errors) <= metadata_binding_errors),
            "candidate_stage_update": dict(update),
        })
    return diagnostics


def stateful_realizer_handoff_diagnostics(
    *, parsed: Mapping[str, Any], allowed_handoff_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Report exact handoff whitelist membership without repairing output."""
    allowed = sorted(set(str(value) for value in allowed_handoff_ids))
    diagnostics = []
    for update in parsed.get("candidate_handoff_updates") or []:
        if not isinstance(update, Mapping):
            continue
        reported = update.get("handoff_id")
        exact_match = isinstance(reported, str) and reported in allowed
        diagnostics.append({
            "reported_handoff_id": reported,
            "allowed_handoff_ids": allowed,
            "valid": exact_match,
            "error": None if exact_match else "unknown_handoff_id",
            "required_action": (None if exact_match else
                "omit this handoff update; no unknown ID may be registered"),
        })
    return diagnostics


def stateful_dialogue_turns_sha256(
    turns: Sequence[Mapping[str, Any]],
) -> str:
    """Canonical SHA used only to attest verbatim metadata-only retries."""
    return _sha_text(json.dumps(list(turns), ensure_ascii=False,
                                sort_keys=True, separators=(",", ":")))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stateful_realizer_retry_error_feedback(
    *, contract: Mapping[str, Any], errors: Sequence[str],
    parsed: Mapping[str, Any], actual_tokens: int, actual_turns: int,
    allowed_handoff_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Expand every Realizer rejection into concrete, non-mutating feedback."""
    feedback: list[dict[str, Any]] = []
    error_set = set(errors)
    hard_tokens = contract["hard_dialogue_token_range"]
    hard_turns = contract["hard_turn_range"]
    if "stateful_scene_token_count_out_of_range" in error_set:
        feedback.append({
            "code": "stateful_scene_token_count_out_of_range",
            "actual_tokens": actual_tokens,
            "minimum_tokens": hard_tokens[0],
            "maximum_tokens": hard_tokens[1],
        })
    if "stateful_scene_turn_count_out_of_range" in error_set:
        feedback.append({
            "code": "stateful_scene_turn_count_out_of_range",
            "actual_turns": actual_turns,
            "minimum_turns": hard_turns[0],
            "maximum_turns": hard_turns[1],
        })
    diagnostics = stateful_realizer_stage_diagnostics(
        contract=contract, parsed=parsed)
    invalid_stages = [row for row in diagnostics if not row["valid"]]
    covered_stage_errors: set[str] = set()
    if error_set & _STAGE_ERROR_CODES:
        for diagnostic in invalid_stages:
            relevant_errors = sorted(
                set(diagnostic["stage_errors"]) & error_set)
            if ("assigned_stage_update_set_mismatch" in error_set and
                    "stage_update_missing" in diagnostic["stage_errors"]):
                relevant_errors = sorted(set(relevant_errors) | {
                    "assigned_stage_update_set_mismatch",
                    "stage_update_missing",
                })
            if not relevant_errors:
                continue
            covered_stage_errors.update(
                error for error in relevant_errors if error in error_set)
            feedback.append({
                "code": "stage_contract_violation",
                **diagnostic,
                "stage_errors": relevant_errors,
                # Backward-compatible diagnostic aliases used by reports/tests.
                "evidence_turn_index": diagnostic[
                    "reported_evidence_turn_index"],
                "actual_evidence_speaker": diagnostic[
                    "actual_indexed_speaker"],
                "actual_result_speaker": diagnostic[
                    "reported_result_speaker"],
            })
    covered = {row["code"] for row in feedback} | covered_stage_errors
    if "unknown_handoff_id" in error_set:
        for diagnostic in stateful_realizer_handoff_diagnostics(
                parsed=parsed, allowed_handoff_ids=allowed_handoff_ids):
            if diagnostic["valid"]:
                continue
            feedback.append({"code": "unknown_handoff_id", **diagnostic})
        if any(row["code"] == "unknown_handoff_id" for row in feedback):
            covered.add("unknown_handoff_id")
    for error in sorted(error_set - covered):
        feedback.append({"code": error})
    return feedback


def stateful_realizer_metadata_only_retry_available(
    *, contract: Mapping[str, Any], errors: Sequence[str],
    parsed: Mapping[str, Any], allowed_handoff_ids: Sequence[str] = (),
) -> bool:
    """Return whether only stage binding metadata needs correction."""
    error_set = set(errors)
    diagnostics = stateful_realizer_stage_diagnostics(
        contract=contract, parsed=parsed)
    invalid_stages = [row for row in diagnostics if not row["valid"]]
    invalid_handoffs = [
        row for row in stateful_realizer_handoff_diagnostics(
            parsed=parsed, allowed_handoff_ids=allowed_handoff_ids)
        if not row["valid"]]
    repairable_errors = _STAGE_ERROR_CODES | \
        _METADATA_ONLY_HANDOFF_ERROR_CODES
    return bool(
        (invalid_stages or invalid_handoffs) and
        not (error_set - repairable_errors) and
        all(row["metadata_only_repair_available"]
            for row in invalid_stages) and
        ("unknown_handoff_id" not in error_set or invalid_handoffs))


def stateful_realizer_dialogue_preserved(
    frozen_dialogue_turns: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, Any],
) -> bool:
    """Compare retry dialogue structurally and verbatim; never rewrite it."""
    return parsed.get("dialogue_turns") == list(frozen_dialogue_turns)


def build_stateful_realizer_retry_suffix(
    *, contract: Mapping[str, Any], errors: Sequence[str],
    parsed: Mapping[str, Any], actual_tokens: int, actual_turns: int,
    frozen_dialogue_turns: Sequence[Mapping[str, Any]] | None = None,
    allowed_handoff_ids: Sequence[str] = (),
) -> str:
    """Build one complete directed retry; never repair model output locally."""
    feedback = stateful_realizer_retry_error_feedback(
        contract=contract, errors=errors, parsed=parsed,
        actual_tokens=actual_tokens, actual_turns=actual_turns,
        allowed_handoff_ids=allowed_handoff_ids)
    diagnostics = stateful_realizer_stage_diagnostics(
        contract=contract, parsed=parsed)
    stage_error_present = bool(set(errors) & _STAGE_ERROR_CODES)
    invalid_stages = [row for row in diagnostics if not row["valid"]]
    if not stage_error_present:
        invalid_stages = []
    non_stage_errors = set(errors) - _STAGE_ERROR_CODES
    freeze_valid_stage_updates = stage_error_present and not non_stage_errors
    valid_stages = ([row for row in diagnostics if row["valid"]]
                    if freeze_valid_stage_updates else [])
    metadata_only_repair = (
        frozen_dialogue_turns is not None or
        stateful_realizer_metadata_only_retry_available(
            contract=contract, errors=errors, parsed=parsed,
            allowed_handoff_ids=allowed_handoff_ids))
    invalid_rows = [{
        "stage_id": row["stage_id"],
        "stage_errors": row["stage_errors"],
        "reported_evidence_turn_index": row[
            "reported_evidence_turn_index"],
        "actual_indexed_speaker": row["actual_indexed_speaker"],
        "required_speakers": row["required_speakers"],
        "required_terminal_markers": row["required_terminal_markers"],
        "marker_turn_candidates": row["marker_turn_candidates"],
        "recommended_evidence_turn_indices": row[
            "recommended_evidence_turn_indices"],
    } for row in invalid_stages]
    valid_rows = [{
        "stage_id": row["stage_id"],
        "candidate_stage_update_to_preserve": row[
            "candidate_stage_update"],
    } for row in valid_stages]
    handoff_rows = stateful_realizer_handoff_diagnostics(
        parsed=parsed, allowed_handoff_ids=allowed_handoff_ids)
    invalid_handoff_rows = [row for row in handoff_rows if not row["valid"]]
    directed_corrections = []
    for row in invalid_rows:
        recommended = row["recommended_evidence_turn_indices"]
        if len(recommended) == 1:
            turn_index = recommended[0]
            candidate = next(
                item for item in row["marker_turn_candidates"]
                if item["turn_index"] == turn_index)
            directed_corrections.append(
                f"Stage {row['stage_id']}: The required marker is already "
                f"present at turn {turn_index} and is spoken by "
                f"{candidate['speaker']}. Set evidence_turn_index to "
                f"{turn_index} and result_speaker to {candidate['speaker']}.")
    for row in invalid_handoff_rows:
        reported = row["reported_handoff_id"]
        if not allowed_handoff_ids:
            directed_corrections.append(
                f"{reported} is not a valid handoff ID. The current ledger "
                "contains no available handoffs, so omit this update and "
                "return candidate_handoff_updates as an empty array.")
        else:
            directed_corrections.append(
                f"{reported} is not a valid handoff ID. Omit this update; "
                "only exact IDs in allowed_handoff_ids may be used.")
    repair_scope = (
        "The dialogue evidence is already sufficient. Preserve dialogue_turns "
        "verbatim, using the frozen copy when provided. Correct only the "
        "listed invalid metadata: use the unique recommended "
        "evidence_turn_index and matching authorized result_speaker; remove "
        "every candidate_handoff_updates object whose handoff_id is not in "
        "allowed_handoff_ids. If allowed_handoff_ids is empty, return "
        "candidate_handoff_updates as an empty array. If a prior metadata-only "
        "retry changed the dialogue, restore the frozen dialogue exactly. "
        "Return the complete JSON object."
        if metadata_only_repair else
        "Regenerate the complete JSON object, changing only fields necessary "
        "to correct the listed errors."
    )
    sections = [
        "DIRECTED REALIZER RETRY — REGENERATE ONE COMPLETE JSON OBJECT",
        "The previous response was rejected. Correct every listed error in "
        "this single retry; do not return a patch or commentary.",
        "COMPLETE ERROR SET\n" + json.dumps(
            feedback, ensure_ascii=False, sort_keys=True, indent=2),
        "STAGES WITH ERRORS — CORRECT ONLY THESE STAGE BINDINGS\n" +
        json.dumps(invalid_rows, ensure_ascii=False, sort_keys=True, indent=2),
        "VALID STAGES — PRESERVE THESE UPDATES EXACTLY\n" + json.dumps(
            valid_rows, ensure_ascii=False, sort_keys=True, indent=2),
        "HANDOFF WHITELIST AND INVALID UPDATES\n" + json.dumps({
            "allowed_handoff_ids": sorted(set(allowed_handoff_ids)),
            "invalid_handoff_updates": invalid_handoff_rows,
        }, ensure_ascii=False, sort_keys=True, indent=2),
        "EXACT METADATA CORRECTIONS\n- " + (
            "\n- ".join(directed_corrections)
            if directed_corrections else "Use the complete error set above."),
        "REPAIR SCOPE\n" + repair_scope,
        "If the scene is too short, expand it naturally through concrete "
        "exchanges, reactions, clarifications, decision process, execution "
        "details, and a natural connection to the supplied parent anchors. "
        "Do not pad with summaries, repeated plans, meta commentary, unrelated "
        "small talk, synonymous loops, new events, new people, or Graph states "
        "not present in the frozen contract.",
        "Preserve the authoritative participants, exact speaker names, scene "
        "event, stage order, anchor references, evidence semantics, evidence "
        "speaker assignment, and terminal marker. If length is the only error, "
        "do not rewrite the event or evidence; only expand the surrounding "
        "dialogue naturally.",
        "Each assigned stage has its own evidence turn. For that stage, "
        "candidate_stage_updates evidence_turn_index must point to a turn "
        "spoken by an authorized speaker and containing that stage's exact "
        "terminal marker; result_speaker must match that turn. Stage markers "
        "do not need to be in the final dialogue turn, and distinct stages do "
        "not need to share one evidence turn. Do not modify candidate stage "
        "updates listed under VALID STAGES.",
        "Do not invent, append, splice, or reuse text locally. Regenerate the "
        "entire schema-valid JSON response from the frozen contract.",
        "PREVIOUS REJECTED JSON\n" + json.dumps(
            parsed, ensure_ascii=False, sort_keys=True, indent=2),
    ]
    if frozen_dialogue_turns is not None:
        sections.insert(-1, "FROZEN DIALOGUE_TURNS — COPY VERBATIM\n" +
                        json.dumps(list(frozen_dialogue_turns),
                                   ensure_ascii=False, sort_keys=True,
                                   indent=2))
    return "\n\n".join(sections)


class StatefulGraphRoundOrchestrator:
    """Execute one ten-scene round without the Graph v1 completion shortcut.

    The class reuses the existing backend request object and request ledger.  It
    intentionally contains no Questioner, Solver, verifier, or training path.
    """

    def __init__(
        self, *, run_root: Path, manifest: Mapping[str, Any],
        contracts: Sequence[Mapping[str, Any]], tokenizer: Any,
        realizer: RealizerBackend, backend_mode: str,
        max_attempts_per_scene: int = 3, max_realizer_generations: int = 30,
        prior_state_instances: Mapping[str, Mapping[str, Any]] | None = None,
        prior_handoffs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if max_attempts_per_scene < 1:
            raise ValueError("stateful_max_attempts_per_scene_invalid")
        self.run_root = run_root
        self.manifest = dict(manifest)
        self.contracts = [dict(contract) for contract in contracts]
        self.tokenizer = tokenizer
        self.realizer = realizer
        self.max_attempts_per_scene = max_attempts_per_scene
        self.max_realizer_generations = max_realizer_generations
        manifest_path = run_root / "config/run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest_sha = sha256_file(manifest_path)
        self.ledger = RequestLedger(run_root, self.manifest_sha, backend_mode)
        self.initial_ledger = initialize_stateful_ledger(
            self.contracts, prior_state_instances=prior_state_instances,
            prior_handoffs=prior_handoffs)

    def _tokens(self, turns: Sequence[Mapping[str, Any]]) -> int:
        text = "\n".join(
            f"{turn.get('speaker', '')}: {turn.get('text', '')}" for turn in turns)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _accepted(self) -> list[dict[str, Any]]:
        paths = sorted((self.run_root / "accepted").glob("session_*.json"))
        if not paths:
            return []
        index_path = self.run_root / "accepted/index.json"
        if not index_path.exists():
            raise RuntimeError("stateful_accepted_index_missing")
        index = _read(index_path)
        actual = {path.name: sha256_file(path) for path in paths}
        if (index.get("run_manifest_sha256") != self.manifest_sha or
                index.get("session_sha256") != actual):
            raise RuntimeError("stateful_accepted_session_integrity_drift")
        return [_read(path) for path in paths]

    def _resume_ledger(self, accepted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not accepted:
            return self.initial_ledger
        scene_id = str(accepted[-1]["graph_scene_id"])
        path = self.run_root / f"frozen/state_ledger_after_{scene_id}.json"
        if not path.exists():
            raise RuntimeError("stateful_resume_ledger_missing")
        ledger = _read(path)
        if ledger.get("ledger_sha256") != accepted[-1].get(
                "state_ledger_sha256_after"):
            raise RuntimeError("stateful_resume_ledger_hash_drift")
        return ledger

    @staticmethod
    def _active_handoffs(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [dict(row) for _, row in sorted(
            (ledger.get("handoffs") or {}).items())
            if row.get("status") != "resolved"]

    @staticmethod
    def _backend_identity(backend: Any) -> dict[str, Any]:
        client = getattr(backend, "client", None)
        config = getattr(client, "config", None)
        identity: dict[str, Any] = {
            "backend_class": type(backend).__name__,
            "model_identifier": str(getattr(backend, "model_identifier", "")),
        }
        if config is not None:
            identity.update({
                "model_identifier": str(getattr(config, "model", "")),
                "temperature": getattr(config, "temperature", None),
                "enable_thinking": getattr(config, "enable_thinking", None),
                "realizer_max_tokens": getattr(
                    config, "realizer_max_tokens", None),
            })
        return identity

    def run(
        self, *, request_limit: int | None = None,
        scene_limit: int | None = None,
    ) -> dict[str, Any]:
        accepted = self._accepted()
        ledger = self._resume_ledger(accepted)
        scene_end = len(self.contracts) if scene_limit is None else scene_limit
        if not len(accepted) <= scene_end <= len(self.contracts):
            raise ValueError("stateful_scene_limit_invalid")
        physical_this_run = 0
        if stateful_completion_report(self.contracts, ledger)["complete"]:
            return {"sessions": accepted, "state_ledger": ledger,
                    "completion": stateful_completion_report(
                        self.contracts, ledger)}
        child_turn_count = sum(len(session.get("dialogue_turns") or [])
                               for session in accepted)
        for position in range(len(accepted), scene_end):
            contract = self.contracts[position]
            active_handoffs = self._active_handoffs(ledger)
            allowed_handoff_ids = sorted(
                str(row["handoff_id"]) for row in active_handoffs)
            base_prompt = build_stateful_graph_realizer_prompt(
                contract=contract, recent_sessions=accepted[-2:],
                active_handoffs=active_handoffs)
            retry_suffix: str | None = None
            metadata_only_dialogue: list[dict[str, Any]] | None = None
            accepted_scene = False
            for attempt in range(1, self.max_attempts_per_scene + 1):
                prompt = base_prompt if retry_suffix is None else (
                    base_prompt + "\n\n" + retry_suffix)
                prompt_sha = _sha_text(prompt)
                request_id = (
                    f"stateful-realizer-s{position + 1:02d}-a{attempt:02d}")
                prompt_path = self.run_root / f"raw/prompts/{request_id}.json"
                if prompt_path.exists():
                    if _read(prompt_path).get("prompt_sha256") != prompt_sha:
                        raise RuntimeError("stateful_prompt_drift")
                else:
                    atomic_json(prompt_path, {
                        "request_id": request_id,
                        "scene_id": contract["scene_id"], "attempt": attempt,
                        "prompt_sha256": prompt_sha, "prompt": prompt,
                    })
                request = RealizerRequest(
                    request_id=request_id, session_index=position + 1,
                    attempt=attempt, prompt=prompt,
                    prompt_sha256=prompt_sha)
                response = self.ledger.invoke(
                    request_id=request_id, intent="stateful_realize_scene",
                    attempt=attempt, prompt_sha256=prompt_sha,
                    call=lambda request=request: self.realizer.generate_session(
                        request),
                    budget_class="realizer",
                    budget_limit=self.max_realizer_generations,
                    total_budget_limit=self.max_realizer_generations,
                    request_metadata={
                        "scene_id": contract["scene_id"],
                        "contract_sha256": contract["contract_sha256"],
                    },
                    backend_identity=self._backend_identity(self.realizer),
                )
                if not response.metadata.get("reused", False):
                    physical_this_run += 1
                atomic_json(
                    self.run_root / f"raw/{request_id}.json",
                    {"request_id": request_id,
                     "raw_response": response.raw_text,
                     "metadata": response.metadata})
                parsed: Any = {}
                tokens = 0
                turn_count = 0
                try:
                    parsed = json.loads(response.raw_text)
                except json.JSONDecodeError:
                    result = {"decision": "reject_stateful_scene",
                              "errors": ["stateful_realizer_json_invalid"],
                              "ledger": ledger}
                    parsed = {}
                else:
                    if not isinstance(parsed, dict):
                        result = {
                            "decision": "reject_stateful_scene",
                            "errors": ["stateful_realizer_schema_invalid"],
                            "ledger": ledger,
                        }
                        parsed = {}
                    else:
                        result = validate_and_apply_stateful_scene(
                            parsed, contract, ledger)
                        turns = parsed.get("dialogue_turns") or []
                        turn_count = len(turns)
                        tokens = self._tokens(turns)
                        hard_turns = contract["hard_turn_range"]
                        hard_tokens = contract["hard_dialogue_token_range"]
                        if not hard_turns[0] <= turn_count <= hard_turns[1]:
                            result["errors"].append(
                                "stateful_scene_turn_count_out_of_range")
                        if not hard_tokens[0] <= tokens <= hard_tokens[1]:
                            result["errors"].append(
                                "stateful_scene_token_count_out_of_range")
                        if (metadata_only_dialogue is not None and
                                not stateful_realizer_dialogue_preserved(
                                    metadata_only_dialogue, parsed)):
                            result["errors"].append(
                                "metadata_only_retry_dialogue_changed")
                        if result["errors"]:
                            result["decision"] = "reject_stateful_scene"
                            result["ledger"] = ledger
                if result["decision"] != "accept_stateful_scene":
                    reasons = sorted(set(result["errors"]))
                    self.ledger.record_decision(
                        request_id, "reject:" + ",".join(reasons))
                    atomic_json(self.run_root / f"reports/{request_id}.json", {
                        "request_id": request_id, "decision": "reject",
                        "reasons": reasons,
                        "actual_dialogue_tokens": tokens,
                        "actual_turn_count": turn_count,
                        "stage_diagnostics": (
                            stateful_realizer_stage_diagnostics(
                                contract=contract, parsed=parsed)
                            if parsed else []),
                        "handoff_diagnostics": (
                            stateful_realizer_handoff_diagnostics(
                                parsed=parsed,
                                allowed_handoff_ids=allowed_handoff_ids)
                            if parsed else []),
                        "allowed_handoff_ids": allowed_handoff_ids,
                        "metadata_only_dialogue_preservation_required":
                            metadata_only_dialogue is not None,
                        "dialogue_turns_sha_before": (
                            stateful_dialogue_turns_sha256(
                                metadata_only_dialogue)
                            if metadata_only_dialogue is not None else None),
                        "dialogue_turns_sha_after": (
                            stateful_dialogue_turns_sha256(
                                parsed.get("dialogue_turns") or [])
                            if metadata_only_dialogue is not None and parsed
                            else None),
                        "metadata_only_dialogue_preserved": (
                            stateful_realizer_dialogue_preserved(
                                metadata_only_dialogue, parsed)
                            if metadata_only_dialogue is not None and parsed
                            else None),
                    })
                    if (attempt < self.max_attempts_per_scene and
                            isinstance(parsed, dict)):
                        if (metadata_only_dialogue is None and
                                stateful_realizer_metadata_only_retry_available(
                                    contract=contract, errors=reasons,
                                    parsed=parsed,
                                    allowed_handoff_ids=
                                        allowed_handoff_ids)):
                            metadata_only_dialogue = deepcopy(
                                parsed.get("dialogue_turns") or [])
                        retry_suffix = build_stateful_realizer_retry_suffix(
                            contract=contract, errors=reasons, parsed=parsed,
                            actual_tokens=tokens, actual_turns=turn_count,
                            frozen_dialogue_turns=metadata_only_dialogue,
                            allowed_handoff_ids=allowed_handoff_ids)
                    if (request_limit is not None and
                            physical_this_run >= request_limit):
                        return {"sessions": accepted, "state_ledger": ledger,
                                "completion": stateful_completion_report(
                                    self.contracts, ledger)}
                    continue

                atomic_json(self.run_root / f"reports/{request_id}.json", {
                    "request_id": request_id, "decision": "accept",
                    "reasons": [],
                    "actual_dialogue_tokens": tokens,
                    "actual_turn_count": turn_count,
                    "stage_diagnostics":
                        stateful_realizer_stage_diagnostics(
                            contract=contract, parsed=parsed),
                    "handoff_diagnostics":
                        stateful_realizer_handoff_diagnostics(
                            parsed=parsed,
                            allowed_handoff_ids=allowed_handoff_ids),
                    "allowed_handoff_ids": allowed_handoff_ids,
                    "metadata_only_dialogue_preservation_required":
                        metadata_only_dialogue is not None,
                    "dialogue_turns_sha_before": (
                        stateful_dialogue_turns_sha256(
                            metadata_only_dialogue)
                        if metadata_only_dialogue is not None else None),
                    "dialogue_turns_sha_after": (
                        stateful_dialogue_turns_sha256(
                            parsed.get("dialogue_turns") or [])
                        if metadata_only_dialogue is not None else None),
                    "metadata_only_dialogue_preserved": (
                        stateful_realizer_dialogue_preserved(
                            metadata_only_dialogue, parsed)
                        if metadata_only_dialogue is not None else None),
                })
                turns = []
                for turn in parsed["dialogue_turns"]:
                    child_turn_count += 1
                    turns.append({**turn, "turn_id": f"C{child_turn_count:04d}",
                                  "session_id": f"session_{position + 1:02d}"})
                next_ledger = result["ledger"]
                session = {
                    **parsed, "dialogue_turns": turns,
                    "session_id": f"session_{position + 1:02d}",
                    "session_index": position + 1,
                    "request_id": request_id, "accepted_attempt": attempt,
                    "graph_scene_id": contract["scene_id"],
                    "graph_scene_contract_version": contract["contract_version"],
                    "graph_scene_contract_sha256": contract["contract_sha256"],
                    "authoritative_stage_updates": result[
                        "accepted_stage_updates"],
                    "authoritative_handoff_updates": result[
                        "accepted_handoff_updates"],
                    "created_handoffs": result["created_handoffs"],
                    "state_ledger_sha256_after": next_ledger["ledger_sha256"],
                    "dialogue_token_count": self._tokens(turns),
                    "final_decision": "accept",
                }
                accepted_path = self.run_root / (
                    f"accepted/session_{position + 1:02d}.json")
                atomic_json(accepted_path, session)
                index_path = self.run_root / "accepted/index.json"
                if index_path.exists():
                    index = _read(index_path)
                else:
                    index = {
                        "schema_version": "stateful_accepted_session_index_v1",
                        "run_id": self.manifest.get("run_id"),
                        "run_manifest_sha256": self.manifest_sha,
                        "session_sha256": {},
                    }
                index["session_sha256"][accepted_path.name] = sha256_file(
                    accepted_path)
                atomic_json(index_path, index)
                atomic_json(self.run_root / (
                    f"frozen/state_ledger_after_{contract['scene_id']}.json"),
                    next_ledger)
                self.ledger.record_decision(request_id, "accept_stateful_scene")
                accepted.append(session)
                ledger = next_ledger
                accepted_scene = True
                if (request_limit is not None and
                        physical_this_run >= request_limit):
                    return {"sessions": accepted, "state_ledger": ledger,
                            "completion": stateful_completion_report(
                                self.contracts, ledger)}
                break
            if not accepted_scene:
                raise RuntimeError(
                    f"stateful_scene_exhausted_attempts:{contract['scene_id']}")
        completion = stateful_completion_report(self.contracts, ledger)
        atomic_json(self.run_root / "reports/stateful_completion_report.json",
                    completion)
        atomic_json(self.run_root / "frozen/final_state_ledger.json", ledger)
        atomic_json(self.run_root / "frozen/stateful_child_dialogue.json", {
            "schema_version": "stateful_child_dialogue_v1",
            "turns": [turn for session in accepted
                      for turn in session["dialogue_turns"]],
        })
        return {"sessions": accepted, "state_ledger": ledger,
                "completion": completion}
