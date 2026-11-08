"""OFF-paired v3: preserve the prior A_OFF prompt and insert one frozen block.

The full response reaches the original canonicalizer unchanged. JSON Schema
checks below are a separate structural audit and never change item admission,
the original reasoning contracts, or deterministic evidence derivation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from ..core import atomic_json, canonical_json, sha256_bytes
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import miner_output_schema
from ..h4_r02_miner_three_arm_diagnostic_v1.adapter import (
    ArmHooks, build_prompt as inherited_build_prompt,
    feedback_payload as inherited_feedback_payload,
)

VERSION = "h4_r02_basic_miner_v3_off_paired_v1"
ARMS = ("BASE_OFF", "V3_OFF")
GUIDANCE_VERSION = "intent_selection_guidance_v3"
GUIDANCE_FILE = Path(__file__).with_name("intent_selection_guidance_v3.txt")
DIALOGUE_MARKER = "\nNUMBERED AUTHORITATIVE DIALOGUE:"
GUIDANCE_BEGIN = "BEGIN_INTENT_SELECTION_GUIDANCE_V3"
GUIDANCE_END = "END_INTENT_SELECTION_GUIDANCE_V3"


def _digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _check_arm(arm_id: str) -> None:
    if arm_id not in ARMS:
        raise ValueError("UNKNOWN_OFF_PAIRED_ARM")


def guidance_text() -> str:
    """Read the literal UTF-8 LF template; never regenerate or rewrite it."""
    data = GUIDANCE_FILE.read_bytes()
    text = data.decode("utf-8")
    if b"\r" in data or not text.startswith(GUIDANCE_BEGIN + "\n") or not text.endswith(GUIDANCE_END + "\n"):
        raise ValueError("V3_GUIDANCE_TEMPLATE_FORMAT_DRIFT")
    if text.count(GUIDANCE_BEGIN) != 1 or text.count(GUIDANCE_END) != 1:
        raise ValueError("V3_GUIDANCE_TEMPLATE_MARKER_DRIFT")
    return text


def guidance_insertion() -> str:
    # This leading LF and the template's terminal LF are part of the frozen
    # exact insertion. The original dialogue marker is preserved verbatim.
    return "\n" + guidance_text()


def output_schema(arm_id: str | None = None) -> dict[str, Any]:
    if arm_id is not None:
        _check_arm(arm_id)
    return miner_output_schema(envelope=False)


def feedback_payload(cards: Any, history_only: Any, root_id: str) -> dict[str, Any]:
    return inherited_feedback_payload("A_OFF", cards, history_only, root_id)


def build_prompt(arm_id: str, original_spec: Mapping[str, Any], cards: Any, history_only: Any, *,
                 expanded: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    """Single production builder; BASE model-visible bytes are prior A_OFF."""
    _check_arm(arm_id)
    base = inherited_build_prompt("A_OFF", original_spec, cards, history_only,
        expanded=expanded, root_id=root_id)
    prompt = str(base["prompt"])
    if prompt.count(DIALOGUE_MARKER) != 1:
        raise ValueError("AUTHORITATIVE_DIALOGUE_MARKER_NOT_UNIQUE")
    if GUIDANCE_BEGIN in prompt or GUIDANCE_END in prompt:
        raise ValueError("GUIDANCE_ALREADY_PRESENT_IN_BASE")
    insertion = guidance_insertion()
    if arm_id == "V3_OFF":
        prompt = prompt.replace(DIALOGUE_MARKER, insertion + DIALOGUE_MARKER, 1)
        if prompt.count(insertion) != 1 or prompt.replace(insertion, "", 1) != base["prompt"]:
            raise ValueError("V3_ONLY_INSERTION_INVARIANT_FAILED")
    return {**base, "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode()),
        "prompt_version": VERSION, "arm_id": arm_id, "inherited_prompt_arm": "A_OFF",
        "inherited_prompt_version": base.get("prompt_version"),
        "base_prompt_sha256": sha256_bytes(str(base["prompt"]).encode()),
        "guidance_version": GUIDANCE_VERSION, "guidance_inserted": arm_id == "V3_OFF",
        "guidance_template_sha256": sha256_bytes(guidance_text().encode()),
        "guidance_insertion_sha256": sha256_bytes(insertion.encode()),
        "response_schema_sha256": _digest(output_schema()), "concrete_transfer_task": False}


def _prompt_text(value: str | Path) -> str:
    return value.read_bytes().decode("utf-8") if isinstance(value, Path) else value


def prompt_pairing_diagnostics(expanded: Mapping[str, Any], cards: Any, history_only: Any,
                               root_id: str, archived_policy_prompt: str | Path,
                               archived_actual_prompt: str | Path) -> dict[str, Any]:
    """Compare both historical baselines and exactly subtract the v3 block."""
    original = build_intent_prompt(expanded)
    specs = {arm: build_prompt(arm, original, cards, history_only, expanded=expanded, root_id=root_id) for arm in ARMS}
    base, v3 = (specs[arm]["prompt"] for arm in ARMS)
    policy, actual = _prompt_text(archived_policy_prompt), _prompt_text(archived_actual_prompt)
    insertion = guidance_insertion()
    payload = feedback_payload(cards, history_only, root_id)
    checks = {
        "base_equals_archived_policy_bytes": base.encode() == policy.encode(),
        "base_equals_archived_actual_bytes": base.encode() == actual.encode(),
        "archived_policy_equals_actual_bytes": policy.encode() == actual.encode(),
        "v3_exactly_one_guidance_insertion": v3.count(insertion) == 1,
        "v3_removal_equals_base_bytes": v3.replace(insertion, "", 1).encode() == base.encode(),
        "guidance_immediately_before_dialogue_marker": insertion + DIALOGUE_MARKER in v3,
        "dialogue_marker_unique_each": base.count(DIALOGUE_MARKER) == v3.count(DIALOGUE_MARKER) == 1,
        "schema_same": specs[ARMS[0]]["response_schema_sha256"] == specs[ARMS[1]]["response_schema_sha256"],
        "payload_same": specs[ARMS[0]]["feedback_payload_sha256"] == specs[ARMS[1]]["feedback_payload_sha256"],
        "history_same": specs[ARMS[0]]["historical_tracking_sha256"] == specs[ARMS[1]]["historical_tracking_sha256"],
        "transcript_same": specs[ARMS[0]]["transcript_canonical_sha256"] == specs[ARMS[1]]["transcript_canonical_sha256"],
        "both_off": payload["feedback_status"] == "NO_ACTIVE_FEEDBACK" and payload["observation_cards"] == [],
    }
    return {"version": VERSION, "root_id": root_id, "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "prompt_sha256": {arm: specs[arm]["prompt_sha256"] for arm in ARMS},
        "archived_policy_prompt_sha256": sha256_bytes(policy.encode()),
        "archived_actual_prompt_sha256": sha256_bytes(actual.encode()),
        "guidance_template_sha256": specs[ARMS[1]]["guidance_template_sha256"],
        "guidance_insertion_sha256": specs[ARMS[1]]["guidance_insertion_sha256"],
        "response_schema_sha256": _digest(output_schema()), "feedback_payload_sha256": _digest(payload)}


def _json_pointer(path: Any) -> str:
    return "".join("/" + str(value).replace("~", "~0").replace("/", "~1") for value in path)


def audit_response_schema(raw: Any) -> dict[str, Any]:
    """Audit the unchanged full schema separately from original admission."""
    schema = output_schema()
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(raw),
        key=lambda error: (_json_pointer(error.absolute_path), _json_pointer(error.absolute_schema_path), error.message))
    return {"audit_version": VERSION, "schema_valid": not errors, "valid": not errors,
        "schema_sha256": _digest(schema), "schema_source": "h4_r02_feedback_transfer_p2_2_v1.transfer.miner_output_schema(envelope=False)",
        "errors": [{"instance_path": _json_pointer(error.absolute_path),
            "schema_path": _json_pointer(error.absolute_schema_path), "keyword": error.validator,
            "message": error.message} for error in errors],
        "raw_response_sha256": _digest(raw), "affects_original_admission": False,
        "audit_scope": "FULL_RESPONSE_JSON_SCHEMA_ONLY; original canonicalizer and contract validity are independent"}


class MinerHooks:
    """Use prior A_OFF complete-response and derivation hooks for both arms."""
    def __init__(self, arm_id: str, cards: Any, history_only: Any, *, transcript_sha256: str | None = None):
        _check_arm(arm_id)
        self.arm_id = arm_id
        self.cards, self.history_only = deepcopy(cards), deepcopy(history_only)
        self._off = ArmHooks("A_OFF", cards, history_only, transcript_sha256=transcript_sha256)
        self.derived = self._off.derived
        self.validations = self._off.validations
        self.previews = self._off.previews

    def miner_prompt(self, prompt_spec: Mapping[str, Any], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> dict[str, Any]:
        result = build_prompt(self.arm_id, prompt_spec, self.cards, self.history_only, expanded=expanded, root_id=root_id)
        atomic_json(Path(root_dir) / "miner_prompt_adaptation.json", {key: value for key, value in result.items() if key != "prompt"})
        return result

    def project_miner_response(self, raw: Any, *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> Any:
        return self._off.project_miner_response(raw, expanded=expanded, root_id=root_id, root_dir=root_dir)

    def after_intents(self, raw: Any, intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> None:
        self._off.after_intents(raw, intent_result, contracts, expanded=expanded, root_id=root_id, root_dir=root_dir)
        audit = audit_response_schema(raw)
        atomic_json(Path(root_dir) / "response_schema_audit.json", {"arm_id": self.arm_id, "root_id": root_id, **audit})
        # The inherited hook's model-independent sidecar names its original
        # A_OFF arm. Record this run's actual identity without altering prompts,
        # intents, contracts, evidence or the inherited processing behavior.
        atomic_json(Path(root_dir) / "arm_response_schema.json", {
            "version": VERSION, "arm_id": self.arm_id, "root_id": root_id,
            "inherited_processing_arm": "A_OFF", "binding_processing": "NOT_APPLICABLE",
            "response_schema_sha256": audit["schema_sha256"], "full_response_schema_valid": audit["schema_valid"],
            "original_intent_validation_errors": list(intent_result.get("errors") or []),
            "raw_response_sha256": audit["raw_response_sha256"], "delivery_status": "NOT_SENT_TO_WRITER"})
