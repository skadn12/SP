"""Arm-aware prompts and thin postprocessing on the formal shared Miner stage.

No Miner pipeline, evidence selector, semantic validator or Writer is defined
here. A/B pass their complete raw JSON directly to the original canonicalizer;
C delegates its response projection and all binding processing to v2.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, canonical_json, sha256_bytes
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import (
    COMMON_BASE_VERSION, MINER_TEMPLATE_VERSION, FIXED_POLICY, model_cards, model_history,
    miner_output_schema as common_output_schema, revise_miner_prompt as common_revise_prompt,
)
from ..h4_r02_miner_binding_v2.transfer import (
    TransferHooksV2, MINER_BINDING_RULES, DELIVERY_STATUS,
    derive_all_intent_evidence, miner_output_schema as v2_output_schema,
    revise_miner_prompt as v2_revise_prompt,
)

VERSION = "h4_r02_miner_three_arm_diagnostic_v1"
ARMS = ("A_OFF", "B_OBSERVATION_ONLY", "C_P2_2_TRANSFER")
SCOPE_FIX_VERSION = "h4_r02_miner_explanation_scope_fix_v1"
EXPLANATION_SCOPE_FIX_VERSION = SCOPE_FIX_VERSION
OLD_SCOPE_SENTENCE = "Do not write questions, options, answers, distractors, or explanations."
NEW_SCOPE_SENTENCE = "Do not generate QA questions, options, answers, distractors, or answer explanations. Output only the fields required by the active response schema."
PAYLOAD_MARKER = "ROOT OBSERVATIONS AND HISTORICAL TRACKING:\n"
SCHEMA_MARKER = "Return exactly one JSON object conforming to this JSON Schema (descriptions are not output values):\n"
DIALOGUE_MARKER = "\nNUMBERED AUTHORITATIVE DIALOGUE:"


def _digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _check_arm(arm_id: str) -> None:
    if arm_id not in ARMS:
        raise ValueError("UNKNOWN_DIAGNOSTIC_ARM")


def output_schema(arm_id: str) -> dict[str, Any]:
    _check_arm(arm_id)
    return v2_output_schema() if arm_id == "C_P2_2_TRANSFER" else common_output_schema(envelope=False)


def feedback_payload(arm_id: str, cards: Any, history_only: Any, root_id: str) -> dict[str, Any]:
    _check_arm(arm_id)
    projected = [] if arm_id == "A_OFF" else model_cards(cards, root_id=root_id)
    return {"root_id": root_id, "feedback_status": "ACTIVE_FEEDBACK" if projected else "NO_ACTIVE_FEEDBACK",
        "observation_cards": projected, "historical_tracking": model_history(history_only, root_id=root_id)}


def _scope_fix(prompt: str) -> str:
    if prompt.count(OLD_SCOPE_SENTENCE) != 1 or NEW_SCOPE_SENTENCE in prompt:
        raise ValueError("EXPLANATION_SCOPE_SOURCE_DRIFT")
    return prompt.replace(OLD_SCOPE_SENTENCE, NEW_SCOPE_SENTENCE, 1)


def build_prompt(arm_id: str, original_spec: Mapping[str, Any], cards: Any, history_only: Any, *,
                 expanded: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    """Pure builder; arm metadata is never rendered into the model prompt."""
    _check_arm(arm_id)
    revised = (v2_revise_prompt(original_spec) if arm_id == "C_P2_2_TRANSFER"
        else common_revise_prompt(original_spec, envelope=False))
    revised_prompt = _scope_fix(str(revised["prompt"]))
    payload = feedback_payload(arm_id, cards, history_only, root_id)
    sections = [revised_prompt]
    if arm_id == "C_P2_2_TRANSFER":
        sections.append(MINER_BINDING_RULES)
    sections.extend([FIXED_POLICY, PAYLOAD_MARKER + json.dumps(payload, ensure_ascii=False, indent=2)])
    prompt = "\n".join(sections)
    return {**revised, "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode()),
        "prompt_version": VERSION + "+" + SCOPE_FIX_VERSION,
        "common_base_version": COMMON_BASE_VERSION, "common_miner_schema_revision": MINER_TEMPLATE_VERSION,
        "explanation_scope_fix_version": SCOPE_FIX_VERSION, "arm_id": arm_id,
        "response_schema_sha256": _digest(output_schema(arm_id)),
        "feedback_payload_sha256": _digest(payload),
        "historical_tracking_sha256": _digest(payload["historical_tracking"]),
        "transcript_canonical_sha256": _digest(expanded),
        "concrete_transfer_task": arm_id == "C_P2_2_TRANSFER"}


def _split_payload(prompt: str) -> tuple[str, dict[str, Any]]:
    if prompt.count(PAYLOAD_MARKER) != 1:
        raise ValueError("PROMPT_PAYLOAD_MARKER_DRIFT")
    prefix, serialized = prompt.split(PAYLOAD_MARKER, 1)
    return prefix, json.loads(serialized)


def _remove_c_task(prompt: str) -> str:
    """Exact-byte removal of only the declared C schema and instruction delta."""
    c_section = SCHEMA_MARKER + json.dumps(v2_output_schema(), ensure_ascii=False, indent=2) + DIALOGUE_MARKER
    ab_section = SCHEMA_MARKER + json.dumps(common_output_schema(envelope=False), ensure_ascii=False, indent=2) + DIALOGUE_MARKER
    transfer_section = "\n" + MINER_BINDING_RULES + "\n"
    if prompt.count(c_section) != 1 or prompt.count(transfer_section) != 1:
        raise ValueError("C_TASK_SECTION_DRIFT")
    return prompt.replace(c_section, ab_section, 1).replace(transfer_section, "\n", 1)


def prompt_pairing_diagnostics(original_spec: Mapping[str, Any], cards: Any, history_only: Any, *,
                               expanded: Mapping[str, Any], root_id: str,
                               legacy_v2_prompt: str | None = None) -> dict[str, Any]:
    """Prove the frozen comparison by exact text replacement, not similarity."""
    specs = {arm: build_prompt(arm, original_spec, cards, history_only, expanded=expanded, root_id=root_id) for arm in ARMS}
    prompts = {arm: spec["prompt"] for arm, spec in specs.items()}
    prefixes, payloads = {}, {}
    for arm in ARMS:
        prefixes[arm], payloads[arm] = _split_payload(prompts[arm])
    a, b, c = ARMS
    normalized_a = deepcopy(payloads[a])
    normalized_a["feedback_status"] = payloads[b]["feedback_status"]
    normalized_a["observation_cards"] = deepcopy(payloads[b]["observation_cards"])
    a_with_b_payload = prefixes[a] + PAYLOAD_MARKER + json.dumps(normalized_a, ensure_ascii=False, indent=2)
    if legacy_v2_prompt is None:
        old_revised = v2_revise_prompt(original_spec)
        legacy_v2_prompt = old_revised["prompt"] + "\n" + MINER_BINDING_RULES + "\n" + FIXED_POLICY + "\n" + PAYLOAD_MARKER + json.dumps(payloads[c], ensure_ascii=False, indent=2)
        legacy_basis = "unchanged_v2_builder_components"
    else:
        legacy_basis = "supplied_historical_v2_prompt_bytes"
    checks = {
        "a_b_payload_only_exact_bytes": a_with_b_payload == prompts[b],
        "a_b_nonpayload_prefix_exact_bytes": prefixes[a] == prefixes[b],
        "b_c_payload_exact_bytes": prompts[b].split(PAYLOAD_MARKER, 1)[1] == prompts[c].split(PAYLOAD_MARKER, 1)[1],
        "b_c_v2_task_only_exact_bytes": _remove_c_task(prompts[c]) == prompts[b],
        "c_v2_only_scope_sentence_exact_bytes": _scope_fix(legacy_v2_prompt) == prompts[c],
        "history_projection_same": len({_digest(payloads[arm]["historical_tracking"]) for arm in ARMS}) == 1,
        "intent_item_schema_same": len({_digest(output_schema(arm)["properties"]["intents"]["items"]) for arm in ARMS}) == 1,
        "scope_fix_once_each": all(prompt.count(NEW_SCOPE_SENTENCE) == 1 and OLD_SCOPE_SENTENCE not in prompt for prompt in prompts.values()),
        "a_cards_empty": payloads[a]["observation_cards"] == [] and payloads[a]["feedback_status"] == "NO_ACTIVE_FEEDBACK",
        "b_c_cards_same": payloads[b]["observation_cards"] == payloads[c]["observation_cards"],
    }
    return {"version": VERSION, "root_id": root_id, "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "c_v2_comparison_basis": legacy_basis,
        "scope_fix": {"version": SCOPE_FIX_VERSION, "old": OLD_SCOPE_SENTENCE, "new": NEW_SCOPE_SENTENCE},
        "prompt_sha256": {arm: specs[arm]["prompt_sha256"] for arm in ARMS},
        "payload_sha256": {arm: _digest(payloads[arm]) for arm in ARMS},
        "response_schema_sha256": {arm: _digest(output_schema(arm)) for arm in ARMS},
        "historical_v2_prompt_sha256": sha256_bytes(legacy_v2_prompt.encode())}


class ArmHooks:
    """All arms use shared generate_root_intents; only response adapters differ."""
    def __init__(self, arm_id: str, cards: Any, history_only: Any, *, transcript_sha256: str | None = None):
        _check_arm(arm_id)
        self.arm_id = arm_id
        self.cards, self.history_only = deepcopy(cards), deepcopy(history_only)
        self.transcript_sha256 = transcript_sha256
        self._v2 = (TransferHooksV2(cards, history_only, transcript_sha256=transcript_sha256)
            if arm_id == "C_P2_2_TRANSFER" else None)
        self.derived: dict[str, Any] = {}
        self.validations: dict[str, Any] = {}
        self.previews: dict[str, Any] = {}

    def miner_prompt(self, prompt_spec: Mapping[str, Any], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> dict[str, Any]:
        result = build_prompt(self.arm_id, prompt_spec, self.cards, self.history_only, expanded=expanded, root_id=root_id)
        atomic_json(Path(root_dir) / "miner_prompt_adaptation.json", {key: value for key, value in result.items() if key != "prompt"})
        return result

    def project_miner_response(self, raw: Any, *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> Any:
        if self._v2 is not None:
            return self._v2.project_miner_response(raw, expanded=expanded, root_id=root_id, root_dir=root_dir)
        # In A/B an unexpected envelope must reach the original strict parser.
        return raw

    def after_intents(self, raw: Any, intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> None:
        if self._v2 is not None:
            self._v2.after_intents(raw, intent_result, contracts, expanded=expanded, root_id=root_id, root_dir=root_dir)
            self.validations[root_id] = self._v2.validations[root_id]
            self.derived[root_id] = self._v2.derived[root_id]
            self.previews[root_id] = self._v2.previews[root_id]
        else:
            self.derived[root_id] = derive_all_intent_evidence(intent_result, contracts, expanded=expanded,
                root_id=root_id, transcript_sha256=self.transcript_sha256)
            atomic_json(Path(root_dir) / "derived_intent_evidence.json", self.derived[root_id])
        atomic_json(Path(root_dir) / "arm_response_schema.json", {
            "version": VERSION, "arm_id": self.arm_id, "root_id": root_id,
            "binding_processing": "V2_UNCHANGED" if self._v2 is not None else "NOT_APPLICABLE",
            "response_schema_sha256": _digest(output_schema(self.arm_id)),
            "original_intent_validation_errors": list(intent_result.get("errors") or []),
            "raw_response_sha256": _digest(raw), "delivery_status": DELIVERY_STATUS})
