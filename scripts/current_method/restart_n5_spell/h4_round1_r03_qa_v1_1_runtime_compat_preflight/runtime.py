"""Prompt construction using frozen task templates and the v1.1 addendum."""

from __future__ import annotations

import json
from typing import Any, Mapping


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_dialogue(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(
        f"[{turn.get('turn_ref')}] {turn.get('speaker')}: {turn.get('text')}"
        for turn in dialogue.get("turns") or []
    )


def targeted_prompt(
    common_feedback: Mapping[str, Any], dialogue: Mapping[str, Any], *,
    frozen_template: str, addendum: str, output_schema: Mapping[str, Any],
    strategy_schema: Mapping[str, Any],
) -> str:
    return "\n".join([
        "You are the Questioner running one bounded feedback-guided targeted-evolution verification.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA. Do not return a list, markdown, commentary, or multiple attempts.",
        "The model chooses the strategy. Use only AUTHORITATIVE_CUMULATIVE_DIALOGUE for facts, options, Gold, and evidence refs.",
        "The frozen Feedback Template controls feedback semantics. QA_OUTPUT_CONTRACT_ADDENDUM_V1_1 controls output mechanics only.",
        "Do not add fields. In particular, emit evidence_refs once; new_r03_evidence_refs is derived locally.",
        "\nFROZEN_FEEDBACK_TEMPLATE_V1:\n" + frozen_template,
        "\nQA_OUTPUT_CONTRACT_ADDENDUM_V1_1:\n" + addendum,
        "\nSTRATEGY_SCHEMA:\n" + canonical(strategy_schema),
        "\nOUTPUT_SCHEMA_V1_1:\n" + canonical(output_schema),
        "\nCOMMON_FEEDBACK_DATA:\n" + canonical(common_feedback),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + render_dialogue(dialogue),
    ])


def multi_pass_prompt(
    pass_name: str, sample: Mapping[str, Any], dialogue: Mapping[str, Any], *,
    frozen_template: str, addendum: str, output_schema: Mapping[str, Any],
) -> str:
    if pass_name not in {"composition", "fresh"}:
        raise ValueError(f"unsupported_pass:{pass_name}")
    return "\n".join([
        f"You are the Questioner running the bounded R03 {pass_name} verification.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA_V1_1. Do not return markdown or extra fields.",
        "Use only AUTHORITATIVE_CUMULATIVE_DIALOGUE as factual authority.",
        "The frozen task template controls generation semantics. QA_OUTPUT_CONTRACT_ADDENDUM_V1_1 controls output mechanics only.",
        "Emit evidence_refs once; new_r03_evidence_refs is derived locally.",
        "\nFROZEN_TASK_TEMPLATE_V1:\n" + frozen_template,
        "\nQA_OUTPUT_CONTRACT_ADDENDUM_V1_1:\n" + addendum,
        "\nOUTPUT_SCHEMA_V1_1:\n" + canonical(output_schema),
        "\nPASS_INPUT:\n" + canonical(sample),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + render_dialogue(dialogue),
    ])


def audit_rendered_prompt(prompt: str, *, schema_id: str, template_sha: str, observed_template_sha: str) -> dict[str, Any]:
    return {
        "schema_id_present": schema_id in prompt,
        "addendum_present": "# QA Output Contract Addendum v1.1" in prompt,
        "single_evidence_instruction_present": "emit evidence_refs once" in prompt.casefold(),
        "local_r03_derivation_instruction_present": "new_r03_evidence_refs is derived locally" in prompt,
        "frozen_template_sha_match": template_sha == observed_template_sha,
        "legacy_schema_id_absent": 'output.v1\"' not in prompt and 'output.v1"' not in prompt,
    }

