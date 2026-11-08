"""v1.1 revision prompt wiring; frozen templates remain unchanged."""

from __future__ import annotations

import json
from typing import Any, Mapping


ADDENDUM = """# QA Output Contract v1.1 Revision 1

Output mechanics:
- Always emit an explicit question_type: single_choice, multi_select, or ordering for QA; null for NO_SAFE_EVOLUTION.
- single_choice uses exactly A-D and one canonical_gold key.
- multi_select uses 4-6 contiguous keys A-D, A-E, or A-F; canonical_gold has 2 through n unique keys and may include all options.
- ordering uses I1-I4, I1-I5, or I1-I6; canonical_gold is the complete unique permutation of all item keys.
- Emit evidence_refs once. Do not emit new_r03_evidence_refs; it is derived locally.
- At least one real dialogue turn ref is required for QA. Evidence completeness is not required in this first pass.
"""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dialogue_text(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(f"[{x.get('turn_ref')}] {x.get('speaker')}: {x.get('text')}" for x in dialogue.get("turns") or [])


def targeted_prompt(common: Mapping[str, Any], dialogue: Mapping[str, Any], *, template: str, schema: Mapping[str, Any], strategy_schema: Mapping[str, Any]) -> str:
    return "\n".join([
        "You are the Questioner running one bounded feedback-guided targeted-evolution attempt.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA. Do not return markdown, a list, commentary, or extra fields.",
        "The frozen Feedback Template controls evolution semantics; the addendum controls output mechanics.",
        "\nFROZEN_FEEDBACK_TEMPLATE_V1:\n" + template,
        "\nQA_OUTPUT_CONTRACT_ADDENDUM_V1_1_REVISION_1:\n" + ADDENDUM,
        "\nSTRATEGY_SCHEMA:\n" + canonical(strategy_schema),
        "\nOUTPUT_SCHEMA:\n" + canonical(schema),
        "\nCOMMON_FEEDBACK_DATA:\n" + canonical(common),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + dialogue_text(dialogue),
    ])


def multi_pass_prompt(pass_name: str, sample: Mapping[str, Any], dialogue: Mapping[str, Any], *, template: str, schema: Mapping[str, Any]) -> str:
    if pass_name not in {"composition", "fresh"}:
        raise ValueError(f"unsupported_pass:{pass_name}")
    return "\n".join([
        f"You are the Questioner running one bounded R03 {pass_name} attempt.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA. Do not return markdown, a list, commentary, or extra fields.",
        "The frozen task template controls generation semantics; the addendum controls output mechanics.",
        "\nFROZEN_TASK_TEMPLATE_V1:\n" + template,
        "\nQA_OUTPUT_CONTRACT_ADDENDUM_V1_1_REVISION_1:\n" + ADDENDUM,
        "\nOUTPUT_SCHEMA:\n" + canonical(schema),
        "\nPASS_INPUT:\n" + canonical(sample),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + dialogue_text(dialogue),
    ])

