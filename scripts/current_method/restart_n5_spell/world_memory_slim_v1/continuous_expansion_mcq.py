"""Small contracts used by the expanded-dialogue MCQ canary."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


MCQ_SOLVER_PROMPT_VERSION = "slim_expanded_dialogue_full_context_solver_v1"
MCQ_SOLVER_MAX_TOKENS = 128


def build_full_context_mcq_solver_prompt(
    *, frozen_dialogue: Sequence[Mapping[str, Any]],
    materialized_qa: Mapping[str, Any],
) -> str:
    options = materialized_qa.get("options")
    if (not isinstance(options, dict) or set(options) != set("ABCD")):
        raise ValueError("mcq_solver_options_invalid")
    rows = [{"turn_id": row["turn_id"], "session_id": row["session_id"],
             "speaker": row["speaker"], "text": row["text"]}
            for row in frozen_dialogue]
    task = {"question": materialized_qa["question"],
            "options": {letter: options[letter] for letter in "ABCD"}}
    schema = {"type": "object", "additionalProperties": False,
              "required": ["answer_letter"],
              "properties": {"answer_letter": {"enum": list("ABCD")}}}
    return "\n\n".join([
        f"PROMPT_VERSION: {MCQ_SOLVER_PROMPT_VERSION}",
        "Use only the full frozen dialogue to answer the multiple-choice question. Compare both the initial parent state and the later child development. Return exactly one JSON object and no explanation.",
        "FULL FROZEN DIALOGUE\n" + json.dumps(
            rows, ensure_ascii=False, separators=(",", ":")),
        "QUESTION AND OPTIONS\n" + json.dumps(
            task, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "OUTPUT SCHEMA\n" + json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ])


def validate_mcq_solver_output(raw: Any) -> dict[str, Any]:
    if (not isinstance(raw, dict) or set(raw) != {"answer_letter"} or
            raw.get("answer_letter") not in set("ABCD")):
        return {"decision": "mcq_solver_schema_invalid",
                "answer_letter": None}
    return {"decision": "mcq_solver_contract_accept",
            "answer_letter": raw["answer_letter"]}

