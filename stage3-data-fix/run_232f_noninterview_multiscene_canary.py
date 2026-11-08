from __future__ import annotations

import argparse
import copy
import json
import runpy
import sys
from pathlib import Path
from typing import Any


BASE_RUNNER = Path(__file__).with_name("run_232b_multiscene_canary.py")

PROMPT_VARIANTS: dict[str, list[dict[str, Any]]] = {
    "joint_planning": [
        {
            "name": "joint_planning_update",
            "fact_order": [1, 0],
            "instruction": (
                "Begin in the middle of a shared planning task. John volunteers a "
                "current decision and a still-open possibility while Tim responds "
                "with practical coordination. Neither speaker interviews the other."
            ),
        },
        {
            "name": "memory_correction_in_passing",
            "fact_order": [1, 0],
            "instruction": (
                "Continue the same planning activity while older notes come into "
                "view. John corrects two stale entries in passing; Tim reacts to the "
                "correction without asking about each entry separately."
            ),
        },
    ],
    "side_confirmation": [
        {
            "name": "shared_note_side_confirmation",
            "fact_order": [0, 1],
            "instruction": (
                "Tim begins with an observation from a shared planning note rather "
                "than a question. John clarifies what is open and what is settled, "
                "then both continue their practical task."
            ),
        },
        {
            "name": "change_log_correction",
            "fact_order": [0, 1],
            "instruction": (
                "John notices that an old planning note needs correction and states "
                "the distinction once. Tim acknowledges it and moves the shared task "
                "forward without requesting a status recap."
            ),
        },
    ],
    "voluntary_addendum": [
        {
            "name": "voluntary_update_during_coordination",
            "fact_order": [1, 0],
            "instruction": (
                "John volunteers a concise update while both participants coordinate "
                "next steps. Tim responds socially or practically; do not structure "
                "the exchange as prompts followed by status answers."
            ),
        },
        {
            "name": "older_options_addendum",
            "fact_order": [0, 1],
            "instruction": (
                "John adds a brief correction about older options as part of the "
                "ongoing conversation. Tim responds to the correction as a whole, "
                "without naming each item again or comparing status labels."
            ),
        },
    ],
}


def _rules(scene_index: int) -> list[str]:
    rules = [
        "Write one coherent natural conversation scene of 4-8 turns using only Tim and John.",
        "Start in the middle of a shared activity or thought. The first turn must not be a greeting, check-in, direct question, or request for an update.",
        "Do not begin any turn with Hey John, Hey, Hi, just checking, checking in, Any updates, or remember when.",
        "Follow the discourse mode. Information must be volunteered, noticed, corrected, or confirmed in passing; do not use an interviewer sequence in which Tim names each fact and John reports its status.",
        "Preserve every typed predicate, subject, value, status, and must_mean exclusion exactly.",
        "Give each fact_ref exactly one core proposition-bearing realization in the whole scene. It may occupy one turn or one adjacent two-turn exchange, but a question alone cannot establish it.",
        "After a fact has been expressed, later turns may react, coordinate, or change the social focus, but must not repeat, summarize, contrast, confirm, or paraphrase that fact's value and status.",
        "Light non-factual social context is allowed. Added context is ancillary and must not add or change a destination, date, cause, source, count, ordering, or plan-strength fact.",
        "Return exactly one JSON object whose sole top-level key is turns. Do not return fact refs, claim IDs, supports, quotes, evidence, explanations, or sidecars.",
    ]
    if scene_index == 2:
        rules.insert(
            8,
            "Use the value-free checkpoint only for conversational continuity. Never mention its IDs, infer missing prior facts, or reopen the closed prior exchange.",
        )
    return rules


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="joint_planning",
    )
    parser.add_argument("--prompt-preview", action="store_true")
    known, remaining = parser.parse_known_args()

    namespace = runpy.run_path(str(BASE_RUNNER), run_name="stage3_232b_base")
    base_globals = namespace["main"].__globals__
    base_prompt = base_globals["_prompt"]
    variant = PROMPT_VARIANTS[known.prompt_variant]

    def prompt_only_override(
        scene_index: int,
        facts: list[dict[str, Any]],
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = base_prompt(scene_index, facts, checkpoint)
        original_facts = copy.deepcopy(prompt["semantic_facts"])
        spec = variant[scene_index - 1]
        prompt["mode"] = "stage3_truth_first_noninterview_multiscene_live_canary_v1"
        prompt["discourse_mode"] = {
            "name": spec["name"],
            "instruction": spec["instruction"],
        }
        prompt["semantic_facts"] = [
            original_facts[index] for index in spec["fact_order"]
        ]
        prompt["rules"] = _rules(scene_index)
        return prompt

    base_globals["_prompt"] = prompt_only_override
    base_globals["MODE_SPECS"] = [
        {
            "name": spec["name"],
            "instruction": spec["instruction"],
        }
        for spec in variant
    ]
    if known.prompt_preview:
        _, claims = base_globals["_load_frozen_claims"]()
        facts = base_globals["_typed_facts"](claims)
        checkpoint = base_globals["_checkpoint"]("John")
        scene1_prompt = prompt_only_override(1, facts[:2], None)
        scene2_prompt = prompt_only_override(2, facts[2:], checkpoint)
        print(
            json.dumps(
                {
                    "prompt_variant": known.prompt_variant,
                    "typed_tuples": [
                        {
                            key: fact[key]
                            for key in (
                                "subject",
                                "predicate",
                                "value",
                                "status",
                                "must_mean",
                            )
                        }
                        for fact in facts
                    ],
                    "checkpoint": checkpoint,
                    "checkpoint_audit": base_globals["_checkpoint_audit"](
                        checkpoint, []
                    ),
                    "scene_modes": [
                        scene1_prompt["discourse_mode"],
                        scene2_prompt["discourse_mode"],
                    ],
                    "scene_fact_orders": [
                        [fact["fact_ref"] for fact in scene1_prompt["semantic_facts"]],
                        [fact["fact_ref"] for fact in scene2_prompt["semantic_facts"]],
                    ],
                    "basic_validator_function": base_globals[
                        "_basic_response_audit"
                    ].__name__,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    sys.argv = [str(BASE_RUNNER), *remaining]
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
