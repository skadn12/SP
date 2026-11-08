from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


NEXT_ACTIONS = (
    "increase_difficulty",
    "moderately_increase_difficulty",
    "keep_as_frontier",
    "slightly_reduce_or_validate",
    "validate_as_risky",
    "regenerate_or_validate",
)

HARD_DIFFICULTY_OPS = (
    "option_confusion",
    "temporal_update",
    "entity_confusion",
    "evidence_dispersion",
    "question_evolution",
)

DATE_HINT_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "yesterday",
    "today",
    "tomorrow",
    "week",
    "month",
    "year",
    "ago",
    "before",
    "after",
}

DURATION_WORDS = {"day", "days", "week", "weeks", "month", "months", "year", "years"}
LOCATION_PREPOSITIONS = (" in ", " from ", " at ", " to ")
RELATIONSHIP_WORDS = {"single", "married", "divorced", "partner", "relationship", "dating", "widowed"}
IDENTITY_WORDS = {"transgender", "woman", "man", "nonbinary", "identity", "gay", "lesbian", "bisexual"}


def infer_question_type(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith("when"):
        return "when"
    if lowered.startswith("where"):
        return "where"
    if lowered.startswith("who"):
        return "who"
    if lowered.startswith("why"):
        return "why"
    if lowered.startswith("how long") or "how long" in lowered:
        return "how_long"
    if lowered.startswith("what"):
        if any(phrase in lowered for phrase in ("identity", "relationship status", "status")):
            return "attribute"
        if any(phrase in lowered for phrase in ("fields", "things", "items", "both")):
            return "list"
        return "what"
    return "other"


def infer_answer_type(answer: Any, question: str) -> str:
    text = str(answer).strip()
    lowered = text.lower()
    question_lowered = question.lower()
    if re.fullmatch(r"\d{4}", text):
        return "year"
    if any(word in lowered for word in DATE_HINT_WORDS) and any(char.isdigit() for char in lowered):
        return "date"
    if any(phrase in lowered for phrase in ("the week", "the sunday", "yesterday", "tomorrow", "last week", "next month")):
        return "relative_time"
    if any(word in lowered.split() for word in DURATION_WORDS):
        return "duration"
    if "," in text or ";" in text:
        return "list"
    if "relationship" in question_lowered or lowered in RELATIONSHIP_WORDS:
        return "relationship_status"
    if "identity" in question_lowered or any(word in lowered for word in IDENTITY_WORDS):
        return "identity"
    if question_lowered.startswith("where") or "which city" in question_lowered or "move from" in question_lowered:
        return "location"
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", text):
        return "person"
    if any(word in lowered for word in ("agency", "school", "store", "studio", "group", "company")):
        return "organization" if any(word in lowered for word in ("agency", "school", "group", "company")) else "event_object"
    if len(text.split()) <= 4:
        return "event_object"
    return "free_text"


def build_qa_profile(result: dict[str, Any]) -> dict[str, Any]:
    qa = result.get("qa", {}) if isinstance(result.get("qa"), dict) else {}
    question = str(qa.get("question", "")).strip()
    answer = qa.get("answer", "")
    return {
        "question_type": infer_question_type(question),
        "answer_type": infer_answer_type(answer, question),
        "question": question,
        "answer": answer,
    }


def model_accuracy_values(model_eval: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for model_item in model_eval.get("models", []):
        if not isinstance(model_item, dict):
            continue
        acc = model_item.get("accuracy")
        if isinstance(acc, (int, float)) and not isinstance(acc, bool):
            values.append(float(acc))
    return values


def has_model_disagreement(model_eval: dict[str, Any]) -> bool:
    values = model_accuracy_values(model_eval)
    return len(values) >= 2 and max(values) - min(values) >= 0.5


def options_look_obvious(result: dict[str, Any]) -> bool:
    qa = result.get("qa", {}) if isinstance(result.get("qa"), dict) else {}
    answer = str(qa.get("correct_answer_text", qa.get("answer", ""))).strip().lower()
    options = qa.get("options", [])
    if not isinstance(options, list) or not answer:
        return False
    normalized_options = [str(option).strip().lower() for option in options]
    if answer not in normalized_options:
        return True
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer))
    if not answer_tokens:
        return False
    overlaps = []
    for option in normalized_options:
        if option == answer:
            continue
        option_tokens = set(re.findall(r"[a-z0-9]+", option))
        overlaps.append(len(answer_tokens & option_tokens) / max(1, len(answer_tokens)))
    return sum(1 for value in overlaps if value >= 0.4) < 2


def evidence_is_short_or_direct(result: dict[str, Any]) -> bool:
    qa = result.get("qa", {}) if isinstance(result.get("qa"), dict) else {}
    answer = str(qa.get("answer", "")).strip().lower()
    evidence_ids = set(str(item) for item in qa.get("evidence", []) if item is not None)
    if not answer or not evidence_ids:
        return False
    for turn in result.get("final_dialogue", []):
        if not isinstance(turn, dict) or str(turn.get("dia_id")) not in evidence_ids:
            continue
        text = str(turn.get("text", "")).lower()
        if answer in text or len(text.split()) <= 18:
            return True
    return False


def choose_difficulty_ops(
    result: dict[str, Any],
    model_eval: dict[str, Any],
    qa_profile: dict[str, Any],
) -> dict[str, Any]:
    question_type = qa_profile.get("question_type")
    answer_type = qa_profile.get("answer_type")
    ops: list[str] = []
    reason = "options_are_too_obvious" if options_look_obvious(result) else "solver_accuracy_too_high"
    expansion_mode = "minimal"
    question_evolution = False

    if question_type == "when" or answer_type in {"date", "year", "relative_time", "duration"}:
        ops.extend(["option_confusion", "temporal_update"])
        question_evolution = True
    elif answer_type == "location":
        ops.extend(["option_confusion", "entity_confusion"])
    elif answer_type == "list":
        ops.extend(["option_confusion", "question_evolution"])
        question_evolution = True
    elif answer_type in {"identity", "relationship_status"}:
        ops.append("option_confusion")
        question_evolution = False
    else:
        ops.append("option_confusion")
        if question_type in {"what", "why", "attribute"}:
            ops.append("entity_confusion")

    if evidence_is_short_or_direct(result):
        reason = "answer_is_directly_visible"
        if answer_type not in {"identity", "relationship_status"}:
            ops.append("evidence_dispersion")
            question_evolution = True
            expansion_mode = "moderate"

    if options_look_obvious(result) and "option_confusion" not in ops:
        ops.insert(0, "option_confusion")
        expansion_mode = "minimal"

    if question_evolution:
        ops.append("question_evolution")

    deduped_ops = [operation for operation in HARD_DIFFICULTY_OPS if operation in set(ops)]
    return {
        "difficulty_reason": reason,
        "difficulty_ops": deduped_ops,
        "regenerate_options": True,
        "dialogue_expansion_mode": expansion_mode,
        "question_evolution": "question_evolution" in deduped_ops,
    }


def build_hard_policy_fields(*, added_turns_delta: int = 2) -> dict[str, Any]:
    return {
        "added_turns_delta": added_turns_delta,
        "difficulty_ops": [
            "option_confusion",
            "temporal_update",
            "entity_confusion",
        ],
        "min_difficulty_ops": 2,
        "answer_policy": "preserve_correct_answer",
        "question_policy": "allow_constraint_evolution",
        "option_policy": "regenerate_adversarial_distractors",
        "regenerate_options": True,
        "min_dialogue_grounded_distractors": 2,
        "required_distractor_types": [
            "outdated_answer",
            "similar_entity",
            "partial_answer",
        ],
    }


def build_policy_output_path(input_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return input_path.with_name(f"{input_path.stem}_challenger_policy_{timestamp}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert solver-panel accuracy from evaluate_model_difficulty.py into "
            "the next challenger generation policy."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an evaluated or frontier-labeled JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the policy JSON. If omitted, a timestamped sidecar file is generated.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new sidecar JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_challenger_policy(model_eval: dict[str, Any], result: dict[str, Any] | None = None) -> dict[str, Any]:
    acc = model_eval.get("average_accuracy")
    result = result or {}
    qa_profile = build_qa_profile(result)

    if acc is None:
        return {
            "next_action": "regenerate_or_validate",
            "difficulty_level": 1,
            "qa_profile": qa_profile,
            "target_skills": [
                "rerun_solver_panel",
                "check_generation_validity",
            ],
        }

    if has_model_disagreement(model_eval) and 0.35 <= float(acc) <= 0.65:
        return {
            "next_action": "keep_as_frontier",
            "difficulty_level": 2,
            "difficulty_reason": "disagreement_frontier",
            "qa_profile": qa_profile,
            "difficulty_ops": [],
            "regenerate_options": False,
            "dialogue_expansion_mode": "minimal",
            "question_evolution": False,
            "target_skills": [
                "preserve_current_difficulty",
                "inspect_model_disagreement",
            ],
        }

    if acc >= 0.8:
        targeted_fields = choose_difficulty_ops(result, model_eval, qa_profile)
        return {
            "next_action": "increase_difficulty",
            "difficulty_level": 3,
            "added_turns_delta": 0 if targeted_fields["dialogue_expansion_mode"] == "minimal" else 2,
            "min_difficulty_ops": 1,
            "answer_policy": "preserve_correct_answer",
            "question_policy": "allow_constraint_evolution" if targeted_fields["question_evolution"] else "preserve_question",
            "option_policy": "regenerate_adversarial_distractors",
            "min_dialogue_grounded_distractors": 2,
            "required_distractor_types": [
                "outdated_answer",
                "similar_entity",
                "partial_answer",
                "temporally_wrong_answer",
                "nearby_but_wrong_fact",
            ],
            "qa_profile": qa_profile,
            **targeted_fields,
            "target_skills": [
                "more_confusable_options",
                "targeted_answer_type_distractors",
                "answer_preserving_question_evolution",
            ],
        }

    if acc >= 0.6:
        targeted_fields = choose_difficulty_ops(result, model_eval, qa_profile)
        targeted_fields["dialogue_expansion_mode"] = "minimal"
        return {
            "next_action": "moderately_increase_difficulty",
            "difficulty_level": 2,
            "added_turns_delta": 0,
            "difficulty_ops": targeted_fields["difficulty_ops"],
            "min_difficulty_ops": 1,
            "answer_policy": "preserve_correct_answer",
            "option_policy": "regenerate_adversarial_distractors",
            "min_dialogue_grounded_distractors": 1,
            "qa_profile": qa_profile,
            **targeted_fields,
            "target_skills": [
                "more_confusable_options",
            ],
        }

    if acc >= 0.4:
        return {
            "next_action": "keep_as_frontier",
            "difficulty_level": 2,
            "difficulty_reason": "within_frontier_band",
            "qa_profile": qa_profile,
            "difficulty_ops": [],
            "regenerate_options": False,
            "dialogue_expansion_mode": "minimal",
            "question_evolution": False,
            "target_skills": [
                "preserve_current_difficulty",
            ],
        }

    if acc >= 0.2:
        return {
            "next_action": "slightly_reduce_or_validate",
            "difficulty_level": 1,
            "difficulty_reason": "possible_overhard_or_ambiguous",
            "qa_profile": qa_profile,
            "difficulty_ops": [],
            "regenerate_options": False,
            "dialogue_expansion_mode": "minimal",
            "question_evolution": False,
            "target_skills": [
                "reduce_ambiguity",
                "check_answer_uniqueness",
            ],
        }

    return {
        "next_action": "validate_as_risky",
        "difficulty_level": 0,
        "difficulty_reason": "risky_low_solver_accuracy",
        "qa_profile": qa_profile,
        "difficulty_ops": [],
        "regenerate_options": False,
        "dialogue_expansion_mode": "minimal",
        "question_evolution": False,
        "target_skills": [
            "check_invalid_question",
            "check_invalid_options",
            "check_evidence_support",
        ],
    }


def collect_result_items(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)], True
    if isinstance(payload, dict) and "model_eval" in payload:
        return [payload], False
    raise ValueError(
        "Input JSON must be either a single evaluated result object or a batch object with a 'results' list."
    )


def attach_policy(result: dict[str, Any]) -> dict[str, Any]:
    model_eval = result.get("model_eval")
    if not isinstance(model_eval, dict):
        policy = update_challenger_policy({"average_accuracy": None}, result)
    else:
        policy = update_challenger_policy(model_eval, result)
        model_eval["challenger_policy"] = policy
    result["challenger_policy"] = policy
    return policy


def build_policy_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = {action: 0 for action in NEXT_ACTIONS}
    difficulty_level_counts: dict[str, int] = {}
    difficulty_op_counts = {operation: 0 for operation in HARD_DIFFICULTY_OPS}
    regenerate_options_count = 0

    for result in results:
        policy = result.get("challenger_policy")
        if not isinstance(policy, dict):
            action = "regenerate_or_validate"
            difficulty_level = 1
            difficulty_ops = []
        else:
            action = str(policy.get("next_action", "regenerate_or_validate"))
            difficulty_level = policy.get("difficulty_level", 1)
            difficulty_ops = policy.get("difficulty_ops", [])

        if action not in action_counts:
            action = "regenerate_or_validate"
        action_counts[action] += 1

        level_key = str(difficulty_level)
        difficulty_level_counts[level_key] = difficulty_level_counts.get(level_key, 0) + 1
        if isinstance(policy, dict) and policy.get("regenerate_options") is True:
            regenerate_options_count += 1
        if isinstance(difficulty_ops, list):
            for operation in difficulty_ops:
                operation_key = str(operation)
                if operation_key in difficulty_op_counts:
                    difficulty_op_counts[operation_key] += 1

    return {
        "policy_case_count": len(results),
        "action_counts": action_counts,
        "difficulty_level_counts": difficulty_level_counts,
        "difficulty_op_counts": difficulty_op_counts,
        "regenerate_options_count": regenerate_options_count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def main() -> int:
    args = parse_args()
    if args.output and args.in_place:
        raise ValueError("Use either --output or --in-place, not both.")

    input_path = Path(args.input)
    payload = load_json(input_path)
    result_items, is_batch = collect_result_items(payload)

    for result in result_items:
        attach_policy(result)

    summary = build_policy_summary(result_items)
    if is_batch and isinstance(payload, dict):
        payload["challenger_policy_summary"] = summary
    elif isinstance(payload, dict):
        payload["challenger_policy_summary"] = summary

    if args.in_place:
        output_path = input_path
    else:
        output_path = Path(args.output) if args.output else build_policy_output_path(input_path)

    save_json(output_path, payload)
    print(f"Saved challenger policy result to: {output_path.resolve()}")
    print(f"Action counts: {summary['action_counts']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        raise
