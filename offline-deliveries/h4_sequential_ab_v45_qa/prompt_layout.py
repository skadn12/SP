"""Position-only task tails with preserved frozen-feedback and final HTTP bindings."""
from copy import deepcopy
import hashlib
import json
import re

import prompt_rules


CONTRACT = "v45_full_data_frozen_feedback_then_current_task_v1"
START = "[CURRENT_QA_TASK_V45]"
END = "[/CURRENT_QA_TASK_V45]"
KINDS = ("multi_select", "ordering", "ordering_review")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def task_tail(kind):
    require(kind in KINDS, "unknown_task_tail_kind")
    guidance = (prompt_rules.review_guidance() if kind == "ordering_review"
                else prompt_rules.generation_guidance(kind))
    return "\n\n" + START + guidance + END


def _user_text(payload):
    messages = payload.get("messages")
    require(isinstance(messages, list) and len(messages) == 1
            and isinstance(messages[0], dict) and messages[0].get("role") == "user"
            and isinstance(messages[0].get("content"), str), "task_tail_single_user_message_required")
    return messages[0]["content"]


def split_task_tail(text, *, expected_kind=None):
    require(isinstance(text, str) and text.count(START) == text.count(END) == 1,
            "task_tail_boundaries_invalid")
    kinds = KINDS if expected_kind is None else (expected_kind,)
    matches = [(kind, task_tail(kind)) for kind in kinds if text.endswith(task_tail(kind))]
    require(len(matches) == 1, "task_tail_missing_changed_or_wrong_kind")
    kind, tail = matches[0]
    return text[:-len(tail)], kind, tail


def append_task_tail(payload, feedback_receipt, kind):
    """Keep the legacy receipt bound to data+feedback; bind the final request too."""
    intermediate = _user_text(payload)
    require(START not in intermediate and END not in intermediate, "task_tail_already_present")
    require(feedback_receipt.get("feedback_appended") is True
            and feedback_receipt.get("request_user_text_sha256") == digest(intermediate.encode()),
            "intermediate_feedback_receipt_invalid")
    final, receipt = deepcopy(payload), deepcopy(feedback_receipt)
    tail = task_tail(kind)
    final["messages"][0]["content"] = intermediate + tail
    receipt["task_tail_receipt"] = {
        "contract": CONTRACT, "kind": kind,
        "feedback_receipt_scope": "intermediate_full_data_plus_frozen_feedback",
        "intermediate_user_text_sha256": digest(intermediate.encode()),
        "intermediate_payload_sha256": digest(payload),
        "task_tail_sha256": digest(tail.encode()),
        "final_user_text_sha256": digest(_user_text(final).encode()),
        "final_payload_sha256": digest(final),
    }
    return final, receipt


def verify_request(payload, receipt, suffix, root, target, source_lock_sha256, *, expected_kind=None):
    """Validate both request layers without weakening the legacy feedback receipt."""
    text = _user_text(payload)
    intermediate, kind, tail = split_task_tail(text, expected_kind=expected_kind)
    require(isinstance(suffix, str) and bool(suffix)
            and intermediate.endswith(suffix) and intermediate.count(suffix) == 1,
            "actual_previous_round_feedback_position_invalid")
    require(receipt.get("feedback_appended") is True and receipt.get("role") == "writer"
            and receipt.get("root_id") == root and receipt.get("target_solver_model") == target
            and receipt.get("suffix_sha256") == digest(suffix.encode())
            and receipt.get("request_user_text_sha256") == digest(intermediate.encode())
            and receipt.get("source_lock_sha256") == source_lock_sha256,
            "actual_previous_round_feedback_binding_invalid")
    intermediate_payload = deepcopy(payload)
    intermediate_payload["messages"][0]["content"] = intermediate
    expected = {"contract": CONTRACT, "kind": kind,
        "feedback_receipt_scope": "intermediate_full_data_plus_frozen_feedback",
        "intermediate_user_text_sha256": digest(intermediate.encode()),
        "intermediate_payload_sha256": digest(intermediate_payload),
        "task_tail_sha256": digest(tail.encode()),
        "final_user_text_sha256": digest(text.encode()),
        "final_payload_sha256": digest(payload)}
    require(receipt.get("task_tail_receipt") == expected, "task_tail_receipt_binding_invalid")
    return expected


def review_feedback_suffix(suffix):
    """Parse only the new layout; semantic and quotation checks stay in the gate."""
    intermediate, _, _ = split_task_tail(suffix, expected_kind="ordering_review")
    match = re.fullmatch(
        r"\[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4\]\s*(.*?)\s*\[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4\]",
        intermediate.strip(), re.S)
    require(match is not None and isinstance(json.loads(match.group(1)), dict),
            "review_unrecognized_prompt_suffix")
    return intermediate
