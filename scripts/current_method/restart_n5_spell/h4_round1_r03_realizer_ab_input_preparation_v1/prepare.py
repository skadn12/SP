"""Prepare, but never send, the two R03 Realizer input variants.

The preparation deliberately bypasses the feedback-view renderer.  That
renderer is allowed to add the authoritative dialogue for Planner and QA, but
the A/B question here is specifically whether the Realizer receives that
dialogue.  Each mode therefore calls the existing Realizer builder directly
and saves the final one-message request that a later authorized call would
send.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_realizer_prompt,
)
from ..h4_round1_r03_feedback_view_v2.renderers import normalize_parent
from ..h4_round1_r03_feedback_view_v2.templates import EXPANSION_TASK_V2


VERSION = "h4_round1_r03_realizer_ab_input_preparation_v1"
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_r03_feedback_view_v2_single_root_debug_live_1"
)
DEFAULT_OUTPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_r03_realizer_ab_input_preparation_v1_live_3"
)
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = _sha_bytes(data)
    path.with_name(path.name + ".sha256").write_text(digest + "\n",
                                                     encoding="utf-8")
    return digest


def _write_json(path: Path, value: Any) -> str:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) +
            "\n").encode("utf-8")
    return _write_bytes(path, data)


def _write_text(path: Path, text: str) -> str:
    return _write_bytes(path, (text.rstrip("\n") + "\n").encode("utf-8"))


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True)


def _dialogue_texts(dialogue: Mapping[str, Any]) -> list[str]:
    return [str(row.get("text") or "") for row in dialogue.get("turns") or []]


def _dialogue_refs(dialogue: Mapping[str, Any]) -> list[str]:
    return [str(row.get("turn_ref") or row.get("turn_id") or "")
            for row in dialogue.get("turns") or []]


def _transcript_line_count(prompt: str, dialogue: Mapping[str, Any]) -> int:
    count = 0
    for row in dialogue.get("turns") or []:
        ref = str(row.get("turn_ref") or row.get("turn_id") or "")
        speaker = str(row.get("speaker") or "")
        text = str(row.get("text") or "")
        if f"{ref} | {speaker} | {text}" in prompt:
            count += 1
    return count


def _exact_source_text_hits(prompt: str, dialogue: Mapping[str, Any]) -> int:
    return sum(1 for text in _dialogue_texts(dialogue)
               if text and text in prompt)


def _mode_boundary(mode: str) -> str:
    if mode == "A_FULL_PARENT":
        return (
            "\n\nAB PREPARATION MODE: A_FULL_PARENT\n"
            "The complete Parent/R01/R02 transcript above is read-only history. "
            "Return only the newly generated append-only Child dialogue for the "
            "three frozen scene plans. Do not output, reproduce, or prepend the "
            "historical transcript.\n"
        )
    return (
        "\n\nAB PREPARATION MODE: B_TEXT_FREE\n"
        "No historical dialogue wording is provided in this mode. Return only "
        "new append-only Child dialogue for the three frozen scene plans. Use "
        "the text-free continuity shape only for form; do not reconstruct, "
        "quote, copy, or closely paraphrase omitted history.\n"
    )


def _shared_realizer_suffix(*, root_id: str, source_cluster_id: str,
                            source_prefix: str) -> str:
    identity = {
        "root_id": root_id,
        "source_cluster_id": source_cluster_id,
        "source_prefix": source_prefix,
    }
    return "\n\n".join([
        EXPANSION_TASK_V2,
        "【当前 Realizer 输入身份】\n" +
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True),
        "【既有输出合同】\nUse the existing Realizer JSON output contract. "
        "Return no QA, fact summary, bundle hash, or evidence index.",
    ])


def _render_mode(*, mode: str, parent: Mapping[str, Any], plan: Mapping[str, Any],
                 style_context: Mapping[str, Any], root_id: str,
                 source_cluster_id: str, source_prefix: str) -> dict[str, Any]:
    full = mode == "A_FULL_PARENT"
    base = build_realizer_prompt(
        parent=parent,
        plan=plan,
        style_excerpt=[],
        style_profile=(style_context.get("style_profile") if not full else None),
        continuity_shape=(style_context.get("continuity_tail") if not full else None),
        include_full_parent=full,
    )
    prompt = "\n\n".join([
        str(base["prompt"]),
        _shared_realizer_suffix(
            root_id=root_id, source_cluster_id=source_cluster_id,
            source_prefix=source_prefix),
        _mode_boundary(mode).strip(),
    ])
    return {
        "mode": mode,
        "prompt_version": base["prompt_version"],
        "builder_metadata": {
            key: base[key]
            for key in (
                "style_excerpt_turn_count", "continuity_shape_included",
                "style_profile_included", "full_parent_included",
            )
        },
        "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
    }


def _check_mode(*, rendered: Mapping[str, Any], dialogue: Mapping[str, Any],
                plan: Mapping[str, Any], style_context: Mapping[str, Any],
                root_id: str, source_cluster_id: str) -> dict[str, Any]:
    mode = str(rendered["mode"])
    prompt = str(rendered["prompt"])
    refs = _dialogue_refs(dialogue)
    transcript_count = _transcript_line_count(prompt, dialogue)
    exact_text_hits = _exact_source_text_hits(prompt, dialogue)
    plan_digest = _sha_bytes(_canonical(plan).encode("utf-8"))
    shape = style_context.get("continuity_tail") or {}
    shape_refs = [str(row.get("turn_ref") or "")
                  for row in shape.get("turn_shapes") or []]
    checks = {
        "one_user_message": rendered["messages"] == [
            {"role": "user", "content": prompt}],
        "plan_present": "FROZEN SCENE PLANS" in prompt,
        "plan_sha": plan_digest,
        "output_contract_present": 'exact top-level key scenes' in prompt,
        "expansion_task_v2_present_unchanged": EXPANSION_TASK_V2 in prompt,
        "root_id_present": root_id in prompt,
        "source_cluster_id_present": source_cluster_id in prompt,
        "mode_boundary_present": f"AB PREPARATION MODE: {mode}" in prompt,
        "history_ref_count": len(refs),
        "history_transcript_line_count": transcript_count,
        "exact_source_text_hits": exact_text_hits,
        "continuity_shape_refs": shape_refs,
        "continuity_shape_present": (
            "CONTINUITY SHAPE ONLY (previous wording omitted)" in prompt),
        "full_parent_marker_present": "FROZEN PARENT TRANSCRIPT" in prompt,
    }
    if mode == "A_FULL_PARENT":
        checks["mode_content_valid"] = (
            transcript_count == len(refs) and
            checks["full_parent_marker_present"] and
            not checks["continuity_shape_present"])
    else:
        checks["mode_content_valid"] = (
            transcript_count == 0 and
            not checks["full_parent_marker_present"] and
            checks["continuity_shape_present"] and
            len(shape_refs) > 0)
    return checks


def prepare(*, source_run: Path = DEFAULT_SOURCE_RUN,
            output_run: Path = DEFAULT_OUTPUT_RUN,
            tokenizer_path: Path = DEFAULT_TOKENIZER) -> dict[str, Any]:
    protected_paths = [
        source_run / "input_identity.json",
        source_run / "feedback_view.json",
        source_run / "execution_plan.json",
        source_run / "planner" / "parsed_plan.json",
        source_run / "realizer" / "style_context.json",
        source_run / "realizer" / "request_01.json",
        source_run / "realizer" / "parsed_01.json",
        source_run / "realizer" / "validation_01.json",
    ]
    protected_before = {str(path): _sha_file(path) for path in protected_paths}
    identity = _read_json(source_run / "input_identity.json")
    dialogue_path = Path(str(identity["input_dialogue_path"]))
    dialogue = _read_json(dialogue_path)
    plan = _read_json(source_run / "planner" / "parsed_plan.json")
    style_context = _read_json(source_run / "realizer" / "style_context.json")
    old_request = _read_json(source_run / "realizer" / "request_01.json")
    old_validation = _read_json(source_run / "realizer" / "validation_01.json")
    old_parsed = _read_json(source_run / "realizer" / "parsed_01.json")
    feedback_view = _read_json(source_run / "feedback_view.json")
    execution_plan = _read_json(source_run / "execution_plan.json")
    old_parent = normalize_parent(dialogue)

    # Re-run the existing validator against the quarantined historical output.
    reproduced = normalize_realizer_output(
        old_parsed, plan=plan, parent=old_parent, round_index=3)
    original_rejection = {
        "stored_valid": old_validation.get("valid"),
        "stored_errors": old_validation.get("errors") or [],
        "reproduced_valid": reproduced.get("valid"),
        "reproduced_errors": reproduced.get("errors") or [],
        "exact_error_sets_match": sorted(old_validation.get("errors") or []) ==
        sorted(reproduced.get("errors") or []),
        "rejected_by_original_validator": reproduced.get("valid") is False,
        "parent_exact_copy_error_count": sum(
            "parent_exact_copy" in str(error)
            for error in reproduced.get("errors") or []),
    }
    if not original_rejection["rejected_by_original_validator"]:
        raise RuntimeError("historical_quarantine_no_longer_rejected")

    parent = old_parent
    root_id = str(identity["root_id"])
    source_cluster_id = str(feedback_view["source_cluster_id"])
    source_prefix = str(identity.get("input_manifest_row", {}).get(
        "source_prefix") or "")
    rendered_a = _render_mode(
        mode="A_FULL_PARENT", parent=parent, plan=plan,
        style_context=style_context, root_id=root_id,
        source_cluster_id=source_cluster_id, source_prefix=source_prefix)
    rendered_b = _render_mode(
        mode="B_TEXT_FREE", parent=parent, plan=plan,
        style_context=style_context, root_id=root_id,
        source_cluster_id=source_cluster_id, source_prefix=source_prefix)
    checks_a = _check_mode(rendered=rendered_a, dialogue=dialogue,
                           plan=plan, style_context=style_context,
                           root_id=root_id,
                           source_cluster_id=source_cluster_id)
    checks_b = _check_mode(rendered=rendered_b, dialogue=dialogue,
                           plan=plan, style_context=style_context,
                           root_id=root_id,
                           source_cluster_id=source_cluster_id)
    tokenizer = _load_tokenizer(tokenizer_path)
    for rendered, checks in ((rendered_a, checks_a), (rendered_b, checks_b)):
        prompt_tokens = len(tokenizer.encode(
            str(rendered["prompt"]), add_special_tokens=False))
        checks["prompt_tokens_local"] = prompt_tokens
        checks["output_token_reserve"] = int(old_request["max_tokens"])
        checks["context_ceiling"] = int(old_request["context_ceiling"])
        checks["prompt_plus_output_reserve"] = (
            prompt_tokens + int(old_request["max_tokens"]))
        checks["within_context_ceiling"] = (
            checks["prompt_plus_output_reserve"] <=
            int(old_request["context_ceiling"]))
    if (not checks_a["mode_content_valid"] or
            not checks_b["mode_content_valid"] or
            not checks_a["within_context_ceiling"] or
            not checks_b["within_context_ceiling"]):
        raise RuntimeError("ab_render_content_check_failed")
    if checks_a["plan_sha"] != checks_b["plan_sha"]:
        raise RuntimeError("ab_plan_drift")

    if output_run.exists() and any(output_run.iterdir()):
        raise RuntimeError(f"output_run_not_empty:{output_run}")
    output_run.mkdir(parents=True, exist_ok=True)
    request_common = {
        key: old_request.get(key)
        for key in (
            "endpoint", "model", "temperature", "max_tokens",
            "context_ceiling", "response_format", "role",
            "automatic_retry_allowed", "provider_fallback_allowed",
        )
    }
    for mode, rendered, checks in (
        ("A_FULL_PARENT", rendered_a, checks_a),
        ("B_TEXT_FREE", rendered_b, checks_b),
    ):
        folder = output_run / ("a_full_parent" if mode == "A_FULL_PARENT"
                               else "b_text_free")
        _write_json(folder / "messages.json", rendered["messages"])
        _write_text(folder / "prompt.txt", rendered["prompt"])
        _write_json(folder / "request.json", {
            **request_common,
            "preparation_version": VERSION,
            "mode": mode,
            "offline_only": True,
            "dispatch_authorized": False,
            "planner_rerun_allowed": False,
            "automatic_retry_allowed": False,
            "planned_realizer_calls_after_separate_authorization": 1,
            "messages": rendered["messages"],
            "prompt_sha256": _sha_bytes(rendered["prompt"].encode("utf-8")),
            "prompt_tokens_local": checks["prompt_tokens_local"],
        })
        _write_json(folder / "prompt_metadata.json", {
            "mode": mode,
            "prompt_version": rendered["prompt_version"],
            "builder_metadata": rendered["builder_metadata"],
            "checks": checks,
            "prompt_sha256": _sha_bytes(rendered["prompt"].encode("utf-8")),
            "message_count": len(rendered["messages"]),
        })

    source_identity = {
        "root_id": identity["root_id"],
        "branch_id": execution_plan["branch_id"],
        "source_cluster_id": source_cluster_id,
        "source_run": str(source_run),
        "dialogue_path": str(dialogue_path),
        "dialogue_sha256": _sha_file(dialogue_path),
        "dialogue_turn_count": len(dialogue.get("turns") or []),
        "dialogue_complete_sha256": dialogue.get("complete_dialogue_sha256"),
        "plan_path": str(source_run / "planner" / "parsed_plan.json"),
        "plan_sha256": _sha_file(source_run / "planner" / "parsed_plan.json"),
        "style_context_path": str(source_run / "realizer" / "style_context.json"),
        "style_context_sha256": _sha_file(source_run / "realizer" / "style_context.json"),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_config_sha256": _sha_file(tokenizer_path /
                                                "tokenizer_config.json"),
        "old_realizer_request_sha256": _sha_file(source_run / "realizer" / "request_01.json"),
        "old_request_parameters": {
            key: old_request.get(key)
            for key in ("model", "temperature", "max_tokens", "context_ceiling",
                        "response_format", "role")
        },
        "validator_path": str(Path(normalize_realizer_output.__code__.co_filename)),
        "validator_sha256": _sha_file(Path(
            normalize_realizer_output.__code__.co_filename)),
    }
    _write_json(output_run / "source_identity.json", source_identity)
    protected_after = {str(path): _sha_file(path) for path in protected_paths}
    protected_source_audit = {
        "files": [{
            "path": path,
            "before_sha256": protected_before[path],
            "after_sha256": protected_after[path],
            "unchanged": protected_before[path] == protected_after[path],
        } for path in sorted(protected_before)],
        "all_unchanged": protected_before == protected_after,
    }
    if not protected_source_audit["all_unchanged"]:
        raise RuntimeError("protected_source_modified")
    _write_json(output_run / "protected_source_sha_audit.json",
                protected_source_audit)
    report = {
        "preparation_version": VERSION,
        "status": "OFFLINE_AB_INPUT_PREPARATION_PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "external_calls": {
            "planner": 0, "realizer": 0, "memorax_search": 0,
            "memorax_status": 0, "memorax_add": 0, "solver": 0,
            "training": 0,
        },
        "root_id": identity["root_id"],
        "source_prefix": identity.get("input_manifest_row", {}).get("source_prefix"),
        "same_frozen_inputs": {
            "dialogue_sha256": source_identity["dialogue_sha256"],
            "plan_sha256": source_identity["plan_sha256"],
            "plan_sha_a": checks_a["plan_sha"],
            "plan_sha_b": checks_b["plan_sha"],
            "same_plan": checks_a["plan_sha"] == checks_b["plan_sha"],
            "same_model": True,
            "same_temperature": True,
            "same_max_tokens": True,
            "same_response_format": True,
            "same_output_contract": (
                checks_a["output_contract_present"] and
                checks_b["output_contract_present"]),
            "same_expansion_task_v2": (
                checks_a["expansion_task_v2_present_unchanged"] and
                checks_b["expansion_task_v2_present_unchanged"]),
            "same_copy_validator": True,
        },
        "modes": {
            "A_FULL_PARENT": checks_a,
            "B_TEXT_FREE": checks_b,
        },
        "original_quarantine_regression": original_rejection,
        "protected_source_sha_audit": protected_source_audit,
        "output_files": {
            "a_messages": str(output_run / "a_full_parent" / "messages.json"),
            "b_messages": str(output_run / "b_text_free" / "messages.json"),
            "a_request": str(output_run / "a_full_parent" / "request.json"),
            "b_request": str(output_run / "b_text_free" / "request.json"),
        },
    }
    _write_json(output_run / "ab_input_preparation_report.json", report)
    md = "\n".join([
        "# R03 Realizer A/B Input Preparation",
        "",
        f"- Status: `{report['status']}`",
        f"- Root: `{identity['root_id']}`",
        "- External calls: `0`",
        f"- Frozen dialogue turns: `{len(dialogue.get('turns') or [])}`",
        "",
        "## A: Full Parent",
        f"- Transcript lines: `{checks_a['history_transcript_line_count']}`",
        f"- Full Parent marker: `{checks_a['full_parent_marker_present']}`",
        f"- Prompt SHA: `{_sha_bytes(rendered_a['prompt'].encode('utf-8'))}`",
        f"- Prompt + output reserve: `"
        f"{checks_a['prompt_tokens_local']} + {checks_a['output_token_reserve']} "
        f"<= {checks_a['context_ceiling']}`",
        "",
        "## B: Text-Free",
        f"- Transcript lines: `{checks_b['history_transcript_line_count']}`",
        f"- Continuity shape: `{checks_b['continuity_shape_present']}`",
        f"- Full Parent marker: `{checks_b['full_parent_marker_present']}`",
        f"- Prompt SHA: `{_sha_bytes(rendered_b['prompt'].encode('utf-8'))}`",
        f"- Prompt + output reserve: `"
        f"{checks_b['prompt_tokens_local']} + {checks_b['output_token_reserve']} "
        f"<= {checks_b['context_ceiling']}`",
        "",
        "## Regression",
        f"- Original quarantined output rejected by unchanged validator: `"
        f"{original_rejection['rejected_by_original_validator']}`",
        f"- Parent exact-copy errors reproduced: `"
        f"{original_rejection['parent_exact_copy_error_count']}`",
        "",
        "The two complete messages are saved under `a_full_parent/` and "
        "`b_text_free/`. Their request envelopes are explicitly offline and "
        "not dispatch-authorized. No model or remote-memory request was made.",
    ])
    _write_text(output_run / "ab_input_preparation_report.md", md)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT_RUN)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    report = prepare(source_run=args.source_run, output_run=args.output_run,
                     tokenizer_path=args.tokenizer)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
