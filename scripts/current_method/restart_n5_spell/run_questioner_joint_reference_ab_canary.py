#!/usr/bin/env python3
"""Run the isolated Questioner joint-reference A/B canary."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from .backends import BackendResponse
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from .ledger import RequestLedger
from .live_backends import ChallengerAPIConfig, load_challenger_config
from .questioner_joint_reference_ab import (
    BASE_PROMPT_VERSION,
    BLIND_SEED,
    CANARY_VERSION,
    FORMAT_ONLY_RETRY_SUFFIX,
    FORMAT_RETRY_VERSION,
    REFERENCE_VERSION,
    SCHEMA_VERSION,
    SYSTEM_MESSAGE,
    PaidRequestBudget,
    alias_lookup,
    build_blind_review,
    build_group_prompts,
    build_turn_aliases,
    candidate_summary,
    format_retry_suffix_sha256,
    load_full_dialogue,
    normalize_group_result,
    prompt_diff,
    render_blind_review_markdown,
    request_bodies_differ_only_by_reference,
    schema_sha256,
    validate_response,
)
from .run_canary import (
    create_source_snapshot,
    project_diff_sha256,
    write_post_run_provenance,
)


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_DIALOGUE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_spell_lineage_three_stage_full_dialogue_canary_v1/"
    "frozen/full_dialogue_challenger_input.json"
)
DEFAULT_REFERENCE = MODULE_ROOT / "prompts/questioner_joint_reference_v1.md"
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_questioner_joint_reference_ab_canary_v1"
)
DEFAULT_API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
TOP_P = 1.0
API_SEED = 20260718
MAX_TOKENS = 4096
API_CONTEXT_LIMIT = 32768


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _request_body(api: ChallengerAPIConfig, user_prompt: str) -> dict[str, Any]:
    return {
        "model": api.model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": api.temperature,
        "top_p": TOP_P,
        "seed": API_SEED,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def _live_complete(api: ChallengerAPIConfig, body: Mapping[str, Any]) -> BackendResponse:
    request = urllib.request.Request(
        api.base_url + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=api.timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    return BackendResponse(
        raw_text=str(choice["message"]["content"]),
        model_identifier=api.model,
        metadata={
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
        },
    )


def _token_count(tokenizer: Any, user_prompt: str) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return len(encoded)


def _assert_inputs(dialogue: Path, reference: Path, run_root: Path) -> None:
    for path in (dialogue, reference):
        if not path.is_file():
            raise FileNotFoundError(f"fixed canary input missing: {path}")
    if run_root.exists():
        raise FileExistsError(f"A/B canary run root already exists: {run_root}")


def preflight(
    *, dialogue: Path, reference: Path, run_root: Path,
    api_config: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    _assert_inputs(dialogue, reference, run_root)
    payload, turns = load_full_dialogue(dialogue)
    alias_map, aliased = build_turn_aliases(turns)
    aliases = alias_lookup(alias_map, aliased)
    reference_text = reference.read_text(encoding="utf-8")
    group_a, group_b, proof = build_group_prompts(aliased, reference_text)
    api = load_challenger_config(api_config)
    if api.model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"unexpected fixed Challenger model: {api.model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt_tokens = {
        "group_a": _token_count(tokenizer, group_a),
        "group_b": _token_count(tokenizer, group_b),
    }
    totals = {group: count + MAX_TOKENS for group, count in prompt_tokens.items()}
    if any(total > API_CONTEXT_LIMIT for total in totals.values()):
        raise RuntimeError(f"A/B prompt preflight exceeds context: {totals}")
    body_a = _request_body(api, group_a)
    body_b = _request_body(api, group_b)
    if not request_bodies_differ_only_by_reference(body_a, body_b, reference_text):
        raise RuntimeError("A/B request bodies differ outside the reference block")
    return {
        "dialogue_payload": payload,
        "turns": turns,
        "alias_map": alias_map,
        "aliased_dialogue": aliased,
        "aliases": aliases,
        "reference_text": reference_text,
        "group_a_prompt": group_a,
        "group_b_prompt": group_b,
        "prompt_difference_proof": proof,
        "prompt_diff": prompt_diff(group_a, group_b),
        "api": api,
        "request_bodies": {"A": body_a, "B": body_b},
        "public": {
            "parent_turn_count": alias_map["parent_turn_count"],
            "child_turn_count": alias_map["child_turn_count"],
            "total_turn_count": alias_map["total_turn_count"],
            "source_dialogue_sha256": sha256_file(dialogue),
            "reference_sha256": sha256_file(reference),
            "model": api.model,
            "temperature": api.temperature,
            "top_p": TOP_P,
            "seed": API_SEED,
            "max_tokens": MAX_TOKENS,
            "system_message_sha256": sha256_bytes(SYSTEM_MESSAGE.encode("utf-8")),
            "prompt_tokens": prompt_tokens,
            "prompt_plus_max_tokens": totals,
            "context_limit": API_CONTEXT_LIMIT,
            "prompt_difference_proof": proof,
            "request_bodies_reference_only_difference": True,
        },
    }


def _source_manifest(
    *, run_root: Path, snapshot: Mapping[str, Any], pre: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "run_id": run_root.name,
        "mode": CANARY_VERSION,
        "project_git_commit": _git("rev-parse", "HEAD"),
        "project_diff_sha256": project_diff_sha256(PROJECT),
        **dict(snapshot),
        "orchestrator_source_sha256": sha256_file(MODULE_ROOT / "orchestrator.py"),
        "validation_source_sha256": sha256_file(MODULE_ROOT / "validation.py"),
        "ledger_source_sha256": sha256_file(MODULE_ROOT / "ledger.py"),
        "run_canary_source_sha256": sha256_file(MODULE_ROOT / "run_canary.py"),
        "mcq_parser_source_sha256": sha256_file(MODULE_ROOT / "core.py"),
        "questioner_prompt_builder_source_sha256": sha256_file(MODULE_ROOT / "prompts.py"),
        "questioner_contract_source_sha256": sha256_file(MODULE_ROOT / "questioner_contract.py"),
        "live_backend_source_sha256": sha256_file(MODULE_ROOT / "live_backends.py"),
        "joint_reference_ab_source_sha256": sha256_file(
            MODULE_ROOT / "questioner_joint_reference_ab.py"
        ),
        "joint_reference_ab_runner_source_sha256": sha256_file(Path(__file__)),
        "canary_version": CANARY_VERSION,
        "base_prompt_version": BASE_PROMPT_VERSION,
        "reference_version": REFERENCE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha256(),
        "format_retry_version": FORMAT_RETRY_VERSION,
        "format_retry_suffix_sha256": format_retry_suffix_sha256(),
        "blind_review_seed": BLIND_SEED,
        "source_dialogue_path": str(DEFAULT_DIALOGUE),
        "source_dialogue_sha256": pre["public"]["source_dialogue_sha256"],
        "reference_path": str(DEFAULT_REFERENCE),
        "reference_sha256": pre["public"]["reference_sha256"],
        "group_a_prompt_sha256": pre["prompt_difference_proof"]["group_a_prompt_sha256"],
        "group_b_prompt_sha256": pre["prompt_difference_proof"]["group_b_prompt_sha256"],
        "prompt_difference_proof": pre["prompt_difference_proof"],
        "challenger": {
            "model": pre["public"]["model"],
            "temperature": pre["public"]["temperature"],
            "top_p": TOP_P,
            "seed": API_SEED,
            "max_tokens": MAX_TOKENS,
            "response_format": "json_object",
            "system_message_sha256": pre["public"]["system_message_sha256"],
        },
        "budget": {
            "group_a_primary": 1,
            "group_a_format_retry_max": 1,
            "group_b_primary": 1,
            "group_b_format_retry_max": 1,
            "maximum_paid_api_requests": 4,
            "realizer": 0,
            "solver": 0,
            "verifier": 0,
            "training": 0,
            "semantic_retry": 0,
        },
    }


def _invoke_attempt(
    *, group: str, attempt_label: str, user_prompt: str,
    api: ChallengerAPIConfig, ledger: RequestLedger,
    budget: PaidRequestBudget, run_root: Path,
    reference_used: bool,
) -> tuple[BackendResponse, dict[str, Any]]:
    request_id = f"group-{group.lower()}-{attempt_label.replace('_', '-')}"
    body = _request_body(api, user_prompt)
    request_path = run_root / f"raw/group_{group.lower()}_{attempt_label}_request.json"
    response_path = run_root / f"raw/group_{group.lower()}_{attempt_label}_response.txt"
    atomic_json(request_path, body)
    budget.reserve()
    prompt_sha = sha256_bytes(user_prompt.encode("utf-8"))
    response = ledger.invoke(
        request_id=request_id,
        intent="questioner_joint_reference_ab",
        attempt=1 if attempt_label == "primary" else 2,
        prompt_sha256=prompt_sha,
        call=lambda: _live_complete(api, body),
        budget_class="paid_api",
        budget_limit=4,
        total_budget_limit=4,
        request_metadata={
            "group": group,
            "attempt_label": attempt_label,
            "reference_used": reference_used,
            "semantic_prompt_sha256": prompt_sha,
            "retry_mode": "format_only" if attempt_label != "primary" else None,
            "format_retry_suffix_sha256": (
                format_retry_suffix_sha256() if attempt_label != "primary" else None
            ),
        },
        backend_identity={
            "backend": "openai_compatible_json_object",
            "model_identifier": api.model,
            "base_url": api.base_url,
            "temperature": api.temperature,
            "top_p": TOP_P,
            "seed": API_SEED,
            "max_tokens": MAX_TOKENS,
            "response_format": "json_object",
            "system_message_sha256": sha256_bytes(SYSTEM_MESSAGE.encode("utf-8")),
        },
    )
    atomic_text(response_path, response.raw_text)
    return response, body


def _run_group(
    *, group: str, semantic_prompt: str,
    api: ChallengerAPIConfig, ledger: RequestLedger,
    budget: PaidRequestBudget, run_root: Path,
    aliases: Mapping[str, Mapping[str, Any]], reference_used: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary, _ = _invoke_attempt(
        group=group, attempt_label="primary", user_prompt=semantic_prompt,
        api=api, ledger=ledger, budget=budget, run_root=run_root,
        reference_used=reference_used,
    )
    primary_validation = validate_response(primary.raw_text, aliases)
    attempts = [{
        "attempt_label": "primary",
        "parse_status": primary_validation["parse_status"],
        "format_errors": primary_validation["format_errors"],
        "alias_errors": primary_validation["alias_errors"],
        "response_metadata": primary.metadata,
    }]
    final_validation = primary_validation
    attempt_used = "primary"
    retry_triggered = bool(primary_validation["retry_eligible"])
    if retry_triggered:
        retry_prompt = semantic_prompt + "\n" + FORMAT_ONLY_RETRY_SUFFIX
        retry, _ = _invoke_attempt(
            group=group, attempt_label="format_retry", user_prompt=retry_prompt,
            api=api, ledger=ledger, budget=budget, run_root=run_root,
            reference_used=reference_used,
        )
        final_validation = validate_response(retry.raw_text, aliases)
        attempt_used = "format_retry"
        attempts.append({
            "attempt_label": "format_retry",
            "parse_status": final_validation["parse_status"],
            "format_errors": final_validation["format_errors"],
            "alias_errors": final_validation["alias_errors"],
            "response_metadata": retry.metadata,
        })
    normalized = normalize_group_result(
        group, final_validation, aliases, attempt_used
    )
    final_decision = (
        "normalized" if final_validation["parse_status"] in {"valid", "alias_invalid"}
        else "questioner_format_invalid_exhausted"
    )
    ledger.record_decision(
        f"group-{group.lower()}-{attempt_used.replace('_', '-')}", final_decision
    )
    return normalized, {
        "group": group,
        "reference_used": reference_used,
        "retry_triggered": retry_triggered,
        "attempts": attempts,
        "final_parse_status": final_validation["parse_status"],
        "final_decision": final_decision,
    }


def _usage(ledger: RequestLedger) -> dict[str, Any]:
    states = sorted(ledger.completed(), key=lambda row: row["request_id"])
    prompt = completion = total = 0
    rows = []
    for state in states:
        usage = (state.get("response_metadata") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
        rows.append({
            "request_id": state["request_id"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": (state.get("response_metadata") or {}).get("finish_reason"),
        })
    return {
        "paid_api_requests": len(states),
        "primary_requests": sum("primary" in row["request_id"] for row in states),
        "format_only_retries": sum("format-retry" in row["request_id"] for row in states),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "requests": rows,
        "questioner_requests": len(states),
        "realizer_generations": 0,
        "solver_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Questioner Joint-Reference A/B Canary",
        "",
        "本报告只记录格式、alias 和计数结果；在人工盲审前不判断 reference 是否有效。",
        "",
        f"- A/B prompt 仅 reference 不同：`{str(report['prompt_diff_reference_only']).lower()}`",
        f"- 付费 API 请求：`{report['api_usage']['paid_api_requests']}`",
        "",
        "| Group | valid | no_valid_candidate | invalid | retry | reasoning types |",
        "|---|---:|---:|---:|---|---|",
    ]
    for group in ("A", "B"):
        summary = report["groups"][group]["summary"]
        request = report["groups"][group]["request"]
        lines.append(
            f"| {group} | {summary['valid']} | {summary['no_valid_candidate']} | "
            f"{summary['invalid_local_or_format']} | "
            f"{str(request['retry_triggered']).lower()} | "
            f"`{json.dumps(summary['reasoning_type_distribution'], ensure_ascii=False)}` |"
        )
    lines.extend([
        "", "盲审包已打乱；请先审阅 packet，再查看 key。", "",
    ])
    return "\n".join(lines)


def _post_provenance(
    run_root: Path, manifest: dict[str, Any], reference_text: str
) -> dict[str, Any]:
    base = write_post_run_provenance(run_root, PROJECT, manifest)
    custom = {
        "joint_reference_ab_source_sha_match": sha256_file(
            MODULE_ROOT / "questioner_joint_reference_ab.py"
        ) == manifest["joint_reference_ab_source_sha256"],
        "joint_reference_ab_runner_source_sha_match": sha256_file(Path(__file__))
        == manifest["joint_reference_ab_runner_source_sha256"],
        "source_dialogue_sha_match": sha256_file(DEFAULT_DIALOGUE)
        == manifest["source_dialogue_sha256"],
        "frozen_dialogue_sha_match": sha256_file(
            run_root / "frozen/source_dialogue_snapshot.json"
        ) == manifest["source_dialogue_sha256"],
        "source_reference_sha_match": sha256_file(DEFAULT_REFERENCE)
        == manifest["reference_sha256"],
        "frozen_reference_sha_match": sha256_file(
            run_root / "frozen/questioner_joint_reference_v1.md"
        ) == manifest["reference_sha256"],
        "group_a_prompt_sha_match": sha256_file(run_root / "prompts/group_a_prompt.txt")
        == manifest["group_a_prompt_sha256"],
        "group_b_prompt_sha_match": sha256_file(run_root / "prompts/group_b_prompt.txt")
        == manifest["group_b_prompt_sha256"],
        "prompt_difference_reference_only": request_bodies_differ_only_by_reference(
            json.loads((run_root / "raw/group_a_primary_request.json").read_text()),
            json.loads((run_root / "raw/group_b_primary_request.json").read_text()),
            reference_text,
        ),
    }
    report = {**base, **custom}
    boolean_checks = [
        value for key, value in report.items()
        if key.endswith("_match") or key == "prompt_difference_reference_only"
    ]
    report["authoritative_release_provenance"] = all(boolean_checks)
    report["provenance_mixed"] = not report["authoritative_release_provenance"]
    atomic_json(run_root / "reports/post_run_provenance.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Questioner joint-reference A/B canary")
    parser.add_argument("--dialogue", type=Path, default=DEFAULT_DIALOGUE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    pre = preflight(
        dialogue=args.dialogue.resolve(), reference=args.reference.resolve(),
        run_root=args.run_root.resolve(), api_config=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
    )
    if args.preflight_only:
        print(json.dumps(pre["public"], ensure_ascii=False, indent=2))
        return 0

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    atomic_json(run_root / "frozen/source_dialogue_snapshot.json", pre["dialogue_payload"])
    atomic_json(run_root / "frozen/source_dialogue_sha256.json", {
        "source_path": str(args.dialogue.resolve()),
        "source_file_sha256": pre["public"]["source_dialogue_sha256"],
    })
    atomic_text(
        run_root / "frozen/questioner_joint_reference_v1.md", pre["reference_text"]
    )
    atomic_text(
        run_root / "frozen/questioner_joint_reference_v1.sha256",
        pre["public"]["reference_sha256"] + "\n",
    )
    atomic_json(run_root / "frozen/turn_alias_map.json", pre["alias_map"])
    atomic_json(run_root / "frozen/full_dialogue_with_aliases.json", pre["aliased_dialogue"])
    atomic_text(run_root / "prompts/group_a_prompt.txt", pre["group_a_prompt"])
    atomic_text(run_root / "prompts/group_b_prompt.txt", pre["group_b_prompt"])
    atomic_text(run_root / "prompts/prompt_diff.txt", pre["prompt_diff"])

    snapshot = create_source_snapshot(PROJECT, run_root, False)
    manifest = _source_manifest(run_root=run_root, snapshot=snapshot, pre=pre)
    atomic_json(run_root / "config/run_manifest.json", manifest)
    atomic_json(run_root / "config/resolved_canary_config.json", pre["public"])
    ledger = RequestLedger(
        run_root, sha256_file(run_root / "config/run_manifest.json"), "live"
    )
    budget = PaidRequestBudget(4)

    group_results: dict[str, dict[str, Any]] = {}
    request_reports: dict[str, dict[str, Any]] = {}
    for group in ("A", "B"):
        normalized, request_report = _run_group(
            group=group,
            semantic_prompt=pre[f"group_{group.lower()}_prompt"],
            api=pre["api"], ledger=ledger, budget=budget, run_root=run_root,
            aliases=pre["aliases"], reference_used=group == "B",
        )
        group_results[group] = normalized
        request_reports[group] = request_report
        atomic_json(run_root / f"normalized/group_{group.lower()}_candidates.json", normalized)

    all_candidates = {
        "canary_version": CANARY_VERSION,
        "groups": [group_results["A"], group_results["B"]],
    }
    atomic_json(run_root / "normalized/all_candidates.json", all_candidates)
    packet, key = build_blind_review([group_results["A"], group_results["B"]])
    atomic_json(run_root / "review/blind_review_packet.json", packet)
    atomic_text(
        run_root / "review/blind_review_packet.md", render_blind_review_markdown(packet)
    )
    atomic_json(run_root / "review/blind_review_key.json", key)

    usage = _usage(ledger)
    atomic_json(run_root / "reports/api_usage.json", usage)
    summaries = {group: candidate_summary(group_results[group]) for group in ("A", "B")}
    report = {
        "canary_version": CANARY_VERSION,
        "semantic_conclusion_deferred_until_blind_review": True,
        "prompt_diff_reference_only": pre["public"][
            "request_bodies_reference_only_difference"
        ],
        "input_attestation": {
            key: pre["public"][key] for key in (
                "parent_turn_count", "child_turn_count", "total_turn_count",
                "source_dialogue_sha256", "reference_sha256",
            )
        },
        "groups": {
            group: {"summary": summaries[group], "request": request_reports[group]}
            for group in ("A", "B")
        },
        "api_usage": usage,
        "blind_review": {
            "seed": BLIND_SEED,
            "packet": "review/blind_review_packet.json",
            "markdown": "review/blind_review_packet.md",
            "key": "review/blind_review_key.json",
        },
        "realizer_generations": 0,
        "solver_generations": 0,
        "verifier_generations": 0,
        "training_runs": 0,
    }
    atomic_json(run_root / "reports/questioner_joint_reference_ab_canary.json", report)
    atomic_text(
        run_root / "reports/questioner_joint_reference_ab_canary.md",
        _report_markdown(report),
    )
    provenance = _post_provenance(run_root, manifest, pre["reference_text"])
    if not provenance["authoritative_release_provenance"]:
        raise RuntimeError("A/B canary post-run provenance is not authoritative")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

