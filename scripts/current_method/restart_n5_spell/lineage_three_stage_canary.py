from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .backends import QuestionerRequest
from .core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .ledger import RequestLedger
from .minimal_admission import (SPAN_IDS_ONLY_SCHEMA_PROFILE,
                                materialize_spell_style_minimal)
from .questioner_contract import default_reasoning_contracts


LINEAGE_THREE_STAGE_CANARY_VERSION = "parent_child_lineage_three_stage_canary_v1"
FORMAT_ONLY_RETRY_VERSION = "lineage_single_object_format_only_retry_v1"
ANSWER_PLACEMENT_VERSION = "lineage_three_stage_answer_placement_v1"

FORMAT_ONLY_RETRY_SUFFIX = """Your previous response was not a valid JSON object matching the required schema.

Regenerate the complete object from scratch for exactly the same semantic task. Follow the same frozen inputs and required IDs. Do not continue, repair, quote, or include any part of the previous response. Do not add semantic requirements. Return only one fully closed JSON object with no Markdown, comments, or surrounding text."""

STAGE_NORMAL_CAPS = {"relation_miner":4, "qa_writer":4,
                     "distractor_writer":8}


def format_only_retry_suffix_sha256() -> str:
    return sha256_bytes((FORMAT_ONLY_RETRY_VERSION + FORMAT_ONLY_RETRY_SUFFIX).encode())


@dataclass
class ApiBudget:
    normal_caps: Mapping[str, int] = field(default_factory=lambda:dict(STAGE_NORMAL_CAPS))
    format_retry_cap: int = 8
    physical_cap: int = 24
    normal_counts: dict[str, int] = field(default_factory=lambda:{
        key:0 for key in STAGE_NORMAL_CAPS})
    retry_counts: dict[str, int] = field(default_factory=lambda:{
        key:0 for key in STAGE_NORMAL_CAPS})
    physical_count: int = 0

    def __post_init__(self) -> None:
        # Keep the historical three-stage defaults while allowing audited canaries
        # to reuse the same budget mechanism with their own fixed stage names.
        for stage in self.normal_caps:
            self.normal_counts.setdefault(stage, 0)
            self.retry_counts.setdefault(stage, 0)

    def reserve(self, stage: str, *, is_retry: bool) -> None:
        if stage not in self.normal_caps:
            raise RuntimeError("unknown_api_budget_stage")
        if self.physical_count >= self.physical_cap:
            raise RuntimeError("global_paid_api_budget_exhausted")
        if is_retry:
            if sum(self.retry_counts.values()) >= self.format_retry_cap:
                raise RuntimeError("global_format_retry_budget_exhausted")
            self.retry_counts[stage] += 1
        else:
            if self.normal_counts[stage] >= int(self.normal_caps[stage]):
                raise RuntimeError(f"{stage}_normal_budget_exhausted")
            self.normal_counts[stage] += 1
        self.physical_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {"relation_miner_normal_requests":self.normal_counts["relation_miner"],
            "relation_miner_format_retries":self.retry_counts["relation_miner"],
            "qa_writer_normal_requests":self.normal_counts["qa_writer"],
            "qa_writer_format_retries":self.retry_counts["qa_writer"],
            "distractor_writer_normal_requests":self.normal_counts["distractor_writer"],
            "distractor_writer_format_retries":self.retry_counts["distractor_writer"],
            "format_only_retries":sum(self.retry_counts.values()),
            "total_paid_api_requests":self.physical_count,
            "normal_request_cap":sum(self.normal_caps.values()),
            "format_only_retry_cap":self.format_retry_cap,
            "physical_paid_api_request_cap":self.physical_cap}


def _prompt_hash(prompt: str) -> str:
    return sha256_bytes(prompt.encode())


def run_single_object_task(*, run_root: Path, ledger: RequestLedger, backend: Any,
                           budget: ApiBudget, stage: str, logical_request_id: str,
                           semantic_prompt: str, lineage_view: Mapping[str, Any],
                           validator: Callable[[Any], dict[str, Any]],
                           schema_retry_decisions: Sequence[str],
                           ledger_budget_limit: int | None = 24,
                           ledger_total_budget_limit: int | None = 24,
                           json_invalid_decision: str | None = None,
                           max_attempts: int = 2,
                           before_paid_request: Callable[
                               [str, bool, int], None] | None = None,
                           retry_prompt_suffix_factory: Callable[
                               [Any, Mapping[str, Any]], str | None] | None = None) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("single_object_task_max_attempts_invalid")
    semantic_sha = _prompt_hash(semantic_prompt)
    attempts: list[dict[str, Any]] = []
    accepted_validation: dict[str, Any] | None = None
    accepted_raw: dict[str, Any] | None = None
    retryable = set(schema_retry_decisions)
    retry_suffix = FORMAT_ONLY_RETRY_SUFFIX
    retry_mode = "full_object_regeneration"
    for attempt_index in range(max_attempts):
        is_retry = attempt_index > 0
        if before_paid_request is not None:
            before_paid_request(stage, is_retry, attempt_index)
        budget.reserve(stage, is_retry=is_retry)
        prompt = semantic_prompt if not is_retry else (
            semantic_prompt + "\n\n" + retry_suffix)
        request_id = f"{logical_request_id}-a{attempt_index:02d}"
        prompt_sha = _prompt_hash(prompt)
        atomic_json(run_root / f"raw/prompts/{request_id}.json", {
            "request_id":request_id, "stage":stage,
            "logical_request_id":logical_request_id,
            "attempt_index":attempt_index, "semantic_prompt_sha256":semantic_sha,
            "format_retry_suffix_sha256":format_only_retry_suffix_sha256()
                if is_retry and retry_mode == "full_object_regeneration" else None,
            "retry_suffix_sha256":sha256_bytes(retry_suffix.encode())
                if is_retry else None,
            "retry_mode":retry_mode if is_retry else None,
            "rendered_prompt_sha256":prompt_sha, "prompt":prompt})
        request = QuestionerRequest(request_id, 0, ("A", "B", "C"), prompt,
            prompt_sha, tuple(turn["turn_id"] for turn in
                              lineage_view["combined_dialogue"]),
            tuple(sorted({turn["session_id"] for turn in
                          lineage_view["combined_dialogue"]})),
            dict(lineage_view["span_catalog"]), tuple())
        response = ledger.invoke(request_id=request_id,
            intent=f"lineage_{stage}", attempt=attempt_index + 1,
            prompt_sha256=prompt_sha,
            call=lambda request=request:backend.generate_batch(request),
            budget_class="paid_api", budget_limit=ledger_budget_limit,
            total_budget_limit=ledger_total_budget_limit,
            request_metadata={"stage":stage,
                "logical_request_id":logical_request_id,
                "attempt_index":attempt_index,
                "retry_of_request_id":f"{logical_request_id}-a{attempt_index - 1:02d}"
                    if is_retry else None,
                "retry_reason":"json_or_schema_invalid" if is_retry else None,
                "retry_mode":retry_mode if is_retry else None,
                "semantic_prompt_sha256":semantic_sha,
                "format_retry_suffix_sha256":format_only_retry_suffix_sha256()
                    if is_retry and retry_mode == "full_object_regeneration" else None,
                "retry_suffix_sha256":sha256_bytes(retry_suffix.encode())
                    if is_retry else None,
                "source_lineage_view_sha256":lineage_view["view_sha256"]},
            backend_identity={"backend":"openai_compatible_single_json_object",
                "model_identifier":backend.client.config.model,
                "base_url":backend.client.config.base_url,
                "temperature":backend.client.config.temperature,
                "max_tokens":backend.max_tokens,
                "response_format":"json_object"})
        try:
            parsed = json.loads(response.raw_text)
            parse_status = "json_parsed"
            validation = validator(parsed)
            decision = str(validation.get("decision") or "schema_invalid")
        except json.JSONDecodeError:
            parsed = None
            parse_status = "json_invalid"
            json_label = json_invalid_decision or {
                "relation_miner":"relation_json_invalid",
                "qa_writer":"qa_json_invalid",
                "distractor_writer":"distractor_json_invalid",
            }.get(stage, f"{stage}_json_invalid")
            validation = {"decision":json_label, "reasons":[json_label]}
            decision = "json_invalid"
        should_retry = (attempt_index < max_attempts - 1 and
            (parse_status == "json_invalid" or decision in retryable))
        ledger_decision = "retry_format_only" if should_retry else decision
        ledger.record_decision(request_id, ledger_decision)
        state = ledger.read(request_id) or {}
        wrapper = {"stage":stage, "logical_request_id":logical_request_id,
            "attempt_index":attempt_index, "request_id":request_id,
            "semantic_prompt_sha256":semantic_sha,
            "rendered_prompt_sha256":prompt_sha, "parse_status":parse_status,
            "validation":validation, "decision":ledger_decision,
            "raw_response":response.raw_text,
            "raw_response_sha256":state.get("raw_response_sha256"),
            "response_metadata":dict(response.metadata)}
        raw_stage = "question_answer_writer" if stage == "qa_writer" else stage
        raw_path = run_root / f"raw/{raw_stage}/{request_id}.json"
        atomic_json(raw_path, wrapper)
        attempts.append({**{key:value for key,value in wrapper.items()
            if key not in {"raw_response"}},
            "raw_record_path":str(raw_path.relative_to(run_root)),
            "raw_record_sha256":sha256_file(raw_path)})
        if should_retry and retry_prompt_suffix_factory is not None:
            custom_suffix = retry_prompt_suffix_factory(parsed, validation)
            if custom_suffix is not None:
                retry_suffix = custom_suffix
                retry_mode = "directed_schema_repair"
        if not should_retry:
            if str(validation.get("decision", "")).endswith("_accept"):
                accepted_validation = validation
                accepted_raw = parsed
            break
    final = attempts[-1]
    return {"stage":stage, "logical_request_id":logical_request_id,
        "semantic_prompt_sha256":semantic_sha, "attempts":attempts,
        "physical_request_count":len(attempts),
        "format_retry_used":len(attempts) > 1,
        "final_decision":final["validation"]["decision"],
        "accepted":accepted_validation is not None,
        "accepted_raw":accepted_raw,
        "accepted_validation":accepted_validation}


def deterministic_answer_placement(arc_id: str, candidate_index: int,
                                   distractor_attempt: int) -> dict[str, Any]:
    arc_offset = {"E1":0, "E4":2}.get(arc_id)
    if arc_offset is None or candidate_index not in (0, 1) or \
            distractor_attempt not in (0, 1):
        raise ValueError("answer_placement_input_invalid")
    candidate_ordinal = arc_offset + candidate_index
    letter = "ABCD"[(candidate_ordinal + 2 * distractor_attempt) % 4]
    seed_material = canonical_json({"version":ANSWER_PLACEMENT_VERSION,
        "arc_id":arc_id, "candidate_index":candidate_index,
        "distractor_attempt":distractor_attempt})
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    return {"answer_placement_version":ANSWER_PLACEMENT_VERSION,
        "placement_seed":seed, "target_letter":letter,
        "content_dependent":False}


def should_stop_after_first_retain_frontier(final_decision: str) -> bool:
    return final_decision == "retain_frontier"


def materialize_three_stage_mcq(*, question_answer: Mapping[str, Any],
                                distractors: Sequence[str], target_letter: str,
                                span_catalog: dict[str, Any]) -> dict[str, Any]:
    raw = {"qa_id":question_answer["qa_id"],
        "question":question_answer["question"],
        "correct_answer_text":question_answer["correct_answer_text"],
        "distractors":list(distractors),
        "answer_support_span_ids":list(question_answer["answer_support_span_ids"]),
        "reasoning_steps":list(question_answer["reasoning_steps"])}
    contract = default_reasoning_contracts((target_letter, target_letter,
                                            target_letter))[0]
    qa, errors, diagnostics = materialize_spell_style_minimal(raw, target_letter,
        span_catalog=span_catalog, reasoning_contract=contract,
        schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
    return {"raw_v8_qa":raw, "minimal_decision":"accept" if not errors else "reject",
        "minimal_reasons":sorted(set(errors)), "minimal_diagnostics":diagnostics,
        "materialized_qa":qa, "reasoning_contract":contract}


def build_relation_diagnostic_contexts(*, lineage_view: Mapping[str, Any]) -> dict[str, Any]:
    parent_span_ids = set(lineage_view["selected_parent_span_ids"])
    child_span_ids = set(lineage_view["selected_child_span_ids"])
    parent_turn_ids = {span_id.rsplit("#S", 1)[0] for span_id in parent_span_ids}
    child_turn_ids = {span_id.rsplit("#S", 1)[0] for span_id in child_span_ids}
    full = [dict(turn) for turn in lineage_view["combined_dialogue"]]
    parent = [turn for turn in full if turn["turn_id"] in parent_turn_ids]
    child = [turn for turn in full if turn["turn_id"] in child_turn_ids]
    if (not parent or not child or len(parent) + len(child) != len(full) or
            set(turn["turn_id"] for turn in parent) &
            set(turn["turn_id"] for turn in child)):
        raise ValueError("lineage_diagnostic_context_partition_invalid")
    return {"parent_only":parent, "child_only":child, "full_lineage":full}


def classify_joint_necessity_v1(*, parent_rewards: Sequence[int],
                                child_rewards: Sequence[int],
                                lineage_rewards: Sequence[int]) -> dict[str, Any]:
    if any(len(values) != 3 or any(value not in (0, 1) for value in values)
           for values in (parent_rewards, child_rewards, lineage_rewards)):
        return {"decision":"joint_necessity_inconclusive",
                "parent_correct":sum(parent_rewards),
                "child_correct":sum(child_rewards),
                "lineage_correct":sum(lineage_rewards),
                "valid_k_complete":False}
    parent_correct, child_correct = sum(parent_rewards), sum(child_rewards)
    lineage_correct = sum(lineage_rewards)
    if parent_correct >= 2:
        decision = "parent_only_sufficient"
    elif child_correct >= 2:
        decision = "child_only_sufficient"
    elif parent_correct <= 1 and child_correct <= 1 and lineage_correct >= 2:
        decision = "lineage_jointly_required"
    elif lineage_correct <= 1:
        decision = "lineage_not_answerable"
    else:
        decision = "joint_necessity_inconclusive"
    return {"decision":decision, "parent_correct":parent_correct,
        "child_correct":child_correct, "lineage_correct":lineage_correct,
        "valid_k_complete":True}


def artifact_sha256_manifest(run_root: Path) -> dict[str, Any]:
    records = []
    included_roots = ("raw/", "frozen/", "materialized/", "reports/")
    exclusions = {"reports/artifact_sha256.json", "reports/post_run_provenance.json"}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(run_root))
        if relative in exclusions or not relative.startswith(included_roots):
            continue
        records.append({"path":relative, "size":path.stat().st_size,
                        "sha256":sha256_file(path)})
    body = {"schema_version":"lineage_three_stage_artifact_manifest_v1",
        "self_exclusions":sorted(exclusions), "files":records}
    body["file_manifest_sha256"] = sha256_bytes(canonical_json(records).encode())
    return body
