"""Offline context accounting through the frozen V4 packet and API builders."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from . import base_qa_executor_v2 as v2
from . import base_qa_executor_v4_single_root as v4
from . import base_qa_pipeline_v1 as qa
from . import qa_api_transport_v2 as transport

PROJECT = Path(__file__).resolve().parents[2]


def _evidence(turn):
    quote = turn["text"][:160]
    if not quote.strip():
        raise ValueError("sample source turn has no usable quote")
    return {"turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": quote}


def _sample_chain(builder, source, warnings):
    turns = source["turns"]
    if len(turns) < 2:
        raise ValueError("context sample requires two source turns")
    left, right = turns[:2]
    left_ev, right_ev = _evidence(left), _evidence(right)
    relation = {
        "relation_id": "context-preflight-relation",
        "author_id": "p0-v4-author-" + v4.ROOT_ID,
        "target": "A synthetic, two-turn relation used only to size requests.",
        "relation_type": "cross_turn_confirmation",
        "why_ask": "The sample chain exercises the complete dependent packet shape.",
        "expected_answer": ["The synthetic relation joins both quoted statements."],
        "requires_deterministic_check": False,
        "facts": [
            {"fact_id": "fact-left", "claim": "First quoted source statement.",
             "evidence": [left_ev], "evidence_turn_ids": [left["turn_ref"]]},
            {"fact_id": "fact-right", "claim": "Second quoted source statement.",
             "evidence": [right_ev], "evidence_turn_ids": [right["turn_ref"]]},
        ],
        "inference_steps": [{"step_id": "join", "inputs": ["fact-left", "fact-right"],
                             "conclusion": "The sample answer joins both statements."}],
        "necessity_groups": [
            {"group_id": "left", "fact_ids": ["fact-left"],
             "turn_refs": [left["turn_ref"]], "why_necessary": "Supplies the first side.",
             "removal_effect": "Without it, only the second side remains."},
            {"group_id": "right", "fact_ids": ["fact-right"],
             "turn_refs": [right["turn_ref"]], "why_necessary": "Supplies the second side.",
             "removal_effect": "Without it, only the first side remains."},
        ],
    }
    if qa.validate_relation(relation, source):
        raise ValueError("synthetic relation does not satisfy the current V1 schema")

    candidate = {
        "candidate_id": "context-preflight-candidate",
        "root_id": v4.ROOT_ID,
        "author_id": relation["author_id"],
        "relation_sha256": qa.legacy.fingerprint(relation),
        "question_type": "single_choice",
        "question": "Which conclusion depends on considering both quoted turns?",
        "options": {"A": "The synthetic relation joins both quoted statements.",
                    "B": "Only the first quoted statement is relevant."},
        "gold": ["A"],
        "ordering_edges": [],
        "deterministic_checks": [],
        "item_analysis": {
            "A": {"status": "supported", "answer_value": relation["expected_answer"][0],
                  "rationale": "The sample relation uses both source turns.",
                  "evidence": [left_ev, right_ev]},
            "B": {"status": "contradicted", "answer_value": "Only the first statement matters.",
                  "rationale": "The proposed relation includes both source turns.",
                  "error_reason": "It drops the second required relation input.",
                  "evidence": [left_ev]},
        },
    }
    audit = qa.validate_candidate(candidate, relation, source)
    if not audit["mechanical_pass"]:
        raise ValueError("synthetic candidate does not satisfy the current V1 schema")

    blind_packet = builder._blind_packet(source, warnings, candidate)
    blind = {
        "packet_sha256": qa.legacy.fingerprint(blind_packet),
        "reviewer_id": "p0-v4-reviewer-" + v4.ROOT_ID,
        "model_id": transport.EXPECTED_MODEL,
        "decision": "SOLVED",
        "independent_gold": ["A"],
        "ambiguities": [],
        "evidence": [left_ev, right_ev],
        "reasoning_steps": ["Read the first quoted turn.", "Read the second quoted turn.",
                             "Join both facts into the sample answer."],
    }
    if qa.validate_blind(candidate, source, blind_packet, blind):
        raise ValueError("synthetic blind response does not satisfy the current V1 schema")

    with tempfile.TemporaryDirectory(prefix="v4-context-preflight-") as temp:
        run = Path(temp)
        v2.freeze(run, "RELATION", relation)
        v2.freeze(run, "CANDIDATE", candidate)
        v2.freeze(run, "BLIND_PACKET", blind_packet)
        v2.freeze(run, "BLIND_SOLUTION", blind)
        comparison = builder._comparison_packet(run, source, warnings)

    return relation, candidate, blind_packet, blind, comparison


def _request(packet, metadata):
    body = transport.build_body(packet, metadata)
    transport.validate_body(body)
    return body


def inspect_context(selection_path=v4.DEFAULT_SELECTION, token_counter=None):
    selection_path = Path(selection_path).resolve()
    selection = qa.read(selection_path)
    row = next(item for item in selection["roots"] if item.get("root_id") == v4.ROOT_ID)
    binding = qa.legacy.file_binding(row["path"])
    source = qa.source_from(binding, v4.ROOT_ID)
    warnings = qa.read(row["warnings_path"]) if row.get("warnings_path") else []
    builder = object.__new__(v4.ExecutorV4)
    relation, candidate, blind_packet, blind, comparison = _sample_chain(builder, source, warnings)

    author_id = "p0-v4-author-" + v4.ROOT_ID
    reviewer_id = "p0-v4-reviewer-" + v4.ROOT_ID
    packets = {
        "DISCOVERY": (builder._discovery_packet(source, warnings), {"author_id": author_id}),
        "AUTHOR": (builder._author_packet(source, warnings, relation),
                   {"author_id": author_id, "relation_sha256": qa.legacy.fingerprint(relation)}),
        "BLIND": (blind_packet, {"reviewer_id": reviewer_id,
                                 "model_id": transport.EXPECTED_MODEL}),
        "COMPARE": (comparison, {"reviewer_id": reviewer_id,
                                 "blind_sha256": qa.legacy.fingerprint(blind),
                                 "candidate_sha256": qa.legacy.fingerprint(candidate)}),
    }

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    profile = qa.read(v4.PROFILE)
    counter = token_counter or v2.LocalTokenizerCounter(profile["tokenizer_directory"])
    stages = []
    for stage in v4.STAGES:
        packet, metadata = packets[stage]
        body = _request(packet, {**metadata, "packet_sha256": qa.legacy.fingerprint(packet)})
        input_tokens = counter(body)
        input_cap = v4.INPUT_LIMITS[stage]
        stages.append({
            "stage": stage,
            "input_estimate_basis": "V4 packet builders + qa_api_transport_v2.build_body + local tokenizer",
            "sample_input_tokens_local": input_tokens,
            "sample_total_context_tokens_local": input_tokens + v4.OUTPUT_LIMIT,
            "schema_maximum_input_tokens": input_tokens if stage == "DISCOVERY" else None,
            "schema_maximum_status": "FIXED_SOURCE_AND_WARNINGS" if stage == "DISCOVERY" else
                "UNBOUNDED_BY_CURRENT_V1_RESPONSE_SCHEMA",
            "v4_enforced_max_input_tokens_local": input_cap,
            "requested_max_output_tokens": v4.OUTPUT_LIMIT,
            "v4_maximum_accepted_total_context_tokens_local": input_cap + v4.OUTPUT_LIMIT,
            "sample_within_v4_input_limit": input_tokens <= input_cap,
        })

    probe_dir = (PROJECT / "research-protocols/selfplay_week_20261019" /
                 "base_qa_proxy_updated_key_probe_20261030_v1")
    probe_path = probe_dir / "PROBE_SUMMARY.json"
    probe = qa.read(probe_path) if probe_path.is_file() else {}
    probe_request_path = probe_dir / "REQUEST.json"
    probe_request = qa.read(probe_request_path) if probe_request_path.is_file() else {}
    service_config_path = (PROJECT / "current_method_github/configs/h4_writer_model_comparison_v1/strong_model.json")
    service_config = qa.read(service_config_path) if service_config_path.is_file() else {}
    readiness_path = (PROJECT / "research-protocols/selfplay_week_20261019" /
                      "base_qa_executor_v3_integration_20261031_v1/LIVE_READINESS.json")
    readiness = qa.read(readiness_path) if readiness_path.is_file() else {}
    prior_report_path = (PROJECT / "restart-n5-runs/h4_adherence_state_v8_tju_live_1/reports/FINAL_REPORT.md")
    frontend_evidence_path = (PROJECT / "current_method_github/configs/h4_writer_model_comparison_v1/" /
                              "service_evidence/tju_public_frontend.md")

    return {
        "version": v4.VERSION,
        "root_id": v4.ROOT_ID,
        "source": {"path": binding["path"], "sha256": binding["sha256"],
                   "bytes": binding["bytes"], "turns": len(source["turns"]),
                   "warnings": len(warnings)},
        "tokenizer": {"path": profile["tokenizer_directory"],
                      "method": "LocalTokenizerCounter.apply_chat_template(add_generation_prompt=True) on each final V4 request body",
                      "network": False, "model_weights_loaded": False,
                      "matches_school_service_tokenizer": False,
                      "historical_difference": "A successful synthetic probe estimated 20 local prompt tokens while school usage reported 24."},
        "stages": stages,
        "schema_limit_finding": {
            "status": "NO_FINITE_SCHEMA_MAX_FOR_DEPENDENT_STAGES",
            "reason": "V1 validation does not cap relation string lengths or most arrays; candidate option count is capped at 26 but option text and analysis evidence are not; blind arrays and comparison necessity groups are also unbounded. The sample requests below are valid-shape sizing examples, not maxima.",
            "source": str(Path(v4.__file__).with_name("base_qa_pipeline_v1.py")),
            "runtime_protection": "V4 counts each exact outgoing body locally and refuses to reserve/send when its stage cap is exceeded; no truncation or splitting occurs.",
        },
        "service_capacity": {
            "verified_remote_context_window_tokens": None,
            "verified_remote_tokenizer": False,
            "verified_max_tokens_4096_enforcement": False,
            "reasoning_accounting_verified": False,
            "historical_probe": {
                "path": str(probe_path), "status": probe.get("status"),
                "request_path": str(probe_request_path),
                "requested_max_tokens": probe_request.get("max_tokens"),
                "returned_model": probe.get("returned_model"), "usage": probe.get("usage"),
                "server_limit_proved": probe.get("server_64_hard_limit_proved"),
            },
            "configured_service_limits": {
                "path": str(service_config_path),
                "limits": service_config.get("service_limits"),
                "these_values_are_service_evidence": False,
            },
            "historical_longer_request_evidence": {
                "path": str(prior_report_path),
                "reported_prompt_tokens": [3676, 3703],
                "reported_completion_tokens": [329, 351],
                "reported_request_max_tokens": 4096,
                "local_vs_school_prompt_count": {"local": 6908, "school": 7379},
                "limit": "Successful requests show only that those smaller requests completed; they do not establish context or hard output limits.",
            },
            "public_frontend_evidence": {
                "path": str(frontend_evidence_path),
                "supports_endpoint_alias_and_bearer_example": True,
                "is_v3_service_contract": False,
                "website_v1_100000_50000_values_are_service_limits": False,
            },
            "v3_readiness": readiness.get("capabilities"),
            "historical_40960_value_is_service_evidence": False,
            "send_size_proven_safe": False,
            "decision": "HOLD_NO_RELIABLE_SCHOOL_PROXY_CONTEXT_CAPACITY_EVIDENCE",
        },
        "request_context_requirements_local_max": {
            stage: {"input_limit": v4.INPUT_LIMITS[stage],
                    "requested_output_limit": v4.OUTPUT_LIMIT,
                    "total": v4.INPUT_LIMITS[stage] + v4.OUTPUT_LIMIT}
            for stage in v4.STAGES
        },
    }


def main():
    print(json.dumps(inspect_context(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
