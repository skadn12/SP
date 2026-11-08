"""Generate QA from frozen Round-0 feedback on existing R02 dialogues only."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config

from . import (
    API_CONFIG,
    DEFAULT_BATCH_RUN,
    DEFAULT_SMOKE_RUN,
    E1Q0_QA_RUN,
    GENERATOR_MAX_TOKENS,
    GENERATOR_TEMPERATURE,
    GENERATOR_VERSION,
    INTENT_MAX_TOKENS,
    ROOTS,
    SLOT_TYPES,
)
from .context import load_feedback_packet, render_prompt_context


def _json(path: Path, value: Any) -> None:
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def _jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    body = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_text(path, body)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_feedback_dialogue(root_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_rows = _load_jsonl(E1Q0_QA_RUN / "paired_dialogue_manifest.jsonl")
    rows = [row for row in manifest_rows if str(row.get("root_id") or "") == root_id]
    if len(rows) != 1:
        raise RuntimeError(f"e1q0_feedback_manifest_not_unique:{root_id}:{len(rows)}")
    manifest = rows[0]
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"e1q0_feedback_dialogue_not_pass:{root_id}")
    path = Path(str(manifest.get("feedback_path") or ""))
    if not path.is_file():
        raise RuntimeError(f"e1q0_feedback_dialogue_missing:{path}")
    observed = sha256_file(path)
    if observed != str(manifest.get("feedback_sha256") or ""):
        raise RuntimeError(f"e1q0_feedback_dialogue_sha_mismatch:{root_id}")
    dialogue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dialogue, dict) or str(dialogue.get("root_id") or "") != root_id:
        raise RuntimeError(f"e1q0_feedback_dialogue_root_mismatch:{root_id}")
    return manifest, dialogue


def _slot_by_intent(output_root: Path) -> dict[str, str]:
    contracts = json.loads((output_root / "reasoning_contracts.json").read_text(encoding="utf-8"))
    if not isinstance(contracts, list):
        return {}
    return {
        str(row.get("intent_id")): SLOT_TYPES[index]
        for index, row in enumerate(contracts[:len(SLOT_TYPES)])
        if isinstance(row, Mapping) and row.get("intent_id")
    }


def _scope_candidates(output_root: Path, root_id: str) -> dict[str, Any]:
    raw_path = output_root / "candidate_pool.json"
    valid_path = output_root / "candidates_valid.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else []
    valid = json.loads(valid_path.read_text(encoding="utf-8")) if valid_path.is_file() else []
    raw = [dict(row) for row in raw if isinstance(row, Mapping)]
    valid_ids = {str(row.get("candidate_id") or "") for row in valid if isinstance(row, Mapping)}
    slots = _slot_by_intent(output_root)
    scoped: list[dict[str, Any]] = []
    valid_scoped: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for ordinal, row in enumerate(raw, 1):
        old_id = str(row.get("candidate_id") or "")
        source = str(row.get("generation_source") or "unknown")
        intent_id = str(row.get("intent_reference") or "")
        slot = slots.get(intent_id, SLOT_TYPES[min(max(ordinal - 1, 0), len(SLOT_TYPES) - 1)])
        new_id = (
            f"{root_id}:R01-R02-feedback:E1Q1_FULL_QA_FEEDBACK_COMPOSITE:"
            f"{slot}:{source}:{ordinal:06d}"
        )
        item = dict(row)
        item["source_candidate_id"] = old_id
        item["candidate_id"] = new_id
        item["root_id"] = root_id
        item["experiment_arm"] = "E1Q1_FULL_QA_FEEDBACK_COMPOSITE"
        item["slot_type"] = slot
        item["feedback_conditioned"] = True
        scoped.append(item)
        if old_id in valid_ids:
            valid_scoped.append(item)
        mapping.append({
            "source_candidate_id": old_id,
            "candidate_id": new_id,
            "slot_type": slot,
            "generation_source": source,
        })
    _json(output_root / "candidate_pool_scoped.json", scoped)
    _json(output_root / "candidates_valid_scoped.json", valid_scoped)
    _jsonl(output_root / "candidate_scope_manifest.jsonl", mapping)
    return {
        "raw_candidate_count": len(scoped),
        "valid_candidate_count": len(valid_scoped),
        "candidate_id_unique": len({row["candidate_id"] for row in mapping}) == len(mapping),
        "candidate_ids": mapping,
    }


def run_root(*, root_id: str, output_root: Path, live: bool) -> dict[str, Any]:
    if root_id not in ROOTS:
        raise ValueError(f"root_not_in_e1q0_success_set:{root_id}")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output_exists_or_nonempty:{output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest, dialogue = _load_feedback_dialogue(root_id)
    packet = load_feedback_packet(root_id)
    feedback_context = render_prompt_context(packet)
    plan = {
        "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE",
        "generator_version": GENERATOR_VERSION,
        "root_id": root_id,
        "feedback_source_round": "round0",
        "dialogue_source": "E1Q0_feedback_cumulative_dialogue",
        "dialogue_path": str(Path(str(manifest["feedback_path"])).resolve()),
        "dialogue_sha256": manifest["feedback_sha256"],
        "source_candidate_id": packet["source_candidate_id"],
        "slot_types": list(SLOT_TYPES),
        "planned_primary_requests": 7,
        "question_generation": {
            "intent_miner_max_tokens": INTENT_MAX_TOKENS,
            "writer_max_tokens": GENERATOR_MAX_TOKENS,
            "temperature": GENERATOR_TEMPERATURE,
            "generation_sources": ["contract_guided", "free_exploration"],
        },
        "memory_add_calls": 0,
        "memorax_search_calls": 0,
        "solver_calls": 0,
        "semantic_auditor_calls": 0,
    }
    _json(output_root / "qa_only_run_contract.json", plan)
    _json(output_root / "feedback_packet.json", packet)
    _json(output_root / "dialogue_source.json", dialogue)
    if not live:
        return {"status": "OFFLINE_PLAN_ONLY", "root_id": root_id, "run_root": str(output_root), "api_calls": 0}

    config = load_challenger_config(API_CONFIG)
    client = OpenAICompatibleChallengerClient(replace(
        config, temperature=GENERATOR_TEMPERATURE, enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds))))
    ledger = RequestLedger(
        output_root,
        sha256_bytes(canonical_json(plan).encode("utf-8")),
        "live",
    )
    generation = generate_root_qa(
        expanded=dialogue,
        root_dir=output_root,
        root_id=root_id,
        api_config=API_CONFIG,
        ledger=ledger,
        client=client,
        feedback_context=feedback_context,
    )
    scoped = _scope_candidates(output_root, root_id)
    request_states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "ledger" / "requests").glob("*.json"))
    ]
    report = {
        "status": "QA_ONLY_GENERATED",
        "root_id": root_id,
        "experiment_arm": plan["experiment_arm"],
        "dialogue_sha256": manifest["feedback_sha256"],
        "feedback_packet_source_candidate_id": packet["source_candidate_id"],
        "generation": generation,
        "scoped_candidates": {key: value for key, value in scoped.items() if key != "candidate_ids"},
        "request_count": len(request_states),
        "completed_request_count": sum(row.get("status") == "completed" for row in request_states),
        "finish_reason_counts": {
            str(reason): sum((row.get("response_metadata") or {}).get("finish_reason") == reason for row in request_states)
            for reason in sorted({(row.get("response_metadata") or {}).get("finish_reason") for row in request_states})
        },
        "online_calls": {
            "qa_writer_and_intent_miner": len(request_states),
            "memorax": 0,
            "solver": 0,
            "semantic_auditor": 0,
        },
    }
    _json(output_root / "final_qa_only_report.json", report)
    atomic_text(
        output_root / "final_qa_only_report.md",
        "# Round-1 QA-only Feedback Generation\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_text(output_root / "final_qa_only_report.md.sha256", sha256_file(output_root / "final_qa_only_report.md") + "  final_qa_only_report.md\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-id", action="append", dest="root_ids")
    parser.add_argument("--output-run", type=Path)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    root_ids = list(args.root_ids or (ROOTS if args.batch else ("cc-train-episode-254852",)))
    base = (args.output_run or (DEFAULT_BATCH_RUN if args.batch else DEFAULT_SMOKE_RUN)).resolve()
    if len(root_ids) == 1 and not args.batch:
        outputs = [base]
    else:
        outputs = [base / root_id for root_id in root_ids]
    reports = [run_root(root_id=root_id, output_root=out, live=args.live) for root_id, out in zip(root_ids, outputs)]
    if len(reports) > 1:
        aggregate = {
            "status": (
                "QA_ONLY_BATCH_GENERATED"
                if all(row.get("status") == "QA_ONLY_GENERATED" for row in reports)
                else "QA_ONLY_BATCH_WITH_GAPS"
            ),
            "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE",
            "generator_version": GENERATOR_VERSION,
            "root_count": len(reports),
            "roots": [
                {
                    "root_id": row.get("root_id"),
                    "status": row.get("status"),
                    "run_root": str(out),
                    "candidate_count": (row.get("generation") or {}).get("candidate_count"),
                    "valid_candidate_count": (row.get("generation") or {}).get("valid_candidate_count"),
                    "request_count": row.get("request_count"),
                    "completed_request_count": row.get("completed_request_count"),
                    "online_calls": row.get("online_calls"),
                }
                for row, out in zip(reports, outputs)
            ],
            "online_calls": {
                "qa_writer_and_intent_miner": sum(int(row.get("request_count") or 0) for row in reports),
                "memorax": 0,
                "solver": 0,
                "semantic_auditor": 0,
            },
        }
        _json(base / "final_qa_only_batch_report.json", aggregate)
        atomic_text(
            base / "final_qa_only_batch_report.md",
            "# Round-1 QA-only Feedback Batch\n\n"
            + json.dumps(aggregate, ensure_ascii=False, indent=2)
            + "\n",
        )
        atomic_text(
            base / "final_qa_only_batch_report.md.sha256",
            sha256_file(base / "final_qa_only_batch_report.md")
            + "  final_qa_only_batch_report.md\n",
        )
    print(json.dumps({"roots": reports}, ensure_ascii=False, indent=2))
    return 0 if all(row.get("status") in {"OFFLINE_PLAN_ONLY", "QA_ONLY_GENERATED"} for row in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
