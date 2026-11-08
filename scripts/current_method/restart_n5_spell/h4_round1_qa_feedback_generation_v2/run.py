"""Generate E1Q1 QA from the frozen v2 QA-feedback packets."""

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
from . import API_CONFIG, COMPILER_RUN, DEFAULT_BATCH_RUN, DEFAULT_SMOKE_RUN, E1Q0_QA_RUN, GENERATOR_VERSION, ROOTS, SLOTS, TEMPERATURE


def _json(path: Path, value: Any) -> None:
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def _jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(x), ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "  " + path.name + "\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_inputs(root: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    packets = {str(x["root_id"]): x for x in _rows(COMPILER_RUN / "source_qa_packets.jsonl")}
    packet = packets.get(root)
    if packet is None:
        raise RuntimeError(f"qa_feedback_packet_missing:{root}")
    manifest = next(x for x in _rows(E1Q0_QA_RUN / "paired_dialogue_manifest.jsonl") if x.get("root_id") == root)
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"e1q0_dialogue_not_pass:{root}")
    path = Path(str(manifest["feedback_path"]))
    if not path.is_file() or sha256_file(path) != manifest.get("feedback_sha256"):
        raise RuntimeError(f"e1q0_dialogue_sha_mismatch:{root}")
    dialogue = json.loads(path.read_text(encoding="utf-8"))
    return packet, manifest, dialogue


def _feedback_context(packet: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    notice = ("The source QA is curriculum feedback, not a fact source. "
              "CURRENT_CUMULATIVE_DIALOGUE is the only source for every new fact, option, Gold answer, and evidence ref.")
    intent = "\n".join(["<QA_FEEDBACK_CURRICULUM>", notice, "FROZEN_SOURCE_QA_PACKET:", text,
                         "Plan exactly three distinct question-evolution intents; do not output questions.", "</QA_FEEDBACK_CURRICULUM>"])
    goals = {
        SLOTS[0]: "Create a harder successor using at least one R02 evidence ref and a distinct fact or relation.",
        SLOTS[1]: "Target the frozen canonical confusion pattern while re-deriving all facts and evidence from the current dialogue.",
        SLOTS[2]: "Cover a different R02 dialogue portion using the feedback gap as a constraint; do not copy the source QA.",
    }
    slots = ["\n".join(["<QA_FEEDBACK_CURRICULUM>", notice, "SLOT_TYPE:", slot, "SLOT_GOAL:", goals[slot],
                         "Do not mention this packet, model names, scores, or metadata in output.", "</QA_FEEDBACK_CURRICULUM>"]) for slot in SLOTS]
    return {"intent_suffix": intent, "slot_suffixes": slots}


def _scope_candidates(root_dir: Path, root: str) -> dict[str, Any]:
    raw = json.loads((root_dir / "candidate_pool.json").read_text(encoding="utf-8"))
    valid_ids = {str(x.get("candidate_id")) for x in json.loads((root_dir / "candidates_valid.json").read_text(encoding="utf-8"))}
    contracts = json.loads((root_dir / "reasoning_contracts.json").read_text(encoding="utf-8"))
    intent_slots = {str(x.get("intent_id")): SLOTS[i] for i, x in enumerate(contracts[:len(SLOTS)])}
    scoped = []; valid = []; mapping = []
    for ordinal, row in enumerate(raw, 1):
        row = dict(row); old = str(row.get("candidate_id") or ""); source = str(row.get("generation_source") or "unknown"); intent = str(row.get("intent_reference") or "")
        slot = intent_slots.get(intent, SLOTS[min(max(ordinal - 1, 0), len(SLOTS) - 1)])
        new = f"{root}:R01-R02-feedback:E1Q1_FULL_QA_FEEDBACK_COMPOSITE:{slot}:{intent}:{source}:{ordinal:06d}"
        row.update({"source_candidate_id": old, "candidate_id": new, "root_id": root, "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE", "slot_type": slot, "feedback_conditioned": True})
        scoped.append(row)
        if old in valid_ids: valid.append(row)
        mapping.append({"source_candidate_id": old, "candidate_id": new, "slot_type": slot, "intent_id": intent, "generation_source": source})
    _json(root_dir / "candidate_pool_scoped.json", scoped); _json(root_dir / "candidates_valid_scoped.json", valid); _jsonl(root_dir / "candidate_scope_manifest.jsonl", mapping)
    return {"raw_candidate_count": len(scoped), "valid_candidate_count": len(valid), "candidate_id_unique": len({x["candidate_id"] for x in mapping}) == len(mapping)}


def _audit_feedback_alignment(root_dir: Path, packet: Mapping[str, Any], root: str) -> dict[str, Any]:
    """Route structurally valid candidates by explicit R02 evidence use.

    The model may see curriculum feedback without applying it.  Keep those
    rows for audit, but only admit R02-aligned targeted/coverage rows to the
    primary feedback analysis set.
    """
    candidates = [dict(x) for x in json.loads((root_dir / "candidates_valid_scoped.json").read_text(encoding="utf-8"))]
    r02_refs = set(map(str, (packet.get("r02_adherence") or {}).get("r02_turn_refs") or []))
    rows = []
    primary = []
    for row in candidates:
        refs = set(map(str, row.get("evidence_refs") or []))
        aligned = bool(refs & r02_refs)
        slot = str(row.get("slot_type") or "")
        if slot in {"TARGETED_EVOLUTION", "R02_COVERAGE_QA"}:
            route = "PRIMARY_FEEDBACK_ALIGNED" if aligned else "SECONDARY_OFF_TARGET_NO_R02_EVIDENCE"
        else:
            route = "PRIMARY_CONTRASTIVE" if refs else "SECONDARY_INVALID_EVIDENCE"
        row["feedback_alignment_status"] = "R02_EVIDENCE_ALIGNED" if aligned else "NO_R02_EVIDENCE"
        row["feedback_analysis_route"] = route
        rows.append(row)
        if route.startswith("PRIMARY"):
            primary.append(row)
    _jsonl(root_dir / "feedback_alignment_audit.jsonl", rows)
    _json(root_dir / "candidates_feedback_primary_scoped.json", primary)
    counts = {}
    for row in rows:
        key = (str(row.get("slot_type") or ""), str(row.get("feedback_analysis_route") or ""))
        counts["|".join(key)] = counts.get("|".join(key), 0) + 1
    return {"candidate_count": len(rows), "primary_candidate_count": len(primary), "r02_evidence_ref_count": len(r02_refs), "route_counts": counts, "all_evidence_refs_valid": all(set(map(str, x.get("evidence_refs") or [])) <= {str(y.get("turn_ref")) for y in json.loads((root_dir / "dialogue_source.json").read_text(encoding="utf-8")).get("turns") or []} for x in rows)}


def run_root(root: str, output: Path, live: bool) -> dict[str, Any]:
    if root not in ROOTS: raise ValueError(f"root_not_in_v2_safe_set:{root}")
    output = output.resolve()
    if output.exists() and any(output.iterdir()): raise RuntimeError(f"output_exists_or_nonempty:{output}")
    output.mkdir(parents=True, exist_ok=False)
    packet, manifest, dialogue = _load_inputs(root)
    plan = {"experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE", "generator_version": GENERATOR_VERSION, "root_id": root, "source_candidate_id": packet["source_candidate_id"], "dialogue_sha256": manifest["feedback_sha256"], "slot_types": list(SLOTS), "planned_primary_requests": 7, "temperature": TEMPERATURE, "max_tokens": 8192, "intent_miner_max_tokens": 4096, "memorax_calls": 0, "solver_calls": 0, "semantic_auditor_calls": 0}
    _json(output / "qa_feedback_generation_contract.json", plan); _json(output / "feedback_packet.json", packet); _json(output / "dialogue_source.json", dialogue)
    if not live: return {"status": "OFFLINE_PLAN_ONLY", "root_id": root, "run_root": str(output), "api_calls": 0}
    config = load_challenger_config(API_CONFIG); client = OpenAICompatibleChallengerClient(replace(config, temperature=TEMPERATURE, enable_thinking=False, timeout_seconds=max(180, int(config.timeout_seconds))))
    ledger = RequestLedger(output, sha256_bytes(canonical_json(plan).encode()), "live")
    generation = generate_root_qa(expanded=dialogue, root_dir=output, root_id=root, api_config=API_CONFIG, ledger=ledger, client=client, feedback_context=_feedback_context(packet))
    scoped = _scope_candidates(output, root)
    alignment = _audit_feedback_alignment(output, packet, root)
    states = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "ledger" / "requests").glob("*.json"))]
    report = {"status": "QA_ONLY_GENERATED", "root_id": root, "experiment_arm": plan["experiment_arm"], "dialogue_sha256": manifest["feedback_sha256"], "feedback_packet_source_candidate_id": packet["source_candidate_id"], "generation": generation, "scoped_candidates": scoped, "feedback_alignment": alignment, "request_count": len(states), "completed_request_count": sum(x.get("status") == "completed" for x in states), "online_calls": {"qa_writer_and_intent_miner": len(states), "memorax": 0, "solver": 0, "semantic_auditor": 0}}
    _json(output / "final_qa_feedback_generation_report.json", report); atomic_text(output / "final_qa_feedback_generation_report.md", "# E1Q1 QA Feedback Generation v2\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n"); atomic_text(output / "final_qa_feedback_generation_report.md.sha256", sha256_file(output / "final_qa_feedback_generation_report.md") + "  final_qa_feedback_generation_report.md\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root-id", action="append", dest="roots"); parser.add_argument("--output-run", type=Path); parser.add_argument("--batch", action="store_true"); parser.add_argument("--live", action="store_true"); args = parser.parse_args()
    roots = list(args.roots or (ROOTS if args.batch else ("cc-train-episode-254852",))); base = (args.output_run or (DEFAULT_BATCH_RUN if args.batch else DEFAULT_SMOKE_RUN)).resolve(); outputs = [base / x for x in roots] if len(roots) > 1 or args.batch else [base]
    reports = [run_root(root, out, args.live) for root, out in zip(roots, outputs)]
    if len(reports) > 1:
        aggregate = {"status": "QA_ONLY_BATCH_GENERATED" if all(x.get("status") == "QA_ONLY_GENERATED" for x in reports) else "QA_ONLY_BATCH_WITH_GAPS", "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE", "root_count": len(reports), "roots": [{"root_id": x.get("root_id"), "status": x.get("status"), "run_root": str(out), "candidate_count": (x.get("generation") or {}).get("candidate_count"), "valid_candidate_count": (x.get("generation") or {}).get("valid_candidate_count"), "request_count": x.get("request_count")} for x, out in zip(reports, outputs)], "online_calls": {"qa_writer_and_intent_miner": sum(int(x.get("request_count") or 0) for x in reports), "memorax": 0, "solver": 0, "semantic_auditor": 0}}
        _json(base / "final_qa_feedback_generation_batch_report.json", aggregate); atomic_text(base / "final_qa_feedback_generation_batch_report.md", "# E1Q1 QA Feedback Generation Batch v2\n\n" + json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"); atomic_text(base / "final_qa_feedback_generation_batch_report.md.sha256", sha256_file(base / "final_qa_feedback_generation_batch_report.md") + "  final_qa_feedback_generation_batch_report.md\n")
    print(json.dumps({"roots": reports}, ensure_ascii=False, indent=2)); return 0 if all(x.get("status") in {"OFFLINE_PLAN_ONLY", "QA_ONLY_GENERATED"} for x in reports) else 1


if __name__ == "__main__": raise SystemExit(main())
