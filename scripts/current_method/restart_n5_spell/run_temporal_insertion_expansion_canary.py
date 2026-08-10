"""Run one temporal-insertion outline expansion, then Questioner and Solver."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from .core import atomic_json, sha256_bytes, sha256_file
from .graph_expansion import (
    GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
    graph_scene_coverage_report,
)
from .ledger import RequestLedger
from .lineage_three_stage_canary import ApiBudget, run_single_object_task
from .live_backends import (
    LiveRealizerBackend,
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from .orchestrator import CanaryOrchestrator
from .run_canary import (
    DEFAULTS,
    create_source_snapshot,
    git_value,
    project_diff_sha256,
    write_post_run_provenance,
)
from .run_rolling_outline_30k_canary import (
    CHALLENGER_MODEL,
    DEFAULT_FEEDBACK,
    DEFAULT_SOURCE,
    MODEL_CONTEXT_LIMIT,
    SOLVER,
    TOKENIZER,
    _character_summary,
    _questioner_round_without_stopping_expansion,
    _source_attestation,
    _tokens,
    _usage,
    _view,
)
from .rolling_outline_30k import remap_baseline_turns
from .temporal_insertion_expansion import (
    TEMPORAL_GRAPH_CONTRACT_VERSION,
    TEMPORAL_INSERTION_CANARY_VERSION,
    TEMPORAL_OUTLINE_PROMPT_VERSION,
    TEMPORAL_OUTLINE_SCHEMA,
    TEMPORAL_OUTLINE_SCHEMA_VERSION,
    TEMPORAL_REALIZER_PROMPT_VERSION,
    build_temporal_outline_prompt,
    build_temporal_realizer_prompt,
    canonical_sha256,
    compile_temporal_scene_contracts,
    merge_temporal_insertions,
    normalize_temporal_outline_plan,
    temporal_realizer_template_hash,
    validate_temporal_outline_plan,
)
from .validation import DEFAULT_DURABLE_PREDICATES, validate_config


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_temporal_insertion_outline_expansion_canary_v3")
PLANNER_MAX_TOKENS = 7000


class TemporalInsertionOrchestrator(CanaryOrchestrator):
    """Reuse Realizer execution while supplying explicit temporal contracts."""

    def __init__(self, *, temporal_contracts: Sequence[Mapping[str, Any]],
                 **kwargs: Any):
        self.temporal_contracts = [dict(contract)
                                   for contract in temporal_contracts]
        super().__init__(**kwargs)

    def _graph_contracts(self) -> list[dict[str, Any]]:
        return [dict(contract) for contract in self.temporal_contracts]

    def _session_prompt(self, index: int, accepted: list[dict[str, Any]],
                        hooks: list[dict[str, str]]) -> str:
        if hooks:
            raise RuntimeError("temporal_insertion_hooks_not_supported")
        outline = self.data["outline"]
        contract = self._graph_contract(index)
        beats = self._assigned_narrative_beats(index, accepted)
        if not beats:
            raise RuntimeError("temporal_scene_has_no_assigned_stage")
        allowed = list(outline["source_parent"]["core_speakers"])
        for character in outline.get("allowed_new_characters") or []:
            introduction = int(str(character.get("introduction_scene") or
                                   "S999").lstrip("S") or 999)
            if introduction <= index:
                allowed.append(str(character["name"]))
        seen = sorted({str(turn["speaker"]) for session in accepted
                       for turn in session.get("dialogue_turns") or []})
        assigned_arc_ids = set(contract["assigned_event_arc_ids"])
        active = [arc for arc in outline["planned_event_arcs"]
                  if arc["event_id"] in assigned_arc_ids]
        continuity = {
            "accepted_inserted_scene_count": len(accepted),
            "all_seen_character_names": seen,
            "current_insertion_after_session_id": contract[
                "insertion_after_session_id"],
            "current_insertion_before_session_id": contract[
                "insertion_before_session_id"],
            "current_insertion_gap_index": contract["insertion_gap_index"],
        }
        outline_text = json.dumps(outline, ensure_ascii=False).casefold()
        durable_subjects = list(allowed)
        durable_subjects.extend(
            phrase for phrase in
            ("application", "priorities brief", "brief", "report", "plan")
            if phrase in outline_text)
        return build_temporal_realizer_prompt(
            graph_scene_contract=contract,
            session_index=index,
            narrative_beats=beats,
            recent_sessions=accepted[-2:],
            continuity_summary=continuity,
            active_plans=active,
            pending_decisions=[],
            superseded_information=[],
            character_knowledge={"all_seen_character_names": seen},
            visibility_boundaries=[],
            unresolved_hooks=[],
            allowed_people=allowed,
            session_location=contract["session_location"],
            allowed_event_arc_ids=contract["assigned_event_arc_ids"],
            forbidden_conflicts=outline["forbidden_conflicts"],
            parent_anchors=contract["parent_anchor_evidence"],
            allowed_durable_subjects=sorted(set(durable_subjects)),
            durable_predicate_whitelist=sorted(DEFAULT_DURABLE_PREDICATES),
        )


def _resolved_config(*, root: Path, api: Any,
                     outline: Mapping[str, Any]) -> dict[str, Any]:
    config = {
        **DEFAULTS,
        "mode": "live_graph_expansion",
        "backend_mode": "live",
        "lineage": "conv-41",
        "run_root": str(root),
        "max_total_paid_api_generations": 30,
        "max_realizer_generations": 30,
        "max_session_attempts_per_session": 3,
        "max_questioner_generations": 0,
        "max_post_freeze_memory_generations": 0,
        "max_verifier_generations": 0,
        "max_sessions": 10,
        "graph_expansion_version": TEMPORAL_INSERTION_CANARY_VERSION,
        "graph_scene_contract_version": TEMPORAL_GRAPH_CONTRACT_VERSION,
        "graph_stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
        "minimum_accepted_child_tokens": int(
            outline["target"]["minimum_accepted_child_tokens"]),
        "target_child_tokens": int(outline["target"]["child_token_growth"]),
        "challenger_model_identifier": api.model,
        "challenger_temperature": api.temperature,
        "challenger_realizer_max_tokens": api.realizer_max_tokens,
        "semantic_repair_generations": 0,
        "training_runs": 0,
    }
    validate_config(config)
    return config


def _write_readable_dialogue(path: Path,
                             turns: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Temporally Interleaved Expanded Dialogue", ""]
    previous = None
    for turn in turns:
        session_id = str(turn["session_id"])
        if session_id != previous:
            lines.extend([f"## {session_id}", ""])
            previous = session_id
        lines.extend([
            f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _character_report(*, original: Sequence[Mapping[str, Any]],
                      combined: Sequence[Mapping[str, Any]],
                      outline: Mapping[str, Any]) -> dict[str, Any]:
    original_speakers = sorted({str(turn["speaker"]) for turn in original})
    final_speakers = sorted({str(turn["speaker"]) for turn in combined})
    declared = [str(row["name"])
                for row in outline.get("allowed_new_characters") or []]
    return {
        "original_unique_speaker_count": len(original_speakers),
        "original_unique_speakers": original_speakers,
        "final_unique_speaker_count": len(final_speakers),
        "final_unique_speakers": final_speakers,
        "declared_new_character_count": len(declared),
        "declared_new_characters": declared,
        "materialized_new_speakers": sorted(
            set(final_speakers) - set(original_speakers)),
    }


def _top_manifest(*, root: Path, source: Mapping[str, Any],
                  snapshot: Mapping[str, Any], project_diff: str,
                  api: Any, prompt: str, feedback_source: Path) -> dict[str, Any]:
    return {
        "run_id": root.name,
        "mode": TEMPORAL_INSERTION_CANARY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256": project_diff,
        "source_snapshot_manifest_path": snapshot[
            "source_snapshot_manifest_path"],
        "source_snapshot_manifest_sha256": snapshot[
            "source_snapshot_manifest_sha256"],
        "source_attestation": {key: value for key, value in source.items()
                               if key != "dialogue"},
        "source_scope": "authoritative_parent_237_turns_only",
        "existing_134_turn_child_used": False,
        "challenger_model_identifier": api.model,
        "challenger_enable_thinking": api.enable_thinking,
        "challenger_temperature": api.temperature,
        "temporal_outline_prompt_version": TEMPORAL_OUTLINE_PROMPT_VERSION,
        "temporal_outline_prompt_sha256": sha256_bytes(prompt.encode()),
        "temporal_outline_schema_version": TEMPORAL_OUTLINE_SCHEMA_VERSION,
        "temporal_outline_schema_sha256": canonical_sha256(
            TEMPORAL_OUTLINE_SCHEMA),
        "temporal_contract_version": TEMPORAL_GRAPH_CONTRACT_VERSION,
        "temporal_realizer_prompt_version": TEMPORAL_REALIZER_PROMPT_VERSION,
        "temporal_realizer_prompt_sha256": temporal_realizer_template_hash(),
        "fixed_positional_stage_schedule_used": False,
        "feedback_source": str(feedback_source.resolve()),
        "feedback_source_sha256": sha256_file(feedback_source),
        "solver_model": str(SOLVER),
        "solver_max_model_len": MODEL_CONTEXT_LIMIT,
        "ordinary_questioner_semantics_changed": False,
        "semantic_repair_used": False,
        "distractor_stage_used": False,
        "verifier_generations": 0,
        "training_runs": 0,
        "historical_runs_modified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--feedback-source", type=Path,
                        default=DEFAULT_FEEDBACK)
    parser.add_argument("--api-config", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--accepted-outline-source", type=Path,
                        help=("Reuse an already accepted raw v3 outline in a "
                              "new run root; no Planner request is sent."))
    parser.add_argument("--accepted-sessions-source-root", type=Path,
                        help=("Seed a new run with a hash-verified prefix of "
                              "accepted Realizer sessions."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_root.exists():
        raise FileExistsError(f"run_root_exists:{args.run_root}")

    source = _source_attestation(args.source_root.resolve())
    original = remap_baseline_turns(source["dialogue"]["parent_turns"])
    if len(original) != 237:
        raise RuntimeError("temporal_source_parent_count_mismatch")
    feedback = json.loads(args.feedback_source.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER,
                                               local_files_only=True)
    api = replace(load_challenger_config(args.api_config),
                  model=CHALLENGER_MODEL, temperature=0.25,
                  enable_thinking=False, timeout_seconds=600)
    solver_config = json.loads((SOLVER / "config.json").read_text(
        encoding="utf-8"))
    if int(solver_config.get("max_position_embeddings") or 0) < \
            MODEL_CONTEXT_LIMIT:
        raise RuntimeError("solver_native_context_too_short")

    args.run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, args.run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    prompt = build_temporal_outline_prompt(original_turns=original,
                                           round_index=1)
    manifest = _top_manifest(root=args.run_root, source=source,
        snapshot=snapshot, project_diff=project_diff, api=api, prompt=prompt,
        feedback_source=args.feedback_source)
    if args.accepted_outline_source is not None:
        accepted_source = args.accepted_outline_source.resolve()
        if not accepted_source.is_file():
            raise FileNotFoundError(
                f"accepted_outline_source_missing:{accepted_source}")
        manifest["accepted_outline_source"] = str(accepted_source)
        manifest["accepted_outline_source_sha256"] = sha256_file(
            accepted_source)
        manifest["planner_reused_without_new_api_request"] = True
    else:
        manifest["planner_reused_without_new_api_request"] = False
    if args.accepted_sessions_source_root is not None:
        sessions_source = args.accepted_sessions_source_root.resolve()
        source_index = sessions_source / "accepted/index.json"
        if not source_index.is_file():
            raise FileNotFoundError(
                f"accepted_sessions_source_missing:{sessions_source}")
        index_payload = json.loads(source_index.read_text(encoding="utf-8"))
        session_hashes = dict(index_payload.get("session_sha256") or {})
        for name, expected_sha in session_hashes.items():
            source_path = sessions_source / "accepted" / name
            if not source_path.is_file() or sha256_file(source_path) != expected_sha:
                raise RuntimeError("accepted_sessions_source_hash_mismatch")
        manifest["accepted_sessions_source_root"] = str(sessions_source)
        manifest["accepted_sessions_source_index_sha256"] = sha256_file(
            source_index)
        manifest["accepted_sessions_source_file_sha256"] = session_hashes
    atomic_json(args.run_root / "config/run_manifest.json", manifest)
    atomic_json(args.run_root / "frozen/source_attestation.json", {
        key: value for key, value in source.items() if key != "dialogue"})
    atomic_json(args.run_root / "frozen/original_parent_dialogue.json", {
        "turns": original,
        "turn_count": len(original),
        "dialogue_tokens": _tokens(tokenizer, original),
        "source_child_turn_count_in_generation_input": 0,
    })
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    preflight = {
        "decision": ("pass" if prompt_tokens + PLANNER_MAX_TOKENS <=
                     MODEL_CONTEXT_LIMIT else "reject_context_overflow"),
        "prompt_tokens": prompt_tokens,
        "max_generation_tokens": PLANNER_MAX_TOKENS,
        "total_tokens": prompt_tokens + PLANNER_MAX_TOKENS,
        "context_limit": MODEL_CONTEXT_LIMIT,
        "source_parent_turn_count": len(original),
        "source_child_turn_count": 0,
        "challenger_model": api.model,
    }
    atomic_json(args.run_root / "reports/preflight.json", preflight)
    if args.dry_run:
        atomic_json(args.run_root / "config/resolved_canary_config.json", {
            "dry_run": True,
            "preflight": preflight,
            "external_api_calls": 0,
            "solver_calls": 0,
        })
        print(json.dumps(preflight, indent=2))
        return 0
    if preflight["decision"] != "pass":
        raise RuntimeError("temporal_outline_context_overflow")

    round_root = args.run_root / "rounds/round_01"
    round_root.mkdir(parents=True)
    round_manifest = {
        **manifest,
        "run_id": round_root.name,
        "mode": "live_graph_expansion",
        "round_index": 1,
        "tokenizer_path": str(TOKENIZER),
        "sampling_parameters": {
            "temperature": api.temperature,
            "planner_max_tokens": PLANNER_MAX_TOKENS,
            "realizer_max_tokens": api.realizer_max_tokens,
        },
        "budget": {
            "outline_normal": (0 if args.accepted_outline_source else 1),
            "outline_format_retry": (0 if args.accepted_outline_source else 1),
            "realizer_physical_max": 30,
            "questioner_physical_max": 2,
            "verifier": 0,
            "training": 0,
        },
    }
    atomic_json(round_root / "config/run_manifest.json", round_manifest)
    client = OpenAICompatibleChallengerClient(api)
    if args.accepted_outline_source is not None:
        raw = json.loads(args.accepted_outline_source.read_text(
            encoding="utf-8"))
        accepted_validation = validate_temporal_outline_plan(
            raw, original_turns=original)
        if accepted_validation["decision"] != "temporal_outline_accept":
            raise ValueError("reused_temporal_outline_no_longer_valid")
        planner = {
            "stage": "temporal_outline_planner",
            "logical_request_id": "reused-accepted-temporal-outline",
            "semantic_prompt_sha256": manifest[
                "temporal_outline_prompt_sha256"],
            "attempts": [],
            "physical_request_count": 0,
            "format_retry_used": False,
            "final_decision": "accepted_outline_reused",
            "accepted": True,
            "accepted_raw": raw,
            "accepted_validation": accepted_validation,
            "accepted_outline_source": str(
                args.accepted_outline_source.resolve()),
            "accepted_outline_source_sha256": sha256_file(
                args.accepted_outline_source),
        }
    else:
        ledger = RequestLedger(round_root, sha256_file(
            round_root / "config/run_manifest.json"), "live")
        budget = ApiBudget(normal_caps={"temporal_outline_planner": 1},
                           format_retry_cap=1, physical_cap=2)
        planner = run_single_object_task(
            run_root=round_root,
            ledger=ledger,
            backend=LiveSemanticRepairBackend(client, PLANNER_MAX_TOKENS),
            budget=budget,
            stage="temporal_outline_planner",
            logical_request_id="temporal-outline-round-01",
            semantic_prompt=prompt,
            lineage_view=_view(original),
            validator=lambda raw: validate_temporal_outline_plan(
                raw, original_turns=original),
            schema_retry_decisions=("temporal_outline_schema_invalid",),
        )
    atomic_json(round_root / "reports/outline_planner_request.json", {
        key: value for key, value in planner.items()
        if key not in {"accepted_raw", "accepted_validation"}})
    if not planner["accepted"]:
        report = {
            "canary_version": TEMPORAL_INSERTION_CANARY_VERSION,
            "decision": planner["final_decision"],
            "expansion_completed": False,
            "questioner_called": False,
            "solver_called": False,
        }
        atomic_json(args.run_root /
                    "reports/temporal_insertion_canary.json", report)
        atomic_json(args.run_root / "reports/api_usage.json",
                    _usage(args.run_root))
        write_post_run_provenance(args.run_root, PROJECT, manifest)
        print(json.dumps(report, indent=2))
        return 0

    raw = planner["accepted_raw"]
    outline = normalize_temporal_outline_plan(
        raw, original_turns=original, round_index=1)
    contracts = compile_temporal_scene_contracts(
        outline, original_turns=original)
    atomic_json(round_root / "frozen/raw_outline_plan.json", raw)
    atomic_json(round_root / "frozen/temporal_expansion_outline.json", outline)
    atomic_json(round_root / "frozen/temporal_scene_contracts.json", {
        "version": TEMPORAL_GRAPH_CONTRACT_VERSION,
        "contracts": contracts,
        "contracts_sha256": canonical_sha256(contracts),
    })
    config = _resolved_config(root=round_root, api=api, outline=outline)
    atomic_json(round_root / "config/resolved_canary_config.json", config)
    orchestrator = TemporalInsertionOrchestrator(
        temporal_contracts=contracts,
        run_root=round_root,
        manifest=round_manifest,
        config=config,
        data={"outline": outline,
              "parent": {"dialogue": {"turns": original}}},
        tokenizer=tokenizer,
        realizer=LiveRealizerBackend(client),
        questioner=None,
        solver=None,
        backend_mode="live",
    )
    if args.accepted_sessions_source_root is not None:
        source_accepted = args.accepted_sessions_source_root.resolve() / \
            "accepted"
        source_index = json.loads((source_accepted / "index.json").read_text(
            encoding="utf-8"))
        names = sorted((source_index.get("session_sha256") or {}).keys())
        if names != [f"session_{index:02d}.json"
                     for index in range(1, len(names) + 1)]:
            raise RuntimeError("accepted_sessions_source_not_contiguous")
        target_accepted = round_root / "accepted"
        target_accepted.mkdir(parents=True, exist_ok=True)
        target_hashes = {}
        for index, name in enumerate(names, 1):
            payload = json.loads((source_accepted / name).read_text(
                encoding="utf-8"))
            if (payload.get("graph_scene_id") != f"S{index}" or
                    payload.get("graph_scene_contract_sha256") !=
                    contracts[index - 1]["contract_sha256"]):
                raise RuntimeError("accepted_session_contract_mismatch")
            target_path = target_accepted / name
            atomic_json(target_path, payload)
            target_hashes[name] = sha256_file(target_path)
        atomic_json(target_accepted / "index.json", {
            "schema_version": "accepted_session_integrity_v2",
            "run_id": round_manifest["run_id"],
            "run_manifest_sha256": sha256_file(
                round_root / "config/run_manifest.json"),
            "session_sha256": target_hashes,
            "seeded_from_root": str(
                args.accepted_sessions_source_root.resolve()),
            "seeded_source_index_sha256": manifest[
                "accepted_sessions_source_index_sha256"],
        })
    sessions = orchestrator.realize()
    child, _ = orchestrator.freeze(sessions)
    child_tokens = orchestrator._tokens(child)
    coverage = graph_scene_coverage_report(
        outline, contracts, sessions, child_tokens)
    atomic_json(round_root / "reports/stage_coverage.json", coverage)
    if not coverage["complete"]:
        raise RuntimeError("temporal_expansion_operationally_incomplete")
    merged = merge_temporal_insertions(
        original_turns=original, sessions=sessions, contracts=contracts,
        round_index=1)
    combined = merged["turns"]
    atomic_json(round_root / "frozen/remapped_inserted_turns.json", {
        "turns": merged["inserted_turns"]})
    atomic_json(round_root / "frozen/insertion_manifest.json", {
        "insertions": merged["insertion_manifest"]})
    atomic_json(round_root / "frozen/interleaved_dialogue.json", {
        "turns": combined,
        "original_turn_count": merged["original_turn_count"],
        "inserted_turn_count": merged["inserted_turn_count"],
        "combined_turn_count": merged["combined_turn_count"],
        "original_order_preserved": merged["original_order_preserved"],
    })
    _write_readable_dialogue(
        round_root / "review/interleaved_dialogue.md", combined)
    expansion_report = {
        "decision": "round_complete",
        "input_turn_count": len(original),
        "input_dialogue_tokens": _tokens(tokenizer, original),
        "inserted_turn_count": len(merged["inserted_turns"]),
        "inserted_dialogue_tokens": _tokens(
            tokenizer, merged["inserted_turns"]),
        "output_turn_count": len(combined),
        "output_dialogue_tokens": _tokens(tokenizer, combined),
        "original_order_preserved": True,
        "insertion_gap_count": len({
            (row["after_session_id"], row["before_session_id"])
            for row in merged["insertion_manifest"]}),
        "fixed_positional_stage_schedule_used": False,
        "coverage": coverage,
        "semantic_transition_completion_claimed": False,
        "character_report": _character_report(
            original=original, combined=combined, outline=outline),
    }
    atomic_json(round_root / "reports/round_expansion.json", expansion_report)

    qa, _solver_backend = _questioner_round_without_stopping_expansion(
        root=round_root,
        round_index=1,
        cumulative=combined,
        tokenizer=tokenizer,
        api=api,
        feedback=feedback,
        feedback_sha=sha256_file(args.feedback_source),
        snapshot=snapshot,
        project_diff=project_diff,
        solver_backend=None,
    )
    decisions = collections.Counter(qa.get("final_decisions") or {})
    result = {
        "canary_version": TEMPORAL_INSERTION_CANARY_VERSION,
        "route": (
            "full_parent_dialogue_to_temporal_outline_to_interleaved_"
            "dialogue_to_unchanged_questioner_solver"),
        "decision": "completed",
        "challenger_model": api.model,
        "baseline_turn_count": len(original),
        "baseline_dialogue_tokens": _tokens(tokenizer, original),
        "final_turn_count": len(combined),
        "final_dialogue_tokens": _tokens(tokenizer, combined),
        "inserted_turn_count": len(merged["inserted_turns"]),
        "insertion_gap_count": expansion_report["insertion_gap_count"],
        "character_report": expansion_report["character_report"],
        "raw_candidate_count": int(qa.get("raw_candidate_count") or 0),
        "minimal_accepted_count": int(qa.get("minimal_accepted_count") or 0),
        "solver_evaluated_count": int(qa.get("solver_evaluated_count") or 0),
        "grounding_pass_count": int(qa.get("grounding_pass_count") or 0),
        "retain_frontier_count": int(qa.get("retain_frontier_count") or 0),
        "final_decisions": dict(sorted(decisions.items())),
        "questioner_report": qa,
        "questioner_semantics_changed": False,
        "semantic_repair_used": False,
        "distractor_stage_used": False,
        "verifier_generations": 0,
        "training_runs": 0,
        "historical_runs_modified": False,
    }
    atomic_json(args.run_root / "frozen/final_dialogue.json", {
        "turns": combined})
    _write_readable_dialogue(
        args.run_root / "review/complete_interleaved_dialogue.md", combined)
    atomic_json(args.run_root / "reports/temporal_insertion_canary.json", result)
    atomic_json(args.run_root / "reports/api_usage.json", _usage(args.run_root))
    write_post_run_provenance(round_root, PROJECT, round_manifest)
    provenance = write_post_run_provenance(
        args.run_root, PROJECT, manifest)
    print(json.dumps({
        "run_root": str(args.run_root),
        "summary": result,
        "api_usage": _usage(args.run_root),
        "provenance": provenance,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
