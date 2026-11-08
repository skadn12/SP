"""Offline final consistency audit and review exports for a completed batch."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text
from .run import PROBE_VERSION, _generation_coverage, _root_dir
from .seed_schedule import stable_seed


TECHNICAL_FAILURES = {
    "parse_failure", "parse_error", "protocol_failure", "runtime_error",
    "runtime_failure", "technical_failure", "timeout", "oom", "OOM",
    "out_of_memory", "guided_decoding_error", "backend_failure",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gold_texts(candidate: Mapping[str, Any]) -> list[str]:
    qtype = str(candidate.get("question_type") or "")
    if qtype == "single_choice":
        options = candidate.get("options") or {}
        return [str(options.get(str(candidate.get("correct_option"))) or "")]
    if qtype == "multi_select":
        options = candidate.get("options") or {}
        return [str(options.get(str(key)) or "") for key in candidate.get("gold") or []]
    items = candidate.get("items") or {}
    return [str(items.get(str(key)) or "") for key in candidate.get("gold") or []]


def _review_item(candidate: Mapping[str, Any], difficulty: Mapping[str, Any],
                 solver: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "root_id": candidate.get("root_id"),
        "intent_id": candidate.get("intent_id"),
        "generation_source": candidate.get("generation_source"),
        "question_type": candidate.get("question_type"),
        "question": candidate.get("question"),
        "options": candidate.get("options") or {},
        "items": candidate.get("items") or {},
        "correct_option": candidate.get("correct_option"),
        "gold": candidate.get("gold") or [],
        "gold_texts": _gold_texts(candidate),
        "evidence_refs": candidate.get("evidence_refs") or [],
        "evidence_groups": candidate.get("evidence_groups") or {},
        "reasoning_groups_used": candidate.get("reasoning_groups_used") or [],
        "hypothesized_reasoning_type": candidate.get("hypothesized_reasoning_type"),
        "discriminative_variable": candidate.get("discriminative_variable"),
        "difficulty": dict(difficulty),
        "position_diagnostics": solver.get("position_diagnostics") or {},
        "context_diagnostics": solver.get("context_diagnostics") or {},
        "gold_rebinding_audit": solver.get("gold_rebinding_audit") or {},
        "semantic_grounding_reviewed": False,
    }


def _markdown(items: list[Mapping[str, Any]], title: str) -> str:
    lines = [f"# {title}", "", (
        "These are Solver-screened candidates only. Evidence refs exist, but "
        "semantic grounding and gold correctness have not been adjudicated."
    ), ""]
    for index, item in enumerate(items, 1):
        difficulty = item.get("difficulty") or {}
        lines.extend([
            f"## {index}. {item.get('candidate_id')}", "",
            f"- Root: `{item.get('root_id')}`",
            f"- Type/source: `{item.get('question_type')}` / `{item.get('generation_source')}`",
            f"- Full/no-context: `{difficulty.get('full_context_accuracy')}` / `{difficulty.get('no_context_accuracy')}`",
            f"- Position range: `{difficulty.get('position_accuracy_range')}`",
            f"- Evidence refs: `{json.dumps(item.get('evidence_refs') or [], ensure_ascii=False)}`",
            "",
            str(item.get("question") or ""),
            "",
        ])
        container = item.get("items") or item.get("options") or {}
        lines.extend(f"- {key}: {value}" for key, value in container.items())
        lines.extend([
            "",
            f"Gold: `{json.dumps(item.get('gold') or item.get('correct_option'), ensure_ascii=False)}`",
            "",
        ])
    return "\n".join(lines) + "\n"


def build_final_audit(run_root: Path) -> dict[str, Any]:
    manifest = _load(run_root / "manifest.json")
    batch = _load(run_root / "report.json")
    errors: list[str] = []
    warnings: list[str] = []
    all_items: list[dict[str, Any]] = []
    hard_items: list[dict[str, Any]] = []
    invalid_items: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    api_prompt_tokens = api_completion_tokens = api_total_tokens = 0
    api_requests = solver_requests = ambiguous_requests = 0
    effective_generation_requests = 0
    total_rollouts = 0
    expected_root_ids = [str(x) for x in manifest.get("selected_root_ids") or []]

    for root_id in expected_root_ids:
        root_dir = _root_dir(run_root, root_id)
        required = [
            "candidate_pool.json", "candidates_valid.json", "solver_results.json",
            "difficulty.json", "report.json", "checkpoint.json",
            "expanded_dialogue_clean.json", "permutations.json",
        ]
        missing = [name for name in required if not (root_dir / name).is_file()]
        if missing:
            errors.append(f"{root_id}:missing_artifacts:{','.join(missing)}")
            continue
        candidates = _load(root_dir / "candidates_valid.json")
        candidate_by_id = {str(row.get("candidate_id")): row for row in candidates}
        solver_payload = _load(root_dir / "solver_results.json")
        solver_rows = list(solver_payload.get("results") or [])
        solver_by_id = {str(row.get("candidate_id")): row for row in solver_rows}
        difficulty = _load(root_dir / "difficulty.json")
        root_difficulty = list(difficulty.get("candidate_summaries") or [])
        difficulty_by_id = {str(row.get("candidate_id")): row for row in root_difficulty}
        if set(candidate_by_id) != set(solver_by_id):
            errors.append(f"{root_id}:candidate_solver_id_mismatch")
        if set(candidate_by_id) != set(difficulty_by_id):
            errors.append(f"{root_id}:candidate_difficulty_id_mismatch")
        checkpoint = _load(root_dir / "checkpoint.json")
        if checkpoint.get("stage") != "REPORT_COMPLETED":
            errors.append(f"{root_id}:checkpoint:{checkpoint.get('stage')}")
        cleaned = _load(root_dir / "expanded_dialogue_clean.json")
        speakers = sorted(set(str(row.get("speaker") or "") for row in cleaned.get("turns") or []))
        if any(re.search(r"(?:^| )(?:A|B)$", speaker) for speaker in speakers):
            errors.append(f"{root_id}:speaker_alias_remaining")

        coverage = _generation_coverage(root_dir)
        generation_rows.append({"root_id": root_id, **coverage})
        if not coverage["minimum_coverage_complete"]:
            warnings.append(f"{root_id}:generation_minimum_gap")
        generation_records = _load(root_dir / "candidate_generation.json")
        effective_generation_requests += 1 + len(generation_records)

        state_dir = root_dir / "ledger" / "requests"
        for state_path in state_dir.glob("*.json"):
            state = _load(state_path)
            if state.get("status") != "completed":
                ambiguous_requests += 1
                errors.append(f"{root_id}:ambiguous_request:{state.get('request_id')}")
                continue
            budget = state.get("budget_class")
            if budget == "fresh_qa_generation":
                api_requests += 1
                usage = ((state.get("response_metadata") or {}).get("usage") or {})
                api_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                api_completion_tokens += int(usage.get("completion_tokens") or 0)
                api_total_tokens += int(usage.get("total_tokens") or 0)
            elif budget == "k3_solver":
                solver_requests += 1

        for result in solver_rows:
            candidate_id = str(result.get("candidate_id"))
            if not bool((result.get("gold_rebinding_audit") or {}).get("pass")):
                errors.append(f"{candidate_id}:gold_rebinding")
            seeds: dict[int, set[int]] = defaultdict(set)
            candidate_container = candidate_by_id.get(candidate_id, {}).get("items") if result.get("question_type") == "ordering" else candidate_by_id.get(candidate_id, {}).get("options")
            expected_variant_count = len(candidate_container or {})
            variant_rows = list(result.get("variants") or [])
            if len(variant_rows) != expected_variant_count * 2:
                errors.append(f"{candidate_id}:variant_mode_count:{len(variant_rows)}")
            for variant in variant_rows:
                summary = variant.get("summary") or {}
                if not summary.get("complete"):
                    errors.append(f"{candidate_id}:{variant.get('variant')}:{variant.get('mode')}:incomplete")
                rollouts = list(variant.get("rollouts") or [])
                if len(rollouts) != 3:
                    errors.append(f"{candidate_id}:{variant.get('variant')}:{variant.get('mode')}:not_k3")
                for rollout in rollouts:
                    total_rollouts += 1
                    slot = int(rollout.get("slot_id"))
                    seed = int(rollout.get("seed"))
                    seeds[slot].add(seed)
                    if seed != stable_seed(root_id, candidate_id, slot):
                        errors.append(f"{candidate_id}:{slot}:seed_drift")
                    if str(rollout.get("failure_type") or "") in TECHNICAL_FAILURES:
                        errors.append(f"{candidate_id}:{rollout.get('request_id')}:technical_failure")
            if set(seeds) != {0, 1, 2} or any(len(values) != 1 for values in seeds.values()):
                errors.append(f"{candidate_id}:paired_seed_failure")

            candidate = candidate_by_id.get(candidate_id)
            summary = difficulty_by_id.get(candidate_id)
            if candidate is None or summary is None:
                continue
            review = _review_item(candidate, summary, result)
            all_items.append(review)
            difficulty_rows.append(dict(summary))
            if summary.get("difficulty_bucket") == "hard_candidate":
                hard_items.append(review)
            elif summary.get("difficulty_bucket") == "invalid_or_ambiguous":
                invalid_items.append(review)

    by_type: dict[str, Any] = {}
    for qtype in sorted({str(row.get("question_type")) for row in difficulty_rows}):
        selected = [row for row in difficulty_rows if row.get("question_type") == qtype]
        counts = Counter(str(row.get("difficulty_bucket")) for row in selected)
        by_type[qtype] = {
            "candidate_count": len(selected),
            "hard_candidate_count": counts.get("hard_candidate", 0),
            "hard_rate": counts.get("hard_candidate", 0) / len(selected) if selected else None,
            "bucket_counts": dict(sorted(counts.items())),
        }
    by_source: dict[str, Any] = {}
    for source in sorted({str(row.get("generation_source")) for row in difficulty_rows}):
        selected = [row for row in difficulty_rows if row.get("generation_source") == source]
        counts = Counter(str(row.get("difficulty_bucket")) for row in selected)
        by_source[source] = {
            "candidate_count": len(selected),
            "hard_candidate_count": counts.get("hard_candidate", 0),
            "hard_rate": counts.get("hard_candidate", 0) / len(selected) if selected else None,
            "bucket_counts": dict(sorted(counts.items())),
        }
    expected_solver_requests = sum(
        len((row.get("items") or {}) if row.get("question_type") == "ordering" else (row.get("options") or {})) * 2 * 3
        for row in all_items
    )
    if expected_solver_requests != solver_requests or total_rollouts != solver_requests:
        errors.append(
            f"solver_request_count:{solver_requests}:{expected_solver_requests}:{total_rollouts}"
        )
    audit = {
        "audit_version": "h4_k3_final_audit_v1",
        "probe_version": PROBE_VERSION,
        "status": "pass" if not errors else "fail",
        "run_root": str(run_root),
        "root_count": len(expected_root_ids),
        "candidate_count": len(all_items),
        "hard_candidate_count": len(hard_items),
        "invalid_or_ambiguous_count": len(invalid_items),
        "semantic_grounding_reviewed": False,
        "gold_semantics_reviewed": False,
        "consistency_errors": errors,
        "warnings": warnings,
        "generation_coverage": {
            "expected_intent_source_batches": sum(
                int(row["expected_intent_source_batches"]) for row in generation_rows),
            "accepted_intent_source_batches": sum(
                int(row["accepted_intent_source_batches"]) for row in generation_rows),
            "failed_intent_source_batches": sum(
                int(row["failed_intent_source_batches"]) for row in generation_rows),
            "roots": generation_rows,
        },
        "request_accounting": {
            "actual_challenger_api_requests": api_requests,
            "effective_current_artifact_generation_requests": effective_generation_requests,
            "superseded_challenger_requests": api_requests - effective_generation_requests,
            "challenger_prompt_tokens": api_prompt_tokens,
            "challenger_completion_tokens": api_completion_tokens,
            "challenger_total_tokens": api_total_tokens,
            "local_solver_requests": solver_requests,
            "expected_local_solver_requests": expected_solver_requests,
            "ambiguous_requests": ambiguous_requests,
        },
        "difficulty_by_question_type": by_type,
        "difficulty_by_generation_source": by_source,
        "batch_summary": batch.get("batch_summary") or {},
        "result_state": "READY_FOR_SEMANTIC_HARD_CANDIDATE_AUDIT" if not errors else "AUDIT_FAILED",
        "interpretation": (
            "The hard list passed structural and execution checks only. It is not training-ready "
            "until semantic evidence/gold adjudication is complete."
        ),
    }
    atomic_json(run_root / "final_audit.json", audit)
    atomic_json(run_root / "all_candidates_with_difficulty.json", {
        "candidate_count": len(all_items), "candidates": all_items,
    })
    atomic_json(run_root / "hard_candidates.json", {
        "candidate_count": len(hard_items), "candidates": hard_items,
    })
    atomic_json(run_root / "invalid_or_ambiguous_candidates.json", {
        "candidate_count": len(invalid_items), "candidates": invalid_items,
    })
    atomic_text(run_root / "hard_candidates.md", _markdown(
        hard_items, "H4 K=3 Hard Candidates - Semantic Review Queue"))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_final_audit(args.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
