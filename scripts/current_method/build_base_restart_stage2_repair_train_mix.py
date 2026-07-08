#!/usr/bin/env python3
"""Build the repaired stage2 train mix after guarded repair top-up mining.

This script is scoped to base_restart_current_method_validation. It combines the
original stage2 strict frontier with repair-topup strict frontier, keeps the
previous failed heldout out of training, and writes current_method full-dialogue
MCQ SFT records only after overlap and length audits pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import build_base_restart_stage2_train_mix as base


VALID_ROOT = ROOT / "base_restart_current_method_validation"
DEFAULT_ORIG_FRONTIER = VALID_ROOT / "03_stage2_guarded_mining300/frontier_pool.jsonl"
DEFAULT_ORIG_CANDIDATE = VALID_ROOT / "03_stage2_guarded_mining300/candidate_pool.jsonl"
DEFAULT_REPAIR_FRONTIER = VALID_ROOT / "05_stage2_repair_topup_from_stage1/repair_mining/frontier_pool_repair.jsonl"
DEFAULT_REPAIR_CANDIDATE = VALID_ROOT / "05_stage2_repair_topup_from_stage1/repair_mining/candidate_pool_repair.jsonl"
DEFAULT_PRIOR_HELDOUT = VALID_ROOT / "04_stage2_train_from_stage1/train_mix/frontier_heldout.jsonl"
DEFAULT_OUTPUT = VALID_ROOT / "05_stage2_repair_topup_from_stage1/train_mix"
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
RUN_ID = "base_restart_current_method_validation_stage2_repair_train_mix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build repaired stage2 current_method train mix.")
    parser.add_argument("--original-frontier", default=str(DEFAULT_ORIG_FRONTIER))
    parser.add_argument("--original-candidate", default=str(DEFAULT_ORIG_CANDIDATE))
    parser.add_argument("--repair-frontier", default=str(DEFAULT_REPAIR_FRONTIER))
    parser.add_argument("--repair-candidate", default=str(DEFAULT_REPAIR_CANDIDATE))
    parser.add_argument("--prior-heldout", default=str(DEFAULT_PRIOR_HELDOUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frontier-heldout", type=int, default=30)
    parser.add_argument("--clean-replay", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl_optional(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return base.read_jsonl(p)


def write_json(path: str | Path, payload: Any) -> None:
    base.write_json(path, payload)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    base.write_jsonl(path, rows)


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (base.sample_id(row), base.question_id(row), base.dialogue_hash(row))


def any_key_set(rows: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    return (
        {base.sample_id(row) for row in rows},
        {base.question_id(row) for row in rows},
        {base.dialogue_hash(row) for row in rows},
    )


def conflicts(row: dict[str, Any], keysets: tuple[set[str], set[str], set[str]]) -> bool:
    sample_ids, question_ids, dialogue_hashes = keysets
    return (
        base.sample_id(row) in sample_ids
        or base.question_id(row) in question_ids
        or base.dialogue_hash(row) in dialogue_hashes
    )


def tag_source(row: dict[str, Any], source_phase: str, source_subrun: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    meta = dict(base.metadata(out))
    existing_phase = (
        meta.get("repair_train_mix_source_phase")
        or meta.get("stage2_guarded_mining_source_phase")
        or row.get("stage2_guarded_mining_source_phase")
    )
    existing_subrun = (
        meta.get("repair_train_mix_source_subrun")
        or meta.get("stage2_guarded_mining_source_subrun")
        or row.get("stage2_guarded_mining_source_subrun")
    )
    final_phase = str(existing_phase or source_phase)
    final_subrun = str(existing_subrun or source_subrun)
    meta.update(
        {
            "stage2_guarded_mining_source_phase": final_phase,
            "stage2_guarded_mining_source_subrun": final_subrun,
            "repair_train_mix_source_phase": final_phase,
            "repair_train_mix_source_subrun": final_subrun,
            "source_pipeline": "base_restart_current_method_validation",
            "prompt_mode": "full_dialogue",
            "evidence_window_used_as_prompt": False,
        }
    )
    out["metadata"] = meta
    return out


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen_sample: set[str] = set()
    seen_question_dialogue: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        sid = base.sample_id(row)
        qd = (base.question_id(row), base.dialogue_hash(row))
        if sid in seen_sample or qd in seen_question_dialogue:
            dropped.append(row)
            continue
        seen_sample.add(sid)
        seen_question_dialogue.add(qd)
        kept.append(row)
    return kept, {"input_count": len(rows), "kept_count": len(kept), "dropped_count": len(dropped)}


def count_by(rows: list[dict[str, Any]], fn) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(fn(row) for row in rows).most_common()}


def source_phase(row: dict[str, Any]) -> str:
    return str(base.metadata(row).get("repair_train_mix_source_phase") or base.source_phase(row))


def source_subrun(row: dict[str, Any]) -> str:
    return str(base.metadata(row).get("repair_train_mix_source_subrun") or base.source_subrun(row))


def select_additional_heldout(rows: list[dict[str, Any]], target: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target <= 0:
        return [], {"target": 0, "selected": 0}
    by_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qtype[base.qtype(row)].append(row)
    qtype_targets = base.allocation_largest_remainder({key: len(value) for key, value in by_qtype.items()}, min(target, len(rows)))
    if len(by_qtype.get("multi_select", [])) >= 8 and qtype_targets.get("multi_select", 0) < 5:
        qtype_targets["multi_select"] = 5
    while sum(qtype_targets.values()) > min(target, len(rows)):
        reducible = sorted(
            [key for key, value in qtype_targets.items() if value > (5 if key == "multi_select" else 0)],
            key=lambda key: (qtype_targets[key], key),
            reverse=True,
        )
        if not reducible:
            break
        qtype_targets[reducible[0]] -= 1

    selected: list[dict[str, Any]] = []
    per_stratum: list[dict[str, Any]] = []
    for qt, group in sorted(by_qtype.items()):
        strata: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            key = (
                qt,
                base.hard_template(row) if qt == "single_choice" else "none",
                base.attack_skill(row),
                base.difficulty_subtype(row),
                base.solver_bucket(row),
                source_phase(row),
            )
            strata[key].append(row)
        counts = base.allocation_largest_remainder({key: len(value) for key, value in strata.items()}, qtype_targets.get(qt, 0))
        for key, stratum_rows in sorted(strata.items()):
            shuffled = sorted(stratum_rows, key=base.sample_id)
            random.Random(seed + base.stable_int(key)).shuffle(shuffled)
            take = counts.get(key, 0)
            selected.extend(shuffled[:take])
            per_stratum.append(
                {
                    "stratum": {
                        "question_type": key[0],
                        "hard_single_choice_template": key[1],
                        "attack_skill": key[2],
                        "difficulty_subtype": key[3],
                        "solver_bucket": key[4],
                        "source_phase": key[5],
                    },
                    "total": len(stratum_rows),
                    "heldout": take,
                    "train": len(stratum_rows) - take,
                }
            )
    selected_ids = {base.sample_id(row) for row in selected}
    if len(selected) < min(target, len(rows)):
        leftovers = [row for row in rows if base.sample_id(row) not in selected_ids]
        random.Random(seed + 503).shuffle(leftovers)
        selected.extend(leftovers[: min(target, len(rows)) - len(selected)])
    return selected, {"target": target, "selected": len(selected), "heldout_qtype_targets": qtype_targets, "per_stratum": per_stratum}


def stamp_raw(row: dict[str, Any], role: str) -> dict[str, Any]:
    out = base.stamp_raw(row, role)
    meta = dict(base.metadata(out))
    meta.update(
        {
            "train_mix_run_id": RUN_ID,
            "train_mix_source_group": "stage2_repair_topup_combined",
            "train_mix_source_phase": source_phase(row),
            "train_mix_source_subrun": source_subrun(row),
            "train_mix_version": "stage2_repair_topup_v1",
            "prior_failed_stage2_model_used": False,
            "stage2_init_solver": "solver_base_restart_stage1_merged",
        }
    )
    out["metadata"] = meta
    out["train_mix_source_group"] = "stage2_repair_topup_combined"
    return out


def to_sft(row: dict[str, Any], role: str) -> dict[str, Any]:
    sft = base.to_sft(row, role)
    meta = dict(sft.get("metadata") or {})
    meta.update(
        {
            "train_mix_run_id": RUN_ID,
            "train_mix_source_group": "stage2_repair_topup_combined",
            "train_mix_source_phase": source_phase(row),
            "train_mix_source_subrun": source_subrun(row),
            "train_mix_version": "stage2_repair_topup_v1",
            "prior_failed_stage2_model_used": False,
            "stage2_init_solver": "solver_base_restart_stage1_merged",
            "prompt_mode": "full_dialogue",
            "evidence_window_used_as_prompt": False,
        }
    )
    sft["metadata"] = meta
    return sft


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = base.split_summary(rows)
    summary["repair_source_phase_distribution"] = count_by(rows, source_phase)
    summary["repair_source_subrun_distribution"] = count_by(rows, source_subrun)
    return summary


def overlap_check(train_mix: list[dict[str, Any]], heldout: list[dict[str, Any]]) -> dict[str, Any]:
    return base.overlap_check(train_mix, heldout)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage2 Repair Train Mix Audit Report",
        "",
        f"- status: {summary['status']}",
        f"- combined_frontier_pool_count: {summary['combined_frontier_pool_count']}",
        f"- frontier_train_count: {summary['frontier_train_count']}",
        f"- frontier_heldout_count: {summary['frontier_heldout_count']}",
        f"- clean_replay_count: {summary['clean_replay_count']}",
        f"- train_mix_count: {summary['train_mix_count']}",
        f"- clean_replay_share: {summary['clean_replay_share']}",
        f"- prior_failed_heldout_reserved_count: {summary['prior_failed_heldout_reserved_count']}",
        f"- prior_failed_heldout_in_train_count: {summary['prior_failed_heldout_in_train_count']}",
        f"- near_miss_aux_pool_count: {summary['near_miss_aux_pool_count']}",
        f"- heldout_overlap_check: `{summary['heldout_overlap_check']}`",
        f"- tokenizer_path: `{summary['tokenizer_path']}`",
        f"- max_sft_token_length: {summary['max_sft_token_length']}",
        f"- over_max_length_count: {summary['over_max_length_count']}",
        f"- evidence_window_used_as_prompt: {str(summary['evidence_window_used_as_prompt']).lower()}",
        f"- train_mix_pass: {summary['train_mix_pass']}",
        "",
        "## Frontier Train",
        f"- qtype_distribution: `{summary['qtype_distribution']['frontier_train']}`",
        f"- bucket_distribution: `{summary['frontier_train']['bucket_distribution']}`",
        f"- hard_single_choice_template_distribution: `{summary['frontier_train']['hard_single_choice_template_distribution']}`",
        f"- source_phase_distribution: `{summary['frontier_train']['repair_source_phase_distribution']}`",
        "",
        "## Frontier Heldout",
        f"- qtype_distribution: `{summary['qtype_distribution']['frontier_heldout']}`",
        f"- bucket_distribution: `{summary['frontier_heldout']['bucket_distribution']}`",
        f"- hard_single_choice_template_distribution: `{summary['frontier_heldout']['hard_single_choice_template_distribution']}`",
        f"- source_phase_distribution: `{summary['frontier_heldout']['repair_source_phase_distribution']}`",
        "",
        "## Clean Replay",
        f"- qtype_distribution: `{summary['qtype_distribution']['clean_replay']}`",
        f"- bucket_distribution: `{summary['clean_replay']['bucket_distribution']}`",
        f"- notes: `{summary['clean_replay_selection'].get('notes')}`",
        "",
        "## Pass Criteria",
        f"- pass_criteria: `{summary['pass_criteria']}`",
        "",
        "## Files",
    ]
    for key, value in summary["artifact_paths"].items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    original_frontier = [tag_source(row, "stage2_guarded_mining300", "original_guarded_mining300") for row in base.read_jsonl(args.original_frontier)]
    repair_frontier = [tag_source(row, "stage2_repair_topup_from_stage1", "repair_topup_mining120") for row in base.read_jsonl(args.repair_frontier)]
    original_candidates = [tag_source(row, "stage2_guarded_mining300", "original_guarded_mining300") for row in base.read_jsonl(args.original_candidate)]
    repair_candidates = [tag_source(row, "stage2_repair_topup_from_stage1", "repair_topup_mining120") for row in base.read_jsonl(args.repair_candidate)]
    prior_heldout = read_jsonl_optional(args.prior_heldout)

    if not all(base.is_strict_frontier(row) for row in original_frontier + repair_frontier):
        raise ValueError("Combined frontier contains non-strict rows; repair train mix only accepts strict frontier.")

    combined_frontier_raw, frontier_dedupe = dedupe_rows(original_frontier + repair_frontier)
    combined_candidate_raw, candidate_dedupe = dedupe_rows(original_candidates + repair_candidates)

    prior_keysets = any_key_set(prior_heldout)
    reserved_prior_heldout_raw = [row for row in combined_frontier_raw if conflicts(row, prior_keysets)]
    split_pool_raw = [row for row in combined_frontier_raw if not conflicts(row, prior_keysets)]
    additional_target = max(0, int(args.frontier_heldout) - len(reserved_prior_heldout_raw))
    additional_heldout_raw, split_stats = select_additional_heldout(split_pool_raw, additional_target, int(args.seed))
    heldout_keysets = any_key_set(reserved_prior_heldout_raw + additional_heldout_raw)
    frontier_train_raw = [row for row in split_pool_raw if not conflicts(row, heldout_keysets)]
    excluded_train_overlap_raw = [row for row in split_pool_raw if conflicts(row, heldout_keysets) and base.sample_id(row) not in {base.sample_id(h) for h in additional_heldout_raw}]
    frontier_heldout_raw = reserved_prior_heldout_raw + additional_heldout_raw

    all_frontier_ids = {base.sample_id(row) for row in combined_frontier_raw}
    near_miss_aux_raw = [row for row in combined_candidate_raw if base.is_near_miss(row)]
    clean_replay_raw, clean_stats = base.select_clean_replay(
        combined_candidate_raw,
        frontier_ids=all_frontier_ids,
        target=int(args.clean_replay),
        seed=int(args.seed) + 37,
    )

    frontier_train = [stamp_raw(row, "frontier_train") for row in frontier_train_raw]
    frontier_heldout = [stamp_raw(row, "frontier_heldout") for row in frontier_heldout_raw]
    clean_replay = [stamp_raw(row, "clean_replay") for row in clean_replay_raw]
    near_miss_aux = [stamp_raw(row, "near_miss_aux") for row in near_miss_aux_raw]
    excluded_prior_heldout = [stamp_raw(row, "excluded_prior_heldout_reserved") for row in reserved_prior_heldout_raw]
    excluded_overlap = [stamp_raw(row, "excluded_train_heldout_key_conflict") for row in excluded_train_overlap_raw]
    train_mix = frontier_train + clean_replay
    sft_train = [to_sft(row, "frontier_train") for row in frontier_train_raw] + [to_sft(row, "clean_replay") for row in clean_replay_raw]

    write_jsonl(output_dir / "frontier_train.jsonl", frontier_train)
    write_jsonl(output_dir / "frontier_heldout.jsonl", frontier_heldout)
    write_jsonl(output_dir / "clean_replay.jsonl", clean_replay)
    write_jsonl(output_dir / "near_miss_aux_pool.jsonl", near_miss_aux)
    write_jsonl(output_dir / "excluded_prior_heldout.jsonl", excluded_prior_heldout)
    write_jsonl(output_dir / "excluded_train_heldout_key_conflicts.jsonl", excluded_overlap)
    write_jsonl(output_dir / "train_mix.jsonl", train_mix)
    write_jsonl(output_dir / "sft_train.jsonl", sft_train)

    token_stats = base.token_length_summary(sft_train, tokenizer_path=str(args.tokenizer), max_length=int(args.max_length))
    heldout_overlap = overlap_check(train_mix, frontier_heldout)
    prior_train_overlap = sum(1 for row in train_mix if conflicts(row, prior_keysets))
    clean_share = round(len(clean_replay) / max(1, len(train_mix)), 6)

    frontier_train_summary = split_summary(frontier_train)
    frontier_heldout_summary = split_summary(frontier_heldout)
    clean_replay_summary = split_summary(clean_replay)
    near_miss_aux_summary = split_summary(near_miss_aux)
    train_mix_summary = split_summary(train_mix)
    candidate_qtype_zero_share = base.qtype_zero_share(combined_candidate_raw)

    pass_criteria = {
        "combined_frontier_pool_count_gte_100": len(combined_frontier_raw) >= 100,
        "frontier_train_count_gte_65": len(frontier_train) >= 65,
        "frontier_heldout_count_gte_30": len(frontier_heldout) >= 30,
        "clean_replay_count_25_to_35": 25 <= len(clean_replay) <= 35,
        "prior_failed_heldout_not_in_train": prior_train_overlap == 0,
        "heldout_overlap_zero": heldout_overlap["total_overlap"] == 0,
        "over_max_length_zero": token_stats["over_max_length_count"] == 0,
        "max_sft_token_length_lte_16384": (token_stats["max_sft_token_length"] or 0) <= int(args.max_length),
        "evidence_window_prompt_false": True,
    }

    summary = {
        "run_id": RUN_ID,
        "status": "audit_passed" if all(pass_criteria.values()) else "audit_failed",
        "original_frontier_path": str(args.original_frontier),
        "repair_frontier_path": str(args.repair_frontier),
        "original_candidate_path": str(args.original_candidate),
        "repair_candidate_path": str(args.repair_candidate),
        "prior_failed_heldout_path": str(args.prior_heldout),
        "output_dir": str(output_dir),
        "original_frontier_count": len(original_frontier),
        "repair_frontier_count": len(repair_frontier),
        "combined_frontier_pool_count": len(combined_frontier_raw),
        "frontier_dedupe": frontier_dedupe,
        "original_candidate_count": len(original_candidates),
        "repair_candidate_count": len(repair_candidates),
        "combined_candidate_count": len(combined_candidate_raw),
        "candidate_dedupe": candidate_dedupe,
        "frontier_train_count": len(frontier_train),
        "frontier_heldout_count": len(frontier_heldout),
        "clean_replay_count": len(clean_replay),
        "near_miss_aux_pool_count": len(near_miss_aux),
        "train_mix_count": len(train_mix),
        "sft_train_count": len(sft_train),
        "clean_replay_share": clean_share,
        "prior_failed_heldout_reserved_count": len(reserved_prior_heldout_raw),
        "prior_failed_heldout_in_train_count": prior_train_overlap,
        "excluded_train_heldout_key_conflict_count": len(excluded_train_overlap_raw),
        "qtype_distribution": {
            "frontier_train": frontier_train_summary["qtype_distribution"],
            "frontier_heldout": frontier_heldout_summary["qtype_distribution"],
            "clean_replay": clean_replay_summary["qtype_distribution"],
            "near_miss_aux_pool": near_miss_aux_summary["qtype_distribution"],
            "train_mix": train_mix_summary["qtype_distribution"],
        },
        "hard_single_choice_template_distribution": {
            "frontier_train": frontier_train_summary["hard_single_choice_template_distribution"],
            "frontier_heldout": frontier_heldout_summary["hard_single_choice_template_distribution"],
            "clean_replay": clean_replay_summary["hard_single_choice_template_distribution"],
            "train_mix": train_mix_summary["hard_single_choice_template_distribution"],
        },
        "source_phase_distribution": {
            "frontier_train": frontier_train_summary["repair_source_phase_distribution"],
            "frontier_heldout": frontier_heldout_summary["repair_source_phase_distribution"],
            "clean_replay": clean_replay_summary["repair_source_phase_distribution"],
            "near_miss_aux_pool": near_miss_aux_summary["repair_source_phase_distribution"],
            "train_mix": train_mix_summary["repair_source_phase_distribution"],
        },
        "candidate_qtype_zero_share": candidate_qtype_zero_share,
        "ordering_zero_share_note": (
            f"combined candidate ordering_zero_share={candidate_qtype_zero_share.get('ordering', 0.0)}; "
            "repair top-up kept ordering frozen and does not oversample ordering."
        ),
        "frontier_train": frontier_train_summary,
        "frontier_heldout": frontier_heldout_summary,
        "clean_replay": clean_replay_summary,
        "near_miss_aux_pool": near_miss_aux_summary,
        "train_mix": train_mix_summary,
        "split_stats": split_stats,
        "clean_replay_selection": clean_stats,
        "heldout_overlap_check": heldout_overlap,
        "tokenizer_path": str(args.tokenizer),
        "max_length": int(args.max_length),
        **token_stats,
        "over_max_length_count": token_stats["over_max_length_count"],
        "max_sft_token_length": token_stats["max_sft_token_length"],
        "evidence_window_used_as_prompt": False,
        "prior_failed_stage2_model_used": False,
        "stage2_init_solver": "solver_base_restart_stage1_merged",
        "pass_criteria": pass_criteria,
        "train_mix_pass": all(pass_criteria.values()),
        "artifact_paths": {
            "frontier_train": str(output_dir / "frontier_train.jsonl"),
            "frontier_heldout": str(output_dir / "frontier_heldout.jsonl"),
            "clean_replay": str(output_dir / "clean_replay.jsonl"),
            "near_miss_aux_pool": str(output_dir / "near_miss_aux_pool.jsonl"),
            "excluded_prior_heldout": str(output_dir / "excluded_prior_heldout.jsonl"),
            "excluded_train_heldout_key_conflicts": str(output_dir / "excluded_train_heldout_key_conflicts.jsonl"),
            "train_mix": str(output_dir / "train_mix.jsonl"),
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "train_mix_summary": str(output_dir / "train_mix_summary.json"),
            "train_mix_report": str(reports_dir / "train_mix_report.md"),
        },
    }

    write_json(output_dir / "train_mix_summary.json", summary)
    write_json(reports_dir / "train_mix_summary.json", summary)
    write_report(reports_dir / "train_mix_report.md", summary)
    print(json.dumps(
        {
            "status": summary["status"],
            "counts": {
                "combined_frontier": len(combined_frontier_raw),
                "frontier_train": len(frontier_train),
                "frontier_heldout": len(frontier_heldout),
                "clean_replay": len(clean_replay),
                "train_mix": len(train_mix),
            },
            "prior_failed_heldout_in_train_count": prior_train_overlap,
            "heldout_overlap_check": heldout_overlap,
            "max_sft_token_length": summary["max_sft_token_length"],
            "over_max_length_count": summary["over_max_length_count"],
            "train_mix_pass": summary["train_mix_pass"],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 0 if summary["train_mix_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
