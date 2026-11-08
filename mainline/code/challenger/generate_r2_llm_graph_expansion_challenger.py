#!/usr/bin/env python3
"""Second-round LLM-assisted graph-expansion challenger.

This generator starts from first-round expanded clean_v1 graph/state-chain
records, extends the inherited chain, realizes only approved extension facts as
synthetic dialogue turns, and emits validated multi-format QA.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_v13_pilot as v13  # noqa: E402
import run_graph_expansion_challenger_v1 as gen  # noqa: E402


DEFAULT_TRAIN_SOURCE = ROOT / "eval_sets/graph_expansion_challenger_v1/train_v13_pilot"
DEFAULT_HELDOUT_SOURCE = ROOT / "eval_sets/graph_expansion_challenger_v1/heldout_v13_pilot"
DEFAULT_OUTPUT_ROOT = ROOT / "eval_sets/graph_expansion_challenger_v1"
DEFAULT_REPORT_DIR = ROOT / "eval_reports/graph_expansion_challenger_v1/r2_llm_graph_expansion_pilot"
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "eval_reports/graph_expansion_challenger_v1/r2_clean_v1_source_manifest.json"
DEFAULT_AUTO_POLICY = ROOT / "eval_reports/graph_expansion_challenger_v1/r2_llm_clean_v1_parent_medium_pilot_eval/challenger_policy_round2_auto_v2.json"
DEFAULT_AUTO_FULL_DIR = ROOT / "eval_sets/graph_expansion_challenger_v1/r2_auto_full"
DEFAULT_AUTO_FULL_REPORT_DIR = ROOT / "eval_reports/graph_expansion_challenger_v1/r2_auto_full_eval"
DEFAULT_CHALLENGER_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHALLENGER_MODEL = "Qwen/Qwen3-14B"
DEFAULT_SOLVER_BASE = "/root/autodl-tmp/Qwen3-4B"
DEFAULT_SOLVER_ADAPTER = "outputs/qwen3_4b_lora_clean_v1_solver_r1_e1"
OPTION_LABELS = tuple("ABCDEF")
QUESTION_TYPES = ("single_choice", "multi_select", "ordering", "state_reconstruction")
LEAK_PATTERNS = tuple(set(v13.LEAK_PATTERNS + gen.GENERATED_DIALOGUE_LEAK_PATTERNS + gen.QUESTION_LEAK_PATTERNS))
DEFAULT_AUTO_TRAIN_SAMPLES = "sample0,sample1,sample2,sample3,sample4,sample5"
DEFAULT_AUTO_DEV_SAMPLES = "sample6"
DEFAULT_AUTO_HELDOUT_SAMPLES = "sample7,sample8,sample9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build r2 LLM-assisted graph-expansion challenger from first-round expanded clean_v1 data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Project config.json with api.base_url/api.api_key/api.model.")
    parser.add_argument("--auto-full", action="store_true", help="Write the r2_auto_full dataset layout requested by the reward-driven round-2 run.")
    parser.add_argument("--policy", default=None, help="Validated auto policy JSON. Used to fill question mix, candidate pool, and default ranges.")
    parser.add_argument("--train-source", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--heldout-source", default=str(DEFAULT_HELDOUT_SOURCE))
    parser.add_argument("--parent-mode", choices=("v13", "clean_v1"), default="v13")
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-dev", type=int, default=None)
    parser.add_argument("--n-heldout", type=int, default=12)
    parser.add_argument("--heldout-per-split", type=int, default=None)
    parser.add_argument("--train-samples", default=DEFAULT_AUTO_TRAIN_SAMPLES, help="Comma-separated clean_v1 train samples for --auto-full parent discovery.")
    parser.add_argument("--dev-samples", default=DEFAULT_AUTO_DEV_SAMPLES, help="Comma-separated clean_v1 dev samples for --auto-full parent discovery.")
    parser.add_argument("--heldout-samples", default=DEFAULT_AUTO_HELDOUT_SAMPLES, help="Comma-separated clean_v1 heldout samples for --auto-full parent discovery.")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--target-extension-delta", type=int, default=4)
    parser.add_argument("--min-extension-delta", type=int, default=3)
    parser.add_argument("--max-extension-delta", type=int, default=None)
    parser.add_argument("--min-added-turns", type=int, default=8)
    parser.add_argument("--max-added-turns", type=int, default=8)
    parser.add_argument("--min-final-depth", type=int, default=1)
    parser.add_argument("--max-final-depth", type=int, default=9999)
    parser.add_argument("--question-type-mix", default="single_choice:0.25,multi_select:0.25,ordering:0.25,state_reconstruction:0.25")
    parser.add_argument("--candidate-pool-per-parent", type=int, default=3)
    parser.add_argument("--candidate-pool-per-parent-max", type=int, default=5)
    parser.add_argument("--solver-sampling-k", type=int, default=5)
    parser.add_argument("--max-attempt-multiplier", type=int, default=80, help="Maximum generation attempts per requested sample.")
    parser.add_argument("--max-llm-retries", type=int, default=2)
    parser.add_argument("--challenger-base-url", default=DEFAULT_CHALLENGER_BASE_URL)
    parser.add_argument("--challenger-model", default=DEFAULT_CHALLENGER_MODEL)
    parser.add_argument("--challenger-api-key", default=None)
    parser.add_argument("--llm-timeout", type=int, default=90)
    parser.add_argument("--llm-workers", type=int, default=None, help="Concurrent challenger planner calls for auto-full generation.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback plans; kept for ablation only.")
    parser.add_argument("--require-llm-plan", action="store_true", help="Skip samples whose challenger planner call did not return a usable plan.")
    parser.add_argument("--skip-solver-eval", action="store_true")
    parser.add_argument("--solver-base", default=DEFAULT_SOLVER_BASE)
    parser.add_argument("--solver-adapter", default=DEFAULT_SOLVER_ADAPTER)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def parse_sample_names(value: str, *, default: str) -> set[str]:
    raw = value if value is not None else default
    names = {item.strip() for item in str(raw).split(",") if item.strip()}
    if not names:
        raise ValueError("sample list must not be empty")
    return names


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.auto_full:
        if args.policy is None:
            args.policy = str(DEFAULT_AUTO_POLICY)
        if args.output_root == str(DEFAULT_OUTPUT_ROOT):
            args.output_root = str(DEFAULT_AUTO_FULL_DIR)
        if args.report_dir == str(DEFAULT_REPORT_DIR):
            args.report_dir = str(DEFAULT_AUTO_FULL_REPORT_DIR)
        args.parent_mode = "clean_v1"
        if args.n_train == 24:
            args.n_train = 1200
        if args.n_dev is None:
            args.n_dev = 200
        if args.heldout_per_split is None and args.n_heldout == 12:
            args.heldout_per_split = 200
        elif args.heldout_per_split is None:
            args.heldout_per_split = 0
        if args.min_added_turns == 8 and args.max_added_turns == 8:
            args.min_added_turns = 80
            args.max_added_turns = 140
        if args.target_extension_delta == 4:
            args.target_extension_delta = 16
        if args.min_extension_delta == 3:
            args.min_extension_delta = 12
        if args.max_extension_delta is None:
            args.max_extension_delta = 20
        if args.min_final_depth == 1:
            args.min_final_depth = 24
        if args.max_final_depth == 9999:
            args.max_final_depth = 40
        if args.llm_workers is None:
            args.llm_workers = 8

    if args.n_dev is None:
        args.n_dev = 0
    if args.heldout_per_split is None:
        args.heldout_per_split = 0
    if args.llm_workers is None:
        args.llm_workers = 1

    if args.policy:
        policy_path = Path(args.policy)
        if policy_path.exists():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            qmix = policy.get("question_type_mix") if isinstance(policy.get("question_type_mix"), dict) else {}
            if qmix:
                args.question_type_mix = ",".join(f"{key}:{float(qmix.get(key, 0.0))}" for key in QUESTION_TYPES)
            turn_range = policy.get("added_turn_count_range")
            if isinstance(turn_range, list) and len(turn_range) == 2 and not args.auto_full:
                args.min_added_turns = int(turn_range[0])
                args.max_added_turns = int(turn_range[1])
            delta_range = policy.get("state_chain_extension_delta_range")
            if isinstance(delta_range, list) and len(delta_range) == 2 and not args.auto_full:
                args.min_extension_delta = int(delta_range[0])
                args.max_extension_delta = int(delta_range[1])
                args.target_extension_delta = round((args.min_extension_delta + args.max_extension_delta) / 2)
            depth_range = policy.get("final_state_chain_depth_range")
            if isinstance(depth_range, list) and len(depth_range) == 2 and not args.auto_full:
                args.min_final_depth = int(depth_range[0])
                args.max_final_depth = int(depth_range[1])
            args.candidate_pool_per_parent = int(policy.get("candidate_pool_per_parent") or 3)
            args.candidate_pool_per_parent_max = int(policy.get("candidate_pool_per_parent_max") or 5)
            args.solver_sampling_k = int(policy.get("solver_sampling_k") or 5)

    config_path = Path(args.config)
    if not config_path.exists():
        return args
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    api = payload.get("api") if isinstance(payload, dict) else {}
    if isinstance(api, dict):
        if args.challenger_base_url == DEFAULT_CHALLENGER_BASE_URL and api.get("base_url"):
            args.challenger_base_url = str(api["base_url"])
        if args.challenger_model == DEFAULT_CHALLENGER_MODEL and api.get("model"):
            args.challenger_model = str(api["model"])
        if args.challenger_api_key is None and api.get("api_key"):
            args.challenger_api_key = str(api["api_key"])
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_hash(text: str) -> str:
    return gen.md5_text(text)[:16]


def load_inherited_samples(source_dir: Path) -> list[dict[str, Any]]:
    sample_path = source_dir / "samples.jsonl"
    dialogue_path = source_dir / "expanded_dialogues.jsonl"
    graph_path = source_dir / "expanded_graphs.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing inherited samples: {sample_path}")
    samples = read_jsonl(sample_path)
    dialogues = {item.get("sample_id"): item for item in read_jsonl(dialogue_path)} if dialogue_path.exists() else {}
    graphs = {item.get("expanded_graph_id"): item for item in read_jsonl(graph_path)} if graph_path.exists() else {}
    for sample in samples:
        sample["_inherited_dialogue_record"] = dialogues.get(sample.get("sample_id"), {})
        sample["_inherited_graph_record"] = graphs.get(sample.get("expanded_graph_id"), {})
    return samples


def load_clean_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing clean source manifest: {path}. Run scripts/build_clean_v1_source_manifest.py first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return payload


def load_clean_parent_samples(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = load_clean_manifest(manifest_path)
    selected = manifest.get("selected") if isinstance(manifest.get("selected"), list) else []
    if split == "train":
        selected = [entry for entry in selected if entry.get("kind") == "clean_v1_first_round_train"]
    else:
        selected = [entry for entry in selected if entry.get("kind") == "clean_v1_first_round_heldout"]
    parents: list[dict[str, Any]] = []
    for entry in selected:
        dialogue_path = ROOT / str(entry["dialogue_path"])
        graph_path = ROOT / str(entry["memory_graph_path"])
        dialogue_payload = json.loads(dialogue_path.read_text(encoding="utf-8"))
        graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
        dialogue_records = gen.flatten_source_records(dialogue_payload)
        graph_records = gen.flatten_source_records(graph_payload)
        dialogue_index = {clean_record_key(record): record for record in dialogue_records}
        for graph_record in graph_records:
            key = clean_record_key(graph_record)
            dialogue_record = dialogue_index.get(key)
            if not dialogue_record:
                continue
            parents.append(build_clean_parent_sample(graph_record, dialogue_record, entry, clean_split_name(entry, split)))
    return parents


def clean_split_name(entry: dict[str, Any], fallback: str) -> str:
    text = str(entry.get("dialogue_path") or "")
    match = re.search(r"/(sample[0-9])/", text)
    if match:
        return match.group(1)
    return fallback


def parent_kind_for_sample(sample_name: str) -> str:
    if sample_name in {"sample0", "sample1", "sample2", "sample3", "sample4", "sample5"}:
        return "clean_v1_first_round_train"
    if sample_name == "sample6":
        return "clean_v1_first_round_dev"
    if sample_name in {"sample7", "sample8", "sample9"}:
        return "clean_v1_first_round_heldout"
    return "unavailable_or_unpaired"


def paired_graph_hardened_entries(sample_names: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    unavailable = sorted(set(sample_names))
    for dialogue_path in sorted((ROOT / "data/locomo_processed_datasets").glob("*/*/*graph_hardened_fast4b.json")):
        if "memory_graph" in dialogue_path.name or "summary" in dialogue_path.name:
            continue
        sample_name = dialogue_path.parent.name
        if sample_name not in sample_names:
            continue
        memory_path = dialogue_path.with_name(dialogue_path.stem + "_memory_graph.json")
        if not memory_path.exists():
            continue
        unavailable = [item for item in unavailable if item != sample_name]
        kind = parent_kind_for_sample(sample_name)
        entries.append(
            {
                "dialogue_path": str(dialogue_path.relative_to(ROOT)),
                "memory_graph_path": str(memory_path.relative_to(ROOT)),
                "kind": kind,
                "selected": True,
                "reason": "selected paired clean_v1 graph-hardened + memory_graph parent",
            }
        )
    return entries, {
        "requested_samples": sorted(sample_names),
        "available_samples": sorted({Path(entry["dialogue_path"]).parent.name for entry in entries}),
        "unavailable_samples_without_paired_graph_memory": unavailable,
    }


def load_clean_parent_entries(entries: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    for entry in entries:
        dialogue_path = ROOT / str(entry["dialogue_path"])
        graph_path = ROOT / str(entry["memory_graph_path"])
        dialogue_payload = json.loads(dialogue_path.read_text(encoding="utf-8"))
        graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
        dialogue_records = gen.flatten_source_records(dialogue_payload)
        graph_records = gen.flatten_source_records(graph_payload)
        dialogue_index = {clean_record_key(record): record for record in dialogue_records}
        for graph_record in graph_records:
            key = clean_record_key(graph_record)
            dialogue_record = dialogue_index.get(key)
            if not dialogue_record:
                continue
            parents.append(build_clean_parent_sample(graph_record, dialogue_record, entry, clean_split_name(entry, split)))
    return parents


def load_auto_full_parents(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train_sample_names = parse_sample_names(args.train_samples, default=DEFAULT_AUTO_TRAIN_SAMPLES)
    dev_sample_names = parse_sample_names(args.dev_samples, default=DEFAULT_AUTO_DEV_SAMPLES)
    heldout_sample_names = parse_sample_names(args.heldout_samples, default=DEFAULT_AUTO_HELDOUT_SAMPLES)
    train_entries, train_availability = paired_graph_hardened_entries(train_sample_names)
    dev_entries, dev_availability = paired_graph_hardened_entries(dev_sample_names)
    heldout_entries, heldout_availability = paired_graph_hardened_entries(heldout_sample_names)
    parents = {
        "train": load_clean_parent_entries(train_entries, "train"),
        "dev": load_clean_parent_entries(dev_entries, "dev"),
        "heldout": load_clean_parent_entries(heldout_entries, "heldout"),
    }
    manifest = {
        "policy": "r2_auto_full uses only paired graph_hardened_fast4b + memory_graph parents under data/locomo_processed_datasets; v12/v13 rule outputs are excluded.",
        "requested_samples": {
            "train": sorted(train_sample_names),
            "dev": sorted(dev_sample_names),
            "heldout": sorted(heldout_sample_names),
        },
        "train_entries": train_entries,
        "dev_entries": dev_entries,
        "heldout_entries": heldout_entries,
        "availability": {
            "train": train_availability,
            "dev": dev_availability,
            "heldout": heldout_availability,
        },
        "parent_counts": {key: len(value) for key, value in parents.items()},
    }
    return parents, manifest


def parse_question_mix(value: str) -> dict[str, float]:
    weights = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw = item.split(":", 1)
        key = key.strip()
        if key not in QUESTION_TYPES:
            raise ValueError(f"Unknown question type in mix: {key}")
        weights[key] = float(raw)
    for key in QUESTION_TYPES:
        weights.setdefault(key, 0.0)
    if sum(weights.values()) <= 0:
        raise ValueError("question type mix must have positive total weight")
    return weights


def clean_record_key(record: dict[str, Any]) -> tuple[str, int | None, int | None]:
    sample_id = str(record.get("sample_id") or record.get("sample_name") or record.get("id") or "").strip()
    sample_index = record.get("sample_index")
    qa_index = record.get("qa_index")
    return (
        sample_id,
        int(sample_index) if isinstance(sample_index, int) else None,
        int(qa_index) if isinstance(qa_index, int) else None,
    )


def build_clean_parent_sample(graph_record: dict[str, Any], dialogue_record: dict[str, Any], manifest_entry: dict[str, Any], split: str) -> dict[str, Any]:
    sample_id = str(graph_record.get("sample_id") or dialogue_record.get("sample_id") or "unknown")
    sample_index = int(graph_record.get("sample_index") if isinstance(graph_record.get("sample_index"), int) else dialogue_record.get("sample_index") or 0)
    qa_index = int(graph_record.get("qa_index") or dialogue_record.get("qa_index") or 0)
    source_id = f"{sample_id}:{qa_index}"
    turns = gen.get_dialogue_turns(dialogue_record)
    nodes: list[dict[str, Any]] = []
    for idx, event in enumerate(graph_record.get("timeline_events") or [], start=1):
        if not isinstance(event, dict):
            continue
        turn_id = str(event.get("turn_id") or "")
        value = normalize(event.get("quote") or event.get("phrase"))
        if not turn_id or not value:
            continue
        nodes.append(
            {
                "node_id": f"{source_id}:event_{idx}",
                "role": f"event_{idx}",
                "label": f"timeline event {idx}",
                "value": value,
                "turn_ids": [turn_id],
                "status": "clean_v1_timeline_event",
                "can_be_gold": True,
                "origin": "clean_v1_memory_graph",
            }
        )
    for idx, update in enumerate(graph_record.get("state_updates") or [], start=1):
        if not isinstance(update, dict):
            continue
        turn_id = str(update.get("current_turn_id") or update.get("turn_id") or "")
        value = normalize(update.get("current_quote") or update.get("quote") or update.get("update_type"))
        if not turn_id or not value:
            continue
        nodes.append(
            {
                "node_id": f"{source_id}:state_update_{idx}",
                "role": f"state_update_{idx}",
                "label": f"state update {idx}",
                "value": value,
                "turn_ids": [turn_id],
                "status": "clean_v1_state_update",
                "can_be_gold": True,
                "origin": "clean_v1_memory_graph",
            }
        )
    if not nodes:
        qa = dialogue_record.get("qa") if isinstance(dialogue_record.get("qa"), dict) else {}
        for idx, turn_id in enumerate(qa.get("evidence") or [], start=1):
            nodes.append(
                {
                    "node_id": f"{source_id}:qa_evidence_{idx}",
                    "role": f"qa_evidence_{idx}",
                    "label": f"qa evidence {idx}",
                    "value": normalize(qa.get("answer") or qa.get("correct_answer_text")),
                    "turn_ids": [str(turn_id)],
                    "status": "clean_v1_qa_evidence",
                    "can_be_gold": True,
                    "origin": "clean_v1_graph_hardened_dialogue",
                }
            )
    chain = {
        "chain_id": f"clean_v1:{source_id}",
        "hard_mode": "clean_v1_first_round_memory_graph",
        "entity": sample_id,
        "slot": "memory_state",
        "slot_label": "memory state",
        "parent_source": "clean_v1_first_round_graph_hardened_memory_graph",
        "source_dialogue_path": manifest_entry.get("dialogue_path"),
        "source_memory_graph_path": manifest_entry.get("memory_graph_path"),
        "nodes": nodes,
        "node_count": len(nodes),
        "target_role": nodes[-1]["role"] if nodes else None,
        "target_node_id": nodes[-1]["node_id"] if nodes else None,
        "valid_node_id": nodes[-1]["node_id"] if nodes else None,
    }
    dialogue_text = "\n".join(gen.dialogue_turn_text_line(turn) for turn in turns)
    return {
        "sample_id": source_id,
        "source_graph_id": source_id,
        "expanded_graph_id": source_id,
        "expanded_dialogue_id": source_id,
        "question_type": "inherited_clean_v1_parent",
        "state_chain": chain,
        "state_chain_id": chain["chain_id"],
        "state_chain_depth": len(nodes),
        "state_entity": sample_id,
        "state_slot": "memory_state",
        "state_slot_label": "memory state",
        "expanded_dialogue_text": dialogue_text,
        "new_turn_count": len(turns),
        "old_turn_count": len(turns),
        "source_split": split,
        "sample_index": sample_index,
        "qa_index": qa_index,
        "parent_pairing_key": {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "qa_index": qa_index,
        },
        "_inherited_dialogue_record": {"new_turns": turns},
        "_inherited_graph_record": graph_record,
        "_clean_parent": True,
    }


def option_value_pool(chain: dict[str, Any]) -> list[str]:
    values = []
    for node in chain.get("nodes") or []:
        value = normalize(node.get("value"))
        if value and value not in values:
            values.append(value)
    return values


def fallback_plan(parent_sample: dict[str, Any], delta: int, seed_text: str) -> dict[str, Any]:
    chain = parent_sample["state_chain"]
    entity = normalize(chain.get("entity") or parent_sample.get("state_entity") or "the packet")
    slot_label = normalize(chain.get("slot_label") or parent_sample.get("state_slot_label") or "entry")
    pool = option_value_pool(chain)
    role_stems = ["handoff_log", "constraint_note", "review_stamp", "routing_gate", "evidence_window", "scope_guard", "audit_anchor", "final_routing"]
    statuses = ["handoff_scope", "constraint_update", "audit_confirmation", "routing_gate", "evidence_window", "scope_guard", "audit_anchor", "final_r2_state"]
    rng = random.Random(stable_hash(seed_text))
    nodes = []
    for idx in range(delta):
        base_value = pool[(idx + rng.randrange(max(1, len(pool)))) % max(1, len(pool))] if pool else f"entry {idx + 1}"
        value = f"{base_value} / r2-{idx + 1}"
        role = role_stems[idx % len(role_stems)] if idx < len(role_stems) else f"r2_extension_{idx + 1}"
        nodes.append(
            {
                "role": role,
                "label": role.replace("_", " "),
                "value": value,
                "status": statuses[idx % len(statuses)],
                "can_be_gold": idx == delta - 1,
                "dialogue_hint": f"Confirm that {entity}'s {slot_label} for {role.replace('_', ' ')} is {value}.",
            }
        )
    return {"extension_nodes": nodes}


def call_challenger_llm(args: argparse.Namespace, parent_sample: dict[str, Any], delta: int, seed_text: str) -> dict[str, Any] | None:
    if args.no_llm:
        return None
    api_key = args.challenger_api_key or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.warning("No challenger API key found; using deterministic fallback plan.")
        return None
    chain = parent_sample["state_chain"]
    compact_nodes = [
        {
            "role": node.get("role"),
            "label": node.get("label"),
            "value": node.get("value"),
            "status": node.get("status"),
            "can_be_gold": node.get("can_be_gold"),
        }
        for node in (chain.get("nodes") or [])[-8:]
        if isinstance(node, dict)
    ]
    prompt = {
        "task": "Propose a compact graph-extension plan only. Do not decide final gold answer.",
        "constraints": [
            "inherit parent chain; do not create a new chain",
            "return JSON only",
            "add exactly the requested number of extension_nodes",
            "each node must have role, label, value, status, can_be_gold, dialogue_hint",
            "keep every value and dialogue_hint short",
            "dialogue_hint must realize only the node fact, with no unsupported facts",
        ],
        "delta": delta,
        "parent_chain": {
            "chain_id": chain.get("chain_id"),
            "entity": chain.get("entity"),
            "slot": chain.get("slot"),
            "slot_label": chain.get("slot_label"),
            "recent_nodes": compact_nodes,
        },
        "seed": seed_text,
    }
    payload = {
        "model": args.challenger_model,
        "messages": [
            {"role": "system", "content": "You are a strict graph-expansion planner. Return valid JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    url = args.challenger_base_url.rstrip("/") + "/chat/completions"
    for attempt in range(args.max_llm_retries + 1):
        try:
            if shutil.which("curl"):
                data = post_json_with_curl(url, payload, api_key, max(1, int(args.llm_timeout)))
            else:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                start = content.find("{")
                end = content.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(content[start : end + 1])
                raise
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            logging.warning("LLM planner attempt %d failed: %s", attempt + 1, exc)
            time.sleep(min(2 + attempt, 5))
    return None


def post_json_with_curl(url: str, payload: dict[str, Any], api_key: str, timeout_seconds: int) -> dict[str, Any]:
    curl = shutil.which("curl")
    if not curl:
        raise urllib.error.URLError("curl not available")
    config_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as config:
            config_path = config.name
            config.write(f'url = "{url}"\n')
            config.write('request = "POST"\n')
            config.write(f'header = "Authorization: Bearer {api_key}"\n')
            config.write('header = "Content-Type: application/json"\n')
            config.write('data-binary = "@-"\n')
            config.write("silent\n")
            config.write("show-error\n")
            config.write("fail-with-body\n")
            config.write(f"max-time = {timeout_seconds}\n")
        proc = subprocess.run(
            [curl, "--config", config_path],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds + 5,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()
            raise urllib.error.URLError(stderr or stdout or f"curl exited {proc.returncode}")
        return json.loads(proc.stdout.decode("utf-8"))
    finally:
        if config_path:
            try:
                os.unlink(config_path)
            except FileNotFoundError:
                pass


def sanitize_plan(plan: dict[str, Any], parent_sample: dict[str, Any], delta: int, seed_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    fallback = fallback_plan(parent_sample, delta, seed_text)["extension_nodes"]
    raw_nodes = plan.get("extension_nodes") if isinstance(plan, dict) else None
    if not isinstance(raw_nodes, list):
        reasons.append("llm_plan_missing_extension_nodes")
        raw_nodes = []
    output: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for idx in range(delta):
        raw = raw_nodes[idx] if idx < len(raw_nodes) and isinstance(raw_nodes[idx], dict) else {}
        fb = fallback[idx]
        role = re.sub(r"[^a-zA-Z0-9_]+", "_", normalize(raw.get("role") or fb["role"]).lower()).strip("_") or fb["role"]
        if role in seen_roles:
            role = f"{role}_{idx + 1}"
        seen_roles.add(role)
        value = normalize(raw.get("value") or fb["value"])
        label = normalize(raw.get("label") or role.replace("_", " "))
        status = re.sub(r"[^a-zA-Z0-9_]+", "_", normalize(raw.get("status") or fb["status"]).lower()).strip("_") or fb["status"]
        hint = normalize(raw.get("dialogue_hint") or fb["dialogue_hint"])
        if not value:
            value = fb["value"]
            reasons.append("empty_llm_value_replaced")
        if any(pattern in value.lower() or pattern in hint.lower() for pattern in LEAK_PATTERNS):
            hint = fb["dialogue_hint"]
            reasons.append("llm_leak_phrase_replaced")
        output.append(
            {
                "role": role,
                "label": label,
                "value": value,
                "status": status,
                "can_be_gold": bool(raw.get("can_be_gold")) if idx == delta - 1 else False,
                "dialogue_hint": hint,
            }
        )
    output[-1]["can_be_gold"] = True
    return output, reasons


def next_turn_ids(parent_sample: dict[str, Any], count: int) -> list[str]:
    dialogue_text = str(parent_sample.get("expanded_dialogue_text") or "")
    keys = [gen.dialogue_turn_id(turn) for turn in parent_sample.get("_inherited_dialogue_record", {}).get("new_turns", []) if isinstance(turn, dict)]
    keys.extend(re.findall(r"\[(D\d+:\d+)\]", dialogue_text))
    max_session = 1
    for key in keys:
        match = re.match(r"D(\d+):", key)
        if match:
            max_session = max(max_session, int(match.group(1)))
    session = max_session + 1
    return [f"D{session}:{idx + 1}" for idx in range(count)]


def realize_turns(parent_sample: dict[str, Any], extension_nodes: list[dict[str, Any]], seed_text: str, added_turn_target: int) -> list[dict[str, Any]]:
    speakers = []
    for line in str(parent_sample.get("expanded_dialogue_text") or "").splitlines():
        match = re.match(r"\[[^\]]+\]\s*([^:]+):", line)
        if match and match.group(1).strip() not in speakers:
            speakers.append(match.group(1).strip())
    speakers = (speakers + ["Alex", "Maya"])[:2]
    added_turn_target = max(len(extension_nodes), added_turn_target)
    turn_ids = next_turn_ids(parent_sample, added_turn_target)
    turns: list[dict[str, Any]] = []
    node_turns: dict[int, list[str]] = {idx: [] for idx in range(len(extension_nodes))}
    for idx, turn_id in enumerate(turn_ids):
        node_turns[idx % len(extension_nodes)].append(turn_id)
    for idx, node in enumerate(extension_nodes):
        node["turn_ids"] = node_turns[idx]
    for turn_index, turn_id in enumerate(turn_ids):
        idx = turn_index % len(extension_nodes)
        node = extension_nodes[idx]
        text_a = normalize(node.pop("dialogue_hint", "")) or f"The {node['label']} now uses {node['value']}."
        variants = [
            text_a,
            f"I logged {node['label']} as {node['value']} and kept it tied to that r2 state-chain step.",
            f"The review note for {node['label']} still points to {node['value']}, not to a neighboring entry.",
            f"When we checked the r2 chain, {node['label']} remained supported by {node['value']}.",
        ]
        text = variants[(turn_index // len(extension_nodes)) % len(variants)]
        turns.append(
            {
                "speaker": speakers[turn_index % 2],
                "dia_id": turn_id,
                "text": text,
                "turn_index": turn_index + 1,
                "timeline_order": idx + 1,
                "session_key": f"r2_session_{turn_id.split(':')[0][1:]}",
                "session_time": "10:00 am on 8 June, 2026",
                "synthetic_segment": True,
                "graph_node_ids": [],
                "graph_edge_ids": [],
                "planned_fact": True,
            }
        )
    for node in extension_nodes:
        for turn in turns:
            if turn["dia_id"] in node["turn_ids"]:
                turn["graph_node_ids"].append(node["node_id"])
    return turns


def choose_delta(args: argparse.Namespace, inherited_depth: int, seed_text: str) -> int:
    min_delta = max(1, int(args.min_extension_delta))
    max_delta = int(args.max_extension_delta) if args.max_extension_delta is not None else max(min_delta, int(args.target_extension_delta))
    max_delta = max(min_delta, max_delta)
    lower = max(min_delta, int(args.min_final_depth) - inherited_depth)
    upper = min(max_delta, int(args.max_final_depth) - inherited_depth)
    if upper < lower:
        return 0
    span = upper - lower + 1
    return lower + (int(stable_hash(seed_text), 16) % span)


def choose_added_turn_target(args: argparse.Namespace, seed_text: str) -> int:
    low = max(1, int(args.min_added_turns))
    high = max(low, int(args.max_added_turns))
    return low + (int(stable_hash(seed_text + ":turns"), 16) % (high - low + 1))


def build_r2_expansion(parent_sample: dict[str, Any], args: argparse.Namespace, index: int, split_name: str) -> dict[str, Any] | None:
    parent_chain = copy.deepcopy(parent_sample["state_chain"])
    inherited_nodes = [copy.deepcopy(node) for node in parent_chain.get("nodes") or [] if isinstance(node, dict)]
    inherited_depth = int(parent_sample.get("state_chain_depth") or parent_chain.get("node_count") or len(inherited_nodes))
    seed_text = f"{split_name}|{index}|{parent_sample.get('sample_id')}|{args.seed}"
    delta = choose_delta(args, inherited_depth, seed_text)
    if delta <= 0:
        return None
    added_turn_target = choose_added_turn_target(args, seed_text)
    llm_plan = call_challenger_llm(args, parent_sample, delta, seed_text)
    if args.require_llm_plan and llm_plan is None:
        return None
    raw_plan = llm_plan or fallback_plan(parent_sample, delta, seed_text)
    extension_nodes, plan_reasons = sanitize_plan(raw_plan, parent_sample, delta, seed_text)
    parent_chain_id = normalize(parent_chain.get("chain_id") or parent_sample.get("state_chain_id") or parent_sample.get("sample_id"))
    chain_id = f"{parent_chain_id}:r2"
    edge_records = []
    for idx, node in enumerate(extension_nodes):
        node["node_id"] = f"{chain_id}:{node['role']}"
        node["origin"] = "r2_llm_assisted_graph_update"
        node["parent_chain_id"] = parent_chain_id
        if idx:
            edge_records.append({"edge_id": f"{chain_id}:edge_{idx}", "source": extension_nodes[idx - 1]["node_id"], "target": node["node_id"], "relation": "state_chain_extension"})
    turns = realize_turns(parent_sample, extension_nodes, seed_text, added_turn_target)
    new_nodes = inherited_nodes + extension_nodes
    state_chain = copy.deepcopy(parent_chain)
    state_chain.update(
        {
            "chain_id": chain_id,
            "parent_chain_id": parent_chain_id,
            "inherited_state_chain_depth": inherited_depth,
            "state_chain_extension_delta": len(extension_nodes),
            "final_state_chain_depth": inherited_depth + len(extension_nodes),
            "nodes": new_nodes,
            "node_count": len(new_nodes),
            "target_role": extension_nodes[-1]["role"],
            "target_node_id": extension_nodes[-1]["node_id"],
            "valid_node_id": extension_nodes[-1]["node_id"],
            "hard_mode": "r2_llm_graph_expansion",
        }
    )
    old_dialogue = str(parent_sample.get("expanded_dialogue_text") or "")
    new_dialogue = "\n".join(gen.dialogue_turn_text_line(turn) for turn in turns)
    expansion_id = stable_hash(f"r2|{seed_text}|{chain_id}")
    return {
        "sample_id": f"{parent_sample.get('sample_id')}_r2_{index:04d}",
        "source_sample_id": parent_sample.get("sample_id"),
        "source_graph_id": parent_sample.get("expanded_graph_id") or parent_sample.get("source_graph_id"),
        "expanded_graph_id": expansion_id,
        "expanded_dialogue_id": f"{parent_sample.get('expanded_dialogue_id') or parent_sample.get('sample_id')}_r2_exp_{index:04d}",
        "parent_sample_id": parent_sample.get("sample_id"),
        "parent_sample_index": parent_sample.get("sample_index"),
        "parent_qa_index": parent_sample.get("qa_index"),
        "parent_pairing_key": parent_sample.get("parent_pairing_key"),
        "state_chain": state_chain,
        "state_chain_nodes": new_nodes,
        "new_nodes": extension_nodes,
        "new_edges": edge_records,
        "turns": turns,
        "new_dialogue_text": new_dialogue,
        "expanded_dialogue_text": "\n".join([old_dialogue, new_dialogue]).strip(),
        "old_turn_count": int(parent_sample.get("new_turn_count") or parent_sample.get("old_turn_count") or 0),
        "added_turn_count": len(turns),
        "new_turn_count": int(parent_sample.get("new_turn_count") or 0) + len(turns),
        "inherited_state_chain_depth": inherited_depth,
        "state_chain_extension_delta": len(extension_nodes),
        "final_state_chain_depth": inherited_depth + len(extension_nodes),
        "parent_chain_id": parent_chain_id,
        "graph_update_validation_reasons": plan_reasons,
        "challenger_model": args.challenger_model,
        "challenger_base_url": args.challenger_base_url,
        "llm_plan_used": llm_plan is not None,
    }


def make_options(values: list[str], gold: str, seed_text: str, count: int = 4) -> dict[str, str]:
    pool = [normalize(v) for v in values if normalize(v) and normalize(v) != normalize(gold)]
    pool = list(dict.fromkeys(pool))
    while len(pool) < count - 1:
        pool.append(f"unsupported r2 distractor {len(pool) + 1}")
    options = [gold] + pool[: count - 1]
    options = v13.stable_shuffle(options, seed_text)
    return {OPTION_LABELS[idx]: value for idx, value in enumerate(options)}


def build_base_sample(expansion: dict[str, Any], qtype: str, training_use: str) -> dict[str, Any]:
    chain = expansion["state_chain"]
    nodes = [node for node in chain.get("nodes") or [] if isinstance(node, dict)]
    target = expansion["new_nodes"][-1]
    values = [node.get("value") for node in nodes]
    options = make_options(values, target["value"], expansion["sample_id"])
    gold_option = next(label for label, value in options.items() if normalize(value) == normalize(target["value"]))
    sample = {
        "sample_id": expansion["sample_id"],
        "parent_sample_id": expansion.get("parent_sample_id"),
        "parent_sample_index": expansion.get("parent_sample_index"),
        "parent_qa_index": expansion.get("parent_qa_index"),
        "parent_pairing_key": expansion.get("parent_pairing_key"),
        "source_graph_id": expansion["source_graph_id"],
        "expanded_graph_id": expansion["expanded_graph_id"],
        "expanded_dialogue_id": expansion["expanded_dialogue_id"],
        "question_type": "single_choice",
        "answer_format": "single_label",
        "question": f"Which {chain.get('slot_label') or 'entry'} is the final r2 state for {chain.get('entity') or 'the packet'}?",
        "options": options,
        "gold_option": gold_option,
        "gold_options": [gold_option],
        "gold_answer": target["value"],
        "gold_node": target["node_id"],
        "gold_node_role": target["role"],
        "gold_origin": "programmatic_from_validated_r2_state_chain",
        "gold_source": target["role"],
        "target_role": target["role"],
        "state_entity": chain.get("entity"),
        "state_slot": chain.get("slot"),
        "state_slot_label": chain.get("slot_label"),
        "state_chain": chain,
        "state_chain_nodes": expansion["state_chain_nodes"],
        "state_chain_id": chain["chain_id"],
        "state_chain_depth": expansion["final_state_chain_depth"],
        "inherited_state_chain_depth": expansion["inherited_state_chain_depth"],
        "state_chain_extension_delta": expansion["state_chain_extension_delta"],
        "final_state_chain_depth": expansion["final_state_chain_depth"],
        "parent_chain_id": expansion["parent_chain_id"],
        "new_nodes": expansion["new_nodes"],
        "new_edges": expansion["new_edges"],
        "evidence_nodes": target["turn_ids"],
        "added_turn_count": expansion["added_turn_count"],
        "old_turn_count": expansion["old_turn_count"],
        "new_turn_count": expansion["new_turn_count"],
        "new_dialogue_text": expansion["new_dialogue_text"],
        "expanded_dialogue_text": expansion["expanded_dialogue_text"],
        "trap_type": "r2_chain_extension",
        "attack_type": "llm_assisted_graph_expansion",
        "training_use": training_use,
        "difficulty_for_solver": "frontier_r2_llm_graph_expansion",
        "quality_tier": "graph_expansion_r2_llm_assisted",
        "synthetic_segment": True,
        "challenger_model": expansion["challenger_model"],
        "challenger_base_url": expansion["challenger_base_url"],
        "llm_plan_used": expansion["llm_plan_used"],
        "graph_dialogue_alignment": True,
        "gold_derivable_from_graph_or_state_chain": True,
        "unconstrained_llm_gold": False,
        "heldout_contamination": False,
        "solver_model_role": "local_solver_r1_fixed_no_retraining",
        "metadata": {
            "training_use": training_use,
            "question_type": "single_choice",
            "gold_origin": "programmatic_from_validated_r2_state_chain",
            "challenger_model": expansion["challenger_model"],
            "challenger_base_url": expansion["challenger_base_url"],
            "llm_plan_used": expansion["llm_plan_used"],
            "solver_base": DEFAULT_SOLVER_BASE,
            "solver_adapter": DEFAULT_SOLVER_ADAPTER,
            "parent_chain_id": expansion["parent_chain_id"],
            "parent_sample_id": expansion.get("parent_sample_id"),
            "parent_sample_index": expansion.get("parent_sample_index"),
            "parent_qa_index": expansion.get("parent_qa_index"),
            "parent_pairing_key": expansion.get("parent_pairing_key"),
            "inherited_state_chain_depth": expansion["inherited_state_chain_depth"],
            "state_chain_extension_delta": expansion["state_chain_extension_delta"],
            "final_state_chain_depth": expansion["final_state_chain_depth"],
            "graph_dialogue_alignment": True,
            "gold_derivable_from_graph_or_state_chain": True,
            "unconstrained_llm_gold": False,
            "heldout_contamination": False,
        },
    }
    return transform_r2_question_type(sample, qtype)


def node_mapping_text(node: dict[str, Any]) -> str:
    return f"{normalize(node.get('label') or node.get('role'))}: {normalize(node.get('value'))}"


def transform_r2_question_type(sample: dict[str, Any], qtype: str) -> dict[str, Any]:
    output = copy.deepcopy(sample)
    output["legacy_question_type"] = output.get("question_type")
    if qtype == "single_choice":
        output["question_type"] = "single_choice"
        output["answer_format"] = "single_label"
        output["gold_options"] = [output["gold_option"]]
    elif qtype == "multi_select":
        nodes = output.get("new_nodes") or []
        correct_nodes = nodes[-2:] if len(nodes) >= 2 else nodes
        if len(correct_nodes) < 2:
            raise ValueError("Need at least two new nodes for multi_select")
        correct_items = [node_mapping_text(node) for node in correct_nodes]
        distractors = []
        inherited_nodes = [node for node in output.get("state_chain", {}).get("nodes", []) if isinstance(node, dict)][:-len(nodes)]
        for idx, node in enumerate(inherited_nodes[:4]):
            distractors.append(node_mapping_text(node))
            if len(distractors) >= 2:
                break
        while len(distractors) < 2:
            distractors.append(f"unsupported mapping {len(distractors) + 1}: not in approved graph")
        items = v13.stable_shuffle(correct_items + distractors[:2], output["sample_id"] + ":r2_multi")
        options = {OPTION_LABELS[idx]: text for idx, text in enumerate(items)}
        gold_options = sorted(label for label, text in options.items() if text in set(correct_items))
        output.update(
            {
                "question_type": "multi_select",
                "answer_format": "multi_label_set",
                "question": f"Select exactly the two r2 role-state mappings approved for {output['state_entity']}.",
                "options": options,
                "gold_options": gold_options,
                "gold_answer": ", ".join(gold_options),
                "gold_answers": correct_items,
            }
        )
    elif qtype == "ordering":
        nodes = output.get("new_nodes") or []
        selected = nodes[:4]
        if len(selected) < 4:
            raise ValueError("Need four new nodes for ordering")
        option_items = v13.stable_shuffle([node_mapping_text(node) for node in selected], output["sample_id"] + ":r2_order")
        options = {OPTION_LABELS[idx]: text for idx, text in enumerate(option_items)}
        text_to_label = {text: label for label, text in options.items()}
        gold_order = [text_to_label[node_mapping_text(node)] for node in selected]
        output.update(
            {
                "question_type": "ordering",
                "answer_format": "ordered_labels",
                "question": f"Order the r2 approved mappings for {output['state_entity']} by their state-chain extension order.",
                "options": options,
                "gold_order": gold_order,
                "gold_answer": " > ".join(gold_order),
            }
        )
    elif qtype == "state_reconstruction":
        output.update(
            {
                "question_type": "state_reconstruction",
                "answer_format": "exact_text",
                "question": f"Reconstruct the exact final r2 memory state for {output['state_entity']}.",
                "options": {},
                "gold_options": [],
                "gold_answer": normalize(output.get("gold_answer")),
            }
        )
    else:
        raise ValueError(f"Unknown question_type: {qtype}")
    output["metadata"].update(
        {
            "question_type": output["question_type"],
            "answer_format": output["answer_format"],
            "gold_options": output.get("gold_options"),
            "gold_order": output.get("gold_order"),
            "gold_answer": output.get("gold_answer"),
        }
    )
    return output


def validate_r2_sample(sample: dict[str, Any], min_delta: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(sample.get("state_chain_extension_delta") or 0) < min_delta:
        reasons.append("state_chain_extension_below_minimum")
    if int(sample.get("final_state_chain_depth") or 0) != int(sample.get("inherited_state_chain_depth") or 0) + int(sample.get("state_chain_extension_delta") or 0):
        reasons.append("state_chain_depth_math_mismatch")
    turn_ids = set(re.findall(r"\[(D\d+:\d+)\]", sample.get("expanded_dialogue_text") or ""))
    for node in sample.get("new_nodes") or []:
        if not set(node.get("turn_ids") or []).issubset(turn_ids):
            reasons.append("new_node_missing_evidence_turn")
            break
    for turn_text in [sample.get("new_dialogue_text"), sample.get("question")]:
        if any(pattern in normalize(turn_text).lower() for pattern in LEAK_PATTERNS):
            reasons.append("leakage_detected")
            break
    target_values = {normalize(node.get("value")) for node in sample.get("new_nodes") or []}
    if sample.get("question_type") in {"single_choice", "state_reconstruction"} and normalize(sample.get("gold_answer")) not in target_values:
        reasons.append("gold_not_derivable_from_extension")
    options = sample.get("options") if isinstance(sample.get("options"), dict) else {}
    if len(set(normalize(value).lower() for value in options.values())) != len(options):
        reasons.append("duplicate_option_texts")
    qtype = sample.get("question_type")
    if qtype == "single_choice":
        gold_option = sample.get("gold_option")
        if gold_option not in options:
            reasons.append("single_choice_gold_option_missing")
        elif normalize(options[gold_option]) != normalize(sample.get("gold_answer")):
            reasons.append("single_choice_gold_option_mismatch")
    elif qtype == "multi_select":
        gold_options = sample.get("gold_options") if isinstance(sample.get("gold_options"), list) else []
        if not (2 <= len(gold_options) <= 4):
            reasons.append("multi_select_correct_count_out_of_range")
        if any(label not in options for label in gold_options):
            reasons.append("multi_select_gold_label_missing")
        if len(options) - len(gold_options) < 1:
            reasons.append("multi_select_missing_distractor")
    elif qtype == "ordering":
        gold_order = sample.get("gold_order") if isinstance(sample.get("gold_order"), list) else []
        if not gold_order or len(gold_order) != len(set(gold_order)):
            reasons.append("ordering_sequence_not_unique")
        if any(label not in options for label in gold_order):
            reasons.append("ordering_label_missing")
    elif qtype == "state_reconstruction":
        if not normalize(sample.get("gold_answer")):
            reasons.append("state_reconstruction_gold_missing")
    else:
        reasons.append("unknown_question_type")
    return not reasons, sorted(set(reasons))


def build_expanded_graph_record(expansion: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_graph_id": expansion["source_graph_id"],
        "expanded_graph_id": expansion["expanded_graph_id"],
        "graph_expansion_round": 2,
        "parent_chain_id": expansion["parent_chain_id"],
        "inherited_state_chain_depth": expansion["inherited_state_chain_depth"],
        "state_chain_extension_delta": expansion["state_chain_extension_delta"],
        "final_state_chain_depth": expansion["final_state_chain_depth"],
        "state_chain": expansion["state_chain"],
        "new_nodes": expansion["new_nodes"],
        "new_edges": expansion["new_edges"],
        "gold_origin": sample.get("gold_origin"),
        "challenger_model": expansion["challenger_model"],
        "challenger_base_url": expansion["challenger_base_url"],
    }


def build_expanded_dialogue_record(expansion: dict[str, Any], sample: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_graph_id": expansion["source_graph_id"],
        "expanded_dialogue_id": expansion["expanded_dialogue_id"],
        "source_split": split_name,
        "old_turn_count": expansion["old_turn_count"],
        "added_turn_count": expansion["added_turn_count"],
        "new_turn_count": expansion["new_turn_count"],
        "new_turns": expansion["turns"],
        "state_chain": expansion["state_chain"],
        "graph_edit_plan": {
            "round": 2,
            "parent_chain_id": expansion["parent_chain_id"],
            "new_nodes": [node["node_id"] for node in expansion["new_nodes"]],
            "new_edges": [edge["edge_id"] for edge in expansion["new_edges"]],
            "state_chain_extension_delta": expansion["state_chain_extension_delta"],
            "dialogue_role": "realization_only",
        },
    }


def choose_question_type(counts: Counter[str], quotas: dict[str, int]) -> str:
    return max(quotas, key=lambda key: (quotas[key] - counts[key], quotas[key], key))


def build_validated_sample(
    parent: dict[str, Any],
    args: argparse.Namespace,
    index: int,
    split_name: str,
    training_use: str,
    qtype: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[bool, list[str]]] | None:
    if not isinstance(parent.get("state_chain"), dict):
        return None
    expansion = build_r2_expansion(parent, args, index, split_name)
    if expansion is None:
        return None
    try:
        sample = build_base_sample(expansion, qtype, training_use)
    except Exception as exc:
        logging.debug("Skipping sample after QA transform error: %s", exc)
        return None
    ok, reasons = validate_r2_sample(sample, args.min_extension_delta)
    sample["option_validation"] = ok
    sample["option_validation_pass"] = ok
    sample["option_validation_reasons"] = reasons
    sample["leakage_free"] = "leakage_detected" not in reasons
    sample["graph_dialogue_alignment"] = ok and "new_node_missing_evidence_turn" not in reasons
    sample["gold_derivable_from_graph_or_state_chain"] = ok and not any(reason.startswith("gold_not_derivable") for reason in reasons)
    sample["unconstrained_llm_gold"] = False
    sample["heldout_contamination"] = False
    sample["metadata"].update(
        {
            "option_validation": ok,
            "leakage_free": sample["leakage_free"],
            "graph_dialogue_alignment": sample["graph_dialogue_alignment"],
            "gold_derivable_from_graph_or_state_chain": sample["gold_derivable_from_graph_or_state_chain"],
            "unconstrained_llm_gold": False,
            "heldout_contamination": False,
            "trap_type": sample.get("trap_type"),
            "attack_type": sample.get("attack_type"),
            "gold_source": sample.get("gold_source"),
            "state_chain_depth": sample.get("state_chain_depth"),
            "added_turn_count": sample.get("added_turn_count"),
            "record_id": sample.get("expanded_graph_id"),
            "sample_name": sample.get("expanded_dialogue_id"),
        }
    )
    if not ok:
        logging.debug("Skipping invalid r2 sample: %s", reasons)
        return None
    return sample, build_expanded_graph_record(expansion, sample), build_expanded_dialogue_record(expansion, sample, split_name), (ok, reasons)


def planned_question_types(n: int, question_type_mix: str) -> list[str]:
    q_quotas = v13.largest_remainder_counts(parse_question_mix(question_type_mix), n)
    q_counts: Counter[str] = Counter()
    planned: list[str] = []
    for _ in range(n):
        qtype = choose_question_type(q_counts, q_quotas)
        planned.append(qtype)
        q_counts[qtype] += 1
    return planned


def generate_split_parallel(
    parents: list[dict[str, Any]],
    n: int,
    args: argparse.Namespace,
    split_name: str,
    training_use: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[tuple[bool, list[str]]]]:
    q_quotas = v13.largest_remainder_counts(parse_question_mix(args.question_type_mix), n)
    q_counts: Counter[str] = Counter()
    results: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[bool, list[str]]]] = {}
    workers = max(1, int(args.llm_workers))
    max_attempts = n * int(args.max_attempt_multiplier)
    submitted = 0
    completed = 0
    next_output_index = 0
    first_failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Any, tuple[int, str]] = {}

        def submit_one() -> None:
            nonlocal submitted
            attempt_index = submitted
            parent = parents[attempt_index % len(parents)]
            qtype = choose_question_type(q_counts, q_quotas)
            futures[
                executor.submit(build_validated_sample, parent, args, attempt_index, split_name, training_use, qtype)
            ] = (attempt_index, qtype)
            submitted += 1

        for _ in range(min(workers * 4, max_attempts)):
            submit_one()

        while futures and len(results) < n:
            future = next(as_completed(futures))
            index, qtype = futures.pop(future)
            completed += 1
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("%s index %d failed: %s", split_name, index, exc)
                item = None
            if item is not None:
                sample, graph, dialogue, validation = item
                suffix = f"{next_output_index:04d}"
                original_sample_id = sample["sample_id"]
                sample["sample_id"] = re.sub(r"_r2_\d{4}$", f"_r2_{suffix}", sample["sample_id"])
                sample["expanded_dialogue_id"] = re.sub(r"_r2_exp_\d{4}$", f"_r2_exp_{suffix}", sample["expanded_dialogue_id"])
                sample["metadata"]["sample_name"] = sample["expanded_dialogue_id"]
                dialogue["sample_id"] = sample["sample_id"]
                dialogue["expanded_dialogue_id"] = sample["expanded_dialogue_id"]
                graph["sample_id"] = sample["sample_id"]
                results[next_output_index] = (sample, graph, dialogue, validation)
                q_counts[str(sample.get("question_type") or qtype)] += 1
                next_output_index += 1
                if original_sample_id == sample["sample_id"]:
                    pass
            elif len(first_failures) < 10:
                first_failures.append(f"attempt={index} qtype={qtype}")
            if completed % 50 == 0 or completed == n:
                logging.info("%s attempts=%d submitted=%d accepted=%d/%d", split_name, completed, submitted, len(results), n)
            while len(results) + len(futures) < n and submitted < max_attempts:
                submit_one()
            while len(futures) < workers * 4 and submitted < max_attempts and len(results) + len(futures) < n * 2:
                submit_one()
    if len(results) < n:
        raise RuntimeError(f"Only generated {len(results)}/{n} samples for {split_name}; first_failures={first_failures}")
    ordered = [results[index] for index in sorted(results)]
    return (
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        [item[2] for item in ordered],
        [item[3] for item in ordered],
    )


def generate_split(parents: list[dict[str, Any]], n: int, args: argparse.Namespace, split_name: str, training_use: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[tuple[bool, list[str]]]]:
    samples: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    dialogues: list[dict[str, Any]] = []
    validations: list[tuple[bool, list[str]]] = []
    q_counts: Counter[str] = Counter()
    q_quotas = v13.largest_remainder_counts(parse_question_mix(args.question_type_mix), n)
    cursor = 0
    attempts = 0
    while len(samples) < n and attempts < n * int(args.max_attempt_multiplier):
        parent = parents[cursor % len(parents)]
        cursor += 1
        attempts += 1
        if not isinstance(parent.get("state_chain"), dict):
            continue
        expansion = build_r2_expansion(parent, args, len(samples), split_name)
        if expansion is None:
            continue
        qtype = choose_question_type(q_counts, q_quotas)
        try:
            sample = build_base_sample(expansion, qtype, training_use)
        except Exception as exc:
            logging.debug("Skipping sample after QA transform error: %s", exc)
            continue
        ok, reasons = validate_r2_sample(sample, args.min_extension_delta)
        sample["option_validation"] = ok
        sample["option_validation_pass"] = ok
        sample["option_validation_reasons"] = reasons
        sample["leakage_free"] = "leakage_detected" not in reasons
        sample["graph_dialogue_alignment"] = ok and "new_node_missing_evidence_turn" not in reasons
        sample["gold_derivable_from_graph_or_state_chain"] = ok and not any(reason.startswith("gold_not_derivable") for reason in reasons)
        sample["unconstrained_llm_gold"] = False
        sample["heldout_contamination"] = False
        sample["metadata"].update(
            {
                "option_validation": ok,
                "leakage_free": sample["leakage_free"],
                "graph_dialogue_alignment": sample["graph_dialogue_alignment"],
                "gold_derivable_from_graph_or_state_chain": sample["gold_derivable_from_graph_or_state_chain"],
                "unconstrained_llm_gold": False,
                "heldout_contamination": False,
                "trap_type": sample.get("trap_type"),
                "attack_type": sample.get("attack_type"),
                "gold_source": sample.get("gold_source"),
                "state_chain_depth": sample.get("state_chain_depth"),
                "added_turn_count": sample.get("added_turn_count"),
                "record_id": sample.get("expanded_graph_id"),
                "sample_name": sample.get("expanded_dialogue_id"),
            }
        )
        if not ok:
            logging.debug("Skipping invalid r2 sample: %s", reasons)
            continue
        samples.append(sample)
        graphs.append(build_expanded_graph_record(expansion, sample))
        dialogues.append(build_expanded_dialogue_record(expansion, sample, split_name))
        validations.append((ok, reasons))
        q_counts[qtype] += 1
    if len(samples) < n:
        raise RuntimeError(f"Only generated {len(samples)}/{n} samples for {split_name}")
    return samples, graphs, dialogues, validations


def summarize(samples: list[dict[str, Any]], validations: list[tuple[bool, list[str]]]) -> dict[str, Any]:
    inherited = [int(sample.get("inherited_state_chain_depth") or 0) for sample in samples]
    delta = [int(sample.get("state_chain_extension_delta") or 0) for sample in samples]
    final = [int(sample.get("final_state_chain_depth") or 0) for sample in samples]
    added = [int(sample.get("added_turn_count") or 0) for sample in samples]
    def stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 4)}
    return {
        "n": len(samples),
        "challenger": {"base_url": DEFAULT_CHALLENGER_BASE_URL, "model": DEFAULT_CHALLENGER_MODEL, "role": "API challenger"},
        "solver": {"base": DEFAULT_SOLVER_BASE, "adapter": DEFAULT_SOLVER_ADAPTER, "role": "local fixed solver_r1; not retrained"},
        "option_validation_pass_count": sum(1 for ok, _ in validations if ok),
        "option_validation_pass_rate": round(sum(1 for ok, _ in validations if ok) / len(samples), 4) if samples else 0.0,
        "leakage_free_count": sum(1 for sample in samples if sample.get("leakage_free")),
        "graph_dialogue_alignment_pass_count": sum(1 for sample in samples if sample.get("option_validation_pass")),
        "added_turn_count_stats": stats(added),
        "inherited_state_chain_depth_stats": stats(inherited),
        "state_chain_extension_delta_stats": stats(delta),
        "final_state_chain_depth_stats": stats(final),
        "min_inherited_state_chain_depth": min(inherited) if inherited else 0,
        "avg_inherited_state_chain_depth": round(sum(inherited) / len(inherited), 4) if inherited else 0,
        "min_state_chain_extension_delta": min(delta) if delta else 0,
        "avg_state_chain_extension_delta": round(sum(delta) / len(delta), 4) if delta else 0,
        "min_final_state_chain_depth": min(final) if final else 0,
        "avg_final_state_chain_depth": round(sum(final) / len(final), 4) if final else 0,
        "question_type_counts": dict(Counter(sample.get("question_type") for sample in samples).most_common()),
        "trap_type_counts": dict(Counter(sample.get("trap_type") for sample in samples).most_common()),
        "gold_origin_counts": dict(Counter(sample.get("gold_origin") for sample in samples).most_common()),
        "llm_plan_used_counts": dict(Counter(str(bool(sample.get("llm_plan_used"))) for sample in samples).most_common()),
        "answer_derivability": {"programmatic_from_validated_graph_state_chain": len(samples), "llm_provisional_final_gold": 0},
    }


def format_status(summary: dict[str, Any], eval_payload: dict[str, Any] | None) -> str:
    challenger = summary.get("challenger", {})
    solver = summary.get("solver", {})
    lines = [
        "# R2 LLM Graph Expansion Status",
        "",
        f"- challenger: API `{challenger.get('model')}` at `{challenger.get('base_url')}`",
        f"- solver: local fixed solver_r1 `{solver.get('base')}` + `{solver.get('adapter')}`",
        "- raw LoCoMo source: not used",
        "- old rule-based synthetic session generator: not used for r2 extension",
        f"- train samples: {summary['train']['n']}",
        f"- heldout samples: {summary['heldout']['n']}",
        f"- train min inherited depth: {summary['train']['min_inherited_state_chain_depth']}",
        f"- train min extension delta: {summary['train']['min_state_chain_extension_delta']}",
        f"- train min final depth: {summary['train']['min_final_state_chain_depth']}",
        f"- heldout min inherited depth: {summary['heldout']['min_inherited_state_chain_depth']}",
        f"- heldout min extension delta: {summary['heldout']['min_state_chain_extension_delta']}",
        f"- heldout min final depth: {summary['heldout']['min_final_state_chain_depth']}",
        f"- gold origins: {summary['train']['gold_origin_counts']} train; {summary['heldout']['gold_origin_counts']} heldout",
        f"- LLM plan used: {summary['train']['llm_plan_used_counts']} train; {summary['heldout']['llm_plan_used_counts']} heldout",
        f"- question types train: {summary['train']['question_type_counts']}",
        f"- question types heldout: {summary['heldout']['question_type_counts']}",
        f"- leakage-free rates: train={summary['train']['leakage_free_count']}/{summary['train']['n']}, heldout={summary['heldout']['leakage_free_count']}/{summary['heldout']['n']}",
    ]
    if eval_payload:
        lines.append(f"- solver_r1 eval: {eval_payload.get('summary', {})}")
    else:
        lines.append("- solver_r1 eval: skipped")
    return "\n".join(lines) + "\n"


def write_dataset(out_dir: Path, sft_name: str, samples: list[dict[str, Any]], graphs: list[dict[str, Any]], dialogues: list[dict[str, Any]], validations: list[tuple[bool, list[str]]]) -> dict[str, Any]:
    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / sft_name, (v13.build_chat_record(sample) for sample in samples))
    write_jsonl(out_dir / "expanded_graphs.jsonl", graphs)
    write_jsonl(out_dir / "expanded_dialogues.jsonl", dialogues)
    summary = summarize(samples, validations)
    write_json(out_dir / "generation_summary.json", summary)
    return summary


def validation_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    depth_math_bad = sum(
        1
        for sample in samples
        if int(sample.get("final_state_chain_depth") or 0)
        != int(sample.get("inherited_state_chain_depth") or 0) + int(sample.get("state_chain_extension_delta") or 0)
    )
    parent_bad = sum(1 for sample in samples if not isinstance(sample.get("parent_pairing_key"), dict))
    train_contamination = sum(
        1
        for sample in samples
        if sample.get("training_use") == "train_candidate"
        and str(sample.get("parent_pairing_key", {}).get("sample_id") or "").startswith(("conv-47", "conv-48", "conv-49", "conv-50"))
    )
    return {
        "total": total,
        "option_validation_bad": sum(1 for sample in samples if sample.get("option_validation_pass") is not True),
        "leakage_bad": sum(1 for sample in samples if sample.get("leakage_free") is not True),
        "graph_dialogue_alignment_bad": sum(1 for sample in samples if sample.get("graph_dialogue_alignment") is not True),
        "gold_not_derivable": sum(1 for sample in samples if sample.get("gold_derivable_from_graph_or_state_chain") is not True),
        "heldout_contamination": sum(1 for sample in samples if sample.get("heldout_contamination") is True) + train_contamination,
        "unconstrained_llm_gold": sum(1 for sample in samples if sample.get("unconstrained_llm_gold") is True),
        "state_depth_math_bad": depth_math_bad,
        "parent_pairing_key_bad": parent_bad,
        "all_checks_passed": (
            total > 0
            and all(sample.get("option_validation_pass") is True for sample in samples)
            and all(sample.get("leakage_free") is True for sample in samples)
            and all(sample.get("graph_dialogue_alignment") is True for sample in samples)
            and all(sample.get("gold_derivable_from_graph_or_state_chain") is True for sample in samples)
            and all(sample.get("unconstrained_llm_gold") is not True for sample in samples)
            and depth_math_bad == 0
            and parent_bad == 0
            and train_contamination == 0
        ),
    }


def write_auto_full_dataset(
    out_dir: Path,
    report_dir: Path,
    train_samples: list[dict[str, Any]],
    dev_samples: list[dict[str, Any]],
    heldout_samples: list[dict[str, Any]],
    graphs: list[dict[str, Any]],
    dialogues: list[dict[str, Any]],
    summaries: dict[str, Any],
    source_manifest: dict[str, Any],
) -> None:
    all_samples = train_samples + dev_samples + heldout_samples
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", all_samples)
    write_jsonl(out_dir / "sft_train.jsonl", (v13.build_chat_record(sample) for sample in train_samples))
    write_jsonl(out_dir / "sft_eval_dev.jsonl", (v13.build_chat_record(sample) for sample in dev_samples))
    write_jsonl(out_dir / "sft_eval_heldout.jsonl", (v13.build_chat_record(sample) for sample in heldout_samples))
    write_jsonl(out_dir / "graph_updates.jsonl", graphs)
    write_jsonl(out_dir / "expanded_dialogues.jsonl", dialogues)
    write_jsonl(out_dir / "expanded_graphs.jsonl", graphs)
    generation_summary = {
        "version": "r2_auto_full",
        "method": "Automated reward-driven r2 LLM-assisted graph expansion from clean paired parents.",
        "source_manifest": source_manifest,
        "split_counts": {
            "train": len(train_samples),
            "dev": len(dev_samples),
            "heldout": len(heldout_samples),
            "total": len(all_samples),
        },
        "train": summaries["train"],
        "dev": summaries["dev"],
        "heldout": summaries["heldout"],
    }
    write_json(out_dir / "generation_summary.json", generation_summary)
    write_json(report_dir / "generation_summary.json", generation_summary)
    validation = validation_summary(all_samples)
    validation["by_split"] = {
        "train": validation_summary(train_samples),
        "dev": validation_summary(dev_samples),
        "heldout": validation_summary(heldout_samples),
    }
    write_json(out_dir / "validation_summary.json", validation)
    write_json(report_dir / "validation_summary.json", validation)


def evaluate_single_choice_subset(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    if args.skip_solver_eval:
        return None
    mcq_samples = [sample for sample in samples if sample.get("question_type") == "single_choice"]
    if not mcq_samples:
        return {"summary": {"n": 0, "accuracy": None}, "results": []}
    eval_args = argparse.Namespace(
        solver_base=args.solver_base,
        solver_adapter=args.solver_adapter,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    normalized = []
    for sample in mcq_samples:
        item = copy.deepcopy(sample)
        item["question_type"] = "mcq"
        normalized.append(item)
    return gen.evaluate_samples(normalized, eval_args)


def main() -> int:
    args = parse_args()
    args = apply_config_defaults(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(args.seed)
    if args.auto_full:
        parent_groups, source_manifest = load_auto_full_parents(args)
        logging.info(
            "Loaded auto-full parents: train=%d dev=%d heldout=%d",
            len(parent_groups["train"]),
            len(parent_groups["dev"]),
            len(parent_groups["heldout"]),
        )
        if not parent_groups["train"] or not parent_groups["dev"] or not parent_groups["heldout"]:
            raise RuntimeError(f"Missing required paired clean parents: {source_manifest['availability']}")
        train_samples, train_graphs, train_dialogues, train_validations = generate_split_parallel(
            parent_groups["train"],
            args.n_train,
            args,
            "r2_auto_full_train",
            "train_candidate",
        )
        dev_samples, dev_graphs, dev_dialogues, dev_validations = generate_split_parallel(
            parent_groups["dev"],
            args.n_dev,
            args,
            "r2_auto_full_dev_sample6",
            "dev_sanity_eval_only",
        )
        for sample in dev_samples:
            sample["heldout_split"] = "sample6"
            sample["metadata"]["heldout_split"] = "sample6"

        heldout_samples: list[dict[str, Any]] = []
        heldout_graphs: list[dict[str, Any]] = []
        heldout_dialogues: list[dict[str, Any]] = []
        heldout_validations: list[tuple[bool, list[str]]] = []
        for split_name in sorted(parse_sample_names(args.heldout_samples, default=DEFAULT_AUTO_HELDOUT_SAMPLES)):
            split_parents = [parent for parent in parent_groups["heldout"] if parent.get("source_split") == split_name]
            if not split_parents:
                raise RuntimeError(f"No parents for heldout split {split_name}")
            split_samples, split_graphs, split_dialogues, split_validations = generate_split_parallel(
                split_parents,
                args.heldout_per_split,
                args,
                f"r2_auto_full_heldout_{split_name}",
                "heldout_eval_only",
            )
            for sample in split_samples:
                sample["heldout_split"] = split_name
                sample["metadata"]["heldout_split"] = split_name
            heldout_samples.extend(split_samples)
            heldout_graphs.extend(split_graphs)
            heldout_dialogues.extend(split_dialogues)
            heldout_validations.extend(split_validations)

        out_dir = Path(args.output_root)
        report_dir = Path(args.report_dir)
        summaries = {
            "train": summarize(train_samples, train_validations),
            "dev": summarize(dev_samples, dev_validations),
            "heldout": summarize(heldout_samples, heldout_validations),
        }
        write_auto_full_dataset(
            out_dir,
            report_dir,
            train_samples,
            dev_samples,
            heldout_samples,
            train_graphs + dev_graphs + heldout_graphs,
            train_dialogues + dev_dialogues + heldout_dialogues,
            summaries,
            source_manifest,
        )
        print(f"Saved r2_auto_full dataset to: {out_dir.resolve()}")
        print(f"Saved r2_auto_full report seed files to: {report_dir.resolve()}")
        return 0

    if args.parent_mode == "clean_v1":
        train_parents = load_clean_parent_samples(Path(args.source_manifest), "train")
        heldout_parents = load_clean_parent_samples(Path(args.source_manifest), "heldout")
    else:
        train_parents = load_inherited_samples(Path(args.train_source))
        heldout_parents = load_inherited_samples(Path(args.heldout_source))
    logging.info("Loaded inherited parents: train=%d heldout=%d", len(train_parents), len(heldout_parents))

    train_samples, train_graphs, train_dialogues, train_validations = generate_split(train_parents, args.n_train, args, "train_r2_llm", "train_candidate")
    if args.parent_mode == "clean_v1" and args.heldout_per_split > 0:
        heldout_samples = []
        heldout_graphs = []
        heldout_dialogues = []
        heldout_validations = []
        for split_name in ("sample7", "sample8", "sample9"):
            split_parents = [parent for parent in heldout_parents if parent.get("source_split") == split_name]
            split_samples, split_graphs, split_dialogues, split_validations = generate_split(
                split_parents,
                args.heldout_per_split,
                args,
                f"heldout_r2_llm_{split_name}",
                "evaluation_only",
            )
            for sample in split_samples:
                sample["heldout_split"] = split_name
                sample["metadata"]["heldout_split"] = split_name
            heldout_samples.extend(split_samples)
            heldout_graphs.extend(split_graphs)
            heldout_dialogues.extend(split_dialogues)
            heldout_validations.extend(split_validations)
    else:
        heldout_samples, heldout_graphs, heldout_dialogues, heldout_validations = generate_split(heldout_parents, args.n_heldout, args, "heldout_r2_llm", "evaluation_only")

    output_root = Path(args.output_root)
    suffix = "clean_v1_parent" if args.parent_mode == "clean_v1" else "graph_expansion"
    train_dir = output_root / f"train_r2_llm_{suffix}_pilot"
    heldout_dir = output_root / f"heldout_r2_llm_{suffix}_pilot"
    train_summary = write_dataset(train_dir, "sft_train.jsonl", train_samples, train_graphs, train_dialogues, train_validations)
    heldout_summary = write_dataset(heldout_dir, "sft_eval.jsonl", heldout_samples, heldout_graphs, heldout_dialogues, heldout_validations)

    try:
        eval_payload = evaluate_single_choice_subset(train_samples, args)
    except Exception as exc:  # pragma: no cover - depends on local model runtime
        eval_payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {"n": 0, "accuracy": None},
            "results": [],
        }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "r2_llm_graph_expansion",
        "method": "LLM-assisted graph update from inherited parent state chains; graph remains source of truth.",
        "parent_mode": args.parent_mode,
        "source_manifest": str(Path(args.source_manifest)) if args.parent_mode == "clean_v1" else None,
        "challenger": {"base_url": args.challenger_base_url, "model": args.challenger_model, "role": "API challenger"},
        "solver": {"base": args.solver_base, "adapter": args.solver_adapter, "role": "local fixed solver_r1; not retrained"},
        "train": train_summary,
        "heldout": heldout_summary,
        "output_dirs": {"train": str(train_dir), "heldout": str(heldout_dir)},
        "solver_r1_eval": eval_payload,
    }
    write_json(report_dir / "generation_summary.json", summary)
    if eval_payload is not None:
        write_json(report_dir / "solver_r1_eval.json", eval_payload)
    (report_dir / "status.md").write_text(format_status(summary, eval_payload), encoding="utf-8")
    print(f"Saved train to: {train_dir.resolve()}")
    print(f"Saved heldout to: {heldout_dir.resolve()}")
    print(f"Saved report to: {report_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
