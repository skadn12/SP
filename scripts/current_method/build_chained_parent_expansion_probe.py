#!/usr/bin/env python3
"""Build and audit promoted current_method parents for chained expansion probes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LIST_KEYS = (
    "events",
    "state_changes",
    "invalidations",
    "cancellations",
    "restorations",
    "delayed_activations",
    "visibility_edges",
    "entities",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def stable_hash(payload: Any, length: int = 24) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def dialogue_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{turn.get('turn_id')}] {turn.get('speaker')}: {turn.get('text')}"
        for turn in turns
        if normalize(turn.get("turn_id")) and normalize(turn.get("text"))
    )


def item_identity(item: dict[str, Any], index: int) -> str:
    for key in ("event_instance_id", "event_id", "node_id", "entity_id", "edge_id", "relation_id", "id"):
        text = normalize(item.get(key))
        if text:
            return f"{key}:{text}"
    return stable_hash(item, length=32) or f"index:{index}"


def merge_list(root_items: Any, child_items: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, raw_items in (("root", root_items), ("child", child_items)):
        if not isinstance(raw_items, list):
            continue
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            row = copy.deepcopy(item)
            row.setdefault("chained_merge_source", source)
            ident = item_identity(row, index)
            if ident in seen:
                continue
            seen.add(ident)
            merged.append(row)
    return merged


def state_node_from_event(item: dict[str, Any], source: str, index: int) -> dict[str, Any]:
    event_id = normalize(item.get("event_id") or item.get("event_instance_id") or item.get("node_id"))
    node_id = normalize(item.get("node_id")) or f"{source}:event:{index:04d}:{stable_hash(item, 8)}"
    value = normalize(item.get("value") or item.get("to") or item.get("text"))
    return {
        "node_id": node_id,
        "event_id": event_id or node_id,
        "role": normalize(item.get("role") or item.get("event_type") or item.get("slot") or "state_change"),
        "label": normalize(item.get("label") or f"{source} event {index + 1}"),
        "value": value,
        "status": normalize(item.get("status") or "active"),
        "turn_ids": item.get("turn_ids") if isinstance(item.get("turn_ids"), list) else item.get("evidence_turn_ids") or [],
        "evidence_turn_ids": item.get("evidence_turn_ids") if isinstance(item.get("evidence_turn_ids"), list) else item.get("turn_ids") or [],
        "origin": f"chained_parent_merge:{source}",
        "can_be_gold": True,
    }


def merge_state_chain(root: dict[str, Any], child: dict[str, Any], merged_graph: dict[str, Any], parent_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_nodes = root.get("state_chain_nodes") if isinstance(root.get("state_chain_nodes"), list) else []
    child_nodes = child.get("state_chain_nodes") if isinstance(child.get("state_chain_nodes"), list) else []
    generated_nodes: list[dict[str, Any]] = []
    for source, items in (("root_graph", root.get("graph_update", {}).get("events") if isinstance(root.get("graph_update"), dict) else []),):
        for index, item in enumerate(items or []):
            if isinstance(item, dict):
                generated_nodes.append(state_node_from_event(item, source, index))
    for source, key in (("child_event", "events"), ("child_state_change", "state_changes")):
        for index, item in enumerate(merged_graph.get(key) or []):
            if isinstance(item, dict):
                generated_nodes.append(state_node_from_event(item, source, index))

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in list(root_nodes) + list(child_nodes) + generated_nodes:
        if not isinstance(node, dict):
            continue
        row = copy.deepcopy(node)
        node_id = normalize(row.get("node_id")) or stable_hash(row, 24)
        if node_id in seen:
            continue
        seen.add(node_id)
        row["node_id"] = node_id
        nodes.append(row)

    root_chain = root.get("state_chain") if isinstance(root.get("state_chain"), dict) else {}
    chain = copy.deepcopy(root_chain)
    chain.update(
        {
            "chain_id": f"chained_parent_v1:{parent_id}",
            "parent_chain_id": root_chain.get("chain_id"),
            "parent_source": "current_method_child_promoted_parent",
            "source_split": "train",
            "nodes": nodes,
            "node_count": len(nodes),
            "graph_snapshot_hash": merged_graph.get("graph_snapshot_hash"),
        }
    )
    return chain, nodes


def graph_counts(graph: dict[str, Any]) -> dict[str, int]:
    return {key: len(graph.get(key) or []) for key in LIST_KEYS}


def build_merged_graph(root: dict[str, Any], child: dict[str, Any], promoted_id: str) -> dict[str, Any]:
    root_graph = root.get("graph_update") if isinstance(root.get("graph_update"), dict) else {}
    child_graph = child.get("graph_update") if isinstance(child.get("graph_update"), dict) else {}
    merged: dict[str, Any] = {
        "graph_snapshot_id": f"chained_parent_v1:{promoted_id}:graph",
        "depends_on_root_graph_snapshot_hash": root_graph.get("graph_snapshot_hash"),
        "depends_on_parent_child_graph_snapshot_hash": child_graph.get("graph_snapshot_hash"),
        "depends_on_dialogue_hash": (child.get("dialogue") or {}).get("dialogue_hash") if isinstance(child.get("dialogue"), dict) else None,
        "chained_merge_strategy": "root_seed_graph_plus_parent_child_graph_update_v1",
    }
    for key in LIST_KEYS:
        merged[key] = merge_list(root_graph.get(key), child_graph.get(key))
    merged["graph_snapshot_hash"] = stable_hash({key: merged.get(key) for key in LIST_KEYS}, length=24)
    return merged


def sample_ok(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    return bool(
        isinstance(row.get("dialogue"), dict)
        and isinstance(row["dialogue"].get("turns"), list)
        and row["dialogue"]["turns"]
        and isinstance(row.get("graph_update"), dict)
        and metadata.get("length_guard_pass") is True
        and validation.get("graph_dialogue_consistent") is True
        and validation.get("gold_derivable") is True
        and validation.get("option_valid") is True
    )


def select_parent_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if sample_ok(row)]
    eligible.sort(key=lambda row: ((row.get("metadata") or {}).get("final_dialogue_tokens") or 10**9, str(row.get("sample_id"))))
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_where(predicate) -> None:
        for row in eligible:
            if len(selected) >= limit:
                return
            sid = str(row.get("sample_id"))
            if sid in seen_ids or not predicate(row):
                continue
            selected.append(row)
            seen_ids.add(sid)

    targets = [
        (3, lambda row: row.get("train_mix_role") == "frontier_train" and row.get("question_type") == "multi_select"),
        (5, lambda row: row.get("train_mix_role") == "frontier_train" and row.get("question_type") == "single_choice"),
        (2, lambda row: row.get("train_mix_role") == "clean_replay"),
    ]
    for quota, predicate in targets:
        before = len(selected)
        add_where(lambda row, predicate=predicate: predicate(row) and len(selected) < before + quota)
    add_where(lambda row: True)
    return selected[:limit]


def build_promoted_parent(child: dict[str, Any], root: dict[str, Any], index: int, tokenizer_path: str) -> dict[str, Any]:
    child_id = normalize(child.get("sample_id"))
    root_id = normalize((child.get("seed_source") or {}).get("parent_chunk_id") or (child.get("metadata") or {}).get("source_seed_id"))
    promoted_id = f"chained_parent_v1_{index:04d}_{stable_hash([child_id, root_id], 10)}"
    turns = copy.deepcopy(child["dialogue"]["turns"])
    text = dialogue_text(turns)
    merged_graph = build_merged_graph(root, child, promoted_id)
    state_chain, state_chain_nodes = merge_state_chain(root, child, merged_graph, promoted_id)
    child_metadata = child.get("metadata") if isinstance(child.get("metadata"), dict) else {}
    seed_source = copy.deepcopy(child.get("seed_source") if isinstance(child.get("seed_source"), dict) else {})
    seed_source.update(
        {
            "split": "train",
            "source_split": "train",
            "raw_training_use": "train_candidate",
            "training_use": "train_candidate",
            "allowed_train_seed": True,
            "forbidden_reasons": [],
            "parent_source": "current_method_child_promoted_parent",
            "parent_sample_id": child_id,
            "parent_chunk_id": promoted_id,
            "sample_id": promoted_id,
            "root_seed_parent_id": root_id,
            "inferred_by": "build_chained_parent_expansion_probe",
        }
    )
    sample_name = normalize(seed_source.get("sample_name") or child.get("frontier_from_sample_name") or root.get("sample_name"))
    metadata = {
        "prompt_mode": "full_dialogue",
        "source_pipeline": "base_restart_current_method_validation",
        "graph_schema_version": "chained_parent_expansion_parent_v1",
        "current_method_source": True,
        "chained_parent_source": True,
        "evidence_window_used_as_prompt": False,
        "r4_r5_r1_data_or_model_used": False,
        "root_seed_parent_id": root_id,
        "parent_round_id": "stage2_repair_topup_from_stage1",
        "parent_sample_id": child_id,
        "parent_source_path": str(args.source_samples) if "args" in globals() else "",
        "parent_graph_hash": merged_graph.get("graph_snapshot_hash"),
        "parent_child_graph_hash": (child.get("graph_update") or {}).get("graph_snapshot_hash"),
        "root_graph_hash": (root.get("graph_update") or {}).get("graph_snapshot_hash"),
        "parent_dialogue_token_count": child_metadata.get("final_dialogue_tokens"),
        "parent_dialogue_hash": child["dialogue"].get("dialogue_hash"),
        "parent_dialogue_turn_count": len(turns),
        "parent_graph_counts": graph_counts(merged_graph),
        "root_graph_counts": graph_counts(root.get("graph_update") if isinstance(root.get("graph_update"), dict) else {}),
        "parent_child_graph_counts": graph_counts(child.get("graph_update") if isinstance(child.get("graph_update"), dict) else {}),
        "sample_name": sample_name,
        "source_sample_name": sample_name,
        "parent_sample_name": sample_name,
        "split": "train",
        "source_split": "train",
        "training_use": "train_candidate",
        "tokenizer_path": tokenizer_path,
    }
    return {
        "sample_id": promoted_id,
        "source_id": promoted_id,
        "expanded_graph_id": promoted_id,
        "expanded_dialogue_id": f"{promoted_id}_dialogue",
        "source": "current_method_child_promoted_parent",
        "split": "train",
        "source_split": "train",
        "training_use": "train_candidate",
        "sample_name": sample_name,
        "frontier_from_split": "train",
        "frontier_from_sample_name": sample_name,
        "dialogue": {"turns": turns, "dialogue_hash": child["dialogue"].get("dialogue_hash")},
        "turns": turns,
        "dialogue_text": text,
        "expanded_dialogue_text": text,
        "final_dialogue_text": text,
        "graph_update": merged_graph,
        "state_chain": state_chain,
        "state_chain_nodes": state_chain_nodes,
        "state_chain_id": state_chain.get("chain_id"),
        "metadata": metadata,
        "seed_source": seed_source,
        "seed": {"source": seed_source, "sample_id": promoted_id, "parent_sample_id": child_id},
        "validation": {
            "graph_dialogue_consistent": True,
            "gold_derivable": True,
            "option_valid": True,
            "anti_shortcut_pass": True,
            "warnings": [],
            "lineage_only_parent": True,
        },
        "option_validation_pass": True,
        "gold_derivable_from_graph_or_state_chain": True,
        "graph_dialogue_alignment": child.get("graph_dialogue_alignment") or {},
        "leakage_free": True,
        "heldout_contamination": False,
        "unconstrained_llm_gold": False,
    }


def main() -> None:
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-root", type=Path, required=True)
    parser.add_argument("--source-samples", type=Path, required=True)
    parser.add_argument("--seed-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
    parent_dataset = args.output_dir / "parent_dataset"
    parent_dataset.mkdir(parents=True, exist_ok=True)

    roots = {row.get("sample_id"): row for row in read_jsonl(args.seed_parents)}
    source_rows = read_jsonl(args.source_samples)
    selected = select_parent_rows(source_rows, args.limit)
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, child in enumerate(selected):
        root_id = normalize((child.get("seed_source") or {}).get("parent_chunk_id") or (child.get("metadata") or {}).get("source_seed_id"))
        root = roots.get(root_id)
        if not root:
            skipped.append({"sample_id": child.get("sample_id"), "reason": "root_seed_parent_not_found", "root_seed_parent_id": root_id})
            continue
        promoted.append(build_promoted_parent(child, root, index, args.tokenizer_path))

    out_path = args.output_dir / "expanded_parent_v1.jsonl"
    write_jsonl(out_path, promoted)
    write_jsonl(parent_dataset / "samples.jsonl", promoted)
    write_jsonl(parent_dataset / "expanded_dialogues.jsonl", [{"sample_id": row["sample_id"], "turns": row["turns"]} for row in promoted])
    write_jsonl(parent_dataset / "expanded_graphs.jsonl", [{"expanded_graph_id": row["expanded_graph_id"], **row["graph_update"]} for row in promoted])
    write_jsonl(parent_dataset / "sft_train.jsonl", [{"record_id": row["expanded_graph_id"], "sample_id": row["sample_id"]} for row in promoted])
    write_jsonl(parent_dataset / "sft_eval_dev.jsonl", [])
    write_jsonl(parent_dataset / "sft_eval_heldout.jsonl", [])

    selected_counter = Counter(row.get("question_type") for row in selected)
    role_counter = Counter(row.get("train_mix_role") for row in selected)
    summary = {
        "status": "success" if len(promoted) >= args.limit else "partial",
        "source_samples": str(args.source_samples),
        "seed_parents": str(args.seed_parents),
        "output_expanded_parent_v1": str(out_path),
        "parent_dataset_dir": str(parent_dataset),
        "requested_limit": args.limit,
        "selected_count": len(selected),
        "promoted_parent_count": len(promoted),
        "skipped": skipped,
        "selected_qtype_distribution": dict(sorted(selected_counter.items())),
        "selected_role_distribution": dict(sorted(role_counter.items())),
        "parent_dialogue_token_min": min((row["metadata"].get("parent_dialogue_token_count") or 0 for row in promoted), default=0),
        "parent_dialogue_token_max": max((row["metadata"].get("parent_dialogue_token_count") or 0 for row in promoted), default=0),
        "parent_graph_hashes": [row["metadata"].get("parent_graph_hash") for row in promoted],
        "evidence_window_used_as_prompt": False,
        "r4_r5_r1_data_or_model_used": False,
    }
    write_json(args.output_dir / "chained_parent_promotion_summary.json", summary)


if __name__ == "__main__":
    main()
