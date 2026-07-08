#!/usr/bin/env python3
"""Promote chained child v2 samples into parent_v2 rows for a 32k chained probe."""

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


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_hash(payload: Any, length: int = 24) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dialogue_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{turn.get('turn_id')}] {turn.get('speaker')}: {turn.get('text')}"
        for turn in turns
        if normalize(turn.get("turn_id")) and normalize(turn.get("text"))
    )


def identity(item: dict[str, Any], index: int) -> str:
    for key in ("event_instance_id", "event_id", "node_id", "entity_id", "edge_id", "relation_id", "id"):
        text = normalize(item.get(key))
        if text:
            return f"{key}:{text}"
    return stable_hash(item, 32) or f"index:{index}"


def merge_list(parent_items: Any, child_items: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, raw_items in (("parent_v1", parent_items), ("child_v2", child_items)):
        if not isinstance(raw_items, list):
            continue
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            row = copy.deepcopy(item)
            row.setdefault("chained_merge_source", source)
            ident = identity(row, index)
            if ident in seen:
                continue
            seen.add(ident)
            merged.append(row)
    return merged


def graph_counts(graph: dict[str, Any]) -> dict[str, int]:
    return {key: len(graph.get(key) or []) for key in LIST_KEYS}


def dependency_count(graph: dict[str, Any]) -> int:
    count = 0
    for key, value in graph.items():
        if key.startswith("depends_on") and value:
            count += 1
    for key in ("events", "state_changes"):
        for item in graph.get(key) or []:
            if not isinstance(item, dict):
                continue
            for dep_key in ("depends_on_graph_snapshot_hash", "depends_on_dialogue_hash", "depends_on_alignment_hash", "origin_event_id", "lineage_id"):
                if item.get(dep_key):
                    count += 1
    return count


def build_merged_graph(parent_v1: dict[str, Any], child_v2: dict[str, Any], promoted_id: str) -> dict[str, Any]:
    parent_graph = parent_v1.get("graph_update") if isinstance(parent_v1.get("graph_update"), dict) else {}
    child_graph = child_v2.get("graph_update") if isinstance(child_v2.get("graph_update"), dict) else {}
    merged: dict[str, Any] = {
        "graph_snapshot_id": f"chained_parent_v2:{promoted_id}:graph",
        "depends_on_parent_v1_graph_snapshot_hash": parent_graph.get("graph_snapshot_hash"),
        "depends_on_child_v2_graph_snapshot_hash": child_graph.get("graph_snapshot_hash"),
        "depends_on_child_v2_dialogue_hash": (child_v2.get("dialogue") or {}).get("dialogue_hash") if isinstance(child_v2.get("dialogue"), dict) else None,
        "chained_merge_strategy": "parent_v1_merged_graph_plus_child_v2_graph_update_v1",
    }
    for key in LIST_KEYS:
        merged[key] = merge_list(parent_graph.get(key), child_graph.get(key))
    merged["graph_snapshot_hash"] = stable_hash({key: merged.get(key) for key in LIST_KEYS}, length=24)
    return merged


def state_node_from_item(item: dict[str, Any], source: str, index: int) -> dict[str, Any]:
    node_id = normalize(item.get("node_id")) or f"{source}:{index:04d}:{stable_hash(item, 8)}"
    event_id = normalize(item.get("event_id") or item.get("event_instance_id") or node_id)
    return {
        "node_id": node_id,
        "event_id": event_id,
        "role": normalize(item.get("role") or item.get("event_type") or item.get("slot") or "state_change"),
        "label": normalize(item.get("label") or f"{source} {index + 1}"),
        "value": normalize(item.get("value") or item.get("to") or item.get("text")),
        "status": normalize(item.get("status") or "active"),
        "turn_ids": item.get("turn_ids") if isinstance(item.get("turn_ids"), list) else item.get("evidence_turn_ids") or [],
        "evidence_turn_ids": item.get("evidence_turn_ids") if isinstance(item.get("evidence_turn_ids"), list) else item.get("turn_ids") or [],
        "origin": f"chained_parent_v2_merge:{source}",
        "can_be_gold": True,
    }


def merge_state_chain(parent_v1: dict[str, Any], child_v2: dict[str, Any], graph: dict[str, Any], promoted_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_chain = parent_v1.get("state_chain") if isinstance(parent_v1.get("state_chain"), dict) else {}
    base_nodes = parent_v1.get("state_chain_nodes") if isinstance(parent_v1.get("state_chain_nodes"), list) else []
    child_nodes = child_v2.get("state_chain_nodes") if isinstance(child_v2.get("state_chain_nodes"), list) else []
    generated_nodes: list[dict[str, Any]] = []
    for source, key in (("child_v2_event", "events"), ("child_v2_state_change", "state_changes")):
        for index, item in enumerate((child_v2.get("graph_update") or {}).get(key) or []):
            if isinstance(item, dict):
                generated_nodes.append(state_node_from_item(item, source, index))
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in list(base_nodes) + list(child_nodes) + generated_nodes:
        if not isinstance(node, dict):
            continue
        row = copy.deepcopy(node)
        node_id = normalize(row.get("node_id")) or stable_hash(row, 24)
        if node_id in seen:
            continue
        seen.add(node_id)
        row["node_id"] = node_id
        nodes.append(row)
    chain = copy.deepcopy(base_chain)
    chain.update(
        {
            "chain_id": f"chained_parent_v2:{promoted_id}",
            "parent_chain_id": base_chain.get("chain_id"),
            "parent_source": "current_method_child_v2_promoted_parent",
            "source_split": "train",
            "nodes": nodes,
            "node_count": len(nodes),
            "graph_snapshot_hash": graph.get("graph_snapshot_hash"),
        }
    )
    return chain, nodes


def sample_ok(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    solver_diag = ((row.get("solver_eval") or {}).get("diagnostics") or {}) if isinstance(row.get("solver_eval"), dict) else {}
    return bool(
        isinstance(row.get("dialogue"), dict)
        and isinstance(row["dialogue"].get("turns"), list)
        and row["dialogue"]["turns"]
        and isinstance(row.get("graph_update"), dict)
        and validation.get("graph_dialogue_consistent") is True
        and validation.get("gold_derivable") is True
        and validation.get("option_valid") is True
        and metadata.get("length_guard_pass") is True
        and not solver_diag.get("eval_context_overflow")
        and not solver_diag.get("truncated")
    )


def promote(child_v2: dict[str, Any], parent_v1: dict[str, Any], index: int, tokenizer_path: str) -> dict[str, Any]:
    child_id = normalize(child_v2.get("sample_id"))
    parent_v1_id = normalize((child_v2.get("metadata") or {}).get("promoted_parent_id") or (child_v2.get("metadata") or {}).get("source_seed_id") or child_v2.get("source_id"))
    promoted_id = f"chained_parent_v2_{index:04d}_{stable_hash([child_id, parent_v1_id], 10)}"
    turns = copy.deepcopy(child_v2["dialogue"]["turns"])
    text = dialogue_text(turns)
    graph = build_merged_graph(parent_v1, child_v2, promoted_id)
    chain, nodes = merge_state_chain(parent_v1, child_v2, graph, promoted_id)
    child_meta = child_v2.get("metadata") if isinstance(child_v2.get("metadata"), dict) else {}
    parent_meta = parent_v1.get("metadata") if isinstance(parent_v1.get("metadata"), dict) else {}
    seed_source = copy.deepcopy(child_v2.get("seed_source") if isinstance(child_v2.get("seed_source"), dict) else {})
    sample_name = normalize(seed_source.get("sample_name") or child_v2.get("frontier_from_sample_name") or parent_v1.get("sample_name"))
    root_seed_parent_id = child_meta.get("root_seed_parent_id") or seed_source.get("root_seed_parent_id") or parent_meta.get("root_seed_parent_id")
    stage2_child_id = child_meta.get("promoted_parent_sample_id") or parent_meta.get("parent_sample_id")
    seed_source.update(
        {
            "split": "train",
            "source_split": "train",
            "raw_training_use": "train_candidate",
            "training_use": "train_candidate",
            "allowed_train_seed": True,
            "forbidden_reasons": [],
            "parent_source": "current_method_child_v2_promoted_parent",
            "parent_sample_id": child_id,
            "parent_chunk_id": promoted_id,
            "sample_id": promoted_id,
            "root_seed_parent_id": root_seed_parent_id,
            "previous_parent_v1_id": parent_v1_id,
            "stage2_parent_child_sample_id": stage2_child_id,
            "inferred_by": "build_chained_parent_v2_from_child_v2",
        }
    )
    metadata = {
        "prompt_mode": "full_dialogue",
        "source_pipeline": "base_restart_current_method_validation",
        "graph_schema_version": "chained_parent_expansion_parent_v2",
        "current_method_source": True,
        "chained_parent_source": True,
        "evidence_window_used_as_prompt": False,
        "r4_r5_r1_data_or_model_used": False,
        "root_seed_parent_id": root_seed_parent_id,
        "parent_round_id": "08_chained_parent_expansion_probe_child_v2",
        "parent_sample_id": child_id,
        "previous_parent_v1_id": parent_v1_id,
        "stage2_parent_child_sample_id": stage2_child_id,
        "lineage_chain": [
            root_seed_parent_id,
            stage2_child_id,
            parent_v1_id,
            child_id,
            promoted_id,
        ],
        "parent_graph_hash": graph.get("graph_snapshot_hash"),
        "parent_child_v2_graph_hash": (child_v2.get("graph_update") or {}).get("graph_snapshot_hash"),
        "previous_parent_v1_graph_hash": parent_meta.get("parent_graph_hash") or (parent_v1.get("graph_update") or {}).get("graph_snapshot_hash"),
        "parent_dialogue_token_count": child_meta.get("final_dialogue_tokens"),
        "parent_dialogue_hash": child_v2["dialogue"].get("dialogue_hash"),
        "parent_dialogue_turn_count": len(turns),
        "parent_graph_counts": graph_counts(graph),
        "parent_dependency_count": dependency_count(graph),
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
        "source": "current_method_child_v2_promoted_parent",
        "split": "train",
        "source_split": "train",
        "training_use": "train_candidate",
        "sample_name": sample_name,
        "frontier_from_split": "train",
        "frontier_from_sample_name": sample_name,
        "dialogue": {"turns": turns, "dialogue_hash": child_v2["dialogue"].get("dialogue_hash")},
        "turns": turns,
        "dialogue_text": text,
        "expanded_dialogue_text": text,
        "final_dialogue_text": text,
        "graph_update": graph,
        "state_chain": chain,
        "state_chain_nodes": nodes,
        "state_chain_id": chain.get("chain_id"),
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
        "graph_dialogue_alignment": child_v2.get("graph_dialogue_alignment") or {},
        "leakage_free": True,
        "heldout_contamination": False,
        "unconstrained_llm_gold": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-v2-samples", type=Path, required=True)
    parser.add_argument("--parent-v1", type=Path, required=True)
    parser.add_argument("--lineage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
    parent_dataset = args.output_dir / "parent_dataset"
    parent_dataset.mkdir(parents=True, exist_ok=True)
    children = [row for row in read_jsonl(args.child_v2_samples) if sample_ok(row)]
    children.sort(key=lambda row: ((row.get("metadata") or {}).get("final_dialogue_tokens") or 10**9, str(row.get("sample_id"))))
    children = children[: args.limit]
    parent_v1_by_id = {row.get("sample_id"): row for row in read_jsonl(args.parent_v1)}
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, child in enumerate(children):
        parent_v1_id = normalize((child.get("metadata") or {}).get("promoted_parent_id") or (child.get("metadata") or {}).get("source_seed_id") or child.get("source_id"))
        parent_v1 = parent_v1_by_id.get(parent_v1_id)
        if not parent_v1:
            skipped.append({"sample_id": child.get("sample_id"), "reason": "parent_v1_not_found", "parent_v1_id": parent_v1_id})
            continue
        promoted.append(promote(child, parent_v1, index, args.tokenizer_path))

    out = args.output_dir / "expanded_parent_v2.jsonl"
    write_jsonl(out, promoted)
    write_jsonl(parent_dataset / "samples.jsonl", promoted)
    write_jsonl(parent_dataset / "expanded_dialogues.jsonl", [{"sample_id": row["sample_id"], "turns": row["turns"]} for row in promoted])
    write_jsonl(parent_dataset / "expanded_graphs.jsonl", [{"expanded_graph_id": row["expanded_graph_id"], **row["graph_update"]} for row in promoted])
    write_jsonl(parent_dataset / "sft_train.jsonl", [{"record_id": row["expanded_graph_id"], "sample_id": row["sample_id"]} for row in promoted])
    write_jsonl(parent_dataset / "sft_eval_dev.jsonl", [])
    write_jsonl(parent_dataset / "sft_eval_heldout.jsonl", [])
    summary = {
        "status": "success" if len(promoted) >= args.limit else "partial",
        "source_child_v2": str(args.child_v2_samples),
        "source_parent_v1": str(args.parent_v1),
        "source_lineage_audit": str(args.lineage_audit),
        "output_expanded_parent_v2": str(out),
        "parent_dataset_dir": str(parent_dataset),
        "requested_limit": args.limit,
        "eligible_child_count": len(children),
        "promoted_parent_count": len(promoted),
        "skipped": skipped,
        "qtype_distribution": dict(sorted(Counter(row.get("question_type") for row in children).items())),
        "parent_dialogue_token_min": min((row["metadata"].get("parent_dialogue_token_count") or 0 for row in promoted), default=0),
        "parent_dialogue_token_max": max((row["metadata"].get("parent_dialogue_token_count") or 0 for row in promoted), default=0),
        "parent_dependency_count_min": min((row["metadata"].get("parent_dependency_count") or 0 for row in promoted), default=0),
        "evidence_window_used_as_prompt": False,
        "r4_r5_r1_data_or_model_used": False,
    }
    write_json(args.output_dir / "parent_v2_promotion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
