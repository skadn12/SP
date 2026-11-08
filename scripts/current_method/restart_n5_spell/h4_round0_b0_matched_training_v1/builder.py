"""Deterministic matched dataset builder for Clean versus inclusive B0 policies."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1 import QUESTION_KEYS
from ..h4_round0_b0_k3_difficulty_v1.contract import (
    STRUCTURED_SYSTEM_PROMPT,
    canonical_gold,
    dialogue_rows,
)
from ..h4_round0_b0_k3_difficulty_v1.io_utils import load_json, load_jsonl, verify_sha_sidecar
from ..prompts import SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT, SOLVER_TRAILING_FORMAT_INSTRUCTION
from . import (
    DATASET_VERSION,
    EVAL_FREEZE_VERSION,
    MATCHING_ALGORITHM_VERSION,
    PERMUTATION_ALGORITHM_VERSION,
    RENDER_ALGORITHM_VERSION,
)


EXPECTED_COUNTS = {
    "difficulty": 3286,
    "train": 2294,
    "dev": 485,
    "test": 507,
    "train_hard": 544,
    "dev_hard": 99,
    "test_hard": 118,
    "train_clean": 386,
    "train_risk": 158,
}
REPLACEMENT_RATES = (0.10, 0.20)
MAX_ACCURACY_DELTA = 0.20
MAX_TOKEN_RELATIVE_DELTA = 0.20
MAX_TRAIN_TOKENS = 16_384
REQUIRED_TRAIN_MAX_LENGTH = 8_192
ALLOWED_BLACKLIST_REASONS = {
    "CONFIRMED_GOLD_WRONG",
    "CONFIRMED_ORDERING_WRONG",
    "CONFIRMED_NON_UNIQUE_ANSWER",
    "CONFIRMED_SUBJECT_OR_SCOPE_WRONG",
}
REQUIRED_DIFFICULTY_FILES = (
    "difficulty_manifest.jsonl",
    "program_gate_results.jsonl",
    "root_split_manifest.jsonl",
    "solver_contract.json",
)
REQUIRED_STRATIFICATION_FILES = (
    "hard_evidence_status_manifest.jsonl",
    "final_stratification_report.json",
)


def stable_int(*parts: Any) -> int:
    payload = canonical_json([str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(bytes.fromhex(sha256_bytes(payload))[:8], "big")


def _relative_delta(left: int | float, right: int | float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1.0)


def _id_map(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"{label}_row_not_object:{index}")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in output:
            raise RuntimeError(f"{label}_candidate_id_missing_or_duplicate:{candidate_id}:{index}")
        output[candidate_id] = dict(row)
    return output


def _source_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"required_source_file_missing:{path}")
        verify_sha_sidecar(path)
        hashes[name] = sha256_file(path)
    return hashes


def load_confirmed_invalid_blacklist(path: Path, *, known_ids: set[str]) -> dict[str, Any]:
    path = path.resolve()
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError("confirmed_invalid_blacklist_not_object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("confirmed_invalid_blacklist_entries_not_list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"blacklist_entry_not_object:{index}")
        candidate_id = str(entry.get("candidate_id") or "")
        reason = str(entry.get("reason_code") or "")
        evidence_path = Path(str(entry.get("review_artifact_path") or "")).resolve()
        expected_sha = str(entry.get("review_artifact_sha256") or "")
        reviewer = str(entry.get("reviewer") or "")
        if not candidate_id or candidate_id in seen or candidate_id not in known_ids:
            raise RuntimeError(f"blacklist_candidate_invalid:{candidate_id}")
        if reason not in ALLOWED_BLACKLIST_REASONS:
            raise RuntimeError(f"blacklist_reason_not_allowed:{candidate_id}:{reason}")
        if not reviewer or not evidence_path.is_file() or sha256_file(evidence_path) != expected_sha:
            raise RuntimeError(f"blacklist_fixed_review_evidence_invalid:{candidate_id}")
        seen.add(candidate_id)
        normalized.append({
            "candidate_id": candidate_id,
            "reason_code": reason,
            "reviewer": reviewer,
            "review_artifact_path": str(evidence_path),
            "review_artifact_sha256": expected_sha,
        })
    return {
        "blacklist_version": str(payload.get("blacklist_version") or ""),
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "entry_count": len(normalized),
        "entries": sorted(normalized, key=lambda row: row["candidate_id"]),
        "excluded_auditor_labels": True,
    }


def _training_user_content(
    candidate: Mapping[str, Any],
    display: Mapping[str, Any],
    dialogue: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    qtype = str(candidate["question_type"])
    chunks = ["Context:\n" + "\n".join(
        f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in dialogue
    )] if qtype == "single_choice" else ["Dialogue:\n" + "\n".join(
        f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in dialogue
    )]
    if qtype == "single_choice":
        chunks.append(str(candidate["question"]))
        chunks.append("\n".join(f"{key}. {value}" for key, value in display["options"].items()))
        chunks.append(SOLVER_TRAILING_FORMAT_INSTRUCTION)
        system = SOLVER_POSITION_NEUTRAL_RESPONDER_PROMPT
    else:
        chunks.append(f"Question ({qtype}):\n{candidate['question']}")
        field = "options" if qtype == "multi_select" else "items"
        heading = "Options" if qtype == "multi_select" else "Items"
        chunks.append(heading + ":\n" + "\n".join(
            f"{key}. {value}" for key, value in display[field].items()
        ))
        chunks.append(
            "Output only selected letters separated by commas, e.g. <box>A,C</box>."
            if qtype == "multi_select" else
            "Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>."
        )
        system = STRUCTURED_SYSTEM_PROMPT
    return system, "\n\n".join(chunks)


def target_signatures(question_type: str, cardinality: int) -> list[tuple[str, ...]]:
    keys = tuple(QUESTION_KEYS[question_type])
    if question_type == "ordering":
        return list(itertools.permutations(keys))
    return list(itertools.combinations(keys, cardinality))


def build_display(
    candidate: Mapping[str, Any],
    *,
    desired_target: Sequence[str],
    namespace: str,
) -> dict[str, Any]:
    qtype = str(candidate["question_type"])
    keys = list(QUESTION_KEYS[qtype])
    field = "items" if qtype == "ordering" else "options"
    original = {key: str((candidate.get(field) or {})[key]) for key in keys}
    gold = canonical_gold(candidate)
    desired = [str(value) for value in desired_target]
    if len(desired) != len(gold) or len(set(desired)) != len(desired):
        raise RuntimeError(f"display_target_cardinality_invalid:{candidate.get('candidate_id')}")
    if any(value not in keys for value in desired):
        raise RuntimeError(f"display_target_label_invalid:{candidate.get('candidate_id')}")
    mapping: dict[str, str] = {}
    for display_key, canonical_key in zip(desired, gold):
        mapping[display_key] = canonical_key
    remaining_display = [key for key in keys if key not in mapping]
    remaining_canonical = [key for key in keys if key not in set(mapping.values())]
    remaining_canonical.sort(key=lambda key: (stable_int(namespace, "remaining", key), key))
    for display_key, canonical_key in zip(remaining_display, remaining_canonical):
        mapping[display_key] = canonical_key
    if set(mapping) != set(keys) or set(mapping.values()) != set(keys):
        raise RuntimeError(f"display_mapping_not_bijective:{candidate.get('candidate_id')}")
    displayed = {display_key: original[mapping[display_key]] for display_key in keys}
    rebound = [mapping[label] for label in desired]
    semantic_match = set(rebound) == set(gold) if qtype == "multi_select" else rebound == gold
    if not semantic_match:
        raise RuntimeError(f"display_gold_rebinding_failed:{candidate.get('candidate_id')}")
    return {
        "display_mapping_version": PERMUTATION_ALGORITHM_VERSION,
        "display_to_canonical": mapping,
        field: displayed,
        "displayed_gold": desired,
        "canonical_gold": gold,
        "semantic_gold_preserved": True,
    }


def render_training_record(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    dialogue: Sequence[Mapping[str, Any]],
    desired_target: Sequence[str],
    policy: str,
    slot_id: str,
    selection_role: str,
    pair_id: str | None,
    order_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = row["candidate"]
    display = build_display(
        candidate, desired_target=desired_target,
        namespace=f"{DATASET_VERSION}:{slot_id}",
    )
    system, user = _training_user_content(candidate, display, dialogue)
    target = "<box>" + ",".join(display["displayed_gold"]) + "</box>"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids.append(tokenizer.eos_token_id)
    supervised = len(full_ids) - min(len(prompt_ids), len(full_ids))
    if supervised <= 0 or len(full_ids) > MAX_TRAIN_TOKENS:
        raise RuntimeError(
            f"training_token_contract_failed:{row['candidate_id']}:{len(full_ids)}:{supervised}"
        )
    metadata = {
        "dataset_version": DATASET_VERSION,
        "render_algorithm_version": RENDER_ALGORITHM_VERSION,
        "permutation_algorithm_version": PERMUTATION_ALGORITHM_VERSION,
        "policy": policy,
        "slot_id": slot_id,
        "order_key": order_key,
        "selection_role": selection_role,
        "pair_id": pair_id,
        "candidate_id": row["candidate_id"],
        "root_id": row["root_id"],
        "split": row["split"],
        "question_type": row["question_type"],
        "gold_cardinality": len(canonical_gold(candidate)),
        "generation_source": row["generation_source"],
        "evidence_status": row["evidence_status"],
        "position_mechanism": row["position_mechanism"],
        "full_context_accuracy": row["full_context_accuracy"],
        "no_context_accuracy": row["no_context_accuracy"],
        "display_to_canonical": display["display_to_canonical"],
        "displayed_target": list(display["displayed_gold"]),
        "canonical_gold": list(display["canonical_gold"]),
        "target_text": target,
        "prompt_token_count": len(prompt_ids),
        "input_token_count": len(full_ids),
        "supervised_token_count": supervised,
        "candidate_pool_path": row["candidate_pool_path"],
        "candidate_pool_sha256": row["candidate_pool_sha256"],
        "dialogue_path": row["dialogue_path"],
        "dialogue_sha256": row["dialogue_sha256"],
        "difficulty_manifest_path": row["difficulty_manifest_path"],
        "difficulty_manifest_sha256": row["difficulty_manifest_sha256"],
    }
    manifest = {**metadata, "messages_sha256": sha256_bytes(canonical_json(messages).encode("utf-8"))}
    training = {"messages": messages, "metadata": metadata}
    return manifest, training


def _profile_row(
    row: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    tokenizer: Any,
    dialogue_cache: dict[str, tuple[list[dict[str, str]], int]],
    difficulty_path: Path,
    difficulty_sha: str,
) -> dict[str, Any]:
    path = str(Path(str(row["dialogue_path"])).resolve())
    if path not in dialogue_cache:
        payload = load_json(Path(path))
        dialogue = dialogue_rows(payload)
        text = "\n".join(f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}" for turn in dialogue)
        dialogue_cache[path] = (dialogue, len(tokenizer.encode(text, add_special_tokens=False)))
    dialogue, context_tokens = dialogue_cache[path]
    candidate = row["candidate"]
    keys = QUESTION_KEYS[str(row["question_type"])]
    identity = build_display(
        candidate,
        desired_target=canonical_gold(candidate),
        namespace=f"{DATASET_VERSION}:profile:{row['candidate_id']}",
    )
    system, user = _training_user_content(candidate, identity, dialogue)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    profile = dict(row)
    profile.update({
        "gold_cardinality": len(canonical_gold(candidate)),
        "context_token_count": context_tokens,
        "training_prompt_token_count": len(tokenizer.encode(prompt_text, add_special_tokens=False)),
        "gate_max_prompt_token_count": int(gate["max_prompt_token_count"]),
        "difficulty_manifest_path": str(difficulty_path),
        "difficulty_manifest_sha256": difficulty_sha,
    })
    if set(identity["display_to_canonical"]) != set(keys):
        raise RuntimeError(f"profile_display_keys_invalid:{row['candidate_id']}")
    return profile


def matching_stratum(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["question_type"]),
        int(row["gold_cardinality"]),
        str(row["position_mechanism"]),
    )


def pair_metrics(risk: Mapping[str, Any], clean: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_accuracy_delta": abs(float(risk["full_context_accuracy"]) - float(clean["full_context_accuracy"])),
        "no_accuracy_delta": abs(float(risk["no_context_accuracy"]) - float(clean["no_context_accuracy"])),
        "context_token_relative_delta": _relative_delta(risk["context_token_count"], clean["context_token_count"]),
        "prompt_token_relative_delta": _relative_delta(risk["training_prompt_token_count"], clean["training_prompt_token_count"]),
        "same_root": risk["root_id"] == clean["root_id"],
    }


def pair_is_eligible(risk: Mapping[str, Any], clean: Mapping[str, Any]) -> bool:
    if matching_stratum(risk) != matching_stratum(clean):
        return False
    metrics = pair_metrics(risk, clean)
    return (
        metrics["full_accuracy_delta"] <= MAX_ACCURACY_DELTA + 1e-12
        and metrics["no_accuracy_delta"] <= MAX_ACCURACY_DELTA + 1e-12
        and metrics["context_token_relative_delta"] <= MAX_TOKEN_RELATIVE_DELTA + 1e-12
        and metrics["prompt_token_relative_delta"] <= MAX_TOKEN_RELATIVE_DELTA + 1e-12
    )


def pair_cost(risk: Mapping[str, Any], clean: Mapping[str, Any]) -> int:
    metrics = pair_metrics(risk, clean)
    base = (
        metrics["full_accuracy_delta"] * 100_000
        + metrics["no_accuracy_delta"] * 100_000
        + metrics["context_token_relative_delta"] * 50_000
        + metrics["prompt_token_relative_delta"] * 50_000
        + (0 if metrics["same_root"] else 1_000)
    )
    return int(round(base)) * 100 + stable_int(
        MATCHING_ALGORITHM_VERSION, risk["candidate_id"], clean["candidate_id"],
    ) % 100


def _maximum_capacity(risks: Sequence[Mapping[str, Any]], cleans: Sequence[Mapping[str, Any]]) -> int:
    graph = nx.Graph()
    for risk in risks:
        for clean in cleans:
            if pair_is_eligible(risk, clean):
                graph.add_edge(f"r:{risk['candidate_id']}", f"c:{clean['candidate_id']}")
    if not graph.edges:
        return 0
    return len(nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True))


def _allocate_with_caps(weights: Mapping[Any, int], caps: Mapping[Any, int], target: int) -> dict[Any, int]:
    keys = sorted(weights, key=str)
    if target > sum(caps.get(key, 0) for key in keys):
        raise RuntimeError(f"matching_capacity_below_target:{target}:{sum(caps.values())}")
    total_weight = sum(weights[key] for key in keys if caps.get(key, 0) > 0)
    raw = {key: target * weights[key] / total_weight if caps.get(key, 0) > 0 else 0.0 for key in keys}
    result = {key: min(caps.get(key, 0), int(math.floor(raw[key]))) for key in keys}
    while sum(result.values()) < target:
        available = [key for key in keys if result[key] < caps.get(key, 0)]
        if not available:
            raise RuntimeError("matching_allocation_exhausted")
        selected = max(
            available,
            key=lambda key: (raw[key] - result[key], weights[key], -stable_int(MATCHING_ALGORITHM_VERSION, key)),
        )
        result[selected] += 1
    return result


def _min_cost_pairs(
    risks: Sequence[Mapping[str, Any]],
    cleans: Sequence[Mapping[str, Any]],
    *,
    target: int,
    stratum: tuple[str, int, str],
    namespace: str,
) -> list[dict[str, Any]]:
    if target == 0:
        return []
    graph = nx.DiGraph()
    source, sink = "source", "sink"
    graph.add_node(source, demand=-target)
    graph.add_node(sink, demand=target)
    risks_by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for risk in risks:
        risks_by_root[str(risk["root_id"])].append(risk)
        rin, rout = f"risk_in:{risk['candidate_id']}", f"risk_out:{risk['candidate_id']}"
        graph.add_edge(rin, rout, capacity=1, weight=0)
    for root_id, root_risks in sorted(risks_by_root.items()):
        for index in range(min(len(root_risks), target)):
            slot = f"root_slot:{root_id}:{index}"
            graph.add_edge(source, slot, capacity=1, weight=index * 3_000_000)
            for risk in root_risks:
                graph.add_edge(slot, f"risk_in:{risk['candidate_id']}", capacity=1, weight=0)
    for clean in cleans:
        graph.add_edge(f"clean:{clean['candidate_id']}", sink, capacity=1, weight=0)
    clean_by_id = {str(row["candidate_id"]): row for row in cleans}
    risk_by_id = {str(row["candidate_id"]): row for row in risks}
    eligible_edges = 0
    for risk in risks:
        for clean in cleans:
            if not pair_is_eligible(risk, clean):
                continue
            eligible_edges += 1
            graph.add_edge(
                f"risk_out:{risk['candidate_id']}", f"clean:{clean['candidate_id']}",
                capacity=1, weight=pair_cost(risk, clean),
            )
    if eligible_edges == 0:
        raise RuntimeError(f"matching_stratum_has_no_edges:{stratum}")
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible as error:
        raise RuntimeError(f"matching_stratum_flow_infeasible:{stratum}:{target}") from error
    pairs: list[dict[str, Any]] = []
    for risk_id, risk in sorted(risk_by_id.items()):
        node = f"risk_out:{risk_id}"
        for target_node, amount in flow.get(node, {}).items():
            if amount != 1 or not target_node.startswith("clean:"):
                continue
            clean_id = target_node.removeprefix("clean:")
            clean = clean_by_id[clean_id]
            metrics = pair_metrics(risk, clean)
            pair_hash = hashlib.sha256(canonical_json({
                "version": MATCHING_ALGORITHM_VERSION,
                "namespace": namespace,
                "risk": risk_id,
                "clean": clean_id,
                "stratum": stratum,
            }).encode("utf-8")).hexdigest()
            pairs.append({
                "pair_id": f"pair-{pair_hash[:20]}",
                "pair_hash": pair_hash,
                "stratum": {
                    "question_type": stratum[0],
                    "gold_cardinality": stratum[1],
                    "position_mechanism": stratum[2],
                },
                "clean_candidate_id": clean_id,
                "risk_candidate_id": risk_id,
                "clean_root_id": clean["root_id"],
                "risk_root_id": risk["root_id"],
                "clean_generation_source": clean["generation_source"],
                "risk_generation_source": risk["generation_source"],
                **metrics,
                "matching_cost": pair_cost(risk, clean),
                "matching_algorithm_version": MATCHING_ALGORITHM_VERSION,
            })
    if len(pairs) != target:
        raise RuntimeError(f"matching_flow_count_mismatch:{stratum}:{len(pairs)}:{target}")
    return sorted(pairs, key=lambda row: row["pair_id"])


def match_at_rate(
    clean_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    *,
    rate: float,
) -> dict[str, Any]:
    target = int(math.floor(len(clean_rows) * rate + 0.5))
    clean_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    risk_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        clean_groups[matching_stratum(row)].append(row)
    for row in risk_rows:
        risk_groups[matching_stratum(row)].append(row)
    strata = sorted(set(clean_groups) | set(risk_groups), key=str)
    capacities = {
        stratum: _maximum_capacity(risk_groups[stratum], clean_groups[stratum])
        for stratum in strata
    }
    weights = {stratum: len(risk_groups[stratum]) for stratum in strata}
    total_capacity = sum(capacities.values())
    allocation_details = {
        "|".join(map(str, key)): {
            "clean_available": len(clean_groups[key]),
            "risk_available": len(risk_groups[key]),
            "eligible_capacity": capacities[key],
            "selected_pairs": 0,
        }
        for key in strata
    }
    if target > total_capacity:
        return {
            "replacement_rate": rate,
            "requested_pair_count": target,
            "feasible": False,
            "infeasible_reason": "eligible_pair_capacity_below_requested_pair_count",
            "eligible_pair_capacity": total_capacity,
            "pair_count": 0,
            "allocation_by_stratum": allocation_details,
            "risk_root_count": 0,
            "clean_root_count": 0,
            "same_root_pair_count": 0,
            "max_pair_deltas": {},
            "mean_pair_deltas": {},
            "pairs": [],
        }
    allocation = _allocate_with_caps(weights, capacities, target)
    pairs: list[dict[str, Any]] = []
    for stratum in strata:
        pairs.extend(_min_cost_pairs(
            risk_groups[stratum], clean_groups[stratum],
            target=allocation[stratum], stratum=stratum,
            namespace=f"{DATASET_VERSION}:rate:{rate:.2f}",
        ))
    clean_ids = [row["clean_candidate_id"] for row in pairs]
    risk_ids = [row["risk_candidate_id"] for row in pairs]
    if len(pairs) != target or len(set(clean_ids)) != target or len(set(risk_ids)) != target:
        raise RuntimeError(f"matching_unique_count_failed:{rate}:{len(pairs)}")
    return {
        "replacement_rate": rate,
        "requested_pair_count": target,
        "feasible": True,
        "eligible_pair_capacity": total_capacity,
        "pair_count": len(pairs),
        "allocation_by_stratum": {
            "|".join(map(str, key)): {
                "clean_available": len(clean_groups[key]),
                "risk_available": len(risk_groups[key]),
                "eligible_capacity": capacities[key],
                "selected_pairs": allocation[key],
            }
            for key in strata
        },
        "risk_root_count": len({row["risk_root_id"] for row in pairs}),
        "clean_root_count": len({row["clean_root_id"] for row in pairs}),
        "same_root_pair_count": sum(row["same_root"] for row in pairs),
        "max_pair_deltas": {
            key: max((float(row[key]) for row in pairs), default=0.0)
            for key in (
                "full_accuracy_delta", "no_accuracy_delta",
                "context_token_relative_delta", "prompt_token_relative_delta",
            )
        },
        "mean_pair_deltas": {
            key: statistics.fmean(float(row[key]) for row in pairs) if pairs else 0.0
            for key in (
                "full_accuracy_delta", "no_accuracy_delta",
                "context_token_relative_delta", "prompt_token_relative_delta",
            )
        },
        "pairs": sorted(pairs, key=lambda row: row["pair_id"]),
    }


def assign_slot_targets(slots: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        groups[(slot["question_type"], slot["gold_cardinality"])].append(slot)
    assignments: dict[str, tuple[str, ...]] = {}
    for group_key, values in sorted(groups.items()):
        signatures = target_signatures(*group_key)
        ordered = sorted(values, key=lambda row: (stable_int(PERMUTATION_ALGORITHM_VERSION, row["slot_id"]), row["slot_id"]))
        offset = stable_int(PERMUTATION_ALGORITHM_VERSION, group_key) % len(signatures)
        for index, slot in enumerate(ordered):
            assignments[slot["slot_id"]] = signatures[(index + offset) % len(signatures)]
    return assignments


def _distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    targets = [tuple(row["displayed_target"]) for row in rows]
    single = Counter(target[0] for row, target in zip(rows, targets) if row["question_type"] == "single_choice")
    multi_rows = [(row, target) for row, target in zip(rows, targets) if row["question_type"] == "multi_select"]
    ordering_rows = [(row, target) for row, target in zip(rows, targets) if row["question_type"] == "ordering"]
    multi_label = Counter(label for _, target in multi_rows for label in target)
    multi_sets = Counter(",".join(target) for _, target in multi_rows)
    rank_counts = {
        str(rank + 1): dict(sorted(Counter(target[rank] for _, target in ordering_rows).items()))
        for rank in range(4)
    }
    return {
        "candidate_count": len(rows),
        "question_type_counts": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "gold_cardinality_counts": dict(sorted(Counter(f"{row['question_type']}:{row['gold_cardinality']}" for row in rows).items())),
        "position_mechanism_counts": dict(sorted(Counter(row["position_mechanism"] for row in rows).items())),
        "generation_source_counts": dict(sorted(Counter(row["generation_source"] for row in rows).items())),
        "evidence_status_counts": dict(sorted(Counter(row["evidence_status"] for row in rows).items())),
        "single_target_label_counts": dict(sorted(single.items())),
        "multi_target_label_occurrence_counts": dict(sorted(multi_label.items())),
        "multi_target_set_counts": dict(sorted(multi_sets.items())),
        "ordering_target_label_by_rank": rank_counts,
        "root_count": len({row["root_id"] for row in rows}),
    }


def _numeric_summary(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[index],
        "max": ordered[-1],
    }


def _argparse_default(path: Path, flag: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise RuntimeError(f"trainer_argument_default_not_found:{flag}:{path}")


def _write_json(path: Path, value: Any) -> None:
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_text(path, text)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def _write_text(path: Path, value: str) -> None:
    atomic_text(path, value)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def _eval_row(row: Mapping[str, Any], evidence: Mapping[str, Any] | None, *, difficulty_path: Path, difficulty_sha: str) -> dict[str, Any]:
    return {
        "eval_freeze_version": EVAL_FREEZE_VERSION,
        "candidate_id": row["candidate_id"],
        "root_id": row["root_id"],
        "split": row["split"],
        "question_type": row["question_type"],
        "generation_source": row["generation_source"],
        "difficulty_bucket": row["difficulty_bucket"],
        "m0_hard": row["difficulty_bucket"] == "hard_candidate",
        "evidence_status": evidence.get("evidence_status") if evidence else "NOT_HARD_NOT_STRATIFIED",
        "position_mechanism": row["position_mechanism"],
        "candidate": row["candidate"],
        "candidate_pool_path": row["candidate_pool_path"],
        "candidate_pool_sha256": row["candidate_pool_sha256"],
        "dialogue_path": row["dialogue_path"],
        "dialogue_sha256": row["dialogue_sha256"],
        "difficulty_manifest_path": str(difficulty_path),
        "difficulty_manifest_sha256": difficulty_sha,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Round-0 B0 Matched Training Dataset",
        "",
        f"- Status: **{report['status']}**",
        f"- Selected replacement rate: `{report['selected_replacement_rate']}`",
        f"- Candidates per policy: `{report['candidate_count_per_policy']}`",
        f"- Shared Clean core: `{report['shared_clean_count']}`",
        f"- Clean controls / Risk replacements: `{report['replacement_count']}` / `{report['replacement_count']}`",
        f"- Confirmed-invalid blacklist entries: `{report['blacklist_entry_count']}`",
        f"- Dev full / M0-hard: `{report['dev_full_count']}` / `{report['dev_m0_hard_count']}`",
        f"- Test full / M0-hard: `{report['test_full_count']}` / `{report['test_m0_hard_count']}`",
        "",
        "No model was loaded for training, no Solver or external API was called, and no training was started.",
        "The experiment compares current production policies; generation source remains a documented confounder.",
        "",
        "## Main Outputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["major_result_paths"])
    return "\n".join(lines) + "\n"


def build(
    *,
    difficulty_run: Path,
    stratification_run: Path,
    blacklist_path: Path,
    tokenizer_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    difficulty_run = difficulty_run.resolve()
    stratification_run = stratification_run.resolve()
    tokenizer_path = tokenizer_path.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"output_run_already_exists:{run_root}")
    difficulty_hashes = _source_hashes(difficulty_run, REQUIRED_DIFFICULTY_FILES)
    stratification_hashes = _source_hashes(stratification_run, REQUIRED_STRATIFICATION_FILES)
    difficulty_path = difficulty_run / "difficulty_manifest.jsonl"
    difficulty_rows = load_jsonl(difficulty_path)
    difficulty = _id_map(difficulty_rows, label="difficulty")
    gate = _id_map(load_jsonl(difficulty_run / "program_gate_results.jsonl"), label="program_gate")
    evidence = _id_map(load_jsonl(stratification_run / "hard_evidence_status_manifest.jsonl"), label="evidence_status")
    if len(difficulty) != EXPECTED_COUNTS["difficulty"] or len(evidence) != 761:
        raise RuntimeError(f"source_count_mismatch:{len(difficulty)}:{len(evidence)}")
    split_counts = Counter(str(row["split"]) for row in difficulty.values())
    for split in ("train", "dev", "test"):
        if split_counts[split] != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"difficulty_split_count_mismatch:{split}:{split_counts[split]}")
    hard_split_counts = Counter(str(row["split"]) for row in difficulty.values() if row["difficulty_bucket"] == "hard_candidate")
    for split in ("train", "dev", "test"):
        if hard_split_counts[split] != EXPECTED_COUNTS[f"{split}_hard"]:
            raise RuntimeError(f"hard_split_count_mismatch:{split}:{hard_split_counts[split]}")
    for candidate_id, status in evidence.items():
        row = difficulty.get(candidate_id)
        if row is None or row["difficulty_bucket"] != "hard_candidate" or row["split"] != status["split"]:
            raise RuntimeError(f"evidence_difficulty_identity_mismatch:{candidate_id}")
    blacklist = load_confirmed_invalid_blacklist(blacklist_path, known_ids=set(difficulty))
    blacklist_ids = {row["candidate_id"] for row in blacklist["entries"]}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True,
    )
    solver_contract = load_json(difficulty_run / "solver_contract.json")
    if str(Path(str(solver_contract["tokenizer"])).resolve()) != str(tokenizer_path):
        raise RuntimeError("tokenizer_path_differs_from_b0_contract")
    chat_template = str(tokenizer.chat_template or "")
    if sha256_bytes(chat_template.encode("utf-8")) != solver_contract["chat_template_sha256"]:
        raise RuntimeError("tokenizer_chat_template_hash_differs_from_b0_contract")

    dialogue_cache: dict[str, tuple[list[dict[str, str]], int]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for candidate_id, status in evidence.items():
        source = difficulty[candidate_id]
        for key in ("candidate_pool_path", "dialogue_path"):
            path = Path(str(source[key]))
            sha_key = key.replace("_path", "_sha256")
            if not path.is_file() or sha256_file(path) != source[sha_key]:
                raise RuntimeError(f"training_lineage_sha_mismatch:{candidate_id}:{key}")
        row = {**source, "evidence_status": status["evidence_status"]}
        profiles[candidate_id] = _profile_row(
            row, gate=gate[candidate_id], tokenizer=tokenizer,
            dialogue_cache=dialogue_cache, difficulty_path=difficulty_path,
            difficulty_sha=difficulty_hashes["difficulty_manifest.jsonl"],
        )
    train_clean = sorted(
        (row for row in profiles.values() if row["split"] == "train" and row["evidence_status"] == "EVIDENCE_CLEAN" and row["candidate_id"] not in blacklist_ids),
        key=lambda row: row["candidate_id"],
    )
    train_risk = sorted(
        (row for row in profiles.values() if row["split"] == "train" and row["evidence_status"] == "EVIDENCE_RISK" and row["candidate_id"] not in blacklist_ids),
        key=lambda row: row["candidate_id"],
    )
    if not blacklist_ids and (len(train_clean), len(train_risk)) != (EXPECTED_COUNTS["train_clean"], EXPECTED_COUNTS["train_risk"]):
        raise RuntimeError(f"train_clean_risk_count_mismatch:{len(train_clean)}:{len(train_risk)}")

    feasibility = [match_at_rate(train_clean, train_risk, rate=rate) for rate in REPLACEMENT_RATES]
    by_rate = {float(row["replacement_rate"]): row for row in feasibility}
    selected_rate = 0.20 if by_rate[0.20]["feasible"] else 0.10
    selected = by_rate[selected_rate]
    if not selected["feasible"]:
        raise RuntimeError("neither_20_nor_10_percent_matching_is_feasible")
    pairs = selected["pairs"]
    clean_by_id = {row["candidate_id"]: row for row in train_clean}
    risk_by_id = {row["candidate_id"]: row for row in train_risk}
    control_ids = {row["clean_candidate_id"] for row in pairs}
    shared_rows = [row for row in train_clean if row["candidate_id"] not in control_ids]
    if len(shared_rows) + len(pairs) != len(train_clean):
        raise RuntimeError("shared_control_partition_mismatch")

    slots: list[dict[str, Any]] = []
    for row in shared_rows:
        slots.append({
            "slot_id": f"shared:{row['candidate_id']}",
            "question_type": row["question_type"],
            "gold_cardinality": row["gold_cardinality"],
            "shared_candidate_id": row["candidate_id"],
            "pair_id": None,
        })
    for pair in pairs:
        slots.append({
            "slot_id": f"replacement:{pair['pair_id']}",
            "question_type": pair["stratum"]["question_type"],
            "gold_cardinality": pair["stratum"]["gold_cardinality"],
            "shared_candidate_id": None,
            "pair_id": pair["pair_id"],
        })
    targets = assign_slot_targets(slots)
    pair_by_id = {row["pair_id"]: row for row in pairs}
    clean_manifest: list[dict[str, Any]] = []
    inclusive_manifest: list[dict[str, Any]] = []
    clean_training: list[dict[str, Any]] = []
    inclusive_training: list[dict[str, Any]] = []
    for slot in slots:
        slot_id = slot["slot_id"]
        order_hash = hashlib.sha256(canonical_json({
            "version": DATASET_VERSION, "slot_id": slot_id, "kind": "train_order",
        }).encode("utf-8")).hexdigest()
        target = targets[slot_id]
        if slot["pair_id"] is None:
            clean_row = inclusive_row = clean_by_id[slot["shared_candidate_id"]]
            roles = ("shared_clean", "shared_clean")
            pair_id = None
        else:
            pair = pair_by_id[slot["pair_id"]]
            clean_row = clean_by_id[pair["clean_candidate_id"]]
            inclusive_row = risk_by_id[pair["risk_candidate_id"]]
            roles = ("clean_control", "risk_replacement")
            pair_id = pair["pair_id"]
        for policy, row, role, manifests, training_rows in (
            ("M_clean_policy", clean_row, roles[0], clean_manifest, clean_training),
            ("M_inclusive_policy", inclusive_row, roles[1], inclusive_manifest, inclusive_training),
        ):
            dialogue = dialogue_cache[str(Path(str(row["dialogue_path"])).resolve())][0]
            manifest, training = render_training_record(
                row, tokenizer=tokenizer, dialogue=dialogue, desired_target=target,
                policy=policy, slot_id=slot_id, selection_role=role,
                pair_id=pair_id, order_key=order_hash,
            )
            manifests.append(manifest)
            training_rows.append(training)
    clean_order = {row["slot_id"]: row["order_key"] for row in clean_manifest}
    inclusive_order = {row["slot_id"]: row["order_key"] for row in inclusive_manifest}
    if clean_order != inclusive_order:
        raise RuntimeError("policy_slot_order_mismatch")
    clean_manifest.sort(key=lambda row: (row["order_key"], row["slot_id"]))
    inclusive_manifest.sort(key=lambda row: (row["order_key"], row["slot_id"]))
    training_by_clean = {row["metadata"]["slot_id"]: row for row in clean_training}
    training_by_inclusive = {row["metadata"]["slot_id"]: row for row in inclusive_training}
    clean_training = [training_by_clean[row["slot_id"]] for row in clean_manifest]
    inclusive_training = [training_by_inclusive[row["slot_id"]] for row in inclusive_manifest]

    clean_distribution = _distribution(clean_manifest)
    inclusive_distribution = _distribution(inclusive_manifest)
    balance_fields = (
        "candidate_count", "question_type_counts", "gold_cardinality_counts",
        "position_mechanism_counts", "single_target_label_counts",
        "multi_target_label_occurrence_counts", "multi_target_set_counts",
        "ordering_target_label_by_rank",
    )
    balance_checks = {field: clean_distribution[field] == inclusive_distribution[field] for field in balance_fields}
    if not all(balance_checks.values()):
        raise RuntimeError(f"policy_balance_mismatch:{balance_checks}")
    clean_by_slot = {row["slot_id"]: row for row in clean_manifest}
    inclusive_by_slot = {row["slot_id"]: row for row in inclusive_manifest}
    if set(clean_by_slot) != set(inclusive_by_slot):
        raise RuntimeError("policy_slot_ids_mismatch")
    prompt_deltas = [
        _relative_delta(clean_by_slot[slot]["prompt_token_count"], inclusive_by_slot[slot]["prompt_token_count"])
        for slot in clean_by_slot
    ]
    input_deltas = [
        _relative_delta(clean_by_slot[slot]["input_token_count"], inclusive_by_slot[slot]["input_token_count"])
        for slot in clean_by_slot
    ]
    if max(prompt_deltas, default=0.0) > MAX_TOKEN_RELATIVE_DELTA + 1e-12:
        raise RuntimeError("rendered_prompt_pair_delta_above_threshold")

    train_roots = {row["root_id"] for row in clean_manifest} | {row["root_id"] for row in inclusive_manifest}
    dev_rows = sorted((row for row in difficulty.values() if row["split"] == "dev"), key=lambda row: row["candidate_id"])
    test_rows = sorted((row for row in difficulty.values() if row["split"] == "test"), key=lambda row: row["candidate_id"])
    dev_roots = {row["root_id"] for row in dev_rows}
    test_roots = {row["root_id"] for row in test_rows}
    if train_roots & dev_roots or train_roots & test_roots or dev_roots & test_roots:
        raise RuntimeError("train_dev_test_root_leakage")
    train_ids = set(clean_by_id) | set(risk_by_id)
    if train_ids & {row["candidate_id"] for row in dev_rows + test_rows}:
        raise RuntimeError("train_dev_test_candidate_leakage")
    dev_eval = [_eval_row(row, evidence.get(row["candidate_id"]), difficulty_path=difficulty_path, difficulty_sha=difficulty_hashes["difficulty_manifest.jsonl"]) for row in dev_rows]
    test_eval = [_eval_row(row, evidence.get(row["candidate_id"]), difficulty_path=difficulty_path, difficulty_sha=difficulty_hashes["difficulty_manifest.jsonl"]) for row in test_rows]
    dev_hard = [row for row in dev_eval if row["m0_hard"]]
    test_hard = [row for row in test_eval if row["m0_hard"]]

    feasibility_report = {
        "matching_feasibility_version": MATCHING_ALGORITHM_VERSION,
        "status": "PASS",
        "clean_pool_count_after_blacklist": len(train_clean),
        "risk_pool_count_after_blacklist": len(train_risk),
        "hard_constraints": ["question_type_exact", "gold_cardinality_exact", "position_mechanism_exact"],
        "max_accuracy_delta": MAX_ACCURACY_DELTA,
        "max_context_and_prompt_token_relative_delta": MAX_TOKEN_RELATIVE_DELTA,
        "rates": feasibility,
        "selected_rate": selected_rate,
        "selection_rule": "prefer_20_percent_when_all_constraints_pass_else_10_percent",
    }
    comparison = {
        "training_distribution_comparison_version": f"{DATASET_VERSION}_distribution_v1",
        "status": "PASS",
        "clean_policy": clean_distribution,
        "inclusive_policy": inclusive_distribution,
        "exact_balance_checks": balance_checks,
        "shared_candidate_count": len(shared_rows),
        "replacement_pair_count": len(pairs),
        "same_slot_target_signature_exact": all(
            clean_by_slot[slot]["displayed_target"] == inclusive_by_slot[slot]["displayed_target"]
            for slot in clean_by_slot
        ),
        "prompt_token_counts": {
            "clean": _numeric_summary([row["prompt_token_count"] for row in clean_manifest]),
            "inclusive": _numeric_summary([row["prompt_token_count"] for row in inclusive_manifest]),
            "paired_relative_delta": _numeric_summary(prompt_deltas),
        },
        "input_token_counts": {
            "clean": _numeric_summary([row["input_token_count"] for row in clean_manifest]),
            "inclusive": _numeric_summary([row["input_token_count"] for row in inclusive_manifest]),
            "paired_relative_delta": _numeric_summary(input_deltas),
        },
        "supervised_token_counts": {
            "clean": _numeric_summary([row["supervised_token_count"] for row in clean_manifest]),
            "inclusive": _numeric_summary([row["supervised_token_count"] for row in inclusive_manifest]),
        },
        "full_context_accuracy": {
            "clean_mean": statistics.fmean(row["full_context_accuracy"] for row in clean_manifest),
            "inclusive_mean": statistics.fmean(row["full_context_accuracy"] for row in inclusive_manifest),
        },
        "no_context_accuracy": {
            "clean_mean": statistics.fmean(row["no_context_accuracy"] for row in clean_manifest),
            "inclusive_mean": statistics.fmean(row["no_context_accuracy"] for row in inclusive_manifest),
        },
        "train_dev_test_leakage": {
            "candidate_overlap_count": 0,
            "root_overlap_count": 0,
            "pass": True,
        },
        "generation_source_is_intentional_confounder": True,
    }
    trainer_path = Path(__file__).resolve().parents[4] / "train_qwen3_lora_sft.py"
    if not trainer_path.is_file():
        raise RuntimeError(f"training_interface_missing:{trainer_path}")
    trainer_default_max_length = int(_argparse_default(trainer_path, "--max-length"))
    model_config_path = tokenizer_path / "config.json"
    model_config = load_json(model_config_path)
    model_max_positions = int(model_config["max_position_embeddings"])
    max_rendered_input = max(row["input_token_count"] for row in clean_manifest + inclusive_manifest)
    if REQUIRED_TRAIN_MAX_LENGTH < max_rendered_input:
        raise RuntimeError(
            f"required_train_max_length_too_small:{REQUIRED_TRAIN_MAX_LENGTH}:{max_rendered_input}"
        )
    if REQUIRED_TRAIN_MAX_LENGTH > model_max_positions:
        raise RuntimeError(
            f"required_train_max_length_above_model_context:{REQUIRED_TRAIN_MAX_LENGTH}:{model_max_positions}"
        )
    training_interface = {
        "training_interface_contract_version": f"{DATASET_VERSION}_training_interface_v1",
        "status": "FROZEN",
        "trainer_path": str(trainer_path),
        "trainer_sha256": sha256_file(trainer_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_chat_template_sha256": solver_contract["chat_template_sha256"],
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "model_max_position_embeddings": model_max_positions,
        "trainer_default_max_length": trainer_default_max_length,
        "minimum_no_truncation_max_length": max_rendered_input,
        "required_max_length_argument": REQUIRED_TRAIN_MAX_LENGTH,
        "trainer_default_requires_override": trainer_default_max_length < max_rendered_input,
        "default_truncation_counts": {
            "M_clean_policy": sum(row["input_token_count"] > trainer_default_max_length for row in clean_manifest),
            "M_inclusive_policy": sum(row["input_token_count"] > trainer_default_max_length for row in inclusive_manifest),
        },
        "left_context_truncation_forbidden": True,
        "assistant_only_supervision_required": True,
        "one_record_per_candidate_per_epoch": True,
        "required_cli_fragment": f"--max-length {REQUIRED_TRAIN_MAX_LENGTH}",
        "training_started": False,
    }
    evaluation_protocol = {
        "evaluation_protocol_version": EVAL_FREEZE_VERSION,
        "status": "FROZEN",
        "models": ["M0", "M_clean_policy", "M_inclusive_policy"],
        "common_candidate_sets_required": True,
        "dev_full_count": len(dev_eval),
        "dev_m0_hard_count": len(dev_hard),
        "test_full_count": len(test_eval),
        "test_m0_hard_count": len(test_hard),
        "b0_protocol": {
            "context_modes": ["no_context", "full_context"],
            "rollouts_per_position": 3,
            "cyclic_positions": True,
            "canonical_rebinding": True,
            "source_solver_contract_path": str(difficulty_run / "solver_contract.json"),
            "source_solver_contract_sha256": difficulty_hashes["solver_contract.json"],
        },
        "primary_metrics": [
            "full_context_accuracy", "no_context_accuracy", "full_minus_no_gap",
            "canonical_flip_rate", "worst_position_accuracy", "m0_hard_to_resolved_or_easy_count",
        ],
        "evidence_status_slicing_allowed": True,
        "policy_specific_test_sets_forbidden": True,
        "evaluation_started": False,
    }
    preflight = {
        "preflight_report_version": f"{DATASET_VERSION}_preflight_v1",
        "status": "PASS",
        "difficulty_run": str(difficulty_run),
        "stratification_run": str(stratification_run),
        "difficulty_source_sha256": difficulty_hashes,
        "stratification_source_sha256": stratification_hashes,
        "blacklist": blacklist,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_chat_template_sha256": solver_contract["chat_template_sha256"],
        "tokenizer_matches_b0_contract": True,
        "counts": {
            "difficulty": len(difficulty), "train": split_counts["train"],
            "dev": split_counts["dev"], "test": split_counts["test"],
            "train_hard": hard_split_counts["train"], "dev_hard": hard_split_counts["dev"],
            "test_hard": hard_split_counts["test"], "train_clean": len(train_clean), "train_risk": len(train_risk),
        },
        "lineage_sha_verified": True,
        "solver_calls": 0,
        "external_api_calls": 0,
        "semantic_auditor_calls": 0,
        "model_training_started": False,
    }

    major_names = (
        "preflight_report.json", "confirmed_invalid_blacklist.json", "matching_feasibility_10_20.json",
        "matched_clean_risk_pairs.jsonl", "shared_clean_core_manifest.jsonl", "clean_controls_manifest.jsonl",
        "risk_replacements_manifest.jsonl", "m_clean_policy_training_manifest.jsonl",
        "m_inclusive_policy_training_manifest.jsonl", "m_clean_policy_train.jsonl", "m_inclusive_policy_train.jsonl",
        "dev_full_eval_manifest.jsonl", "dev_m0_hard_eval_manifest.jsonl", "test_full_eval_manifest.jsonl",
        "test_m0_hard_eval_manifest.jsonl", "evaluation_protocol.json", "training_interface_contract.json",
        "training_distribution_comparison.json",
        "final_dataset_report.json", "final_dataset_report.md",
    )
    final = {
        "final_dataset_report_version": f"{DATASET_VERSION}_final_v1",
        "status": "COMPLETE",
        "dataset_version": DATASET_VERSION,
        "selected_replacement_rate": selected_rate,
        "candidate_count_per_policy": len(clean_manifest),
        "shared_clean_count": len(shared_rows),
        "replacement_count": len(pairs),
        "blacklist_entry_count": blacklist["entry_count"],
        "clean_policy_clean_count": sum(row["evidence_status"] == "EVIDENCE_CLEAN" for row in clean_manifest),
        "inclusive_policy_clean_count": sum(row["evidence_status"] == "EVIDENCE_CLEAN" for row in inclusive_manifest),
        "inclusive_policy_risk_count": sum(row["evidence_status"] == "EVIDENCE_RISK" for row in inclusive_manifest),
        "risk_replacement_root_count": len({row["root_id"] for row in inclusive_manifest if row["selection_role"] == "risk_replacement"}),
        "dev_full_count": len(dev_eval),
        "dev_m0_hard_count": len(dev_hard),
        "test_full_count": len(test_eval),
        "test_m0_hard_count": len(test_hard),
        "exact_balance_checks": balance_checks,
        "same_slot_target_signature_exact": comparison["same_slot_target_signature_exact"],
        "train_dev_test_leakage_pass": True,
        "max_prompt_token_count": max(row["prompt_token_count"] for row in clean_manifest + inclusive_manifest),
        "max_input_token_count": max(row["input_token_count"] for row in clean_manifest + inclusive_manifest),
        "required_training_max_length": REQUIRED_TRAIN_MAX_LENGTH,
        "trainer_default_requires_override": training_interface["trainer_default_requires_override"],
        "major_result_paths": [str(run_root / name) for name in major_names],
        "solver_calls": 0,
        "external_api_calls": 0,
        "semantic_auditor_calls": 0,
        "model_training_started": False,
        "matched_training_run_started": False,
        "source_modified": False,
    }

    run_root.mkdir(parents=True, exist_ok=False)
    _write_json(run_root / "preflight_report.json", preflight)
    _write_json(run_root / "confirmed_invalid_blacklist.json", blacklist)
    _write_json(run_root / "matching_feasibility_10_20.json", feasibility_report)
    _write_jsonl(run_root / "matched_clean_risk_pairs.jsonl", pairs)
    _write_jsonl(run_root / "shared_clean_core_manifest.jsonl", [row for row in clean_manifest if row["selection_role"] == "shared_clean"])
    _write_jsonl(run_root / "clean_controls_manifest.jsonl", [row for row in clean_manifest if row["selection_role"] == "clean_control"])
    _write_jsonl(run_root / "risk_replacements_manifest.jsonl", [row for row in inclusive_manifest if row["selection_role"] == "risk_replacement"])
    _write_jsonl(run_root / "m_clean_policy_training_manifest.jsonl", clean_manifest)
    _write_jsonl(run_root / "m_inclusive_policy_training_manifest.jsonl", inclusive_manifest)
    _write_jsonl(run_root / "m_clean_policy_train.jsonl", clean_training)
    _write_jsonl(run_root / "m_inclusive_policy_train.jsonl", inclusive_training)
    _write_jsonl(run_root / "dev_full_eval_manifest.jsonl", dev_eval)
    _write_jsonl(run_root / "dev_m0_hard_eval_manifest.jsonl", dev_hard)
    _write_jsonl(run_root / "test_full_eval_manifest.jsonl", test_eval)
    _write_jsonl(run_root / "test_m0_hard_eval_manifest.jsonl", test_hard)
    _write_json(run_root / "evaluation_protocol.json", evaluation_protocol)
    _write_json(run_root / "training_interface_contract.json", training_interface)
    _write_json(run_root / "training_distribution_comparison.json", comparison)
    _write_json(run_root / "final_dataset_report.json", final)
    _write_text(run_root / "final_dataset_report.md", _markdown(final))
    return final


__all__ = [
    "assign_slot_targets", "build", "build_display", "match_at_rate",
    "pair_is_eligible", "render_training_record", "target_signatures",
]
