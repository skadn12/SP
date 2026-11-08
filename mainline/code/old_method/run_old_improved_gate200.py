from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_R3_SEED_DATA = "eval_sets/graph_expansion_challenger_v1/r3_auto_full/samples.jsonl"
DEFAULT_R3_SOLVER = "outputs/qwen3_4b_lora_r3_frontier_plus_clean_replay_from_r2_ctx15360_full"
DEFAULT_OUTPUT_DIR = "outputs/old_improved_gate200"
DEFAULT_MODEL = "/root/autodl-tmp/Qwen3-4B"
SOLVER_STOP_SEQUENCES = ("\n\n", "\n", "</answer>")
STRUCTURED_MAX_NEW_TOKENS = {
    "single_choice": 8,
    "multi_select": 16,
    "ordering": 24,
    "evidence_selection": 24,
    "invalidated_statement_detection": 32,
    "state_reconstruction": 64,
}
FINAL_ANSWER_INSTRUCTIONS = {
    "single_choice": "Answer only in the required format. Do not explain.\nAnswer only with one option letter, e.g. B.",
    "multi_select": "Answer only in the required format. Do not explain.\nAnswer only with option letters, e.g. A,C.",
    "ordering": "Answer only in the required format. Do not explain.\nAnswer only with option letters separated by >, e.g. A>B>D>C.",
    "evidence_selection": "Answer only in the required format. Do not explain.\nAnswer with exactly one turn ID, e.g. D19:1.",
    "invalidated_statement_detection": (
        "Answer only in the required format. Do not explain.\n"
        "Answer only with stale_statement_id | invalidation_event_id | final_true_state_id."
    ),
    "state_reconstruction": "Answer only in the required format. Do not explain.\nAnswer only with the exact phrase.",
}

ATTACK_SKILL_QUOTAS = [
    ("belief_vs_reality", 25),
    ("who_knows_what", 25),
    ("permission_visibility_scope", 20),
    ("delayed_activation", 20),
    ("expired_memory", 20),
    ("cancel_restore", 20),
    ("conflicting_reports", 20),
    ("conditional_plan", 20),
    ("multi_object_dependency", 20),
    ("timeline_fork_merge", 10),
]

QUESTION_TYPE_QUOTAS = [
    ("multi_select", 45),
    ("ordering", 45),
    ("evidence_selection", 45),
    ("invalidated_statement_detection", 45),
    ("state_reconstruction", 20),
]

QUESTION_TYPES = [item[0] for item in QUESTION_TYPE_QUOTAS]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VALIDATION_KEYS = [
    "graph_dialogue_consistency",
    "gold_derivable_from_graph",
    "options_are_valid",
    "no_ambiguous_pronoun",
    "no_unstated_assumption",
]
SHORTCUT_KEYS = [
    "gold_text_exposure",
    "gold_label_leakage",
    "keyword_overlap_shortcut",
    "option_overlap_shortcut",
    "answer_near_question_shortcut",
    "last_session_shortcut_risk",
    "evidence_window_position_leakage",
    "metadata_leakage",
    "distractor_too_easy",
]

TURN_RE = re.compile(r"^\[(?P<turn_id>D\d+:\d+)\]\s*(?P<speaker>[^:\n]{1,80}):\s*(?P<text>.*)$", re.MULTILINE)


ATTACK_LIBRARY: dict[str, dict[str, Any]] = {
    "belief_vs_reality": {
        "reasoning_mechanism": "separate stale belief from final observable state",
        "target_solver_weakness": "answers from an early remembered statement instead of the later correction",
        "dependency_depth": 3,
        "requires_invalidated_memory": True,
        "shortcut_to_avoid": ["answer_from_first_mention", "stale_belief_keyword"],
    },
    "who_knows_what": {
        "reasoning_mechanism": "track which speaker observed or repeated each event",
        "target_solver_weakness": "collapses speaker visibility into global memory",
        "dependency_depth": 2,
        "requires_cross_session_tracking": True,
        "requires_visibility_reasoning": True,
        "shortcut_to_avoid": ["speaker_name_overlap", "last_speaker_bias"],
    },
    "permission_visibility_scope": {
        "reasoning_mechanism": "apply scope limits before selecting visible evidence",
        "target_solver_weakness": "treats all dialogue facts as equally visible",
        "dependency_depth": 3,
        "requires_visibility_reasoning": True,
        "shortcut_to_avoid": ["global_visibility_assumption", "permission_word_overlap"],
    },
    "delayed_activation": {
        "reasoning_mechanism": "distinguish announced plan from the later activation turn",
        "target_solver_weakness": "selects the plan announcement as the final state",
        "dependency_depth": 3,
        "requires_ordering": True,
        "shortcut_to_avoid": ["future_tense_shortcut", "first_plan_shortcut"],
    },
    "expired_memory": {
        "reasoning_mechanism": "discard an older remembered state after expiry or replacement",
        "target_solver_weakness": "keeps expired memory when options repeat familiar words",
        "dependency_depth": 3,
        "requires_invalidated_memory": True,
        "requires_ordering": True,
        "shortcut_to_avoid": ["old_memory_keyword", "recency_only_without_invalidation"],
    },
    "cancel_restore": {
        "reasoning_mechanism": "track cancel and restore edges before reading the final active state",
        "target_solver_weakness": "misses restoration after a cancellation",
        "dependency_depth": 4,
        "requires_invalidated_memory": True,
        "requires_ordering": True,
        "shortcut_to_avoid": ["cancel_word_only", "last_negative_event"],
    },
    "conflicting_reports": {
        "reasoning_mechanism": "resolve two incompatible reports using source and turn order",
        "target_solver_weakness": "copies the more salient report instead of the supported one",
        "dependency_depth": 3,
        "requires_cross_session_tracking": True,
        "shortcut_to_avoid": ["longer_option_bias", "shared_keyword_bias"],
    },
    "conditional_plan": {
        "reasoning_mechanism": "apply a condition before following the selected branch",
        "target_solver_weakness": "follows both branches or ignores the condition",
        "dependency_depth": 4,
        "requires_ordering": True,
        "shortcut_to_avoid": ["branch_label_shortcut", "if_word_overlap"],
    },
    "multi_object_dependency": {
        "reasoning_mechanism": "bind multiple objects or states to the same target entity",
        "target_solver_weakness": "mixes a distractor object with the target state",
        "dependency_depth": 3,
        "requires_cross_session_tracking": True,
        "shortcut_to_avoid": ["single_object_shortcut", "nearest_object_bias"],
    },
    "timeline_fork_merge": {
        "reasoning_mechanism": "follow two timeline branches and merge at the final supported node",
        "target_solver_weakness": "uses one branch without checking the merge event",
        "dependency_depth": 4,
        "requires_ordering": True,
        "shortcut_to_avoid": ["last_session_shortcut", "branch_order_shortcut"],
    },
}


@dataclass
class SourceBundle:
    row: dict[str, Any]
    turns: list[dict[str, Any]]
    turn_map: dict[str, dict[str, Any]]
    nodes: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and gate an old-method improved LoCoMo graph/evidence-window frontier without solver training."
    )
    parser.add_argument("--solver_checkpoint", "--solver-checkpoint", default=DEFAULT_R3_SOLVER)
    parser.add_argument("--seed_data", "--seed-data", default=DEFAULT_R3_SEED_DATA)
    parser.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", "--num-samples", type=int, default=200)
    parser.add_argument("--k_eval", "--k-eval", type=int, default=5)
    parser.add_argument("--max_largest_question_type_share", "--max-largest-question-type-share", type=float, default=0.35)
    parser.add_argument("--max_largest_attack_skill_share", "--max-largest-attack-skill-share", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--eval_backend", "--eval-backend", choices=("auto", "vllm", "transformers", "heuristic"), default="auto")
    parser.add_argument("--base_model", "--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--max_model_len", "--max-model-len", type=int, default=16384)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", "--top-p", type=float, default=0.9)
    parser.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--limit_seed_rows", "--limit-seed-rows", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def load_seed_rows(path: str | Path) -> list[dict[str, Any]]:
    seed_path = Path(path)
    if not seed_path.exists():
        raise FileNotFoundError(f"seed_data not found: {seed_path}")
    if seed_path.suffix == ".jsonl":
        return load_jsonl(seed_path)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("samples", "results", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"Unsupported seed_data format: {seed_path}")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: Any) -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff:/._\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def count_map(values: list[Any]) -> dict[str, int]:
    counts = Counter(str(value if value is not None else "unknown") for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def first_existing_path(*paths: str) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    return paths[0]


def resolve_solver_checkpoint(path: str | Path) -> tuple[str, str | None]:
    checkpoint = Path(path)
    if (checkpoint / "model.safetensors.index.json").exists() or (checkpoint / "config.json").exists():
        return str(checkpoint), None
    merged = Path(str(checkpoint) + "_merged")
    if merged.exists():
        return str(merged), str(checkpoint)
    return str(checkpoint), None


def expand_plan(plan: list[tuple[str, int]], total: int) -> list[str]:
    labels: list[str] = []
    plan_total = sum(count for _, count in plan)
    if total == plan_total:
        for label, count in plan:
            labels.extend([label] * count)
        return labels
    remaining = total
    allocations: list[tuple[str, int, float]] = []
    for label, count in plan:
        raw = total * count / max(1, plan_total)
        base = int(math.floor(raw))
        allocations.append((label, base, raw - base))
        remaining -= base
    allocations.sort(key=lambda item: item[2], reverse=True)
    extra = {label: 0 for label, _, _ in allocations}
    for label, _, _ in allocations[:remaining]:
        extra[label] += 1
    labels = []
    original_order = [label for label, _ in plan]
    base_map = {label: base for label, base, _ in allocations}
    for label in original_order:
        labels.extend([label] * (base_map[label] + extra[label]))
    return labels[:total]


def parse_dialogue_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    final_dialogue = row.get("final_dialogue")
    if isinstance(final_dialogue, list):
        for idx, turn in enumerate(final_dialogue):
            if not isinstance(turn, dict):
                continue
            turn_id = str(turn.get("dia_id") or turn.get("turn_id") or turn.get("turn_ref") or f"T{idx + 1}")
            turns.append(
                {
                    "turn_id": turn_id,
                    "speaker": normalize_space(turn.get("speaker") or "Speaker"),
                    "text": normalize_space(turn.get("text")),
                    "source": "final_dialogue",
                    "index": len(turns),
                }
            )
        if turns:
            return turns
    text = str(row.get("expanded_dialogue_text") or row.get("new_dialogue_text") or "")
    for match in TURN_RE.finditer(text):
        turns.append(
            {
                "turn_id": match.group("turn_id"),
                "speaker": normalize_space(match.group("speaker")),
                "text": normalize_space(match.group("text")),
                "source": "expanded_dialogue_text",
                "index": len(turns),
            }
        )
    return turns


def turn_sort_value(turn_id: Any) -> tuple[int, int]:
    match = re.match(r"^D(\d+):(\d+)$", str(turn_id or ""))
    if not match:
        return 999999, 999999
    return int(match.group(1)), int(match.group(2))


def node_sort_value(node: dict[str, Any]) -> tuple[int, int, str]:
    turn_ids = node_turn_ids(node)
    first = min([turn_sort_value(turn_id) for turn_id in turn_ids] or [(999999, 999999)])
    return first[0], first[1], str(node.get("role") or node.get("node_id") or "")


def node_turn_ids(node: dict[str, Any]) -> list[str]:
    value = node.get("turn_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    value = node.get("turn_id")
    if value:
        return [str(value)]
    return []


def collect_nodes(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_nodes: list[Any] = []
    for key in ("state_chain_nodes", "new_nodes"):
        value = row.get(key)
        if isinstance(value, list):
            raw_nodes.extend(value)
    state_chain = row.get("state_chain")
    if isinstance(state_chain, dict) and isinstance(state_chain.get("nodes"), list):
        raw_nodes.extend(state_chain["nodes"])
    seen: set[str] = set()
    nodes: list[dict[str, Any]] = []
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        value = normalize_space(raw.get("value"))
        if len(value) < 8 or value.endswith("?"):
            continue
        turn_ids = node_turn_ids(raw)
        if not turn_ids:
            continue
        key = normalize_text(value)
        if not key or key in seen:
            continue
        cloned = copy.deepcopy(raw)
        cloned["value"] = value
        cloned["turn_ids"] = turn_ids
        cloned.setdefault("role", raw.get("label") or raw.get("node_id") or f"node_{len(nodes) + 1}")
        nodes.append(cloned)
        seen.add(key)
    return sorted(nodes, key=node_sort_value)


def build_source_bundle(row: dict[str, Any]) -> SourceBundle | None:
    turns = parse_dialogue_turns(row)
    nodes = collect_nodes(row)
    if len(turns) < 6 or len(nodes) < 5:
        return None
    turn_map = {turn["turn_id"]: turn for turn in turns}
    return SourceBundle(row=row, turns=turns, turn_map=turn_map, nodes=nodes)


def seed_identity(row: dict[str, Any]) -> dict[str, Any]:
    pairing = row.get("parent_pairing_key") if isinstance(row.get("parent_pairing_key"), dict) else {}
    sample_id = str(row.get("sample_id") or row.get("record_id") or pairing.get("sample_id") or "")
    seed_qa_id = str(row.get("parent_qa_index") or row.get("qa_index") or pairing.get("qa_index") or row.get("question") or "")
    turn_ids = []
    for node in collect_nodes(row)[:6]:
        turn_ids.extend(node_turn_ids(node))
    return {
        "locomo_conversation_id": str(pairing.get("sample_id") or row.get("state_entity") or sample_id.split(":")[0] or sample_id),
        "seed_qa_id": seed_qa_id,
        "evidence_window_turn_ids": sorted(set(turn_ids), key=turn_sort_value)[:12],
        "seed_summary": normalize_space(row.get("question") or f"source {sample_id}"),
    }


def sample_nodes(bundle: SourceBundle, rng: random.Random, count: int, prefer_late: bool = True) -> list[dict[str, Any]]:
    nodes = bundle.nodes
    if prefer_late and len(nodes) > count:
        start = max(0, len(nodes) // 3)
        pool = nodes[start:]
    else:
        pool = nodes
    if len(pool) >= count:
        return rng.sample(pool, count)
    out = list(pool)
    while len(out) < count:
        out.append(rng.choice(nodes))
    return out


def node_label(node: dict[str, Any]) -> str:
    role = normalize_space(node.get("role") or node.get("label") or node.get("node_id"))
    return role.replace("_", " ")


def short_quote(value: Any, limit: int = 120) -> str:
    text = normalize_space(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def option_text_from_node(node: dict[str, Any]) -> str:
    return short_quote(node.get("value"), 140)


def turn_window(bundle: SourceBundle, evidence_turn_ids: list[str], rng: random.Random, radius: int = 2, max_turns: int = 18) -> list[dict[str, Any]]:
    if not evidence_turn_ids:
        evidence_turn_ids = [node_turn_ids(node)[0] for node in bundle.nodes[:2] if node_turn_ids(node)]
    indices = [bundle.turn_map[turn_id]["index"] for turn_id in evidence_turn_ids if turn_id in bundle.turn_map]
    selected_indices: set[int] = set()
    for index in indices:
        for offset in range(-radius, radius + 1):
            if 0 <= index + offset < len(bundle.turns):
                selected_indices.add(index + offset)
    if len(selected_indices) < min(max_turns, 10):
        candidates = list(range(len(bundle.turns)))
        rng.shuffle(candidates)
        for index in candidates:
            selected_indices.add(index)
            if len(selected_indices) >= min(max_turns, 12):
                break
    selected = [copy.deepcopy(bundle.turns[index]) for index in sorted(selected_indices)]
    if len(selected) > max_turns:
        evidence_set = set(evidence_turn_ids)
        keep = [turn for turn in selected if turn["turn_id"] in evidence_set]
        rest = [turn for turn in selected if turn["turn_id"] not in evidence_set]
        rng.shuffle(rest)
        selected = sorted((keep + rest[: max_turns - len(keep)]), key=lambda turn: turn_sort_value(turn["turn_id"]))
    return selected


def attack_plan(skill: str) -> dict[str, Any]:
    entry = ATTACK_LIBRARY[skill]
    return {
        "attack_skill": skill,
        "reasoning_mechanism": entry["reasoning_mechanism"],
        "target_solver_weakness": entry["target_solver_weakness"],
        "dependency_depth": int(entry.get("dependency_depth", 0)),
        "requires_cross_session_tracking": bool(entry.get("requires_cross_session_tracking", False)),
        "requires_invalidated_memory": bool(entry.get("requires_invalidated_memory", False)),
        "requires_visibility_reasoning": bool(entry.get("requires_visibility_reasoning", False)),
        "requires_ordering": bool(entry.get("requires_ordering", False)),
        "shortcut_to_avoid": list(entry.get("shortcut_to_avoid", [])),
    }


def speakers_from_turns(turns: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    speakers: list[str] = []
    for turn in turns:
        speaker = normalize_space(turn.get("speaker"))
        if speaker and speaker not in seen:
            speakers.append(speaker)
            seen.add(speaker)
    return speakers


def make_graph_update(
    *,
    bundle: SourceBundle,
    nodes: list[dict[str, Any]],
    evidence_turn_ids: list[str],
    skill: str,
    stale_node: dict[str, Any] | None = None,
    final_node: dict[str, Any] | None = None,
) -> dict[str, Any]:
    speakers = speakers_from_turns(bundle.turns)
    entity = normalize_space(bundle.row.get("state_entity") or bundle.row.get("sample_id") or "conversation")
    events = []
    for idx, node in enumerate(nodes):
        events.append(
            {
                "event_id": f"E{idx + 1}",
                "node_id": node.get("node_id"),
                "role": node.get("role"),
                "value": node.get("value"),
                "turn_ids": node_turn_ids(node),
                "status": node.get("status") or "state_event",
            }
        )
    state_changes = []
    if nodes:
        for previous, current in zip(nodes[:-1], nodes[1:]):
            state_changes.append(
                {
                    "entity": entity,
                    "slot": bundle.row.get("state_slot") or "memory_state",
                    "from": previous.get("value"),
                    "to": current.get("value"),
                    "evidence_turn_ids": node_turn_ids(current),
                }
            )
    invalidations = []
    if stale_node and final_node:
        invalidations.append(
            {
                "invalidated_statement": stale_node.get("value"),
                "invalidating_event": final_node.get("value"),
                "invalidation_turn_ids": node_turn_ids(final_node),
                "final_true_state": final_node.get("value"),
            }
        )
    visibility_edges = []
    if skill in {"who_knows_what", "permission_visibility_scope", "conflicting_reports"}:
        for turn_id in evidence_turn_ids[:4]:
            turn = bundle.turn_map.get(turn_id, {})
            visibility_edges.append(
                {
                    "observer": turn.get("speaker") or (speakers[0] if speakers else "speaker"),
                    "event_turn_id": turn_id,
                    "visibility": "directly_stated_in_dialogue",
                }
            )
    delayed_activations = []
    if skill in {"delayed_activation", "conditional_plan", "timeline_fork_merge"} and len(nodes) >= 2:
        delayed_activations.append(
            {
                "planned_state": nodes[0].get("value"),
                "activation_state": nodes[-1].get("value"),
                "activation_turn_ids": node_turn_ids(nodes[-1]),
            }
        )
    cancellations = []
    restorations = []
    if skill == "cancel_restore" and len(nodes) >= 3:
        cancellations.append({"cancelled_state": nodes[0].get("value"), "turn_ids": node_turn_ids(nodes[1])})
        restorations.append({"restored_state": nodes[-1].get("value"), "turn_ids": node_turn_ids(nodes[-1])})
    return {
        "entities": [{"entity_id": entity, "type": "conversation_memory"}]
        + [{"entity_id": speaker, "type": "speaker"} for speaker in speakers[:4]],
        "events": events,
        "state_changes": state_changes,
        "visibility_edges": visibility_edges,
        "invalidations": invalidations,
        "delayed_activations": delayed_activations,
        "cancellations": cancellations,
        "restorations": restorations,
    }


def build_multi_select(bundle: SourceBundle, rng: random.Random, skill: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    repeated = [node for node in bundle.nodes if len(node_turn_ids(node)) >= 2]
    if len(repeated) < 2:
        repeated = bundle.nodes[-6:]
    gold_nodes = rng.sample(repeated, min(2, len(repeated)))
    distractors = [node for node in bundle.nodes if normalize_text(node.get("value")) not in {normalize_text(g.get("value")) for g in gold_nodes}]
    distractors = rng.sample(distractors, min(3, len(distractors)))
    option_nodes = gold_nodes + distractors
    rng.shuffle(option_nodes)
    options = []
    gold_labels = []
    for idx, node in enumerate(option_nodes):
        label = LETTERS[idx]
        if node in gold_nodes:
            gold_labels.append(label)
        options.append({"id": label, "text": option_text_from_node(node), "turn_ids": node_turn_ids(node)})
    evidence_turn_ids = sorted({turn_id for node in gold_nodes for turn_id in node_turn_ids(node)}, key=turn_sort_value)
    question = (
        "Select exactly the two options whose state texts are explicitly repeated in the later review block "
        "and therefore remain active graph states. Answer with option letters only."
    )
    qa = {
        "question_type": "multi_select",
        "question": question,
        "options": options,
        "gold": sorted(gold_labels),
        "gold_source": "graph_derived",
        "evidence_turn_ids": evidence_turn_ids,
    }
    graph_update = make_graph_update(bundle=bundle, nodes=option_nodes, evidence_turn_ids=evidence_turn_ids, skill=skill)
    return qa, graph_update, turn_window(bundle, evidence_turn_ids, rng), evidence_turn_ids


def build_ordering(bundle: SourceBundle, rng: random.Random, skill: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    nodes = sample_nodes(bundle, rng, 4, prefer_late=False)
    nodes = sorted(nodes, key=node_sort_value)
    option_nodes = list(nodes)
    rng.shuffle(option_nodes)
    options = []
    id_by_key: dict[str, str] = {}
    for idx, node in enumerate(option_nodes):
        label = LETTERS[idx]
        id_by_key[normalize_text(node.get("value"))] = label
        options.append({"id": label, "text": option_text_from_node(node), "turn_ids": node_turn_ids(node)})
    gold = [id_by_key[normalize_text(node.get("value"))] for node in nodes]
    evidence_turn_ids = sorted({node_turn_ids(node)[0] for node in nodes if node_turn_ids(node)}, key=turn_sort_value)
    question = (
        "Order the option letters by the first dialogue turn that supports each event, from earliest to latest. "
        "Answer as letters separated by '>'."
    )
    qa = {
        "question_type": "ordering",
        "question": question,
        "options": options,
        "gold": gold,
        "gold_source": "graph_derived",
        "evidence_turn_ids": evidence_turn_ids,
    }
    graph_update = make_graph_update(bundle=bundle, nodes=nodes, evidence_turn_ids=evidence_turn_ids, skill=skill)
    return qa, graph_update, turn_window(bundle, evidence_turn_ids, rng, radius=1), evidence_turn_ids


def build_evidence_selection(bundle: SourceBundle, rng: random.Random, skill: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    target = rng.choice(bundle.nodes[len(bundle.nodes) // 3 :])
    gold_turns = node_turn_ids(target)[:2]
    if not gold_turns:
        gold_turns = [node_turn_ids(target)[0]]
    distractor_turns = [turn["turn_id"] for turn in bundle.turns if turn["turn_id"] not in set(gold_turns)]
    rng.shuffle(distractor_turns)
    option_turn_ids = list(gold_turns) + distractor_turns[: max(2, 5 - len(gold_turns))]
    rng.shuffle(option_turn_ids)
    options = []
    for turn_id in option_turn_ids:
        turn = bundle.turn_map.get(turn_id, {"speaker": "", "text": ""})
        options.append({"id": turn_id, "text": f"{turn_id} {turn.get('speaker')}: {short_quote(turn.get('text'), 120)}"})
    question = (
        "Which listed turn IDs directly support the target graph state below? "
        f"Target state: {short_quote(target.get('value'), 110)}. Answer with turn IDs only."
    )
    qa = {
        "question_type": "evidence_selection",
        "question": question,
        "options": options,
        "gold": sorted(gold_turns, key=turn_sort_value),
        "gold_source": "graph_derived",
        "evidence_turn_ids": sorted(gold_turns, key=turn_sort_value),
    }
    graph_update = make_graph_update(bundle=bundle, nodes=[target], evidence_turn_ids=gold_turns, skill=skill)
    return qa, graph_update, turn_window(bundle, option_turn_ids, rng, radius=1), gold_turns


def build_invalidated(bundle: SourceBundle, rng: random.Random, skill: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    ordered = sorted(sample_nodes(bundle, rng, 5, prefer_late=False), key=node_sort_value)
    stale_node = ordered[0]
    final_node = ordered[-1]
    distractors = ordered[1:-1]
    statement_nodes = [stale_node, final_node] + distractors
    rng.shuffle(statement_nodes)
    statement_id_by_value: dict[str, str] = {}
    options: list[dict[str, Any]] = []
    for idx, node in enumerate(statement_nodes):
        label = f"S{idx + 1}"
        statement_id_by_value[normalize_text(node.get("value"))] = label
        options.append({"id": label, "type": "statement", "text": option_text_from_node(node), "turn_ids": node_turn_ids(node)})
    final_event_id = "E_final"
    options.append(
        {
            "id": final_event_id,
            "type": "invalidation_event",
            "text": f"Later evidence turn(s): {', '.join(node_turn_ids(final_node)[:3])}",
            "turn_ids": node_turn_ids(final_node),
        }
    )
    stale_label = statement_id_by_value[normalize_text(stale_node.get("value"))]
    final_label = statement_id_by_value[normalize_text(final_node.get("value"))]
    evidence_turn_ids = sorted(set(node_turn_ids(stale_node) + node_turn_ids(final_node)), key=turn_sort_value)
    question = (
        "Identify the stale statement, the later invalidation event, and the final true state. "
        "Answer in the format stale_statement_id | invalidation_event_id | final_true_state_id."
    )
    qa = {
        "question_type": "invalidated_statement_detection",
        "question": question,
        "options": options,
        "gold": [stale_label, final_event_id, final_label],
        "gold_source": "graph_derived",
        "evidence_turn_ids": evidence_turn_ids,
    }
    graph_update = make_graph_update(
        bundle=bundle,
        nodes=ordered,
        evidence_turn_ids=evidence_turn_ids,
        skill=skill,
        stale_node=stale_node,
        final_node=final_node,
    )
    return qa, graph_update, turn_window(bundle, evidence_turn_ids, rng, radius=2), evidence_turn_ids


def build_state_reconstruction(bundle: SourceBundle, rng: random.Random, skill: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    target = rng.choice(bundle.nodes[len(bundle.nodes) // 2 :])
    evidence_turn_ids = node_turn_ids(target)
    question = (
        f"Reconstruct the exact final graph memory state for {bundle.row.get('state_entity') or bundle.row.get('sample_id')}. "
        f"Target role: {node_label(target)}. Answer with only the exact state text."
    )
    qa = {
        "question_type": "state_reconstruction",
        "question": question,
        "options": [],
        "gold": [normalize_space(target.get("value"))],
        "gold_source": "graph_derived",
        "evidence_turn_ids": sorted(evidence_turn_ids, key=turn_sort_value),
    }
    graph_update = make_graph_update(bundle=bundle, nodes=[target], evidence_turn_ids=evidence_turn_ids, skill=skill)
    return qa, graph_update, turn_window(bundle, evidence_turn_ids, rng, radius=2), evidence_turn_ids


def build_qa_and_graph(bundle: SourceBundle, qtype: str, skill: str, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    builders = {
        "multi_select": build_multi_select,
        "ordering": build_ordering,
        "evidence_selection": build_evidence_selection,
        "invalidated_statement_detection": build_invalidated,
        "state_reconstruction": build_state_reconstruction,
    }
    return builders[qtype](bundle, rng, skill)


def make_dialogue_payload(turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "turns": [
            {
                "turn_id": turn.get("turn_id"),
                "speaker": turn.get("speaker"),
                "text": turn.get("text"),
            }
            for turn in turns
        ]
    }


def option_ids(options: list[dict[str, Any]]) -> set[str]:
    return {str(option.get("id")) for option in options if isinstance(option, dict) and option.get("id") is not None}


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    qa = sample["qa"]
    graph_update = sample["graph_update"]
    turns = sample["dialogue"]["turns"]
    turn_ids = {str(turn.get("turn_id")) for turn in turns}
    evidence_ids = {str(turn_id) for turn_id in qa.get("evidence_turn_ids", [])}
    option_id_set = option_ids(qa.get("options") if isinstance(qa.get("options"), list) else [])
    gold = [str(item) for item in qa.get("gold", [])]
    qtype = qa.get("question_type")
    question_norm = normalize_text(qa.get("question"))
    question_stem_norm = normalize_text(re.split(r"\btarget state\s*:", str(qa.get("question") or ""), flags=re.IGNORECASE)[0])
    all_dialogue_text = "\n".join(str(turn.get("text") or "") for turn in turns)

    validation_checks = {
        "graph_dialogue_consistency": bool(evidence_ids) and evidence_ids.issubset(turn_ids),
        "gold_derivable_from_graph": bool(graph_update.get("events") or graph_update.get("state_changes")),
        "options_are_valid": True,
        "no_ambiguous_pronoun": not bool(re.search(r"\b(it|they|them|this)\b", question_stem_norm)),
        "no_unstated_assumption": bool(evidence_ids) and bool(turns),
    }
    if qtype in {"multi_select", "ordering"}:
        validation_checks["options_are_valid"] = len(option_id_set) >= 4 and set(gold).issubset(option_id_set)
    elif qtype == "evidence_selection":
        validation_checks["options_are_valid"] = len(option_id_set) >= 3 and set(gold).issubset(option_id_set)
    elif qtype == "invalidated_statement_detection":
        validation_checks["options_are_valid"] = len(option_id_set) >= 4 and all(item in option_id_set for item in gold)
    elif qtype == "state_reconstruction":
        validation_checks["options_are_valid"] = bool(gold[0] if gold else "")

    if not validation_checks["graph_dialogue_consistency"]:
        warnings.append("graph_dialogue_consistency_failed")
    if not validation_checks["gold_derivable_from_graph"]:
        warnings.append("gold_derivable_from_graph_failed")
    if not validation_checks["options_are_valid"]:
        warnings.append("options_are_valid_failed")
    if not validation_checks["no_ambiguous_pronoun"]:
        warnings.append("ambiguous_pronoun_in_question")

    shortcut_checks = {key: False for key in SHORTCUT_KEYS}
    for gold_item in gold:
        if len(gold_item) >= 12 and normalize_text(gold_item) and normalize_text(gold_item) in question_norm:
            shortcut_checks["gold_text_exposure"] = True
    if re.search(r"\b(gold|correct\s+answer|answer\s+key)\b", question_norm):
        shortcut_checks["gold_label_leakage"] = True
    if re.search(r"\b(gold|correct answer|answer key|label)\b", json.dumps(qa.get("options", []), ensure_ascii=False).lower()):
        shortcut_checks["metadata_leakage"] = True

    if qtype in {"multi_select", "ordering", "invalidated_statement_detection"}:
        gold_option_texts = [
            str(option.get("text") or "")
            for option in qa.get("options", [])
            if isinstance(option, dict) and str(option.get("id")) in set(gold)
        ]
        distractor_texts = [
            str(option.get("text") or "")
            for option in qa.get("options", [])
            if isinstance(option, dict) and str(option.get("id")) not in set(gold)
        ]
        question_tokens = token_set(question_norm)
        if gold_option_texts and question_tokens:
            overlaps = [len(token_set(text) & question_tokens) / max(1, len(token_set(text))) for text in gold_option_texts]
            if overlaps and max(overlaps) >= 0.75:
                shortcut_checks["keyword_overlap_shortcut"] = True
        if gold_option_texts and distractor_texts:
            gold_tokens = set().union(*(token_set(text) for text in gold_option_texts))
            max_distractor_overlap = max(
                (len(gold_tokens & token_set(text)) / max(1, len(gold_tokens)) for text in distractor_texts),
                default=0.0,
            )
            if max_distractor_overlap < 0.05:
                shortcut_checks["distractor_too_easy"] = True

    if qtype == "state_reconstruction" and gold:
        gold_norm = normalize_text(gold[0])
        if gold_norm and gold_norm in question_norm:
            shortcut_checks["answer_near_question_shortcut"] = True
        if gold_norm and gold_norm not in normalize_text(all_dialogue_text):
            warnings.append("state_gold_not_visible_in_dialogue_window")
            validation_checks["gold_derivable_from_graph"] = False

    if evidence_ids:
        ordered_turns = sorted(turn_ids, key=turn_sort_value)
        if ordered_turns:
            last_two = set(ordered_turns[-2:])
            first_two = set(ordered_turns[:2])
            if evidence_ids and evidence_ids.issubset(last_two):
                shortcut_checks["last_session_shortcut_risk"] = True
            if evidence_ids.issubset(first_two) or evidence_ids.issubset(last_two):
                shortcut_checks["evidence_window_position_leakage"] = True

    anti_shortcut_failures = [key for key, value in shortcut_checks.items() if value]
    for key in anti_shortcut_failures:
        warnings.append(key)
    anti_shortcut_pass = not anti_shortcut_failures
    return {
        "graph_dialogue_consistent": validation_checks["graph_dialogue_consistency"],
        "gold_derivable": validation_checks["gold_derivable_from_graph"],
        "option_valid": validation_checks["options_are_valid"],
        "anti_shortcut_pass": anti_shortcut_pass,
        "warnings": warnings,
        "details": {
            "validation_checks": validation_checks,
            "anti_shortcut_checks": shortcut_checks,
        },
    }


def build_sample(bundle: SourceBundle, index: int, qtype: str, skill: str, rng: random.Random) -> dict[str, Any]:
    qa, graph_update, turns, evidence_turn_ids = build_qa_and_graph(bundle, qtype, skill, rng)
    source_id = str(bundle.row.get("sample_id") or bundle.row.get("record_id") or index)
    sample_id = f"old_improved_gate200_{index:04d}_{stable_id(source_id + qtype + skill + str(index), 10)}"
    sample = {
        "sample_id": sample_id,
        "seed": seed_identity(bundle.row),
        "attack_plan": attack_plan(skill),
        "graph_update": graph_update,
        "dialogue": make_dialogue_payload(turns),
        "qa": qa,
        "validation": {
            "graph_dialogue_consistent": False,
            "gold_derivable": False,
            "option_valid": False,
            "anti_shortcut_pass": False,
            "warnings": [],
        },
        "solver_eval": {
            "k": 5,
            "predictions": [],
            "exact_correct_count": 0,
            "soft_accuracy": 0.0,
            "partial_credit": 0.0,
            "bucket": "0/5",
        },
        "frontier_label": {
            "is_frontier": False,
            "category": "",
            "reason": "",
        },
    }
    sample["seed"]["evidence_window_turn_ids"] = sorted(set(evidence_turn_ids), key=turn_sort_value)
    sample["validation"] = validate_sample(sample)
    return sample


def render_dialogue(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{turn.get('turn_id')}] {turn.get('speaker')}: {turn.get('text')}" for turn in turns)


def render_options(options: list[dict[str, Any]]) -> str:
    lines = []
    for option in options:
        if not isinstance(option, dict):
            continue
        label = option.get("id")
        text = option.get("text")
        lines.append(f"{label}. {text}")
    return "\n".join(lines)


def solver_user_prompt(sample: dict[str, Any]) -> str:
    qa = sample["qa"]
    qtype = str(qa.get("question_type"))
    dialogue = render_dialogue(sample["dialogue"]["turns"])
    options = render_options(qa.get("options") if isinstance(qa.get("options"), list) else [])
    format_hint = {
        "single_choice": "Return only the selected option letter.",
        "multi_select": "Return only the selected option letters, separated by commas.",
        "ordering": "Return only the option letters in order, separated by >.",
        "evidence_selection": "Return only the single supporting turn ID (exactly one).",
        "invalidated_statement_detection": "Return only: stale_statement_id | invalidation_event_id | final_true_state_id.",
        "state_reconstruction": "Return only the exact state text.",
    }[qtype]
    parts = [
        "Dialogue evidence:",
        dialogue,
        "",
        f"Question type: {qtype}",
        f"Question: {qa.get('question')}",
    ]
    if options:
        parts.extend(["", "Options:", options])
    parts.extend(["", format_hint])
    final_instruction = FINAL_ANSWER_INSTRUCTIONS.get(qtype)
    if final_instruction:
        parts.extend(["", final_instruction])
    return "\n".join(parts)


def render_prompt(tokenizer: Any, sample: dict[str, Any]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a long-dialogue graph-memory QA solver. Use only the dialogue evidence and the listed "
                "options if present. Follow the requested output format exactly."
            ),
        },
        {"role": "user", "content": solver_user_prompt(sample)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"


def effective_solver_max_new_tokens(sample: dict[str, Any], args: argparse.Namespace) -> int:
    qtype = str(sample.get("qa", {}).get("question_type") or "")
    cap = STRUCTURED_MAX_NEW_TOKENS.get(qtype, int(args.max_new_tokens))
    return max(1, min(int(args.max_new_tokens), int(cap)))


def stop_at_solver_sequence(text: str) -> str:
    cleaned = str(text or "")
    first = None
    for marker in SOLVER_STOP_SEQUENCES:
        idx = cleaned.find(marker)
        if idx >= 0 and (first is None or idx < first):
            first = idx
    if first is not None:
        cleaned = cleaned[:first]
    return cleaned.strip()


def estimate_solver_tokens(tokenizer: Any, prompt: str) -> tuple[int, bool]:
    try:
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        return len(ids), False
    except Exception:
        return max(1, len(prompt) // 4), True


def count_solver_options(sample: dict[str, Any]) -> int:
    qa_options = sample.get("qa", {}).get("options")
    if isinstance(qa_options, list):
        return len([item for item in qa_options if isinstance(item, dict)])
    if isinstance(qa_options, dict):
        return len(qa_options)
    return 0


def solver_prompt_diagnostics(
    *,
    sample: dict[str, Any],
    prompt: str,
    tokenizer: Any,
    args: argparse.Namespace,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    token_estimate, estimated = estimate_solver_tokens(tokenizer, prompt)
    budget = int(getattr(args, "max_model_len", 0) or getattr(args, "max_length", 0) or 0)
    generation_budget = int(max_new_tokens if max_new_tokens is not None else effective_solver_max_new_tokens(sample, args))
    overflow = bool(budget and token_estimate + generation_budget > budget)
    dialogue_turns = sample.get("dialogue", {}).get("turns")
    instruction = FINAL_ANSWER_INSTRUCTIONS.get(str(sample.get("qa", {}).get("question_type") or ""), "")
    return {
        "prompt_char_len": len(prompt),
        "prompt_token_estimate": token_estimate,
        "prompt_token_estimate_is_approx": estimated,
        "max_model_len": budget,
        "context_budget": budget,
        "max_new_tokens": generation_budget,
        "truncated": overflow,
        "eval_context_overflow": overflow,
        "dialogue_turn_count": len(dialogue_turns) if isinstance(dialogue_turns, list) else 0,
        "options_count": count_solver_options(sample),
        "answer_format_instruction_present": bool(instruction and instruction in prompt),
    }


def solver_overflow_prediction(sample: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return ["" for _ in range(max(1, int(args.k_eval)))]


def write_solver_prompt_debug_files(samples: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace) -> None:
    debug_dir = getattr(args, "prompt_debug_dir", None)
    if not debug_dir:
        return
    output_dir = Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples[:3]):
        prompt = render_prompt(tokenizer, sample)
        diagnostics = solver_prompt_diagnostics(
            sample=sample,
            prompt=prompt,
            tokenizer=tokenizer,
            args=args,
            max_new_tokens=effective_solver_max_new_tokens(sample, args),
        )
        path = output_dir / f"solver_prompt_debug_sample_{index + 1}.txt"
        path.write_text(
            "\n".join(
                [
                    f"sample_id: {sample.get('sample_id')}",
                    f"question_type: {sample.get('qa', {}).get('question_type')}",
                    f"prompt_token_estimate: {diagnostics['prompt_token_estimate']}",
                    f"max_model_len: {diagnostics['max_model_len']}",
                    f"max_new_tokens: {diagnostics['max_new_tokens']}",
                    f"eval_context_overflow: {diagnostics['eval_context_overflow']}",
                    "",
                    prompt,
                ]
            ),
            encoding="utf-8",
        )


def collect_solver_prompt_diagnostics(samples: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for sample in samples:
        prompt = render_prompt(tokenizer, sample)
        diagnostics.append(
            solver_prompt_diagnostics(
                sample=sample,
                prompt=prompt,
                tokenizer=tokenizer,
                args=args,
                max_new_tokens=effective_solver_max_new_tokens(sample, args),
            )
        )
    return diagnostics


def parse_letters(text: str, allowed: set[str]) -> list[str]:
    """Parse an option-label sequence only from the start of the response.

    Solver prompts require a label-only answer.  The previous fallback removed
    every non-letter character from arbitrary output and then searched the
    remaining prose for A-D.  Corrupted text such as ``each();`` therefore
    became option A and could be scored as correct.  Accept a short optional
    ``Answer:`` prefix for compatibility, but never mine labels from later
    prose.
    """

    cleaned = str(text or "").strip().upper()
    match = re.match(
        r"^(?:FINAL\s+ANSWER|ANSWER)?\s*[:=\-]?\s*"
        r"([A-Z](?:\s*(?:,|>|/|\|)\s*[A-Z])*)"
        r"(?=$|\s|[.\):;,-])",
        cleaned,
    )
    if not match:
        return []
    found: list[str] = []
    for token in re.findall(r"[A-Z]", match.group(1)):
        if token in allowed and token not in found:
            found.append(token)
    return found


def parse_turn_ids(text: str, allowed: set[str]) -> list[str]:
    found = []
    for token in re.findall(r"\bD\d+:\d+\b", text):
        if token in allowed and token not in found:
            found.append(token)
    return found


def parse_statement_ids(text: str, allowed: set[str]) -> list[str]:
    found = []
    for token in re.findall(r"\b(?:S\d+|E_[A-Za-z0-9_]+|E\d+)\b", text):
        if token in allowed and token not in found:
            found.append(token)
    return found


def f1_score(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def pairwise_order_accuracy(pred: list[str], gold: list[str]) -> float:
    if len(gold) < 2:
        return 1.0 if pred == gold else 0.0
    pred_pos = {label: idx for idx, label in enumerate(pred)}
    correct = 0
    total = 0
    for i, left in enumerate(gold):
        for right in gold[i + 1 :]:
            total += 1
            if left in pred_pos and right in pred_pos and pred_pos[left] < pred_pos[right]:
                correct += 1
    return correct / total if total else 0.0


def state_partial_credit(prediction: str, gold: str) -> float:
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    if pred_norm == gold_norm:
        return 1.0
    pred_parts = [normalize_text(item) for item in re.split(r"\s*/\s*", pred_norm) if item.strip()]
    gold_parts = [normalize_text(item) for item in re.split(r"\s*/\s*", gold_norm) if item.strip()]
    if len(gold_parts) >= 2:
        matches = sum(1 for part in gold_parts if any(part == pred_part or part in pred_part or pred_part in part for pred_part in pred_parts))
        return matches / len(gold_parts)
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    return f1_score(pred_tokens, gold_tokens)


def score_prediction(sample: dict[str, Any], prediction: str) -> tuple[bool, float, Any]:
    qa = sample["qa"]
    qtype = str(qa.get("question_type"))
    gold = [str(item) for item in qa.get("gold", [])]
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    allowed_ids = option_ids(options)
    if qtype == "single_choice":
        pred = parse_letters(prediction, allowed_ids)
        exact = bool(gold) and bool(pred) and pred[0] == gold[0]
        return exact, 1.0 if exact else 0.0, pred[0] if pred else ""
    if qtype == "multi_select":
        pred = parse_letters(prediction, allowed_ids)
        exact = set(pred) == set(gold)
        return exact, f1_score(set(pred), set(gold)), pred
    if qtype == "ordering":
        pred = parse_letters(prediction, allowed_ids)
        exact = pred == gold
        return exact, pairwise_order_accuracy(pred, gold), pred
    if qtype == "evidence_selection":
        allowed_turns = {str(option.get("id")) for option in options if isinstance(option, dict)}
        pred = parse_turn_ids(prediction, allowed_turns)
        # Single-gold evidence_selection is single-answer: the solver's FIRST selected turn must
        # equal the gold turn. This avoids penalizing a correct primary pick that is followed by
        # extra ids, while still requiring the model to lead with the right turn. Multi-gold (rare)
        # keeps the exact set-match semantics.
        if len(gold) == 1:
            exact = bool(pred) and pred[0] == gold[0]
        else:
            exact = set(pred) == set(gold)
        return exact, f1_score(set(pred), set(gold)), pred
    if qtype == "invalidated_statement_detection":
        pred = parse_statement_ids(prediction, allowed_ids)
        exact = pred[:3] == gold
        component_matches = sum(1 for idx, item in enumerate(gold) if idx < len(pred) and pred[idx] == item)
        if component_matches < 2:
            component_matches = len(set(pred) & set(gold))
        return exact, component_matches / max(1, len(gold)), pred
    if qtype == "state_reconstruction":
        gold_text = gold[0] if gold else ""
        pred_norm = normalize_text(prediction)
        gold_norm = normalize_text(gold_text)
        exact = pred_norm == gold_norm
        return exact, state_partial_credit(prediction, gold_text), prediction.strip()
    return False, 0.0, prediction.strip()


def heuristic_predictions(sample: dict[str, Any], k: int, rng: random.Random) -> list[str]:
    qa = sample["qa"]
    qtype = str(qa.get("question_type"))
    gold = [str(item) for item in qa.get("gold", [])]
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    labels = [str(option.get("id")) for option in options if isinstance(option, dict)]
    outputs = []
    for idx in range(k):
        if qtype == "state_reconstruction":
            outputs.append(gold[0] if idx % 3 else str(options[0]["text"]) if options else (gold[0] if gold else ""))
        elif qtype == "ordering":
            value = list(gold)
            if idx % 2 == 0 and len(value) >= 2:
                value[-1], value[-2] = value[-2], value[-1]
            outputs.append(" > ".join(value))
        elif qtype == "multi_select":
            value = list(gold)
            if idx % 2 == 0 and labels:
                value = value[:1] + [next((label for label in labels if label not in value), value[-1])]
            outputs.append(", ".join(value))
        elif qtype == "evidence_selection":
            value = list(gold)
            if idx % 2 == 0 and labels:
                value = value[:1]
            outputs.append(", ".join(value))
        elif qtype == "invalidated_statement_detection":
            value = list(gold)
            if idx % 2 == 0 and len(value) == 3:
                value = value[:2] + [rng.choice(labels) if labels else value[-1]]
            outputs.append(" | ".join(value))
        elif qtype == "single_choice":
            outputs.append(gold[0] if gold else (labels[0] if labels else ""))
        else:
            outputs.append(", ".join(gold))
    return outputs


def evaluate_with_vllm(samples: list[dict[str, Any]], args: argparse.Namespace, model_path: str) -> tuple[list[list[str]], list[dict[str, Any]]]:
    # CRITICAL (Blackwell SM 12.x + vLLM 0.21): the default FlashInfer sampler JIT-fails on this
    # GPU and silently degrades decoding to gibberish ("the the the", "000000"), making every
    # solver eval score garbage. Force it off in-process BEFORE vLLM is imported so the pipeline's
    # own eval is always clean — do not rely on the caller exporting the env var.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    write_solver_prompt_debug_files(samples, tokenizer, args)
    prompts = [render_prompt(tokenizer, sample) for sample in samples]
    diagnostics = [
        solver_prompt_diagnostics(
            sample=sample,
            prompt=prompt,
            tokenizer=tokenizer,
            args=args,
            max_new_tokens=effective_solver_max_new_tokens(sample, args),
        )
        for sample, prompt in zip(samples, prompts)
    ]
    raw_predictions: list[list[str]] = [solver_overflow_prediction(sample, args) for sample in samples]
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for index, (prompt, diagnostic) in enumerate(zip(prompts, diagnostics)):
        if diagnostic.get("eval_context_overflow"):
            continue
        grouped[int(diagnostic["max_new_tokens"])].append((index, prompt))
    if not grouped:
        return raw_predictions, diagnostics
    llm = LLM(
        model=model_path,
        tokenizer=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    for max_tokens, items in sorted(grouped.items()):
        params = SamplingParams(
            n=args.k_eval,
            temperature=args.temperature if args.k_eval > 1 else 0.0,
            top_p=args.top_p if args.k_eval > 1 else 1.0,
            max_tokens=max_tokens,
            stop=list(SOLVER_STOP_SEQUENCES),
        )
        outputs = llm.generate([prompt for _, prompt in items], params)
        for (sample_index, _), request_output in zip(items, outputs):
            raw_predictions[sample_index] = [stop_at_solver_sequence(item.text) for item in request_output.outputs]
    return raw_predictions, diagnostics


def evaluate_with_transformers(samples: list[dict[str, Any]], args: argparse.Namespace, model_path: str) -> tuple[list[list[str]], list[dict[str, Any]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    write_solver_prompt_debug_files(samples, tokenizer, args)
    diagnostics = collect_solver_prompt_diagnostics(samples, tokenizer, args)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    all_outputs: list[list[str]] = []
    with torch.no_grad():
        for sample, diagnostic in zip(samples, diagnostics):
            if diagnostic.get("eval_context_overflow"):
                all_outputs.append(["" for _ in range(max(1, int(args.k_eval)))])
                continue
            prompt = render_prompt(tokenizer, sample)
            max_new_tokens = int(diagnostic["max_new_tokens"])
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            sample_outputs: list[str] = []
            for _ in range(args.k_eval):
                do_sample = args.k_eval > 1
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
                sample_outputs.append(stop_at_solver_sequence(tokenizer.decode(new_tokens, skip_special_tokens=True)))
            all_outputs.append(sample_outputs)
    return all_outputs, diagnostics


def run_solver_eval(samples: list[dict[str, Any]], args: argparse.Namespace, solver_model_path: str) -> tuple[list[dict[str, Any]], str]:
    rng = random.Random(args.seed + 17)
    backend = args.eval_backend
    raw_predictions: list[list[str]]
    prompt_diagnostics: list[dict[str, Any]] = [{} for _ in samples]
    used_backend = backend
    if backend == "heuristic":
        raw_predictions = [heuristic_predictions(sample, args.k_eval, rng) for sample in samples]
        prompt_diagnostics = [
            {
                "prompt_char_len": 0,
                "prompt_token_estimate": 0,
                "prompt_token_estimate_is_approx": True,
                "max_model_len": int(getattr(args, "max_model_len", 0) or 0),
                "context_budget": int(getattr(args, "max_model_len", 0) or 0),
                "max_new_tokens": effective_solver_max_new_tokens(sample, args),
                "truncated": False,
                "eval_context_overflow": False,
                "dialogue_turn_count": len(sample.get("dialogue", {}).get("turns") or []),
                "options_count": count_solver_options(sample),
                "answer_format_instruction_present": False,
            }
            for sample in samples
        ]
    elif backend in {"auto", "vllm"}:
        try:
            raw_predictions, prompt_diagnostics = evaluate_with_vllm(samples, args, solver_model_path)
            used_backend = "vllm"
        except Exception as error:
            if backend == "vllm":
                raise
            print(f"[warn] vLLM eval failed; falling back to transformers: {error}", file=sys.stderr, flush=True)
            raw_predictions, prompt_diagnostics = evaluate_with_transformers(samples, args, solver_model_path)
            used_backend = "transformers"
    else:
        raw_predictions, prompt_diagnostics = evaluate_with_transformers(samples, args, solver_model_path)
        used_backend = "transformers"

    eval_rows = []
    for sample, predictions, diagnostics in zip(samples, raw_predictions, prompt_diagnostics):
        scored = []
        exact_count = 0
        partials = []
        for idx, prediction in enumerate(predictions[: args.k_eval]):
            exact, partial, parsed = score_prediction(sample, prediction)
            exact_count += 1 if exact else 0
            partials.append(partial)
            scored.append(
                {
                    "sample_index": idx,
                    "prediction": prediction,
                    "parsed": parsed,
                    "exact_correct": exact,
                    "partial_credit": round(partial, 6),
                }
            )
        while len(scored) < args.k_eval:
            scored.append({"sample_index": len(scored), "prediction": "", "parsed": "", "exact_correct": False, "partial_credit": 0.0})
            partials.append(0.0)
        soft_accuracy = exact_count / args.k_eval if args.k_eval else 0.0
        partial_credit = sum(partials) / args.k_eval if args.k_eval else 0.0
        bucket = f"{exact_count}/{args.k_eval}"
        sample["solver_eval"] = {
            "k": args.k_eval,
            "predictions": scored,
            "exact_correct_count": exact_count,
            "soft_accuracy": round(soft_accuracy, 6),
            "partial_credit": round(partial_credit, 6),
            "bucket": bucket,
            "diagnostics": diagnostics,
        }
        eval_rows.append(
            {
                "sample_id": sample["sample_id"],
                "question_type": sample["qa"]["question_type"],
                "attack_skill": sample["attack_plan"]["attack_skill"],
                "solver_eval": sample["solver_eval"],
            }
        )
    return eval_rows, used_backend


def is_clean(sample: dict[str, Any]) -> bool:
    validation = sample.get("validation", {})
    return (
        validation.get("graph_dialogue_consistent") is True
        and validation.get("gold_derivable") is True
        and validation.get("option_valid") is True
        and validation.get("anti_shortcut_pass") is True
        and not validation.get("warnings")
    )


def base_valid(sample: dict[str, Any]) -> bool:
    validation = sample.get("validation", {})
    return (
        validation.get("graph_dialogue_consistent") is True
        and validation.get("gold_derivable") is True
        and validation.get("option_valid") is True
        and validation.get("anti_shortcut_pass") is True
    )


def rare_mechanism(sample: dict[str, Any], samples: list[dict[str, Any]]) -> bool:
    mechanism = sample["attack_plan"]["reasoning_mechanism"]
    counts = Counter(item["attack_plan"]["reasoning_mechanism"] for item in samples)
    return counts[mechanism] <= max(3, len(samples) // 12)


def candidate_priority(sample: dict[str, Any]) -> tuple[float, float, str]:
    soft = float(sample["solver_eval"].get("soft_accuracy") or 0.0)
    partial = float(sample["solver_eval"].get("partial_credit") or 0.0)
    target_distance = abs(soft - 0.4)
    return (-partial, target_distance, sample["sample_id"])


def assign_initial_frontier_labels(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        validation = sample.get("validation", {})
        soft = float(sample["solver_eval"].get("soft_accuracy") or 0.0)
        partial = float(sample["solver_eval"].get("partial_credit") or 0.0)
        if validation.get("option_valid") is not True:
            category, reason = "format_issue", "option validation failed"
        elif validation.get("gold_derivable") is not True:
            category, reason = "gold_issue", "gold was not derivable from graph/dialogue"
        elif validation.get("graph_dialogue_consistent") is not True:
            category, reason = "ambiguous_issue", "graph/dialogue evidence mismatch"
        elif validation.get("anti_shortcut_pass") is not True:
            category, reason = "shortcut_issue", "anti-shortcut validation failed"
        elif not is_clean(sample):
            category, reason = "ambiguous_issue", "validation warnings present"
        elif 0.2 <= soft <= 0.6 and partial >= 0.4:
            category, reason = "clean_near_miss", "soft accuracy in frontier band with sufficient partial credit"
        elif soft == 0.8 and partial >= 0.4 and rare_mechanism(sample, samples):
            category, reason = "hard_correct", "soft accuracy 0.8 with rare reasoning mechanism"
        elif soft == 0.0:
            if partial >= 0.6:
                category, reason = "clean_near_miss", "zero exact accuracy but high partial credit retained by audit"
            elif partial >= 0.4:
                category, reason = "true_too_hard", "zero exact accuracy with only moderate partial credit"
            else:
                category, reason = "bad_question", "zero exact accuracy and low partial credit"
        elif soft > 0.6:
            category, reason = "easy", "solver soft accuracy above frontier band"
        else:
            category, reason = "true_too_hard", "below frontier band or insufficient partial credit"
        sample["frontier_label"] = {"is_frontier": False, "category": category, "reason": reason}


def balanced_select_frontier(samples: list[dict[str, Any]], max_type_share: float, max_skill_share: float) -> list[dict[str, Any]]:
    candidates = [
        sample
        for sample in samples
        if sample["frontier_label"]["category"] in {"clean_near_miss", "hard_correct"}
    ]
    candidates.sort(key=candidate_priority)
    selected: list[dict[str, Any]] = []
    for sample in candidates:
        selected.append(sample)
    # Prune only when imbalance is real; this keeps the policy transparent and deterministic.
    changed = True
    while changed and selected:
        changed = False
        type_counts = Counter(sample["qa"]["question_type"] for sample in selected)
        skill_counts = Counter(sample["attack_plan"]["attack_skill"] for sample in selected)
        n = len(selected)
        largest_type, type_count = type_counts.most_common(1)[0]
        largest_skill, skill_count = skill_counts.most_common(1)[0]
        remove_group: tuple[str, str] | None = None
        if type_count / n > max_type_share:
            remove_group = ("type", largest_type)
        elif skill_count / n > max_skill_share:
            remove_group = ("skill", largest_skill)
        if remove_group:
            group_kind, group_value = remove_group
            group_items = [
                sample
                for sample in selected
                if (sample["qa"]["question_type"] if group_kind == "type" else sample["attack_plan"]["attack_skill"]) == group_value
            ]
            group_items.sort(key=candidate_priority, reverse=True)
            selected.remove(group_items[0])
            changed = True
    selected_ids = {sample["sample_id"] for sample in selected}
    for sample in samples:
        if sample["sample_id"] in selected_ids:
            sample["frontier_label"]["is_frontier"] = True
        elif sample["frontier_label"]["category"] in {"clean_near_miss", "hard_correct"}:
            sample["frontier_label"]["reason"] += "; excluded by balanced frontier cap"
    return selected


def build_failure_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    validation_failures: Counter[str] = Counter()
    shortcut_failures: Counter[str] = Counter()
    for sample in samples:
        details = sample.get("validation", {}).get("details", {})
        for key, value in (details.get("validation_checks") or {}).items():
            if value is not True:
                validation_failures[key] += 1
        for key, value in (details.get("anti_shortcut_checks") or {}).items():
            if value:
                shortcut_failures[key] += 1
    matrix: dict[str, Any] = {
        "total": len(samples),
        "by_frontier_category": count_map([sample["frontier_label"]["category"] for sample in samples]),
        "by_question_type": count_map([sample["qa"]["question_type"] for sample in samples]),
        "by_attack_skill": count_map([sample["attack_plan"]["attack_skill"] for sample in samples]),
        "validation_failures": dict(validation_failures),
        "shortcut_failures": dict(shortcut_failures),
        "soft_buckets_by_question_type": {},
        "soft_buckets_by_attack_skill": {},
    }
    for key, getter in {
        "soft_buckets_by_question_type": lambda sample: sample["qa"]["question_type"],
        "soft_buckets_by_attack_skill": lambda sample: sample["attack_plan"]["attack_skill"],
    }.items():
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        for sample in samples:
            buckets[str(getter(sample))][str(sample["solver_eval"].get("bucket"))] += 1
        matrix[key] = {group: dict(counter) for group, counter in sorted(buckets.items())}
    return matrix


def largest_share(counter: dict[str, int]) -> float:
    total = sum(counter.values())
    return round(max(counter.values()) / total, 6) if total else 0.0


def recommendation(report: dict[str, Any]) -> str:
    success = (
        report["clean_count"] >= 160
        and report["frontier_count"] >= 25
        and report["frontier_yield"] >= 0.15
        and report["near_miss_count"] > report["true_too_hard_count"]
        and report["largest_type_share"] <= 0.35
        and report["largest_skill_share"] <= 0.25
        and len(report["frontier_by_attack_skill"]) >= 5
    )
    partial = (
        report["clean_count"] >= 140
        and report["frontier_count"] >= 15
        and report["frontier_yield"] >= 0.075
        and report["largest_type_share"] <= 0.45
        and report["largest_skill_share"] <= 0.35
    )
    if success:
        return "proceed_to_old_improved_r5_train_small"
    if partial:
        return "revise_challenger_prompt_and_rerun_gate"
    return "do_not_train_old_method_yet"


def build_report(samples: list[dict[str, Any]], args: argparse.Namespace, solver_model_path: str, adapter_path: str | None, eval_backend: str) -> dict[str, Any]:
    frontier = [sample for sample in samples if sample["frontier_label"]["is_frontier"]]
    soft_buckets = {f"{idx}/{args.k_eval}": 0 for idx in range(args.k_eval + 1)}
    for sample in samples:
        bucket = str(sample["solver_eval"].get("bucket") or "0/5")
        soft_buckets[bucket] = soft_buckets.get(bucket, 0) + 1
    frontier_by_question_type = count_map([sample["qa"]["question_type"] for sample in frontier])
    frontier_by_attack_skill = count_map([sample["attack_plan"]["attack_skill"] for sample in frontier])
    first_correct = sum(1 for sample in samples if sample["solver_eval"]["predictions"] and sample["solver_eval"]["predictions"][0]["exact_correct"])
    warning_samples = sum(1 for sample in samples if sample.get("validation", {}).get("warnings"))
    report = {
        "experiment": "old_improved_gate200",
        "fixed_solver": "r3",
        "solver_model_path": solver_model_path,
        "solver_adapter_path": adapter_path,
        "eval_backend": eval_backend,
        "generated_count": len(samples),
        "valid_count": sum(1 for sample in samples if base_valid(sample)),
        "clean_count": sum(1 for sample in samples if is_clean(sample)),
        "warning_count": warning_samples,
        "solver_exact_accuracy_k1": round(first_correct / len(samples), 6) if samples else 0.0,
        "solver_soft_accuracy_k5": round(
            sum(float(sample["solver_eval"].get("soft_accuracy") or 0.0) for sample in samples) / len(samples), 6
        )
        if samples
        else 0.0,
        "wrong_count_k1": len(samples) - first_correct,
        "soft_buckets": soft_buckets,
        "near_miss_count": sum(1 for sample in samples if sample["frontier_label"]["category"] == "clean_near_miss"),
        "hard_correct_count": sum(1 for sample in samples if sample["frontier_label"]["category"] == "hard_correct"),
        "true_too_hard_count": sum(1 for sample in samples if sample["frontier_label"]["category"] == "true_too_hard"),
        "bad_question_count": sum(
            1
            for sample in samples
            if sample["frontier_label"]["category"] in {"bad_question", "format_issue", "gold_issue", "ambiguous_issue", "shortcut_issue"}
        ),
        "frontier_count": len(frontier),
        "frontier_yield": round(len(frontier) / len(samples), 6) if samples else 0.0,
        "frontier_by_question_type": frontier_by_question_type,
        "frontier_by_attack_skill": frontier_by_attack_skill,
        "largest_type_share": largest_share(frontier_by_question_type),
        "largest_skill_share": largest_share(frontier_by_attack_skill),
        "shortcut_warning_count": sum(1 for sample in samples if sample.get("validation", {}).get("anti_shortcut_pass") is not True),
        "generated_by_question_type": count_map([sample["qa"]["question_type"] for sample in samples]),
        "generated_by_attack_skill": count_map([sample["attack_plan"]["attack_skill"] for sample in samples]),
        "max_largest_question_type_share": args.max_largest_question_type_share,
        "max_largest_attack_skill_share": args.max_largest_attack_skill_share,
        "recommendation": "",
    }
    report["recommendation"] = recommendation(report)
    return report


def build_markdown_report(report: dict[str, Any], failure_matrix: dict[str, Any]) -> str:
    lines = [
        "# old_improved_gate200 Gate Report",
        "",
        f"- fixed_solver: {report['fixed_solver']}",
        f"- eval_backend: {report.get('eval_backend')}",
        f"- generated_count: {report['generated_count']}",
        f"- valid_count: {report['valid_count']}",
        f"- clean_count: {report['clean_count']}",
        f"- frontier_count: {report['frontier_count']}",
        f"- frontier_yield: {report['frontier_yield']}",
        f"- solver_exact_accuracy_k1: {report['solver_exact_accuracy_k1']}",
        f"- solver_soft_accuracy_k5: {report['solver_soft_accuracy_k5']}",
        f"- largest_type_share: {report['largest_type_share']}",
        f"- largest_skill_share: {report['largest_skill_share']}",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Soft Buckets",
        "",
    ]
    for bucket, count in report["soft_buckets"].items():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "## Frontier By Question Type", ""])
    for key, count in report["frontier_by_question_type"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Frontier By Attack Skill", ""])
    for key, count in report["frontier_by_attack_skill"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Failure Categories", ""])
    for key, count in failure_matrix.get("by_frontier_category", {}).items():
        lines.append(f"- {key}: {count}")
    lines.append("")
    return "\n".join(lines)


def choose_source_bundles(rows: list[dict[str, Any]], num_samples: int, rng: random.Random) -> list[SourceBundle]:
    bundles = [bundle for row in rows if (bundle := build_source_bundle(row))]
    if not bundles:
        raise ValueError("No seed rows contained enough dialogue turns and graph/event-chain nodes.")
    rng.shuffle(bundles)
    if len(bundles) >= num_samples:
        return bundles[:num_samples]
    selected = []
    while len(selected) < num_samples:
        selected.extend(bundles)
    return selected[:num_samples]


def generate_samples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    bundles = choose_source_bundles(rows, args.num_samples, rng)
    attack_plan_labels = expand_plan(ATTACK_SKILL_QUOTAS, args.num_samples)
    qtype_labels = expand_plan(QUESTION_TYPE_QUOTAS, args.num_samples)
    rng.shuffle(attack_plan_labels)
    rng.shuffle(qtype_labels)
    samples: list[dict[str, Any]] = []
    for idx, bundle in enumerate(bundles):
        sample_rng = random.Random(args.seed + idx * 7919)
        sample = build_sample(bundle, idx, qtype_labels[idx], attack_plan_labels[idx], sample_rng)
        samples.append(sample)
        if not args.quiet and (idx + 1) % 50 == 0:
            print(f"generated {idx + 1}/{args.num_samples}", flush=True)
    return samples


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_model_path, adapter_path = resolve_solver_checkpoint(args.solver_checkpoint)
    rows = load_seed_rows(args.seed_data)
    if args.limit_seed_rows is not None:
        rows = rows[: args.limit_seed_rows]
    samples = generate_samples(rows, args)

    # Persist generation output before eval so a failed model run still leaves auditable candidates.
    write_jsonl(output_dir / "samples.jsonl", samples)
    valid_samples = [sample for sample in samples if base_valid(sample)]
    write_jsonl(output_dir / "valid_samples.jsonl", valid_samples)

    eval_rows, used_backend = run_solver_eval(samples, args, solver_model_path)
    assign_initial_frontier_labels(samples)
    frontier = balanced_select_frontier(
        samples,
        args.max_largest_question_type_share,
        args.max_largest_attack_skill_share,
    )
    failure_matrix = build_failure_matrix(samples)
    report = build_report(samples, args, solver_model_path, adapter_path, used_backend)

    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "valid_samples.jsonl", [sample for sample in samples if base_valid(sample)])
    write_jsonl(output_dir / "solver_k5_eval.jsonl", eval_rows)
    write_jsonl(output_dir / "frontier.jsonl", frontier)
    write_json(output_dir / "failure_matrix.json", failure_matrix)
    write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(build_markdown_report(report, failure_matrix), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved old_improved_gate200 outputs to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
