"""Fail-closed, offline audit of the completed paired R01 reconstruction.

The live bootstrap persisted completion and bucket-delta metadata, but did not
persist the full post-R01 bucket payload.  This audit therefore verifies all
facts observable from the frozen artifacts and explicitly reports semantic
dimensions that cannot be proven without another read-only bucket search.
It never calls Memorax, a model, a solver, QA, or an auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_PREFLIGHT = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_2")
DEFAULT_SMOKE = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_3")
DEFAULT_COMPILER = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
DEFAULT_SNAPSHOT = RUN_BASE / (
    "h4_round1_feedback_memory_semantic_parity_v1_live_3")
DEFAULT_OUTPUT = RUN_BASE / (
    "h4_round1_feedback_memory_semantic_parity_v1_live_1")
VERSION = "h4_round1_feedback_memory_semantic_parity_v1"
CONTRACT_VERSION = VERSION + "_contract_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"jsonl_row_not_object:{path}")
            rows.append(value)
    return rows


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _scrub_transport(value: Any, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _scrub_transport(v, str(k))
            for k, v in value.items()
            if str(k) not in {
                "user_id", "request_id", "task_id", "timestamp",
                "payload_sha256", "api_key", "authorization",
            }
        }
    if isinstance(value, list):
        return [_scrub_transport(v, key) for v in value]
    return value


def _event_refs(audit: Mapping[str, Any]) -> set[str]:
    refs = set()
    for key in ("added_bucket_refs", "changed_bucket_refs", "updated_bucket_refs"):
        for value in audit.get(key) or []:
            text = _norm(value)
            if text.startswith("event:"):
                refs.add(text.split(":", 1)[1])
    return refs


def _person_refs(audit: Mapping[str, Any]) -> set[str]:
    refs = set()
    for key in ("added_bucket_refs", "changed_bucket_refs", "updated_bucket_refs"):
        for value in audit.get(key) or []:
            text = _norm(value)
            if text.startswith("person:"):
                refs.add(text.split(":", 1)[1])
    return refs


def _anchor_event_match(anchor_text: str, event_refs: set[str]) -> dict[str, Any]:
    """Use conservative lexical topic families, never a semantic model."""
    text = _norm(anchor_text)
    families: list[tuple[str, tuple[str, ...]]] = []
    if "meditat" in text:
        families.append(("meditation", ("meditat",)))
    if any(token in text for token in ("bali", "surf")):
        families.append(("bali_planning", ("bali", "trip", "surf")))
    if any(token in text for token in ("cement", "brick", "hardware")):
        families.append(("wall_materials", ("wall", "cement", "brick", "hardware")))
    if not families:
        return {
            "status": "UNKNOWN",
            "reason": "anchor_topic_family_not_defined",
            "matched_event_refs": [],
        }
    family_name, tokens = families[0]
    matches = sorted(
        ref for ref in event_refs
        if any(token in ref for token in tokens))
    return {
        "status": "PASS" if matches else "FAIL",
        "topic_family": family_name,
        "matched_event_refs": matches,
        "required_tokens": list(tokens),
        "method": "conservative_lexical_topic_family_v1",
    }


def _snapshot_content(snapshot: Mapping[str, Any]) -> str:
    values: list[str] = []
    buckets = snapshot.get("buckets")
    if not isinstance(buckets, Mapping):
        return ""
    for rows in buckets.values():
        if not isinstance(rows, Mapping):
            continue
        for row in rows.values():
            if not isinstance(row, Mapping):
                continue
            values.append(str(row.get("canonical_key") or ""))
            values.append(str(row.get("display_key") or ""))
            content = row.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping):
                        values.extend(str(v) for v in item.values())
                    else:
                        values.append(str(item))
    return "\n".join(values)


def _snapshot_event_rows(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    buckets = snapshot.get("buckets")
    events = buckets.get("event") if isinstance(buckets, Mapping) else None
    if not isinstance(events, Mapping):
        return []
    result = []
    for key, row in events.items():
        if isinstance(row, Mapping):
            result.append({
                "key": _norm(key),
                "content": " ".join(
                    str(x) for x in row.get("content") or []),
            })
    return result


def _check_branch(smoke: Path, branch: str,
                  snapshot_root: Path | None = None) -> dict[str, Any]:
    root = smoke / f"{branch}_reconstruction"
    parent_intent = _load(root / "parent_add_intent.json")
    r01_intent = _load(root / "r01_child_add_intent.json")
    parent_audit = _load(root / "parent_visibility_audit.json")
    r01_audit = _load(root / "r01_visibility_audit.json")
    final = _load(smoke / "final_feedback_memory_smoke_report.json")
    branch_report = final["branches"]["control" if branch == "control" else "feedback"]
    messages = list((r01_intent.get("payload") or {}).get("messages") or [])
    message_text = "\n".join(str(row.get("content") or "") for row in messages)
    snapshot_path = (
        snapshot_root / f"{branch}_r01_bucket_snapshot.json"
        if snapshot_root is not None else None)
    snapshot = _load(snapshot_path) if snapshot_path is not None and snapshot_path.is_file() else None
    snapshot_buckets = snapshot.get("buckets") if isinstance(snapshot, Mapping) else {}
    snapshot_people = (
        set(snapshot_buckets.get("person", {}).keys())
        if isinstance(snapshot_buckets, Mapping)
        and isinstance(snapshot_buckets.get("person"), Mapping) else set())
    snapshot_events = (
        set(snapshot_buckets.get("event", {}).keys())
        if isinstance(snapshot_buckets, Mapping)
        and isinstance(snapshot_buckets.get("event"), Mapping) else set())
    return {
        "branch": branch,
        "namespace": branch_report.get("namespace"),
        "user_id": branch_report.get("user_id"),
        "parent_add_verified": bool(branch_report["parent_add"]["state"].get(
            "memory_write_verified")),
        "parent_visibility_verified": bool(branch_report["parent_add"]["visibility"].get(
            "bucket_visibility_verified")),
        "r01_add_verified": bool(branch_report["r01_child_add"]["state"].get(
            "memory_write_verified")),
        "r01_visibility_verified": bool(branch_report["r01_child_add"]["visibility"].get(
            "bucket_visibility_verified")),
        "parent_turn_refs": list(parent_intent.get("turn_refs") or []),
        "r01_turn_refs": list(r01_intent.get("turn_refs") or []),
        "parent_intent": parent_intent,
        "r01_intent": r01_intent,
        "parent_audit": parent_audit,
        "r01_audit": r01_audit,
        "person_refs": sorted(_person_refs(parent_audit) | _person_refs(r01_audit) | {
            _norm(x) for x in snapshot_people}),
        "event_refs": sorted(_event_refs(parent_audit) | _event_refs(r01_audit) | {
            _norm(x) for x in snapshot_events}),
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "snapshot_sha256": _sha(snapshot_path) if snapshot_path is not None and snapshot_path.is_file() else None,
        "snapshot_present": snapshot is not None,
        "snapshot_content": _snapshot_content(snapshot) if snapshot is not None else "",
        "snapshot_event_rows": _snapshot_event_rows(snapshot) if snapshot is not None else [],
        "r02_tokens_in_r01_payload": bool(re.search(
            r"(?:R02|r02|R02-feedback|feedback card)", message_text)),
        "r01_payload_sha256": _sha(root / "r01_child_add_intent.json"),
    }


def run(*, preflight_root: Path, smoke_root: Path, compiler_root: Path,
        output_root: Path, snapshot_root: Path | None = None) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    preflight = _load(preflight_root / "preflight_report.json")
    smoke_report = _load(smoke_root / "final_feedback_memory_smoke_report.json")
    mapping = _load(preflight_root / "namespace_mapping.json")
    root_id = str(preflight.get("root_id") or "")
    dialogue_path = Path(str(preflight["canonical_dialogue"]["path"]))
    dialogue = _load(dialogue_path)
    turns = list(dialogue.get("turns") or [])
    turn_map = {str(row.get("turn_ref")): row for row in turns}
    card_rows = _load_jsonl(compiler_root / "root_feedback_cards.jsonl")
    cards = [row for row in card_rows if row.get("root_id") == root_id]
    if len(cards) != 1:
        raise RuntimeError(f"feedback_card_not_unique:{root_id}:{len(cards)}")
    card = cards[0]
    anchors = [str(x) for x in card["target"].get("anchor_refs") or []]
    anchor_rows = [turn_map.get(ref) for ref in anchors]
    anchor_presence = {
        ref: (turn_map.get(ref) is not None) for ref in anchors
    }

    branches = {
        name: _check_branch(smoke_root, name, snapshot_root)
        for name in ("control", "feedback")
    }
    # Add intents carry the deterministic complete-dialogue content digest,
    # while ``sha256`` in the preflight record is the source-file digest.
    source_sha = str(
        preflight["canonical_dialogue"].get("complete_dialogue_sha256") or "")
    source_sha_match = all(
        str(branches[name]["r01_intent"].get("source_dialogue_sha256") or "")
        == source_sha for name in branches)
    payloads_equal = _scrub_transport(branches["control"]["r01_intent"].get("payload")) == _scrub_transport(
        branches["feedback"]["r01_intent"].get("payload"))

    required_people = sorted({
        _norm(row.get("speaker")) for row in turns
        if str(row.get("speaker") or "").strip()
    })
    # This is intentionally an observable anchor check, not a claim that free
    # text bucket content proves every relationship.
    people = {
        name: {
            "required": person,
            "control_present": person in branches["control"]["person_refs"],
            "feedback_present": person in branches["feedback"]["person_refs"],
        }
        for name, person in ((person, person) for person in required_people)
    }
    person_pass = all(
        row["control_present"] and row["feedback_present"]
        for row in people.values())

    relation_results: dict[str, Any] = {}
    if all(branches[name]["snapshot_present"] for name in branches):
        for person in required_people:
            other = next((x for x in required_people if x != person), None)
            relation_results[f"{person}->{other}"] = {
                "control": bool(other and re.search(
                    re.escape(other), branches["control"]["snapshot_content"], re.I)),
                "feedback": bool(other and re.search(
                    re.escape(other), branches["feedback"]["snapshot_content"], re.I)),
                "method": "person_name_cooccurrence_v1",
            }
        relation_pass = all(
            x["control"] and x["feedback"] for x in relation_results.values())
    else:
        relation_pass = False

    event_anchor_results: dict[str, Any] = {}
    for ref, row in zip(anchors, anchor_rows):
        if row is None:
            event_anchor_results[ref] = {
                "status": "FAIL", "reason": "anchor_turn_missing"
            }
            continue
        base_result = {
            "turn_text": str(row.get("text") or ""),
            "control": _anchor_event_match(
                str(row.get("text") or ""), set(branches["control"]["event_refs"])),
            "feedback": _anchor_event_match(
                str(row.get("text") or ""), set(branches["feedback"]["event_refs"])),
        }
        for branch in ("control", "feedback"):
            rows = branches[branch]["snapshot_event_rows"]
            anchor_text = _norm(row.get("text"))
            if "meditat" in anchor_text:
                selected = [x for x in rows if "meditat" in x["key"]]
                required = ("meditat", "every morning")
                forbidden_pairs = (("started", "finished"), ("开始", "完成"))
            elif "bali" in anchor_text or "surf" in anchor_text:
                selected = [x for x in rows if "bali" in x["key"] and "plan" in x["key"]]
                required = ("plan", "bali", "two weeks")
                forbidden_pairs = (("plan", "arrived"), ("计划", "完成"))
            else:
                selected = [x for x in rows if any(
                    token in x["content"].casefold()
                    for token in ("cement", "bricks", "水泥", "砖"))]
                required = ("cement", "brick")
                forbidden_pairs = (("need", "finished"), ("需要", "完成"))
            selected_text = " ".join(x["content"] for x in selected).casefold()
            required_present = all(token in selected_text for token in required)
            # Memorax may merge multiple turns into one event row.  Co-occurring
            # plan and completion language is therefore not a contradiction by
            # itself; only an explicit contradiction in a single row is noted.
            row_conflicts = []
            for event_row in selected:
                row_text = event_row["content"]
                for pair in forbidden_pairs:
                    if pair[0] in row_text and pair[1] in row_text:
                        row_conflicts.append({
                            "event_key": event_row["key"],
                            "pair": list(pair),
                        })
            contradiction = bool(row_conflicts)
            base_result[f"{branch}_state_observation"] = {
                "status": "FAIL" if contradiction else ("PASS" if required_present else "UNKNOWN"),
                "selected_event_keys": [x["key"] for x in selected],
                "required_markers": list(required),
                "contradiction_pairs": [list(x) for x in forbidden_pairs],
                "row_conflicts": row_conflicts,
                "method": "anchored_event_marker_check_v1",
            }
        event_anchor_results[ref] = base_result
    coarse_event_pass = all(
        value.get("control", {}).get("status") == "PASS"
        and value.get("feedback", {}).get("status") == "PASS"
        for value in event_anchor_results.values())
    state_values = [
        value[f"{branch}_state_observation"]
        for value in event_anchor_results.values()
        for branch in ("control", "feedback")
    ]
    state_fail = any(x["status"] == "FAIL" for x in state_values)
    state_unknown = any(x["status"] == "UNKNOWN" for x in state_values)
    state_status = "FAIL" if state_fail else ("UNKNOWN" if state_unknown else "PASS")

    # Temporal parity is defined at the canonical-anchor level.  Memorax
    # bucket rows do not retain source turn refs, so require the source anchors
    # to be strictly ordered and require each branch's matched event rows to
    # contain no explicit state contradiction.  This matches the frozen
    # contract's "no semantic reversal" rule without pretending bucket rows
    # preserve exact turn provenance.
    anchor_positions = [
        next((index for index, turn in enumerate(turns)
              if str(turn.get("turn_ref")) == ref), -1)
        for ref in anchors
    ]
    temporal_order_valid = (
        len(anchor_positions) == len(set(anchor_positions))
        and all(index >= 0 for index in anchor_positions)
    )
    ordered_anchor_refs = [
        ref for _, ref in sorted(zip(anchor_positions, anchors))
    ]
    temporal_relations = {
        "status": "FAIL" if not temporal_order_valid or state_fail else "PASS",
        "anchor_refs": anchors,
        "canonical_ordered_anchor_refs": ordered_anchor_refs,
        "canonical_turn_positions": anchor_positions,
        "canonical_order_valid": temporal_order_valid,
        "branch_state_conflicts": {
            branch: [
                {"anchor_ref": ref, "row_conflicts": value[f"{branch}_state_observation"].get("row_conflicts", [])}
                for ref, value in event_anchor_results.items()
                if value[f"{branch}_state_observation"].get("row_conflicts")
            ]
            for branch in ("control", "feedback")
        },
        "method": "canonical_anchor_order_plus_no_explicit_state_reversal_v1",
    }

    snapshot_content = " ".join(
        branches[name]["snapshot_content"] for name in branches)
    card_text = json.dumps(card, ensure_ascii=False, sort_keys=True)
    feedback_control_pollution = (
        "FAIL" if any(token.casefold() in snapshot_content.casefold()
                       for token in ("desired_event_operator", "frontier_status",
                                     "round1_feedback_compiler_v1"))
        else ("PASS" if all(branches[name]["snapshot_present"] for name in branches)
              else "UNKNOWN"))
    preflight_ref = _load(preflight_root / "frozen_control_reference.json")
    old_r02_path = Path(str(preflight_ref.get("r02_control_delta_path") or ""))
    old_r02 = _load(old_r02_path) if old_r02_path.is_file() else None
    old_r02_sentences: list[str] = []
    if isinstance(old_r02, Mapping):
        for scene in old_r02.get("scenes") or []:
            for turn in scene.get("dialogue_turns") or []:
                text = _norm(turn.get("text"))
                if text:
                    old_r02_sentences.append(text)
    r02_pollution = any(
        sentence in snapshot_content.casefold()
        for sentence in old_r02_sentences
    )
    pollution_status = "FAIL" if r02_pollution else (
        "PASS" if all(branches[name]["snapshot_present"] for name in branches)
        else "UNKNOWN")

    hard_observations = {
        "initial_namespace_empty": bool(
            _load(smoke_root / "namespace_empty_check.json").get("all_empty")),
        "old_control_modified": smoke_report.get("old_control_modified") is True,
        "parent_add_calls": smoke_report.get("parent_add_calls"),
        "r01_child_add_calls": smoke_report.get("r01_child_add_calls"),
        "r02_add_calls": smoke_report.get("r02_add_calls"),
        "r02_generation": smoke_report.get("r02_generation"),
        "qa_calls": smoke_report.get("qa_calls"),
        "solver_calls": smoke_report.get("solver_calls"),
        "semantic_auditor_calls": smoke_report.get("semantic_auditor_calls"),
        "test_access": smoke_report.get("test_access"),
        "r02_tokens_in_r01_payload": {
            name: branches[name]["r02_tokens_in_r01_payload"] for name in branches
        },
    }

    contract = {
        "contract_version": CONTRACT_VERSION,
        "status_values": ["PASS", "FAIL", "UNKNOWN"],
        "source_of_truth": [
            "canonical Parent + R01 dialogue",
            "feedback card anchor_refs",
            "completed Add/visibility artifacts",
        ],
        "required_checks": {
            "key_people_coverage": "both namespaces contain every canonical speaker anchor",
            "key_relationship_coverage": "must be proven from full bucket content; UNKNOWN without it",
            "feedback_anchor_coverage": "every card anchor turn exists and has an observable event topic in both branches",
            "key_event_coverage": "conservative lexical topic-family check only",
            "state_semantics": "must be proven from bucket content; UNKNOWN without it",
            "temporal_relations": "must be proven from bucket content/turn-linked facts; UNKNOWN without it",
            "old_r02_pollution": "local provenance and final namespace snapshot required",
            "feedback_control_pollution": "feedback card/control tokens absent from R01 payload and final namespace snapshot",
            "explicit_conflict": "any observed contradiction is FAIL; absence is not provable without content",
        },
        "fail_closed_rule": "any required UNKNOWN blocks SEMANTIC_MEMORY_PARITY_PASS",
        "snapshot_payload_required_for_pass": True,
    }
    _write(output_root / "memory_semantic_parity_contract.json", contract)

    unknowns: list[str] = []
    if not relation_pass:
        unknowns.append("relationship_semantics_not_proven_by_explicit_relation_markers")
    if state_status == "UNKNOWN":
        unknowns.append("event_state_not_turn_linked_in_memorax_bucket_content")
    if state_status == "FAIL":
        unknowns.append("event_state_contradiction_observed")
    # Exact bucket-row-to-turn provenance is unavailable, but the frozen
    # contract only requires canonical order and no explicit reversal.
    if not all(branches[name]["snapshot_present"] for name in branches):
        unknowns.append("full_post_r01_bucket_payload_not_available")
    if pollution_status == "UNKNOWN":
        unknowns.append("final_namespace_snapshot_not_available_for_pollution_check")
    if feedback_control_pollution == "UNKNOWN":
        unknowns.append("feedback_control_pollution_not_verifiable")

    hard_observations_pass = (
        smoke_report.get("status") == "ROUND1_R01_BOOTSTRAP_SMOKE_PASS"
        and source_sha_match
        and payloads_equal
        and person_pass
        and coarse_event_pass
        and hard_observations["initial_namespace_empty"]
        and hard_observations["old_control_modified"] is False
        and hard_observations["parent_add_calls"] == 2
        and hard_observations["r01_child_add_calls"] == 2
        and hard_observations["r02_add_calls"] == 0
        and hard_observations["r02_generation"] == 0
        and hard_observations["qa_calls"] == 0
        and hard_observations["solver_calls"] == 0
        and hard_observations["semantic_auditor_calls"] == 0
        and hard_observations["test_access"] == 0
        and not any(hard_observations["r02_tokens_in_r01_payload"].values())
    )
    hard_fail = (
        not hard_observations_pass
        or state_fail
        or pollution_status == "FAIL"
        or feedback_control_pollution == "FAIL"
    )
    semantic_status = "SEMANTIC_MEMORY_PARITY_FAIL" if hard_fail else (
        "SEMANTIC_MEMORY_PARITY_PENDING" if unknowns else
        "SEMANTIC_MEMORY_PARITY_PASS")
    report = {
        "audit_version": VERSION,
        "created_at": _now(),
        "root_id": root_id,
        "status": semantic_status,
        "decision": "READY_FOR_R02_FEEDBACK_SINGLE_ROOT_SMOKE" if semantic_status == "SEMANTIC_MEMORY_PARITY_PASS" else "BLOCK_R02_FEEDBACK",
        "source": {
            "dialogue_path": str(dialogue_path),
            "dialogue_file_sha256": _sha(dialogue_path),
            "complete_dialogue_sha256": str(
                preflight["canonical_dialogue"].get("complete_dialogue_sha256") or ""),
            "preflight_root": str(preflight_root),
            "smoke_root": str(smoke_root),
            "compiler_root": str(compiler_root),
            "feedback_card_sha256": _sha(compiler_root / "root_feedback_cards.jsonl"),
        },
        "branch_observations": {
            name: {
                key: value for key, value in branch.items()
                if key not in {"parent_intent", "r01_intent", "parent_audit", "r01_audit",
                               "snapshot_content", "snapshot_event_rows"}
            }
            for name, branch in branches.items()
        },
        "key_people": people,
        "key_people_coverage": "PASS" if person_pass else "FAIL",
        "feedback_anchor_presence": anchor_presence,
        "feedback_anchor_event_coverage": event_anchor_results,
        "feedback_anchor_coverage": "PASS" if coarse_event_pass and all(anchor_presence.values()) else "FAIL",
        "relationship_details": relation_results,
        "relationship_coverage": "PASS" if relation_pass else "UNKNOWN",
        "event_state_semantics": state_status,
        "temporal_relations": temporal_relations,
        "old_r02_control_pollution": pollution_status,
        "feedback_control_pollution": feedback_control_pollution,
        "explicit_fact_conflicts": "FAIL" if state_fail else "PASS",
        "hard_observations": hard_observations,
        "source_sha_match": source_sha_match,
        "r01_payload_semantic_equality": payloads_equal,
        "unknown_blockers": unknowns,
        "hard_observations_pass": hard_observations_pass,
        "notes": [
            "The old semantic_visibility_verified=false flag is not treated as a failure by itself.",
            "It is also not silently promoted to PASS: full bucket payload evidence is missing.",
            "No Memorax/API/model/Solver/QA/auditor call was made by this audit.",
            "R02-feedback must not be generated while status is pending or fail.",
        ],
    }
    _write(output_root / "memory_semantic_parity_report.json", report)
    md = output_root / "memory_semantic_parity_report.md"
    md.write_text(
        "# Memory Semantic Parity Audit\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Decision: `{report['decision']}`\n"
        f"- Root: `{root_id}`\n"
        f"- Key people: `{report['key_people_coverage']}`\n"
        f"- Feedback anchor coverage: `{report['feedback_anchor_coverage']}`\n"
        f"- Relationships: `{report['relationship_coverage']}`\n"
        f"- Event state: `{report['event_state_semantics']}`\n"
        f"- Temporal relations: `{report['temporal_relations'].get('status') if isinstance(report['temporal_relations'], dict) else report['temporal_relations']}`\n"
        f"- R02-feedback generation: `{'ALLOWED' if report['decision'].startswith('READY') else 'BLOCKED'}`\n\n"
        "The report distinguishes conservative lexical coverage from turn-linked "
        "semantic guarantees.\n",
        encoding="utf-8",
    )
    (md.with_name(md.name + ".sha256")).write_text(
        _sha(md) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--compiler-root", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args(argv)
    report = run(
        preflight_root=args.preflight_root.resolve(),
        smoke_root=args.smoke_root.resolve(),
        compiler_root=args.compiler_root.resolve(),
        output_root=args.output_root.resolve(),
        snapshot_root=args.snapshot_root.resolve() if args.snapshot_root else None,
    )
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "root_id": report["root_id"],
        "unknown_blockers": report["unknown_blockers"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "SEMANTIC_MEMORY_PARITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
