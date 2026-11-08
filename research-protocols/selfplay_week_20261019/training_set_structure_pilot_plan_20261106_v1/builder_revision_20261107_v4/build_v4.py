#!/usr/bin/env python3
"""Build the frozen, anonymous input layer for structure-pilot v2.

This command is deliberately limited to controller-side preparation.  It
copies and hashes the eight already-admitted canonical sources, makes an
anonymous turn projection, writes role/selector packet *templates*, and
creates the public/restricted schedules.  It never calls a model, creates a
slot, creates a candidate question, or changes an existing execution.

The output directory is created with exclusive-write semantics.  Re-running
against a populated execution is refused unless ``--replace-empty`` is used
and every existing file is zero bytes (normally a typo, so it should not be
needed).  This keeps a v2 execution auditable and prevents silent mutation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_V2 = Path(__file__).resolve().parents[1] / "execution_20261107_v4"
V2 = FINAL_V2
PLAN = FINAL_V2.parent
V1_BINDINGS = PLAN / "INPUT_BINDINGS.json"
HIST_DIR = PLAN.parent / "training_set_rebuild_20261105_FG_exploration_run01" / "private_semantic_reference"
FEEDBACK_PATH = PLAN.parent / "training_set_rebuild_20261105_FG_exploration_run01" / "ABSTRACT_FEEDBACK_F.json"
HIST_COVERAGE = PLAN / "execution_20261106_v1" / "manifests" / "HISTORICAL_COVERAGE.json"
SCHEDULE_SEED = 2026110707
EXPECTED_ROOTS = [
    "cc-train-episode-138343", "cc-train-episode-14304",
    "cc-train-episode-162896", "cc-train-episode-191749",
    "cc-train-episode-225575", "cc-train-episode-25652",
    "cc-train-episode-75241", "cc-train-episode-77882",
]
SYNTHETIC_ROOTS = {
    "cc-train-episode-25652",
    "cc-train-episode-75241",
    "cc-train-episode-77882",
}
SYNTHETIC_TURN_RE = re.compile(
    r"R03:(A|B):(S0[1-3]):T(00[1-8])\Z"
)
GROUPS = ("OLD_F", "OLD_G", "NEW_F", "NEW_G")
ROOT_GROUPS = ("R02", "R03")


class BuildError(RuntimeError):
    pass


def strict_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_pairs,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    return obj


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encode_json(obj)
    path.write_bytes(raw)
    return digest(raw)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(V2.resolve()))


def load_bindings() -> list[dict[str, Any]]:
    data = read_json(V1_BINDINGS)
    rows = data.get("roots")
    if not isinstance(rows, list):
        raise BuildError("INPUT_BINDINGS.json has no roots list")
    if len(rows) != len(EXPECTED_ROOTS):
        raise BuildError(
            f"binding root whitelist requires exactly {len(EXPECTED_ROOTS)} rows; got {len(rows)}"
        )
    if any(not isinstance(x, dict) or not isinstance(x.get("root_id"), str) for x in rows):
        raise BuildError("binding rows must be objects with string root_id")
    ids = [x["root_id"] for x in rows]
    if len(set(ids)) != len(ids):
        raise BuildError("duplicate canonical binding root_id")
    unexpected = sorted(set(ids) - set(EXPECTED_ROOTS))
    if unexpected:
        raise BuildError(f"binding root outside whitelist: {unexpected}")
    by_root = {x["root_id"]: x for x in rows}
    missing = [x for x in EXPECTED_ROOTS if x not in by_root]
    if missing:
        raise BuildError(f"missing canonical bindings: {missing}")
    return [by_root[x] for x in EXPECTED_ROOTS]


def turn_projection(source: dict[str, Any], root: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate canonical turns and create a strictly anonymous projection.

    Synthetic R03 extensions are admitted only for the three explicitly
    approved roots and only as the complete A/B x S01-S03 x T001-T008 grid.
    Their internal origin and turn references remain controller-private.
    """
    turns = source.get("turns")
    if not isinstance(turns, list) or not turns:
        raise BuildError(f"canonical source has no non-empty turns list: {root}")
    if root not in EXPECTED_ROOTS:
        raise BuildError(f"canonical root outside whitelist: {root}")
    sessions: dict[str, str] = {}
    counters: dict[str, int] = {}
    out: list[dict[str, str]] = []
    synthetic_rows: list[dict[str, str]] = []
    seen_turn_refs: set[str] = set()
    synthetic_refs: list[str] = []
    for item in turns:
        if not isinstance(item, dict):
            raise BuildError(f"turn entry is not an object: {root}")
        for key in ("turn_ref", "speaker", "text", "origin"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise BuildError(f"turn missing non-empty string {key}: {root}")
        turn_ref = item["turn_ref"]
        if turn_ref in seen_turn_refs:
            raise BuildError(f"duplicate canonical turn_ref {turn_ref}: {root}")
        seen_turn_refs.add(turn_ref)
        origin = item["origin"]
        is_synthetic = origin == "codex_synthetic_extension"
        session = item.get("session_ref")
        unbound_synthetic = is_synthetic and (
            not isinstance(session, str) or not session.strip()
        )
        match = SYNTHETIC_TURN_RE.fullmatch(turn_ref) if unbound_synthetic else None
        if unbound_synthetic:
            if root not in SYNTHETIC_ROOTS:
                raise BuildError(f"unbound synthetic extension on non-approved root: {root}")
            if match is None:
                raise BuildError(f"malformed unbound synthetic turn_ref: {turn_ref}")
            route, scene, ordinal = match.groups()
            synthetic_scene = f"R03:{route}:{scene}"
            synthetic_refs.append(turn_ref)
            session_key = f"__synthetic_segment__:{synthetic_scene}"
            synthetic_rows.append({
                "turn_ref": turn_ref,
                "synthetic_segment": synthetic_scene,
                "session_ref_present": "session_ref" in item,
            })
        else:
            if root in SYNTHETIC_ROOTS and turn_ref.startswith("R03:"):
                # R03 refs are reserved for the admitted synthetic grid.
                if is_synthetic:
                    raise BuildError(f"bound synthetic R03 turn_ref is not allowed: {turn_ref}")
            if not isinstance(session, str) or not session.strip():
                raise BuildError(f"turn missing non-empty session_ref: {turn_ref}")
            session_key = session.strip()
        if session_key not in sessions:
            sessions[session_key] = f"S{len(sessions) + 1:02d}"
            counters[session_key] = 0
        counters[session_key] += 1
        out.append({
            "turn_ref": f"{sessions[session_key]}:T{counters[session_key]:03d}",
            "speaker": item["speaker"],
            "text": item["text"],
        })

    expected_synthetic = [
        f"R03:{route}:S{scene:02d}:T{turn:03d}"
        for route in ("A", "B")
        for scene in range(1, 4)
        for turn in range(1, 9)
    ]
    if root in SYNTHETIC_ROOTS:
        if synthetic_refs != expected_synthetic:
            raise BuildError(
                f"synthetic extension must be exactly ordered six segments x eight turns for {root}"
            )
    elif synthetic_refs:
        raise BuildError(f"synthetic extension present on non-synthetic root: {root}")

    projection = {
        "schema_version": "anonymous_source_projection_v3",
        "projection_policy": "anonymous dialogue projection; local session references follow first appearance",
        "turn_count": len(out),
        "session_count": len(sessions),
        "turns": out,
    }
    neutral = {
        "schema_version": "anonymous_neutral_projection_v3",
        "turn_count": len(out),
        "session_count": len(sessions),
        "complete_dialogue_preserved": True,
        "evidence_policy": "Match every citation to the exact anonymous session, speaker and text. Preserve reported, planned, suggested and completed distinctions; do not infer relations from mention order.",
    }
    private = {
        "synthetic_turn_count": len(synthetic_rows),
        "derived_count": len(synthetic_rows),
        "synthetic_turns": synthetic_rows,
        "synthetic_grid_validated": root in SYNTHETIC_ROOTS,
        "provenance_claim": False,
    }
    return projection, neutral, private

def metadata_audit(source: dict[str, Any], actual_turn_count: int) -> list[dict[str, Any]]:
    issues = []
    if source.get("baseline_turn_count") is not None and source.get("new_turn_count") is not None and source.get("cumulative_new_turn_count") is not None:
        b, n, c = source.get("baseline_turn_count"), source.get("new_turn_count"), source.get("cumulative_new_turn_count")
        if isinstance(b, int) and isinstance(n, int) and isinstance(c, int) and b + c != actual_turn_count:
            issues.append({"code": "TURN_METADATA_FORMULA_DISCREPANCY", "message": "baseline_turn_count + cumulative_new_turn_count differs from actual turns length", "baseline": b, "cumulative_new": c, "actual": actual_turn_count})
    if source.get("previous_turn_count") is not None and source.get("new_turn_count") is not None:
        p, n = source.get("previous_turn_count"), source.get("new_turn_count")
        if isinstance(p, int) and isinstance(n, int) and p + n != actual_turn_count:
            issues.append({"code": "TURN_METADATA_PREVIOUS_FORMULA_DISCREPANCY", "message": "previous_turn_count + new_turn_count differs from actual turns length", "previous": p, "new": n, "actual": actual_turn_count})
    if source.get("baseline_turn_count") is not None and source.get("baseline_turn_count") + (source.get("cumulative_new_turn_count") or 0) != actual_turn_count:
        # Keep a separate marker for sources where generated extensions are not
        # reflected in legacy counters.  It is a recorded admission warning,
        # never an implicit correction to the canonical bytes.
        if not any(x["code"] == "TURN_METADATA_FORMULA_DISCREPANCY" for x in issues):
            issues.append({"code": "TURN_METADATA_DISCREPANCY", "message": "legacy turn counters do not describe canonical turn list"})
    return issues


def encode_json(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")


def source_rows(bindings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    """Validate every canonical source before returning any writeable artifact."""
    rng = random.Random(SCHEDULE_SEED)
    shuffled = EXPECTED_ROOTS.copy()
    rng.shuffle(shuffled)
    root_to_src = {root: f"SRC_{i+1:02d}" for i, root in enumerate(shuffled)}
    root_to_root_code = {root: f"R{i+1:02d}" for i, root in enumerate(shuffled)}
    rows: list[dict[str, Any]] = []
    private: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    artifacts: dict[str, bytes] = {}
    by_root = {b["root_id"]: b for b in bindings}
    for root in EXPECTED_ROOTS:
        b = by_root[root]
        src_path = Path(b["canonical_source"]["path"])
        if not src_path.exists():
            raise BuildError(f"canonical source missing: {src_path}")
        raw = src_path.read_bytes()
        actual_hash = digest(raw)
        expected_hash = b["canonical_source"].get("sha256")
        if actual_hash != expected_hash:
            raise BuildError(f"canonical hash mismatch for {root}: {actual_hash} != {expected_hash}")
        expected_bytes = b["canonical_source"].get("bytes")
        if not isinstance(expected_bytes, int) or len(raw) != expected_bytes:
            raise BuildError(f"canonical byte-size mismatch for {root}: {len(raw)} != {expected_bytes}")
        source = read_json(src_path)
        if source.get("root_id") != root:
            raise BuildError(f"canonical root_id mismatch for {root}")
        actual_count = len(source.get("turns", [])) if isinstance(source.get("turns"), list) else -1
        expected_count = b.get("turn_count", actual_count)
        if not isinstance(expected_count, int) or actual_count != expected_count:
            raise BuildError(f"binding turn_count mismatch for {root}: {actual_count} != {expected_count}")
        projection, neutral, derivation = turn_projection(source, root)
        source_code = root_to_src[root]
        source_rel = f"inputs/anonymous_source/{source_code}/SOURCE.json"
        neutral_rel = f"inputs/anonymous_source/{source_code}/NEUTRAL.json"
        source_raw = encode_json(projection)
        neutral_raw = encode_json(neutral)
        artifacts[source_rel] = source_raw
        artifacts[neutral_rel] = neutral_raw
        source_hash = digest(source_raw)
        neutral_hash = digest(neutral_raw)
        audit = metadata_audit(source, actual_count)
        if derivation["synthetic_turn_count"]:
            audit.append({
                "code": "SESSION_REF_DERIVED_SYNTHETIC_SEGMENT",
                "message": "canonical extension turns omit session_ref; structured segment boundaries are used without claiming original session provenance",
                "derived_count": derivation["synthetic_turn_count"],
                "provenance_claim": False,
            })
        if audit:
            warnings.append({"source_code": source_code, "root_code": root_to_root_code[root], "root_binding_private": True, "issues": audit})
        private[root] = {
            "source_code": source_code,
            "root_code": root_to_root_code[root],
            "canonical_path": str(src_path),
            "canonical_sha256": actual_hash,
            "canonical_bytes": len(raw),
            "turn_count": actual_count,
            "source_round": b.get("source_round"),
            "metadata_audit": audit,
            "session_ref_derivation": derivation,
            "projection_path": source_rel,
            "projection_sha256": source_hash,
            "neutral_path": neutral_rel,
            "neutral_sha256": neutral_hash,
        }
        rows.append({"source_code": source_code, "turn_count": actual_count, "source_sha256": source_hash, "neutral_sha256": neutral_hash, "source_path": source_rel, "neutral_path": neutral_rel})
    return rows, private, warnings, artifacts

def historical_projection() -> dict[str, Any]:
    # The frozen inventory is controller-readable only.  Its root/candidate/path
    # identifiers are intentionally discarded before the projection reaches any
    # role.  Keep the full 966-row denominator, including unknown/pending rows.
    records = []
    if not HIST_COVERAGE.exists():
        raise BuildError(f"historical coverage manifest missing: {HIST_COVERAGE}")
    inventory = read_json(HIST_COVERAGE)
    for i, item in enumerate(inventory.get("records", []), 1):
        if not isinstance(item, dict):
            continue
        qtype = item.get("question_type") if isinstance(item.get("question_type"), str) else "unknown"
        op = item.get("task_operation") or item.get("operation_family_provisional") or "unknown_pending"
        records.append({"history_item_id": f"H{i:04d}", "question_type": qtype,
                        "answer_operation": op if isinstance(op, str) else "unknown_pending",
                        "operation_status": item.get("operation_coding_status") if isinstance(item.get("operation_coding_status"), str) else "UNKNOWN_PENDING"})
    from collections import Counter
    type_counts = Counter(x["question_type"] for x in records)
    op_counts = Counter(x["answer_operation"] for x in records)
    return {
        "schema_version": "anonymous_historical_coverage_v2",
        "status": "PARTIAL_UNKNOWN_PENDING",
        "projection_policy": "Removed root, arm, candidate, question, option, answer, score, solver and path identity fields; UNKNOWN is retained where independent quality/operation review is absent.",
        "record_count": len(records),
        "unknown_count": sum(1 for x in records if "UNKNOWN" in x["operation_status"] or "PENDING" in x["operation_status"]),
        "question_type_counts": dict(sorted(type_counts.items())),
        "operation_counts": dict(sorted(op_counts.items())),
        "records": records,
    }


def feedback_projection() -> dict[str, Any]:
    """Return the abstract F observations without their experiment identity.

    The author may use these observations to discover a natural information
    need, but must not learn the source experiment, checkpoint, arm or card
    identifier.  This is deliberately a small allow-list projection instead
    of a recursive redaction, so a future feedback-card schema cannot leak a
    new identity field by accident.
    """
    if not FEEDBACK_PATH.exists():
        raise BuildError(f"feedback card missing: {FEEDBACK_PATH}")
    raw = read_json(FEEDBACK_PATH)
    if not isinstance(raw, dict):
        raise BuildError("feedback card must be an object")
    fields = {
        "abstract_search_hypotheses",
        "batch_review_assessment",
        "alternative_explanations",
        "applicability",
        "do_not_force",
        "exploration_direction",
    }
    card = {key: copy.deepcopy(raw[key]) for key in fields if key in raw}
    if set(card) != fields:
        raise BuildError(f"feedback card missing projected fields: {sorted(fields - set(card))}")
    return {
        "schema_version": "anonymous_context_card_v2",
        "purpose": "abstract discovery guidance only; it does not authorize generation or scoring",
        "identity_removed": True,
        "observations": card,
        "no_private_case_wording_options_gold_blind_or_solver_answers": True,
    }


def contracts() -> tuple[str, str, dict[str, Any]]:
    author = """AUTHOR CONTRACT V2 (ANONYMOUS ROLE)\n\nInput is limited to the anonymous SOURCE.json, NEUTRAL.json, this contract, and (for F only) the supplied anonymous context card. Do not access controller_private, canonical paths, historical raw records, other packets, solver/training artifacts, or previous executions.\n\nThe packet may expose only workflow OLD/NEW and an anonymous opportunity code. Never request or emit root, group, arm, candidate, solver, history, path, or gold identifiers. OLD produces at most one candidate. NEW first produces a NEW_SLOTS.json and SLOTS_FREEZE.json; implementation occurs only after an independent selector result. No retries, fallback, or second candidate are allowed.\n\nGeneration is authorized only when the controller authorization artifact says AUTHOR_ONLY; the current v2 freeze is NONE.\n"""
    selector = """SELECTOR CONTRACT V2 (INDEPENDENT ANONYMOUS ROLE)\n\nRead only the anonymous SOURCE.json, NEUTRAL.json, NEW_SLOTS.json when present, and anonymous historical coverage projection. Do not access author packets, workflow, F/G, root, group, arm, candidate, solver, path, gold, or feedback identity.\n\nReturn exactly one JSON object with exactly these keys: selected_slot_id (string or null), status (SELECTED, NO_SAFE_QA, or FORM_UNSUPPORTED), reason (concise string), input_hashes (object with source_sha256, neutral_sha256, slots_sha256, history_sha256). No extra keys.\n\nNO_SAFE_QA, FORM_UNSUPPORTED, malformed input, and access violations are terminal. Do not retry or fallback.\n"""
    metrics = {
        "schema_version": "pre_registered_metrics_v2",
        "fixed_denominators": {"opportunities": 32, "roots": 8, "groups": ["OLD_F", "OLD_G", "NEW_F", "NEW_G"]},
        "metrics": ["opportunity_count", "slot_count", "selected_count", "first_draft_count", "quality_pass_count", "history_deduplicated_valid_N", "non_D1_D4_structural_count_S", "root_coverage", "answer_operation_coverage", "protocol_failure_rate", "valid_yield", "quality_pass_rate", "history_dedup_rate", "structural_increment", "token_context_time_cost"],
        "missing_usage_policy": "Unavailable usage is null; never impute.",
        "advancement_gate": {"N_NEW_min": 10, "root_coverage_NEW_min": 6, "N_NEW_ge_N_OLD": True, "S_NEW_minus_S_OLD_min": 4, "improved_roots_min": 3, "distinct_operation_or_relation_patterns_min": 2, "protocol_contamination_allowed": False},
    }
    return author, selector, metrics


def _build_into() -> None:
    bindings = load_bindings()
    rows, private_rows, warnings, source_artifacts = source_rows(bindings)
    hist = historical_projection()
    hist_path = V2 / "inputs" / "anonymous_history" / "HISTORICAL_COVERAGE_ANON.json"
    hist_hash = dump(hist_path, hist)
    context = feedback_projection()
    context_path = V2 / "inputs" / "anonymous_context" / "CONTEXT_CARD.json"
    context_hash = dump(context_path, context)
    for relative, raw in source_artifacts.items():
        path = V2 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    author, selector, metrics = contracts()
    (V2 / "contracts").mkdir(parents=True, exist_ok=True)
    author_path = V2 / "contracts" / "AUTHOR_CONTRACT_v2.txt"
    selector_path = V2 / "contracts" / "SELECTOR_CONTRACT_v2.txt"
    author_path.write_text(author, encoding="utf-8")
    selector_path.write_text(selector, encoding="utf-8")
    author_hash = digest(author_path.read_bytes())
    selector_hash = digest(selector_path.read_bytes())
    dump(V2 / "contracts" / "PRE_REGISTERED_METRICS_v2.json", metrics)
    # Explicitly separate private controller bindings from all role-readable files.
    dump(V2 / "PRIVATE_BINDINGS.json", {"schema_version": "private_bindings_v2", "visibility": "controller_only", "schedule_seed": SCHEDULE_SEED, "sources": private_rows})
    dump(V2 / "SOURCE_ADMISSION_AUDIT.json", {"schema_version": "source_admission_audit_v2", "status": "PASS_WITH_EXPLICIT_METADATA_WARNINGS" if warnings else "PASS", "warnings": warnings, "actual_turns_are_authoritative": True})
    dump(V2 / "PROTOCOL_FREEZE.json", {"schema_version": "execution_freeze_v2", "execution_id": "execution_20261107_v2", "status": "FROZEN_PRE_AUTHOR", "created_utc": datetime.now(timezone.utc).isoformat(), "generation_authorization": "NONE", "solver_training_export": "FORBIDDEN", "source_projection": "anonymous_only", "schedule_seed": SCHEDULE_SEED})
    auth_payload = {"schema_version": "authorization_v2", "generation_authorization": "NONE", "scope": {"author": False, "selector": False, "solver": False, "training": False, "export": False, "retry": False, "fallback": False}, "reason": "Awaiting Stage-A and Stage-B preflight; this builder grants no execution permission."}
    dump(V2 / "AUTHORIZATION.json", auth_payload)
    dump(V2 / "AUTHOR_LAUNCH_AUTHORIZATION.json", {"schema_version": "author_launch_authorization_v2", "generation_authorization": "NONE", "author_launch_authorized": False, "scope": auth_payload["scope"], "reason": "Preflight has not granted an author launch."})
    dump(V2 / "ACCESS_RULES.json", {"schema_version": "access_rules_v2", "current_generation_authorization": "NONE", "role_readable_roots": ["inputs/anonymous_source", "inputs/anonymous_history", "role_packets", "selector_packets", "contracts"], "controller_only_roots": ["PRIVATE_BINDINGS.json", "PRIVATE_OPPORTUNITY_MAPPING.json", "SELECTOR_OPERATION_SUMMARY.json", "AUTHOR_LAUNCH_AUTHORIZATION.json", "controller_private", "SOURCE_ADMISSION_AUDIT.json"], "forbidden_tokens": ["root_id", "canonical_path", "source_id", "scene_ref", "group_id", "arm", "candidate_id", "solver", "gold", "history_question", "absolute_path"]})
    dump(V2 / "INPUT_MANIFEST.json", {"schema_version": "anonymous_input_manifest_v2", "status": "FROZEN_ANONYMOUS_PROJECTIONS", "sources": rows, "historical_coverage": {"path": rel(hist_path), "sha256": hist_hash, "record_count": hist["record_count"], "unknown_count": hist["unknown_count"]}, "context_card": {"path": rel(context_path), "sha256": context_hash}})
    public = []
    restricted = []
    rng = random.Random(SCHEDULE_SEED + 1)
    opps = []
    n = 0
    for source_code in [x["source_code"] for x in rows]:
        for group in GROUPS:
            n += 1
            code = f"OPP_{n:03d}"
            workflow = "OLD" if group.startswith("OLD") else "NEW"
            feedback = "CONTEXT_CARD" if group.endswith("F") else "NONE"
            opps.append((code, source_code, group, workflow, feedback))
    rng.shuffle(opps)
    for code, source_code, group, workflow, feedback in opps:
        role_dir = V2 / "role_packets" / code
        role_dir.mkdir(parents=True, exist_ok=True)
        selector_dir = V2 / "selector_packets" / code
        if workflow == "NEW":
            selector_dir.mkdir(parents=True, exist_ok=True)
        src = next(x for x in rows if x["source_code"] == source_code)
        role = {"schema_version": "anonymous_author_packet_v2", "role": "author", "opportunity_code": code, "workflow": workflow, "source": {"path": src["source_path"], "sha256": src["source_sha256"], "turn_count": src["turn_count"]}, "neutral": {"path": src["neutral_path"], "sha256": src["neutral_sha256"]}, "contract": {"path": rel(author_path), "sha256": author_hash}, "output": {"directory": f"author_output/{code}"}, "opportunity_ceiling": 1, "slots_ceiling": 3 if workflow == "NEW" else 0, "generation_authorization": "NONE"}
        if feedback == "CONTEXT_CARD":
            role["context_card"] = {"path": rel(context_path), "sha256": context_hash}
        dump(role_dir / "INPUT.json", role)
        # Selector packets intentionally carry no opportunity/arm/workflow
        # label.  The packet directory is an anonymous transport boundary;
        # the selector only sees the data needed for coverage selection.
        if workflow == "NEW":
            selector = {"schema_version": "anonymous_selector_packet_v2", "role": "selector", "source": {"path": src["source_path"], "sha256": src["source_sha256"]}, "neutral": {"path": src["neutral_path"], "sha256": src["neutral_sha256"]}, "history": {"path": rel(hist_path), "sha256": hist_hash}, "slots": {"path": "NEW_SLOTS.json", "sha256": None}, "contract": {"path": rel(selector_path), "sha256": selector_hash}, "generation_authorization": "NONE", "expected_result_schema": {"selected_slot_id": "string|null", "status": ["SELECTED", "NO_SAFE_QA", "FORM_UNSUPPORTED"], "reason": "string", "input_hashes": {"source_sha256": "hex64", "neutral_sha256": "hex64", "slots_sha256": "hex64", "history_sha256": "hex64"}}}
            dump(selector_dir / "INPUT.json", selector)
        public.append({"opportunity_code": code})
        restricted.append({"opportunity_code": code, "source_code": source_code, "group": group, "workflow": workflow, "feedback": feedback})
    dump(V2 / "SCHEDULE_PUBLIC.json", {"schema_version": "public_schedule_v2", "generation_authorization": "NONE", "opportunities": public})
    dump(V2 / "SCHEDULE_RESTRICTED.json", {"schema_version": "restricted_schedule_v2", "visibility": "controller_only", "generation_authorization": "NONE", "opportunities": restricted})
    dump(V2 / "PRIVATE_OPPORTUNITY_MAPPING.json", {"schema_version": "private_opportunity_mapping_v2", "visibility": "controller_only", "opportunities": [{"opportunity_code": code, "source_code": source_code, "group": group, "workflow": workflow, "feedback": feedback} for code, source_code, group, workflow, feedback in opps]})
    operation_counts: dict[str, int] = {}
    for record in hist.get("records", []):
        operation = record.get("answer_operation")
        if isinstance(operation, str):
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
    dump(V2 / "SELECTOR_OPERATION_SUMMARY.json", {"schema_version": "anonymous_selector_operation_summary_v2", "visibility": "controller_only", "status": "PREPARED_NOT_STARTED", "generation_authorization": "NONE", "selector_opportunity_count": sum(1 for x in opps if x[3] == "NEW"), "historical_coverage_sha256": hist_hash, "historical_record_count": hist["record_count"], "historical_unknown_count": hist["unknown_count"], "answer_operation_counts": operation_counts, "allowed_result_keys": ["selected_slot_id", "status", "reason", "input_hashes"], "result_count": 0})
    packet_rows = []
    for code, source_code, group, workflow, feedback in opps:
        role_input = V2 / "role_packets" / code / "INPUT.json"
        selector_input = V2 / "selector_packets" / code / "INPUT.json"
        item = {"opportunity_code": code, "source_code": source_code, "group": group, "workflow": workflow, "role_input_path": rel(role_input), "role_input_sha256": digest(role_input.read_bytes())}
        if workflow == "NEW":
            item.update({"selector_input_path": rel(selector_input), "selector_input_sha256": digest(selector_input.read_bytes())})
        packet_rows.append(item)
    dump(V2 / "PACKET_MANIFEST.json", {"schema_version": "anonymous_packet_manifest_v2", "status": "PACKETS_PREPARED_NOT_LAUNCHED", "generation_authorization": "NONE", "packet_count": len(packet_rows), "selector_packet_count": sum(1 for item in packet_rows if "selector_input_path" in item), "packets": packet_rows})
    dump(V2 / "BUILD_MANIFEST.json", {"schema_version": "v2_build_manifest", "status": "BUILT_FROZEN_METADATA_ONLY", "source_count": len(rows), "opportunity_count": len(opps), "generation_started": False, "selector_started": False, "solver_started": False, "training_started": False, "export_started": False, "artifacts": sorted(str(p.relative_to(V2)) for p in V2.rglob("*") if p.is_file())})
    print(json.dumps({"status": "PASS", "execution": str(FINAL_V2), "staging_validated": True, "source_count": len(rows), "opportunity_count": len(opps), "historical_records": hist["record_count"], "metadata_warnings": len(warnings)}, ensure_ascii=False))


def build() -> None:
    """Build in a sibling temporary directory, then atomically publish it."""
    global V2
    if FINAL_V2.exists():
        raise BuildError(f"refusing to mutate existing execution: {FINAL_V2}")
    staging = Path(tempfile.mkdtemp(prefix=FINAL_V2.name + ".tmp-", dir=str(FINAL_V2.parent)))
    V2 = staging
    try:
        _build_into()
        # os.replace is atomic when source and destination share a filesystem.
        staging.replace(FINAL_V2)
        V2 = FINAL_V2
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        V2 = FINAL_V2
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the new execution")
    args = ap.parse_args()
    if not args.build:
        ap.error("use --build to write a new v2 execution")
    try:
        build()
    except Exception as e:
        print(f"BUILD_FAILED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
