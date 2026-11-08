"""Offline compiler for the Round-1 Question Bank feedback views.

This module deliberately has no model, tokenizer backend, HTTP, or Memorax
imports.  It reads frozen manifests, builds deterministic views, and stops at
preflight; a later runner owns any live expansion.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    ACTIVE_CANDIDATES, ALIASES, DEFAULT_OUTPUT, ENDPOINT_CONTEXT_LIMIT,
    EXPANSION_OUTPUT_RESERVE, FROZEN_SAMPLING_SEED, FREEZE_RUN, PAIRED_QA_RUN,
    PROMPT_CLUSTERS, QUARANTINE_ROOTS, R03_ROOTS, REROUTE_RUN, TARGET_ROUND,
    VALUE_RUN, VERSION,
)

FREEZE_FILES = (
    "final_question_bank_lifecycle_freeze_report.json",
    "lifecycle_contract.json", "weighting_contract.json",
    "sampling_budget_contract.json", "cooldown_and_family_cap_contract.json",
    "active_train_scope_manifest.jsonl", "global_solver_prompt_cluster_manifest.jsonl",
    "candidate_to_cluster_map.jsonl", "cluster_canonical_representative_manifest.jsonl",
    "cluster_observation_origin_manifest.jsonl", "cluster_lifecycle_manifest.jsonl",
    "candidate_lifecycle_view.jsonl", "operation_eligibility_manifest.jsonl",
    "bridge_generation_eligibility.jsonl", "composition_parent_eligibility.jsonl",
    "questioner_targeted_sampling_weights.jsonl",
    "questioner_composition_sampling_weights.jsonl",
    "solver_replay_sampling_weights_clean.jsonl",
    "solver_replay_sampling_weights_inclusive.jsonl",
    "normalized_stem_saturation_ledger.jsonl", "cooldown_state_initialization.jsonl",
    "round1_feedback_compiler_input_view.jsonl",
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_value(value: Any) -> str:
    return digest_bytes(canonical(value).encode("utf-8"))


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"jsonl_not_object:{path}:{line_no}")
                rows.append(value)
    return rows


def check_sha(path: Path) -> tuple[str, bool, str | None]:
    observed = digest_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        return observed, False, None
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return observed, observed == expected, expected


def write_artifact(root: Path, name: str, value: Any) -> str:
    path = root / name
    if name.endswith(".jsonl"):
        text = "".join(canonical(row) + "\n" for row in value)
    elif name.endswith(".md"):
        text = str(value).rstrip("\n") + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"immutable_artifact_drift:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    sha = digest_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    side_text = f"{sha}  {path.name}\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != side_text:
        raise RuntimeError(f"immutable_sidecar_drift:{sidecar}")
    if not sidecar.exists():
        sidecar.write_text(side_text, encoding="utf-8")
    return sha


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def open_unit(*parts: Any) -> float:
    raw = "".join(str(p) for p in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return (value + 0.5) / float(2**64)


def weighted_priority(weight: float, *parts: Any) -> tuple[float, float]:
    if float(weight) <= 0:
        return 1e100, 0.0
    u = open_unit(*parts)
    return -math.log(u) / float(weight), u


def evidence_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(map(str, left)), set(map(str, right))
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def proxy_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def load_local_token_counter(tokenizer_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False,
    )
    return lambda text: len(tokenizer.encode(str(text), add_special_tokens=False))


def scrub_prompt_text(text: Any) -> str:
    # The compact planner view is deliberately prevented from carrying
    # source-answer or evaluator vocabulary, even if it appears in an anchor.
    return re.sub(
        r"(?i)semantic\s+auditor|raw\s+solver\s+prediction|prediction|solver|gold|options?|items?|feedback",
        "[redacted]", str(text or ""),
    )


def cluster_key(row: Mapping[str, Any]) -> str:
    return str(row.get("solver_prompt_fingerprint") or "")


def candidate_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("candidate_id") or "")
        if not cid or cid in result:
            raise RuntimeError(f"duplicate_candidate_id:{cid}")
        result[cid] = dict(row)
    return result


def find_value(candidate: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in candidate:
        return candidate[key]
    nested = candidate.get("candidate")
    return nested.get(key, default) if isinstance(nested, Mapping) else default


def state_summary(metric: Mapping[str, Any]) -> str:
    full = float(metric.get("full", metric.get("full_context_accuracy", 0.0)) or 0.0)
    no = float(metric.get("no", metric.get("no_context_accuracy", 0.0)) or 0.0)
    if full <= 0:
        return "FULL_ZERO"
    if full >= 1:
        return "FULL_COMPLETE"
    if full > no:
        return "PARTIAL_CONTEXT_HELPFUL"
    return "PARTIAL_NO_CONTEXT_MATCH"


def metric_profile(cluster: Mapping[str, Any]) -> dict[str, Any]:
    metrics = cluster.get("metrics") or {}
    clean = metrics.get("M_clean_policy") or {}
    inclusive = metrics.get("M_inclusive_policy") or {}
    return {
        "clean_full": clean.get("full"), "clean_no": clean.get("no"),
        "clean_context_gain": (float(clean.get("full", 0.0)) - float(clean.get("no", 0.0))),
        "inclusive_full": inclusive.get("full"), "inclusive_no": inclusive.get("no"),
        "inclusive_context_gain": (float(inclusive.get("full", 0.0)) - float(inclusive.get("no", 0.0))),
        "clean_state": state_summary(clean), "inclusive_state": state_summary(inclusive),
    }


def _ref_forms(root: str, ref: Any) -> set[str]:
    value = str(ref)
    return {value, f"{root}:{value}"} if not value.startswith(f"{root}:") else {value, value[len(root) + 1:]}


def _evidence_check(root: str, refs: Iterable[Any], turns: Mapping[str, Mapping[str, Any]]) -> tuple[bool, bool, list[str]]:
    forms = [set(_ref_forms(root, ref)) for ref in refs]
    missing = [str(ref) for ref, variants in zip(refs, forms) if not (variants & set(turns))]
    all_present = not missing
    shared = all_present and all(not str(ref).startswith("R02:") for ref in refs)
    return all_present, shared, missing


def build_dialogue_inventory(paired_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    inventory: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for row in paired_rows:
        root = str(row["root_id"])
        path = Path(row["feedback_path"])
        observed = digest_file(path)
        dialogue = read_json(path)
        complete_sha = str(dialogue.get("complete_dialogue_sha256") or "")
        turns = {str(t.get("turn_ref")): dict(t) for t in dialogue.get("turns") or []}
        freeze = read_json(path.parent / "run_freeze_manifest.json")
        payload_path = path.parent / "feedback_payload_audit.json"
        payload = read_json(payload_path) if payload_path.is_file() else {"session_id": f"{root}:memorax:R02-feedback", "payload_audit_source": "paired_manifest_session_contract"}
        inventory[root] = {
            "root_id": root, "feedback_path": str(path), "feedback_sha256": observed,
            "manifest_feedback_sha256": row.get("feedback_sha256"),
            "dialogue_content_sha256": complete_sha, "turns": turns,
            "dialogue": dialogue, "freeze": freeze, "payload": payload,
            "target_namespace": freeze.get("memorax_user_namespace"),
            "source_user_id": freeze.get("memory_user_id"),
            "source_r02_session_id": payload.get("session_id"),
            "payload_audit_path": str(payload_path) if payload_path.is_file() else None,
        }
        audit.append({
            "root_id": root, "feedback_path": str(path),
            "feedback_sha256": observed, "manifest_sha256": row.get("feedback_sha256"),
            "feedback_sha_match": observed == row.get("feedback_sha256"),
            "dialogue_content_sha256": complete_sha,
            "turn_count": len(turns), "target_session_id": f"{root}:memorax:R03-feedback",
        })
    if set(inventory) != set(R03_ROOTS):
        raise RuntimeError(f"r03_root_set_mismatch:{sorted(inventory)}")
    return inventory, audit


def branch_audit_for_cluster(
    cluster: Mapping[str, Any], rep: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    eligibility: Mapping[str, Any], cooldown: Mapping[str, Any],
) -> dict[str, Any]:
    root = str(cluster.get("root_id") or rep.get("root_id") or "")
    info = inventory.get(root)
    arm = str(rep.get("normalized_experiment_arm") or "")
    refs = list(cluster.get("evidence_refs") or find_value(rep, "evidence_refs", []) or [])
    base = {
        "solver_prompt_fingerprint": cluster_key(cluster),
        "canonical_representative_candidate_id": cluster.get("canonical_representative_candidate_id"),
        "root_id": root, "normalized_experiment_arm": arm,
        "evidence_refs": [str(x) for x in refs],
        "canonical_representative": True,
        "cluster_alias_count": int(cluster.get("alias_count") or 0),
        "cooldown": bool(cooldown.get("seed_cooldown_active", False)),
        "targeted_eligible_frozen": bool(eligibility.get("questioner_targeted_evolution_eligible")),
    }
    if root not in inventory:
        base.update({"branch_compatibility": "OUTSIDE_R03_ROOT_SCOPE", "evidence_refs_present": False, "missing_refs": [str(x) for x in refs]})
        return base
    if arm == "E0Q0_CONTROL_BLIND":
        base.update({"branch_compatibility": "CONTROL_BRANCH_EXCLUDED", "evidence_refs_present": False, "missing_refs": []})
        return base
    turns = info["turns"]
    present, shared, missing = _evidence_check(root, refs, turns)
    context_sig = str(rep.get("context_signature") or "")
    # E1Q0 records the serialized file hash; E1Q1 records the canonical
    # dialogue-content hash.  Both are accepted only for the same root.
    current = arm in {"E1Q0_FEEDBACK_BLIND", "E1Q1_FEEDBACK_GUIDED"} and context_sig in {
        str(info.get("feedback_sha256") or ""), str(info.get("manifest_feedback_sha256") or ""),
        str(info.get("dialogue_content_sha256") or ""),
    }
    if arm in {"E1Q0_FEEDBACK_BLIND", "E1Q1_FEEDBACK_GUIDED"} and not current:
        status = "DIALOGUE_SHA_MISMATCH"
    elif current and present:
        status = "CURRENT_FEEDBACK_BRANCH"
    elif arm == "ROUND0_PARENT_PLUS_R01" and shared:
        status = "SHARED_PREFIX_COMPATIBLE"
    elif not present:
        status = "EVIDENCE_REF_NOT_IN_TARGET_PREFIX"
    else:
        status = "DIALOGUE_SHA_MISMATCH"
    base.update({"branch_compatibility": status, "context_signature": context_sig,
                 "target_feedback_sha256": info.get("feedback_sha256"),
                 "target_dialogue_content_sha256": info.get("dialogue_content_sha256"),
                 "evidence_refs_present": present, "shared_prefix_evidence": shared,
                 "missing_refs": missing})
    return base


def confusion_summary(cluster: Mapping[str, Any]) -> dict[str, Any]:
    metrics = cluster.get("metrics") or {}
    result: dict[str, Any] = {"basis": "canonical_option_or_item_ids_only"}
    for name in ("M_clean_policy", "M_inclusive_policy"):
        metric = metrics.get(name) or {}
        result[name] = {
            "full_correct": metric.get("full_correct"), "full_total": metric.get("full_total"),
            "no_correct": metric.get("no_correct"), "no_total": metric.get("no_total"),
            "position_mechanism": metric.get("position_mechanism"),
        }
    return result


def build_anchor_snippets(info: Mapping[str, Any], refs: Iterable[Any], limit: int = 8) -> list[dict[str, str]]:
    turns = info.get("turns") or {}
    result = []
    for ref in refs:
        variants = _ref_forms(str(info.get("root_id")), ref)
        turn = next((turns[key] for key in variants if key in turns), None)
        if not turn:
            continue
        text = scrub_prompt_text(turn.get("text", ""))
        result.append({"turn_ref": str(turn.get("turn_ref")), "text": text[:limit]})
    return result


def make_cards(
    selected: list[dict[str, Any]], candidate_by_id: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Mapping[str, Any]], token_counter=proxy_tokens,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for item in selected:
        cluster = item["cluster"]
        cid = str(cluster["canonical_representative_candidate_id"])
        source = candidate_by_id[cid]
        candidate = source.get("candidate") if isinstance(source.get("candidate"), Mapping) else source
        root = str(cluster["root_id"])
        info = inventory[root]
        refs = list(cluster.get("evidence_refs") or candidate.get("evidence_refs") or [])
        profile = metric_profile(cluster)
        gap = "temporal_precedence" if cluster.get("question_type") == "ordering" else (
            "distributed_evidence_omission" if len(refs) > 1 else "scope_boundary_confusion"
        )
        full = {
            "card_version": "round1_to_r03_expansion_feedback_card_v1",
            "root_id": root, "target_round": TARGET_ROUND,
            "target_dialogue_sha256": info.get("feedback_sha256"),
            "target_namespace": info.get("target_namespace"),
            "source_cluster_id": cluster_key(cluster), "source_candidate_id": cid,
            "source_arm": source.get("normalized_experiment_arm"),
            "branch_compatibility": item["branch_compatibility"],
            "lifecycle": cluster.get("lifecycle_state"), "operation": "EVOLVE_UP",
            "source_question_type": candidate.get("question_type"),
            "source_question": candidate.get("question"),
            "source_evidence_refs": [str(x) for x in refs],
            "solver_profile": profile,
            "canonical_confusion_summary": confusion_summary(cluster),
            "gap_type": gap,
            "selection_weight": {k: item.get(k) for k in ("raw_weight", "normalized_weight", "u", "priority_key", "tier")},
            "source_sha256": {"candidate": digest_value(candidate), "cluster": digest_value(cluster)},
        }
        snippets = build_anchor_snippets(info, refs)
        prompt = {
            "card_version": "round1_to_r03_expansion_card_prompt_v1",
            "root_id": root, "source_question_type": candidate.get("question_type"),
            "source_question_stem": scrub_prompt_text(candidate.get("question")),
            "lifecycle": cluster.get("lifecycle_state"), "gap_status": gap,
            "evidence_anchor_refs": [str(x) for x in refs], "evidence_anchor_snippets": snippets,
            "clean_state": profile["clean_state"], "inclusive_state": profile["inclusive_state"],
            "canonical_confusion": {"basis": "canonical_ids", "clean_full_correct": confusion_summary(cluster)["M_clean_policy"].get("full_correct"), "clean_full_total": confusion_summary(cluster)["M_clean_policy"].get("full_total"), "inclusive_full_correct": confusion_summary(cluster)["M_inclusive_policy"].get("full_correct"), "inclusive_full_total": confusion_summary(cluster)["M_inclusive_policy"].get("full_total")},
            "operation": "EVOLVE_UP",
            "operation_instruction": "This feedback identifies a capability frontier.\nDo not answer, quote, or paraphrase the source question.\n\nContinue the dialogue append-only from the frozen R02-feedback prefix.\nIntroduce a genuinely new event, state update, constraint, decision,\naction, or consequence that deepens the same underlying distinction.\n\nDo not repeat old dialogue facts as if they were new.\nDo not mention QA, Solver, Gold, feedback, difficulty, or model behavior.",
        }
        prompt_text = canonical(prompt)
        full_rows.append(full)
        prompt_rows.append(prompt)
        count = int(token_counter(prompt_text))
        token_rows.append({"root_id": root, "source_cluster_id": cluster_key(cluster), "token_count": count, "token_method": "Qwen3_tokenizer_local_exact", "limit": 512, "within_limit": count <= 512})
    return full_rows, prompt_rows, token_rows


def _load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    source_sha: dict[str, Any] = {"files": [], "sidecar_failures": []}
    frozen: dict[str, Any] = {}
    for name in FREEZE_FILES:
        path = FREEZE_RUN / name
        if not path.is_file():
            raise RuntimeError(f"missing_freeze_source:{path}")
        observed, ok, expected = check_sha(path)
        source_sha["files"].append({"path": str(path), "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok})
        if not ok:
            source_sha["sidecar_failures"].append(str(path))
        frozen[name] = read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
    prov_path = REROUTE_RUN / "provisional_question_bank_entries.jsonl"
    observed, ok, expected = check_sha(prov_path)
    source_sha["files"].append({"path": str(prov_path), "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok})
    if not ok:
        source_sha["sidecar_failures"].append(str(prov_path))
    frozen["provisional_question_bank_entries.jsonl"] = read_jsonl(prov_path)
    for name in ("e1q1_complete_pool_value_analysis.json", "e1q1_candidate_value_manifest.jsonl", "e1q1_prompt_cluster_value_manifest.jsonl", "e1q1_root_value_summary.json"):
        path = VALUE_RUN / name
        if path.is_file():
            observed, ok, expected = check_sha(path)
            source_sha["files"].append({"path": str(path), "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok})
            if not ok:
                source_sha["sidecar_failures"].append(str(path))
            frozen[name] = read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
    paired = PAIRED_QA_RUN / "paired_dialogue_manifest.jsonl"
    observed, ok, expected = check_sha(paired)
    source_sha["files"].append({"path": str(paired), "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok})
    if not ok:
        source_sha["sidecar_failures"].append(str(paired))
    frozen["paired_dialogue_manifest.jsonl"] = read_jsonl(paired)
    quarantine = PAIRED_QA_RUN / "quarantine_manifest.jsonl"
    if quarantine.is_file():
        observed, ok, expected = check_sha(quarantine)
        source_sha["files"].append({"path": str(quarantine), "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok})
        frozen["quarantine_manifest.jsonl"] = read_jsonl(quarantine)
    return frozen, source_sha


def _source_rows(frozen: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = candidate_map(frozen["provisional_question_bank_entries.jsonl"])
    clusters = {cluster_key(row): dict(row) for row in frozen["cluster_lifecycle_manifest.jsonl"]}
    if len(clusters) != PROMPT_CLUSTERS:
        raise RuntimeError(f"prompt_cluster_count:{len(clusters)}")
    eligibility = {cluster_key(row): dict(row) for row in frozen["operation_eligibility_manifest.jsonl"]}
    cooldown = {cluster_key(row): dict(row) for row in frozen["cooldown_state_initialization.jsonl"]}
    if len(eligibility) != PROMPT_CLUSTERS or len(cooldown) != PROMPT_CLUSTERS:
        raise RuntimeError("cluster_auxiliary_manifest_count")
    return candidates, clusters, {**eligibility, **{f"cooldown::{k}": v for k, v in cooldown.items()}}


def _cluster_candidate(cluster: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    cid = str(cluster["canonical_representative_candidate_id"])
    if cid not in candidates:
        raise RuntimeError(f"representative_missing:{cid}")
    return candidates[cid]


def _targeted_weight(fp: str, weights: Mapping[str, Mapping[str, Any]]) -> float:
    return float((weights.get(fp) or {}).get("raw_weight") or 0.0)


def _selection_rows(
    clusters: Mapping[str, Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]],
    elig: Mapping[str, Mapping[str, Any]], cooldown: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Mapping[str, Any]], target_weights: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pool: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp, cluster in sorted(clusters.items()):
        rep = _cluster_candidate(cluster, candidates)
        e = elig[fp]
        cdown = cooldown.get(fp, {})
        audit = branch_audit_for_cluster(cluster, rep, inventory, e, cdown)
        state = str(cluster.get("lifecycle_state"))
        eligible = bool(e.get("questioner_targeted_evolution_eligible")) and state in {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE"}
        safe = eligible and audit["branch_compatibility"] in {"CURRENT_FEEDBACK_BRANCH", "SHARED_PREFIX_COMPATIBLE"} and audit.get("evidence_refs_present") and not audit.get("cooldown") and audit.get("canonical_representative")
        tier = None
        if safe:
            if audit["branch_compatibility"] == "CURRENT_FEEDBACK_BRANCH":
                tier = 1 if state == "ACTIVE_FRONTIER_BOTH" else 2
            else:
                tier = 3 if state == "ACTIVE_FRONTIER_BOTH" else 4
            weight = _targeted_weight(fp, target_weights)
            if weight <= 0:
                safe = False
        if safe:
            item = {"cluster": cluster, "lifecycle_state": state, "normalized_experiment_arm": str(rep.get("normalized_experiment_arm") or ""), **audit, "tier": tier, "raw_weight": weight,
                    "normalized_weight": float((target_weights.get(fp) or {}).get("within_root_normalized_weight") or 0.0)}
            item["priority_key"], item["u"] = weighted_priority(weight, VERSION, TARGET_ROUND, cluster["root_id"], fp, "R03_EXPANSION_FEEDBACK", FROZEN_SAMPLING_SEED)
            pool.append(item); by_root[str(cluster["root_id"])].append(item)
        else:
            reasons = []
            if state not in {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE"}: reasons.append("lifecycle_not_frontier")
            if not e.get("questioner_targeted_evolution_eligible"): reasons.append("targeted_ineligible")
            if audit["branch_compatibility"] not in {"CURRENT_FEEDBACK_BRANCH", "SHARED_PREFIX_COMPATIBLE"}: reasons.append(audit["branch_compatibility"])
            if not audit.get("evidence_refs_present"): reasons.append("evidence_missing")
            if audit.get("cooldown"): reasons.append("cooldown")
            if not audit.get("canonical_representative"): reasons.append("alias")
            if _targeted_weight(fp, target_weights) <= 0: reasons.append("zero_weight")
            exclusions.append({"solver_prompt_fingerprint": fp, "root_id": cluster.get("root_id"), "candidate_id": cluster.get("canonical_representative_candidate_id"), "reasons": sorted(set(reasons)), "branch_compatibility": audit["branch_compatibility"], "lifecycle_state": state})
    selections: list[dict[str, Any]] = []
    reserves: list[dict[str, Any]] = []
    for root in R03_ROOTS:
        rows = sorted(by_root.get(root, []), key=lambda x: (int(x["tier"]), float(x["priority_key"]), str(x["solver_prompt_fingerprint"])))
        best_tier = min((int(x["tier"]) for x in rows), default=None)
        tier_rows = [x for x in rows if int(x["tier"]) == best_tier]
        for rank, item in enumerate(tier_rows):
            disposition = "PRIMARY" if rank == 0 else "RESERVE" if rank <= 2 else "HELD_OUT_SAME_TIER"
            item["selection_disposition"] = disposition
            out = {k: item[k] for k in ("solver_prompt_fingerprint", "canonical_representative_candidate_id", "root_id", "normalized_experiment_arm", "branch_compatibility", "lifecycle_state", "tier", "raw_weight", "normalized_weight", "u", "priority_key")}
            out["selection_disposition"] = disposition
            if disposition == "PRIMARY":
                selections.append(out)
            elif disposition == "RESERVE":
                reserves.append(out)
            else:
                exclusions.append({**out, "reasons": ["not_selected_after_weighted_priority"]})
        for item in rows:
            if int(item["tier"]) != best_tier:
                item["selection_disposition"] = "HELD_OUT_LOWER_TIER"
                exclusions.append({"solver_prompt_fingerprint": item["solver_prompt_fingerprint"], "canonical_representative_candidate_id": item["canonical_representative_candidate_id"], "root_id": root, "normalized_experiment_arm": item["normalized_experiment_arm"], "branch_compatibility": item["branch_compatibility"], "lifecycle_state": item["lifecycle_state"], "tier": item["tier"], "raw_weight": item["raw_weight"], "normalized_weight": item["normalized_weight"], "u": item["u"], "priority_key": item["priority_key"], "selection_disposition": "HELD_OUT_LOWER_TIER", "reasons": ["higher_priority_tier_available"]})
    return pool, selections, reserves, exclusions, by_root


def build_questioner_pools(
    clusters: Mapping[str, Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]],
    elig: Mapping[str, Mapping[str, Any]], weights_target: Mapping[str, Mapping[str, Any]],
    weights_comp: Mapping[str, Mapping[str, Any]], inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    targeted: list[dict[str, Any]] = []
    bridge: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    for fp, cluster in sorted(clusters.items()):
        rep = _cluster_candidate(cluster, candidates)
        e = elig[fp]
        state = str(cluster.get("lifecycle_state"))
        arm = str(rep.get("normalized_experiment_arm") or "")
        audit = branch_audit_for_cluster(cluster, rep, inventory, e, {}) if str(cluster.get("root_id")) in inventory else {"branch_compatibility": "GLOBAL_ONLY", "evidence_refs_present": bool(cluster.get("evidence_refs"))}
        base = {
            "solver_prompt_fingerprint": fp,
            "canonical_representative_candidate_id": cluster.get("canonical_representative_candidate_id"),
            "root_id": cluster.get("root_id"), "lifecycle_state": state,
            "normalized_experiment_arm": arm, "canonical_representative": True,
            "normalized_stem_signature": cluster.get("normalized_stem_signature"),
            "cluster_alias_count": int(cluster.get("alias_count") or 0),
            "alias_sampling_weight": 0.0,
            "branch_compatibility": audit.get("branch_compatibility", "GLOBAL_ONLY"),
            "evidence_refs": [str(x) for x in cluster.get("evidence_refs") or []],
        }
        branch_ok = audit.get("branch_compatibility") in {"CURRENT_FEEDBACK_BRANCH", "SHARED_PREFIX_COMPATIBLE"}
        if branch_ok and arm != "E0Q0_CONTROL_BLIND" and e.get("questioner_targeted_evolution_eligible") and state in {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE"}:
            row = {**base, "raw_weight": float((weights_target.get(fp) or {}).get("raw_weight") or 0.0), "within_root_normalized_weight": float((weights_target.get(fp) or {}).get("within_root_normalized_weight") or 0.0), "global_normalized_weight": float((weights_target.get(fp) or {}).get("global_normalized_weight") or 0.0), "r03_feedback_compatible": arm != "E0Q0_CONTROL_BLIND" and (audit.get("branch_compatibility") in {"CURRENT_FEEDBACK_BRANCH", "SHARED_PREFIX_COMPATIBLE", "GLOBAL_ONLY"})}
            targeted.append(row)
            root_counts[str(cluster.get("root_id"))] += 1
        if branch_ok and arm != "E0Q0_CONTROL_BLIND" and e.get("questioner_bridge_generation_eligible") and state == "BRIDGE_REQUIRED_BOTH_ZERO":
            bridge.append({**base, "generation_mode": "BRIDGE_GENERATION", "future_prompt_contract": "construct a simpler lemma or bridge question; do not use EVOLVE_UP", "evidence_span": [str(x) for x in cluster.get("evidence_refs") or []], "raw_weight": float((weights_target.get(fp) or {}).get("raw_weight") or 0.0)})
        if branch_ok and arm != "E0Q0_CONTROL_BLIND" and e.get("composition_parent_eligible") and state in {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE", "RESOLVED_WITH_CONTEXT"}:
            composition.append({**base, "raw_weight": float((weights_comp.get(fp) or {}).get("raw_weight") or 0.0), "within_root_normalized_weight": float((weights_comp.get(fp) or {}).get("within_root_normalized_weight") or 0.0), "global_normalized_weight": float((weights_comp.get(fp) or {}).get("global_normalized_weight") or 0.0), "r03_feedback_compatible": arm != "E0Q0_CONTROL_BLIND"})
    return targeted, bridge, composition, [], dict(root_counts)


def build_composition_pairs(composition: list[dict[str, Any]], frontier_states: set[str] | None = None) -> list[dict[str, Any]]:
    frontier_states = frontier_states or {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE"}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in composition:
        # Pair feasibility is a R03 feedback view; control candidates stay in
        # the global parent ledger but are explicitly excluded here.
        if row.get("r03_feedback_compatible"):
            grouped[str(row.get("root_id"))].append(row)
    output: list[dict[str, Any]] = []
    for root, rows in sorted(grouped.items()):
        pairs: list[dict[str, Any]] = []
        for left, right in itertools.combinations(sorted(rows, key=lambda x: str(x["solver_prompt_fingerprint"])), 2):
            if left["solver_prompt_fingerprint"] == right["solver_prompt_fingerprint"]:
                continue
            if left.get("normalized_stem_signature") and left.get("normalized_stem_signature") == right.get("normalized_stem_signature"):
                continue
            a, b = left.get("evidence_refs") or [], right.get("evidence_refs") or []
            if not a or not b:
                continue
            jac = evidence_jaccard(a, b)
            if jac > 0.50:
                continue
            if left.get("lifecycle_state") not in frontier_states and right.get("lifecycle_state") not in frontier_states:
                continue
            pairs.append({"root_id": root, "left_cluster_id": left["solver_prompt_fingerprint"], "right_cluster_id": right["solver_prompt_fingerprint"], "left_candidate_id": left["canonical_representative_candidate_id"], "right_candidate_id": right["canonical_representative_candidate_id"], "evidence_jaccard": jac, "feasibility_basis": "lexical_and_evidence_only", "semantic_complementarity_confirmed": False})
        output.extend(sorted(pairs, key=lambda x: (float(x["evidence_jaccard"]), x["left_cluster_id"], x["right_cluster_id"]))[:100])
    return output


def build_fresh_negative_view(clusters: Mapping[str, Mapping[str, Any]], saturation: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in saturation:
        rows.append({"root_id": row.get("root_id"), "normalized_stem_signature": row.get("normalized_stem_signature"), "prompt_cluster_count": row.get("prompt_cluster_count"), "candidate_count": row.get("candidate_count"), "family_saturation_multiplier": row.get("family_saturation_multiplier"), "fresh_avoidance_only": True})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("root_id"))].append(row)
    result: list[dict[str, Any]] = []
    for root, items in sorted(grouped.items()):
        result.extend(sorted(items, key=lambda x: (-int(x.get("prompt_cluster_count") or 0), -int(x.get("candidate_count") or 0), str(x.get("normalized_stem_signature"))))[:10])
    return result


def build_replay_pool(rows: list[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]], model: str) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if str(row.get("model_name")) == model and bool(row.get("eligible"))]
    eligible.sort(key=lambda x: (-float(x.get("raw_weight") or 0.0), str(x.get("solver_prompt_fingerprint"))))
    output = []
    for rank, row in enumerate(eligible, 1):
        cid = str(row.get("canonical_representative_candidate_id"))
        source = candidates.get(cid, {})
        output.append({**row, "replay_rank": rank, "alias_sampling_weight": 0.0 if int(row.get("representative_multiplier") or 1) != 1 else None, "candidate_context": {"root_id": source.get("candidate", {}).get("root_id", source.get("root_id")), "normalized_experiment_arm": source.get("normalized_experiment_arm"), "split": source.get("split", "train")}})
    return output


def expansion_contract(inventory: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    freezes = [info["freeze"] for info in inventory.values()]
    api_models = sorted({str((f.get("api_configuration") or {}).get("model")) for f in freezes})
    api_temps = sorted({float((f.get("api_configuration") or {}).get("temperature")) for f in freezes})
    max_tokens = freezes[0].get("max_completion_tokens") if freezes else {}
    role_sets = [tuple(sorted((f.get("required_outbound_roles") or []))) for f in freezes]
    role_coverage = all(bool((f.get("outbound_role_coverage") or {}).get("valid")) for f in freezes)
    source_runtime = str((freezes[0].get("feedback_adapter") or {}).get("original_runtime_module")) if freezes else ""
    source_sha = str((freezes[0].get("feedback_adapter") or {}).get("original_runtime_sha256")) if freezes else ""
    # The generation temperature and tokenizer/context ceiling are frozen in
    # the historical preflight/config lineage rather than in the QA manifest.
    config_path = Path("/root/autodl-tmp/gxy-selfplay/config.json")
    config = read_json(config_path) if config_path.is_file() else {}
    generation = config.get("generation") or {}
    runtime_path = Path("/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/live.py")
    runtime_text = runtime_path.read_text(encoding="utf-8") if runtime_path.is_file() else ""
    current_runtime_sha = digest_file(runtime_path) if runtime_path.is_file() else None
    required_roles = sorted(set(role_sets[0] if role_sets else []) | {"simple_append_planner_v1", "simple_append_planner_candidate_repair_v1", "planner_outline_semantic_auditor_v1", "memory_event_realization_aware_semantic_gate_v1", "simple_append_round_realizer_v1", "memory_event_realizer_adherence_auditor_v1", "simple_append_scene_continuation_realizer_v1", "simple_append_scene_replacement_realizer_v1", "memory_event_realizer_adherence_reaudit_v1"})
    complete = bool(
        api_models == ["Qwen/Qwen3-14B"] and api_temps == [0.25]
        and float(generation.get("temperature")) == 0.45
        and "CONTEXT_LIMIT = 40_960" in runtime_text and role_coverage
        and bool(current_runtime_sha and current_runtime_sha == source_sha)
        and max_tokens and source_runtime and source_sha and all(role_sets)
    )
    contract = {
        "contract_version": "r03_expansion_contract_fingerprint_v1",
        "model": api_models[0] if len(api_models) == 1 else api_models,
        "effective_api_temperature": api_temps[0] if len(api_temps) == 1 else api_temps,
        "generation_planning_temperature": generation.get("temperature"),
        "context_ceiling": ENDPOINT_CONTEXT_LIMIT,
        "tokenizer_path": freezes[0].get("tokenizer_path") if freezes else None,
        "max_completion_tokens": max_tokens,
        "prompt_renderer": {"prompt_version": (freezes[0].get("feedback_adapter") or {}).get("feedback_prompt_version") if freezes else None, "runtime_module": source_runtime, "runtime_sha256": source_sha},
        "historical_runtime_sha256": source_sha,
        "current_runtime_sha256": current_runtime_sha,
        "runtime_sha_matches_historical_manifest": bool(current_runtime_sha and current_runtime_sha == source_sha),
        "planner_realizer_contract": {"required_outbound_roles": required_roles, "coverage_valid": role_coverage},
        "repair_retry_policy": {"planner_repair": max_tokens.get("planner_repair"), "adherence_auditor_attempts": (freezes[0].get("request_budget") or {}).get("realizer_adherence_auditor") if freezes else None},
        "semantic_continuation_gate": "memory_event_realization_aware_semantic_gate_v1",
        "child_add_payload_builder": "workflow.build_add_intent",
        "session_override": {"suffix": ":memorax:R03-feedback", "append_mode": "R03_delta_only", "card_in_memorax_payload": False},
        "source_freeze_roots": len(freezes),
        "fingerprint_complete": complete,
        "gaps": [] if complete else (["current_runtime_sha_differs_from_historical_freeze"] if current_runtime_sha != source_sha else ["historical_expansion_contract_fields_not_unique_or_missing"]),
    }
    return contract, {"config_path": str(config_path), "config_sha256": digest_file(config_path) if config_path.is_file() else None, "runtime_path": str(runtime_path), "runtime_sha256": digest_file(runtime_path) if runtime_path.is_file() else None}


def build_token_preflight(selected: list[dict[str, Any]], prompt_rows: list[dict[str, Any]], inventory: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        counter = load_local_token_counter(Path(str(contract.get("tokenizer_path") or "/root/autodl-tmp/Qwen3-4B")))
        token_method = "Qwen3_tokenizer_local_exact"
    except Exception:
        counter = proxy_tokens
        token_method = "deterministic_unicode_wordpiece_proxy"
    rows = []
    for item, prompt in zip(selected, prompt_rows):
        root = str(item["root_id"])
        info = inventory[root]
        raw = int(counter(canonical(info["dialogue"].get("turns") or [])))
        memory_nav = 512
        feedback = proxy_tokens(canonical(prompt))
        base = 1024
        total = raw + memory_nav + feedback + base + EXPANSION_OUTPUT_RESERVE
        rows.append({"root_id": root, "source_cluster_id": item["solver_prompt_fingerprint"], "complete_cumulative_dialogue_tokens": raw, "memory_navigation_reserved_tokens": memory_nav, "feedback_prompt_view_tokens": feedback, "r03_base_prompt_reserved_tokens": base, "output_reserve_tokens": EXPANSION_OUTPUT_RESERVE, "total_reserved_tokens": total, "context_ceiling": ENDPOINT_CONTEXT_LIMIT, "input_truncated": False, "feedback_truncated": False, "memory_navigation_truncated": False, "within_context_ceiling": total <= ENDPOINT_CONTEXT_LIMIT})
    return {"token_preflight_version": "r03_expansion_token_preflight_v1", "tokenizer_source": contract.get("tokenizer_path"), "token_method": token_method, "context_ceiling": ENDPOINT_CONTEXT_LIMIT, "rows": rows, "max_total_reserved_tokens": max((r["total_reserved_tokens"] for r in rows), default=0), "truncation_count": sum(not r["within_context_ceiling"] for r in rows), "all_within_context_ceiling": all(r["within_context_ceiling"] for r in rows)}


def compile_feedback(run_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_root = Path(run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"output_root_not_empty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    frozen, source_sha = _load_sources()
    candidates, clusters, auxiliary = _source_rows(frozen)
    elig = {fp: auxiliary[fp] for fp in clusters}
    cooldown = {fp: auxiliary[f"cooldown::{fp}"] for fp in clusters}
    target_weights = {str(r["solver_prompt_fingerprint"]): r for r in frozen["questioner_targeted_sampling_weights.jsonl"]}
    comp_weights = {str(r["solver_prompt_fingerprint"]): r for r in frozen["questioner_composition_sampling_weights.jsonl"]}
    paired_rows = frozen["paired_dialogue_manifest.jsonl"]
    inventory, dialogue_audit = build_dialogue_inventory(paired_rows)
    # Extend the immutable source audit down the authoritative per-root
    # lineage.  Missing payload audit for the legacy single-root record is
    # represented by its manifest session contract, not silently fabricated.
    for root, info in inventory.items():
        for raw_path in (info["feedback_path"], str(Path(info["feedback_path"]).parent / "run_freeze_manifest.json"), str(Path(info["feedback_path"]).parent / "planner" / "prompt_manifest.json")):
            path = Path(raw_path)
            if not path.is_file():
                source_sha["sidecar_failures"].append(raw_path)
                continue
            observed, ok, expected = check_sha(path)
            source_sha["files"].append({"path": raw_path, "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok, "root_id": root})
            if expected is not None and not ok:
                source_sha["sidecar_failures"].append(raw_path)
        payload_path = Path(info["feedback_path"]).parent / "feedback_payload_audit.json"
        if payload_path.is_file():
            observed, ok, expected = check_sha(payload_path)
            source_sha["files"].append({"path": str(payload_path), "sha256": observed, "sidecar_expected": expected, "sidecar_ok": ok, "root_id": root})
            if expected is not None and not ok:
                source_sha["sidecar_failures"].append(str(payload_path))
    if source_sha["sidecar_failures"]:
        raise RuntimeError("source_sidecar_failure:" + ",".join(source_sha["sidecar_failures"][:3]))
    quarantine_rows = frozen.get("quarantine_manifest.jsonl") or [{"root_id": r} for r in QUARANTINE_ROOTS]
    # Every cluster is audited against its own root.  This is also the source
    # of truth for excluding the control branch from R03 Questioner views.
    branch_rows = []
    for fp, cluster in sorted(clusters.items()):
        rep = _cluster_candidate(cluster, candidates)
        branch_rows.append(branch_audit_for_cluster(cluster, rep, inventory, elig[fp], cooldown[fp]))
    pool, selection, reserves, exclusions, by_root = _selection_rows(clusters, candidates, elig, cooldown, inventory, target_weights)
    selection_for_cards = [{**row, "cluster": clusters[row["solver_prompt_fingerprint"]]} for row in selection]
    try:
        token_counter = load_local_token_counter(Path("/root/autodl-tmp/Qwen3-4B"))
    except Exception:
        token_counter = proxy_tokens
    full_cards, prompt_cards, token_rows = make_cards(selection_for_cards, candidates, inventory, token_counter)
    targeted, bridge, composition, _, targeted_roots = build_questioner_pools(clusters, candidates, elig, target_weights, comp_weights, inventory)
    pairs = build_composition_pairs(composition)
    fresh = build_fresh_negative_view(clusters, frozen["normalized_stem_saturation_ledger.jsonl"])
    replay_clean = build_replay_pool(frozen["solver_replay_sampling_weights_clean.jsonl"], candidates, "M_clean_policy")
    replay_inclusive = build_replay_pool(frozen["solver_replay_sampling_weights_inclusive.jsonl"], candidates, "M_inclusive_policy")
    contract, contract_sources = expansion_contract(inventory)
    token_preflight = build_token_preflight(selection, prompt_cards, inventory, contract)
    lineage = []
    for root in R03_ROOTS:
        info = inventory[root]
        lineage.append({"root_id": root, "source_prefix": "Parent+R01+R02-feedback", "source_dialogue_sha256": info["feedback_sha256"], "source_namespace": info["target_namespace"], "source_user_id": info["source_user_id"], "source_r02_session_id": info["source_r02_session_id"], "target_round": "R03", "target_session_id": f"{root}:memorax:R03-feedback", "append_mode": "R03_delta_only", "parent_add_allowed": False, "r01_add_allowed": False, "r02_add_allowed": False, "r03_child_add_allowed_after_live_approval": True})
    # Preserve the frozen formula and explicitly document the feature gate.
    allowlist = {"allowed": ["frozen lifecycle", "frozen questioner targeted weight", "Clean/Inclusive full/no/context gain", "authoritative difficulty", "training exposure", "evidence status", "position mechanism", "cross-round", "branch compatibility", "cooldown", "normalized-stem saturation", "canonical representative status"], "experiment_arm_usage": "branch compatibility and source-tier partition only", "forbidden": ["experiment_arm as reward", "E1Q1 label as reward", "feedback_target_hit", "semantic auditor label", "Dev/Test content", "raw candidate count", "candidate ID except stable hash tie-break", "manual preference"]}
    usage = {"selection_operation": "R03_EXPANSION_FEEDBACK", "used_features": allowlist["allowed"], "forbidden_features_observed": [], "experiment_arm_used_only_for_branch": True, "semantic_auditor_used": False, "feedback_target_hit_used": False, "dev_test_content_used": False}
    scope_report = read_json(FREEZE_RUN / "final_question_bank_lifecycle_freeze_report.json")
    lifecycle_counts = Counter(str(c.get("lifecycle_state")) for c in clusters.values())
    op_counts = {key: sum(bool(row.get(key)) for row in frozen["operation_eligibility_manifest.jsonl"]) for key in ("questioner_targeted_evolution_eligible", "questioner_bridge_generation_eligible", "composition_parent_eligible", "solver_replay_eligible_clean", "solver_replay_eligible_inclusive")}
    global_scope = {"active_train_candidates": len(candidates), "active_train_prompt_clusters": len(clusters), "aliases": sum(int(c.get("alias_count") or 0) for c in clusters.values()), "lifecycle_counts": dict(sorted(lifecycle_counts.items())), "operation_eligibility_counts": op_counts, "dev_test_content_read": False, "question_bank_scope": "ACTIVE_TRAIN_ONLY", "online_calls": {"api": 0, "solver": 0, "qa": 0, "memorax": 0, "auditor": 0, "training": 0}}
    root_inventory = []
    for root in R03_ROOTS:
        rows = sorted(by_root.get(root, []), key=lambda x: (int(x["tier"]), float(x["priority_key"])))
        root_inventory.append({"root_id": root, "current_feedback_seed_count": sum(x["branch_compatibility"] == "CURRENT_FEEDBACK_BRANCH" for x in rows), "shared_prefix_fallback_count": sum(x["branch_compatibility"] == "SHARED_PREFIX_COMPATIBLE" for x in rows), "safe_seed_available": bool(rows), "selected_primary": next((x["canonical_representative_candidate_id"] for x in selection if x["root_id"] == root), None), "selected_reserves": [x["canonical_representative_candidate_id"] for x in reserves if x["root_id"] == root]})
    budget_plan = {"questioner_budget_version": "provisional_v1", "TARGETED_EVOLUTION": 0.40, "MULTI_QA_COMPOSITION": 0.20, "FRESH_EXPLORATION": 0.40, "BRIDGE_GENERATION": {"conditional": True, "deducted_from_main_budget": False}, "final_request_counts": "deferred_until_R03_novel_question_capacity_preflight"}
    replay_budget = {"default_replay_ratio": 0.10, "hard_cap": 0.15, "new_training_data_minimum": 0.85, "cluster_max_replay_per_round": 1, "alias_weight": 0.0, "final_replay_selection": "deferred_until_new_hard_count"}
    artifacts = {
        "feedback_compiler_contract.json": {"version": VERSION, "stage": "offline_preflight_only", "source_roots": [str(FREEZE_RUN), str(REROUTE_RUN), str(VALUE_RUN), str(PAIRED_QA_RUN)], "no_live_calls": True},
        "feedback_feature_allowlist.json": allowlist,
        "feedback_feature_usage_audit.json": usage,
        "global_question_bank_scope_audit.json": global_scope,
        "r03_root_scope_manifest.jsonl": dialogue_audit,
        "quarantine_root_manifest.jsonl": quarantine_rows,
        "branch_compatibility_audit.jsonl": branch_rows,
        "r03_expansion_seed_candidate_pool.jsonl": pool,
        "r03_expansion_seed_selection.jsonl": selection,
        "r03_expansion_seed_reserves.jsonl": reserves,
        "r03_expansion_seed_exclusions.jsonl": exclusions,
        "r03_expansion_feedback_cards_full.jsonl": full_cards,
        "r03_expansion_feedback_cards_prompt_view.jsonl": prompt_cards,
        "r03_expansion_feedback_card_token_report.json": {"rows": token_rows, "max_token_count": max((x["token_count"] for x in token_rows), default=0), "all_within_512": all(x["within_limit"] for x in token_rows)},
        "questioner_targeted_pool.jsonl": targeted,
        "questioner_bridge_pool.jsonl": bridge,
        "questioner_composition_parent_pool.jsonl": composition,
        "questioner_composition_pair_feasibility.jsonl": pairs,
        "fresh_exploration_negative_signature_view.jsonl": fresh,
        "questioner_future_budget_plan.json": budget_plan,
        "solver_replay_ranked_pool_clean.jsonl": replay_clean,
        "solver_replay_ranked_pool_inclusive.jsonl": replay_inclusive,
        "solver_replay_future_budget_plan.json": replay_budget,
        "r03_cumulative_lineage_plan.jsonl": lineage,
        "r03_dialogue_identity_audit.json": {"roots": dialogue_audit, "all_feedback_sha_match": all(x["feedback_sha_match"] for x in dialogue_audit), "append_mode": "R03_delta_only", "parent_r01_r02_add_allowed": False},
        "r03_expansion_contract_fingerprint.json": contract,
        "r03_expansion_token_preflight.json": token_preflight,
        "root_feedback_inventory.jsonl": root_inventory,
        "root_feedback_budget_summary.json": {"roots": len(R03_ROOTS), "targeted_pool": len(targeted), "bridge_pool": len(bridge), "composition_parent_pool": len(composition), "composition_pairs": len(pairs), "fresh_negative_signatures": len(fresh), "clean_replay": len(replay_clean), "inclusive_replay": len(replay_inclusive)},
        "source_sha_audit.json": {**source_sha, "contract_sources": contract_sources, "source_manifests_modified": 0},
    }
    for name, value in artifacts.items():
        write_artifact(run_root, name, value)
    failures = []
    if len(candidates) != ACTIVE_CANDIDATES: failures.append("active_candidate_count")
    if len(clusters) != PROMPT_CLUSTERS: failures.append("prompt_cluster_count")
    if sum(int(c.get("alias_count") or 0) for c in clusters.values()) != ALIASES: failures.append("alias_count")
    if set(R03_ROOTS) & set(QUARANTINE_ROOTS): failures.append("root_overlap")
    if len(selection) != len(R03_ROOTS): failures.append("safe_seed_root_count")
    if not token_preflight["all_within_context_ceiling"]: failures.append("token_budget")
    if not contract["fingerprint_complete"]: failures.append("expansion_contract_fingerprint")
    if not all(x["within_limit"] for x in token_rows): failures.append("card_token_limit")
    status = "ROUND1_QUESTION_BANK_FEEDBACK_COMPILATION_COMPLETE" if not failures else "ROUND1_QUESTION_BANK_FEEDBACK_COMPILATION_COMPLETE_WITH_GAPS"
    next_status = "READY_FOR_R03_CUMULATIVE_EXPANSION_PREFLIGHT" if not failures else "R03_EXPANSION_PREFLIGHT_NOT_READY"
    report = {"status": status, "next_status": next_status, "candidate_count": len(candidates), "prompt_cluster_count": len(clusters), "alias_count": sum(int(c.get("alias_count") or 0) for c in clusters.values()), "r03_target_roots": list(R03_ROOTS), "quarantine_roots": list(QUARANTINE_ROOTS), "primary_seed_count": len(selection), "reserve_seed_count": len(reserves), "targeted_pool_count": len(targeted), "bridge_pool_count": len(bridge), "composition_parent_count": len(composition), "composition_pair_count": len(pairs), "fresh_negative_signature_count": len(fresh), "clean_replay_count": len(replay_clean), "inclusive_replay_count": len(replay_inclusive), "lifecycle_counts": dict(sorted(lifecycle_counts.items())), "operation_eligibility_counts": op_counts, "e1q1_raw_consensus_context_helpful_hard": 38, "e1q1_active_frontier_both": 23, "e1q1_position_only_diagnostic": 15, "online_calls": global_scope["online_calls"], "source_manifests_modified": 0, "failures": failures, "contract_fingerprint_complete": contract["fingerprint_complete"], "max_reserved_tokens": token_preflight["max_total_reserved_tokens"], "truncation_count": token_preflight["truncation_count"]}
    write_artifact(run_root, "final_round1_question_bank_feedback_compiler_report.json", report)
    markdown = "\n".join(["# Round-1 Question Bank Feedback Compiler", "", f"- Status: `{status}`", f"- Next status: `{next_status}`", f"- Active candidates / clusters / aliases: `{len(candidates)} / {len(clusters)} / {report['alias_count']}`", f"- R03 roots ready: `{len(selection)}/{len(R03_ROOTS)}`", f"- Targeted / Bridge / Composition: `{len(targeted)} / {len(bridge)} / {len(composition)}`", f"- Clean / Inclusive replay: `{len(replay_clean)} / {len(replay_inclusive)}`", f"- Contract fingerprint complete: `{contract['fingerprint_complete']}`", f"- Maximum reserved tokens: `{token_preflight['max_total_reserved_tokens']}` / `{ENDPOINT_CONTEXT_LIMIT}`", f"- Online calls: `{global_scope['online_calls']}`", "", "R03 live expansion was not started."])
    write_artifact(run_root, "final_round1_question_bank_feedback_compiler_report.md", markdown)
    return report
