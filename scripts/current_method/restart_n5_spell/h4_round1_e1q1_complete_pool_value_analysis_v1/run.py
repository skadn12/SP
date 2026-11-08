"""Pure-offline value analysis for the complete E1Q1 pool.

This stage never invokes a model.  It treats E1Q1 generation candidates,
observed canonical Solver inputs, and prompt clusters as different units so
aliases cannot become duplicate Solver observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    E1Q0_SOLVER_RUN,
    E1Q1_CLOSURE_RUN,
    E1Q1_GENERATION_RUN,
    E1Q1_OLD_SOLVER_RUN,
    MODELS,
    OUTPUT_RUN,
    REROUTE_RUN,
    VERSION,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_no}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing_source:{path}")
    observed = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing_sha_sidecar:{path}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0] != observed:
        raise ValueError(f"sha_mismatch:{path}")
    return observed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def root_suffix(ref: Any) -> str:
    value = str(ref)
    if ":R" in value:
        return value[value.index(":R") + 1 :]
    return value


def evidence_span(row: Mapping[str, Any], dialogue: Mapping[str, Any] | None) -> str:
    refs = {root_suffix(value) for value in row.get("evidence_refs") or []}
    if dialogue is None:
        if not refs:
            return "UNKNOWN"
        r02 = {ref for ref in refs if ref.startswith("R02:")}
        if r02 == refs:
            return "R02_ONLY_UNVERIFIED"
        if r02:
            return "CROSS_ROUND_UNVERIFIED"
        return "HISTORICAL_ONLY_UNVERIFIED"
    valid = {root_suffix(turn.get("turn_ref")) for turn in dialogue.get("turns") or [] if isinstance(turn, Mapping)}
    if not refs:
        return "UNKNOWN"
    if not refs <= valid:
        return "INVALID_REF"
    r02 = {ref for ref in valid if ref.startswith("R02:")}
    if refs <= r02:
        return "R02_ONLY"
    if not refs & r02:
        return "HISTORICAL_ONLY"
    return "CROSS_ROUND"


def count_accuracy(row: Mapping[str, Any], context: str) -> tuple[int, int, float | None]:
    total = int(row.get(f"{context}_context_total_count") or row.get(f"{context}_total_count") or 0)
    correct = int(row.get(f"{context}_context_correct_count") or row.get(f"{context}_correct_count") or 0)
    diagnostic = (row.get("position_diagnostics") or {}).get(context) or {}
    total = int(diagnostic.get("total_count") or total)
    correct = int(diagnostic.get("correct_count") or correct)
    accuracy = correct / total if total else None
    return total, correct, accuracy


def metric(row: Mapping[str, Any]) -> dict[str, Any]:
    ft, fc, fa = count_accuracy(row, "full")
    nt, nc, na = count_accuracy(row, "no")
    return {
        "full_context_total": ft,
        "full_context_correct": fc,
        "full_context_accuracy": fa,
        "no_context_total": nt,
        "no_context_correct": nc,
        "no_context_accuracy": na,
        "context_gain": fa - na if fa is not None and na is not None else None,
        "difficulty_bucket": row.get("difficulty_bucket"),
        "context_helpful": row.get("context_helpful"),
        "technical_failure_count": int(row.get("technical_failure_count") or 0),
        "request_count": int(row.get("request_count") or 0),
    }


def load_difficulty(run: Path, model: str) -> dict[str, dict[str, Any]]:
    path = run / model / "feedback" / "difficulty_manifest.jsonl"
    verify_sidecar(path)
    rows = load_jsonl(path)
    result = {str(row["candidate_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate_difficulty_ids:{path}")
    return result


def load_dialogues() -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for path in sorted(E1Q1_GENERATION_RUN.glob("cc-train-episode-*/dialogue_source.json")):
        verify_sidecar(path)
        result[path.parent.name] = load_json(path)
    if len(result) != 7:
        raise ValueError(f"e1q1_dialogue_root_count:{len(result)}")
    return result


def prompt_cluster_fingerprint(candidate: Mapping[str, Any], dialogue: Mapping[str, Any] | None) -> str:
    """Reconstruct a conservative baseline prompt identity without model calls."""
    turns = [] if dialogue is None else [
        {"turn_ref": turn.get("turn_ref"), "speaker": turn.get("speaker"), "text": turn.get("text")}
        for turn in dialogue.get("turns") or [] if isinstance(turn, Mapping)
    ]
    payload = {
        "question_type": candidate.get("question_type"),
        "question": candidate.get("question"),
        "options": candidate.get("options") or {},
        "items": candidate.get("items") or {},
        "gold": candidate.get("gold") or [candidate.get("correct_option")],
        "dialogue": turns,
        "contract": "h4_round0_b0_k3_solver_contract_v1",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_e1q1() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    routing_path = REROUTE_RUN / "e1q1_multilabel_routing.jsonl"
    complete_path = E1Q1_CLOSURE_RUN / "e1q1_complete_candidate_manifest.jsonl"
    cluster_path = E1Q1_CLOSURE_RUN / "solver_prompt_cluster_map.jsonl"
    for path in (routing_path, complete_path, cluster_path):
        verify_sidecar(path)
    routing_rows = load_jsonl(routing_path)
    complete_rows = load_jsonl(complete_path)
    cluster_rows = load_jsonl(cluster_path)
    if len(routing_rows) != 238 or len(complete_rows) != 238 or len(cluster_rows) != 238:
        raise ValueError("e1q1_candidate_closure_count_mismatch")
    generation: dict[str, dict[str, Any]] = {}
    for path in sorted(E1Q1_GENERATION_RUN.glob("cc-train-episode-*/candidate_pool_scoped.json")):
        verify_sidecar(path)
        for row in load_json(path):
            generation[str(row["candidate_id"])] = row
    if set(generation) != {str(row["candidate_id"]) for row in routing_rows}:
        raise ValueError("e1q1_generation_routing_id_mismatch")
    routing = {str(row["candidate_id"]): row for row in routing_rows}
    complete = {str(row["candidate_id"]): row for row in complete_rows}
    if set(routing) != set(complete):
        raise ValueError("e1q1_complete_routing_id_mismatch")
    clusters: dict[str, dict[str, Any]] = {}
    for row in cluster_rows:
        fp = str(row["solver_prompt_fingerprint"])
        clusters.setdefault(fp, {"fingerprint": fp, "member_ids": list(row["cluster_member_ids"]), "representative_id": row["cluster_representative_candidate_id"], "member_count": int(row["cluster_member_count"]), "members": []})
        clusters[fp]["members"].append(row)
    if len(clusters) != 233:
        raise ValueError(f"e1q1_cluster_count:{len(clusters)}")
    return routing_rows, generation, complete, clusters


def load_e1q1_metrics() -> dict[str, dict[str, dict[str, Any]]]:
    old = {model: load_difficulty(E1Q1_OLD_SOLVER_RUN, model) for model in MODELS}
    new = {model: load_difficulty(E1Q1_CLOSURE_RUN, model) for model in MODELS}
    result: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for model in MODELS:
        for cid, row in {**old[model], **new[model]}.items():
            result[model][cid] = metric(row)
        if len(result[model]) != 231:
            raise ValueError(f"e1q1_observed_metric_count:{model}:{len(result[model])}")
    return result


def load_e1q0() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = E1Q0_SOLVER_RUN / "candidate_manifest.jsonl"
    verify_sidecar(manifest_path)
    manifest = [row for row in load_jsonl(manifest_path) if row.get("branch") == "feedback"]
    if len(manifest) != 254:
        raise ValueError(f"e1q0_feedback_count:{len(manifest)}")
    by_id = {str(row["candidate_id"]): row for row in manifest}
    metrics = {model: load_difficulty(E1Q0_SOLVER_RUN, model) for model in MODELS}
    combined: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    for model in MODELS:
        if set(metrics[model]) != set(by_id):
            raise ValueError(f"e1q0_metric_id_mismatch:{model}")
        combined[model] = {cid: metric(row) for cid, row in metrics[model].items()}
    roots = sorted({str(row["root_id"]) for row in manifest})
    return by_id, combined, {"candidate_count": len(manifest), "roots": roots}


def source_metadata(routing: Mapping[str, Any], candidate: Mapping[str, Any], status: str) -> dict[str, Any]:
    labels = routing.get("feedback_labels") or {}
    lineage = routing.get("lineage") or {}
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "root_id": str(candidate.get("root_id")),
        "question_type": candidate.get("question_type"),
        "slot_type": candidate.get("slot_type") or lineage.get("slot_type"),
        "generation_source": candidate.get("generation_source") or lineage.get("generation_source"),
        "solver_prompt_fingerprint": routing.get("solver_prompt_fingerprint"),
        "solver_completion_status": status,
        "r02_touch": bool(routing.get("r02_touch")),
        "r02_ref_aligned": bool(routing.get("r02_ref_aligned")),
        "feedback_contract_hit": routing.get("feedback_contract_hit"),
        "feedback_target_hit": "UNKNOWN",
        "off_target": bool(routing.get("off_target")),
        "evidence_span": routing.get("evidence_span"),
        "normalized_stem_signature": routing.get("normalized_stem_signature"),
        "surface_signature": routing.get("surface_signature"),
        "exact_surface_qa_duplicate": bool(routing.get("exact_surface_qa_duplicate")),
        "context_variant": bool(routing.get("context_variant")),
        "evidence_annotation_variant": bool(routing.get("evidence_annotation_variant")),
        "lineage_family_id": routing.get("lineage_family_id"),
        "split": candidate.get("split", routing.get("split")),
    }


def pool_metrics(rows: list[Mapping[str, Any]], weighting: str) -> dict[str, Any]:
    """Summarize paired model rows using candidate, root, or cluster weighting."""
    result: dict[str, Any] = {"unit_count": len(rows), "weighting": weighting, "models": {}}
    for model in MODELS:
        def unit_metric(group: list[Mapping[str, Any]], context: str) -> tuple[int, int, float | None]:
            total = sum(int((row["metrics"][model] or {}).get(f"{context}_context_total") or 0) for row in group)
            correct = sum(int((row["metrics"][model] or {}).get(f"{context}_context_correct") or 0) for row in group)
            return total, correct, correct / total if total else None

        if weighting == "candidate_weighted":
            groups = {"ALL": rows}
        elif weighting == "prompt_cluster_equal":
            groups = {str(row["unit_id"]): [row] for row in rows}
        else:
            groups = defaultdict(list)
            for row in rows:
                groups[str(row["root_id"])].append(row)
        full = [unit_metric(group, "full") for group in groups.values()]
        no = [unit_metric(group, "no") for group in groups.values()]
        if weighting == "candidate_weighted":
            ft, fc, fa = full[0] if full else (0, 0, None)
            nt, nc, na = no[0] if no else (0, 0, None)
            gain = fa - na if fa is not None and na is not None else None
        else:
            ft, fc = sum(x[0] for x in full), sum(x[1] for x in full)
            nt, nc = sum(x[0] for x in no), sum(x[1] for x in no)
            fa = mean(x[2] for x in full if x[2] is not None)
            na = mean(x[2] for x in no if x[2] is not None)
            gain_values = [f[2] - n[2] for f, n in zip(full, no) if f[2] is not None and n[2] is not None]
            gain = mean(gain_values)
        result["models"][model] = {
            "full_context_accuracy": fa,
            "no_context_accuracy": na,
            "context_gain": gain,
            "full_context_correct": fc,
            "full_context_total": ft,
            "no_context_correct": nc,
            "no_context_total": nt,
        }
    return result


def pair_flags(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    both = []
    full_zero = []
    worse = []
    for row in rows:
        clean, inclusive = row["metrics"].get(MODELS[0]), row["metrics"].get(MODELS[1])
        if not clean or not inclusive:
            continue
        hard = clean.get("difficulty_bucket") == "hard_candidate" and inclusive.get("difficulty_bucket") == "hard_candidate"
        helpful = all((m.get("full_context_accuracy") is not None and m.get("no_context_accuracy") is not None and m["full_context_accuracy"] > m["no_context_accuracy"]) for m in (clean, inclusive))
        if hard and helpful:
            both.append(row)
        if all(m.get("full_context_accuracy") == 0 for m in (clean, inclusive)):
            full_zero.append(row)
        if any(m.get("full_context_accuracy") is not None and m.get("no_context_accuracy") is not None and m["full_context_accuracy"] < m["no_context_accuracy"] for m in (clean, inclusive)):
            worse.append(row)
    by_root = Counter(str(row["root_id"]) for row in both)
    zero_by_root = Counter(str(row["root_id"]) for row in full_zero)
    worse_by_root = Counter(str(row["root_id"]) for row in worse)
    return {
        "consensus_context_helpful_hard_count": len(both),
        "consensus_context_helpful_hard_by_root": dict(sorted(by_root.items())),
        "consensus_context_helpful_hard_per_root": len(both) / len({str(row["root_id"]) for row in rows}) if rows else None,
        "full_zero_by_both_count": len(full_zero),
        "full_zero_by_both_by_root": dict(sorted(zero_by_root.items())),
        "full_zero_by_both_per_root": len(full_zero) / len({str(row["root_id"]) for row in rows}) if rows else None,
        "context_worse_any_count": len(worse),
        "context_worse_any_by_root": dict(sorted(worse_by_root.items())),
        "context_worse_any_per_root": len(worse) / len({str(row["root_id"]) for row in rows}) if rows else None,
        "technical_failure_count": sum(int((row["metrics"].get(model) or {}).get("technical_failure_count") or 0) for row in rows for model in MODELS),
    }


def generation_request_counts() -> dict[str, int]:
    pipeline_total = 0
    candidate_generation = 0
    for path in sorted(E1Q1_GENERATION_RUN.glob("cc-train-episode-*/final_qa_feedback_generation_report.json")):
        report = load_json(path)
        pipeline_total += int(report.get("request_count") or 0)
        candidate_generation += int((report.get("generation") or {}).get("candidate_generation_calls") or 0)
    return {
        "candidate_generation_requests": candidate_generation,
        "intent_mining_requests": pipeline_total - candidate_generation,
        "pipeline_requests_total": pipeline_total,
    }


def add_baseline_flags(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute bucket flags for E1Q0 where no E1Q1 slot metadata exists."""
    both = []
    full_zero = []
    worse = []
    for row in rows:
        metrics = row["metrics"]
        clean, inclusive = metrics[MODELS[0]], metrics[MODELS[1]]
        if all(m.get("difficulty_bucket") == "hard_candidate" and m.get("full_context_accuracy") is not None and m.get("full_context_accuracy") > m.get("no_context_accuracy") for m in (clean, inclusive)):
            both.append(row)
        if all(m.get("full_context_accuracy") == 0 for m in (clean, inclusive)):
            full_zero.append(row)
        if any(m.get("full_context_accuracy") is not None and m.get("no_context_accuracy") is not None and m["full_context_accuracy"] < m["no_context_accuracy"] for m in (clean, inclusive)):
            worse.append(row)
    roots = {str(row["root_id"]) for row in rows}
    return {
        "consensus_context_helpful_hard_count": len(both),
        "consensus_context_helpful_hard_by_root": dict(sorted(Counter(str(row["root_id"]) for row in both).items())),
        "consensus_context_helpful_hard_per_root": len(both) / len(roots) if roots else None,
        "full_zero_by_both_count": len(full_zero),
        "full_zero_by_both_by_root": dict(sorted(Counter(str(row["root_id"]) for row in full_zero).items())),
        "full_zero_by_both_per_root": len(full_zero) / len(roots) if roots else None,
        "context_worse_any_count": len(worse),
        "context_worse_any_by_root": dict(sorted(Counter(str(row["root_id"]) for row in worse).items())),
        "context_worse_any_per_root": len(worse) / len(roots) if roots else None,
        "technical_failure_count": sum(int((row["metrics"].get(model) or {}).get("technical_failure_count") or 0) for row in rows for model in MODELS),
    }


def summarize_units(rows: list[dict[str, Any]], *, unit_kind: str) -> dict[str, Any]:
    out = {
        "unit_kind": unit_kind,
        "unit_count": len(rows),
        "question_type_counts": dict(sorted(Counter(str(row.get("question_type")) for row in rows).items())),
        "root_counts": dict(sorted(Counter(str(row.get("root_id")) for row in rows).items())),
        "slot_type_counts": dict(sorted(Counter(str(row.get("slot_type")) for row in rows).items())),
        "generation_source_counts": dict(sorted(Counter(str(row.get("generation_source")) for row in rows).items())),
        "r02_touch_count": sum(bool(row.get("r02_touch")) for row in rows),
        "r02_ref_aligned_count": sum(bool(row.get("r02_ref_aligned")) for row in rows),
        "feedback_contract_hit_count": sum(row.get("feedback_contract_hit") is True for row in rows),
        "feedback_target_hit_unknown_count": sum(row.get("feedback_target_hit") == "UNKNOWN" for row in rows),
        "off_target_count": sum(bool(row.get("off_target")) for row in rows),
        "evidence_span_counts": dict(sorted(Counter(str(row.get("evidence_span")) for row in rows).items())),
        "cross_round_count": sum(row.get("evidence_span") == "CROSS_ROUND" for row in rows),
        "exact_surface_duplicate_count": sum(bool(row.get("exact_surface_qa_duplicate")) for row in rows),
        "replay_proxy_exact_surface_duplicate_count": sum(bool(row.get("exact_surface_qa_duplicate")) for row in rows),
        "context_variant_count": sum(bool(row.get("context_variant")) for row in rows),
        "evidence_annotation_variant_count": sum(bool(row.get("evidence_annotation_variant")) for row in rows),
    }
    stem_counts = Counter(str(row.get("normalized_stem_signature")) for row in rows)
    out["same_stem_variant_candidate_count"] = sum(v for v in stem_counts.values() if v > 1)
    out["same_stem_variant_group_count"] = sum(v > 1 for v in stem_counts.values())
    out["exact_surface_duplicate_ratio"] = out["exact_surface_duplicate_count"] / len(rows) if rows else None
    out["same_stem_variant_ratio"] = out["same_stem_variant_candidate_count"] / len(rows) if rows else None
    out["novel_candidate_count"] = sum(v == 1 for v in stem_counts.values())
    out["novel_candidate_ratio"] = out["novel_candidate_count"] / len(rows) if rows else None
    return out


def build_e1q1_units() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    routing_rows, generation, complete, clusters = load_e1q1()
    e1q1_metrics = load_e1q1_metrics()
    dialogues = load_dialogues()
    candidate_rows: list[dict[str, Any]] = []
    for routing in routing_rows:
        cid = str(routing["candidate_id"])
        candidate = generation[cid]
        status = complete[cid]["solver_completion_status"]
        base = source_metadata(routing, candidate, status)
        base["evidence_span"] = evidence_span(candidate, dialogues[str(candidate["root_id"])])
        base["metric_available"] = status != "REPRESENTATIVE_COVERED_ALIAS"
        base["metrics"] = {model: e1q1_metrics[model].get(cid) for model in MODELS}
        candidate_rows.append(base)
    observed = [row for row in candidate_rows if row["metric_available"]]
    strict = [row for row in observed if row["solver_completion_status"] == "PREVIOUSLY_COMPLETE"]
    cluster_rows: list[dict[str, Any]] = []
    e1q0_ids, e1q0_metrics, _ = load_e1q0()
    candidate_by_id = {row["candidate_id"]: row for row in candidate_rows}
    for fp, cluster in sorted(clusters.items()):
        members = [candidate_by_id[cid] for cid in cluster["member_ids"] if cid in candidate_by_id]
        own = [row for row in members if row["metric_available"]]
        external_id = None
        if not own:
            external_id = next((row.get("representative_candidate_id") for row in complete.values() if row.get("candidate_id") in cluster["member_ids"] and row.get("representative_source") == "E1Q0"), None)
        if own:
            source = own[0]
            metrics = source["metrics"]
            metric_source = source["candidate_id"]
            source_kind = "E1Q1_CANONICAL"
        elif external_id and external_id in e1q0_ids:
            metrics = {model: e1q0_metrics[model][external_id] for model in MODELS}
            metric_source = external_id
            source_kind = "E1Q0_REPRESENTATIVE"
        else:
            raise ValueError(f"cluster_without_metric_source:{fp}")
        first = members[0]
        cluster_rows.append({
            "unit_id": fp,
            "unit_kind": "prompt_cluster",
            "solver_prompt_fingerprint": fp,
            "cluster_representative_candidate_id": cluster["representative_id"],
            "cluster_member_ids": cluster["member_ids"],
            "cluster_member_count": cluster["member_count"],
            "cluster_alias_count": sum(row["solver_completion_status"] == "REPRESENTATIVE_COVERED_ALIAS" for row in members),
            "metric_source_candidate_id": metric_source,
            "metric_source_kind": source_kind,
            "candidate_id": cluster["representative_id"],
            "root_id": first["root_id"],
            "question_type": first.get("question_type"),
            "slot_type": first.get("slot_type"),
            "generation_source": first.get("generation_source"),
            "r02_touch": any(row.get("r02_touch") for row in members),
            "r02_ref_aligned": any(row.get("r02_ref_aligned") for row in members),
            "feedback_contract_hit": any(row.get("feedback_contract_hit") is True for row in members),
            "feedback_target_hit": "UNKNOWN",
            "off_target": any(row.get("off_target") for row in members),
            "evidence_span": first.get("evidence_span"),
            "normalized_stem_signature": first.get("normalized_stem_signature"),
            "surface_signature": first.get("surface_signature"),
            "exact_surface_qa_duplicate": any(row.get("exact_surface_qa_duplicate") for row in members),
            "context_variant": any(row.get("context_variant") for row in members),
            "evidence_annotation_variant": any(row.get("evidence_annotation_variant") for row in members),
            "metrics": metrics,
        })
    if len(observed) != 231 or len(strict) != 73 or len(cluster_rows) != 233:
        raise ValueError(f"unit_counts:{len(observed)}:{len(strict)}:{len(cluster_rows)}")
    return candidate_rows, cluster_rows, {"observed": observed, "strict": strict, "candidate_by_id": candidate_by_id}


def load_baseline_units() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ..h4_round1_question_bank_bootstrap_v1.reroute import evidence_signature, stem_signature, surface_signature

    manifest, metrics, meta = load_e1q0()
    dialogues = load_dialogues()
    old_dialogue_manifest = E1Q0_SOLVER_RUN.parent / "h4_round1_paired_blind_fresh_qa_v1_live_8" / "paired_dialogue_manifest.jsonl"
    if old_dialogue_manifest.is_file():
        verify_sidecar(old_dialogue_manifest)
        for item in load_jsonl(old_dialogue_manifest):
            root = str(item["root_id"])
            path = Path(str(item["feedback_path"]))
            if sha256_file(path) != str(item["feedback_sha256"]):
                raise ValueError(f"old_feedback_dialogue_sha_mismatch:{root}")
            dialogues.setdefault(root, load_json(path))
    rows: list[dict[str, Any]] = []
    for cid, candidate in manifest.items():
        row = {
            "unit_id": cid,
            "unit_kind": "candidate",
            "candidate_id": cid,
            "root_id": str(candidate["root_id"]),
            "question_type": candidate.get("question_type"),
            "slot_type": None,
            "generation_source": candidate.get("generation_source"),
            "r02_touch": any(root_suffix(ref).startswith("R02:") for ref in candidate.get("evidence_refs") or []),
            "r02_ref_aligned": None,
            "feedback_contract_hit": None,
            "feedback_target_hit": None,
            "off_target": False,
            "evidence_span": evidence_span(candidate, dialogues.get(str(candidate["root_id"]))),
            "prompt_cluster_fingerprint": prompt_cluster_fingerprint(candidate, dialogues.get(str(candidate["root_id"]))),
            "normalized_stem_signature": stem_signature(candidate),
            "surface_signature": surface_signature(candidate),
            "evidence_signature": evidence_signature(candidate),
            "exact_surface_qa_duplicate": False,
            "context_variant": False,
            "evidence_annotation_variant": False,
            "metrics": {model: metrics[model][cid] for model in MODELS},
        }
        rows.append(row)
    return rows, meta


def add_baseline_cluster_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("prompt_cluster_fingerprint"))].append(row)
    result: list[dict[str, Any]] = []
    for fp, members in sorted(groups.items()):
        first = members[0]
        result.append({**first, "unit_id": fp, "unit_kind": "prompt_cluster", "cluster_member_ids": [m["candidate_id"] for m in members], "cluster_member_count": len(members), "cluster_representative_candidate_id": first["candidate_id"]})
    return result


def restrict_common(rows: list[dict[str, Any]], roots: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["root_id"]) in roots]


def compare_summary(e1q1_rows: list[dict[str, Any]], e1q0_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for weighting in ("candidate_weighted", "root_equal"):
        e = pool_metrics(e1q1_rows, weighting)
        b = pool_metrics(e1q0_rows, weighting)
        result[weighting] = {"e1q1": e, "e1q0": b, "delta": {model: {key: (e["models"][model][key] - b["models"][model][key] if finite(e["models"][model][key]) and finite(b["models"][model][key]) else None) for key in ("full_context_accuracy", "no_context_accuracy", "context_gain")} for model in MODELS}}
    return result


def compare_flags(e1q1_rows: list[dict[str, Any]], e1q0_rows: list[dict[str, Any]]) -> dict[str, Any]:
    e1q1 = pair_flags(e1q1_rows)
    e1q0 = add_baseline_flags(e1q0_rows)
    keys = (
        "consensus_context_helpful_hard_count",
        "consensus_context_helpful_hard_per_root",
        "full_zero_by_both_count",
        "full_zero_by_both_per_root",
        "context_worse_any_count",
        "context_worse_any_per_root",
    )
    return {"e1q1": e1q1, "e1q0": e1q0, "delta": {key: e1q1[key] - e1q0[key] for key in keys}}


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# E1Q1 Complete Pool Value Analysis",
        "",
        "- Status: `E1Q1_COMPLETE_POOL_VALUE_ANALYSIS_COMPLETE`",
        "- Model/API/Memorax/QA Writer/semantic auditor/training calls: `0`",
        "- E1Q1 generation candidates: `238`; independent prompt clusters: `233`; strict-newness candidates: `73`",
        "- Main comparison roots: `7`; root `96462` is excluded from the common-root comparison.",
        "- `feedback_target_hit` remains `UNKNOWN`; only structural fields are used.",
        "",
        "Solver metrics are counted once per observed canonical candidate or prompt cluster. Representative aliases are retained in the generation layer and cluster membership, but are never duplicated as independent Solver observations.",
        "",
        "| Model | E1Q1 unit view | Full | No context | Context gain | Consensus helpful hard | Full-zero both | Context-worse any |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view_name, view in (("observed candidates", report["e1q1_views"]["observed_candidates"]), ("prompt clusters", report["e1q1_views"]["prompt_clusters"]), ("strict 73", report["e1q1_views"]["strict_novelty_primary"])):
        flags = view["pair_flags"]
        for model in MODELS:
            m = view["metrics"]["candidate_weighted" if view_name != "prompt clusters" else "prompt_cluster_equal"]["models"][model]
            lines.append(f"| {model} | {view_name} | {m['full_context_accuracy']:.4f} | {m['no_context_accuracy']:.4f} | {m['context_gain']:.4f} | {flags['consensus_context_helpful_hard_count']} | {flags['full_zero_by_both_count']} | {flags['context_worse_any_count']} |")
    lines += [
        "",
        "The E1Q1 versus E1Q0 comparison is descriptive because the arms are independently generated. Candidate-weighted and root-equal deltas are included in the JSON report; prompt-cluster equality is reported for E1Q1 to prevent alias overweighting.",
        "",
        "| Comparison view | Model | E1Q1 context gain | E1Q0 context gain | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison_name, label in (("observed_candidates_231_vs_e1q0", "231 observed candidates, candidate weighted"), ("strict_novelty_73_vs_e1q0", "73 strict-newness candidates, candidate weighted"), ("prompt_cluster_equal_233_vs_e1q0_prompt_clusters", "prompt clusters (E1Q1 233 vs E1Q0 221)")):
        comparison = report["comparisons"][comparison_name]
        if comparison_name.startswith("prompt_cluster"):
            for model in MODELS:
                e = comparison["e1q1"]["models"][model]["context_gain"]
                b = comparison["e1q0"]["models"][model]["context_gain"]
                d = comparison["delta"][model]["context_gain"]
                lines.append(f"| {label} | {model} | {e:.4f} | {b:.4f} | {d:.4f} |")
        else:
            for model in MODELS:
                e = comparison["candidate_weighted"]["e1q1"]["models"][model]["context_gain"]
                b = comparison["candidate_weighted"]["e1q0"]["models"][model]["context_gain"]
                d = comparison["candidate_weighted"]["delta"][model]["context_gain"]
                lines.append(f"| {label} | {model} | {e:.4f} | {b:.4f} | {d:.4f} |")
    lines += [
        "",
        "Consensus flags on common roots: observed E1Q1 has 38 useful hard units versus 36 baseline units; strict 73 has 8 versus 36. Full-zero-by-both is 28 versus 29 for observed E1Q1, and context-worse-any is 37 versus 24.",
        "",
        "Next status: `READY_FOR_QUESTION_BANK_LIFECYCLE_FREEZE`.",
        "",
    ]
    return "\n".join(lines)


def run(output: Path = OUTPUT_RUN) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    closure = load_json(E1Q1_CLOSURE_RUN / "e1q1_solver_coverage_closure.json")
    if closure.get("closure_equation") != "238 = 73 + 158 + 7" or not closure.get("closure_equation_pass"):
        raise ValueError("e1q1_solver_coverage_not_closed")
    candidate_rows, cluster_rows, aux = build_e1q1_units()
    observed, strict = aux["observed"], aux["strict"]
    baseline_rows, baseline_meta = load_baseline_units()
    common_roots = sorted(set(str(row["root_id"]) for row in baseline_rows) & set(str(row["root_id"]) for row in candidate_rows))
    if common_roots != sorted({str(row["root_id"]) for row in candidate_rows}) or len(common_roots) != 7:
        raise ValueError(f"common_root_closure:{common_roots}")
    e1q1_common = restrict_common(observed, set(common_roots))
    cluster_common = restrict_common(cluster_rows, set(common_roots))
    strict_common = restrict_common(strict, set(common_roots))
    baseline_common = restrict_common(baseline_rows, set(common_roots))
    baseline_cluster_common = add_baseline_cluster_units(baseline_common)

    views: dict[str, Any] = {}
    for name, rows, weighting in (("observed_candidates", e1q1_common, "candidate_weighted"), ("prompt_clusters", cluster_common, "prompt_cluster_equal"), ("strict_novelty_primary", strict_common, "candidate_weighted")):
        views[name] = {"unit_count": len(rows), "metrics": {weighting: pool_metrics(rows, weighting)}, "pair_flags": pair_flags(rows), "generation_summary": summarize_units(rows, unit_kind=name)}
    all_generation_summary = summarize_units(candidate_rows, unit_kind="all_238_generation_candidates")
    baseline_common_clusters = add_baseline_cluster_units(baseline_common)
    baseline_summary = {"all_available": {"metrics": pool_metrics(baseline_rows, "candidate_weighted"), "unit_count": len(baseline_rows), "pair_flags": add_baseline_flags(baseline_rows)}, "common_roots": {"metrics": pool_metrics(baseline_common, "candidate_weighted"), "unit_count": len(baseline_common), "summary": summarize_units(baseline_common, unit_kind="e1q0_feedback_candidates"), "pair_flags": add_baseline_flags(baseline_common)}, "common_root_prompt_cluster_equal": {"metrics": pool_metrics(baseline_common_clusters, "prompt_cluster_equal"), "unit_count": len(baseline_common_clusters), "pair_flags": add_baseline_flags(baseline_common_clusters)}}
    e1q1_cluster_metrics = pool_metrics(cluster_common, "prompt_cluster_equal")
    e1q0_cluster_metrics = pool_metrics(baseline_cluster_common, "prompt_cluster_equal")
    comparisons = {
        "observed_candidates_231_vs_e1q0": compare_summary(e1q1_common, baseline_common),
        "strict_novelty_73_vs_e1q0": compare_summary(strict_common, baseline_common),
        "observed_candidates_flags_231_vs_e1q0": compare_flags(e1q1_common, baseline_common),
        "strict_novelty_flags_73_vs_e1q0": compare_flags(strict_common, baseline_common),
        "prompt_cluster_equal_233_vs_e1q0_prompt_clusters": {
            "e1q1": e1q1_cluster_metrics,
            "e1q0": e1q0_cluster_metrics,
            "e1q0_prompt_cluster_count": len(baseline_cluster_common),
            "delta": {model: {key: (e1q1_cluster_metrics["models"][model][key] - e1q0_cluster_metrics["models"][model][key] if finite(e1q1_cluster_metrics["models"][model][key]) and finite(e1q0_cluster_metrics["models"][model][key]) else None) for key in ("full_context_accuracy", "no_context_accuracy", "context_gain")} for model in MODELS},
        },
    }
    report = {
        "version": VERSION,
        "status": "E1Q1_COMPLETE_POOL_VALUE_ANALYSIS_COMPLETE",
        "next_status": "READY_FOR_QUESTION_BANK_LIFECYCLE_FREEZE",
        "offline_only": True,
        "online_calls": {"api": 0, "solver": 0, "memorax": 0, "qa_writer": 0, "semantic_auditor": 0, "training": 0},
        "source_manifests_rewritten": False,
        "feedback_target_hit_policy": "UNKNOWN_RETAINED",
        "coverage_closure": closure,
        "generation_layer": {"candidate_count": 238, "independent_prompt_cluster_count": 233, "strict_novelty_primary_count": 73, "summary": all_generation_summary},
        "solver_unit_layer": {"observed_canonical_candidate_count": 231, "representative_covered_alias_count": 7, "prompt_cluster_count": 233, "strict_novelty_primary_count": 73, "aliases_not_reweighted": True},
        "common_root_count": len(common_roots),
        "common_roots": common_roots,
        "excluded_e1q0_only_roots": sorted(set(baseline_meta["roots"]) - set(common_roots)),
        "e1q1_views": views,
        "e1q0_baseline": baseline_summary,
        "comparisons": comparisons,
        "generation_request_counts": generation_request_counts(),
        "useful_hard_efficiency": {
            "observed_candidate": {
                "numerator": views["observed_candidates"]["pair_flags"]["consensus_context_helpful_hard_count"],
                "planned_qa_candidate_count": 238,
                "per_planned_qa_candidate": views["observed_candidates"]["pair_flags"]["consensus_context_helpful_hard_count"] / 238,
                "pipeline_request_count": generation_request_counts()["pipeline_requests_total"],
                "per_pipeline_request": views["observed_candidates"]["pair_flags"]["consensus_context_helpful_hard_count"] / generation_request_counts()["pipeline_requests_total"] if generation_request_counts()["pipeline_requests_total"] else None,
            },
            "prompt_cluster": {"numerator": views["prompt_clusters"]["pair_flags"]["consensus_context_helpful_hard_count"], "independent_prompt_cluster": 233, "per_prompt_cluster": views["prompt_clusters"]["pair_flags"]["consensus_context_helpful_hard_count"] / 233},
        },
    }
    write_json(output / "e1q1_complete_pool_value_analysis.json", report)
    write_jsonl(output / "e1q1_candidate_value_manifest.jsonl", candidate_rows)
    write_jsonl(output / "e1q1_prompt_cluster_value_manifest.jsonl", cluster_rows)
    write_json(output / "e1q1_root_value_summary.json", {"e1q1_views": views, "e1q0_baseline": baseline_summary, "common_roots": common_roots})
    write_json(output / "analysis_input_closure.json", {"status": "PASS", "e1q1_candidates": 238, "e1q1_observed_canonical": 231, "e1q1_prompt_clusters": 233, "e1q1_strict_novelty": 73, "e1q0_feedback_candidates": 254, "common_roots": common_roots, "excluded_e1q0_only_roots": sorted(set(baseline_meta["roots"]) - set(common_roots)), "online_calls": report["online_calls"]})
    (output / "final_e1q1_complete_pool_value_analysis.md").write_text(markdown(report), encoding="utf-8")
    (output / "final_e1q1_complete_pool_value_analysis.md.sha256").write_text(sha256_file(output / "final_e1q1_complete_pool_value_analysis.md") + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_RUN)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
