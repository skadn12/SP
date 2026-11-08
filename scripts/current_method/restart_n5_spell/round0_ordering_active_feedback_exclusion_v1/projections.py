"""Pure projections of frozen cluster observations for active feedback only."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Any


MODELS = ("M_clean_policy", "M_inclusive_policy")
ACTIVE_ELIGIBILITY_KEYS = (
    "questioner_targeted_evolution_eligible",
    "questioner_bridge_generation_eligible",
    "composition_parent_eligible",
)


def _index(rows: list, key: str, name: str) -> dict:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate_identity:{name}")
    return result


def _candidate_root(candidate: dict) -> str:
    return str(candidate.get("root_id") or (candidate.get("candidate") or {}).get("root_id") or "")


def _independent(origin: dict, candidate_id: str) -> bool:
    return bool(
        origin.get("source_independent_observation_available")
        and origin.get("counted_as_independent_observation")
        and origin.get("canonical_representative_candidate_id") == candidate_id
        and origin.get("observation_origin") == "INDEPENDENT_OBSERVATION"
    )


def _valid_metric(metric: Any) -> bool:
    if not isinstance(metric, dict):
        return False
    for prefix in ("full", "no"):
        correct, total = metric.get(f"{prefix}_correct"), metric.get(f"{prefix}_total")
        if isinstance(correct, bool) or isinstance(total, bool):
            return False
        if not isinstance(correct, int) or not isinstance(total, int) or not 0 <= correct <= total or total <= 0:
            return False
    return True


def _normalize(rows: list) -> None:
    global_total = sum(row["raw_weight"] for row in rows)
    root_totals = Counter()
    for row in rows:
        root_totals[row["root_id"]] += row["raw_weight"]
    for row in rows:
        raw, root_total = row["raw_weight"], root_totals[row["root_id"]]
        row["global_normalized_weight"] = raw / global_total if global_total else 0.0
        row["within_root_normalized_weight"] = raw / root_total if root_total else 0.0


def _project_weight(source: dict, cluster: dict, family_count: int, eligible: bool) -> dict:
    row = copy.deepcopy(source)
    row.update({
        "canonical_representative_candidate_id": cluster["canonical_representative_candidate_id"],
        "root_id": cluster["root_id"],
        "lifecycle_state": cluster["lifecycle_state"],
        "eligible": eligible,
        "active_feedback_only": True,
        "active_family_cluster_count": family_count,
        "family_saturation_multiplier": 1.0 / math.sqrt(max(1, family_count)),
        "weight_status": "REBUILT_FROM_FROZEN_INDIVIDUAL_COMPONENTS",
    })
    if cluster["active_observation_status"] != "RETAINED_INDEPENDENT_OBSERVATION":
        for key in tuple(row):
            if key in {"base_score", "mean_frontier_uncertainty", "mean_positive_context_gain", "transition_bonus", "exposure_bonus", "cross_round_bonus", "multi_round_evidence_bonus", "evidence_multiplier", "additive_score"}:
                row[key] = None
        row.update({"eligible": False, "raw_weight": 0.0, "weight_status": "UNKNOWN_OBSERVATION_ORIGIN"})
        return row
    components = ("additive_score", "evidence_multiplier", "cooldown_multiplier", "representative_multiplier")
    if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in components):
        raise ValueError(f"unreconstructible_weight_components:{cluster['solver_prompt_fingerprint']}")
    row["raw_weight"] = (
        row["additive_score"] * row["evidence_multiplier"]
        * row["family_saturation_multiplier"] * row["cooldown_multiplier"]
        * row["representative_multiplier"]
    ) if eligible else 0.0
    return row


def _root_aggregates(clusters: list, candidates: dict, targeted: list, composition: list) -> list:
    grouped = defaultdict(list)
    for cluster in clusters:
        grouped[cluster["root_id"]].append(cluster)
    roots = sorted({_candidate_root(candidate) for candidate in candidates.values()} - {""})
    output = []
    for root in roots:
        rows = grouped[root]
        known = [row for row in rows if row["active_observation_status"] == "RETAINED_INDEPENDENT_OBSERVATION"]
        unknown_count = len(rows) - len(known)
        models = {}
        for model in MODELS:
            valid = [row["metrics"][model] for row in known if _valid_metric(row["metrics"].get(model))]
            missing_count = len(rows) - len(valid)
            totals = {key: sum(metric[key] for metric in valid) for key in ("full_correct", "full_total", "no_correct", "no_total")}
            complete = bool(rows) and missing_count == 0
            models[model] = {
                "status": "COMPLETE" if complete else "UNKNOWN",
                "independent_observation_count": len(valid),
                "missing_or_unseparable_cluster_count": missing_count,
                **{key: value if complete else None for key, value in totals.items()},
                "full": totals["full_correct"] / totals["full_total"] if complete and totals["full_total"] else None,
                "no": totals["no_correct"] / totals["no_total"] if complete and totals["no_total"] else None,
                "known_subset_counts": totals,
            }
        output.append({
            "root_id": root,
            "active_feedback_only": True,
            "aggregation_unit": "retained_independent_cluster_observation",
            "aggregation_method": "sum_correct_counts_divided_by_sum_total_counts",
            "status": "COMPLETE" if rows and all(value["status"] == "COMPLETE" for value in models.values()) else "UNKNOWN",
            "retained_candidate_count": sum(len(row["member_candidate_ids"]) for row in rows),
            "retained_cluster_count": len(rows),
            "independent_observation_count": len(known),
            "unknown_observation_cluster_count": unknown_count,
            "lifecycle_counts": dict(sorted(Counter(row["lifecycle_state"] for row in rows).items())),
            "models": models,
            "targeted_eligible_cluster_count": sum(row["eligible"] for row in targeted if row["root_id"] == root),
            "targeted_raw_weight_sum": sum(row["raw_weight"] for row in targeted if row["root_id"] == root),
            "composition_eligible_cluster_count": sum(row["eligible"] for row in composition if row["root_id"] == root),
            "composition_raw_weight_sum": sum(row["raw_weight"] for row in composition if row["root_id"] == root),
        })
    return output


def project_clusters(
    *, clusters: list, members: list, origins: list, eligibility: list,
    target_weights: list, composition_weights: list, excluded_ids: set,
    candidates: dict,
) -> dict:
    """Mask members before deriving weights; never mutate a frozen input.

    ``clusters`` are cluster_lifecycle_manifest rows, ``members`` are
    candidate_to_cluster_map rows, and ``origins`` are observation-origin rows.
    Metrics are representative observations, never means over member aliases.
    A removed representative cannot lend its metrics to a surviving alias.
    """
    member_index = _index(members, "candidate_id", "members")
    members_by_cluster = defaultdict(set)
    for row in members:
        members_by_cluster[row["solver_prompt_fingerprint"]].add(row["candidate_id"])
    origin_index = _index(origins, "candidate_id", "origins")
    eligibility_index = _index(eligibility, "solver_prompt_fingerprint", "eligibility")
    target_index = _index(target_weights, "solver_prompt_fingerprint", "target_weights")
    composition_index = _index(composition_weights, "solver_prompt_fingerprint", "composition_weights")
    _index(clusters, "solver_prompt_fingerprint", "clusters")
    projected, audits = [], []
    for source in clusters:
        fp = source["solver_prompt_fingerprint"]
        original_members = set(source["member_candidate_ids"])
        representative = source["canonical_representative_candidate_id"]
        if original_members != members_by_cluster[fp]:
            raise ValueError(f"cluster_membership_or_origin_mismatch:{fp}")
        if representative not in original_members:
            raise ValueError(f"representative_not_member:{fp}")
        for cid in original_members:
            link = member_index.get(cid)
            origin = origin_index.get(cid)
            if cid not in candidates or not link or not origin or link["solver_prompt_fingerprint"] != fp or origin["solver_prompt_fingerprint"] != fp:
                raise ValueError(f"cluster_membership_or_origin_mismatch:{cid}")
            if link["canonical_representative_candidate_id"] != representative or origin["canonical_representative_candidate_id"] != representative:
                raise ValueError(f"cluster_representative_origin_mismatch:{cid}")
        retained = sorted(original_members - excluded_ids)
        removed = sorted(original_members & excluded_ids)
        audit = {
            "solver_prompt_fingerprint": fp,
            "original_representative_candidate_id": representative,
            "original_member_candidate_ids": sorted(original_members),
            "retained_member_candidate_ids": retained,
            "excluded_member_candidate_ids": removed,
            "mixed_scope_cluster": bool(retained and removed),
        }
        if not retained:
            audit.update({"status": "EXCLUDED_ALL_MEMBERS", "active_observation_candidate_id": None})
            audits.append(audit)
            continue
        # Frozen metrics belong only to the canonical representative.  A flag
        # saying an alias once had data does not provide that member's metrics.
        reliable = (
            representative in retained
            and _independent(origin_index[representative], representative)
            and all(_valid_metric((source.get("metrics") or {}).get(model)) for model in MODELS)
        )
        status = "RETAINED_INDEPENDENT_OBSERVATION" if reliable else "UNKNOWN_UNSEPARABLE_OBSERVATION_ORIGIN"
        row = copy.deepcopy(source)
        row.update({
            "member_candidate_ids": retained,
            "member_count": len(retained),
            "alias_count": len(retained) - 1,
            "active_feedback_only": True,
            "active_observation_status": status,
            "active_observation_candidate_id": representative if reliable else None,
        })
        if not reliable:
            retained_rep = min(retained)
            candidate = candidates[retained_rep]
            question = candidate.get("candidate") or candidate
            row.update({
                "canonical_representative_candidate_id": retained_rep,
                "root_id": _candidate_root(candidate),
                "normalized_experiment_arm": candidate.get("normalized_experiment_arm"),
                "question_type": question.get("question_type"),
                "evidence_refs": copy.deepcopy(question.get("evidence_refs") or []),
                "evidence_status": "UNKNOWN",
                "evidence_rounds": [],
                "position_mechanism": "UNKNOWN",
                "metrics": {model: None for model in MODELS},
                "auxiliary_labels": {},
                "lifecycle_state": "UNKNOWN_ACTIVE_OBSERVATION_AFTER_SCOPE_FILTER",
                "feedback_target_hit": "UNKNOWN",
            })
        audit.update({"status": status, "active_observation_candidate_id": row["active_observation_candidate_id"]})
        audits.append(audit)
        projected.append(row)

    families = defaultdict(list)
    for row in projected:
        families[(row["root_id"], str(row["normalized_stem_signature"]))].append(row)
    active_families = [
        {
            "root_id": root, "normalized_stem_signature": stem,
            "candidate_count": sum(row["member_count"] for row in rows),
            "prompt_cluster_count": len(rows),
            "family_saturation_multiplier": 1.0 / math.sqrt(len(rows)),
            "active_feedback_only": True,
            "historical_coverage_replacement": False,
        }
        for (root, stem), rows in sorted(families.items())
    ]
    projected_eligibility, projected_target, projected_composition = [], [], []
    for cluster in projected:
        fp = cluster["solver_prompt_fingerprint"]
        if fp not in eligibility_index or fp not in target_index or fp not in composition_index:
            raise ValueError(f"missing_cluster_active_inputs:{fp}")
        source = eligibility_index[fp]
        reliable = cluster["active_observation_status"] == "RETAINED_INDEPENDENT_OBSERVATION"
        eligible = {
            "solver_prompt_fingerprint": fp,
            "canonical_representative_candidate_id": cluster["canonical_representative_candidate_id"],
            "root_id": cluster["root_id"],
            "normalized_experiment_arm": cluster["normalized_experiment_arm"],
            "lifecycle_state": cluster["lifecycle_state"],
            "active_feedback_only": True,
            "eligibility_status": "REUSED_INDIVIDUAL_RULES_AFTER_SCOPE_MASK" if reliable else "UNKNOWN_OBSERVATION_ALL_ACTIVE_USES_DISABLED",
            **{key: bool(source.get(key)) and reliable for key in ACTIVE_ELIGIBILITY_KEYS},
        }
        projected_eligibility.append(eligible)
        family_count = len(families[(cluster["root_id"], str(cluster["normalized_stem_signature"]))])
        projected_target.append(_project_weight(target_index[fp], cluster, family_count, eligible["questioner_targeted_evolution_eligible"]))
        projected_composition.append(_project_weight(composition_index[fp], cluster, family_count, eligible["composition_parent_eligible"]))
    _normalize(projected_target)
    _normalize(projected_composition)
    return {
        "clusters": projected,
        "cluster_audit": audits,
        "eligibility": projected_eligibility,
        "target_weights": projected_target,
        "composition_weights": projected_composition,
        "root_aggregates": _root_aggregates(projected, candidates, projected_target, projected_composition),
        "active_families": active_families,
    }
