"""Offline loading, profiling, sampling, and permutation construction.

The module is deliberately independent from production QA admission.  It only
reads the frozen K=3 probe and writes immutable manifests below a new run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import canonical_json, sha256_file


SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_batch30_memory_event_k3_probe_v1_live_1"
)
PROBE_VERSION = "h4_position_permutation_ab_probe_v1"
SAMPLE_ALGORITHM = "position_ablation_stratified_sample_v1"
PERMUTATION_ALGORITHM = "position_ablation_permutation_manifest_v1"
MODEL_LEN = 16_384
OUTPUT_TOKENS_LABEL = 32
QUESTION_TYPES = ("single_choice", "multi_select", "ordering")
TYPE_QUOTAS = {"single_choice": 6, "multi_select": 12, "ordering": 12}
SOURCE_TARGETS = {"contract_guided": 11, "free_exploration": 19}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _obj(path: Path) -> dict[str, Any]:
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_score(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _candidate_list(path: Path) -> list[dict[str, Any]]:
    value = _json(path)
    if not isinstance(value, list):
        raise ValueError(f"candidate_pool_must_be_list:{path}")
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _difficulty_summary(path: Path) -> list[dict[str, Any]]:
    value = _obj(path)
    rows = value.get("candidate_summaries")
    if not isinstance(rows, list):
        raise ValueError(f"difficulty_summaries_missing:{path}")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _solver_results(path: Path) -> dict[str, dict[str, Any]]:
    value = _obj(path)
    rows = value.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"solver_results_missing:{path}")
    return {
        str(row["candidate_id"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("candidate_id")
    }


def _validate_rollout_history(candidate: Mapping[str, Any], solver: Mapping[str, Any]) -> list[str]:
    """Require recoverable per-position, per-slot history; never infer it."""
    errors: list[str] = []
    qtype = str(candidate.get("question_type") or "")
    expected_positions = {"single_choice": 4, "multi_select": 5, "ordering": 4}.get(qtype)
    if expected_positions is None:
        return ["unsupported_question_type"]
    variants = solver.get("variants")
    if not isinstance(variants, list) or len(variants) != expected_positions * 2:
        errors.append(f"variant_count:{len(variants) if isinstance(variants, list) else 'missing'}:{expected_positions * 2}")
        return errors
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for variant in variants:
        if not isinstance(variant, Mapping):
            errors.append("variant_not_object")
            continue
        by_mode[str(variant.get("mode"))].append(variant)
        summary = variant.get("summary")
        if not isinstance(summary, Mapping):
            errors.append("position_summary_missing")
            continue
        if int(summary.get("requested_total") or 0) != 3:
            errors.append("requested_rollout_count_not_3")
        if not summary.get("complete"):
            errors.append("historical_position_incomplete")
        rollouts = summary.get("rollouts")
        if not isinstance(rollouts, list) or len(rollouts) != 3:
            errors.append("rollout_history_missing")
            continue
        for rollout in rollouts:
            if not isinstance(rollout, Mapping):
                errors.append("rollout_not_object")
                continue
            for key in ("raw_output", "parsed_answer", "correct", "seed"):
                if key not in rollout:
                    errors.append(f"rollout_field_missing:{key}")
    for mode in ("no_context", "full_context"):
        if len(by_mode.get(mode, [])) != expected_positions:
            errors.append(f"mode_position_count:{mode}")
    return sorted(set(errors))


def load_hard_pool(source_run: Path = SOURCE_RUN) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and fail closed on the exact 71-row hard pool contract."""
    source_run = source_run.resolve()
    candidates: dict[str, dict[str, Any]] = {}
    difficulties: dict[str, dict[str, Any]] = {}
    solver_by_id: dict[str, dict[str, Any]] = {}
    lineage: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    root_dirs = sorted(source_run.glob("shards/*/qa/roots/*"))
    for root_dir in root_dirs:
        files = {
            "candidate_pool": root_dir / "candidate_pool.json",
            "difficulty": root_dir / "difficulty.json",
            "dialogue": root_dir / "expanded_dialogue_clean.json",
            "solver_results": root_dir / "solver_results.json",
        }
        if not all(path.is_file() for path in files.values()):
            continue
        try:
            pool = _candidate_list(files["candidate_pool"])
            summary_rows = _difficulty_summary(files["difficulty"])
            solver_rows = _solver_results(files["solver_results"])
        except Exception as exc:
            errors.append(f"{root_dir.name}:load:{type(exc).__name__}:{exc}")
            continue
        root_id = root_dir.name
        for row in pool:
            cid = str(row.get("candidate_id") or "")
            if not cid:
                errors.append(f"{root_id}:candidate_id_missing")
                continue
            if cid in candidates:
                errors.append(f"duplicate_candidate_id:{cid}")
            candidates[cid] = row
            lineage[cid] = {
                "candidate_pool_path": str(files["candidate_pool"]),
                "candidate_pool_sha256": sha256_file(files["candidate_pool"]),
                "difficulty_path": str(files["difficulty"]),
                "difficulty_sha256": sha256_file(files["difficulty"]),
                "dialogue_path": str(files["dialogue"]),
                "dialogue_sha256": sha256_file(files["dialogue"]),
                "solver_results_path": str(files["solver_results"]),
                "solver_results_sha256": sha256_file(files["solver_results"]),
            }
        for row in summary_rows:
            cid = str(row.get("candidate_id") or "")
            if cid in difficulties:
                errors.append(f"duplicate_difficulty_candidate:{cid}")
            difficulties[cid] = row
        for cid, row in solver_rows.items():
            if cid in solver_by_id:
                errors.append(f"duplicate_solver_candidate:{cid}")
            solver_by_id[cid] = row

    hard: list[dict[str, Any]] = []
    for cid, difficulty in difficulties.items():
        if difficulty.get("difficulty_bucket") != "hard_candidate":
            continue
        candidate = candidates.get(cid)
        solver = solver_by_id.get(cid)
        if candidate is None:
            errors.append(f"difficulty_candidate_missing_from_pool:{cid}")
            continue
        if solver is None:
            errors.append(f"difficulty_candidate_missing_solver:{cid}")
            continue
        root_id = str(candidate.get("root_id") or "")
        if root_id != str(difficulty.get("root_id") or "") or root_id != cid.split(":", 1)[0]:
            errors.append(f"root_identity_mismatch:{cid}")
        if str(candidate.get("question_type") or "") != str(difficulty.get("question_type") or ""):
            errors.append(f"question_type_mismatch:{cid}")
        if str(candidate.get("generation_source") or "") != str(difficulty.get("generation_source") or ""):
            errors.append(f"generation_source_mismatch:{cid}")
        errors.extend(f"{cid}:{error}" for error in _validate_rollout_history(candidate, solver))
        hard.append({
            "candidate": candidate,
            "difficulty": difficulty,
            "solver": solver,
            "root_id": root_id,
            "candidate_id": cid,
            "lineage": lineage.get(cid, {}),
            "candidate_sha256": _sha(candidate),
        })

    counts = {
        "hard_total": len(hard),
        "question_type": dict(sorted(Counter(str(row["candidate"].get("question_type") or "") for row in hard).items())),
        "generation_source": dict(sorted(Counter(str(row["candidate"].get("generation_source") or "") for row in hard).items())),
        "root_count": len({row["root_id"] for row in hard}),
        "root_counts": dict(sorted(Counter(row["root_id"] for row in hard).items())),
    }
    expected = {
        "hard_total": 71,
        "question_type": {"multi_select": 35, "ordering": 27, "single_choice": 9},
        "generation_source": {"contract_guided": 27, "free_exploration": 44},
        "root_count": 8,
    }
    if counts != expected | {"root_counts": counts["root_counts"]}:
        # Compare only the declared fields; root-level 4-12 is checked below.
        for key in ("hard_total", "question_type", "generation_source", "root_count"):
            if counts[key] != expected[key]:
                errors.append(f"hard_pool_{key}_mismatch:{counts[key]}:{expected[key]}")
    if any(value < 4 or value > 12 for value in counts["root_counts"].values()):
        errors.append("root_hard_count_out_of_range")
    if len(counts["root_counts"]) != 8:
        errors.append("root_count_not_8")
    report = {
        "source_run": str(source_run),
        "source_run_sha256": _sha({"path": str(source_run), "files": sorted(lineage.values(), key=lambda x: x.get("candidate_pool_path", ""))}),
        "expected": expected | {"root_hard_count_range": [4, 12]},
        "observed": counts,
        "errors": sorted(set(errors)),
        "passed": not errors,
        "candidate_pool_total": len(candidates),
        "difficulty_total": len(difficulties),
        "solver_total": len(solver_by_id),
    }
    return hard, report


def profile_hard(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    solver = row["solver"]
    qtype = str(candidate.get("question_type") or "")
    positions = 5 if qtype == "multi_select" else 4
    full = [variant for variant in solver["variants"] if variant.get("mode") == "full_context"]
    no = [variant for variant in solver["variants"] if variant.get("mode") == "no_context"]

    def values(rows: Iterable[Mapping[str, Any]]) -> list[float]:
        ordered = sorted(rows, key=lambda x: (int(x.get("shift") or 0), str(x.get("variant") or "")))
        return [float((x.get("summary") or {}).get("accuracy")) for x in ordered]

    def metrics(values_: list[float], source_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        if len(values_) != positions:
            raise ValueError(f"profile_position_count:{len(values_)}:{positions}")
        mean = sum(values_) / len(values_)
        within = sum(4.0 * p * (1.0 - p) for p in values_) / len(values_)
        between = 4.0 * sum((p - mean) ** 2 for p in values_) / len(values_)
        total = 4.0 * mean * (1.0 - mean)
        if abs(total - within - between) > 1e-9:
            raise ValueError("variance_decomposition_failed")
        counts = Counter()
        for variant in sorted(source_rows, key=lambda x: int(x.get("shift") or 0)):
            for rollout in (variant.get("summary") or {}).get("rollouts") or []:
                parsed = json.dumps(rollout.get("parsed_answer") or [], sort_keys=True)
                counts[parsed] += 1
        max_count = max(counts.values()) if counts else 0
        total_answers = sum(counts.values())
        return {
            "position_accuracies": values_,
            "p_bar": mean,
            "within_position_uncertainty": within,
            "between_position_sensitivity": between,
            "total_variance": total,
            "canonical_flip_rate": 1.0 - (max_count / total_answers if total_answers else 0.0),
            "canonical_answer_entropy": (
                -sum((n / total_answers) * __import__("math").log2(n / total_answers) for n in counts.values())
                if total_answers else 0.0
            ),
        }

    full_values = values(full)
    no_values = values(no)
    full_metrics = metrics(full_values, full)
    no_metrics = metrics(no_values, no)
    within = full_metrics["within_position_uncertainty"]
    between = full_metrics["between_position_sensitivity"]
    eps = 1e-9
    if within <= eps and between > eps:
        mechanism = "position_only_hard"
    elif within > eps and between <= eps:
        mechanism = "rollout_only_hard"
    elif within > eps and between > eps:
        mechanism = "mixed_rollout_position_hard"
    else:
        mechanism = "other_or_unclassified"
    return {
        "candidate_id": row["candidate_id"],
        "root_id": row["root_id"],
        "question_type": qtype,
        "generation_source": candidate.get("generation_source"),
        "hard_mechanism": mechanism,
        "full_context": full_metrics,
        "no_context": no_metrics,
        "historical_full_accuracy": full_metrics["p_bar"],
        "historical_no_accuracy": no_metrics["p_bar"],
        "historical_context_gap": full_metrics["p_bar"] - no_metrics["p_bar"],
    }


def _sample_key(row: Mapping[str, Any], source_run_id: str) -> tuple[int, str]:
    return (stable_score(source_run_id, row["root_id"], row["candidate_id"], SAMPLE_ALGORITHM), row["candidate_id"])


def stratified_sample(rows: list[dict[str, Any]], profiles: Mapping[str, Mapping[str, Any]], *, source_run_id: str) -> list[dict[str, Any]]:
    """Deterministic greedy sample satisfying exact type/root bounds."""
    if len(rows) < 30:
        raise ValueError("sample_pool_under_30")
    enriched = []
    for row in rows:
        profile = profiles[row["candidate_id"]]
        enriched.append({**row, "profile": profile, "stable_hash_score": _sample_key(row, source_run_id)[0]})
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    type_counts: Counter[str] = Counter()
    root_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    mechanism_seen: set[str] = set()
    # First establish two per root.  The scoring keeps exact type quotas feasible.
    for _ in range(2):
        for root in sorted({row["root_id"] for row in enriched}):
            options = [row for row in enriched if row["root_id"] == root and row["candidate_id"] not in selected_ids]
            options = [row for row in options if type_counts[row["candidate"]["question_type"]] < TYPE_QUOTAS[row["candidate"]["question_type"]]]
            if not options:
                raise ValueError(f"root_minimum_unmet:{root}")
            options.sort(key=lambda row: (
                -(row["profile"]["hard_mechanism"] not in mechanism_seen),
                -(SOURCE_TARGETS.get(str(row["candidate"].get("generation_source")), 0) - source_counts[str(row["candidate"].get("generation_source"))]),
                _sample_key(row, source_run_id),
            ))
            chosen = options[0]
            selected.append(chosen); selected_ids.add(chosen["candidate_id"])
            qtype = str(chosen["candidate"].get("question_type")); source = str(chosen["candidate"].get("generation_source"))
            type_counts[qtype] += 1; root_counts[root] += 1; source_counts[source] += 1
            mechanism_seen.add(chosen["profile"]["hard_mechanism"])
    while len(selected) < 30:
        options = [row for row in enriched if row["candidate_id"] not in selected_ids and root_counts[row["root_id"]] < 5]
        options = [row for row in options if type_counts[str(row["candidate"].get("question_type"))] < TYPE_QUOTAS[str(row["candidate"].get("question_type"))]]
        if not options:
            raise ValueError("sample_quota_unmet")
        def score(row: Mapping[str, Any]) -> tuple[Any, ...]:
            qtype = str(row["candidate"].get("question_type")); source = str(row["candidate"].get("generation_source"))
            type_need = TYPE_QUOTAS[qtype] - type_counts[qtype]
            source_need = SOURCE_TARGETS.get(source, 0) - source_counts[source]
            root_need = 2 - root_counts[row["root_id"]]
            mech_bonus = row["profile"]["hard_mechanism"] not in mechanism_seen
            return (-type_need, -root_need, -int(mech_bonus), -source_need, _sample_key(row, source_run_id))
        chosen = sorted(options, key=score)[0]
        selected.append(chosen); selected_ids.add(chosen["candidate_id"])
        qtype = str(chosen["candidate"].get("question_type")); source = str(chosen["candidate"].get("generation_source"))
        type_counts[qtype] += 1; root_counts[chosen["root_id"]] += 1; source_counts[source] += 1
        mechanism_seen.add(chosen["profile"]["hard_mechanism"])
    if dict(type_counts) != TYPE_QUOTAS or any(value < 2 or value > 5 for value in root_counts.values()):
        raise ValueError("sample_postcondition_failed")
    return selected


def _canonical_items(candidate: Mapping[str, Any]) -> tuple[list[str], dict[str, str], list[str]]:
    qtype = str(candidate.get("question_type") or "")
    if qtype == "single_choice":
        keys = ["A", "B", "C", "D"]; mapping = {key: str((candidate.get("options") or {})[key]) for key in keys}; gold = [str(candidate["correct_option"])]
    elif qtype == "multi_select":
        keys = ["A", "B", "C", "D", "E"]; mapping = {key: str((candidate.get("options") or {})[key]) for key in keys}; gold = [str(x) for x in candidate.get("gold") or []]
    elif qtype == "ordering":
        keys = ["I1", "I2", "I3", "I4"]; mapping = {key: str((candidate.get("items") or {})[key]) for key in keys}; gold = [str(x) for x in candidate.get("gold") or []]
    else:
        raise ValueError(f"unsupported_question_type:{qtype}")
    if len(mapping) != len(set(mapping.values())):
        raise ValueError(f"duplicate_option_text:{candidate.get('candidate_id')}")
    return keys, mapping, gold


def build_permutations(row: Mapping[str, Any], *, source_run_id: str) -> list[dict[str, Any]]:
    candidate = row["candidate"]
    qtype = str(candidate.get("question_type") or "")
    keys, texts, gold = _canonical_items(candidate)
    all_orders = list(itertools.permutations(keys))
    if qtype == "multi_select":
        cyclic = [tuple(keys[shift:] + keys[:shift]) for shift in range(len(keys))]
        cyclic_set = set(cyclic)
        rest = [order for order in all_orders if order not in cyclic_set]
        rest.sort(key=lambda order: stable_score(source_run_id, row["candidate_id"], "permutation", "|".join(order), PERMUTATION_ALGORITHM))
        orders = cyclic + rest[:25]
    else:
        orders = all_orders
    result: list[dict[str, Any]] = []
    for index, order in enumerate(orders):
        display_labels = keys
        display_to_canonical = {label: canonical for label, canonical in zip(display_labels, order)}
        inverse = {canonical: label for label, canonical in display_to_canonical.items()}
        # The displayed answer is always expressed in the labels shown to the
        # Solver.  This is especially important for ordering: the canonical
        # item IDs must be rebound through the permutation just like MCQ
        # option labels.  Runtime scoring repeats this mapping independently.
        displayed_gold = [inverse[item] for item in gold]
        result.append({
            "candidate_id": row["candidate_id"],
            "root_id": row["root_id"],
            "question_type": qtype,
            "permutation_id": f"perm_{index:03d}",
            "is_cyclic": tuple(order) in {tuple(keys[shift:] + keys[:shift]) for shift in range(len(keys))},
            "display_order": list(order),
            "display_to_canonical_mapping": display_to_canonical,
            "canonical_to_display_mapping": inverse,
            "displayed_gold": displayed_gold,
            "canonical_gold": list(gold),
            "permutation_hash": _sha(list(order)),
        })
    expected = 30 if qtype == "multi_select" else 24
    if len(result) != expected or len({item["permutation_hash"] for item in result}) != expected:
        raise ValueError(f"permutation_count_or_duplicate:{qtype}")
    return result
