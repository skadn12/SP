"""Closed-loop difficulty controller for the graph_conditioned easy_v2 challenger.

Design (per user decisions):
  - SIGNAL SOURCE: the controller obtains a CLEAN per-cell soft signal itself, using the
    already-validated diagnostic eval harness with the FlashInfer sampler disabled
    (VLLM_USE_FLASHINFER_SAMPLER=0). It does NOT trust a gate's embedded solver_eval, which
    has repeatedly been corrupted to gibberish by the Blackwell sampler bug.
  - ADJUSTMENT OBJECT: cell SAMPLING WEIGHTS only (not within-cell D tiers). The controller
    emits a `difficulty_plan` (question_type_shares + cell_weights) consumed by the existing
    DeterministicQuotaScheduler. No new continuous difficulty dimension.
  - BAD CELLS: cells whose 0-soft is driven by bad_question (e.g. state_reconstruction with
    non-reproducible synthetic gold) are flagged INVALID, not "too hard". The controller does
    not try to make them easier; it down-weights them. It does NOT rewrite gold.

Difficulty cell = "question_type|attack_skill".

This module is pure orchestration + JSON state. It shells the diagnostic harness for the
clean eval (subprocess, sampler env forced) so it never re-imports vLLM in-process.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DIAG_HARNESS = ROOT / "scripts/current_method/diagnostics_context_ablation.py"
BALANCED_MINING_STAGE = "frontier_mining_balanced_1"
MANUAL_POLICY_CORRECTION = "manual_policy_correction_to_balanced_mining"

# Frontier band (kept identical to frontier_selection / policy seed; controller only reads it).
BAND_LOW = 0.2
BAND_HIGH = 0.6
MASTERED = 0.6           # soft strictly above -> mastered (down-weight, room to add difficulty)
TOO_HARD = 0.2           # soft strictly below -> too hard (down-weight unless it is a band entry)
# A cell is BAD (invalid signal) rather than too-hard when bad_question dominates its samples.
BAD_QUESTION_FRACTION = 0.5
MIN_CELL_N = 8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cell_of(sample: dict[str, Any]) -> str:
    cell = sample.get("difficulty_cell")
    if isinstance(cell, str) and cell:
        return cell
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    qtype = str(qa.get("question_type") or sample.get("question_type") or "unknown")
    skill = str(sample.get("attack_skill") or "unknown")
    return f"{qtype}|{skill}"


def run_clean_per_cell_eval(
    samples_path: str | Path,
    out_dir: str | Path,
    *,
    venv_python: str | Path,
    base_model: str | None = None,
    solver_model: str | None = None,
    solver_tokenizer: str | None = None,
    k_eval: int = 5,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Clean re-eval of a samples.jsonl via the diagnostic harness with FlashInfer OFF.

    Returns the parsed verify eval rows joined with each sample's difficulty_cell.
    """
    samples_path = Path(samples_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import os

    env = dict(os.environ)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # mandatory on this Blackwell GPU
    env["PYTHONPATH"] = str(ROOT / "scripts") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        str(venv_python),
        str(DIAG_HARNESS),
        "--task", "verify",
        "--samples", str(samples_path),
        "--k-eval", str(k_eval),
        "--out-dir", str(out_dir),
    ]
    if base_model:
        cmd.extend(["--base-model", str(base_model)])
    if solver_model:
        cmd.extend(["--solver-model", str(solver_model)])
    if solver_tokenizer:
        cmd.extend(["--solver-tokenizer", str(solver_tokenizer)])
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    eval_path = out_dir / "generation" / "verify_k5_eval.jsonl"
    if proc.returncode != 0 and not eval_path.exists():
        raise RuntimeError(f"clean per-cell eval failed (rc={proc.returncode}): {proc.stderr[-2000:]}")
    eval_rows = _read_jsonl(eval_path)
    samples = {s.get("sample_id"): s for s in _read_jsonl(samples_path)}
    joined: list[dict[str, Any]] = []
    for row in eval_rows:
        sid = row.get("sample_id")
        src = samples.get(sid, {})
        se = row.get("solver_eval", {})
        bucket = se.get("bucket")
        exact_count = se.get("exact_correct_count")
        try:
            exact_count_int = int(exact_count)
        except (TypeError, ValueError):
            exact_count_int = 0
        predictions = se.get("predictions") if isinstance(se.get("predictions"), list) else []
        first_exact = bool(predictions and isinstance(predictions[0], dict) and predictions[0].get("exact_correct") is True)
        k = se.get("k")
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = k_eval
        joined.append(
            {
                "sample_id": sid,
                "difficulty_cell": _cell_of(src),
                "soft_accuracy": float(se.get("soft_accuracy") or 0.0),
                "bucket": bucket,
                "exact_correct_count": exact_count_int,
                "first_exact": first_exact,
                "k": k_int,
                "frontier_category": src.get("frontier_label", {}).get("category"),
            }
        )
    return {"rows": joined, "stdout_tail": proc.stdout[-1500:], "eval_path": str(eval_path)}


def _qtype_of_cell(cell: str) -> str:
    return str(cell).split("|", 1)[0] if "|" in str(cell) else str(cell or "unknown")


def _normalize_shares(shares: dict[str, float]) -> dict[str, float]:
    cleaned = {str(k): max(0.0, float(v)) for k, v in shares.items() if max(0.0, float(v)) > 0}
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in cleaned.items()}


def _redistribute_capped_share(
    shares: dict[str, float],
    *,
    cap_by_qtype: dict[str, float],
    sink_qtype: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjusted = {q: max(0.0, float(v)) for q, v in shares.items() if max(0.0, float(v)) > 0}
    capped: dict[str, dict[str, float]] = {}
    overflow = 0.0
    for q, cap in cap_by_qtype.items():
        if q not in adjusted:
            continue
        before = adjusted[q]
        after = min(before, max(0.0, float(cap)))
        if after < before:
            adjusted[q] = after
            overflow += before - after
            capped[q] = {"before": round(before, 6), "after": round(after, 6), "cap": round(float(cap), 6)}
    if overflow > 0:
        adjusted[sink_qtype] = adjusted.get(sink_qtype, 0.0) + overflow
    return _normalize_shares(adjusted), {"capped": capped, "overflow_to": sink_qtype if overflow > 0 else None}


def _apply_exploration_floor(
    shares: dict[str, float],
    *,
    floor_by_qtype: dict[str, float],
    source_qtype: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjusted = {q: max(0.0, float(v)) for q, v in shares.items() if max(0.0, float(v)) > 0}
    raised: dict[str, dict[str, float]] = {}
    needed = 0.0
    for q, floor in floor_by_qtype.items():
        if q not in adjusted:
            continue
        before = adjusted[q]
        after = max(before, max(0.0, float(floor)))
        if after > before:
            adjusted[q] = after
            needed += after - before
            raised[q] = {"before": round(before, 6), "after": round(after, 6), "floor": round(float(floor), 6)}
    if needed > 0 and source_qtype in adjusted:
        adjusted[source_qtype] = max(0.0, adjusted[source_qtype] - needed)
    return _normalize_shares(adjusted), {"raised": raised, "source": source_qtype if needed > 0 else None}


def aggregate_qtype_metrics(joined_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate clean verify rows into the qtype metrics used by the curriculum gate."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        grouped[_qtype_of_cell(str(row.get("difficulty_cell") or "unknown"))].append(row)
    out: dict[str, dict[str, Any]] = {}
    frontier_categories = {"clean_near_miss", "hard_correct"}
    for qtype, rows in sorted(grouped.items()):
        n = len(rows)
        if not n:
            continue
        soft_sum = sum(float(r.get("soft_accuracy") or 0.0) for r in rows)
        first_exact = sum(1 for r in rows if r.get("first_exact") is True)
        buckets = Counter(str(r.get("bucket") or "unknown") for r in rows)
        frontier_count = sum(1 for r in rows if str(r.get("frontier_category") or "") in frontier_categories)
        five_count = int(buckets.get("5/5", 0))
        out[qtype] = {
            "n": n,
            "acc": round(first_exact / n, 6),
            "exact_accuracy_k1": round(first_exact / n, 6),
            "soft_accuracy_k5": round(soft_sum / n, 6),
            "frontier_count": frontier_count,
            "frontier_yield": round(frontier_count / n, 6),
            "bucket_counts": {k: int(v) for k, v in sorted(buckets.items())},
            "five_of_five_share": round(five_count / n, 6),
        }
    return out


def aggregate_cells(joined_rows: list[dict[str, Any]], frontier_categories: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Aggregate clean per-cell soft + bad_question fraction.

    frontier_categories: optional {sample_id: category} from the gate's frontier labels
    (used to classify BAD cells). If absent, falls back to the category carried on each row.
    """
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in joined_rows:
        by_cell[r["difficulty_cell"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for cell, rows in sorted(by_cell.items()):
        n = len(rows)
        soft = sum(r["soft_accuracy"] for r in rows) / n if n else 0.0
        buckets = Counter(str(r.get("bucket") or "unknown") for r in rows)
        cats = Counter()
        for r in rows:
            cat = (frontier_categories or {}).get(r["sample_id"]) or r.get("frontier_category") or "unknown"
            cats[str(cat)] += 1
        bad_frac = cats.get("bad_question", 0) / n if n else 0.0
        exact_frontier_yield = cats.get("hard_correct", 0) / n if n else 0.0
        near_miss_yield = cats.get("clean_near_miss", 0) / n if n else 0.0
        zero_share = buckets.get("0/5", 0) / n if n else 0.0
        five_share = buckets.get("5/5", 0) / n if n else 0.0
        utility = (
            1.0 * exact_frontier_yield
            + 0.5 * near_miss_yield
            - 0.4 * max(0.0, zero_share - 0.40)
            - 0.2 * max(0.0, five_share - 0.65)
        )
        out[cell] = {
            "n": n,
            "mean_soft_accuracy": round(soft, 6),
            "by_frontier_category": dict(cats),
            "bucket_counts": {k: int(v) for k, v in sorted(buckets.items())},
            "exact_frontier_yield": round(exact_frontier_yield, 6),
            "near_miss_yield": round(near_miss_yield, 6),
            "zero_share": round(zero_share, 6),
            "five_share": round(five_share, 6),
            "frontier_utility": round(utility, 6),
            "bad_question_fraction": round(bad_frac, 6),
        }
    return out


def classify_cell(stats: dict[str, Any]) -> str:
    """Return one of: band | mastered | too_hard | bad | unknown.

    Ordering matters: a cell with high clean soft is MASTERED, never BAD. The bad_question
    fraction is only trusted to mark a cell invalid when soft is also low — because a high
    clean soft empirically proves the cell IS solvable (so any bad_question label came from a
    corrupted eval or a mislabeled gold that the solver nonetheless answered).
    """
    n = int(stats.get("n") or 0)
    if n < MIN_CELL_N:
        return "unknown"
    soft = float(stats.get("mean_soft_accuracy") or 0.0)
    # Mastery takes precedence over any category label: clean soft is the ground truth.
    if soft > MASTERED:
        return "mastered"
    if soft >= BAND_LOW:  # in or near band, not bad regardless of labels
        return "band"
    # soft is below band here. Distinguish genuinely-too-hard from invalid-gold (bad).
    if float(stats.get("bad_question_fraction") or 0.0) >= BAD_QUESTION_FRACTION:
        return "bad"
    return "too_hard"


# Multiplicative weight adjustment per class (applied to a cell's prior weight, then renormalized).
CLASS_WEIGHT_DELTA = {
    "band": 1.5,        # band cells are the target -> up-weight
    "mastered": 0.5,    # solver already solves -> down-weight (room for harder via future D tiers)
    "too_hard": 0.6,    # genuinely too hard -> down-weight but keep some presence
    "bad": 0.1,         # invalid gold signal -> strongly down-weight (do NOT treat as difficulty)
    "unknown": 1.0,
}


def _feasible_share_cap(shares: dict[str, float], hard_cap: float) -> dict[str, float]:
    """Cap per-type shares at `hard_cap` via water-filling, guaranteeing feasibility.

    Two guarantees so the downstream generator can always fill n_samples:
      1. No type exceeds hard_cap (matches the generation-side per-type hard cap).
      2. If too few types remain to cover 100% at hard_cap (active*cap < 1), the cap is
         relaxed to 1/active so the plan is mathematically satisfiable. This prevents the
         deadlock where the controller collapsed onto 2 types but the 40% cap needs >=3.
    """
    active = [q for q, s in shares.items() if s > 0]
    if not active:
        return shares
    effective_cap = max(hard_cap, 1.0 / len(active))  # relax if cap can't cover 100%
    values = {q: max(0.0, shares.get(q, 0.0)) for q in active}
    total = sum(values.values()) or 1.0
    values = {q: v / total for q, v in values.items()}
    for _ in range(100):
        offenders = [q for q, v in values.items() if v > effective_cap + 1e-9]
        if not offenders:
            break
        excess = sum(values[q] - effective_cap for q in offenders)
        for q in offenders:
            values[q] = effective_cap
        receivers = [q for q in active if q not in offenders and values[q] < effective_cap - 1e-9]
        if not receivers:
            break
        rtotal = sum(values[q] for q in receivers) or float(len(receivers))
        for q in receivers:
            values[q] += excess * (values[q] / rtotal if rtotal else 1.0 / len(receivers))
    denom = sum(values.values()) or 1.0
    return {q: round(v / denom, 6) for q, v in values.items()}


def update_state_and_plan(
    *,
    prior_state: dict[str, Any] | None,
    cell_stats: dict[str, dict[str, Any]],
    qtype_metrics: dict[str, dict[str, Any]] | None = None,
    controller_source_solver: dict[str, Any] | None = None,
    supported_question_types: list[str],
    cells_by_question_type: dict[str, list[str]],
    max_question_type_share: float = 0.40,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update persistent controller state and emit the next-round difficulty_plan.

    Returns (new_state, difficulty_plan). difficulty_plan = {question_type_shares, cell_weights}.
    max_question_type_share MUST match the generator's per-type hard cap so the plan never
    asks for a distribution the hard cap will reject (which previously deadlocked the run).
    """
    prior = prior_state or {}
    prior_cell_weights: dict[str, float] = dict(prior.get("cell_weights") or {})
    new_cell_weights: dict[str, float] = {}
    classifications: dict[str, str] = {}
    mastery_counts: dict[str, int] = dict(prior.get("mastery_counts") or {})
    prior_plan = prior.get("difficulty_plan") if isinstance(prior.get("difficulty_plan"), dict) else {}
    prior_curriculum = prior_plan.get("qtype_curriculum") if isinstance(prior_plan.get("qtype_curriculum"), dict) else {}
    prior_stage = str(prior_curriculum.get("stage") or "")

    # all candidate cells = those seen now plus prior weights plus the currently supported action
    # cells. Newly-unlocked qtypes have no prior samples yet, but the generator still needs their
    # cells represented so the bandit/quota scheduler can allocate them.
    candidate_cells = {cell for cells in cells_by_question_type.values() for cell in cells}
    all_cells = set(cell_stats) | set(prior_cell_weights) | candidate_cells
    for cell in sorted(all_cells):
        stats = cell_stats.get(cell, {})
        klass = classify_cell(stats) if stats else "unknown"
        classifications[cell] = klass
        base_w = float(prior_cell_weights.get(cell, 1.0))
        utility = float(stats.get("frontier_utility") or 0.0)
        sample_n = int(stats.get("n") or 0)
        utility_weight = 1.0 if sample_n < MIN_CELL_N else max(0.25, min(2.0, 1.0 + utility))
        new_cell_weights[cell] = max(0.01, base_w * CLASS_WEIGHT_DELTA.get(klass, 1.0) * utility_weight)
        # track consecutive mastery for future D-tier escalation (recorded, not yet acted on)
        if klass == "mastered":
            mastery_counts[cell] = int(mastery_counts.get(cell, 0)) + 1
        elif klass in ("too_hard", "bad"):
            mastery_counts[cell] = 0

    # Derive question_type_shares from the summed (non-bad) cell weights per type.
    qtype_weight: dict[str, float] = {q: 0.0 for q in supported_question_types}
    for q in supported_question_types:
        for cell in cells_by_question_type.get(q, []):
            if classifications.get(cell) == "bad":
                continue  # do not let bad cells prop up a question type's share
            qtype_weight[q] += new_cell_weights.get(cell, 1.0)
    total_qw = sum(qtype_weight.values())
    if total_qw <= 0:
        # all cells bad/zero -> fall back to uniform over supported types so the run can proceed
        question_type_shares = {q: 1.0 / len(supported_question_types) for q in supported_question_types}
    else:
        question_type_shares = {q: round(w / total_qw, 6) for q, w in qtype_weight.items() if w > 0}

    # Anti-collapse + feasibility guardrail: cap each type at the generation hard-cap, and
    # relax to 1/active if too few types remain to fill the run (prevents the deadlock where
    # the plan collapsed to 2 types but the 40% hard cap requires >=3 to reach 100%).
    question_type_shares = _feasible_share_cap(question_type_shares, max_question_type_share)

    # --- qtype curriculum gate (overrides weight-derived shares) ---------------------------
    # Difficulty-ordered unlocking: don't sample harder qtypes until the easier prerequisite is
    # in-band for the CURRENT solver. This is the core fix: weight tuning alone kept sampling
    # too-hard qtypes; the gate hard-locks them out instead.
    qtype_soft = per_qtype_soft(cell_stats)
    qtype_metrics = qtype_metrics or {
        q: {
            "n": int(info.get("n") or 0),
            "acc": float(info.get("mean_soft_accuracy") or 0.0),
            "exact_accuracy_k1": float(info.get("mean_soft_accuracy") or 0.0),
            "soft_accuracy_k5": float(info.get("mean_soft_accuracy") or 0.0),
            "frontier_yield": 0.0,
            "bucket_counts": {},
            "five_of_five_share": 0.0,
        }
        for q, info in qtype_soft.items()
    }
    consec_low, frozen = _update_qtype_freeze(prior.get("qtype_freeze") or {}, qtype_soft, supported_question_types)
    qtype_curriculum = qtype_curriculum_gate(qtype_metrics, supported_question_types, frozen, prior_state=prior)
    # The gate's shares are intentional curriculum decisions, NOT weight noise — do NOT re-apply
    # the anti-collapse feasibility cap (which would distort e.g. 0.8/0.2 into 0.5/0.5). The
    # generator already makes its per-type hard cap feasibility-aware (max(global_cap, planned
    # quota)), so an intentional 0.8 share is honored downstream without deadlock.
    question_type_shares = {
        q: round(v, 6)
        for q, v in qtype_curriculum.get("question_type_shares", {}).items()
        if v > 0
    }
    if prior_stage == BALANCED_MINING_STAGE:
        balanced_shares = {"single_choice": 0.40, "multi_select": 0.35, "ordering": 0.25}
        balanced_allowed = ["single_choice", "multi_select", "ordering"]
        question_type_shares = {q: v for q, v in balanced_shares.items() if q in supported_question_types and v > 0}
        qtype_curriculum = {
            **qtype_curriculum,
            "stage": BALANCED_MINING_STAGE,
            "manual_policy_correction": MANUAL_POLICY_CORRECTION,
            "allowed_question_types": balanced_allowed,
            "question_type_shares": balanced_shares,
            "share_policy": {
                "ordinary_single_choice": 0.05,
                "hard_single_choice": 0.35,
                "multi_select": 0.35,
                "ordering": 0.25,
                "qtype_delta_cap": 0.05,
                "single_choice_floor": 0.05,
                "hard_single_choice_min": 0.25,
                "hard_single_choice_default": 0.35,
                "hard_single_choice_max": 0.40,
                "multi_select_min": 0.25,
                "multi_select_default": 0.35,
                "multi_select_max": 0.40,
                "ordering_min": 0.15,
                "ordering_default": 0.25,
                "ordering_max": 0.30,
            },
        }

    # skill_shares: when the gate forces the easiest setting (single_choice-only / base<LOW),
    # bias skills toward the easiest (delayed_activation). Otherwise leave skill selection to
    # the generator's per-cell round-robin (emit empty -> generator default).
    base_qtype = next((q for q in QTYPE_DIFFICULTY_ORDER if q in question_type_shares), None)
    base_soft = float((qtype_soft.get(base_qtype) or {}).get("mean_soft_accuracy", 0.0) or 0.0) if base_qtype else 0.0
    skill_shares: dict[str, float] = {}
    if len(question_type_shares) == 1 or base_soft < GATE_LOW:
        skill_shares = {k: v for k, v in EASIEST_SKILL_SHARES.items() if v > 0}

    if prior_stage == BALANCED_MINING_STAGE:
        skill_shares = {
            "conditional_plan": 0.45,
            "permission_visibility_scope": 0.08,
            "delayed_activation": 0.22,
            "timeline_fork_merge": 0.25,
        }

    # Fold skill_shares into cell_weights so the generator's cell-biased scheduler honors them
    # without a separate code path. For each allowed qtype, set cell weight = skill_share for
    # that cell's skill (cells whose skill has 0 share get a small floor, not zero, to stay feasible).
    if skill_shares:
        folded = dict(new_cell_weights)
        for q in question_type_shares:
            for cell in cells_by_question_type.get(q, []):
                skill = cell.split("|", 1)[1] if "|" in cell else ""
                cell_n = int((cell_stats.get(cell) or {}).get("n") or 0)
                skill_factor = 1.0 if cell_n < MIN_CELL_N else max(0.05, float(skill_shares.get(skill, 0.0)) or 0.05)
                folded[cell] = max(0.01, float(folded.get(cell, 1.0)) * skill_factor)
        new_cell_weights = folded
    elif prior_stage == BALANCED_MINING_STAGE:
        folded = dict(new_cell_weights)
        for q in question_type_shares:
            for cell in cells_by_question_type.get(q, []):
                skill = cell.split("|", 1)[1] if "|" in cell else ""
                cell_n = int((cell_stats.get(cell) or {}).get("n") or 0)
                skill_factor = 1.0 if cell_n < MIN_CELL_N else float({
                    "conditional_plan": 0.45,
                    "permission_visibility_scope": 0.08,
                    "delayed_activation": 0.22,
                    "timeline_fork_merge": 0.25,
                }.get(skill, 0.05))
                folded[cell] = max(0.01, float(folded.get(cell, 1.0)) * skill_factor)
        new_cell_weights = folded

    single_choice_subtype_shares = dict(prior_plan.get("single_choice_subtype_shares") or {})
    single_choice_subtype_hardness_mix = dict(prior_plan.get("single_choice_subtype_hardness_mix") or {})
    if prior_stage == BALANCED_MINING_STAGE or not single_choice_subtype_shares:
        single_choice_subtype_shares = {
            "ordinary_single_choice": 0.125,
            "hard_single_choice": 0.875,
        }
    if prior_stage == BALANCED_MINING_STAGE or not single_choice_subtype_hardness_mix:
        single_choice_subtype_hardness_mix = {
            "hard_single_choice": {"h0": 0.50, "h1": 0.40, "h2": 0.10},
            "ordinary_single_choice": {"h0": 0.90, "h1": 0.10, "h2": 0.00},
        }

    difficulty_plan = {
        "question_type_shares": question_type_shares,
        "allowed_question_types": list(qtype_curriculum.get("allowed_question_types") or list(question_type_shares)),
        "qtype_curriculum": {
            **qtype_curriculum,
            "source_solver_name": (controller_source_solver or {}).get("solver_name"),
            "source_solver_model": (controller_source_solver or {}).get("solver_model"),
            "source_solver_tokenizer": (controller_source_solver or {}).get("solver_tokenizer"),
            "created_by": "manual_acc_threshold_curriculum_gate",
            "manual_policy_correction": qtype_curriculum.get("manual_policy_correction"),
        },
        "qtype_shares_created_by": "manual_acc_threshold_curriculum_gate",
        "skill_shares": skill_shares,
        "single_choice_subtype_shares": single_choice_subtype_shares,
        "single_choice_subtype_hardness_mix": single_choice_subtype_hardness_mix,
        "cell_weights": {c: round(w, 6) for c, w in sorted(new_cell_weights.items())},
        "cell_classifications": classifications,
        "qtype_soft": qtype_soft,
        "qtype_metrics": qtype_metrics,
        "frozen_qtypes": sorted(frozen),
        "gate_base_qtype": base_qtype,
        "gate_base_soft": round(base_soft, 6),
        "controller_source_solver": controller_source_solver or {},
    }
    prior_rollout_counts = {
        str(q): int(v or 0)
        for q, v in (prior.get("qtype_rollout_counts") if isinstance(prior.get("qtype_rollout_counts"), dict) else {}).items()
    }
    rollout_counts = dict(prior_rollout_counts)
    for q in qtype_curriculum.get("allowed_question_types") or []:
        if q != base_qtype:
            rollout_counts[str(q)] = rollout_counts.get(str(q), 0) + 1
    for q in QTYPE_DIFFICULTY_ORDER:
        if q not in (qtype_curriculum.get("allowed_question_types") or []) and q in rollout_counts:
            rollout_counts[q] = 0

    new_state = {
        "round_index": int(prior.get("round_index", 0)) + 1,
        "controller_source_solver": controller_source_solver or {},
        "cell_weights": difficulty_plan["cell_weights"],
        "cell_classifications": classifications,
        "mastery_counts": mastery_counts,
        "qtype_freeze": {"consec_low": consec_low, "frozen": sorted(frozen)},
        "qtype_rollout_counts": rollout_counts,
        "qtype_soft": qtype_soft,
        "qtype_metrics": qtype_metrics,
        "cell_stats": cell_stats,
        "difficulty_plan": difficulty_plan,
    }
    return new_state, difficulty_plan


EASY_V2_CELLS_BY_QUESTION_TYPE = {
    "single_choice": ["single_choice|conditional_plan", "single_choice|delayed_activation"],
    "multi_select": ["multi_select|conditional_plan", "multi_select|permission_visibility_scope"],
    "ordering": ["ordering|timeline_fork_merge", "ordering|conditional_plan"],
    "state_reconstruction": ["state_reconstruction|conditional_plan", "state_reconstruction|delayed_activation"],
}
EASY_V2_SUPPORTED_QUESTION_TYPES = ["single_choice", "multi_select", "ordering", "state_reconstruction"]

# ----------------------------------------------------------------------------------------
# qtype curriculum gate
# ----------------------------------------------------------------------------------------
# Difficulty order (easy -> hard). r4 hardening can unlock multi_select and ordering when the
# current fixed solver has mastered the easier base qtype. evidence_selection remains metadata-like
# and is not part of this curriculum.
QTYPE_DIFFICULTY_ORDER = [
    "single_choice",
    "multi_select",
    "ordering",
    "state_reconstruction",
]
GATE_LOW = 0.2      # below this on the base qtype => stay locked to the base qtype only
GATE_BAND_HI = 0.6  # in [LOW, HI] => base qtype dominant, only the next tier cracked open
FREEZE_AFTER_CONSEC_LOW = 2  # qtype frozen after this many consecutive batches < GATE_LOW
FIRST_UNLOCK_NEXT_MAX_SHARE = 0.25
FIRST_UNLOCK_NEXT_NEXT_MAX_SHARE = 0.10
UNLOCKED_QTYPE_EXPLORATION_FLOOR = 0.05
EXPLORATION_FLOOR_GATES = 1
# Permanently-frozen qtypes. evidence_selection is support metadata (evidence_turn_ids), not a
# standalone solver task for r4 hardening. multi_select/ordering are deliberately not hard-frozen:
# the qtype curriculum gate owns when they unlock.
HARD_FROZEN_QTYPES = {"evidence_selection"}
# Skill difficulty preference when the gate forces the easiest setting (single_choice only).
EASIEST_SKILL_SHARES = {
    "delayed_activation": 0.6,
    "permission_visibility_scope": 0.2,
    "conditional_plan": 0.2,
    "timeline_fork_merge": 0.0,
}


def per_qtype_soft(cell_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-cell stats up to per-question_type soft accuracy + sample count."""
    agg: dict[str, dict[str, float]] = {}
    for cell, s in cell_stats.items():
        qtype = cell.split("|", 1)[0]
        n = int(s.get("n") or 0)
        if n <= 0:
            continue
        e = agg.setdefault(qtype, {"soft_sum": 0.0, "n": 0})
        e["soft_sum"] += float(s.get("mean_soft_accuracy") or 0.0) * n
        e["n"] += n
    return {
        q: {"mean_soft_accuracy": round(e["soft_sum"] / e["n"], 6) if e["n"] else 0.0, "n": e["n"]}
        for q, e in agg.items()
    }


def _update_qtype_freeze(
    prior_freeze: dict[str, Any],
    qtype_soft: dict[str, dict[str, Any]],
    supported: list[str],
) -> tuple[dict[str, int], set[str]]:
    """Track consecutive low batches per qtype; return (consec_low_counts, frozen_set).

    A qtype with soft < GATE_LOW for FREEZE_AFTER_CONSEC_LOW consecutive batches is frozen.
    A qtype that reaches >= GATE_LOW resets its counter and unfreezes. HARD_FROZEN_QTYPES are
    always frozen. A qtype not evaluated this batch keeps its prior counter unchanged.
    """
    consec = {q: int((prior_freeze.get("consec_low") or {}).get(q, 0)) for q in supported}
    for q in supported:
        info = qtype_soft.get(q)
        if info is None:
            continue  # not sampled this batch -> keep prior counter
        if float(info.get("mean_soft_accuracy") or 0.0) < GATE_LOW:
            consec[q] += 1
        else:
            consec[q] = 0
    frozen = {q for q in supported if consec.get(q, 0) >= FREEZE_AFTER_CONSEC_LOW}
    frozen |= (HARD_FROZEN_QTYPES & set(supported))
    return consec, frozen


def qtype_curriculum_gate(
    qtype_metrics: dict[str, dict[str, Any]],
    supported: list[str],
    frozen: set[str],
    prior_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the qtype curriculum decision from clean fixed-solver metrics.

    The threshold table is intentionally manual curriculum policy, but the per-round shares are
    computed from the previous fixed solver's measured qtype metrics. The bandit may tune actions
    inside these allowed qtypes later, but it must not erase this unlock decision.
    """
    order = [q for q in QTYPE_DIFFICULTY_ORDER if q in supported]
    unlocked_order = [q for q in order if q not in frozen]
    unlock_decisions: dict[str, dict[str, Any]] = {}
    if not unlocked_order:
        # everything frozen: fall back to the single easiest supported qtype to keep run alive
        base = order[0] if order else (supported[0] if supported else "single_choice")
        shares = {base: 1.0}
        return {
            "stage": "all_frozen_fallback",
            "base_qtype": base,
            "base_metrics": {},
            "qtype_metrics": qtype_metrics,
            "unlock_decisions": {
                base: {"unlocked": True, "share": 1.0, "reason": "fallback_when_all_supported_qtypes_frozen"}
            },
            "allowed_question_types": [base],
            "question_type_shares": shares,
        }

    # The current base qtype is the easiest non-frozen qtype that was actually measured this round.
    # For r4 probe197 this is single_choice, yielding mastered_advance from 195/200 5-of-5.
    base = next((q for q in unlocked_order if int((qtype_metrics.get(q) or {}).get("n") or 0) > 0), unlocked_order[0])
    base_index = unlocked_order.index(base)
    next_qtype = unlocked_order[base_index + 1] if base_index + 1 < len(unlocked_order) else None
    next_next_qtype = unlocked_order[base_index + 2] if base_index + 2 < len(unlocked_order) else None
    base_metrics = qtype_metrics.get(base) or {}
    acc = float(base_metrics.get("acc", base_metrics.get("exact_accuracy_k1", base_metrics.get("soft_accuracy_k5", 0.0))) or 0.0)
    frontier_yield = float(base_metrics.get("frontier_yield") or 0.0)
    five_of_five_share = float(base_metrics.get("five_of_five_share") or 0.0)

    if acc < 0.20:
        stage = "too_hard"
        shares = {base: 1.0}
    elif acc < 0.60:
        stage = "frontier"
        shares = {base: 0.85}
        if next_qtype:
            shares[next_qtype] = 0.15
    elif acc < 0.85:
        stage = "learnable_to_mastering"
        shares = {base: 0.60}
        if next_qtype:
            shares[next_qtype] = 0.40
    elif frontier_yield >= 0.05:
        stage = "mastered_but_frontier_exists"
        shares = {base: 0.50}
        if next_qtype:
            shares[next_qtype] = 0.40
        if next_next_qtype:
            shares[next_next_qtype] = 0.10
    elif five_of_five_share >= 0.70:
        stage = "mastered_advance"
        shares = {base: 0.20}
        if next_qtype:
            shares[next_qtype] = 0.50
        if next_next_qtype:
            shares[next_next_qtype] = 0.30
    else:
        stage = "mastered_uncertain"
        shares = {base: 0.50}
        if next_qtype:
            shares[next_qtype] = 0.40
        if next_next_qtype:
            shares[next_next_qtype] = 0.10

    raw_threshold_shares = _normalize_shares({q: v for q, v in shares.items() if q in unlocked_order and v > 0})
    shares = dict(raw_threshold_shares)
    if not shares:
        shares = {base: 1.0}

    prior = prior_state or {}
    prior_plan = prior.get("difficulty_plan") if isinstance(prior.get("difficulty_plan"), dict) else {}
    prior_curriculum = prior_plan.get("qtype_curriculum") if isinstance(prior_plan.get("qtype_curriculum"), dict) else {}
    rollout_counts = {
        str(q): int(v or 0)
        for q, v in (prior.get("qtype_rollout_counts") if isinstance(prior.get("qtype_rollout_counts"), dict) else {}).items()
    }
    newly_unlocked = [q for q in shares if q != base and rollout_counts.get(q, 0) <= 0]
    first_unlock_caps: dict[str, float] = {}
    if next_qtype in newly_unlocked:
        first_unlock_caps[next_qtype] = FIRST_UNLOCK_NEXT_MAX_SHARE
    if next_next_qtype in newly_unlocked:
        first_unlock_caps[next_next_qtype] = FIRST_UNLOCK_NEXT_NEXT_MAX_SHARE
    cap_report: dict[str, Any] = {"enabled": bool(first_unlock_caps), "newly_unlocked_qtypes": newly_unlocked}
    if first_unlock_caps:
        shares, cap_report_extra = _redistribute_capped_share(shares, cap_by_qtype=first_unlock_caps, sink_qtype=base)
        cap_report.update(cap_report_extra)

    exploration_qtypes = [
        q
        for q in shares
        if q != base and (q in newly_unlocked or rollout_counts.get(q, 0) < EXPLORATION_FLOOR_GATES)
    ]
    floor_report: dict[str, Any] = {"enabled": bool(exploration_qtypes), "qtypes": exploration_qtypes}
    if exploration_qtypes:
        shares, floor_report_extra = _apply_exploration_floor(
            shares,
            floor_by_qtype={q: UNLOCKED_QTYPE_EXPLORATION_FLOOR for q in exploration_qtypes},
            source_qtype=base,
        )
        floor_report.update(floor_report_extra)

    for q in order:
        share = float(shares.get(q, 0.0))
        if q in frozen:
            reason = "hard_or_dynamic_frozen"
        elif q == base:
            reason = f"base_qtype_stage:{stage}"
        elif share > 0:
            reason = f"unlocked_by_stage:{stage}"
        else:
            reason = f"not_reached_by_stage:{stage}"
        unlock_decisions[q] = {
            "unlocked": share > 0,
            "share": round(share, 6),
            "reason": reason,
            "metrics": qtype_metrics.get(q, {}),
            "first_unlocked_this_gate": q in newly_unlocked,
            "rollout_count_before": rollout_counts.get(q, 0),
        }
    return {
        "stage": stage,
        "base_qtype": base,
        "base_metrics": base_metrics,
        "qtype_metrics": qtype_metrics,
        "unlock_decisions": unlock_decisions,
        "allowed_question_types": [q for q in order if shares.get(q, 0.0) > 0],
        "question_type_shares": shares,
        "raw_threshold_question_type_shares": raw_threshold_shares,
        "first_unlock_cap": {
            **cap_report,
            "next_max_share": FIRST_UNLOCK_NEXT_MAX_SHARE,
            "next_next_max_share": FIRST_UNLOCK_NEXT_NEXT_MAX_SHARE,
        },
        "exploration_floor": {
            **floor_report,
            "min_share": UNLOCKED_QTYPE_EXPLORATION_FLOOR,
            "gates": EXPLORATION_FLOOR_GATES,
        },
        "thresholds": {
            "too_hard_acc_lt": 0.20,
            "frontier_acc_range": [0.20, 0.60],
            "learnable_to_mastering_acc_range": [0.60, 0.85],
            "mastered_acc_gte": 0.85,
            "mastered_frontier_yield_gte": 0.05,
            "mastered_advance_frontier_yield_lt": 0.05,
            "mastered_advance_five_of_five_share_gte": 0.70,
        },
    }




def run_controller_update(
    *,
    samples_path: str | Path,
    state_path: str | Path,
    work_dir: str | Path,
    venv_python: str | Path,
    frontier_categories: dict[str, str] | None = None,
    k_eval: int = 5,
    base_model: str | None = None,
    solver_name: str | None = None,
    solver_model: str | None = None,
    solver_tokenizer: str | None = None,
    source_run: str | None = None,
    source_eval_file: str | None = None,
) -> dict[str, Any]:
    """End-to-end: clean per-cell eval -> aggregate -> update state -> persist -> return plan.

    state_path persists across rounds (the closed loop's memory). The returned difficulty_plan
    is what the NEXT generation round should inject into policy["difficulty_plan"].
    """
    state_path = Path(state_path)
    prior_state = None
    if state_path.exists():
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior_state = None
    clean = run_clean_per_cell_eval(
        samples_path,
        work_dir,
        venv_python=venv_python,
        base_model=base_model,
        solver_model=solver_model,
        solver_tokenizer=solver_tokenizer,
        k_eval=k_eval,
    )
    cell_stats = aggregate_cells(clean["rows"], frontier_categories=frontier_categories)
    qtype_metrics = aggregate_qtype_metrics(clean["rows"])
    controller_source_solver = {
        "solver_name": solver_name,
        "solver_model": solver_model,
        "solver_tokenizer": solver_tokenizer or base_model,
        "source_run": source_run,
        "source_eval_file": source_eval_file or clean.get("eval_path"),
    }
    # Match the controller's per-type cap to the generator's hard cap so the emitted plan is
    # always feasible for the next round (no question_type_cap deadlock).
    try:
        try:
            from current_method import native_api_graph_dialogue_expansion as _gen
        except ModuleNotFoundError:
            from scripts.current_method import native_api_graph_dialogue_expansion as _gen
        hard_cap = float(getattr(_gen, "EASY_V2_MAX_QUESTION_TYPE_SHARE", 0.40))
    except Exception:
        hard_cap = 0.40
    new_state, difficulty_plan = update_state_and_plan(
        prior_state=prior_state,
        cell_stats=cell_stats,
        qtype_metrics=qtype_metrics,
        controller_source_solver=controller_source_solver,
        supported_question_types=EASY_V2_SUPPORTED_QUESTION_TYPES,
        cells_by_question_type=EASY_V2_CELLS_BY_QUESTION_TYPE,
        max_question_type_share=hard_cap,
    )

    # ---- Action-Level Bandit Controller (paper: Gated Proposer / Learnability Reward) ----
    # Run the bandit over the SAME clean-evaluated samples to produce the next-round action
    # policy (question_type | attack_skill | distractor_hardness) and project it onto the
    # generator's difficulty_plan knobs (shares + cell_weights + cell_hardness_mix). This is the
    # automatic proposer: no per-batch hand-set weights.
    bandit_report = None
    try:
        try:
            from current_method import proposer_bandit as pb
        except ModuleNotFoundError:
            from scripts.current_method import proposer_bandit as pb
        samples = _read_jsonl(Path(samples_path))
        # join the clean verify eval (sample_id -> solver_eval) back onto the samples so the
        # bandit's validity gate + bucket reward use the CLEAN (FlashInfer-off) eval, not the
        # gate's possibly-corrupted embedded eval.
        eval_by_id = {}
        for row in _read_jsonl(Path(clean["eval_path"])):
            if isinstance(row, dict) and row.get("sample_id"):
                eval_by_id[row["sample_id"]] = row.get("solver_eval")
        for s in samples:
            ce = eval_by_id.get(s.get("sample_id"))
            if isinstance(ce, dict):
                s["solver_eval"] = ce
        prior_action_policy = (prior_state or {}).get("action_policy") if isinstance(prior_state, dict) else None
        curriculum = difficulty_plan.get("qtype_curriculum") if isinstance(difficulty_plan.get("qtype_curriculum"), dict) else {}
        allowed_question_types = tuple(curriculum.get("allowed_question_types") or difficulty_plan.get("allowed_question_types") or EASY_V2_SUPPORTED_QUESTION_TYPES)
        curriculum_shares = (
            curriculum.get("question_type_shares")
            if isinstance(curriculum.get("question_type_shares"), dict)
            else difficulty_plan.get("question_type_shares")
        )
        bandit_report = pb.run_bandit_step(
            samples,
            prior_action_policy,
            allowed_question_types=allowed_question_types,
        )
        # The curriculum gate is authoritative for question-type unlocking. The bandit projection
        # may tune action/cell/hardness weights within allowed qtypes, but must not collapse
        # question_type_shares back to single_choice-only.
        bplan = bandit_report["next_difficulty_plan"]
        merged_shares, merge_report = pb.merge_qtype_shares(
            curriculum_shares if isinstance(curriculum_shares, dict) else {},
            bplan.get("question_type_shares") if isinstance(bplan.get("question_type_shares"), dict) else {},
            allowed_question_types,
            exploration_floor=UNLOCKED_QTYPE_EXPLORATION_FLOOR,
        )
        merged_cell_weights = dict(difficulty_plan.get("cell_weights") or {})
        for cell, weight in (bplan.get("cell_weights") or {}).items():
            qtype = str(cell).split("|", 1)[0]
            if qtype in allowed_question_types:
                merged_cell_weights[cell] = weight
        difficulty_plan["question_type_shares"] = merged_shares
        difficulty_plan["allowed_question_types"] = list(allowed_question_types)
        difficulty_plan["cell_weights"] = merged_cell_weights
        difficulty_plan["cell_hardness_mix"] = bplan["cell_hardness_mix"]
        difficulty_plan["action_weights"] = bplan["action_weights"]
        difficulty_plan["bandit_report"] = bandit_report
        difficulty_plan["qtype_share_merge"] = merge_report
        difficulty_plan["qtype_share_merge_mode"] = merge_report.get("mode")
        difficulty_plan["curriculum_allowed_question_types"] = list(allowed_question_types)
        difficulty_plan["bandit_allowed_question_types"] = list(bandit_report.get("allowed_question_types") or [])
        difficulty_plan["bandit_overrode_curriculum"] = False
        difficulty_plan["unlocked_qtype_exploration_floor"] = UNLOCKED_QTYPE_EXPLORATION_FLOOR
        difficulty_plan["exploration_floor_gates"] = EXPLORATION_FLOOR_GATES
        new_state["action_policy"] = bandit_report["next_action_policy"]
        new_state["difficulty_plan"] = difficulty_plan
    except Exception as exc:  # never let the bandit crash a completed round
        difficulty_plan["bandit_error"] = str(exc)

    _write_json(state_path, new_state)
    _write_json(Path(work_dir) / "difficulty_plan.json", difficulty_plan)
    if bandit_report is not None:
        _write_json(Path(work_dir) / "bandit_report.json", bandit_report)
    return {
        "difficulty_plan": difficulty_plan,
        "cell_stats": cell_stats,
        "state_path": str(state_path),
        "clean_eval_path": clean["eval_path"],
        "bandit_report": bandit_report,
    }
