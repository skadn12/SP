"""Action-Level Bandit Controller for the Self-Play challenger (current_method).

Paper framing (use these names in the paper, NOT "manual tuning"):
  * Learnability Reward          - per-sample reward peaking where the fixed solver is
                                   half-right (frontier band), not where difficulty is maximal.
  * Gated Proposer Reward        - the learnability reward multiplied by a hard validity gate;
                                   invalid / leaking / wrong-phase samples contribute 0.
  * Action-Level Bandit Controller - an exponentiated-weights bandit over actions
                                   (question_type | attack_skill | distractor_hardness) whose
                                   reward is the mean gated proposer reward of the samples that
                                   action produced, with shrinkage for small action counts.
  * Distractor Hardness Adaptation - when an action saturates (high 5/5 rate) the controller
                                   shifts probability mass to higher-hardness variants of the
                                   same (question_type, attack_skill) rather than merely
                                   down-weighting it.

The challenger therefore maximizes frontier-band learnability against the CURRENT fixed solver,
discovered automatically from solver bucket feedback — no per-batch hand-set cell weights.

This module is pure / offline-testable: it consumes already-evaluated samples (k=5 solver eval)
plus the prior policy, and emits the next-round action policy + a report block. It does not call
the solver, the API, or re-run eval.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

# --------------------------------------------------------------------------------------------
# Curriculum-gated action space
# --------------------------------------------------------------------------------------------
# The qtype curriculum gate decides which question types are allowed for the next round. This
# module only tunes actions inside that allowed set; it must not globally freeze multi_select or
# ordering, and it must not collapse question_type_shares back to single_choice-only.
PHASE1_ALLOWED_QUESTION_TYPES = ("single_choice",)
PHASE1_ALLOWED_ATTACK_SKILLS = ("conditional_plan", "delayed_activation")
ATTACK_SKILLS_BY_QUESTION_TYPE = {
    "single_choice": ("conditional_plan", "delayed_activation"),
    "multi_select": ("conditional_plan", "permission_visibility_scope"),
    "ordering": ("timeline_fork_merge", "conditional_plan"),
    "state_reconstruction": ("conditional_plan", "delayed_activation"),
}
HARDNESS_LEVELS = ("h0", "h1", "h2")
# Question types that must always receive zero proposer weight (never solver tasks).
FORBIDDEN_QUESTION_TYPES = ("evidence_selection",)

# Learnability reward by k=5 exact-correct count c. Peaks at the frontier band (2/5, 3/5),
# half-credit at the secondary band (1/5, 4/5), zero at the useless extremes (0/5 too hard/buggy,
# 5/5 mastered). This is the single most important design choice: reward learnability, not difficulty.
LEARNABILITY_REWARD_BY_BUCKET = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0}

# Bandit hyperparameters (paper-reportable defaults).
DEFAULT_ALPHA = 5.0          # shrinkage strength toward the global mean reward
DEFAULT_TEMPERATURE = 0.3    # exponentiated-weights temperature
DEFAULT_ETA = 0.3            # soft policy interpolation rate
DEFAULT_EPSILON = 0.05       # minimum exploration (uniform floor over allowed actions)
DEFAULT_MAX_ACTION_SHARE = 0.7  # anti-collapse cap on any single action

# Blend weight between the Cell-Level Learnability reward (continuous action pass-rate tent,
# robust to per-sample k=5 degeneracy) and the per-sample frontier-band reward. 1.0 = pure
# cell-level; 0.0 = pure per-sample. We lean cell-level because single_choice per-sample buckets
# are near-bimodal (0/5 or 5/5) for a confident solver, while action pass rate is continuous.
CELL_REWARD_WEIGHT = 0.7

# Adaptation thresholds.
EASY_RATE_HARDEN_THRESHOLD = 0.5   # 5/5 rate above which we escalate to higher hardness
HARD_RATE_BUGCHECK_THRESHOLD = 0.5  # 0/5 rate above which we down-weight + flag bug_check
HIGH_INVALID_RATE_FREEZE = 0.5      # invalid rate above which we freeze the action


def build_phase1_action_space(allowed_question_types: tuple[str, ...] | None = None) -> list[str]:
    """All allowed action keys: question_type|attack_skill|hardness."""
    actions: list[str] = []
    qtypes = allowed_question_types or PHASE1_ALLOWED_QUESTION_TYPES
    for qt in qtypes:
        if qt in FORBIDDEN_QUESTION_TYPES:
            continue
        skills = ATTACK_SKILLS_BY_QUESTION_TYPE.get(qt, PHASE1_ALLOWED_ATTACK_SKILLS)
        for sk in skills:
            for h in HARDNESS_LEVELS:
                actions.append(f"{qt}|{sk}|{h}")
    return actions


# --------------------------------------------------------------------------------------------
# Sample-level: validity gate + gated learnability (proposer) reward
# --------------------------------------------------------------------------------------------
def _exact_count(sample: dict[str, Any]) -> int | None:
    se = sample.get("solver_eval") if isinstance(sample.get("solver_eval"), dict) else {}
    c = se.get("exact_correct_count")
    if isinstance(c, (int, float)):
        return int(c)
    bucket = str(se.get("bucket") or "")
    if "/" in bucket:
        try:
            return int(bucket.split("/", 1)[0])
        except ValueError:
            return None
    return None


def sample_question_type(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    return str(qa.get("question_type") or sample.get("question_type") or "unknown")


def sample_attack_skill(sample: dict[str, Any]) -> str:
    ap = sample.get("attack_plan") if isinstance(sample.get("attack_plan"), dict) else {}
    return str(ap.get("attack_skill") or sample.get("attack_skill") or "unknown")


def sample_hardness(sample: dict[str, Any]) -> str:
    """distractor_hardness recorded by the generator; default h0 for legacy samples."""
    for container in (sample, sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {},
                      sample.get("qa") if isinstance(sample.get("qa"), dict) else {}):
        if isinstance(container, dict):
            h = container.get("distractor_hardness")
            if h in HARDNESS_LEVELS:
                return str(h)
    return "h0"


def action_key(sample: dict[str, Any]) -> str:
    return f"{sample_question_type(sample)}|{sample_attack_skill(sample)}|{sample_hardness(sample)}"


def validity_gate(sample: dict[str, Any], *, allowed_question_types: tuple[str, ...]) -> tuple[int, str]:
    """Return (gate, reason). gate=1 only if the sample is a valid trainable solver task.

    Gate conditions (all required):
      clean validation, missing_refs==0, no meta leak, no solver-visible meta leak,
      gold graph-derived/validated, dialogue length-expanded vs parent, no dev/heldout leakage,
      and question_type allowed for the current phase.
    """
    qt = sample_question_type(sample)
    if qt not in allowed_question_types:
        return 0, f"qtype_not_allowed:{qt}"
    v = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
    if v.get("graph_dialogue_consistent") is not True:
        return 0, "graph_dialogue_inconsistent"
    if v.get("gold_derivable") is not True:
        return 0, "gold_not_derivable"
    if v.get("option_valid") is not True:
        return 0, "option_invalid"
    if v.get("anti_shortcut_pass") is not True:
        return 0, "anti_shortcut_fail"
    if v.get("warnings"):
        return 0, "validation_warnings"
    # leakage / refs
    nc = sample.get("native_contract") if isinstance(sample.get("native_contract"), dict) else {}
    if int(nc.get("missing_ref_count") or 0) != 0:
        return 0, "missing_refs"
    if int(nc.get("anchor_leakage_count") or 0) != 0:
        return 0, "anchor_leak"
    if sample.get("meta_leak") is True or sample.get("solver_visible_meta_leak") is True:
        return 0, "meta_leak"
    # length expansion vs parent (native_contract length_guard_pass when present)
    if "length_guard_pass" in nc and nc.get("length_guard_pass") is not True:
        return 0, "not_length_expanded"
    # dev/heldout leakage
    seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else {}
    split = sample.get("frontier_from_split") or seed_source.get("split") or sample.get("split")
    if seed_source.get("allowed_train_seed") is not True or split != "train":
        return 0, "dev_or_heldout_or_nontrain_seed"
    return 1, "ok"


def learnability_reward(exact_count: int | None) -> float:
    if exact_count is None:
        return 0.0
    return LEARNABILITY_REWARD_BY_BUCKET.get(int(exact_count), 0.0)


def gated_proposer_reward(sample: dict[str, Any], *, allowed_question_types: tuple[str, ...]) -> tuple[float, int, str]:
    """sample_reward = validity_gate * learnability_reward. Returns (reward, gate, reason)."""
    gate, reason = validity_gate(sample, allowed_question_types=allowed_question_types)
    if gate == 0:
        return 0.0, 0, reason
    return learnability_reward(_exact_count(sample)), 1, "ok"


# --------------------------------------------------------------------------------------------
# Action-level reward aggregation (+ shrinkage)
# --------------------------------------------------------------------------------------------
def aggregate_actions(
    samples: list[dict[str, Any]],
    *,
    allowed_question_types: tuple[str, ...] = PHASE1_ALLOWED_QUESTION_TYPES,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Aggregate per-action stats and the shrinkage-smoothed reward.

    Returns {actions: {action: {...}}, global_mean_reward, proposer_reward_total/mean, n_valid}.
    """
    per_action: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "valid_count": 0, "sum_reward": 0.0, "buckets": Counter(), "invalid_reasons": Counter(), "valid_exact_sum": 0}
    )
    total_reward = 0.0
    valid_rewards: list[float] = []
    for s in samples:
        act = action_key(s)
        reward, gate, reason = gated_proposer_reward(s, allowed_question_types=allowed_question_types)
        c = _exact_count(s)
        a = per_action[act]
        a["count"] += 1
        a["buckets"][f"{c}/5" if c is not None else "na"] += 1
        if gate == 1:
            a["valid_count"] += 1
            a["sum_reward"] += reward
            a["valid_exact_sum"] += int(c or 0)
            total_reward += reward
            valid_rewards.append(reward)
        else:
            a["invalid_reasons"][reason] += 1
    global_mean = (sum(valid_rewards) / len(valid_rewards)) if valid_rewards else 0.0

    actions_out: dict[str, Any] = {}
    for act, a in per_action.items():
        cnt = a["count"]
        b = a["buckets"]
        def bn(k): return int(b.get(k, 0))
        primary = bn("2/5") + bn("3/5")
        secondary = bn("1/5") + bn("4/5")
        easy = bn("5/5")
        hard = bn("0/5")
        invalid = cnt - a["valid_count"]
        # Cell-Level Learnability (paper): per-sample k=5 buckets are near-degenerate for a
        # confident solver (single_choice gives only 0/5 or 5/5), so the per-sample frontier-band
        # reward is noisy. The ACTION pass rate (mean exact / 5) is, by contrast, a CONTINUOUS
        # difficulty signal (e.g. delayed h0=0.96 -> h1=0.84 -> h2=0.33). We reward actions whose
        # pass rate sits near 0.5 via a tent function, and blend it with the per-sample reward.
        cell_pass_rate = (a["valid_exact_sum"] / (5.0 * a["valid_count"])) if a["valid_count"] else 0.0
        cell_learnability = max(0.0, 1.0 - 2.0 * abs(cell_pass_rate - 0.5))  # tent: 1.0 at 0.5, 0 at {0,1}
        per_sample_mean = (a["sum_reward"] / a["valid_count"]) if a["valid_count"] else 0.0
        blended = CELL_REWARD_WEIGHT * cell_learnability + (1.0 - CELL_REWARD_WEIGHT) * per_sample_mean
        # shrinkage toward the global mean: avoids small-count actions swinging the policy.
        smoothed = (blended * a["valid_count"] + alpha * global_mean) / (a["valid_count"] + alpha) if (a["valid_count"] + alpha) > 0 else global_mean
        actions_out[act] = {
            "count": cnt,
            "valid_count": a["valid_count"],
            "bucket_distribution": {k: int(v) for k, v in sorted(b.items())},
            "primary_frontier_count": primary,
            "secondary_frontier_count": secondary,
            "easy_count": easy,
            "hard_count": hard,
            "invalid_count": invalid,
            "mean_sample_reward": round(a["sum_reward"] / a["valid_count"], 6) if a["valid_count"] else 0.0,
            "cell_pass_rate": round(cell_pass_rate, 6),
            "cell_learnability": round(cell_learnability, 6),
            "smoothed_reward": round(smoothed, 6),
            "easy_rate": round(easy / cnt, 6) if cnt else 0.0,
            "hard_rate": round(hard / cnt, 6) if cnt else 0.0,
            "invalid_rate": round(invalid / cnt, 6) if cnt else 0.0,
            "invalid_reasons": {k: int(v) for k, v in a["invalid_reasons"].most_common()},
        }
    return {
        "actions": actions_out,
        "global_mean_reward": round(global_mean, 6),
        "proposer_reward_total": round(total_reward, 6),
        "proposer_reward_mean": round(total_reward / len(samples), 6) if samples else 0.0,
        "n_samples": len(samples),
        "n_valid": len(valid_rewards),
    }


# --------------------------------------------------------------------------------------------
# Action-Level Bandit policy update (exponentiated weights)
# --------------------------------------------------------------------------------------------
def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, v) / total for k, v in weights.items()}


def merge_qtype_shares(
    curriculum_shares: dict[str, float],
    bandit_shares: dict[str, float],
    allowed_question_types: tuple[str, ...] | list[str],
    *,
    exploration_floor: float = 0.05,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Merge qtype shares without letting the bandit erase curriculum unlocks.

    The curriculum gate is authoritative for qtype unlocking and coarse shares. Bandit shares are
    accepted only as a fallback when the curriculum did not provide a positive value for an allowed
    qtype. This keeps every unlocked qtype non-zero and records that the bandit did not override the
    curriculum.
    """
    allowed = [str(q) for q in allowed_question_types if str(q) and str(q) not in FORBIDDEN_QUESTION_TYPES]
    if not allowed:
        allowed = ["single_choice"]
    curriculum_clean = {
        q: max(0.0, float(curriculum_shares.get(q, 0.0) or 0.0))
        for q in allowed
    }
    bandit_clean = {
        q: max(0.0, float(bandit_shares.get(q, 0.0) or 0.0))
        for q in allowed
    }
    final: dict[str, float] = {}
    for q in allowed:
        if curriculum_clean.get(q, 0.0) > 0:
            final[q] = curriculum_clean[q]
        elif bandit_clean.get(q, 0.0) > 0:
            final[q] = bandit_clean[q]
    mode = "curriculum_authoritative_bandit_missing_qtype_fallback"
    fallback_reason = None
    if not final:
        final = {q: v for q, v in bandit_clean.items() if v > 0}
        mode = "bandit_fallback_no_curriculum_positive_shares"
    if not final:
        final = {q: 1.0 / len(allowed) for q in allowed}
        mode = "uniform_fallback_no_positive_shares"
    normalized = _normalize(final)
    unlocked = [q for q, v in curriculum_clean.items() if v > 0]
    missing_unlocked = [q for q in unlocked if normalized.get(q, 0.0) <= 0]
    if missing_unlocked:
        fallback_reason = f"bandit_or_merge_zeroed_unlocked_qtypes:{missing_unlocked}"
        normalized = _normalize({q: v for q, v in curriculum_clean.items() if v > 0})
        mode = "fallback_to_curriculum_illegal_bandit_output"
    floor_applied: dict[str, dict[str, float]] = {}
    floor = max(0.0, float(exploration_floor or 0.0))
    if floor > 0:
        adjusted = dict(normalized)
        needed = 0.0
        for q in unlocked:
            before = float(adjusted.get(q, 0.0) or 0.0)
            if 0 < before < floor:
                adjusted[q] = floor
                needed += floor - before
                floor_applied[q] = {"before": round(before, 6), "after": round(floor, 6)}
        if needed > 0:
            donors = [q for q in adjusted if q not in floor_applied and adjusted[q] > floor]
            donor_total = sum(adjusted[q] for q in donors)
            if donors and donor_total > 0:
                for q in donors:
                    adjusted[q] = max(0.0, adjusted[q] - needed * (adjusted[q] / donor_total))
                normalized = _normalize(adjusted)
            else:
                normalized = _normalize({q: v for q, v in curriculum_clean.items() if v > 0})
                mode = "fallback_to_curriculum_exploration_floor_infeasible"
    return (
        {q: round(v, 6) for q, v in normalized.items() if v > 0},
        {
            "mode": mode,
            "allowed_question_types": allowed,
            "curriculum_shares": {q: round(v, 6) for q, v in curriculum_clean.items()},
            "bandit_shares": {q: round(v, 6) for q, v in bandit_clean.items()},
            "final_question_type_shares": {q: round(v, 6) for q, v in normalized.items() if v > 0},
            "bandit_overrode_curriculum": False,
            "unlocked_qtype_exploration_floor": floor,
            "exploration_floor_applied": floor_applied,
            "fallback_reason": fallback_reason,
        },
    )


def _higher_hardness(action: str) -> str | None:
    qt, sk, h = action.split("|")
    nxt = {"h0": "h1", "h1": "h2", "h2": None}[h]
    return f"{qt}|{sk}|{nxt}" if nxt else None


def update_policy(
    prior_policy: dict[str, float] | None,
    aggregate: dict[str, Any],
    *,
    action_space: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    eta: float = DEFAULT_ETA,
    epsilon: float = DEFAULT_EPSILON,
    max_action_share: float = DEFAULT_MAX_ACTION_SHARE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Exponentiated-weights bandit update with shrinkage, exploration, anti-collapse, and
    distractor-hardness adaptation. Returns the new policy + a full reportable trace.
    """
    actions = action_space or build_phase1_action_space()
    action_stats = aggregate.get("actions", {})
    global_mean = float(aggregate.get("global_mean_reward") or 0.0)

    # prior over the allowed action space (uniform if none / missing keys)
    pi_t: dict[str, float] = {}
    for a in actions:
        pi_t[a] = float((prior_policy or {}).get(a, 0.0))
    if sum(pi_t.values()) <= 0:
        pi_t = {a: 1.0 / len(actions) for a in actions}
    else:
        pi_t = _normalize(pi_t)

    frozen: dict[str, str] = {}
    hardening_triggered: dict[str, str] = {}
    reasons: dict[str, str] = {}
    smoothed: dict[str, float] = {}

    for a in actions:
        st = action_stats.get(a, {})
        cnt = int(st.get("count") or 0)
        # shrinkage reward (recompute from stats so unseen actions get the prior=global_mean)
        if cnt == 0:
            smoothed[a] = global_mean
            reasons[a] = "unseen->global_mean_prior"
            continue
        smoothed[a] = float(st.get("smoothed_reward") or 0.0)
        invalid_rate = float(st.get("invalid_rate") or 0.0)
        hard_rate = float(st.get("hard_rate") or 0.0)
        easy_rate = float(st.get("easy_rate") or 0.0)
        # PROTECTION: freeze high-invalid actions
        if invalid_rate >= HIGH_INVALID_RATE_FREEZE:
            frozen[a] = f"invalid_rate={invalid_rate:.2f}>={HIGH_INVALID_RATE_FREEZE}"
            continue
        # PROTECTION: high 0/5 with no learnability signal -> down-weight + bug_check
        if hard_rate >= HARD_RATE_BUGCHECK_THRESHOLD:
            reasons[a] = f"bug_check:hard_rate={hard_rate:.2f}; downweight"
            smoothed[a] = min(smoothed[a], 0.0)  # 0/5 is not high reward
        # HARDNESS ADAPTATION (0/5-aware): a saturated action (high 5/5) should escalate to the
        # next hardness ONLY if that next level is not already drowning in 0/5. Escalating into a
        # too-hard sibling just trades 5/5 waste for 0/5 waste. If the sibling's hard_rate is high,
        # HOLD at the current hardness instead of pushing further into "too hard".
        if easy_rate >= EASY_RATE_HARDEN_THRESHOLD:
            nxt = _higher_hardness(a)
            nxt_hard_rate = float((action_stats.get(nxt) or {}).get("hard_rate") or 0.0) if nxt else 1.0
            nxt_seen = int((action_stats.get(nxt) or {}).get("count") or 0) if nxt else 0
            if nxt and nxt in actions and not (nxt_seen > 0 and nxt_hard_rate >= HARD_RATE_BUGCHECK_THRESHOLD):
                hardening_triggered[a] = f"easy_rate={easy_rate:.2f}->boost {nxt} (nxt_hard_rate={nxt_hard_rate:.2f})"
                # discourage staying at this saturated hardness
                smoothed[a] = min(smoothed[a], global_mean * 0.5)
            elif nxt and nxt_seen > 0 and nxt_hard_rate >= HARD_RATE_BUGCHECK_THRESHOLD:
                # next level is too hard -> do NOT escalate; hold here (this hardness is the band edge)
                hardening_triggered[a] = f"easy_rate={easy_rate:.2f}; HOLD (next {nxt} too hard, hard_rate={nxt_hard_rate:.2f})"
                smoothed[a] = max(smoothed[a], global_mean)  # keep this level as the best available
            else:
                hardening_triggered[a] = f"easy_rate={easy_rate:.2f}; max hardness saturated -> downweight skill"
                smoothed[a] = min(smoothed[a], 0.0)

    # exponentiated-weights proposal over non-frozen actions
    proposal: dict[str, float] = {}
    for a in actions:
        if a in frozen:
            proposal[a] = 0.0
            continue
        proposal[a] = pi_t[a] * math.exp(smoothed[a] / max(1e-6, temperature))
    # if hardening triggered, give the higher-hardness sibling an exploration bump — but ONLY when
    # we actually decided to escalate (msg contains "boost"), not when we are holding because the
    # next level is already too hard.
    for a, _msg in hardening_triggered.items():
        if "boost" not in _msg:
            continue
        nxt = _higher_hardness(a)
        if nxt and nxt in proposal and nxt not in frozen:
            proposal[nxt] = max(proposal[nxt], pi_t.get(nxt, 0.0) * math.exp(global_mean / max(1e-6, temperature)) + 0.05)
    proposal = _normalize(proposal)

    # soft interpolation
    pi_next = {a: (1 - eta) * pi_t[a] + eta * proposal[a] for a in actions}
    # epsilon exploration floor over NON-frozen allowed actions
    live = [a for a in actions if a not in frozen]
    if live:
        unif = {a: (1.0 / len(live) if a in frozen else 0.0) for a in actions}
        unif = {a: (1.0 / len(live) if a not in frozen else 0.0) for a in actions}
        pi_next = {a: (1 - epsilon) * pi_next[a] + epsilon * unif[a] for a in actions}
    # frozen actions -> 0
    for a in frozen:
        pi_next[a] = 0.0
    pi_next = _normalize(pi_next)

    # anti-collapse: cap any single action, redistribute excess to other live actions
    pi_next = _cap_action_share(pi_next, frozen, max_action_share)

    # build change reasons
    for a in actions:
        if a in frozen:
            reasons[a] = f"FROZEN:{frozen[a]}"
        elif a in hardening_triggered:
            reasons.setdefault(a, hardening_triggered[a])
        else:
            delta = pi_next[a] - pi_t[a]
            reasons.setdefault(a, f"{'up' if delta>0 else 'down' if delta<0 else 'flat'}weight (R̂={smoothed[a]:.3f})")

    return {
        "old_action_weights": {a: round(pi_t[a], 6) for a in actions},
        "new_action_weights": {a: round(pi_next[a], 6) for a in actions},
        "smoothed_rewards": {a: round(smoothed[a], 6) for a in actions},
        "action_weight_change_reason": reasons,
        "frozen_actions": frozen,
        "hardening_triggered_actions": hardening_triggered,
        "hyperparameters": {
            "alpha": alpha, "temperature": temperature, "eta": eta,
            "epsilon": epsilon, "max_action_share": max_action_share,
        },
    }


def _cap_action_share(policy: dict[str, float], frozen: dict[str, str], max_share: float) -> dict[str, float]:
    if not policy:
        return policy
    live = [a for a in policy if a not in frozen]
    if len(live) <= 1:
        return policy
    vals = dict(policy)
    for _ in range(100):
        total = sum(vals.values()) or 1.0
        vals = {a: v / total for a, v in vals.items()}
        offenders = [a for a in live if vals[a] > max_share + 1e-9]
        if not offenders:
            break
        excess = sum(vals[a] - max_share for a in offenders)
        for a in offenders:
            vals[a] = max_share
        receivers = [a for a in live if a not in offenders]
        rtotal = sum(vals[a] for a in receivers) or float(len(receivers))
        for a in receivers:
            vals[a] += excess * (vals[a] / rtotal if rtotal else 1.0 / len(receivers))
    return _normalize(vals)


# --------------------------------------------------------------------------------------------
# Policy -> generator difficulty_plan (question_type_shares + cell_weights + per-cell hardness)
# --------------------------------------------------------------------------------------------
def policy_to_difficulty_plan(action_policy: dict[str, float]) -> dict[str, Any]:
    """Project the action policy onto the generator's plan knobs.

    question_type_shares : summed action mass per question_type.
    cell_weights         : summed action mass per (qtype|attack_skill) cell.
    cell_hardness_mix    : per cell, the hardness distribution (h0/h1/h2) the generator should use.
    """
    qt_share: dict[str, float] = defaultdict(float)
    cell_w: dict[str, float] = defaultdict(float)
    cell_hard: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for action, w in action_policy.items():
        if w <= 0:
            continue
        qt, sk, h = action.split("|")
        qt_share[qt] += w
        cell = f"{qt}|{sk}"
        cell_w[cell] += w
        cell_hard[cell][h] += w
    # normalize hardness mix within each cell
    cell_hardness_mix = {}
    for cell, hmix in cell_hard.items():
        tot = sum(hmix.values()) or 1.0
        cell_hardness_mix[cell] = {h: round(v / tot, 6) for h, v in sorted(hmix.items())}
    return {
        "question_type_shares": {k: round(v, 6) for k, v in _normalize(dict(qt_share)).items()},
        "cell_weights": {k: round(v, 6) for k, v in dict(cell_w).items()},
        "cell_hardness_mix": cell_hardness_mix,
        "action_weights": {k: round(v, 6) for k, v in action_policy.items() if v > 0},
        "source": "action_level_bandit_controller",
    }


def run_bandit_step(
    samples: list[dict[str, Any]],
    prior_action_policy: dict[str, float] | None,
    *,
    allowed_question_types: tuple[str, ...] = PHASE1_ALLOWED_QUESTION_TYPES,
    alpha: float = DEFAULT_ALPHA,
    temperature: float = DEFAULT_TEMPERATURE,
    eta: float = DEFAULT_ETA,
    epsilon: float = DEFAULT_EPSILON,
    max_action_share: float = DEFAULT_MAX_ACTION_SHARE,
) -> dict[str, Any]:
    """One controller step: gated learnability reward -> action aggregation -> bandit update ->
    next-round difficulty_plan. Pure / offline. Returns a full report + the next policy/plan.
    """
    allowed_question_types = tuple(q for q in allowed_question_types if q not in FORBIDDEN_QUESTION_TYPES)
    if not allowed_question_types:
        allowed_question_types = PHASE1_ALLOWED_QUESTION_TYPES
    action_space = build_phase1_action_space(allowed_question_types)
    agg = aggregate_actions(samples, allowed_question_types=allowed_question_types, alpha=alpha)
    upd = update_policy(
        prior_action_policy, agg, action_space=action_space,
        temperature=temperature, eta=eta, epsilon=epsilon, max_action_share=max_action_share, alpha=alpha,
    )
    next_plan = policy_to_difficulty_plan(upd["new_action_weights"])
    # layered frontier accounting from buckets
    bsum = Counter()
    for a, st in agg["actions"].items():
        for k, v in st["bucket_distribution"].items():
            bsum[k] += v
    primary = int(bsum.get("2/5", 0)) + int(bsum.get("3/5", 0))
    secondary = int(bsum.get("1/5", 0)) + int(bsum.get("4/5", 0))
    report = {
        "controller": "action_level_bandit_controller",
        "reward_model": "gated_proposer_reward(learnability)",
        "allowed_question_types": list(allowed_question_types),
        "forbidden_question_types": list(FORBIDDEN_QUESTION_TYPES),
        "proposer_reward_total": agg["proposer_reward_total"],
        "proposer_reward_mean": agg["proposer_reward_mean"],
        "global_mean_reward": agg["global_mean_reward"],
        "by_action_reward": {a: st["smoothed_reward"] for a, st in agg["actions"].items()},
        "by_action_mean_sample_reward": {a: st["mean_sample_reward"] for a, st in agg["actions"].items()},
        "by_action_bucket_distribution": {a: st["bucket_distribution"] for a, st in agg["actions"].items()},
        "by_action_easy_rate": {a: st["easy_rate"] for a, st in agg["actions"].items()},
        "by_action_hard_rate": {a: st["hard_rate"] for a, st in agg["actions"].items()},
        "by_action_invalid_rate": {a: st["invalid_rate"] for a, st in agg["actions"].items()},
        "old_action_weights": upd["old_action_weights"],
        "new_action_weights": upd["new_action_weights"],
        "action_weight_change_reason": upd["action_weight_change_reason"],
        "frozen_actions": upd["frozen_actions"],
        "hardening_triggered_actions": upd["hardening_triggered_actions"],
        "primary_frontier_count": primary,
        "secondary_frontier_count": secondary,
        "frontier_candidate_count": primary + secondary,
        "weighted_frontier_candidate_score": round(1.0 * primary + 0.5 * secondary, 4),
        "easy_rate": round(int(bsum.get("5/5", 0)) / agg["n_samples"], 6) if agg["n_samples"] else 0.0,
        "hard_rate": round(int(bsum.get("0/5", 0)) / agg["n_samples"], 6) if agg["n_samples"] else 0.0,
        "hyperparameters": upd["hyperparameters"],
        "next_difficulty_plan": next_plan,
        "next_action_policy": upd["new_action_weights"],
    }
    return report
