#!/usr/bin/env python3
"""Generic auto graph-expansion self-play runner.

The historical r2 generator is kept as the low-level implementation for LLM
planning, graph extension, dialogue realization, QA construction, and hard
validation. This runner makes the self-play round explicit:

  clean_v1 parents  -> round r2
  round dataset r2  -> round r3
  round dataset r3  -> round r4

The child round naming, gold source, trap type, fixed solver, output paths, and
policy provenance are all parameters rather than r2-specific constants.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# vLLM's FlashInfer sampler JIT-fails on SM 12.x (Blackwell) GPUs with CUDA < 12.9,
# silently corrupting solver decoding into repetitive gibberish ("the the", "000000")
# and forcing every sample to 0/5. Disabling it before vLLM is imported (the import is
# lazy, inside the eval functions) restores correct decoding for both the in-process
# eval (report_builder -> base.evaluate_with_vllm) and any subprocess eval
# (eval_v13_pilot_vllm.py). setdefault lets an explicit override still win.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

for _thread_env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    _thread_env_raw = str(os.environ.get(_thread_env_key, "")).strip()
    try:
        _thread_env_valid = int(_thread_env_raw) > 0
    except ValueError:
        _thread_env_valid = False
    if not _thread_env_valid:
        os.environ[_thread_env_key] = "1"



ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_r2_llm_graph_expansion_challenger as roundgen  # noqa: E402
import generate_v13_pilot as v13  # noqa: E402
from current_method import api_challenger_generation as current_generation  # noqa: E402
from current_method import challenger_policy_update as current_policy  # noqa: E402
from current_method import difficulty_controller as current_difficulty  # noqa: E402
from current_method import frontier_selection as current_frontier  # noqa: E402
from current_method import native_challenger_generation as native_generation  # noqa: E402
from current_method import report_builder as current_reports  # noqa: E402
from current_method import seed_audit as current_seed_audit  # noqa: E402


QUESTION_TYPES = ("single_choice", "multi_select", "ordering", "state_reconstruction")
OPERATORS = (
    "delayed_invalidation",
    "scope_limited_update",
    "cancellation_then_restoration",
    "cross_storyline_conflict",
    "stale_memory_option_trap",
    "trial_state_vs_main_state",
    "removed_state_distractor",
    "earlier_valid_later_invalid",
    "multi_hop_state_reconstruction",
    "ordering_under_revisions",
)

DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_R2_PARENT_DIR = ROOT / "eval_sets/graph_expansion_challenger_v1/r2_auto_full"
DEFAULT_R2_EVAL_DIR = ROOT / "eval_reports/graph_expansion_challenger_v1/solver_r2_ctx9216_full_eval"
DEFAULT_R2_MERGED = ROOT / "outputs/qwen3_4b_lora_r2_frontier_plus_clean_replay_s0_s2_from_r1_e1_ctx9216_full_merged"
DEFAULT_R2_ADAPTER = ROOT / "outputs/qwen3_4b_lora_r2_frontier_plus_clean_replay_s0_s2_from_r1_e1_ctx9216_full"
DEFAULT_RUN_NAME = "current_method_r3_from_r2"
DEFAULT_RUN_ROOT = ROOT / "current_method_runs/current_method_r3_from_r2"
DEFAULT_R2_SAMPLE0_5_PARENT_DIR = ROOT / "eval_sets/graph_expansion_challenger_v1/r2_auto_full_sample0_5_complete"


@dataclass(frozen=True)
class RoundSpec:
    round_id: str
    round_number: int
    parent_round_id: str | None
    parent_round_number: int | None
    fixed_solver_name: str
    generator_child_id: str = "r2"

    @property
    def gold_origin(self) -> str:
        return f"programmatic_from_validated_{self.round_id}_state_chain"

    @property
    def trap_type(self) -> str:
        return f"{self.round_id}_chain_extension"

    @property
    def attack_type(self) -> str:
        return f"llm_assisted_graph_expansion_{self.round_id}"

    @property
    def hard_mode(self) -> str:
        return f"{self.round_id}_llm_graph_expansion"

    @property
    def node_origin(self) -> str:
        return f"{self.round_id}_llm_assisted_graph_update"

    @property
    def difficulty(self) -> str:
        return f"frontier_{self.round_id}_auto_from_{safe_name(self.fixed_solver_name)}"

    @property
    def quality_tier(self) -> str:
        return f"graph_expansion_{self.round_id}_auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic solver-in-the-loop auto graph-expansion round runner.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT.relative_to(ROOT)))
    parser.add_argument("--round-id", default="current_method_r3", help="Child round id, e.g. current_method_r3.")
    parser.add_argument("--round-number", type=int, default=3)
    parser.add_argument("--parent-round-id", default="r2")
    parser.add_argument("--parent-round-number", type=int, default=2)
    parser.add_argument("--fixed-solver-name", default="solver_r2_ctx9216_full", help="For provenance, e.g. solver_r2_ctx9216_full.")
    parser.add_argument("--parent-mode", choices=("clean_v1", "round_dataset"), default="round_dataset")
    parser.add_argument("--parent-dir", default=str(DEFAULT_R2_SAMPLE0_5_PARENT_DIR), help="Required for --parent-mode round_dataset.")
    parser.add_argument("--policy-source-eval-dir", default=str(DEFAULT_R2_EVAL_DIR))
    parser.add_argument("--policy-solver-key", default=None, help="Accuracy key prefix in policy source, e.g. solver_r2.")
    parser.add_argument("--policy", default=None, help="Existing policy JSON. If omitted, policy is built from --policy-source-eval-dir.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-samples", default="sample0,sample1,sample2,sample3,sample4,sample5")
    parser.add_argument("--dev-samples", default="sample6")
    parser.add_argument("--heldout-samples", default="sample7,sample8,sample9")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-dev", type=int, default=75)
    parser.add_argument("--heldout-per-split", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="Use 1000/150/450 split size.")
    parser.add_argument("--train-only", action="store_true", help="Generate/evaluate only the train split. Useful for incremental train-only parent datasets.")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--min-added-turns", type=int, default=48)
    parser.add_argument("--max-added-turns", type=int, default=88)
    parser.add_argument("--min-extension-delta", type=int, default=8)
    parser.add_argument("--max-extension-delta", type=int, default=14)
    parser.add_argument("--target-extension-delta", type=int, default=12)
    parser.add_argument("--min-final-depth", type=int, default=36)
    parser.add_argument("--max-final-depth", type=int, default=56)
    parser.add_argument("--llm-workers", type=int, default=2)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--max-llm-retries", type=int, default=1)
    parser.add_argument("--max-attempt-multiplier", type=int, default=120)
    parser.add_argument("--allow-fallback-plan", action="store_true")
    parser.add_argument("--allow-policy-fallback", action="store_true")
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=1,
        help=(
            "Number of challenger samples to generate concurrently (graph_conditioned_api_dialogue only). "
            "1 (default) is the byte-equivalent serial path. 4-6 is a safe speedup given the API is I/O-bound; "
            "higher risks API rate limits (429). Planned-cell assignment, hard caps, counts, and checkpoints "
            "stay serial in the collector, so the accepted distribution remains deterministic."
        ),
    )
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=0,
        help=(
            "Per-slot attempt cap for graph_conditioned_api_dialogue (0/None = off, old behavior). "
            "When >0, a target sample slot that fails this many times is SKIPPED (the driver stops "
            "submitting API calls for it and advances the plan cursor to the next feasible cell), "
            "instead of letting a hard/bad cell soak unbounded retries. Decision is made BEFORE "
            "submitting the next attempt. Recommended 4 for closed-loop / easy_v2 runs."
        ),
    )
    parser.add_argument(
        "--generation-backend",
        choices=("old_gate_compat", "native_current_method"),
        default="old_gate_compat",
        help="Generation backend for current_method rounds. Default keeps the old_gate compatibility path.",
    )
    parser.add_argument(
        "--native-challenger-mode",
        choices=("template", "graph_conditioned_api_dialogue"),
        default="template",
        help="Native current_method mode. Default template preserves native v0 behavior; graph_conditioned_api_dialogue is opt-in.",
    )
    parser.add_argument(
        "--native-curriculum-mode",
        choices=("default", "r3_long_easy", "r3_long_easy_v2"),
        default="default",
        help=(
            "Opt-in curriculum profile for graph_conditioned_api_dialogue. 'default' preserves the current "
            "r3_long distribution exactly. 'r3_long_easy' reduces required events, appended turns, token delta, "
            "and hard attack-skill / question-type share. 'r3_long_easy_v2' adds a deterministic per-sample "
            "question_type/attack_skill quota, hard caps on the accepted set, and forbids silent fallback to "
            "ordering (the reliable D=1 foundation for the closed-loop difficulty controller)."
        ),
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--closed-loop-difficulty",
        action="store_true",
        help=(
            "Opt-in: before generating, inject the closed-loop difficulty controller's plan "
            "(read from --difficulty-state-path) into the easy_v2 quota scheduler; after the "
            "round, re-run a CLEAN per-cell eval (FlashInfer sampler forced off) and update the "
            "persistent controller state. Requires --native-curriculum-mode r3_long_easy_v2."
        ),
    )
    parser.add_argument(
        "--disable-smoke-controller-local-state",
        action="store_true",
        help=(
            "When closed-loop difficulty is enabled, always write controller updates to the "
            "persistent difficulty-state path instead of diverting smoke-sized runs to a per-run "
            ".smoke.json file."
        ),
    )
    parser.add_argument(
        "--skip-closed-loop-controller-update",
        action="store_true",
        help=(
            "With --closed-loop-difficulty, inject the prior difficulty plan before generation "
            "but skip the expensive post-round clean per-cell controller eval/update. Useful when "
            "an outer adaptive smoke loop will update the difficulty state from the round's k1/k5 eval."
        ),
    )
    parser.add_argument(
        "--difficulty-state-path",
        default=None,
        help="Path to the persistent difficulty controller state JSON (defaults to <run_root>/../difficulty_controller_state.json).",
    )
    parser.add_argument(
        "--controller-state-path",
        dest="difficulty_state_path",
        default=None,
        help="Alias for --difficulty-state-path; prefer a solver-specific state for r4 hardening.",
    )
    parser.add_argument(
        "--min-length-delta-tokens",
        type=int,
        default=0,
        help="For graph_conditioned_api_dialogue hardening, require final dialogue tokens to exceed parent by this many tokens.",
    )
    parser.add_argument(
        "--length-guard-parent-baseline-tokens",
        type=int,
        default=0,
        help=(
            "For r4+ hardening, raise the official length-guard parent baseline to at least this many tokens "
            "before applying --min-length-delta-tokens. Use this to pin the guard to the previous round's "
            "full-dialogue acceptance baseline when tokenizer/summary token definitions differ."
        ),
    )
    parser.add_argument("--run-k5", action="store_true")
    parser.add_argument("--eval-backend", choices=("auto", "vllm", "transformers", "heuristic"), default="auto")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--solver-model", default=str(DEFAULT_R2_MERGED))
    parser.add_argument("--solver-tokenizer", default=None)
    parser.add_argument("--solver-adapter", default=str(DEFAULT_R2_ADAPTER))
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--max-length", type=int, default=15360)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k5-eval-temperature", type=float, default=1.0,
                        help="Sampling temperature for the k=5 learnability-bucket eval ONLY (k=1 true-accuracy eval is unaffected). "
                             "Higher than --temperature so a confident solver does not return identical answers 5x, which collapsed "
                             "single_choice buckets to only 0/5 or 5/5. 1.0 gives per-sample bucket diversity; set 0 to reuse --temperature.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--challenger-temperature", type=float, default=0.25)
    parser.add_argument("--challenger-max-tokens", type=int, default=2048,
                        help="Max output tokens for the dialogue-expansion API call. Lowered from 4096: observed appended dialogue is ~1082 tokens, so 2048 gives ample headroom while truncating runaway generations and reducing latency.")
    parser.add_argument("--graph-expansion-max-tokens", type=int, default=2048,
                        help="Max output tokens for the graph/event-plan API call. Raised from 1024 to 2048 after observing invalid_json truncation failures; the event-plan JSON for 3-8 events can exceed 1024.")
    parser.add_argument("--parent-context-tail-turns", type=int, default=12)
    parser.add_argument("--parent-context-relevant-turns", type=int, default=24)
    parser.add_argument("--dialogue-api-parent-token-budget", type=int, default=2048)
    parser.add_argument(
        "--chained-qa-dependency-targeting",
        action="store_true",
        help=(
            "Opt-in for chained parent expansion probes: QA-last must build questions whose "
            "gold/evidence uses both promoted parent context and newly appended child context, "
            "with cross-context evidence distance and non-obvious distractors. Default off."
        ),
    )
    parser.add_argument(
        "--boundary-aware-qa-diagnostics",
        action="store_true",
        help=(
            "Opt-in for chained QA-last: emit boundary-aware item diagnostics and hard-reject only "
            "certain bad QA items such as exact duplicate options, identical gold pairs, or "
            "gold/evidence inconsistency. Default off."
        ),
    )
    parser.add_argument(
        "--boundary-aware-qa-planner",
        action="store_true",
        help=(
            "Opt-in with chained QA-last: build a graph-derived item plan before constructing "
            "question/options, then use boundary-aware diagnostics and repair as fallback. Default off."
        ),
    )
    parser.add_argument(
        "--stage33-compat-boundary-planner",
        action="store_true",
        help=(
            "Opt-in recovery path for Stage33/35-compatible boundary generation. When the "
            "permission_visibility_scope planner is enabled and the graph API does not provide "
            "natural hard-negative paths, force anchored deterministic permission hard negatives "
            "instead of zero-yielding on no_permission_visibility_scope_hard_negative_plan. "
            "Default off."
        ),
    )
    parser.add_argument(
        "--qa-operation-answer-type-diversification",
        action="store_true",
        help=(
            "Opt-in API selfplay mode for anchor-first QA operation / answer-type diversification. "
            "The API challenger must emit qa_operation/answer_type/query_plan; default generation remains unchanged."
        ),
    )
    parser.add_argument(
        "--boundary-item-difficulty-target",
        choices=("easy", "boundary", "harder_boundary", "hard"),
        default=None,
        help=(
            "Opt-in difficulty target for --boundary-aware-qa-planner. easy uses shallow/obvious "
            "grounded checks, boundary targets medium-hard 1/5-4/5 items, harder_boundary is a "
            "narrower medium-high profile for over-easy cells, and hard keeps diagnostic deep/close "
            "hard negatives. Default unset preserves the existing planner behavior."
        ),
    )
    parser.add_argument(
        "--deterministic-permission-hard-negatives",
        action="store_true",
        help=(
            "Opt-in with --boundary-aware-qa-planner for multi_select|permission_visibility_scope: "
            "when the graph API does not provide natural hard-negative paths, synthesize anchored "
            "graph-derived scope/override distractor events instead of falling back to placeholder "
            "archive/report false surfaces. Default off."
        ),
    )
    parser.add_argument(
        "--boundary-aware-qa-repair",
        action="store_true",
        help=(
            "Opt-in with --boundary-aware-qa-diagnostics: repair QA/options/distractors locally "
            "before hard-rejecting repairable boundary-aware item risks. Graph/dialogue/alignment "
            "are kept fixed. Default off."
        ),
    )
    parser.add_argument(
        "--boundary-aware-qa-repair-attempts",
        type=int,
        default=2,
        help="Number of local boundary-aware QA/options repair attempts before hard reject. Default 2.",
    )
    parser.add_argument(
        "--api-yield-normalizers",
        action="store_true",
        help=(
            "Opt-in safe API yield normalizers for graph/dialogue JSON wrappers. Only unwraps "
            "existing schema/content such as graph_update.events or dialogue_proposal; does not "
            "invent graph events, dialogue, QA, or evidence."
        ),
    )
    parser.add_argument(
        "--graph-proposal-cache",
        action="store_true",
        help=(
            "Opt-in valid graph proposal cache for graph_conditioned_api_dialogue. Validated graph "
            "plans are cached atomically and reused for dialogue retries/resume; invalid/leaky/"
            "contradictory graphs are not cached."
        ),
    )
    parser.add_argument(
        "--graph-proposal-cache-dir",
        default=None,
        help="Optional directory for --graph-proposal-cache; defaults to <output_dir>/graph_proposal_cache.",
    )
    parser.add_argument(
        "--query-conditioned-graph-expansion",
        action="store_true",
        help=(
            "Opt-in graph-first experiment: condition the graph/event-plan API on a hidden "
            "downstream query objective before dialogue verbalization and QA-last rendering. "
            "Default off; currently supports multi_select|permission_visibility_scope."
        ),
    )
    parser.add_argument(
        "--controller-patch-v2",
        action="store_true",
        help=(
            "Opt-in provenance guard for challenger_autodesigned_policy_v2 probes. Requires the injected "
            "difficulty_plan qtype_curriculum to carry controller_patch_v2=true; default off."
        ),
    )
    parser.add_argument("--partial-samples", default=None)
    parser.add_argument("--failed-api-calls", default=None)
    parser.add_argument("--max-api-failures", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--frontier-splits", default="train")
    parser.add_argument("--forbid-frontier-from-dev-heldout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-largest-question-type-share", type=float, default=0.35)
    parser.add_argument("--max-largest-attack-skill-share", type=float, default=0.25)
    parser.add_argument("--require-quality-audit-before-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-training-pipeline", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "unknown"


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if not source.exists():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {"base_url": roundgen.DEFAULT_CHALLENGER_BASE_URL, "model": roundgen.DEFAULT_CHALLENGER_MODEL, "api_key": None}
    payload = read_json(path)
    api = payload.get("api") if isinstance(payload, dict) else {}
    if not isinstance(api, dict):
        api = {}
    return {
        "base_url": api.get("base_url") or roundgen.DEFAULT_CHALLENGER_BASE_URL,
        "model": api.get("model") or roundgen.DEFAULT_CHALLENGER_MODEL,
        "api_key": api.get("api_key"),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def ratio_weights(raw: dict[str, float], *, floor: float = 0.03, smoothing: float = 0.20) -> dict[str, float]:
    keys = list(raw)
    if not keys:
        return {}
    positive = {key: max(0.0, float(raw.get(key, 0.0))) for key in keys}
    total = sum(positive.values())
    base = {key: positive[key] / total for key in keys} if total > 0 else {key: 1.0 / len(keys) for key in keys}
    uniform = 1.0 / len(keys)
    mixed = {key: (1.0 - smoothing) * base[key] + smoothing * uniform for key in keys}
    if floor * len(keys) >= 1.0:
        return {key: round(uniform, 6) for key in keys}
    remaining = 1.0 - floor * len(keys)
    weights = {key: floor + remaining * mixed[key] for key in keys}
    denom = sum(weights.values())
    return {key: round(weights[key] / denom, 6) for key in keys}


def solver_accuracy(item: dict[str, Any], solver_key: str | None) -> float | None:
    if not isinstance(item, dict):
        return None
    candidate_keys = []
    if solver_key:
        candidate_keys.append(f"{solver_key}_accuracy")
    candidate_keys.extend(("solver_accuracy", "current_solver_accuracy", "accuracy"))
    for key in candidate_keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    # Fall back to the strongest non-r1 solver key in comparison reports.
    for key, value in sorted(item.items()):
        if key.startswith("solver_") and key.endswith("_accuracy") and key != "solver_r1_accuracy" and isinstance(value, (int, float)):
            return float(value)
    return None


def build_policy(args: argparse.Namespace, spec: RoundSpec, config_info: dict[str, str | None]) -> dict[str, Any]:
    report_dir = Path(args.report_dir)
    if args.policy:
        policy = read_json(args.policy)
        write_json(report_dir / f"challenger_policy_{spec.round_id}_auto.json", policy)
        return policy

    source_dir = Path(args.policy_source_eval_dir)
    paths = {
        "overall_summary": source_dir / "overall_summary.json",
        "status": source_dir / "status.md",
        "question_type_breakdown": source_dir / "question_type_breakdown.json",
        "trap_type_breakdown": source_dir / "trap_type_breakdown.json",
        "state_chain_depth_breakdown": source_dir / "state_chain_depth_breakdown.json",
        "truncation_summary": source_dir / "truncation_summary.json",
        "retention_summary": source_dir / "retention_summary.json",
        "k5_soft_summary": source_dir / "k5_dev_heldout_soft_summary.json",
    }
    loaded = {name: read_json(path) for name, path in paths.items() if path.suffix == ".json" and path.exists()}
    q_payload = loaded.get("question_type_breakdown") or {}
    if isinstance(q_payload, dict) and isinstance(q_payload.get("r2_sets_only"), dict):
        q_stats = q_payload["r2_sets_only"]
    else:
        q_stats = q_payload if isinstance(q_payload, dict) else {}
    q_scores: dict[str, float] = {}
    for key in QUESTION_TYPES:
        item = q_stats.get(key) if isinstance(q_stats, dict) else None
        acc = solver_accuracy(item or {}, args.policy_solver_key)
        n = float((item or {}).get("n", 0.0)) if isinstance(item, dict) else 0.0
        q_scores[key] = max(0.0, n * (1.0 - acc)) if acc is not None else 0.0
    question_mix = ratio_weights(q_scores, floor=0.08, smoothing=0.35)

    op_scores = {key: 0.0 for key in OPERATORS}
    op_scores["ordering_under_revisions"] += q_scores.get("ordering", 0.0)
    op_scores["delayed_invalidation"] += q_scores.get("ordering", 0.0) + 0.5 * q_scores.get("state_reconstruction", 0.0)
    op_scores["multi_hop_state_reconstruction"] += q_scores.get("state_reconstruction", 0.0)
    op_scores["cross_storyline_conflict"] += 0.8 * q_scores.get("state_reconstruction", 0.0)
    op_scores["cancellation_then_restoration"] += 0.6 * q_scores.get("state_reconstruction", 0.0)
    op_scores["stale_memory_option_trap"] += q_scores.get("multi_select", 0.0) + 0.5 * q_scores.get("single_choice", 0.0)
    op_scores["scope_limited_update"] += q_scores.get("single_choice", 0.0) + 0.4 * q_scores.get("multi_select", 0.0)
    op_scores["trial_state_vs_main_state"] += q_scores.get("single_choice", 0.0) + 0.3 * q_scores.get("ordering", 0.0)
    op_scores["removed_state_distractor"] += 0.8 * q_scores.get("multi_select", 0.0)
    op_scores["earlier_valid_later_invalid"] += sum(q_scores.values()) / max(1, len(q_scores))

    policy = {
        "policy_id": f"challenger_policy_{spec.round_id}_auto",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round_spec": asdict(spec),
        "source_summary": {
            "input_report_paths": {name: str(path) for name, path in paths.items() if path.exists()},
            "solver_used_to_build_policy": spec.fixed_solver_name,
            "eval_metric_summary": loaded.get("overall_summary", {}),
        },
        "failure_distribution": {
            "question_type_failure": q_stats,
            "question_type_error_scores": {key: round(value, 6) for key, value in q_scores.items()},
            "trap_type_failure": loaded.get("trap_type_breakdown"),
            "state_chain_depth_failure": loaded.get("state_chain_depth_breakdown"),
            "truncation_notes": loaded.get("truncation_summary"),
            "retention_notes": loaded.get("retention_summary"),
            "k5_soft_summary": loaded.get("k5_soft_summary"),
        },
        "generation_targets": {
            "target_token_range": [12000, 16000],
            "state_chain_extension_delta_range": [args.min_extension_delta, args.max_extension_delta],
            "target_extension_delta": args.target_extension_delta,
            "added_turn_count_range": [args.min_added_turns, args.max_added_turns],
            "final_state_chain_depth_range": [args.min_final_depth, args.max_final_depth],
            "parent_mode": args.parent_mode,
            "parent_dataset": str(args.parent_dir) if args.parent_mode == "round_dataset" else "clean_v1 paired graph_hardened parents",
        },
        "question_type_mix": question_mix,
        "adversarial_operators": list(OPERATORS),
        "operator_weights": ratio_weights(op_scores, floor=0.03, smoothing=0.25),
        "operator_weight_derivation": {
            "method": "solver residual error by question type, smoothed for compositional coverage",
            "raw_operator_scores": {key: round(value, 6) for key, value in op_scores.items()},
        },
        "safety_and_validity_gates": [
            "option validation",
            "leakage check",
            "graph-dialogue alignment",
            "gold derivability",
            "no unconstrained LLM gold",
            "heldout contamination check",
            "state depth math check",
        ],
        "gold_source_policy": {
            "allowed": ["inherited gold", "graph-derived gold", "llm_provisional_validated", "programmatic_from_validated_state_chain"],
            "forbidden": ["unconstrained_llm_gold"],
            "expected": spec.gold_origin,
        },
        "challenger": {
            "model": config_info.get("model"),
            "base_url": config_info.get("base_url"),
            "config_file": str(args.config),
        },
    }
    write_json(report_dir / f"challenger_policy_{spec.round_id}_auto.json", policy)
    return policy


def generation_args(args: argparse.Namespace, policy: dict[str, Any], config_info: dict[str, str | None]) -> argparse.Namespace:
    qmix = policy.get("question_type_mix") if isinstance(policy.get("question_type_mix"), dict) else {}
    if not qmix:
        qmix = {key: 1.0 / len(QUESTION_TYPES) for key in QUESTION_TYPES}
    question_type_mix = ",".join(f"{key}:{float(qmix.get(key, 0.0))}" for key in QUESTION_TYPES)
    return argparse.Namespace(
        no_llm=False,
        require_llm_plan=not args.allow_fallback_plan,
        max_llm_retries=args.max_llm_retries,
        challenger_api_key=config_info.get("api_key"),
        challenger_model=config_info.get("model") or roundgen.DEFAULT_CHALLENGER_MODEL,
        challenger_base_url=config_info.get("base_url") or roundgen.DEFAULT_CHALLENGER_BASE_URL,
        llm_timeout=args.llm_timeout,
        llm_workers=args.llm_workers,
        seed=args.seed,
        min_extension_delta=args.min_extension_delta,
        max_extension_delta=args.max_extension_delta,
        target_extension_delta=args.target_extension_delta,
        min_added_turns=args.min_added_turns,
        min_length_delta_tokens=args.min_length_delta_tokens,
        length_guard_parent_baseline_tokens=args.length_guard_parent_baseline_tokens,
        max_added_turns=args.max_added_turns,
        min_final_depth=args.min_final_depth,
        max_final_depth=args.max_final_depth,
        question_type_mix=question_type_mix,
        max_attempt_multiplier=args.max_attempt_multiplier,
        qa_operation_answer_type_diversification=bool(args.qa_operation_answer_type_diversification),
    )


def record_ids(path: Path) -> set[str]:
    return {str(row.get("record_id")) for row in read_jsonl(path) if row.get("record_id")}


def load_round_dataset_parents(parent_dir: Path) -> dict[str, list[dict[str, Any]]]:
    parents = roundgen.load_inherited_samples(parent_dir)
    split_ids = {
        "train": record_ids(parent_dir / "sft_train.jsonl"),
        "dev": record_ids(parent_dir / "sft_eval_dev.jsonl"),
        "heldout": record_ids(parent_dir / "sft_eval_heldout.jsonl"),
    }
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "heldout": []}
    for parent in parents:
        rid = str(parent.get("expanded_graph_id") or "")
        for split, ids in split_ids.items():
            if rid in ids:
                groups[split].append(parent)
                break
    return groups


def load_clean_v1_parents(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    clean_args = argparse.Namespace(
        train_samples=args.train_samples,
        dev_samples=args.dev_samples,
        heldout_samples=args.heldout_samples,
    )
    parents, _manifest = roundgen.load_auto_full_parents(clean_args)
    return parents


def replace_child_round_text(value: Any, spec: RoundSpec) -> Any:
    if not isinstance(value, str):
        return value
    source = re.escape(spec.generator_child_id)
    value = re.sub(rf"\b{source}\b", spec.round_id, value)
    value = value.replace(f"{spec.generator_child_id}-", f"{spec.round_id}-")
    value = value.replace(spec.generator_child_id.upper(), spec.round_id.upper())
    return value


def replace_last_child_suffix(value: str, spec: RoundSpec, kind: str) -> str:
    child = spec.generator_child_id
    if kind == "sample":
        new = re.sub(rf"_{re.escape(child)}_(\d{{4}})$", f"_{spec.round_id}_" + r"\1", value)
    else:
        new = re.sub(rf"_{re.escape(child)}_exp_(\d{{4}})$", f"_{spec.round_id}_exp_" + r"\1", value)
    return new if new != value else f"{value}_{spec.round_id}"


def patch_round_records(
    sample: dict[str, Any],
    graph: dict[str, Any],
    dialogue: dict[str, Any],
    *,
    spec: RoundSpec,
    split: str,
    heldout_split: str | None,
    parent_source: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample = copy.deepcopy(sample)
    graph = copy.deepcopy(graph)
    dialogue = copy.deepcopy(dialogue)

    old_record_id = str(sample.get("expanded_graph_id") or "")
    old_sample_id = str(sample.get("sample_id") or "")
    old_dialogue_id = str(sample.get("expanded_dialogue_id") or "")
    new_record_id = stable_hash(f"{spec.round_id}|{old_record_id}|{old_sample_id}|{old_dialogue_id}")
    new_sample_id = replace_last_child_suffix(old_sample_id, spec, "sample")
    new_dialogue_id = replace_last_child_suffix(old_dialogue_id, spec, "dialogue")

    state_chain = sample.get("state_chain") if isinstance(sample.get("state_chain"), dict) else {}
    generated_chain_id = str(state_chain.get("chain_id") or "")
    parent_chain_id = str(state_chain.get("parent_chain_id") or sample.get("parent_chain_id") or "")
    new_chain_id = f"{parent_chain_id}:{spec.round_id}" if parent_chain_id else f"{generated_chain_id}:{spec.round_id}"
    old_new_node_ids = {str(node.get("node_id")) for node in sample.get("new_nodes", []) if isinstance(node, dict)}
    old_value_map = {
        normalize(node.get("value")).lower(): replace_child_round_text(node.get("value"), spec)
        for node in sample.get("new_nodes", [])
        if isinstance(node, dict) and normalize(node.get("value"))
    }

    id_map = {}
    for old_id in old_new_node_ids:
        if generated_chain_id and old_id.startswith(generated_chain_id):
            id_map[old_id] = old_id.replace(generated_chain_id, new_chain_id, 1)
        else:
            id_map[old_id] = old_id.replace(f":{spec.generator_child_id}:", f":{spec.round_id}:")

    def patch_node(node: dict[str, Any], *, child_node: bool) -> dict[str, Any]:
        row = copy.deepcopy(node)
        old_id = str(row.get("node_id") or "")
        if old_id in id_map:
            row["node_id"] = id_map[old_id]
        if child_node:
            for key in ("role", "label", "value", "status", "origin"):
                if key in row:
                    row[key] = replace_child_round_text(row[key], spec)
            row["origin"] = spec.node_origin
            row["parent_chain_id"] = parent_chain_id
        return row

    new_nodes = [patch_node(node, child_node=True) for node in sample.get("new_nodes", []) if isinstance(node, dict)]

    patched_chain = copy.deepcopy(state_chain)
    patched_chain.update(
        {
            "chain_id": new_chain_id,
            "parent_chain_id": parent_chain_id,
            "hard_mode": spec.hard_mode,
            "target_role": replace_child_round_text(patched_chain.get("target_role"), spec),
            "target_node_id": id_map.get(str(patched_chain.get("target_node_id")), patched_chain.get("target_node_id")),
            "valid_node_id": id_map.get(str(patched_chain.get("valid_node_id")), patched_chain.get("valid_node_id")),
        }
    )
    patched_chain["nodes"] = [
        patch_node(node, child_node=str(node.get("node_id")) in old_new_node_ids)
        for node in patched_chain.get("nodes", [])
        if isinstance(node, dict)
    ]
    patched_chain["node_count"] = len(patched_chain["nodes"])

    patched_edges = []
    for edge in sample.get("new_edges") or []:
        if not isinstance(edge, dict):
            continue
        row = copy.deepcopy(edge)
        for key in ("edge_id", "source", "target"):
            if key in row:
                text = str(row[key])
                if generated_chain_id and text.startswith(generated_chain_id):
                    text = text.replace(generated_chain_id, new_chain_id, 1)
                row[key] = text.replace(f":{spec.generator_child_id}:", f":{spec.round_id}:")
        patched_edges.append(row)

    child_dialogue_text = replace_child_round_text(sample.get("new_dialogue_text") or "", spec)
    full_text = str(sample.get("expanded_dialogue_text") or "")
    old_child_text = str(sample.get("new_dialogue_text") or "")
    if old_child_text and full_text.endswith(old_child_text):
        parent_text = full_text[: -len(old_child_text)].strip()
        expanded_dialogue_text = "\n".join(part for part in (parent_text, child_dialogue_text) if part).strip()
    else:
        expanded_dialogue_text = replace_child_round_text(full_text, spec)

    sample.update(
        {
            "sample_id": new_sample_id,
            "expanded_graph_id": new_record_id,
            "expanded_dialogue_id": new_dialogue_id,
            f"{spec.round_id}_parent_sample_id": old_sample_id,
            f"{spec.round_id}_parent_record_id": old_record_id,
            f"{spec.round_id}_parent_dataset": parent_source,
            "graph_expansion_round": spec.round_number,
            "question": replace_child_round_text(sample.get("question"), spec),
            "gold_origin": spec.gold_origin,
            "gold_source": replace_child_round_text(sample.get("gold_source"), spec),
            "gold_node": id_map.get(str(sample.get("gold_node")), sample.get("gold_node")),
            "gold_node_role": replace_child_round_text(sample.get("gold_node_role"), spec),
            "target_role": replace_child_round_text(sample.get("target_role"), spec),
            "state_chain": patched_chain,
            "state_chain_nodes": patched_chain.get("nodes", []),
            "state_chain_id": new_chain_id,
            "new_nodes": new_nodes,
            "new_edges": patched_edges,
            "new_dialogue_text": child_dialogue_text,
            "expanded_dialogue_text": expanded_dialogue_text,
            "trap_type": spec.trap_type,
            "attack_type": spec.attack_type,
            "difficulty_for_solver": spec.difficulty,
            "quality_tier": spec.quality_tier,
            "split": split,
            "heldout_split": heldout_split,
            "gold_derivable_from_graph_or_state_chain": True,
            "unconstrained_llm_gold": False,
            "heldout_contamination": False,
        }
    )
    if sample.get("question_type") in {"single_choice", "state_reconstruction"} and sample.get("gold_answer"):
        sample["gold_answer"] = replace_child_round_text(sample["gold_answer"], spec)
    if isinstance(sample.get("options"), dict):
        sample["options"] = {label: old_value_map.get(normalize(value).lower(), value) for label, value in sample["options"].items()}
    if isinstance(sample.get("gold_answers"), list):
        sample["gold_answers"] = [old_value_map.get(normalize(value).lower(), value) for value in sample["gold_answers"]]

    metadata = dict(sample.get("metadata") or {})
    metadata.update(
        {
            "record_id": new_record_id,
            "sample_name": new_dialogue_id,
            "split": split,
            "heldout_split": heldout_split,
            "graph_expansion_round": spec.round_number,
            "parent_round": spec.parent_round_id,
            f"{spec.round_id}_parent_record_id": old_record_id,
            f"{spec.round_id}_parent_sample_id": old_sample_id,
            f"{spec.round_id}_parent_dataset": parent_source,
            "fixed_solver_name": spec.fixed_solver_name,
            "gold_origin": spec.gold_origin,
            "gold_source": sample.get("gold_source"),
            "trap_type": spec.trap_type,
            "attack_type": spec.attack_type,
            "difficulty_for_solver": spec.difficulty,
            "quality_tier": spec.quality_tier,
            "state_chain_depth": sample.get("state_chain_depth"),
            "final_state_chain_depth": sample.get("final_state_chain_depth"),
            "state_chain_extension_delta": sample.get("state_chain_extension_delta"),
            "added_turn_count": sample.get("added_turn_count"),
            "option_validation": True,
            "option_validation_pass": True,
            "leakage_free": True,
            "graph_dialogue_alignment": True,
            "gold_derivable_from_graph_or_state_chain": True,
            "unconstrained_llm_gold": False,
            "heldout_contamination": False,
        }
    )
    sample["metadata"] = metadata

    graph.update(
        {
            "sample_id": new_sample_id,
            "expanded_graph_id": new_record_id,
            "graph_expansion_round": spec.round_number,
            "parent_graph_expansion_round": spec.parent_round_number,
            "parent_chain_id": parent_chain_id,
            "state_chain": patched_chain,
            "new_nodes": new_nodes,
            "new_edges": patched_edges,
            "gold_origin": spec.gold_origin,
            f"{spec.round_id}_parent_record_id": old_record_id,
            f"{spec.round_id}_parent_dataset": parent_source,
        }
    )

    patched_turns = []
    for turn in dialogue.get("new_turns") or []:
        if not isinstance(turn, dict):
            continue
        row = copy.deepcopy(turn)
        row["text"] = replace_child_round_text(row.get("text"), spec)
        row["session_key"] = replace_child_round_text(row.get("session_key"), spec)
        row["graph_node_ids"] = [id_map.get(str(item), str(item)) for item in row.get("graph_node_ids") or []]
        patched_turns.append(row)
    dialogue.update(
        {
            "sample_id": new_sample_id,
            "expanded_dialogue_id": new_dialogue_id,
            "source_split": f"{spec.round_id}_auto_{heldout_split or split}",
            "new_turns": patched_turns,
            "state_chain": patched_chain,
            f"{spec.round_id}_parent_record_id": old_record_id,
            f"{spec.round_id}_parent_dataset": parent_source,
        }
    )
    graph_plan = dialogue.get("graph_edit_plan") if isinstance(dialogue.get("graph_edit_plan"), dict) else {}
    graph_plan.update(
        {
            "round": spec.round_number,
            "parent_round": spec.parent_round_number,
            "parent_chain_id": parent_chain_id,
            "new_nodes": sorted(str(node.get("node_id")) for node in new_nodes),
            "new_edges": [edge.get("edge_id") for edge in patched_edges],
        }
    )
    dialogue["graph_edit_plan"] = graph_plan
    return sample, graph, dialogue


def hard_gate_ok(sample: dict[str, Any]) -> bool:
    return (
        sample.get("option_validation_pass") is True
        and sample.get("leakage_free") is True
        and sample.get("graph_dialogue_alignment") is True
        and sample.get("gold_derivable_from_graph_or_state_chain") is True
        and sample.get("unconstrained_llm_gold") is not True
        and sample.get("heldout_contamination") is not True
        and int(sample.get("final_state_chain_depth") or 0)
        == int(sample.get("inherited_state_chain_depth") or 0) + int(sample.get("state_chain_extension_delta") or 0)
    )


def int_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 4)}


def split_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(samples),
        "question_type_counts": dict(Counter(str(sample.get("question_type")) for sample in samples).most_common()),
        "trap_type_counts": dict(Counter(str(sample.get("trap_type")) for sample in samples).most_common()),
        "added_turn_count_stats": int_stats([int(sample.get("added_turn_count") or 0) for sample in samples]),
        "inherited_state_chain_depth_stats": int_stats([int(sample.get("inherited_state_chain_depth") or 0) for sample in samples]),
        "state_chain_extension_delta_stats": int_stats([int(sample.get("state_chain_extension_delta") or 0) for sample in samples]),
        "final_state_chain_depth_stats": int_stats([int(sample.get("final_state_chain_depth") or 0) for sample in samples]),
        "gold_origin_counts": dict(Counter(str(sample.get("gold_origin")) for sample in samples).most_common()),
        "llm_plan_used_counts": dict(Counter(str(bool(sample.get("llm_plan_used"))) for sample in samples).most_common()),
    }


def validation_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    depth_bad = sum(
        1
        for sample in samples
        if int(sample.get("final_state_chain_depth") or 0)
        != int(sample.get("inherited_state_chain_depth") or 0) + int(sample.get("state_chain_extension_delta") or 0)
    )
    valid = sum(1 for sample in samples if hard_gate_ok(sample))
    return {
        "total": total,
        "valid": valid,
        "validation_pass_rate": round(valid / total, 6) if total else None,
        "option_validation_bad": sum(1 for sample in samples if sample.get("option_validation_pass") is not True),
        "leakage_bad": sum(1 for sample in samples if sample.get("leakage_free") is not True),
        "graph_dialogue_alignment_bad": sum(1 for sample in samples if sample.get("graph_dialogue_alignment") is not True),
        "gold_not_derivable": sum(1 for sample in samples if sample.get("gold_derivable_from_graph_or_state_chain") is not True),
        "heldout_contamination": sum(1 for sample in samples if sample.get("heldout_contamination") is True),
        "unconstrained_llm_gold": sum(1 for sample in samples if sample.get("unconstrained_llm_gold") is True),
        "state_depth_math_bad": depth_bad,
        "all_checks_passed": total > 0 and valid == total,
    }


def generate_split(
    parents: list[dict[str, Any]],
    n: int,
    gen_args: argparse.Namespace,
    split_name: str,
    training_use: str,
    *,
    spec: RoundSpec,
    split: str,
    heldout_split: str | None,
    parent_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[tuple[bool, list[str]]]]:
    samples, graphs, dialogues, validations = roundgen.generate_split_parallel(parents, n, gen_args, split_name, training_use)
    patched = [
        patch_round_records(sample, graph, dialogue, spec=spec, split=split, heldout_split=heldout_split, parent_source=parent_source)
        for sample, graph, dialogue in zip(samples, graphs, dialogues)
    ]
    return [row[0] for row in patched], [row[1] for row in patched], [row[2] for row in patched], validations


def generate_round(args: argparse.Namespace, spec: RoundSpec, policy: dict[str, Any], config_info: dict[str, str | None]) -> dict[str, Any]:
    if args.parent_mode == "clean_v1":
        parent_groups = load_clean_v1_parents(args)
        parent_source = "clean_v1 paired graph_hardened parents under data/locomo_processed_datasets"
    else:
        parent_groups = load_round_dataset_parents(Path(args.parent_dir))
        parent_source = str(Path(args.parent_dir))

    required_splits = ("train",) if args.train_only else ("train", "dev", "heldout")
    missing = {key: len(parent_groups.get(key, [])) for key in required_splits if not parent_groups.get(key)}
    if missing:
        raise RuntimeError(f"Missing parent groups: {missing}")
    logging.info("Loaded parents: %s", {key: len(value) for key, value in parent_groups.items()})

    gen_args = generation_args(args, policy, config_info)
    train_samples, train_graphs, train_dialogues, train_validations = generate_split(
        parent_groups["train"], args.n_train, gen_args, f"{spec.round_id}_auto_train", "train_candidate",
        spec=spec, split="train", heldout_split=None, parent_source=parent_source
    )
    if args.train_only:
        dev_samples: list[dict[str, Any]] = []
        dev_graphs: list[dict[str, Any]] = []
        dev_dialogues: list[dict[str, Any]] = []
        dev_validations: list[tuple[bool, list[str]]] = []
    else:
        dev_samples, dev_graphs, dev_dialogues, dev_validations = generate_split(
            parent_groups["dev"], args.n_dev, gen_args, f"{spec.round_id}_auto_dev", "dev_sanity_eval_only",
            spec=spec, split="dev", heldout_split=parse_csv(args.dev_samples)[0] if parse_csv(args.dev_samples) else None, parent_source=parent_source
        )

    heldout_samples: list[dict[str, Any]] = []
    heldout_graphs: list[dict[str, Any]] = []
    heldout_dialogues: list[dict[str, Any]] = []
    heldout_validations: list[tuple[bool, list[str]]] = []
    for heldout_name in ([] if args.train_only else parse_csv(args.heldout_samples)):
        split_parents = [
            parent
            for parent in parent_groups["heldout"]
            if str(parent.get("heldout_split") or parent.get("source_split") or parent.get("metadata", {}).get("heldout_split")) == heldout_name
        ]
        if not split_parents and args.parent_mode == "clean_v1":
            split_parents = [parent for parent in parent_groups["heldout"] if str(parent.get("source_split")) == heldout_name]
        if not split_parents:
            raise RuntimeError(f"No parents for heldout split {heldout_name}")
        samples, graphs, dialogues, validations = generate_split(
            split_parents,
            args.heldout_per_split,
            gen_args,
            f"{spec.round_id}_auto_heldout_{heldout_name}",
            "heldout_eval_only",
            spec=spec,
            split="heldout",
            heldout_split=heldout_name,
            parent_source=parent_source,
        )
        heldout_samples.extend(samples)
        heldout_graphs.extend(graphs)
        heldout_dialogues.extend(dialogues)
        heldout_validations.extend(validations)

    all_samples = train_samples + dev_samples + heldout_samples
    all_graphs = train_graphs + dev_graphs + heldout_graphs
    all_dialogues = train_dialogues + dev_dialogues + heldout_dialogues
    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    write_jsonl(out_dir / "samples.jsonl", all_samples)
    write_jsonl(out_dir / "sft_train.jsonl", (v13.build_chat_record(sample) for sample in train_samples))
    write_jsonl(out_dir / "sft_eval_dev.jsonl", (v13.build_chat_record(sample) for sample in dev_samples))
    write_jsonl(out_dir / "sft_eval_heldout.jsonl", (v13.build_chat_record(sample) for sample in heldout_samples))
    write_jsonl(out_dir / "graph_updates.jsonl", all_graphs)
    write_jsonl(out_dir / "expanded_graphs.jsonl", all_graphs)
    write_jsonl(out_dir / "expanded_dialogues.jsonl", all_dialogues)

    generation_summary = {
        "version": f"{spec.round_id}_auto_graph_expansion",
        "round_spec": asdict(spec),
        "method": "Generic auto self-play graph expansion with explicit round and parent-source controls.",
        "policy_path": str(report_dir / f"challenger_policy_{spec.round_id}_auto.json"),
        "parent_mode": args.parent_mode,
        "parent_source": parent_source,
        "parent_counts": {key: len(value) for key, value in parent_groups.items()},
        "train_only": bool(args.train_only),
        "split_counts": {"train": len(train_samples), "dev": len(dev_samples), "heldout": len(heldout_samples), "total": len(all_samples)},
        "requested_size": {"train": args.n_train, "dev": args.n_dev, "heldout_per_split": args.heldout_per_split},
        "size_note": "Full requested size selected." if args.full else "Pilot size selected to control API/eval budget while preserving split structure.",
        "challenger": policy.get("challenger"),
        "fixed_solver": {"name": spec.fixed_solver_name, "model": args.solver_model, "adapter": args.solver_adapter, "modified_or_trained": False},
        "train": split_summary(train_samples),
        "dev": split_summary(dev_samples),
        "heldout": split_summary(heldout_samples),
        "all": split_summary(all_samples),
    }
    validation = validation_summary(all_samples)
    validation["by_split"] = {
        "train": validation_summary(train_samples),
        "dev": validation_summary(dev_samples),
        "heldout": validation_summary(heldout_samples),
    }
    write_json(out_dir / "generation_summary.json", generation_summary)
    write_json(out_dir / "validation_summary.json", validation)
    write_json(report_dir / "generation_summary.json", generation_summary)
    write_json(report_dir / "validation_summary.json", validation)
    return generation_summary


def run_eval(args: argparse.Namespace, sampling_k: int, output_dir: Path) -> None:
    input_paths = [Path(args.output_dir) / "sft_train.jsonl"]
    if not args.train_only:
        input_paths.extend([Path(args.output_dir) / "sft_eval_dev.jsonl", Path(args.output_dir) / "sft_eval_heldout.jsonl"])
    cmd = [
        sys.executable,
        str(SCRIPTS / "eval_v13_pilot_vllm.py"),
        "--input",
        *[str(path) for path in input_paths],
        "--output-dir",
        str(output_dir),
        "--model",
        str(args.solver_model),
        "--adapter-dir",
        str(args.solver_adapter),
        "--max-length",
        str(args.max_length),
        "--max-model-len",
        str(args.max_model_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--sampling-k",
        str(sampling_k),
        "--chunk-size",
        str(args.chunk_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.solver_tokenizer:
        cmd.extend(["--tokenizer", str(args.solver_tokenizer)])
    if args.eval_limit is not None:
        cmd.extend(["--limit", str(args.eval_limit)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def grouped_accuracy(results: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(key_fn(row) or "unknown")].append(row)
    output = {}
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        n = len(rows)
        correct = sum(1 for row in rows if row.get("correct") is True)
        soft = [float(row.get("soft_accuracy", 0.0) or 0.0) for row in rows]
        output[key] = {
            "n": n,
            "correct": correct,
            "wrong": n - correct,
            "first_sample_accuracy": round(correct / n, 6) if n else None,
            "mean_soft_accuracy": round(sum(soft) / n, 6) if n else None,
        }
    return output


def truncation_summary(overall: dict[str, Any]) -> dict[str, Any]:
    total = 0
    max_seen = None
    weighted_means = []
    by_eval = {}
    for item in overall.get("eval_sets", []) or []:
        stats = item.get("prompt_length_stats") if isinstance(item, dict) else {}
        if not isinstance(stats, dict):
            continue
        total += int(stats.get("truncated_count") or 0)
        if stats.get("max") is not None:
            max_seen = max(int(stats["max"]), max_seen or 0)
        n = item.get("summary", {}).get("n")
        if stats.get("mean") is not None and n:
            weighted_means.append((float(stats["mean"]), int(n)))
        by_eval[str(item.get("eval_set"))] = stats
    denom = sum(n for _, n in weighted_means)
    return {
        "max_length": overall.get("max_length"),
        "max_model_len": overall.get("max_model_len"),
        "total_truncated": total,
        "max_prompt_tokens_seen": max_seen,
        "mean_prompt_tokens": round(sum(mean * n for mean, n in weighted_means) / denom, 3) if denom else None,
        "nonzero_truncation": total > 0,
        "by_eval_set": by_eval,
    }


def summarize_k5(k5: dict[str, Any] | None) -> dict[str, Any] | None:
    if not k5:
        return None
    results = [row for row in k5.get("results", []) if isinstance(row, dict)]
    attempts = sum(int(row.get("successful_count") or 0) for row in results)
    correct = sum(int(row.get("correct_count") or 0) for row in results)
    any_correct = sum(1 for row in results if int(row.get("correct_count") or 0) > 0)
    n = len(results)
    return {
        "n": n,
        "sampling_k": k5.get("sampling_k"),
        "mean_soft_accuracy": round(correct / attempts, 6) if attempts else None,
        "any_correct_rate": round(any_correct / n, 6) if n else None,
        "by_eval_set": grouped_accuracy(results, lambda row: row.get("eval_set")),
        "by_question_type": grouped_accuracy(results, lambda row: row.get("question_type")),
    }


def build_eval_reports(args: argparse.Namespace, spec: RoundSpec, k1: dict[str, Any], k5: dict[str, Any] | None) -> dict[str, Any]:
    samples = {str(row.get("expanded_graph_id")): row for row in read_jsonl(Path(args.output_dir) / "samples.jsonl")}
    results = [row for row in k1.get("results", []) if isinstance(row, dict)]

    def sample_meta(row: dict[str, Any]) -> dict[str, Any]:
        sample = samples.get(str(row.get("record_id")), {})
        meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        return {**sample, **meta}

    split_breakdown = grouped_accuracy(results, lambda row: sample_meta(row).get("split"))
    sample_breakdown = grouped_accuracy(results, lambda row: sample_meta(row).get("heldout_split") or sample_meta(row).get("split"))
    question_type_breakdown = grouped_accuracy(results, lambda row: row.get("question_type"))
    trap_type_breakdown = grouped_accuracy(results, lambda row: row.get("trap_type"))
    depth_breakdown = grouped_accuracy(results, lambda row: row.get("state_chain_depth"))
    t_summary = truncation_summary(k1)
    summary = {
        "metric": "first_sample_deterministic_accuracy",
        "fixed_solver": spec.fixed_solver_name,
        "model": args.solver_model,
        "adapter_recorded": args.solver_adapter,
        "sampling_k_first_sample": 1,
        "overall": k1.get("summary"),
        "split_breakdown": split_breakdown,
        "sample_breakdown": sample_breakdown,
        "question_type_breakdown": question_type_breakdown,
        "trap_type_breakdown": trap_type_breakdown,
        "state_chain_depth_breakdown": depth_breakdown,
        "k5_soft_accuracy": summarize_k5(k5),
    }
    report_dir = Path(args.report_dir)
    solver_prefix = "solver_" + safe_name(spec.fixed_solver_name.split("_")[1]) if spec.fixed_solver_name.startswith("solver_") else safe_name(spec.fixed_solver_name)
    write_json(report_dir / "solver_eval_summary.json", summary)
    write_json(report_dir / f"{solver_prefix}_eval_summary.json", summary)
    write_json(report_dir / "split_breakdown.json", split_breakdown)
    write_json(report_dir / "sample_breakdown.json", sample_breakdown)
    write_json(report_dir / "question_type_breakdown.json", question_type_breakdown)
    write_json(report_dir / "trap_type_breakdown.json", trap_type_breakdown)
    write_json(report_dir / "state_chain_depth_breakdown.json", depth_breakdown)
    write_json(report_dir / "truncation_summary.json", t_summary)
    write_json(report_dir / "overall_summary.json", k1)
    return summary


def build_reward_and_frontier(args: argparse.Namespace, spec: RoundSpec, k1: dict[str, Any], k5: dict[str, Any] | None) -> dict[str, Any]:
    sample_by_id = {str(row.get("expanded_graph_id")): row for row in read_jsonl(Path(args.output_dir) / "samples.jsonl")}
    eval_by_id = {str(row.get("record_id")): row for row in k1.get("results", []) if isinstance(row, dict)}
    k5_by_id = {str(row.get("record_id")): row for row in (k5 or {}).get("results", []) if isinstance(row, dict)}
    rows = []
    frontier = []
    for rid, sample in sample_by_id.items():
        eval_row = eval_by_id.get(rid, {})
        k5_row = k5_by_id.get(rid, {})
        soft = float(k5_row.get("soft_accuracy")) if isinstance(k5_row.get("soft_accuracy"), (int, float)) else (1.0 if eval_row.get("correct") is True else 0.0)
        valid = hard_gate_ok(sample)
        label = "rejected" if not valid else ("wrong" if eval_row.get("correct") is False or soft <= 0.2 else ("hard_correct" if soft < 0.8 else "easy"))
        delta = int(sample.get("state_chain_extension_delta") or 0)
        chain_reward = 0.4 * min(delta / max(1, int(args.target_extension_delta)), 1.0)
        reward = 0.0 if not valid else ((1.2 if label == "wrong" else 0.8 if label == "hard_correct" else -0.7) + chain_reward)
        row = {
            "sample_id": sample.get("sample_id"),
            "record_id": rid,
            "split": sample.get("split"),
            "heldout_split": sample.get("heldout_split"),
            "question_type": sample.get("question_type"),
            "trap_type": sample.get("trap_type"),
            "soft_accuracy": round(soft, 6),
            "first_sample_correct": eval_row.get("correct"),
            "solver_prediction": eval_row.get("prediction"),
            "hard_gate_valid": valid,
            "reward": round(reward, 6),
            "frontier_label": label,
            "state_chain_extension_delta": delta,
            "final_state_chain_depth": sample.get("final_state_chain_depth"),
        }
        rows.append(row)
        if label in {"wrong", "hard_correct"}:
            frontier.append({"reward": row, "eval": eval_row, "sample": sample})
    frontier.sort(key=lambda item: (-float(item["reward"].get("reward") or 0.0), str(item["reward"].get("record_id"))))
    reward_report = {
        "reward_version": f"reward_v2_style_{spec.round_id}_{safe_name(spec.fixed_solver_name)}",
        "formula": "R(x)=G_valid(x)*R_soft(x); wrong and hard_correct are frontier candidates, easy is penalized.",
        "target_delta": args.target_extension_delta,
        "label_counts": dict(Counter(row["frontier_label"] for row in rows).most_common()),
        "hard_gate_summary": {"total": len(rows), "valid": sum(1 for row in rows if row["hard_gate_valid"]), "invalid": sum(1 for row in rows if not row["hard_gate_valid"])},
        "samples": rows,
    }
    frontier_summary = {
        "frontier_count": len(frontier),
        "wrong_count": sum(1 for item in frontier if item["reward"].get("frontier_label") == "wrong"),
        "hard_correct_count": sum(1 for item in frontier if item["reward"].get("frontier_label") == "hard_correct"),
        "by_split": dict(Counter(str(item["reward"].get("split")) for item in frontier).most_common()),
        "selection_policy": f"valid {spec.round_id} samples labeled wrong or hard_correct under fixed {spec.fixed_solver_name}; no next solver training performed.",
    }
    solver_prefix = "solver_" + safe_name(spec.fixed_solver_name.split("_")[1]) if spec.fixed_solver_name.startswith("solver_") else safe_name(spec.fixed_solver_name)
    report_dir = Path(args.report_dir)
    write_json(report_dir / f"reward_report_{spec.round_id}.json", reward_report)
    write_json(report_dir / "frontier_summary.json", frontier_summary)
    write_jsonl(report_dir / f"{solver_prefix}_frontier_candidates.jsonl", frontier)
    return {"reward_report": reward_report, "frontier_summary": frontier_summary}


def write_placeholder_reports(args: argparse.Namespace, spec: RoundSpec) -> None:
    report_dir = Path(args.report_dir)
    for name, payload in (
        ("solver_eval_summary.json", {"skipped": True}),
        ("split_breakdown.json", {}),
        ("sample_breakdown.json", {}),
        ("question_type_breakdown.json", {}),
        ("trap_type_breakdown.json", {}),
        ("state_chain_depth_breakdown.json", {}),
        ("truncation_summary.json", {"skipped": True}),
        (f"reward_report_{spec.round_id}.json", {"skipped": True}),
        ("frontier_summary.json", {"skipped": True}),
    ):
        write_json(report_dir / name, payload)


def write_status(args: argparse.Namespace, spec: RoundSpec, policy: dict[str, Any], generation: dict[str, Any], validation: dict[str, Any], solver_summary: dict[str, Any] | None, frontier: dict[str, Any] | None) -> None:
    report_dir = Path(args.report_dir)
    truncation = read_json(report_dir / "truncation_summary.json") if (report_dir / "truncation_summary.json").exists() else {}
    split_counts = generation.get("split_counts", {})
    solver_overall = (solver_summary or {}).get("overall", {})
    frontier_summary = (frontier or {}).get("frontier_summary", {})
    lines = [
        f"# {spec.round_id}_auto_graph_expansion Status",
        "",
        f"- round_id: `{spec.round_id}`",
        f"- parent_round_id: `{spec.parent_round_id}`",
        f"- parent_mode: `{args.parent_mode}`",
        f"- train_only: `{args.train_only}`",
        f"- fixed solver: `{spec.fixed_solver_name}`",
        "- fixed solver modified or trained in this run: no",
        f"- policy generated automatically: {'no, existing policy supplied' if args.policy else 'yes'}",
        f"- policy path: `{report_dir / f'challenger_policy_{spec.round_id}_auto.json'}`",
        f"- output dataset: `{args.output_dir}`",
        f"- generated samples: train={split_counts.get('train')}, dev={split_counts.get('dev')}, heldout={split_counts.get('heldout')}",
        f"- validation pass rate: {validation.get('validation_pass_rate')}",
        f"- option failures: {validation.get('option_validation_bad')}",
        f"- leakage failures: {validation.get('leakage_bad')}",
        f"- graph-dialogue failures: {validation.get('graph_dialogue_alignment_bad')}",
        f"- unconstrained LLM gold cases: {validation.get('unconstrained_llm_gold')}",
        f"- heldout contamination cases: {validation.get('heldout_contamination')}",
        f"- prompt truncation count: {truncation.get('total_truncated', 'not evaluated')}",
        f"- max prompt tokens: {truncation.get('max_prompt_tokens_seen', 'not evaluated')}",
        f"- first-sample accuracy: {solver_overall.get('accuracy', 'not evaluated')}",
        f"- frontier count: {frontier_summary.get('frontier_count', 'not selected')}",
        f"- wrong frontier: {frontier_summary.get('wrong_count', 'not selected')}",
        f"- hard_correct frontier: {frontier_summary.get('hard_correct_count', 'not selected')}",
    ]
    if not args.full:
        lines.append("- caveat: pilot size used unless `--full` is specified.")
    if args.skip_eval:
        lines.append("- caveat: solver evaluation was skipped.")
    if args.train_only:
        lines.append("- caveat: train-only incremental generation; dev/heldout splits intentionally omitted.")
    elif not args.run_k5:
        lines.append("- caveat: k=5 soft accuracy was not run.")
    (report_dir / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    overall_lines = [
        f"# {spec.round_id}_auto_graph_expansion Overall Summary",
        "",
        f"- policy path: `{report_dir / f'challenger_policy_{spec.round_id}_auto.json'}`",
        f"- generated dataset path: `{args.output_dir}`",
        f"- report directory: `{args.report_dir}`",
        f"- train_only: `{args.train_only}`",
        f"- generated samples: {split_counts.get('total')} (train={split_counts.get('train')}, dev={split_counts.get('dev')}, heldout={split_counts.get('heldout')})",
        f"- validation pass rate: {validation.get('validation_pass_rate')}",
        f"- fixed solver accuracy: {solver_overall.get('accuracy', 'not evaluated')}",
        f"- frontier count: {frontier_summary.get('frontier_count', 'not selected')}",
        f"- truncation count: {truncation.get('total_truncated', 'not evaluated')}",
        "- next solver training: not run",
    ]
    (report_dir / "overall_summary.md").write_text("\n".join(overall_lines) + "\n", encoding="utf-8")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def optional_project_path(value: str | Path | None) -> str:
    text = normalize(value)
    if not text or text.lower() in {"none", "null", "no", "false"}:
        return ""
    return str(project_path(text))


def reject_legacy_output_name(path: Path, label: str) -> None:
    resolved = path.resolve()
    parts = set(resolved.parts)
    if "eval_sets" in parts or "eval_reports" in parts:
        raise RuntimeError(f"{label} must not write new current_method outputs under eval_sets/eval_reports: {resolved}")
    if any(part.startswith("old_improved") or part.startswith("old_method") for part in resolved.parts):
        raise RuntimeError(f"{label} must not use legacy old_* naming: {resolved}")


def finalize_current_method_args(args: argparse.Namespace) -> argparse.Namespace:
    if "old_improved" in args.run_name or "old_method" in args.run_name:
        raise RuntimeError(f"run_name must not contain legacy naming: {args.run_name}")
    if args.full:
        args.n_train = 1800 if not args.train_only else 1000
        args.n_dev = 150
        args.heldout_per_split = 150
    args.run_root = str(project_path(args.run_root))
    run_root = Path(args.run_root)
    args.parent_dir = str(project_path(args.parent_dir))
    args.policy_source_eval_dir = str(project_path(args.policy_source_eval_dir))
    args.config = str(project_path(args.config))
    args.solver_model = str(project_path(args.solver_model))
    args.solver_adapter = optional_project_path(args.solver_adapter)
    args.base_model = str(project_path(args.base_model))
    if args.difficulty_state_path:
        args.difficulty_state_path = str(project_path(args.difficulty_state_path))
    if args.solver_tokenizer:
        args.solver_tokenizer = str(project_path(args.solver_tokenizer))
    args.output_dir = str(project_path(args.output_dir) if args.output_dir else run_root / "generation")
    args.report_dir = str(project_path(args.report_dir) if args.report_dir else run_root / "reports")
    args.policy_dir = str(run_root / "policy")
    args.train_mix_dir = str(run_root / "train_mix")
    args.models_dir = str(run_root / "models")
    args.logs_dir = str(run_root / "logs")
    args.partial_samples = str(project_path(args.partial_samples) if args.partial_samples else Path(args.output_dir) / "samples.partial.jsonl")
    args.failed_api_calls = str(project_path(args.failed_api_calls) if args.failed_api_calls else Path(args.output_dir) / "failed_api_calls.jsonl")
    for label, path in {
        "run_root": run_root,
        "output_dir": Path(args.output_dir),
        "report_dir": Path(args.report_dir),
        "policy_dir": Path(args.policy_dir),
        "train_mix_dir": Path(args.train_mix_dir),
        "models_dir": Path(args.models_dir),
        "logs_dir": Path(args.logs_dir),
    }.items():
        reject_legacy_output_name(path, label)
        path.mkdir(parents=True, exist_ok=True)
    if Path(args.partial_samples).name != "samples.partial.jsonl":
        Path(args.partial_samples).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.failed_api_calls).name != "failed_api_calls.jsonl":
        Path(args.failed_api_calls).parent.mkdir(parents=True, exist_ok=True)
    return args


def resolve_current_method_seed_path(args: argparse.Namespace) -> Path:
    parent_dir = project_path(args.parent_dir)
    if parent_dir.is_file():
        return parent_dir
    preferred = parent_dir.parent / "r2_auto_full_sample0_5_complete" / "samples.jsonl"
    if args.parent_round_id == "r2" and preferred.exists():
        return preferred
    return parent_dir / "samples.jsonl"


def _round_parent_source_dialogue_path(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    state_chain = row.get("state_chain") if isinstance(row.get("state_chain"), dict) else {}
    for value in (
        row.get("source_dialogue_path"),
        metadata.get("source_dialogue_path"),
        seed_source.get("source_dialogue_path"),
        state_chain.get("source_dialogue_path"),
    ):
        if value:
            return str(value)
    return ""


def _round_parent_source_sample_name(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    for value in (
        row.get("sample_name"),
        metadata.get("sample_name"),
        seed_source.get("sample_name"),
        row.get("frontier_from_sample_name"),
        row.get("merge_source_sample_name"),
        metadata.get("merge_source_sample_name"),
    ):
        text = normalize(value)
        match = re.search(r"\b(sample\d+)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    source_path = _round_parent_source_dialogue_path(row)
    match = re.search(r"/(sample\d+)/", source_path)
    return match.group(1).lower() if match else None


def _round_parent_has_heldout_or_dev_marker(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    state_chain = row.get("state_chain") if isinstance(row.get("state_chain"), dict) else {}
    path = _round_parent_source_dialogue_path(row).lower()
    values = [
        row.get("heldout_split"),
        metadata.get("heldout_split"),
        seed_source.get("heldout_split"),
        row.get("split"),
        metadata.get("split"),
        seed_source.get("split"),
        row.get("source_split"),
        metadata.get("source_split"),
        seed_source.get("source_split"),
        state_chain.get("source_split"),
    ]
    text_values = {normalize(value).lower() for value in values if normalize(value)}
    if text_values.intersection({"dev", "test", "heldout", "dev_sanity_eval_only", "heldout_eval_only"}):
        return True
    return bool(re.search(r"/(dev|test|heldout)/sample\d+/", path))


def _populate_round_parent_seed_provenance(row: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(row)
    metadata = prepared.get("metadata") if isinstance(prepared.get("metadata"), dict) else {}
    metadata = dict(metadata)
    existing_seed_source = prepared.get("seed_source") if isinstance(prepared.get("seed_source"), dict) else {}
    state_chain = prepared.get("state_chain") if isinstance(prepared.get("state_chain"), dict) else {}
    sample_name = (
        normalize(prepared.get("merge_source_sample_name"))
        or normalize(metadata.get("merge_source_sample_name"))
        or normalize(existing_seed_source.get("sample_name"))
        or normalize(prepared.get("frontier_from_sample_name"))
        or _round_parent_source_sample_name(prepared)
    )
    sample_match = re.search(r"\b(sample\d+)\b", sample_name or "", re.IGNORECASE)
    sample_name = sample_match.group(1).lower() if sample_match else (_round_parent_source_sample_name(prepared) or None)
    source_path = _round_parent_source_dialogue_path(prepared)
    raw_training_use = normalize(
        prepared.get("training_use")
        or metadata.get("training_use")
        or existing_seed_source.get("raw_training_use")
        or existing_seed_source.get("training_use")
    )
    split_hint = normalize(
        prepared.get("split")
        or metadata.get("split")
        or existing_seed_source.get("split")
        or prepared.get("frontier_from_split")
        or prepared.get("source_split")
        or metadata.get("source_split")
        or existing_seed_source.get("source_split")
    ).lower()
    has_forbidden_marker = _round_parent_has_heldout_or_dev_marker(prepared)
    forbidden_reasons: list[str] = []
    if not sample_name:
        forbidden_reasons.append("sample_name_unrecoverable")
    if raw_training_use and raw_training_use.lower() not in {"train", "train_candidate"}:
        forbidden_reasons.append(f"training_use={raw_training_use}")
    if not raw_training_use:
        if split_hint == "train" and existing_seed_source.get("allowed_train_seed") is True:
            raw_training_use = "train_candidate"
        else:
            forbidden_reasons.append("training_use_missing")
    if has_forbidden_marker:
        forbidden_reasons.append("heldout_or_dev_marker_present")
    allowed_train_seed = (
        not forbidden_reasons
        and raw_training_use.lower() == "train_candidate"
        and existing_seed_source.get("allowed_train_seed", True) is not False
    )
    source_split = "train" if allowed_train_seed else None

    if sample_name:
        metadata["sample_name"] = sample_name
        metadata.setdefault("source_sample_name", sample_name)
        metadata.setdefault("parent_sample_name", sample_name)
        prepared["frontier_from_sample_name"] = sample_name
        prepared["sample_name"] = sample_name
    if source_split:
        metadata["source_split"] = source_split
        metadata["split"] = source_split
        prepared["frontier_from_split"] = source_split
        prepared["source_split"] = source_split
        prepared["split"] = source_split
    if raw_training_use:
        metadata["training_use"] = raw_training_use
    if source_path:
        metadata["source_dialogue_path"] = source_path
    seed_source = {
        **copy.deepcopy(existing_seed_source),
        "split": source_split or "unknown",
        "source_split": source_split or "unknown",
        "sample_name": sample_name or "unknown",
        "raw_training_use": raw_training_use or "unknown",
        "training_use": raw_training_use or "unknown",
        "source_dialogue_path": source_path,
        "parent_source": state_chain.get("parent_source") or existing_seed_source.get("parent_source"),
        "parent_sample_id": (
            prepared.get("parent_sample_id")
            or metadata.get("parent_sample_id")
            or existing_seed_source.get("parent_sample_id")
            or prepared.get("source_id")
        ),
        "sample_id": prepared.get("sample_id") or existing_seed_source.get("sample_id"),
        "allowed_train_seed": bool(allowed_train_seed),
        "forbidden_reasons": forbidden_reasons,
        "inferred_by": "prepare_current_method_seed_path.round_dataset_parent_provenance",
    }
    prepared["seed_source"] = seed_source
    prepared["seed"] = {
        "source": seed_source,
        "sample_id": prepared.get("sample_id"),
        "parent_sample_id": seed_source.get("parent_sample_id"),
    }
    prepared["metadata"] = metadata
    return prepared


def _round_parent_is_train_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    source_path = _round_parent_source_dialogue_path(row)
    training_use = normalize(
        row.get("training_use")
        or metadata.get("training_use")
        or seed_source.get("raw_training_use")
        or seed_source.get("training_use")
    ).lower()
    split_text = normalize(
        row.get("split")
        or metadata.get("split")
        or seed_source.get("split")
        or row.get("frontier_from_split")
        or row.get("source_split")
        or metadata.get("source_split")
        or seed_source.get("source_split")
    ).lower()
    if seed_source.get("allowed_train_seed") is True and split_text == "train":
        return True
    return (
        training_use in {"train", "train_candidate"}
        or split_text == "train"
        or bool(re.search(r"/train/sample\d+/", source_path))
    )


def prepare_current_method_seed_path(args: argparse.Namespace, seed_path: Path) -> Path:
    if args.parent_mode != "round_dataset" or not args.train_only or not seed_path.name.endswith(".jsonl"):
        return seed_path
    rows = read_jsonl(seed_path)
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        if not _round_parent_is_train_row(row):
            continue
        prepared = _populate_round_parent_seed_provenance(row)
        prepared_rows.append(prepared)
    if not prepared_rows:
        return seed_path
    prepared_path = Path(args.report_dir) / "round_dataset_train_seed_rows.jsonl"
    write_jsonl(prepared_path, prepared_rows)
    return prepared_path


def parse_frontier_splits(value: str) -> set[str]:
    splits = {item.strip() for item in str(value or "").split(",") if item.strip()}
    return splits or {"train"}


def round_label(args: argparse.Namespace) -> str:
    if isinstance(args.round_number, int) and args.round_number > 0:
        return f"r{args.round_number}"
    match = re.search(r"\b(r\d+)\b", str(args.round_id))
    return match.group(1) if match else str(args.round_id)


def quality_audit_passed(report_dir: Path) -> bool:
    candidates = [
        report_dir / "quality_audit.json",
        report_dir / "current_method_r3_quality_audit.json",
        report_dir / "quality_audit_summary.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = read_json(candidate)
        except json.JSONDecodeError:
            continue
        if payload.get("pass") is True or payload.get("audit_pass") is True or payload.get("quality_audit_pass") is True:
            return True
    return False


def maybe_run_training_pipeline(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.run_training_pipeline:
        return
    report_dir = Path(args.report_dir)
    if args.require_quality_audit_before_train and not quality_audit_passed(report_dir):
        raise RuntimeError("Training blocked: no passing quality audit file found in reports/.")
    required = {
        "dev_or_heldout_frontier_count": report.get("dev_or_heldout_frontier_count") == 0,
        "frontier_count": int(report.get("frontier_count") or 0) > 0,
        "policy_validation_pass": report.get("policy_validation_pass") is True,
        "policy_fallback_used": report.get("policy_fallback_used") is False,
        "seed_source_audit_pass": report.get("seed_source_audit_pass") is True,
    }
    if not all(required.values()):
        raise RuntimeError(f"Training blocked by current_method gates: {required}")
    model_dir = Path(args.models_dir) / "qwen3_4b_lora_current_method_r3_from_r2_ctx15360"
    merged_dir = Path(args.models_dir) / "qwen3_4b_lora_current_method_r3_from_r2_ctx15360_merged"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_round_solver_training_pipeline.py"),
        "--round-id",
        "r3",
        "--round-number",
        "3",
        "--fixed-solver-name",
        str(args.fixed_solver_name),
        "--solver-name",
        "solver_current_method_r3_ctx15360_full",
        "--frontier",
        str(Path(args.output_dir) / "frontier.jsonl"),
        "--round-sft-train",
        str(Path(args.output_dir) / "sft_train.jsonl"),
        "--train-output-dir",
        str(args.train_mix_dir),
        "--report-dir",
        str(Path(args.report_dir) / "training"),
        "--base-model",
        str(args.base_model),
        "--init-adapter-dir",
        str(args.solver_adapter),
        "--adapter-output-dir",
        str(model_dir),
        "--merged-output-dir",
        str(merged_dir),
        "--max-length",
        str(args.max_length),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _native_weight_map(raw: Any, defaults: dict[str, float], allowed_keys: Iterable[str]) -> dict[str, float]:
    allowed = list(allowed_keys)
    weights: dict[str, float] = {}
    source = raw if isinstance(raw, dict) else {}
    for key in allowed:
        try:
            value = float(source.get(key)) if source.get(key) is not None else float(defaults.get(key, 0.0))
        except (TypeError, ValueError):
            value = float(defaults.get(key, 0.0))
        weights[key] = max(0.0, value)
    total = sum(weights.values())
    if total <= 0:
        weights = {key: float(defaults.get(key, 0.0)) for key in allowed}
        total = sum(weights.values())
    return {key: round(value / total, 8) for key, value in weights.items() if value > 0}


def build_native_current_method_policy(
    *,
    args: argparse.Namespace,
    policy_seed: dict[str, Any],
    policy_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    native_mode = getattr(args, "native_challenger_mode", "template")
    graph_conditioned = native_mode == "graph_conditioned_api_dialogue"
    native_version = "graph_conditioned_api_dialogue_v1" if graph_conditioned else "v0"
    created_by = "native_current_method_graph_conditioned_api_dialogue" if graph_conditioned else "native_current_method_v0"
    challenger_api_called = bool(graph_conditioned)
    source_policy = read_json(project_path(args.policy)) if args.policy else {}
    source_question_weights = (
        source_policy.get("question_type_weights")
        if isinstance(source_policy, dict) and isinstance(source_policy.get("question_type_weights"), dict)
        else {}
    )
    source_attack_weights = (
        source_policy.get("attack_skill_weights")
        if isinstance(source_policy, dict) and isinstance(source_policy.get("attack_skill_weights"), dict)
        else {}
    )
    question_weights = _native_weight_map(
        source_question_weights,
        native_generation.DEFAULT_QUESTION_TYPE_WEIGHTS,
        native_generation.SUPPORTED_QUESTION_TYPES,
    )
    attack_weights = _native_weight_map(
        source_attack_weights,
        current_policy.DEFAULT_ATTACK_SKILL_TAXONOMY,
        current_policy.DEFAULT_ATTACK_SKILL_TAXONOMY.keys(),
    )
    raw_policy = {
        "policy_id": f"{args.run_name}_native_current_method_{native_version}_policy",
        "created_by": created_by,
        "method": "current_method",
        "native_challenger_mode": native_mode,
        "parent_run": args.parent_round_id,
        "target_round": round_label(args),
        "fixed_solver": args.fixed_solver_name,
        "attack_hypotheses": (
            [
                "Graph-conditioned native mode expands a frozen current graph/event_plan with the API.",
                "Dialogue is generated from current graph anchors and then verified before QA-last construction.",
                "Final gold/evidence/options are rebuilt from current graph plus verified current alignment.",
            ]
            if graph_conditioned
            else [
                "Native v0 uses structured event plans as source of truth.",
                "LLM challenger API is bypassed; gold and evidence are generated programmatically.",
            ]
        ),
        "attack_skill_weights": attack_weights,
        "question_type_weights": question_weights,
        "difficulty_targets": {
            "target_soft_accuracy_range": [0.2, 0.6],
            "avoid_easy_threshold": 0.8,
            "avoid_too_hard_threshold": 0.0,
        },
        "anti_shortcut_rules": [
            "do not expose answer labels in question text",
            "do not place all evidence in only the first or last two turns",
            "derive ordering gold from earliest visible support turns",
        ],
        "generation_instructions": (
            [
                "use native graph_conditioned_api_dialogue challenger mode",
                "call API for graph/event_plan expansion",
                "call API for graph-conditioned dialogue expansion with internal current anchors",
                "strip anchors from solver-visible dialogue",
                "verify current graph-dialogue alignment before QA generation",
                "generate QA last from current graph and verified current alignment",
                "do not route through old_gate.generate_samples",
            ]
            if graph_conditioned
            else [
                "use native current_method event_plan generator v0",
                "do not call challenger API",
                "do not route through old_gate.generate_samples",
                "keep invalidated_statement_detection disabled",
            ]
        ),
        "expected_frontier_profile": policy_seed.get("frontier_target_profile", {}),
    }
    compiled_policy = {
        **raw_policy,
        "policy_source": "native_current_method_local_policy",
        "policy_validation_pass": True,
        "policy_fallback_used": False,
        "policy_retry_count": 0,
        "required_api_payload_keys": [],
        "hard_validation_gates": [
            "forbid_dev_heldout_seed",
            "require_gold_graph_or_state_chain_derivable",
            "require_anti_shortcut_pass",
            "require_native_contract_validation",
        ],
        "disabled_question_types": ["invalidated_statement_detection"],
        "audit_only_question_types": ["invalidated_statement_detection"],
        "generation_backend": "native_current_method",
        "native_challenger_mode": native_mode,
        "native_generator_version": native_version,
    }
    policy_meta = {
        "policy_created_by": created_by,
        "policy_source": "native_current_method_local_policy",
        "policy_retry_count": 0,
        "policy_fallback_used": False,
        "challenger_api_called": challenger_api_called,
        "native_challenger_mode": native_mode,
    }
    write_json(policy_dir / "challenger_policy.raw.json", raw_policy)
    write_json(policy_dir / "challenger_policy.compiled.json", compiled_policy)
    return compiled_policy, policy_meta


def ensure_solver_eval_placeholder_for_skip_eval(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        if isinstance(sample.get("solver_eval"), dict):
            continue
        sample["solver_eval"] = {
            "skipped": True,
            "bucket": "not_evaluated",
            "soft_accuracy": 0.0,
            "partial_credit": 0.0,
            "exact_correct_count": 0,
            "k": 0,
        }


def run_current_method(args: argparse.Namespace) -> int:
    args = finalize_current_method_args(args)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    # Persistent closed-loop difficulty controller state path (defined once, used pre- and post-round).
    if args.difficulty_state_path:
        difficulty_state_path = Path(args.difficulty_state_path)
    elif int(args.round_number or 0) >= 4:
        if "probe197" in str(args.fixed_solver_name):
            state_name = "difficulty_controller_state_r4_probe197.json"
        else:
            state_name = f"difficulty_controller_state_{safe_name(round_label(args))}_{safe_name(args.fixed_solver_name)}.json"
        difficulty_state_path = Path(args.run_root).parent / state_name
    else:
        difficulty_state_path = Path(args.run_root).parent / "difficulty_controller_state.json"
    controller_source_solver = {
        "solver_name": args.fixed_solver_name,
        "solver_model": args.solver_model,
        "solver_tokenizer": args.solver_tokenizer or args.base_model,
        "source_run": args.run_root,
        "source_eval_file": None,
    }
    seed_path = resolve_current_method_seed_path(args)
    seed_path = prepare_current_method_seed_path(args, seed_path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed source not found: {seed_path}")

    seed_report_path = Path(args.report_dir) / "seed_source_audit.json"
    seed_audit_report, seed_rows = current_seed_audit.audit_seed_file(
        seed_path,
        report_path=seed_report_path,
        run_name=args.run_name,
        round_id=round_label(args),
        parent_round_id=args.parent_round_id,
    )

    policy_dir = Path(args.policy_dir)
    policy_seed = current_policy.build_policy_seed(
        run_name=args.run_name,
        run_root=args.run_root,
        policy_dir=policy_dir,
        reports_dir=args.report_dir,
        round_id=round_label(args),
        parent_round_id=args.parent_round_id,
        fixed_solver=args.fixed_solver_name,
        seed_source=seed_path,
        seed_scope=current_seed_audit.SEED_SCOPE,
        policy_source_eval_dir=args.policy_source_eval_dir,
    )
    if args.generation_backend == "native_current_method":
        compiled_policy, policy_meta = build_native_current_method_policy(
            args=args,
            policy_seed=policy_seed,
            policy_dir=policy_dir,
        )
    else:
        raw_policy, policy_meta = current_policy.challenger_update_policy(
            policy_seed=policy_seed,
            config_path=args.config,
            raw_policy_path=policy_dir / "challenger_policy.raw.json",
            max_retries=args.max_llm_retries,
            timeout=args.llm_timeout,
            max_tokens=args.challenger_max_tokens,
            allow_policy_fallback=args.allow_policy_fallback,
        )
        compiled_policy = current_policy.validate_and_compile_policy(
            raw_policy=raw_policy,
            policy_seed=policy_seed,
            compiled_policy_path=policy_dir / "challenger_policy.compiled.json",
            max_largest_question_type_share=args.max_largest_question_type_share,
            max_largest_attack_skill_share=args.max_largest_attack_skill_share,
            policy_meta=policy_meta,
        )
        if compiled_policy.get("policy_fallback_used"):
            raise RuntimeError("current_method policy fallback was used; default flow requires challenger_api policy.")

    if args.skip_generation:
        samples = current_generation.load_samples(Path(args.output_dir) / "samples.jsonl")
        generation_summary = read_json(Path(args.output_dir) / "generation_summary.json")
    elif args.generation_backend == "native_current_method":
        native_api_args = None
        if args.native_challenger_mode == "graph_conditioned_api_dialogue":
            native_api_args = generation_args(args, compiled_policy, load_config(Path(args.config)))
            native_api_args.challenger_temperature = args.challenger_temperature
            native_api_args.challenger_max_tokens = args.challenger_max_tokens
            stage64_length_guard = bool(args.qa_operation_answer_type_diversification)
            if (int(args.round_number or 0) >= 4 or stage64_length_guard) and int(args.min_length_delta_tokens or 0) > 0:
                native_api_args.challenger_max_tokens = max(
                    int(native_api_args.challenger_max_tokens),
                    int(args.min_length_delta_tokens) + 1024,
                )
            native_api_args.graph_expansion_max_tokens = args.graph_expansion_max_tokens
            native_api_args.generation_concurrency = args.generation_concurrency
            native_api_args.max_attempts_per_sample = args.max_attempts_per_sample
            native_api_args.max_api_failures = args.max_api_failures
            native_api_args.checkpoint_every = args.checkpoint_every
            native_api_args.parent_context_tail_turns = args.parent_context_tail_turns
            native_api_args.parent_context_relevant_turns = args.parent_context_relevant_turns
            native_api_args.dialogue_api_parent_token_budget = args.dialogue_api_parent_token_budget
            native_api_args.chained_qa_dependency_targeting = bool(args.chained_qa_dependency_targeting)
            native_api_args.boundary_aware_qa_diagnostics = bool(args.boundary_aware_qa_diagnostics)
            native_api_args.boundary_aware_qa_planner = bool(args.boundary_aware_qa_planner)
            native_api_args.stage33_compat_boundary_planner = bool(args.stage33_compat_boundary_planner)
            native_api_args.qa_operation_answer_type_diversification = bool(args.qa_operation_answer_type_diversification)
            native_api_args.boundary_item_difficulty_target = args.boundary_item_difficulty_target
            native_api_args.deterministic_permission_hard_negatives = bool(args.deterministic_permission_hard_negatives)
            native_api_args.boundary_aware_qa_repair = bool(args.boundary_aware_qa_repair)
            native_api_args.boundary_aware_qa_repair_attempts = int(args.boundary_aware_qa_repair_attempts or 0)
            native_api_args.api_yield_normalizers = bool(args.api_yield_normalizers)
            native_api_args.graph_proposal_cache = bool(args.graph_proposal_cache)
            native_api_args.graph_proposal_cache_dir = args.graph_proposal_cache_dir
            native_api_args.query_conditioned_graph_expansion = bool(args.query_conditioned_graph_expansion)
            native_api_args.round_id = args.round_id
            native_api_args.run_name = args.run_name
            native_api_args.round_number = args.round_number
            native_api_args.native_challenger_mode = args.native_challenger_mode
            native_api_args.native_curriculum_mode = args.native_curriculum_mode
            native_api_args.min_length_delta_tokens = args.min_length_delta_tokens
            native_api_args.length_guard_parent_baseline_tokens = args.length_guard_parent_baseline_tokens
            native_api_args.solver_tokenizer = args.solver_tokenizer or args.base_model
            native_api_args.tokenizer_path = args.solver_tokenizer or args.base_model
            native_api_args.base_model = args.base_model
            if (int(args.round_number or 0) >= 4 or stage64_length_guard) and int(args.min_length_delta_tokens or 0) > 0:
                native_api_args.challenger_max_tokens = max(
                    int(native_api_args.challenger_max_tokens),
                    int(args.min_length_delta_tokens or 0)
                    + int(args.length_guard_parent_baseline_tokens or 0)
                    + 1536,
                )
        # Closed-loop difficulty: inject the prior round's controller plan (if any) so this
        # round generates against the adjusted cell distribution. The plan lives on the policy.
        if args.closed_loop_difficulty and args.native_curriculum_mode == "r3_long_easy_v2":
            injected = False
            if difficulty_state_path.exists():
                try:
                    prior_state = json.loads(difficulty_state_path.read_text(encoding="utf-8"))
                    prior_plan = prior_state.get("difficulty_plan") if isinstance(prior_state, dict) else None
                    if isinstance(prior_plan, dict) and prior_plan:
                        compiled_policy = dict(compiled_policy)
                        compiled_policy["difficulty_plan"] = prior_plan
                        compiled_policy["qtype_curriculum"] = prior_plan.get("qtype_curriculum")
                        compiled_policy["question_type_shares"] = prior_plan.get("question_type_shares")
                        compiled_policy["allowed_question_types"] = prior_plan.get("allowed_question_types")
                        compiled_policy["single_choice_subtype_shares"] = prior_plan.get("single_choice_subtype_shares")
                        compiled_policy["single_choice_subtype_hardness_mix"] = prior_plan.get("single_choice_subtype_hardness_mix")
                        compiled_policy["hard_single_choice_template_shares"] = prior_plan.get("hard_single_choice_template_shares")
                        compiled_policy["difficulty_plan_stage"] = (
                            prior_plan.get("qtype_curriculum") or {}
                        ).get("stage")
                        compiled_policy["difficulty_plan_manual_policy_correction"] = (
                            prior_plan.get("qtype_curriculum") or {}
                        ).get("manual_policy_correction")
                        injected = True
                        hardness_mix = prior_plan.get("cell_hardness_mix")
                        logging.info(
                            "closed_loop_difficulty: injected prior difficulty_plan from %s (shares=%s, hardness_mix=%s)",
                            difficulty_state_path, prior_plan.get("question_type_shares"),
                            "present" if hardness_mix else "ABSENT",
                        )
                except (json.JSONDecodeError, OSError) as exc:
                    logging.warning("closed_loop_difficulty: could not load prior state (%s); proceeding with defaults", exc)
            if not injected:
                # Loud: closed-loop is ON but generation will fall back to DEFAULT h0 distribution.
                # This silently wasted a whole gate (phase16): all samples came out h0 because the
                # state path had no difficulty_plan at generation time. Warn unmistakably.
                logging.warning(
                    "closed_loop_difficulty: NO difficulty_plan injected (state_path=%s exists=%s) -> "
                    "generation will use DEFAULT hardness (all h0). The bandit/hardness plan is NOT in effect. "
                    "Point --difficulty-state-path at a state file that already contains a difficulty_plan.",
                    difficulty_state_path, difficulty_state_path.exists(),
                    )
            compiled_policy = dict(compiled_policy)
            compiled_policy["controller_state_path"] = str(difficulty_state_path)
            compiled_policy["controller_source_solver"] = controller_source_solver
            compiled_policy["length_guard"] = {
                "length_guard_enabled": bool(int(args.round_number or 0) >= 4 and int(args.min_length_delta_tokens or 0) > 0),
                "round_number": int(args.round_number or 0),
                "min_length_delta_tokens": int(args.min_length_delta_tokens or 0),
                "length_guard_parent_baseline_tokens": int(args.length_guard_parent_baseline_tokens or 0),
                "length_guard_baseline_mode": (
                    "max_actual_parent_and_configured_baseline"
                    if int(args.length_guard_parent_baseline_tokens or 0) > 0
                    else "actual_parent_dialogue_tokens"
                ),
            }
            if isinstance(compiled_policy.get("difficulty_plan"), dict):
                plan = compiled_policy["difficulty_plan"]
                compiled_policy["qtype_curriculum"] = plan.get("qtype_curriculum")
                compiled_policy["question_type_shares"] = plan.get("question_type_shares")
                compiled_policy["allowed_question_types"] = plan.get("allowed_question_types")
                compiled_policy["single_choice_subtype_shares"] = plan.get("single_choice_subtype_shares")
                compiled_policy["single_choice_subtype_hardness_mix"] = plan.get("single_choice_subtype_hardness_mix")
                compiled_policy["hard_single_choice_template_shares"] = plan.get("hard_single_choice_template_shares")
                compiled_policy["difficulty_plan_stage"] = (plan.get("qtype_curriculum") or {}).get("stage")
                compiled_policy["difficulty_plan_manual_policy_correction"] = (plan.get("qtype_curriculum") or {}).get("manual_policy_correction")
                compiled_policy["controller_patch_v2_enabled"] = bool(args.controller_patch_v2)
                if args.controller_patch_v2 and not (plan.get("qtype_curriculum") or {}).get("controller_patch_v2"):
                    raise RuntimeError(
                        "--controller-patch-v2 was enabled, but the injected difficulty_plan does not "
                        "declare qtype_curriculum.controller_patch_v2=true"
                    )
            write_json(policy_dir / "challenger_policy.compiled.json", compiled_policy)
        samples = native_generation.generate_native_samples(
            seed_rows=seed_rows,
            policy=compiled_policy,
            n_samples=int(args.n_train),
            output_dir=args.output_dir,
            rng_seed=int(args.seed),
            native_challenger_mode=args.native_challenger_mode,
            api_args=native_api_args,
            round_id=args.round_id,
            partial_samples=args.partial_samples,
            no_resume=args.no_resume,
        )
        generation_summary = read_json(Path(args.output_dir) / "generation_summary.json")
    else:
        samples, generation_summary = current_generation.generate_samples(
            seed_rows=seed_rows,
            args=args,
            compiled_policy=compiled_policy,
            output_dir=args.output_dir,
            partial_samples=args.partial_samples,
            failed_api_calls=args.failed_api_calls,
            seed_source=seed_path,
        )

    solver_summary = current_reports.run_solver_evaluations(
        samples=samples,
        args=args,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    if args.skip_eval:
        ensure_solver_eval_placeholder_for_skip_eval(samples)
    frontier, frontier_summary, failure_matrix = current_frontier.select_frontier(
        samples=samples,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        frontier_splits=parse_frontier_splits(args.frontier_splits),
        forbid_frontier_from_dev_heldout=args.forbid_frontier_from_dev_heldout,
        max_question_type_share=args.max_largest_question_type_share,
        max_attack_skill_share=args.max_largest_attack_skill_share,
    )
    current_generation.write_jsonl(Path(args.output_dir) / "samples.jsonl", samples)
    current_generation.write_jsonl(Path(args.output_dir) / "valid_samples.jsonl", [sample for sample in samples if current_reports.base.base_valid(sample)])

    report = current_reports.write_reports(
        args=args,
        run_root=args.run_root,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        samples=samples,
        seed_audit_report=seed_audit_report,
        generation_summary=generation_summary,
        solver_summary=solver_summary,
        frontier_summary=frontier_summary,
        failure_matrix=failure_matrix,
        compiled_policy=compiled_policy,
        policy_meta=policy_meta,
    )
    maybe_run_training_pipeline(args, report)

    # Closed-loop difficulty: after the round, run a CLEAN per-cell eval (sampler forced off)
    # and update the persistent controller state so the NEXT round's plan reflects the current
    # solver's true per-cell mastery. This is the round-to-round closed loop.
    if (
        args.closed_loop_difficulty
        and args.native_curriculum_mode == "r3_long_easy_v2"
        and not args.skip_eval
        and not bool(getattr(args, "skip_closed_loop_controller_update", False))
    ):
        try:
            samples_path = Path(args.output_dir) / "samples.jsonl"
            frontier_categories = {
                str(s.get("sample_id")): str(s.get("frontier_label", {}).get("category"))
                for s in samples
                if isinstance(s, dict)
            }
            is_smoke_controller_probe = (
                (int(args.n_train or 0) <= 20 or "smoke" in str(args.run_name))
                and not bool(getattr(args, "disable_smoke_controller_local_state", False))
            )
            controller_update_state_path = (
                Path(args.run_root) / "difficulty_controller_state.smoke.json"
                if is_smoke_controller_probe
                else difficulty_state_path
            )
            controller_out = current_difficulty.run_controller_update(
                samples_path=samples_path,
                state_path=controller_update_state_path,
                work_dir=Path(args.run_root) / "difficulty_controller",
                venv_python=sys.executable,
                frontier_categories=frontier_categories,
                k_eval=5,
                base_model=args.base_model,
                solver_name=args.fixed_solver_name,
                solver_model=args.solver_model,
                solver_tokenizer=args.solver_tokenizer or args.base_model,
                source_run=args.run_root,
            )
            logging.info(
                "closed_loop_difficulty: updated %s state %s; next-round shares=%s",
                "smoke-local" if is_smoke_controller_probe else "persistent",
                controller_update_state_path,
                controller_out["difficulty_plan"]["question_type_shares"],
            )
            try:
                plan_path = Path(args.run_root) / "difficulty_controller" / "difficulty_plan.json"
                state_path = Path(controller_update_state_path)
                plan_payload = read_json(plan_path) if plan_path.exists() else dict(controller_out.get("difficulty_plan") or {})
                plan_payload["recommendation"] = report.get("recommendation")
                plan_payload["recommendation_source"] = "current_method_report_builder"
                write_json(plan_path, plan_payload)
                if state_path.exists():
                    state_payload = read_json(state_path)
                    if isinstance(state_payload.get("difficulty_plan"), dict):
                        state_payload["difficulty_plan"]["recommendation"] = report.get("recommendation")
                        state_payload["difficulty_plan"]["recommendation_source"] = "current_method_report_builder"
                    write_json(state_path, state_payload)
            except Exception as rec_exc:
                logging.warning("closed_loop_difficulty: could not persist recommendation: %s", rec_exc)
        except Exception as exc:  # never let the controller crash a completed round
            logging.warning("closed_loop_difficulty: controller update failed: %s", exc)

    print((Path(args.report_dir) / "report.md").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    args = parse_args()
    return run_current_method(args)


if __name__ == "__main__":
    raise SystemExit(main())
