#!/usr/bin/env python3
"""Generate and audit Stage3 semantic graphs before dialogue, QA, or solver work."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402


DEFAULT_OPERATIONS = ",".join(native.STAGE64_QA_OPERATIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paid-API graph-only smoke for the Stage2 -> Stage3 semantic graph boundary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="QA-free Stage3 semantic parent JSONL (canonical parent_v3 recommended).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--round-id", default="stage3_semantic_graph_smoke")
    parser.add_argument("--operations", default=DEFAULT_OPERATIONS)
    parser.add_argument("--max-attempts-per-sample", type=int, default=2)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.70)
    parser.add_argument("--challenger-temperature", type=float, default=0.25)
    parser.add_argument("--graph-expansion-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--graph-semantic-retries",
        type=int,
        default=1,
        help="Validator-guided graph correction calls after a parseable but noncanonical response.",
    )
    parser.add_argument("--max-llm-retries", type=int, default=1)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--api-yield-normalizers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--confirm-api-calls",
        action="store_true",
        help="Required acknowledgement that this command makes paid external API calls.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_api_args(args: argparse.Namespace) -> SimpleNamespace:
    config = read_json(args.config) if args.config.exists() else {}
    api = config.get("api") if isinstance(config.get("api"), dict) else {}
    api_key = api.get("api_key") or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = api.get("base_url")
    model = api.get("model")
    if not api_key or not base_url or not model:
        raise RuntimeError("Missing API key/base_url/model in config or environment")
    return SimpleNamespace(
        challenger_api_key=api_key,
        challenger_base_url=base_url,
        challenger_model=model,
        challenger_temperature=args.challenger_temperature,
        challenger_max_tokens=args.graph_expansion_max_tokens,
        graph_expansion_max_tokens=args.graph_expansion_max_tokens,
        graph_semantic_retries=args.graph_semantic_retries,
        max_llm_retries=args.max_llm_retries,
        llm_timeout=args.llm_timeout,
        api_yield_normalizers=args.api_yield_normalizers,
        query_conditioned_graph_expansion=False,
    )


def operation_targets(operations: list[str], n: int) -> list[str]:
    if not operations:
        raise ValueError("At least one operation is required")
    unknown = sorted(set(operations) - set(native.STAGE64_QA_OPERATIONS))
    if unknown:
        raise ValueError(f"Unsupported operations: {unknown}")
    return [operations[index % len(operations)] for index in range(n)]


def policy_for(operation: str, target: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a graph policy with the complete immutable Stage64 target.

    The graph-only probe is deliberately subject to the same target contract as
    the formal runner.  Supplying only ``qa_operation`` here would silently
    disable the reasoning-subtype and hard-negative checks.
    """
    stage64_target = copy.deepcopy(target) if isinstance(target, dict) else {
        "qa_operation": operation,
        "preferred_answer_type": "Number" if operation in {"count", "comparison"} else "Event",
        "target_basis": "stage3_graph_only_smoke_fallback",
    }
    if stage64_target.get("qa_operation") != operation:
        raise ValueError(
            f"Stage64 target operation mismatch: target={stage64_target.get('qa_operation')!r}, "
            f"operation={operation!r}"
        )
    return {
        "native_curriculum_mode": native.EASY_V2_MODE,
        "qa_operation_answer_type_diversification": True,
        "semantic_graph_v2": True,
        "planned_cell": {
            "question_type": "single_choice",
            "attack_skill": "conditional_plan",
            "difficulty_subtype": "hard_single_choice",
            "distractor_hardness": "hard",
        },
        "stage64_target": stage64_target,
    }


def exception_graph_context(exc: Exception) -> tuple[Any, Any]:
    return (
        copy.deepcopy(getattr(exc, "api_graph_payload", None)),
        copy.deepcopy(getattr(exc, "api_graph_meta", None)),
    )


def run_graph_smoke(
    *,
    parents: list[dict[str, Any]],
    api_args: SimpleNamespace,
    operations: list[str],
    n: int,
    seed: int,
    round_id: str,
    max_attempts_per_sample: int,
    output_dir: Path,
    minimum_pass_rate: float,
) -> dict[str, Any]:
    if not parents:
        raise RuntimeError("Stage3 parent input is empty")
    if n < 1 or max_attempts_per_sample < 1:
        raise ValueError("--n and --max-attempts-per-sample must be positive")
    if not 0.0 <= minimum_pass_rate <= 1.0:
        raise ValueError("--minimum-pass-rate must be in [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    valid: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    targets = operation_targets(operations, n)
    required_valid_count = max(0, math.ceil(minimum_pass_rate * n))
    # Materialize one canonical target per requested operation before any API
    # call.  This keeps the smoke deterministic while exercising the complete
    # formal Stage64 reasoning identity (subtype/template/hops/hard negative).
    canonical_target_records: dict[str, dict[str, Any]] = {}
    canonical_targets: dict[str, dict[str, Any]] = {}
    for operation in sorted(set(targets)):
        canonical = native._build_stage64_canonical_targets(
            1,
            {operation: 1.0},
            None,
        )
        if len(canonical) != 1 or not isinstance(canonical[0].get("stage64_target"), dict):
            raise RuntimeError(f"Could not materialize canonical target for {operation!r}")
        canonical_target_records[operation] = copy.deepcopy(canonical[0])
        canonical_targets[operation] = copy.deepcopy(
            canonical_target_records[operation]["stage64_target"]
        )
    parent_order = list(range(len(parents)))
    random.Random(seed).shuffle(parent_order)
    attempted_target_count = 0
    early_stopped = False

    for sample_index, operation in enumerate(targets):
        remaining_including_current = n - sample_index
        if len(valid) + remaining_including_current < required_valid_count:
            early_stopped = True
            break
        attempted_target_count += 1
        accepted = False
        for attempt in range(1, max_attempts_per_sample + 1):
            parent_index = parent_order[(sample_index + attempt - 1) % len(parent_order)]
            parent = parents[parent_index]
            target_record = canonical_target_records[operation]
            frozen_target = canonical_targets[operation]
            target_audit_fields = {
                "stage64_target": copy.deepcopy(frozen_target),
                "stage64_target_identity": target_record.get("target_identity"),
                "stage64_target_id": target_record.get("target_id"),
                "stage64_target_plan_sha256": target_record.get(
                    "target_plan_sha256"
                ),
                "stage64_target_index": target_record.get("target_index"),
            }
            policy = policy_for(operation, frozen_target)
            rng = random.Random(seed + 1_000_003 * (sample_index + 1) + attempt)
            legacy = native.bootstrap_legacy_alignment(parent)
            raw_graph: Any = None
            graph_meta: Any = None
            try:
                plan, graph_meta, raw_graph = native.call_api_graph_expansion(
                    parent,
                    legacy,
                    policy,
                    api_args,
                    rng,
                    sample_index=sample_index,
                    round_id=round_id,
                )
                target = policy.get("stage64_target") if isinstance(policy, dict) else None
                issues = native.semantic_graph_plan_quality_issues(
                    plan,
                    operation,
                    target if isinstance(target, dict) else None,
                )
                if issues:
                    raise native.NativeGraphDialogueReject(
                        f"stage3 graph semantic quality failed: {issues}"
                    )
                valid.append(
                    {
                        "sample_id": f"{round_id}:S{sample_index:04d}",
                        "parent_sample_id": parent.get("sample_id") or parent.get("source_id"),
                        "target_operation": operation,
                        **copy.deepcopy(target_audit_fields),
                        "attempt": attempt,
                        "raw_graph": raw_graph,
                        "graph_meta": graph_meta,
                        "semantic_graph": plan,
                        "quality_issues": [],
                    }
                )
                accepted = True
                break
            except Exception as exc:
                attached_raw, attached_meta = exception_graph_context(exc)
                rejects.append(
                    {
                        "sample_index": sample_index,
                        "parent_sample_id": parent.get("sample_id") or parent.get("source_id"),
                        "target_operation": operation,
                        **copy.deepcopy(target_audit_fields),
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_graph": raw_graph if raw_graph is not None else attached_raw,
                        "graph_meta": graph_meta if graph_meta is not None else attached_meta,
                    }
                )
            finally:
                native.write_jsonl(output_dir / "graph_valid.jsonl", valid)
                native.write_jsonl(output_dir / "graph_rejects.jsonl", rejects)
        if not accepted:
            continue

    pass_rate = len(valid) / n
    summary = {
        "method": "stage3_semantic_graph_only_smoke",
        "schema_version": "semantic_graph_v2",
        "graph_gate_version": native.GRAPH_CACHE_SCHEMA_VERSION,
        "requested_count": n,
        "valid_count": len(valid),
        "rejected_attempt_count": len(rejects),
        "failed_target_count": n - len(valid),
        "attempted_target_count": attempted_target_count,
        "unattempted_target_count": n - attempted_target_count,
        "early_stopped": early_stopped,
        "required_valid_count": required_valid_count,
        "pass_rate": round(pass_rate, 6),
        "minimum_pass_rate": minimum_pass_rate,
        "gate_passed": pass_rate >= minimum_pass_rate,
        "target_operation_distribution": dict(Counter(targets)),
        "valid_operation_distribution": dict(Counter(row["target_operation"] for row in valid)),
        "reject_error_types": dict(Counter(row["error_type"] for row in rejects)),
        "dialogue_generated": False,
        "qa_generated": False,
        "solver_loaded": False,
    }
    native.write_json_atomic(output_dir / "graph_smoke_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if not args.confirm_api_calls:
        raise RuntimeError("Refusing paid API calls without --confirm-api-calls")
    parents = native.read_jsonl_if_exists(args.input)
    operations = [item.strip() for item in args.operations.split(",") if item.strip()]
    summary = run_graph_smoke(
        parents=parents,
        api_args=load_api_args(args),
        operations=operations,
        n=args.n,
        seed=args.seed,
        round_id=args.round_id,
        max_attempts_per_sample=args.max_attempts_per_sample,
        output_dir=args.output_dir,
        minimum_pass_rate=args.minimum_pass_rate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
