#!/usr/bin/env python3
"""Run bounded live-first canaries against an admitted Truth-first artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402
from scripts.current_method import stage3_claim_ir  # noqa: E402
from scripts.current_method import stage3_truth_first_discourse  # noqa: E402


TRUTH_FIRST_GENERATION_CONTRACT = "stage3_truth_first_generate_by_construction_v1"
REPORT_SCHEMA_VERSION = "stage3_truth_first_live_probe_report_v2"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def claim_time_key(claim: dict[str, Any]) -> tuple[int, int, str]:
    canonical = claim.get("canonical_support_claim")
    canonical = canonical if isinstance(canonical, dict) else {}
    time_value = canonical.get("time")
    time_value = time_value if isinstance(time_value, dict) else {}
    return (
        int(time_value.get("global_time_index") or -1),
        int(time_value.get("time_index") or -1),
        normalized(claim.get("claim_id")),
    )


def select_coherent_claim_groups(
    claims: Iterable[dict[str, Any]], call_count: int
) -> list[list[dict[str, Any]]]:
    """Select real same-subject/slot claims without artifact-specific IDs."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        if normalized(claim.get("claim_type")).casefold() != "node":
            continue
        canonical = claim.get("canonical_support_claim")
        if not isinstance(canonical, dict):
            continue
        key = (
            normalized(canonical.get("subject")).casefold(),
            normalized(canonical.get("predicate")).casefold(),
        )
        if all(key):
            grouped[key].append(copy.deepcopy(claim))
    candidates = sorted(
        (
            sorted(rows, key=claim_time_key)
            for rows in grouped.values()
            if len(rows) >= 3
        ),
        key=lambda rows: (-len(rows), claim_time_key(rows[0])),
    )
    if not candidates:
        raise ValueError(
            "the source artifact has no coherent group of at least three node claims"
        )
    base = candidates[0][:4]
    if len(base) == 3:
        patterns = [(0, 1, 2)]
    else:
        patterns = [
            (0, 1, 2),
            (1, 2, 3),
            (0, 1, 3),
            (0, 2, 3),
            (0, 1, 2, 3),
        ]
    return [
        [copy.deepcopy(base[index]) for index in patterns[i % len(patterns)]]
        for i in range(call_count)
    ]


def load_parent_roster(
    parent_path: Path,
    *,
    seed_sample_id: str,
    expected_parent_graph_id: str,
) -> tuple[list[str], dict[str, Any]]:
    matches: list[tuple[int, dict[str, Any]]] = []
    with parent_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            identities = {
                normalized(row.get("sample_id")),
                normalized(row.get("source_id")),
            }
            if normalized(seed_sample_id) in identities:
                matches.append((line_number, row))
    if len(matches) != 1:
        raise ValueError(
            "the locked parent file must contain exactly one row for the source artifact"
        )
    line_number, row = matches[0]
    graph = row.get("semantic_parent_graph")
    graph = graph if isinstance(graph, dict) else {}
    if normalized(graph.get("graph_id")) != normalized(expected_parent_graph_id):
        raise ValueError("source parent graph id does not match the frozen artifact")
    turns = row.get("turns")
    if not isinstance(turns, list):
        dialogue = row.get("dialogue")
        turns = dialogue.get("turns") if isinstance(dialogue, dict) else None
    if not isinstance(turns, list):
        raise ValueError("source parent row has no dialogue turns")
    frequency: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        speaker = normalized(turn.get("speaker") or turn.get("role"))
        if not speaker:
            continue
        frequency[speaker] += 1
        first_seen.setdefault(speaker, index)
    ranked = sorted(
        frequency,
        key=lambda speaker: (-frequency[speaker], first_seen[speaker], speaker),
    )
    if len(ranked) < 2 or frequency[ranked[1]] < 2:
        raise ValueError("source parent lacks two established dialogue speakers")
    roster = ranked[:2]
    return roster, {
        "source_parent_line_number": line_number,
        "source_parent_sample_id": normalized(row.get("sample_id")),
        "source_parent_graph_id": normalized(graph.get("graph_id")),
        "source_parent_turn_count": len(turns),
        "speaker_frequency": dict(
            sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
        ),
        "selected_speaker_roster": roster,
        "selection_rule": "top_two_provenance_backed_parent_speakers_by_frequency",
    }


def load_validated_frozen_quotes(
    quote_source: Path,
    selected_claims: list[dict[str, Any]],
    *,
    topic: str,
    setting: str,
) -> tuple[list[str], dict[str, Any]]:
    source = read_json(quote_source)
    claim_ids = [normalized(claim.get("claim_id")) for claim in selected_claims]
    for row in source.get("calls") or []:
        if not isinstance(row, dict) or row.get("machine_clean") is not True:
            continue
        validation = row.get("validation")
        validation = validation if isinstance(validation, dict) else {}
        quote_map = validation.get("quote_by_claim_id")
        if not isinstance(quote_map, dict) or not all(
            isinstance(quote_map.get(claim_id), str) and quote_map[claim_id]
            for claim_id in claim_ids
        ):
            continue
        quotes = [quote_map[claim_id] for claim_id in claim_ids]
        contract = (
            stage3_truth_first_discourse.build_live_content_binding_probe_contract(
                selected_claims, topic=topic, setting=setting
            )
        )
        replay = {
            "realizations": [
                {"claim_ref": f"C{index + 1}", "exact_support_quote": quote}
                for index, quote in enumerate(quotes)
            ]
        }
        report = (
            stage3_truth_first_discourse.validate_live_content_binding_probe_response(
                contract, replay, claims=selected_claims
            )
        )
        if report.get("pass") is not True:
            continue
        return quotes, {
            "source_run_id": normalized(source.get("run_id")),
            "source_canary_id": normalized(row.get("canary_id")),
            "source_prompt_sha256": normalized(row.get("prompt_sha256")),
            "source_validation_replay_pass": True,
            "claim_ids": claim_ids,
        }
    raise ValueError("quote source has no machine-clean call covering both claims")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    display_name = {
        "single_scene_live_canary": "Single-scene",
        "frozen_quote_scene_live_canary": "Frozen-quote scene",
    }.get(summary["probe_kind"], "Content-binding")
    lines = [
        f"# Stage3 Truth-first Live {display_name} Canary",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Source artifact SHA256: `{summary['source_artifact_sha256']}`",
        f"- Model: `{summary['model']}`",
        f"- Planned/executed calls: {summary['planned_call_count']}/{summary['executed_call_count']}",
        f"- Semantic-clean: {summary['semantic_clean_count']}/{summary['executed_call_count']}",
        f"- Aggregate tokens: {summary['aggregate_usage']['total_tokens']}",
        f"- Stop loss triggered: `{str(summary['stop_loss_triggered']).lower()}`",
        f"- Machine gate passed: `{str(summary['machine_gate_passed']).lower()}`",
        f"- Proceed to next layer: `{str(summary['proceed_to_next_layer']).lower()}`",
        "",
        "## Calls",
        "",
    ]
    for row in summary["calls"]:
        validation = row.get("validation") or {}
        meta = row.get("api_meta") or {}
        usage = meta.get("usage") or {}
        lines.extend(
            [
                f"### {row['canary_id']}",
                "",
                f"- Claims: {row['requested_claim_count']}",
                f"- API used: `{str(bool(meta.get('used'))).lower()}`",
                f"- Finish reason: `{normalized(meta.get('finish_reason')) or 'n/a'}`",
                f"- Tokens: {int(usage.get('total_tokens') or 0)}",
                f"- Machine clean: `{str(bool(row.get('machine_clean'))).lower()}`",
                f"- Issues: `{json.dumps(validation.get('issue_counts_by_code') or {}, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            summary["decision"],
            "",
            "Raw API responses are retained unchanged in `calls.jsonl`. No local repair or API retry was used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-kind",
        choices=("content_binding", "single_scene", "frozen_quote_scene"),
        default="content_binding",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expect-artifact-sha256", required=True)
    parser.add_argument("--parent-file", type=Path)
    parser.add_argument("--expect-parent-sha256")
    parser.add_argument("--quote-source", type=Path)
    parser.add_argument("--expect-quote-source-sha256")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--model-override")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.probe_kind == "content_binding" and args.calls != 5:
        raise ValueError("the content-binding canary is fixed to exactly five calls")
    if args.probe_kind in {"single_scene", "frozen_quote_scene"} and args.calls != 1:
        raise ValueError("the first scene canary is fixed to exactly one call")
    if args.output.exists():
        raise FileExistsError(
            f"fresh live probe output already exists; refusing resume: {args.output}"
        )
    artifact_sha256 = file_sha256(args.artifact)
    if artifact_sha256 != normalized(args.expect_artifact_sha256).casefold():
        raise ValueError(
            "source artifact SHA256 differs from the explicitly locked value"
        )
    artifact = read_json(args.artifact)
    if artifact.get("generation_contract") != TRUTH_FIRST_GENERATION_CONTRACT:
        raise ValueError("source artifact is not an admitted Truth-first artifact")
    plan = artifact.get("current_event_plan")
    materialized = (
        plan.get("_truth_first_materialized") if isinstance(plan, dict) else None
    )
    if not isinstance(materialized, dict) or not isinstance(
        materialized.get("graph"), dict
    ):
        raise ValueError("source artifact lacks its frozen materialized graph")
    if not isinstance(materialized.get("binding_validation"), dict) or (
        materialized["binding_validation"].get("pass") is not True
    ):
        raise ValueError("source artifact binding validation did not pass")

    claim_bundle = stage3_claim_ir.compile_claim_ir(materialized["graph"])
    claim_groups = select_coherent_claim_groups(
        claim_bundle.get("claims") or [], args.calls
    )
    if args.probe_kind in {"single_scene", "frozen_quote_scene"}:
        claim_groups = [group[:2] for group in claim_groups]
    discourse = materialized.get("discourse_plan")
    discourse = discourse if isinstance(discourse, dict) else {}
    speaker_roster: list[str] = []
    parent_audit: dict[str, Any] = {}
    parent_sha256: str | None = None
    if args.probe_kind in {"single_scene", "frozen_quote_scene"}:
        if args.parent_file is None or not normalized(args.expect_parent_sha256):
            raise ValueError(
                "single_scene requires --parent-file and --expect-parent-sha256"
            )
        parent_sha256 = file_sha256(args.parent_file)
        if parent_sha256 != normalized(args.expect_parent_sha256).casefold():
            raise ValueError("source parent SHA256 differs from the locked value")
        parent_graph = artifact.get("parent_graph")
        parent_graph = parent_graph if isinstance(parent_graph, dict) else {}
        speaker_roster, parent_audit = load_parent_roster(
            args.parent_file,
            seed_sample_id=normalized(artifact.get("seed_sample_id")),
            expected_parent_graph_id=normalized(parent_graph.get("graph_id")),
        )
    frozen_quotes: list[str] = []
    quote_source_audit: dict[str, Any] = {}
    quote_source_sha256: str | None = None
    if args.probe_kind == "frozen_quote_scene":
        if args.quote_source is None or not normalized(
            args.expect_quote_source_sha256
        ):
            raise ValueError(
                "frozen_quote_scene requires --quote-source and its locked SHA256"
            )
        quote_source_sha256 = file_sha256(args.quote_source)
        if quote_source_sha256 != normalized(
            args.expect_quote_source_sha256
        ).casefold():
            raise ValueError("quote source SHA256 differs from the locked value")
        frozen_quotes, quote_source_audit = load_validated_frozen_quotes(
            args.quote_source,
            claim_groups[0],
            topic=normalized(discourse.get("topic")),
            setting=normalized(discourse.get("setting")),
        )

    config = read_json(args.config)
    api = config.get("api") if isinstance(config.get("api"), dict) else {}
    api_key = api.get("api_key")
    base_url = api.get("base_url")
    model = normalized(args.model_override) or api.get("model")
    if not all((api_key, base_url, model)):
        raise ValueError("config api_key/base_url/model must all be present")
    api_args = SimpleNamespace(
        challenger_api_key=str(api_key),
        challenger_base_url=str(base_url),
        challenger_model=str(model),
        challenger_temperature=float(args.temperature),
        challenger_max_tokens=int(args.max_tokens),
        max_llm_retries=0,
        llm_timeout=int(args.timeout),
        api_yield_normalizers=False,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    calls_path = args.output / "calls.jsonl"
    call_rows: list[dict[str, Any]] = []
    issue_frequency: Counter[str] = Counter()
    issue_call_frequency: Counter[str] = Counter()
    aggregate_usage = Counter()
    consecutive_transport_failures = 0
    stop_loss_reason = ""

    for call_index, selected_claims in enumerate(claim_groups, start=1):
        if args.probe_kind == "frozen_quote_scene":
            row = native.call_api_truth_first_frozen_quote_scene_probe(
                selected_claims,
                frozen_quotes,
                api_args,
                topic=normalized(discourse.get("topic")),
                setting=normalized(discourse.get("setting")),
                speaker_roster=speaker_roster,
                canary_id=f"FQ{call_index:02d}",
                max_tokens=int(args.max_tokens),
            )
        elif args.probe_kind == "single_scene":
            row = native.call_api_truth_first_single_scene_probe(
                selected_claims,
                api_args,
                topic=normalized(discourse.get("topic")),
                setting=normalized(discourse.get("setting")),
                speaker_roster=speaker_roster,
                canary_id=f"SC{call_index:02d}",
                max_tokens=int(args.max_tokens),
            )
        else:
            row = native.call_api_truth_first_content_binding_probe(
                selected_claims,
                api_args,
                topic=normalized(discourse.get("topic")),
                setting=normalized(discourse.get("setting")),
                canary_id=f"CB{call_index:02d}",
                max_tokens=int(args.max_tokens),
            )
        row.update(
            {
                "run_id": args.run_id,
                "call_index": call_index,
                "source_artifact_sha256": artifact_sha256,
                "operation": normalized(materialized["graph"].get("operation")),
                "no_resume": True,
                "manual_naturalness_pass": None,
            }
        )
        call_rows.append(row)
        append_jsonl(calls_path, row)
        meta = row.get("api_meta") or {}
        usage = meta.get("usage") or {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            aggregate_usage[field] += int(usage.get(field) or 0)
        validation = row.get("validation") or {}
        issue_counts = validation.get("issue_counts_by_code") or {}
        issue_frequency.update(issue_counts)
        issue_call_frequency.update(set(issue_counts))
        if meta.get("used") is True:
            consecutive_transport_failures = 0
        else:
            consecutive_transport_failures += 1
        if consecutive_transport_failures >= 2:
            stop_loss_reason = "two_consecutive_transport_or_response_contract_failures"
            break
        repeated_severe = sorted(
            code
            for code, count in issue_call_frequency.items()
            if count >= 3
            and code
            not in {
                "live_binding_near_duplicate_quote",
                "live_binding_exact_duplicate_quote",
            }
        )
        if repeated_severe:
            stop_loss_reason = "same_semantic_failure_in_three_calls:" + ",".join(
                repeated_severe
            )
            break

    clean_count = sum(bool(row.get("machine_clean")) for row in call_rows)
    required_clean = int(math.ceil(args.calls * 0.8))
    machine_gate_passed = (
        len(call_rows) == args.calls
        and clean_count >= required_clean
        and not stop_loss_reason
    )
    if args.probe_kind == "frozen_quote_scene":
        proceed = False
        decision = (
            "The frozen-quote scene passed the minimal machine gate. Inspect the raw conversation manually before expanding to five calls."
            if machine_gate_passed
            else "The frozen-quote scene failed. Inspect the raw conversation and stop this composition route."
        )
        probe_kind = "frozen_quote_scene_live_canary"
    elif args.probe_kind == "single_scene":
        proceed = False
        decision = (
            "The first single scene passed the minimal machine gate. Inspect the raw response manually before any additional scene call."
            if machine_gate_passed
            else "The first single scene failed the minimal machine gate. Inspect this raw response and do not run another scene yet."
        )
        probe_kind = "single_scene_live_canary"
    else:
        proceed = machine_gate_passed
        decision = (
            "The content-binding live gate passed. The next permitted layer is one fresh single-scene live call."
            if proceed
            else "The content-binding live gate failed or hit its stop loss. Do not run a scene or complete n=1 probe."
        )
        probe_kind = "content_binding_live_canary"
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "probe_kind": probe_kind,
        "source_artifact_path": str(args.artifact.resolve()),
        "source_artifact_sha256": artifact_sha256,
        "source_generation_contract": artifact.get("generation_contract"),
        "source_parent_path": (
            str(args.parent_file.resolve()) if args.parent_file else None
        ),
        "source_parent_sha256": parent_sha256,
        "parent_roster_audit": parent_audit,
        "quote_source_path": (
            str(args.quote_source.resolve()) if args.quote_source else None
        ),
        "quote_source_sha256": quote_source_sha256,
        "quote_source_audit": quote_source_audit,
        "operation": normalized(materialized["graph"].get("operation")),
        "model": str(model),
        "base_url": str(base_url),
        "temperature": float(args.temperature),
        "max_completion_tokens_per_call": int(args.max_tokens),
        "no_resume": True,
        "api_retries_per_call": 0,
        "local_repair_allowed": False,
        "planned_call_count": int(args.calls),
        "executed_call_count": len(call_rows),
        "semantic_clean_count": clean_count,
        "required_clean_count": required_clean,
        "clean_rate": clean_count / len(call_rows) if call_rows else 0.0,
        "aggregate_usage": {
            field: int(aggregate_usage[field])
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "issue_frequency": dict(sorted(issue_frequency.items())),
        "issue_call_frequency": dict(sorted(issue_call_frequency.items())),
        "stop_loss_triggered": bool(stop_loss_reason),
        "stop_loss_reason": stop_loss_reason or None,
        "machine_gate_passed": machine_gate_passed,
        "manual_review_required": args.probe_kind
        in {"single_scene", "frozen_quote_scene"},
        "proceed_to_next_layer": proceed,
        "decision": decision,
        "calls": call_rows,
    }
    write_json(args.output / "summary.json", summary)
    write_markdown(args.output / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "executed": len(call_rows),
                "clean": clean_count,
                "stop_loss": stop_loss_reason or None,
                "machine_gate_passed": machine_gate_passed,
                "proceed": proceed,
                "total_tokens": int(aggregate_usage["total_tokens"]),
                "report": str((args.output / "REPORT.md").resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if machine_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
