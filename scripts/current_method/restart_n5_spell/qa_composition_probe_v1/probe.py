"""Offline CLI for compiling frozen atomic QA pairs into Composer requests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from .composer import build_composer_request
from .contracts import AtomicQA, COMPOSITION_TYPES
from .input_adapter import build_atomic_ir, load_atomic_rows, load_evaluations
from .pair_compiler import compile_pairs


OUTPUT_FILES = (
    "atomic_qa_normalized.jsonl",
    "pair_candidates.jsonl",
    "pair_rejections.jsonl",
    "composer_requests.jsonl",
    "static_audit.jsonl",
    "report.md",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _prepare_run_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_run_root:{path}")
    path.mkdir(parents=True, exist_ok=True)


def _request_preflight(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
    required_ids = payload.get("required_qa_ids")
    groups = payload.get("required_evidence_groups")
    errors: list[str] = []
    if not isinstance(required_ids, list) or len(required_ids) != 2 or len(set(required_ids)) != 2:
        errors.append("exactly_two_required_qa_ids")
    if not isinstance(groups, list) or len(groups) != 2:
        errors.append("exactly_two_required_evidence_groups")
        groups = []
    refs = [set(str(ref) for ref in group.get("turn_refs") or [])
            for group in groups if isinstance(group, Mapping)]
    if len(refs) != 2 or not all(refs):
        errors.append("nonempty_evidence_groups")
    elif refs[0] == refs[1]:
        errors.append("evidence_sets_differ")
    return {
        "pair_id": request.get("pair_id"),
        "audit_stage": "composer_request_preflight",
        "status": "ready_for_future_composer" if not errors else "blocked",
        "valid": not errors,
        "errors": errors,
        "composer_response_present": False,
        "response_static_audit": "pending_composer_response",
        "future_solver_gates": {
            "no_context": "not_run",
            "remove_E1": "not_run",
            "remove_E2": "not_run",
            "full_context": "not_run",
        },
    }


def _atomic_rejections(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("record_type") == "ineligible":
            rows.append({
                "record_type": "atomic_ineligible",
                "qa_id": record.get("qa_id"),
                "reason": ",".join(record.get("ineligibility_reasons") or ["other"]),
                "grounding_decision": record.get("grounding_decision"),
            })
    return rows


def _relation_expectation(candidate_type: str) -> str:
    return {
        "state_change": "compare the earlier and later values along the shared state dimension",
        "proposal_vs_outcome": "contrast an earlier proposal with the later adopted outcome",
        "actor_outcome_binding": "bind the actor's stated preference to the matching final outcome",
        "uncertain": "no reliable closed relation was inferred; keep out of Composer",
    }[candidate_type]


def _render_report(
    *, source_batch: Path, transcript_path: Path, evaluation_path: Path,
    records: list[dict[str, Any]], atomic_qas: list[AtomicQA],
    candidates: list[dict[str, Any]], pair_rejections: list[dict[str, Any]],
    composer_requests: list[dict[str, Any]],
) -> str:
    atomic_by_id = {qa.qa_id: qa for qa in atomic_qas}
    type_counts = Counter(row["candidate_type"] for row in candidates)
    atomic_reasons = Counter()
    grounding_decisions = Counter()
    for record in records:
        grounding_decisions[str(record.get("grounding_decision") or "unknown")] += 1
        for reason in record.get("ineligibility_reasons") or []:
            atomic_reasons[str(reason)] += 1
    pair_reasons = Counter(row["reason"] for row in pair_rejections)
    ranked = sorted(
        candidates,
        key=lambda row: (
            not bool(row.get("composer_ready")),
            -(row.get("evidence_distance") or 0),
            str(row["pair_id"]),
        ),
    )[:10]
    lines = [
        "# QA Composition Probe v1 Report",
        "",
        "This is an offline QA-level composition probe. No Writer, Composer,",
        "Distractor, Solver, or Memorax API was called.",
        "",
        "## Frozen Inputs",
        "",
        f"- Atomic batch: `{source_batch}`",
        f"- Transcript: `{transcript_path}`",
        f"- Evaluation reports: `{evaluation_path}`",
        "- Atomic schema: materialized `simple_append` MCQ rows using",
        "  `reference_answer_letter`, `canonical_answer_text`, and",
        "  canonical `evidence_turn_ids`.",
        "- Eligibility policy: the materialized row must be structurally complete,",
        "  its frozen no-context evaluation must be `pass_grounding`, and its",
        "  frozen full-context K=5 result must be 5/5 correct.",
        "",
        "## Counts",
        "",
        f"- Atomic QA total: {len(records)}",
        f"- Atomic QA eligible: {len(atomic_qas)}",
        f"- Atomic QA ineligible: {len(records) - len(atomic_qas)}",
        f"- Exactly-two pair universe: {len(atomic_qas) * (len(atomic_qas) - 1) // 2}",
        f"- Candidate pairs total: {len(candidates)}",
        f"- Pair rejections total: {len(pair_rejections)}",
        f"- `state_change`: {type_counts['state_change']}",
        f"- `proposal_vs_outcome`: {type_counts['proposal_vs_outcome']}",
        f"- `actor_outcome_binding`: {type_counts['actor_outcome_binding']}",
        f"- `uncertain`: {type_counts['uncertain']}",
        f"- Final pairs ready for future Composer: {len(composer_requests)}",
        "",
        "## Rejections",
        "",
        "Atomic ineligibility:",
        "",
    ]
    if atomic_reasons:
        lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(atomic_reasons.items()))
    else:
        lines.append("- None")
    lines.extend(["", "Frozen grounding decisions:", ""])
    lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(grounding_decisions.items()))
    lines.extend(["", "Pair rejection reasons:", ""])
    if pair_reasons:
        lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(pair_reasons.items()))
    else:
        lines.append("- None")
    lines.extend(["", "## Candidate Pair Review", ""])
    if not ranked:
        lines.append("No pair passed the conservative relation prefilter.")
    for index, pair in enumerate(ranked, 1):
        qa1, qa2 = (atomic_by_id[qa_id] for qa_id in pair["qa_ids"])
        lines.extend([
            f"### {index}. {pair['pair_id']} ({pair['candidate_type']})",
            "",
            f"- QA1: {qa1.question}",
            f"- QA1 gold: {qa1.gold_text}",
            f"- QA1 evidence: {' | '.join(f'{ref}: {text}' for ref, text in zip(qa1.evidence_turn_refs, qa1.evidence_texts))}",
            f"- QA2: {qa2.question}",
            f"- QA2 gold: {qa2.gold_text}",
            f"- QA2 evidence: {' | '.join(f'{ref}: {text}' for ref, text in zip(qa2.evidence_turn_refs, qa2.evidence_texts))}",
            f"- Candidate type: `{pair['candidate_type']}`",
            f"- Expected relation: {_relation_expectation(pair['candidate_type'])}",
            f"- Composer ready: {str(bool(pair['composer_ready'])).lower()}",
            "",
        ])
    lines.extend([
        "## Conclusion",
        "",
        ("The frozen grounded pool contains at least one conservative pair for a "
         "future live Composer probe." if composer_requests else
         "The frozen grounded pool does not contain a conservative Composer-ready pair."),
        "This report does not claim that any composite question exists yet;",
        "`composer_requests.jsonl` contains requests only, not fabricated responses.",
        "",
    ])
    return "\n".join(lines)


def run_probe(
    *, atomic_batch: Path, expanded_dialogue: Path, evaluations: Path,
    run_root: Path, require_grounding: bool = True,
    require_full_context_stable: bool = True,
) -> dict[str, Any]:
    _prepare_run_root(run_root)
    rows = load_atomic_rows(atomic_batch)
    transcript = _read_json(expanded_dialogue)
    frozen_evaluations = load_evaluations(evaluations)
    atomic_qas, records = build_atomic_ir(
        rows, transcript=transcript, evaluations=frozen_evaluations,
        source_path=str(atomic_batch), require_grounding=require_grounding,
        require_full_context_stable=require_full_context_stable,
    )
    candidates, pair_rejections = compile_pairs(atomic_qas)
    atomic_by_id = {qa.qa_id: qa for qa in atomic_qas}
    composer_requests = [
        build_composer_request(
            pair,
            qa1=atomic_by_id[pair["qa_ids"][0]],
            qa2=atomic_by_id[pair["qa_ids"][1]],
        )
        for pair in candidates if pair.get("composer_ready")
    ]
    static_audits = [_request_preflight(request) for request in composer_requests]
    _write_jsonl(run_root / "atomic_qa_normalized.jsonl", records)
    _write_jsonl(run_root / "pair_candidates.jsonl", candidates)
    _write_jsonl(run_root / "pair_rejections.jsonl", [
        {"record_type": "pair_rejection", **row} for row in pair_rejections])
    _write_jsonl(run_root / "composer_requests.jsonl", composer_requests)
    _write_jsonl(run_root / "static_audit.jsonl", static_audits)
    report = _render_report(
        source_batch=atomic_batch, transcript_path=expanded_dialogue,
        evaluation_path=evaluations, records=records, atomic_qas=atomic_qas,
        candidates=candidates, pair_rejections=pair_rejections,
        composer_requests=composer_requests,
    )
    (run_root / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "probe_version": "qa_composition_probe_v1",
        "offline_only": True,
        "require_grounding": require_grounding,
        "require_full_context_stable": require_full_context_stable,
        "inputs": {
            "atomic_batch": {"path": str(atomic_batch), "sha256": _sha256(atomic_batch)},
            "expanded_dialogue": {"path": str(expanded_dialogue), "sha256": _sha256(expanded_dialogue)},
            "evaluations": str(evaluations),
        },
        "implementation": _implementation_metadata(),
        "outputs": list(OUTPUT_FILES),
        "counts": {
            "atomic_total": len(records),
            "atomic_eligible": len(atomic_qas),
            "candidate_pairs": len(candidates),
            "composer_ready_pairs": len(composer_requests),
        },
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _implementation_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[4]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repository,
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_status_available": False}
    return {
        "git_sha": sha,
        "git_status_available": True,
        "working_tree_dirty": bool(status),
        "working_tree_status_line_count": len(status),
        "probe_module": "qa_composition_probe_v1",
        "composer_request_schema_version": "qa_composition_composer_request_v1",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline exactly-two-atomic-QA composition probe")
    parser.add_argument("--atomic-batch", type=Path, required=True)
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--allow-non-grounded", action="store_true",
        help="diagnostic only: do not require frozen pass_grounding evaluations",
    )
    parser.add_argument(
        "--allow-full-context-failures", action="store_true",
        help="diagnostic only: do not require a frozen full-context 5/5 result",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_probe(
        atomic_batch=args.atomic_batch,
        expanded_dialogue=args.expanded_dialogue,
        evaluations=args.evaluations,
        run_root=args.run_root,
        require_grounding=not args.allow_non_grounded,
        require_full_context_stable=not args.allow_full_context_failures,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
