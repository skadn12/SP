"""Offline CLI and report writer for relation_candidate_probe_v2."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from .contracts import Fact, RelationCandidate, Rejection, Turn
from .fact_extractor import extract_facts
from .input_adapter import load_structured_sources, load_transcript
from .paired_qa_planner import build_paired_qa_target
from .relation_miner import mine_relations


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


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _bucket(distance: int) -> str:
    if distance <= 2:
        return "1-2 turns"
    if distance <= 5:
        return "3-5 turns"
    if distance <= 10:
        return "6-10 turns"
    if distance <= 20:
        return "11-20 turns"
    return ">20 turns"


def _implementation_metadata() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], cwd=repo,
                                check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "working_tree_dirty": None}
    return {"git_sha": sha, "working_tree_dirty": bool(status),
            "working_tree_status_line_count": len(status),
            "probe_version": "relation_candidate_probe_v2"}


def _turn_counts(turns: list[Turn]) -> dict[str, int]:
    return {
        "parent_turns": sum(turn.origin == "source_grounded" for turn in turns),
        "child_turns": sum(turn.origin == "generated_extension" for turn in turns),
        "total_turns": len(turns),
    }


def _fact_rows(facts: Iterable[Fact]) -> list[dict[str, Any]]:
    return [fact.to_dict() for fact in facts]


def _relation_case(relation: RelationCandidate) -> str:
    return "same" if relation.value_relation == "same" else "changed"


def _render_fact(fact: Mapping[str, Any]) -> str:
    return (f"{fact['turn_refs'][0]} ({fact.get('speaker', '')}): "
            f"{fact['evidence_texts'][0]} -> `{fact['value']}`")


def _render_report(
    *, root_id: str, transcript_path: Path, structured_paths: list[Path],
    turns: list[Turn], proposals: list[Fact], outcomes: list[Fact],
    relations: list[RelationCandidate], rejections: list[Rejection],
    paired_targets: list[dict[str, Any]],
) -> str:
    status_counts = Counter(relation.status for relation in relations)
    all_value_counts = Counter(relation.value_relation for relation in relations)
    value_counts = Counter(relation.value_relation for relation in relations
                           if relation.status == "eligible")
    distance_counts = Counter(_bucket(relation.turn_distance) for relation in relations)
    scope_counts = Counter(relation.scope for relation in relations)
    cross_count = sum(relation.cross_parent_child for relation in relations)
    rejection_counts = Counter(rejection.reason for rejection in rejections)
    eligible = [relation for relation in relations if relation.status == "eligible"][:10]
    uncertain = [relation for relation in relations if relation.status == "uncertain"][:10]
    rejected = rejections[:10]
    lines = [
        "# Relation Candidate Probe v2",
        "",
        "Offline transcript-first mining of exactly one relation type: `proposal_outcome`.",
        "No external API, Writer, Solver, Distractor service, or Memorax call was made.",
        "",
        "## Inputs",
        "",
        f"- Root: `{root_id}`",
        f"- Transcript: `{transcript_path}`",
        "- Structured annotations (not evidence):",
    ]
    lines.extend(f"  - `{path}`" for path in structured_paths)
    lines.extend([
        "- Evidence source: only canonical transcript `turn_ref`, speaker, origin, and text.",
        "",
        "## Counts",
        "",
        f"- Roots scanned: 1",
        f"- Parent turns: {sum(turn.origin == 'source_grounded' for turn in turns)}",
        f"- Child turns: {sum(turn.origin == 'generated_extension' for turn in turns)}",
        f"- Total turns: {len(turns)}",
        f"- Proposal facts detected: {len(proposals)}",
        f"- Outcome facts detected: {len(outcomes)}",
        f"- Candidate relation pairs: {len(relations)}",
        f"- Eligible: {status_counts['eligible']}",
        f"- Uncertain: {status_counts['uncertain']}",
        f"- Rejected: {len(rejections)}",
        f"- Eligible value_relation same: {value_counts['same']}",
        f"- Eligible value_relation changed: {value_counts['changed']}",
        f"- Cross Parent/Child: {cross_count}",
        f"- All candidate value_relation same: {all_value_counts['same']}",
        f"- All candidate value_relation changed: {all_value_counts['changed']}",
        "",
        "### Distance",
        "",
    ])
    for label in ("1-2 turns", "3-5 turns", "6-10 turns", "11-20 turns", ">20 turns"):
        lines.append(f"- `{label}`: {distance_counts[label]}")
    lines.extend(["", "### Scope", ""])
    for label in ("parent_only", "child_only", "parent_child"):
        lines.append(f"- `{label}`: {scope_counts[label]}")
    lines.extend(["", "### Rejection reasons", ""])
    if rejection_counts:
        lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(rejection_counts.items()))
    else:
        lines.append("- None")
    lines.extend(["", "## Eligible Examples", ""])
    if not eligible:
        lines.append("No eligible relation candidate.")
    for relation in eligible:
        proposal, outcome = relation.proposal, relation.outcome
        target = next((item for item in paired_targets
                       if item["relation_id"] == relation.relation_id), None)
        lines.extend([
            f"### {relation.relation_id}", "",
            f"- Proposal: {_render_fact(proposal)}",
            f"- Outcome: {_render_fact(outcome)}",
            f"- event_key: `{relation.event_key}`",
            f"- same event/decision slot: exact `event_key` match (`{relation.event_key}`)",
            f"- value_relation: `{relation.value_relation}`",
            f"- turn_distance: {relation.turn_distance}",
            f"- cross_parent_child: {str(relation.cross_parent_child).lower()}",
            f"- scope: `{relation.scope}`",
            f"- confidence: `{relation.confidence}`",
            f"- reason: {relation.reason}",
            f"- source_structured_refs: {', '.join(relation.source_structured_refs) or 'none'}",
        ])
        if target:
            lines.extend([
                f"- Planned atomic target A: {target['atomic_targets'][0]['target_id']} "
                f"({target['atomic_targets'][0]['target_answer']})",
                f"- Planned atomic target B: {target['atomic_targets'][1]['target_id']} "
                f"({target['atomic_targets'][1]['target_answer']})",
                f"- Future composite intent: {target['composite_intent']}",
            ])
        lines.append("")
    lines.extend(["## Uncertain Examples", ""])
    if not uncertain:
        lines.append("No uncertain relation candidate.")
    for relation in uncertain:
        lines.extend([
            f"### {relation.relation_id}", "",
            f"- Proposal: {_render_fact(relation.proposal)}",
            f"- Outcome: {_render_fact(relation.outcome)}",
            f"- event_key: `{relation.event_key}`; distance: {relation.turn_distance}",
            f"- reason: {relation.reason}; warnings are retained as uncertain",
            "",
        ])
    lines.extend(["## Typical Rejections", ""])
    if not rejected:
        lines.append("No rejected relation pair.")
    for item in rejected:
        lines.extend([
            f"- `{item.pair_id}`: `{item.reason}` "
            f"(proposal={item.proposal_fact_id}, outcome={item.outcome_fact_id})",
        ])
    lines.extend([
        "", "## Interpretation", "",
        ("The transcript contains eligible proposal-to-outcome candidates that can "
         "support a future paired atomic QA plan." if eligible else
         "The transcript did not yield a high-confidence eligible proposal-to-outcome pair."),
        "Paired targets contain evidence refs and answer targets only; no natural-language QA was generated.",
        "",
    ])
    return "\n".join(lines)


def run_probe(
    *, transcript_path: Path, structured_paths: list[Path], run_root: Path,
) -> dict[str, Any]:
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_run_root:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    root_id, turns, _raw_transcript = load_transcript(transcript_path)
    structured = load_structured_sources(structured_paths)
    proposals, outcomes = extract_facts(turns, structured_sources=structured)
    relations, rejections = mine_relations(proposals, outcomes)
    paired_targets = [
        build_paired_qa_target(relation.to_dict())
        for relation in relations if relation.status == "eligible"
    ]
    _write_jsonl(run_root / "normalized_turns.jsonl", [turn.to_dict() for turn in turns])
    _write_jsonl(run_root / "proposal_facts.jsonl", _fact_rows(proposals))
    _write_jsonl(run_root / "outcome_facts.jsonl", _fact_rows(outcomes))
    _write_jsonl(run_root / "relation_candidates.jsonl",
                 [relation.to_dict() for relation in relations])
    _write_jsonl(run_root / "relation_rejections.jsonl",
                 [rejection.to_dict() for rejection in rejections])
    _write_jsonl(run_root / "paired_qa_targets.jsonl", paired_targets)
    report = _render_report(
        root_id=root_id, transcript_path=transcript_path,
        structured_paths=structured_paths, turns=turns, proposals=proposals,
        outcomes=outcomes, relations=relations, rejections=rejections,
        paired_targets=paired_targets,
    )
    (run_root / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "probe_version": "relation_candidate_probe_v2",
        "offline_only": True,
        "relation_type": "proposal_outcome",
        "root_id": root_id,
        "inputs": {
            "transcript": {"path": str(transcript_path), "sha256": _sha256(transcript_path)},
            "structured_sources": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in structured_paths if path.exists()
            ],
        },
        "counts": _turn_counts(turns) | {
            "proposal_facts": len(proposals), "outcome_facts": len(outcomes),
            "relation_candidates": len(relations),
            "eligible": sum(r.status == "eligible" for r in relations),
            "uncertain": sum(r.status == "uncertain" for r in relations),
            "rejected": len(rejections), "paired_qa_targets": len(paired_targets),
        },
        "implementation": _implementation_metadata(),
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline proposal-outcome relation miner")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--structured-source", type=Path, action="append", default=[])
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_probe(
        transcript_path=args.transcript,
        structured_paths=list(args.structured_source),
        run_root=args.run_root,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
