"""Create a compact, read-only manual-review bundle from a completed A/B run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_candidate(run_root: Path, candidate_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample = None
    for line in (run_root / "sample_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("candidate_id") == candidate_id:
            sample = json.loads(line)
            break
    if sample is None:
        raise SystemExit(f"candidate_not_in_sample_manifest:{candidate_id}")
    lineage = sample["lineage"]
    candidates = load(Path(lineage["candidate_pool_path"]))
    candidate = next((row for row in candidates if row.get("candidate_id") == candidate_id), None)
    if candidate is None:
        raise SystemExit(f"candidate_not_in_pool:{candidate_id}")
    dialogue = load(Path(lineage["dialogue_path"]))
    return sample, candidate, dialogue


def find_permutations(run_root: Path, candidate_id: str) -> dict[str, dict[str, Any]]:
    result = {}
    for line in (run_root / "permutation_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("candidate_id") == candidate_id:
                result[str(row["permutation_id"])] = row
    return result


def find_rows(run_root: Path, candidate_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_root / "results").glob("*.json")):
        row = load(path)
        if row.get("candidate_id") == candidate_id:
            rows.append(row)
    return rows


def canonical_prediction(row: dict[str, Any], permutation: dict[str, Any]) -> list[str]:
    parsed = [str(value) for value in row.get("parsed_answer") or []]
    if row.get("output_mode") == "label":
        mapping = permutation["display_to_canonical_mapping"]
        return [str(mapping[value]) for value in parsed if value in mapping]
    return [str(value) for value in (row.get("parse") or {}).get("parsed_canonical") or parsed]


def display_options(candidate: dict[str, Any], permutation: dict[str, Any]) -> dict[str, str]:
    values = candidate.get("options") if candidate.get("question_type") != "ordering" else candidate.get("items")
    return {
        label: str(values[canonical])
        for label, canonical in permutation["display_to_canonical_mapping"].items()
    }


def compact_row(row: dict[str, Any], permutation: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    prediction = canonical_prediction(row, permutation)
    gold = [str(value) for value in row.get("canonical_gold") or []]
    return {
        "request_id": row.get("request_id"),
        "permutation_id": row.get("permutation_id"),
        "is_cyclic": row.get("is_cyclic"),
        "display_order": row.get("display_order"),
        "display_to_canonical_mapping": permutation.get("display_to_canonical_mapping"),
        "display_options": display_options(candidate, permutation),
        "context_mode": row.get("context_mode"),
        "output_mode": row.get("output_mode"),
        "seed": row.get("seed"),
        "temperature": row.get("temperature"),
        "output_reserve": row.get("output_reserve"),
        "canonical_gold": gold,
        "displayed_gold_from_manifest": permutation.get("displayed_gold"),
        "raw_output": row.get("raw_output"),
        "parsed_answer": row.get("parsed_answer"),
        "canonical_prediction_after_rebinding": prediction,
        "correct_after_rebinding": prediction == gold,
        "failure_type": row.get("failure_type"),
        "parse": row.get("parse"),
    }


def dialogue_markdown(dialogue: dict[str, Any]) -> str:
    lines = [
        "# Frozen Expanded Dialogue",
        "",
        f"- Root: `{dialogue.get('root_id')}`",
        f"- Turns: `{dialogue.get('turn_count')}`",
        f"- Dialogue SHA256: `{dialogue.get('complete_dialogue_sha256')}`",
        f"- Snapshot: `{dialogue.get('expanded_dialogue_version')}`",
        "",
    ]
    for turn in dialogue.get("turns") or []:
        lines.append(f"**[{turn.get('turn_ref')}] {turn.get('speaker')}**  ")
        lines.append(str(turn.get("text") or ""))
        lines.append("")
    return "\n".join(lines)


def review_markdown(sample: dict[str, Any], candidate: dict[str, Any], outputs: list[dict[str, Any]], out_dir: Path) -> str:
    lines = [
        "# A/B Manual Review Bundle",
        "",
        "This is an offline, read-only review copy. The frozen source run and raw result files were not modified.",
        "",
        f"- Candidate: `{candidate['candidate_id']}`",
        f"- Root: `{candidate.get('root_id')}`",
        f"- Type/source: `{candidate.get('question_type')}` / `{candidate.get('generation_source')}`",
        f"- Source dialogue: [`original_dialogue.md`]({(out_dir / 'original_dialogue.md').resolve()})",
        f"- Candidate JSON: [`candidate.json`]({(out_dir / 'candidate.json').resolve()})",
        f"- Selected A/B outputs: [`ab_outputs.json`]({(out_dir / 'ab_outputs.json').resolve()})",
        f"- Full corrected analysis: [`corrected_summary.md`]({(Path(sample['_run_root']) / 'corrected_summary.md').resolve()})",
        f"- Full raw result directory: [`results/`]({(Path(sample['_run_root']) / 'results').resolve()})",
        "",
        "## Question",
        "",
        candidate.get("question", ""),
        "",
        "### Options and submitted Gold",
        "",
    ]
    for label, text in (candidate.get("options") or {}).items():
        marker = " **<-- submitted Gold**" if label == candidate.get("correct_option") else ""
        lines.append(f"- **{label}.** {text}{marker}")
    gold_label = str(candidate.get("correct_option") or ((candidate.get("gold") or [""])[0]))
    gold_text = (candidate.get("options") or {}).get(gold_label, "")
    question = str(candidate.get("question") or "")
    match = re.search(r"about ([A-Z][a-z]+)|did ([A-Z][a-z]+)", question)
    subject = next((group for group in match.groups() if group), "") if match else ""
    lines += [
        "",
        f"Evidence refs supplied by generator: `{', '.join(candidate.get('evidence_refs') or [])}`",
        "",
        "## Selected A/B outputs",
        "",
        "`perm_000` is cyclic/identity. `perm_001` is a non-cyclic swap of C and D. Predictions are rebound to canonical option IDs before scoring.",
        "",
        "| permutation | context | mode | displayed Gold | raw model output | canonical prediction | correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in outputs:
        raw = str(row.get("raw_output") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{row['permutation_id']}` ({'cyclic' if row['is_cyclic'] else 'non-cyclic'}) | "
            f"`{row['context_mode']}` | `{row['output_mode']}` | `{','.join(row.get('displayed_gold_from_manifest') or [])}` | "
            f"`{raw}` | `{','.join(row.get('canonical_prediction_after_rebinding') or [])}` | `{row['correct_after_rebinding']}` |"
        )
    lines += [
        "",
        "For the unabridged structured rows, including display mapping, prompt seed, parser result, and output reserve, open [`ab_outputs.json`].",
        "",
        "## Review focus",
        "",
        f"1. Is the submitted Gold (`{gold_label}`, {gold_text}) semantically the only best answer?",
        f"2. Does the full dialogue support the question's wording{(' about ' + subject) if subject else ''}?",
        "3. Do the A/B outputs show a label/position effect, a content-selection effect, or ordinary reasoning errors?",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--permutation-ids", nargs="+", default=["perm_000", "perm_001"])
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    sample, candidate, dialogue = find_candidate(run_root, args.candidate_id)
    sample["_run_root"] = str(run_root)
    permutations = find_permutations(run_root, args.candidate_id)
    rows = find_rows(run_root, args.candidate_id)
    wanted = []
    for row in rows:
        # A: selected cyclic and non-cyclic permutations; B0/B1/B2: cyclic only.
        if row.get("output_mode") == "label" and row.get("permutation_id") in args.permutation_ids:
            wanted.append(row)
        elif row.get("permutation_id") == "perm_000" and row.get("output_mode") in {"content_visible", "content_hidden"}:
            wanted.append(row)
    wanted.sort(key=lambda row: (
        args.permutation_ids.index(row["permutation_id"]) if row["permutation_id"] in args.permutation_ids else 99,
        {"no_context": 0, "full_context": 1}[row["context_mode"]],
        {"label": 0, "content_visible": 1, "content_hidden": 2}[row["output_mode"]],
    ))
    compact = [compact_row(row, permutations[row["permutation_id"]], candidate) for row in wanted]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.candidate_id)
    out_dir = run_root / f"manual_review_{safe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(out_dir / "candidate.json", {
        "candidate": candidate,
        "sample_manifest_entry": sample,
        "source_candidate_path": sample["lineage"]["candidate_pool_path"],
        "source_dialogue_path": sample["lineage"]["dialogue_path"],
    })
    dump(out_dir / "ab_outputs.json", {
        "run_root": str(run_root),
        "candidate_id": args.candidate_id,
        "selection": "A perm_000 + perm_001; B0/B1/B2 perm_000; both context modes",
        "rows": compact,
    })
    (out_dir / "original_dialogue.md").write_text(dialogue_markdown(dialogue), encoding="utf-8")
    (out_dir / "review.md").write_text(review_markdown(sample, candidate, compact, out_dir), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
