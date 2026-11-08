"""Build an isolated semantic-audit input fixture from a B subbatch manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest_row_not_object:{path}")
            rows.append(dict(value))
    return rows


def build(*, b_run: Path, input_run: Path) -> dict[str, Any]:
    b_run, input_run = b_run.resolve(), input_run.resolve()
    if input_run.exists():
        raise RuntimeError(f"input_run_already_exists:{input_run}")
    rows = _load_jsonl(b_run / "sample_manifest.jsonl")
    if len(rows) != 30:
        raise ValueError(f"expected_30_rows:{len(rows)}")
    candidates: list[dict[str, Any]] = []
    root_dialogues: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate_candidate:{candidate_id}")
        seen.add(candidate_id)
        pool = _load(Path(str(row["lineage"]["candidate_pool_path"])))
        matches = [item for item in pool if str(item.get("candidate_id")) == candidate_id]
        if len(matches) != 1:
            raise ValueError(f"candidate_lineage_not_unique:{candidate_id}:{len(matches)}")
        candidates.append(dict(matches[0]))
        root_id = str(row["root_id"])
        dialogue_path = Path(str(row["lineage"]["dialogue_path"]))
        dialogue = _load(dialogue_path)
        if root_id in root_dialogues and root_dialogues[root_id] != dialogue:
            raise ValueError(f"root_dialogue_mismatch:{root_id}")
        root_dialogues[root_id] = dialogue
    input_run.mkdir(parents=True, exist_ok=False)
    for root_id, dialogue in sorted(root_dialogues.items()):
        root_dir = input_run / "roots" / root_id
        root_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(root_dir / "expanded_dialogue_clean.json", dialogue)
    atomic_json(input_run / "hard_candidates.json", {"candidates": candidates})
    atomic_json(input_run / "invalid_or_ambiguous_candidates.json", {"candidates": []})
    atomic_json(input_run / "report.json", {
        "probe_version": "h4_semantic_audit_input_fixture_v1",
        "status": "completed",
        "candidate_count": len(candidates),
        "root_count": len(root_dialogues),
        "source_b_run": str(b_run),
        "semantic_admission_applied": False,
        "difficulty_filter_applied": False,
    })
    atomic_json(input_run / "final_audit.json", {
        "audit_version": "h4_semantic_audit_input_fixture_v1",
        "status": "completed",
        "candidate_count": len(candidates),
        "root_count": len(root_dialogues),
        "structural_input_only": True,
    })
    provenance = {
        "b_run": str(b_run),
        "b_sample_manifest_sha256": sha256_file(b_run / "sample_manifest.jsonl"),
        "candidate_count": len(candidates),
        "root_count": len(root_dialogues),
        "semantic_admission_applied": False,
        "difficulty_filter_applied": False,
    }
    atomic_json(input_run / "provenance.json", provenance)
    atomic_text(input_run / "summary.md", "# Isolated semantic audit input\n\n" + json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return {"status": "written", "input_run": str(input_run), **provenance}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--input-run", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(b_run=args.b_run, input_run=args.input_run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
