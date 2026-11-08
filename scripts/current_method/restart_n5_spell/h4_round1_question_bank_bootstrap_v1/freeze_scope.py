"""Freeze the source scope for the Question Replay & Evolution Bank.

This stage is intentionally offline and stops before duplicate rerouting,
family construction, lifecycle assignment, or sampling-weight computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from . import (
    DEFAULT_RUN,
    E1Q0_RUN,
    E1Q1_COMPILER_RUN,
    E1Q1_FILTER_RUN,
    E1Q1_GENERATION_RUN,
    E1Q1_SOLVER_RUN,
    ROUND0_DIFFICULTY_RUN,
    ROUND0_REEVAL_RUN,
    SCOPE_VERSION,
    VERSION,
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    if path.suffix == ".jsonl":
        text = "".join(canonical(row) + "\n" for row in value)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def assert_sidecar(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_source_sidecar:{sidecar}")
    observed = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if observed != expected:
        raise RuntimeError(f"source_sidecar_mismatch:{path}")
    return observed


def require_unique(rows: Iterable[dict[str, Any]], key: str, corpus: str) -> None:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values):
        raise RuntimeError(f"missing_{key}:{corpus}")
    duplicate_count = len(values) - len(set(values))
    if duplicate_count:
        raise RuntimeError(f"duplicate_{key}:{corpus}:{duplicate_count}")


def schema_sha(rows: list[dict[str, Any]]) -> str:
    keys = sorted({key for row in rows for key in row})
    return hashlib.sha256(canonical(keys).encode("utf-8")).hexdigest()


def root_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["root_id"]) for row in rows).items()))


def corpus_manifest(
    *,
    corpus_id: str,
    source_round: str,
    rows: list[dict[str, Any]],
    source_files: list[Path],
    scope_note: str,
) -> dict[str, Any]:
    require_unique(rows, "candidate_id", corpus_id)
    return {
        "corpus_id": corpus_id,
        "source_round": source_round,
        "candidate_count": len(rows),
        "root_count": len(root_counts(rows)),
        "root_candidate_counts": root_counts(rows),
        "candidate_schema_sha256": schema_sha(rows),
        "scope_note": scope_note,
        "source_files": [
            {"path": str(path.resolve()), "sha256": assert_sidecar(path)} for path in source_files
        ],
    }


def freeze(run: Path = DEFAULT_RUN) -> dict[str, Any]:
    run = run.resolve()
    if run.exists():
        raise RuntimeError(f"output_already_exists:{run}")
    run.mkdir(parents=True, exist_ok=False)

    round0_path = ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl"
    round0 = load_jsonl(round0_path)
    round0_train = [row for row in round0 if row.get("split") == "train"]
    round0_split_counts = dict(sorted(Counter(str(row.get("split")) for row in round0).items()))

    exposure_path = ROUND0_REEVAL_RUN / "candidate_training_exposure_manifest.jsonl"
    transition_path = ROUND0_REEVAL_RUN / "candidate_transition_matrix.jsonl"
    exposure = load_jsonl(exposure_path)
    transition = load_jsonl(transition_path)
    require_unique(exposure, "candidate_id", "round0_train_exposure")
    require_unique(transition, "candidate_id", "round0_train_transition")
    if {row["candidate_id"] for row in round0_train} != {row["candidate_id"] for row in exposure}:
        raise RuntimeError("round0_train_exposure_candidate_set_mismatch")
    if {row["candidate_id"] for row in round0_train} != {row["candidate_id"] for row in transition}:
        raise RuntimeError("round0_train_transition_candidate_set_mismatch")

    e1q0_path = E1Q0_RUN / "all_candidates_valid.jsonl"
    e1q0_dialogue_path = E1Q0_RUN / "paired_dialogue_manifest.jsonl"
    e1q0 = load_jsonl(e1q0_path)
    e1q0_dialogues = load_jsonl(e1q0_dialogue_path)
    require_unique(e1q0_dialogues, "root_id", "e1q0_dialogues")

    success_path = E1Q1_COMPILER_RUN / "successful_root_manifest.jsonl"
    source_packet_path = E1Q1_COMPILER_RUN / "source_qa_packets.jsonl"
    e1q1_success = load_jsonl(success_path)
    source_packets = load_jsonl(source_packet_path)
    require_unique(source_packets, "root_id", "e1q1_source_packets")
    packet_roots = {str(row["root_id"]) for row in source_packets}

    e1q1: list[dict[str, Any]] = []
    e1q1_files: list[Path] = []
    dialogue_identity: list[dict[str, Any]] = []
    for root_dir in sorted(E1Q1_GENERATION_RUN.glob("cc-train-episode-*")):
        candidate_path = root_dir / "candidates_valid_scoped.json"
        dialogue_path = root_dir / "dialogue_source.json"
        if not candidate_path.is_file() or not dialogue_path.is_file():
            continue
        candidate_sha = assert_sidecar(candidate_path)
        dialogue_sha = assert_sidecar(dialogue_path)
        candidates = load_json(candidate_path)
        dialogue = load_json(dialogue_path)
        root_id = str(dialogue.get("root_id") or root_dir.name)
        if root_id != root_dir.name:
            raise RuntimeError(f"e1q1_dialogue_root_mismatch:{root_dir.name}:{root_id}")
        if any(str(row.get("root_id")) != root_id for row in candidates):
            raise RuntimeError(f"e1q1_candidate_root_mismatch:{root_id}")
        # The producer computes complete_dialogue_sha256 over the body without
        # the embedded hash field. Reproduce that contract exactly.
        dialogue_body = dict(dialogue)
        dialogue_body.pop("complete_dialogue_sha256", None)
        computed_dialogue_sha = hashlib.sha256(canonical(dialogue_body).encode("utf-8")).hexdigest()
        recorded_dialogue_sha = str(dialogue.get("complete_dialogue_sha256") or "")
        dialogue_identity.append(
            {
                "root_id": root_id,
                "candidate_count": len(candidates),
                "candidate_file": str(candidate_path.resolve()),
                "candidate_file_sha256": candidate_sha,
                "dialogue_file": str(dialogue_path.resolve()),
                "dialogue_file_sha256": dialogue_sha,
                "dialogue_content_sha256": computed_dialogue_sha,
                "recorded_complete_dialogue_sha256": recorded_dialogue_sha,
                "recorded_complete_dialogue_sha_matches": computed_dialogue_sha == recorded_dialogue_sha,
                "turn_count": len(dialogue.get("turns") or []),
                "qa_fact_source": dialogue.get("qa_fact_source"),
                "memory_is_qa_fact_source": dialogue.get("memory_is_qa_fact_source"),
            }
        )
        e1q1.extend(candidates)
        e1q1_files.append(candidate_path)

    e1q1_roots = set(root_counts(e1q1))
    if e1q1_roots != packet_roots:
        raise RuntimeError("e1q1_generated_roots_do_not_match_source_packets")
    if not all(row["recorded_complete_dialogue_sha_matches"] for row in dialogue_identity):
        raise RuntimeError("e1q1_dialogue_content_sha_mismatch")
    if not all(row["qa_fact_source"] == "complete_parent_plus_child" for row in dialogue_identity):
        raise RuntimeError("e1q1_invalid_qa_fact_source")
    if any(row["memory_is_qa_fact_source"] is not False for row in dialogue_identity):
        raise RuntimeError("e1q1_memory_fact_source_violation")

    expected_e1q0_roots = {str(row["root_id"]) for row in e1q0_dialogues}
    successful_roots = {str(row["root_id"]) for row in e1q1_success}
    if expected_e1q0_roots != successful_roots:
        raise RuntimeError("e1q0_success_root_manifest_mismatch")

    all_id_sets = {
        "ROUND0_ALL": {str(row["candidate_id"]) for row in round0},
        "E1Q0": {str(row["candidate_id"]) for row in e1q0},
        "E1Q1": {str(row["candidate_id"]) for row in e1q1},
    }
    namespace_overlaps = []
    names = sorted(all_id_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(all_id_sets[left] & all_id_sets[right])
            namespace_overlaps.append(
                {"left": left, "right": right, "overlap_count": len(overlap), "overlap_ids": overlap}
            )
    if any(row["overlap_count"] for row in namespace_overlaps):
        raise RuntimeError("candidate_namespace_collision")

    manifests = [
        corpus_manifest(
            corpus_id="ROUND0_ALL",
            source_round="ROUND0_PARENT_PLUS_R01",
            rows=round0,
            source_files=[round0_path],
            scope_note="All frozen Round-0 difficulty candidates across train/dev/test.",
        ),
        corpus_manifest(
            corpus_id="ROUND0_TRAIN",
            source_round="ROUND0_PARENT_PLUS_R01",
            rows=round0_train,
            source_files=[round0_path, exposure_path, transition_path],
            scope_note="Train-only Round-0 subset with trained-model transition and exposure metadata.",
        ),
        corpus_manifest(
            corpus_id="E1Q0",
            source_round="E1Q0_R01_R02_CONTROL_FEEDBACK",
            rows=e1q0,
            source_files=[e1q0_path, e1q0_dialogue_path],
            scope_note="Eight-root Control and Feedback QA baseline; branch remains part of candidate identity.",
        ),
        corpus_manifest(
            corpus_id="E1Q1",
            source_round="E1Q1_FULL_QA_FEEDBACK_COMPOSITE",
            rows=e1q1,
            source_files=e1q1_files + [success_path, source_packet_path],
            scope_note="Seven generated roots only; cc-train-episode-96462 is an explicit missing-generation root.",
        ),
    ]

    root_rows: list[dict[str, Any]] = []
    all_roots = sorted(set().union(*(set(manifest["root_candidate_counts"]) for manifest in manifests)))
    manifest_by_id = {manifest["corpus_id"]: manifest for manifest in manifests}
    for root_id in all_roots:
        root_rows.append(
            {
                "root_id": root_id,
                "round0_all_candidate_count": manifest_by_id["ROUND0_ALL"]["root_candidate_counts"].get(root_id, 0),
                "round0_train_candidate_count": manifest_by_id["ROUND0_TRAIN"]["root_candidate_counts"].get(root_id, 0),
                "e1q0_candidate_count": manifest_by_id["E1Q0"]["root_candidate_counts"].get(root_id, 0),
                "e1q1_candidate_count": manifest_by_id["E1Q1"]["root_candidate_counts"].get(root_id, 0),
                "e1q0_success_root": root_id in successful_roots,
                "e1q1_source_packet_present": root_id in packet_roots,
                "e1q1_generation_status": (
                    "GENERATED" if root_id in e1q1_roots else
                    "NO_SAFE_QA_FEEDBACK_SEED" if root_id == "cc-train-episode-96462" else
                    "OUT_OF_E1Q1_SCOPE"
                ),
            }
        )

    source_artifacts = {
        "e1q1_filter_report": E1Q1_FILTER_RUN / "final_qa_feedback_filter_report.json",
        "e1q1_solver_report": E1Q1_SOLVER_RUN / "final_trained_solver_report.json",
    }
    source_artifact_identity = [
        {"name": name, "path": str(path.resolve()), "sha256": assert_sidecar(path)}
        for name, path in source_artifacts.items()
    ]

    summary = {
        "status": "QUESTION_BANK_SCOPE_FROZEN",
        "next_status": "READY_FOR_E1Q1_MULTILABEL_REROUTING",
        "version": VERSION,
        "scope_version": SCOPE_VERSION,
        "corpus_candidate_counts": {row["corpus_id"]: row["candidate_count"] for row in manifests},
        "corpus_root_counts": {row["corpus_id"]: row["root_count"] for row in manifests},
        "round0_split_counts": round0_split_counts,
        "e1q0_success_root_count": len(successful_roots),
        "e1q1_generated_root_count": len(e1q1_roots),
        "e1q1_missing_generation_roots": sorted(successful_roots - e1q1_roots),
        "candidate_namespace_overlap_count": sum(row["overlap_count"] for row in namespace_overlaps),
        "online_calls": {"qa_writer": 0, "solver": 0, "memorax": 0, "semantic_auditor": 0, "expansion": 0},
        "deferred_stages": [
            "E1Q1_MULTILABEL_REROUTING",
            "QUESTION_FAMILY_CONSTRUCTION",
            "QUESTION_BANK_LIFECYCLE_ASSIGNMENT",
            "MISSING_SOLVER_MANIFEST",
            "SAMPLING_WEIGHT_COMPUTATION",
        ],
    }

    write(run / "corpus_scope_manifest.jsonl", manifests)
    write(run / "root_scope_manifest.jsonl", root_rows)
    write(run / "e1q1_dialogue_identity_manifest.jsonl", dialogue_identity)
    write(run / "candidate_namespace_audit.json", {"status": "PASS", "pairs": namespace_overlaps})
    write(run / "source_artifact_identity.json", source_artifact_identity)
    write(run / "final_scope_freeze_report.json", summary)
    write(
        run / "final_scope_freeze_report.md",
        "# Question Bank Scope Freeze v1\n\n"
        + "```json\n"
        + json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(freeze(args.run), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
