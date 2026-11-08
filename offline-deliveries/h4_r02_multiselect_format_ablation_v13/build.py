"""Replay frozen Qwen drafts with citation gates disabled, without authoring edits.

This is a diagnostic ablation. It neither certifies Gold semantics nor promotes
any resulting question to the formal training pool.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "h4_r02_multiselect_format_ablation_v13"
MODEL = "Qwen/Qwen3.6-35B-A3B"
CITATION_GATE = "DIAGNOSTIC_ONLY_DISABLED_BY_USER"
TARGETS = (("A", "cc-train-episode-104814"), ("B", "cc-train-episode-138343"))
LABELS = "ABCDEF"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def scoreability_errors(draft: dict) -> list[str]:
    """Only requirements necessary to present and score a six-option question."""
    errors = []
    if not isinstance(draft.get("question"), str) or not draft["question"].strip():
        errors.append("UNSCORABLE_QUESTION_TEXT")
    options = draft.get("options")
    if not isinstance(options, list) or len(options) != 6:
        return errors + ["UNSCORABLE_OPTION_COUNT"]
    propositions = [option.get("proposition") if isinstance(option, dict) else None
                    for option in options]
    if (any(not isinstance(text, str) or not text.strip() for text in propositions)
            or len({text.strip().casefold() for text in propositions if isinstance(text, str)}) != 6):
        errors.append("UNSCORABLE_DISTINCT_OPTION_TEXT")
    relations = [option.get("relation") if isinstance(option, dict) else None
                 for option in options]
    if any(relation not in {"supports", "contradicts"} for relation in relations):
        errors.append("UNSCORABLE_RELATION_VALUES")
    if not 2 <= relations.count("supports") <= 6:
        errors.append("UNSCORABLE_GOLD_CARDINALITY")
    return errors


def build(source_run: Path, run: Path) -> dict:
    source_run, run = source_run.resolve(), run.resolve()
    if run.exists():
        raise FileExistsError(f"WRITE_ONCE_RUN_ALREADY_EXISTS:{run}")
    if source_run == run or source_run in run.parents:
        raise ValueError("OUTPUT_MUST_NOT_MODIFY_HISTORICAL_RUN")
    source_manifest_path = source_run / "MANIFEST.json"
    source_manifest = read(source_manifest_path)
    if source_manifest.get("generator_model") != MODEL:
        raise ValueError("SOURCE_GENERATOR_IDENTITY_MISMATCH")
    originals: dict[Path, str] = {source_manifest_path: sha(source_manifest_path)}
    top_bindings = {Path(row["path"]).resolve(): row["sha256"]
                    for row in source_manifest.get("files", [])}
    latest: dict[str, dict] = {}
    snapshots: list[dict] = []
    source_rows: dict[tuple[str, str], dict] = {}
    outcomes: dict[str, dict] = {}
    archives: dict[Path, Path] = {source_manifest_path: Path("raw/source_run_MANIFEST.json")}

    for branch, root in TARGETS:
        generation = source_run / branch / "generation"
        manifest_path = generation / "MANIFEST.json"
        manifest = read(manifest_path)
        if manifest.get("generator_model") != MODEL:
            raise ValueError("GENERATION_MODEL_MISMATCH:" + branch)
        matching_sources = [row for row in manifest["source_rows"] if row["root_id"] == root]
        if len(matching_sources) != 1:
            raise ValueError("UNIQUE_FROZEN_SOURCE_REQUIRED:" + root)
        source_row = matching_sources[0]
        source_path = Path(source_row["source_path"]).resolve()
        source_sha = sha(source_path)
        if source_sha != source_row["source_sha256"] or top_bindings.get(source_path) != source_sha:
            raise ValueError("FROZEN_SOURCE_HASH_MISMATCH:" + root)
        source = read(source_path)
        if source.get("root_id") != root or not isinstance(source.get("turns"), list) or not source["turns"]:
            raise ValueError("FROZEN_SOURCE_ROOT_OR_TURNS_INVALID:" + root)
        refs = [turn.get("turn_ref") for turn in source["turns"]]
        if len(set(refs)) != len(refs):
            raise ValueError("FROZEN_SOURCE_TURN_REFS_NOT_UNIQUE:" + root)
        originals[source_path] = source_sha
        source_rows[branch, root] = {**source_row, "original_source_path": str(source_path),
                                    "source_path": str(run / "sources" / branch / (root + ".json"))}
        archives[source_path] = Path("sources") / branch / (root + ".json")
        archives[manifest_path] = Path("raw") / branch / "GENERATION_MANIFEST.json"
        outcome_path = generation / "outcomes" / (root + ".multi_select.json")
        outcome = read(outcome_path)
        if outcome.get("root_id") != root or outcome.get("question_type") != "multi_select":
            raise ValueError("ORIGINAL_OUTCOME_IDENTITY_MISMATCH:" + root)
        archives[outcome_path] = Path("raw") / branch / "ORIGINAL_OUTCOME.json"
        for rejected in outcome.get("mechanical_rejected", []):
            outcomes[rejected["candidate_id"]] = copy.deepcopy(rejected)
        responses = sorted((generation / "requests").glob("*multi*/RESPONSE.json"))
        if not responses:
            raise ValueError("NO_RAW_RESPONSES:" + root)
        for response_path in responses:
            request_path = response_path.with_name("REQUEST.json")
            request, response = read(request_path), read(response_path)
            metadata = response.get("metadata") or {}
            if (request.get("model") != MODEL or metadata.get("requested_model") != MODEL
                    or metadata.get("returned_model") != MODEL):
                raise ValueError("RAW_REQUEST_RETURNED_MODEL_MISMATCH:" + str(response_path))
            parsed = json.loads(response["text"])
            if parsed.get("root_id") != root or not isinstance(parsed.get("questions"), list):
                raise ValueError("RAW_RESPONSE_ROOT_OR_QUESTIONS_INVALID:" + str(response_path))
            for file_path in (request_path, response_path):
                archives[file_path] = Path("raw") / branch / response_path.parent.name / file_path.name
            validation_path = response_path.with_name("VALIDATION.json")
            if validation_path.is_file():
                archives[validation_path] = Path("raw") / branch / response_path.parent.name / validation_path.name
            seen = set()
            for index, draft in enumerate(parsed["questions"]):
                qid = draft.get("candidate_id")
                if not isinstance(qid, str) or not qid.startswith(root + "-") or qid in seen:
                    raise ValueError("RAW_CANDIDATE_ID_INVALID_OR_DUPLICATE:" + str(qid))
                seen.add(qid)
                snapshot = {"candidate_id": qid, "root_id": root, "branch": branch,
                            "response_path": str(response_path), "response_sha256": sha(response_path),
                            "request_path": str(request_path), "request_sha256": sha(request_path),
                            "response_question_index": index, "requested_model": MODEL,
                            "returned_model": metadata["returned_model"], "draft": copy.deepcopy(draft)}
                snapshots.append(snapshot)
                latest[qid] = snapshot
        branch_ids = [qid for qid, item in latest.items() if item["branch"] == branch]
        if len(branch_ids) != 4:
            raise ValueError("EXPECTED_FOUR_RAW_CANDIDATE_IDS:" + branch)
    if len(latest) != 8 or set(latest) != set(outcomes):
        raise ValueError("RAW_CANDIDATE_OUTCOME_SET_MISMATCH")
    for path in archives:
        originals[path] = sha(path)

    questions, diagnostics = [], []
    for qid, snapshot in sorted(latest.items()):
        draft, branch, root = snapshot["draft"], snapshot["branch"], snapshot["root_id"]
        errors = scoreability_errors(draft)
        original = outcomes[qid]
        diagnostic = {"candidate_id": qid, "root_id": root, "branch": branch,
                      "diagnostic_only": True, "citation_gate": CITATION_GATE,
                      "raw_draft_policy": "LAST_RESPONSE_CONTAINING_CANDIDATE_NO_FALLBACK",
                      "raw_response_path": snapshot["response_path"],
                      "raw_response_sha256": snapshot["response_sha256"],
                      "original_outcome": original,
                      "original_errors": copy.deepcopy(original.get("errors", [])),
                      "historical_duplicate_diagnostic": "duplicate_full_statement_set" in original.get("errors", []),
                      "evaluation_status": "UNSCORABLE" if errors else "READY_FOR_DIAGNOSTIC_SOLVER",
                      "scoreability_errors": errors}
        diagnostics.append(diagnostic)
        if errors:
            continue
        options = dict(zip(LABELS, [option["proposition"] for option in draft["options"]]))
        evidence = {}
        for label, option in zip(LABELS, draft["options"]):
            evidence[label] = {key: copy.deepcopy(value) for key, value in option.items()
                               if key != "proposition"}
            evidence[label]["reason"] = copy.deepcopy(option.get("rationale"))
        source_row = source_rows[branch, root]
        questions.append({"candidate_id": qid, "root_id": root, "branch": branch,
                          "question_type": "multi_select", "question": draft["question"],
                          "options": options,
                          "gold": [label for label, option in zip(LABELS, draft["options"])
                                   if option["relation"] == "supports"],
                          "option_evidence": evidence, "source_path": source_row["source_path"],
                          "source_sha256": source_row["source_sha256"], "generator_model": MODEL,
                          "citation_gate": CITATION_GATE, "diagnostic_only": True,
                          "original_outcome_errors": diagnostic["original_errors"],
                          "raw_response_sha256": snapshot["response_sha256"]})
    pool = {"version": VERSION, "diagnostic_only": True, "citation_gate": CITATION_GATE,
            "formal_training_promotion": False,
            "provenance": {"requested_model": MODEL, "returned_model": MODEL,
                           "raw_draft_policy": "LAST_RESPONSE_CONTAINING_CANDIDATE_NO_FALLBACK",
                           "source_run": str(source_run), "codex_question_edits": False,
                           "new_generation_calls": 0, "new_semantic_review_calls": 0},
            "questions": questions}
    # All source checks finish before creating a write-once output directory.
    run.mkdir(parents=True, exist_ok=False)
    for original, relative in archives.items():
        destination = run / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        if sha(destination) != originals[original]:
            raise ValueError("COPY_HASH_MISMATCH:" + str(original))
    write(run / "RAW_SNAPSHOTS.json", {"version": VERSION, "snapshots": snapshots})
    write(run / "DIAGNOSTICS.json", {"version": VERSION, "diagnostic_only": True,
                                     "candidates": diagnostics})
    write(run / "SOLVER_POOL.json", pool)
    for original, expected in originals.items():
        if sha(original) != expected:
            raise ValueError("ORIGINAL_CHANGED_DURING_BUILD:" + str(original))
    summary = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
               "source_run": str(source_run), "source_run_manifest_sha256": originals[source_manifest_path],
               "diagnostic_only": True, "citation_gate": CITATION_GATE,
               "raw_candidates": len(latest), "raw_snapshots": len(snapshots),
               "scoreable_candidates": len(questions), "unscorable_candidates": len(latest) - len(questions),
               "new_generation_calls": 0, "ground_calls": 0, "training_calls": 0,
               "formal_training_promotion": False, "originals_unchanged": True,
               "source_rows": list(source_rows.values()),
               "original_files": [{"path": str(path), "sha256": digest} for path, digest in originals.items()],
               "files": [{"path": str(path.resolve()), "sha256": sha(path)}
                         for path in sorted(run.rglob("*")) if path.is_file()],
               "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))}}
    write(run / "MANIFEST.json", summary)
    return {key: summary[key] for key in ("version", "raw_candidates", "raw_snapshots",
                                         "scoreable_candidates", "unscorable_candidates", "originals_unchanged")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_run, args.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
