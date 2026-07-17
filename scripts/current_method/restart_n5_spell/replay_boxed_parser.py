"""Read-only legacy solver replay under the boxed admission parser."""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import (atomic_json, parse_boxed_mcq_answer, parse_mcq_response,
                   sha256_file)


def replay(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"replay destination already exists: {destination}")
    qa_path = source / "frozen/solver_qa.json"
    import json
    qa_payload = json.loads(qa_path.read_text(encoding="utf-8"))
    qa = qa_payload.get("qa", qa_payload)
    options = qa["options"]
    rows = []
    for path in sorted((source / "raw/backend_responses").glob("*full_context*.txt")):
        raw = path.read_text(encoding="utf-8")
        legacy = parse_mcq_response(raw, options)
        boxed = parse_boxed_mcq_answer(raw, options)
        rows.append({"source_path": str(path), "source_raw_sha256": sha256_file(path),
            "raw_response": raw, "legacy_parse_status": legacy["parse_status"],
            "legacy_parsed_letter": legacy.get("parsed_letter"),
            "boxed_parse_status": boxed["parse_status"],
            "boxed_parsed_letter": boxed.get("parsed_letter")})
    destination.mkdir(parents=True)
    report = {"mode":"offline_parser_replay", "source_run":str(source),
        "source_qa_sha256":sha256_file(qa_path), "parser_mode_legacy":"legacy_audit_only",
        "parser_mode_admission":"boxed_letter_v1", "vllm_generations":0,
        "paid_api_generations":0, "records":rows,
        "interpretation":("Old free-text output being invalid under boxed parsing does not imply "
            "semantic ambiguity; it means only that the output violates the new frozen protocol.")}
    atomic_json(destination / "reports/boxed_parser_replay.json", report)
    atomic_json(destination / "config/run_manifest.json", {"mode":"offline_parser_replay",
        "source_run":str(source), "source_qa_sha256":sha256_file(qa_path),
        "boxed_parser_version":"boxed_letter_v1", "authoritative_old_decisions_modified":False})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    replay(args.source.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
