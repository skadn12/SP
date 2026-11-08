"""Offline, lossless option-shape normalization of the frozen v5 responses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "h4_crossroot_multiselect_batch_v5"
ROOTS = (
    "cc-train-episode-136202",
    "cc-train-episode-197635",
    "cc-train-episode-254852",
)
LABELS = tuple("ABCDEF")
OPTION_RE = re.compile(r"^([A-F])\. ([^\s].*)$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def v5_module():
    path = V5 / "generate.py"
    spec = importlib.util.spec_from_file_location("h4_crossroot_multiselect_batch_v5", path)
    if spec is None or spec.loader is None:
        raise ImportError("v5_validator_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_options(options: object) -> tuple[dict | None, str | None]:
    if isinstance(options, dict):
        if set(options) == set(LABELS):
            return options.copy(), "ALREADY_OBJECT"
        return None, "OPTIONS_OBJECT_LABELS_INVALID"
    if not isinstance(options, list) or len(options) != 6:
        return None, "OPTIONS_NOT_SIX_STRINGS"
    normalized = {}
    for expected, value in zip(LABELS, options):
        if not isinstance(value, str):
            return None, f"OPTION_{expected}_NOT_STRING"
        match = OPTION_RE.fullmatch(value)
        if match is None or match.group(1) != expected:
            return None, f"OPTION_{expected}_MISSING_OR_AMBIGUOUS_PREFIX"
        normalized[expected] = match.group(2)
    return normalized, "ORDERED_PREFIX_ARRAY_NORMALIZED"


def checked_input(root: str) -> tuple[dict, dict, dict, dict, dict]:
    directory = V5 / root
    prepared = read(V5 / "BATCH_PREPARED.json")
    if (prepared.get("version") != V5.name
            or prepared.get("script_sha256") != sha(V5 / "generate.py")
            or prepared.get("instructions_sha256") != sha(V5 / "instructions.txt")):
        raise ValueError("V5_PREPARED_GENERATOR_BINDING_MISMATCH")
    entry = next((row for row in prepared["roots"] if row["root_id"] == root), None)
    if entry is None:
        raise ValueError("ROOT_NOT_IN_V5_PREPARED:" + root)
    source = read(directory / "source.json")
    parsed = read(directory / "PARSED_RESPONSE.json")
    response = read(directory / "RESPONSE.json")
    validation = read(directory / "VALIDATION.json")
    if (source.get("root_id") != root or sha(directory / "source.json") != entry["source_sha256"]
            or response.get("source_sha256") != entry["source_sha256"]
            or response.get("payload_sha256") != entry["payload_sha256"]
            or sha(directory / "prompt.txt") != entry["prompt_sha256"]
            or validation.get("root_id") != root
            or validation.get("provenance", {}).get("source_sha256") != entry["source_sha256"]
            or validation.get("provenance", {}).get("payload_sha256") != entry["payload_sha256"]
            or parsed != json.loads(response["raw_text"])):
        raise ValueError("V5_FROZEN_INPUT_BINDING_MISMATCH:" + root)
    return source, parsed, response, validation, entry


def normalize_root(root: str, validator) -> dict:
    source, parsed, response, validation, entry = checked_input(root)
    original = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(original, list):
        original = []
    checks = validation.get("checks", [])
    normalized, derived, mappings, dispositions = [], [], [], []
    global_errors = validation.get("global_errors", [])
    for index, row in enumerate(original, 1):
        prior = checks[index - 1] if index <= len(checks) else None
        prior_errors = prior.get("errors") if isinstance(prior, dict) else None
        if global_errors or not isinstance(prior_errors, list):
            dispositions.append({"index": index, "status": "REJECTED_V5_GLOBAL_OR_CHECKS_INVALID"})
            continue
        if not isinstance(row, dict):
            dispositions.append({"index": index, "status": "REJECTED_CANDIDATE_NOT_OBJECT"})
            continue
        options, shape = normalize_options(row.get("options"))
        if options is None:
            dispositions.append({"index": index, "candidate_id": row.get("candidate_id"),
                                 "status": "REJECTED_UNRECOVERABLE_OPTION_SHAPE", "reason": shape})
            continue
        candidate = dict(row)
        candidate["options"] = options
        errors = validator.check_candidate(candidate, root, source["turns"])
        if errors:
            dispositions.append({"index": index, "candidate_id": row.get("candidate_id"),
                                 "status": "REJECTED_AFTER_V5_REVALIDATION", "errors": errors,
                                 "option_shape": shape})
            continue
        relabeled, mapping = validator.relabel(candidate)
        derived_errors = validator.check_candidate(relabeled, root, source["turns"])
        if derived_errors:
            raise ValueError("DERIVED_REVALIDATION_FAILED:" + root + ":" + str(derived_errors))
        mapping = {**mapping, "normalized_from_v5_option_shape": shape,
                   "v5_original_candidate_index": index}
        normalized.append(candidate)
        derived.append(relabeled)
        mappings.append(mapping)
        dispositions.append({"index": index, "candidate_id": row.get("candidate_id"),
                             "status": "MECHANICALLY_VALID_CONTENT_AUDIT_PENDING",
                             "option_shape": shape, "derived_gold": relabeled["gold"]})
    source_dir = V5 / root
    hashes = {name: sha(source_dir / name) for name in (
        "source.json", "PARSED_RESPONSE.json", "RESPONSE.json", "VALIDATION.json",
        "generation_view.json", "prompt.txt", "request.json")}
    status = ("MECHANICALLY_VALID_CONTENT_AUDIT_PENDING" if normalized else
              "REJECTED_NO_RECOVERABLE_CANDIDATE")
    return {
        "version": HERE.name, "root_id": root, "status": status,
        "source_batch": V5.name, "source_model": validation["provenance"]["returned_model"],
        "source_response_id": validation["provenance"]["response_id"],
        "source_sha256": entry["source_sha256"],
        "source_file_sha256": hashes,
        "v5_generate_py_sha256": sha(V5 / "generate.py"),
        "v6_normalize_py_sha256": sha(Path(__file__)),
        "v5_original_validation_status": validation["status"],
        "v5_original_validation_errors": validation["checks"],
        "source_response_finish_reason": response["metadata"].get("finish_reason"),
        "original_candidates": original,
        "normalized_candidates": normalized,
        "derived_candidates": derived,
        "relabel_mappings": mappings,
        "dispositions": dispositions,
        "candidate_count": len(original),
        "mechanically_normalized_count": len(normalized),
        "content_audited": False,
        "solver_evaluated": False,
        "selected_for_training": False,
        "training_started": False,
    }


def main() -> None:
    if (HERE / "BATCH_RESULTS.json").exists() or any((HERE / root).exists() for root in ROOTS):
        raise FileExistsError("V6_OUTPUT_ALREADY_EXISTS_NO_OVERWRITE")
    validator = v5_module()
    results = [normalize_root(root, validator) for root in ROOTS]
    for result in results:
        directory = HERE / result["root_id"]
        directory.mkdir()
        write_new(directory / "NORMALIZATION.json", result)
    write_new(HERE / "BATCH_RESULTS.json", {
        "version": HERE.name, "source_batch": V5.name,
        "roots": [{"root_id": row["root_id"], "status": row["status"],
                   "candidate_count": row["candidate_count"],
                   "mechanically_normalized_count": row["mechanically_normalized_count"],
                   "dispositions": row["dispositions"]} for row in results],
        "candidate_count": sum(row["candidate_count"] for row in results),
        "mechanically_normalized_count": sum(row["mechanically_normalized_count"] for row in results),
        "content_audited_count": 0, "solver_calls": 0, "training_started": False,
    })
    print(json.dumps({"status": "OFFLINE_NORMALIZATION_DONE", "roots": len(results),
                      "candidate_count": sum(row["candidate_count"] for row in results),
                      "mechanically_normalized_count": sum(row["mechanically_normalized_count"]
                                                           for row in results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
