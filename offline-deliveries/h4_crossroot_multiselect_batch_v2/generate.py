"""Independent v2 cross-root multi-select batch using the frozen v1 runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "h4_crossroot_multiselect_batch_v1/generate.py"
BASE_SHA256 = "626752711e39aaef285e4b936048f3eb48ccedab6868c5cb11baeed5dfc26b46"


def runner():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("v1_runner_dependency_changed")
    spec = importlib.util.spec_from_file_location("h4_crossroot_multiselect_batch_v1_runner", BASE)
    if spec is None or spec.loader is None:
        raise ImportError("v1_runner_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.VERSION = HERE.name
    module.__file__ = __file__

    original_check = module.check_candidate
    expected_fields = {
        "candidate_id", "root_id", "question_type", "topic", "question",
        "options", "gold", "option_evidence", "answer_reason",
    }

    def check_candidate(row: object, root_id: str, turns: list[dict]) -> list[str]:
        errors = original_check(row, root_id, turns)
        if not isinstance(row, dict):
            return errors
        if set(row) != expected_fields:
            errors.append("candidate_schema_fields_mismatch")
        evidence = row.get("option_evidence")
        if isinstance(evidence, dict):
            for label, item in evidence.items():
                if label in module.LABELS and isinstance(item, dict) and set(item) != {
                    "turn_ref", "quote", "relation", "reason"
                }:
                    errors.append(f"{label}:evidence_schema_fields_mismatch")
        return errors

    module.check_candidate = check_candidate
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    actions.add_argument("--validate", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    actions.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    batch = runner()
    if args.preflight:
        batch.preflight()
    elif args.execute:
        batch.execute(args.execute)
    elif args.validate:
        print(json.dumps(batch.validate(args.validate), ensure_ascii=False))
    elif args.summary:
        batch.summary()
    else:
        batch.prepare()
