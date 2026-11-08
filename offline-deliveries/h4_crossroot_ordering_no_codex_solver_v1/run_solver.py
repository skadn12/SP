"""Run the existing ordering Solver protocol on an automatic-review-only pool."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from build import HERE, BATCH, make_pool, read, sha, require


BASE_PATH = HERE.parent / "h4_crossroot_ordering_solver_selection_v1" / "run_solver.py"
POOL = HERE / "READY_FOR_EVALUATION.json"
AUDIT = HERE / "AUTOMATIC_REVIEW_AUDIT.json"
RUN = HERE / "run"


def load_base():
    spec = importlib.util.spec_from_file_location("auto_review_ordering_base_runner", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
base.POOL = POOL
base.AUDIT = AUDIT
base.BATCH = BATCH
base.RUN = RUN
base.SEED_BASE = 2026100900


def checked_inputs():
    pool, audit = read(POOL), read(AUDIT)
    expected_pool, expected_audit = make_pool()
    require(pool == expected_pool and audit == expected_audit,
            "automatic_review_pool_or_audit_changed")
    questions = pool["questions"]
    require(questions and len(questions) == audit["candidate_count"],
            "automatic_review_candidate_count_invalid")
    sources = {}
    for question in questions:
        root = question["root_id"]
        source_path = BATCH / root / "source.json"
        source = read(source_path)
        require(source["root_id"] == root
                and question["source_path"] == str(source_path)
                and question["source_sha256"] == sha(source_path),
                "automatic_review_source_invalid:" + question["candidate_id"])
        sources[root] = {"path": source_path, "source": source,
                         "sha256": sha(source_path),
                         "turn_count": len(source["turns"])}
    return pool, audit, questions, sources


base.checked_inputs = checked_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "live"))
    args = parser.parse_args()
    {"preflight": base.preflight, "live": base.live}[args.phase]()


if __name__ == "__main__":
    main()
