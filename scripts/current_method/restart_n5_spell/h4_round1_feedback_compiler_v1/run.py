"""Run the frozen Round-1 offline feedback compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_RUN_ROOT
from .compiler import compile_feedback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    report = compile_feedback(parse_args().run_root)
    print(json.dumps({
        "status": report["status"],
        "next_status": report["next_status"],
        "candidate_count": report["candidate_count"],
        "primary_transition_counts": report["primary_transition_counts"],
        "routing_counts": report["routing_counts"],
        "formal_root_count": report["formal_root_count"],
        "reserve_root_count": report["reserve_root_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

