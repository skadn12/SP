"""Run the offline Memory-bucket QA feedback compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_RUN_ROOT
from .compiler import compile_memory_bucket_feedback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    report = compile_memory_bucket_feedback(args.run_root)
    print(json.dumps({
        "status": report["status"],
        "next_status": report["next_status"],
        "safe_root_count": report["safe_root_count"],
        "root_count": report["root_count"],
        "blockers": sorted({b for row in report["root_reports"] for b in row["blockers"]}),
        "online_calls": report["online_calls"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

