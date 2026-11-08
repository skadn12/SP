"""Run the offline Round-1 QA feedback compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_RUN_ROOT
from .compiler import compile_qa_feedback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    report = compile_qa_feedback(args.run_root)
    print(json.dumps({"status": report["status"], "next_status": report["next_status"], "safe_root_count": report["safe_root_count"], "root_count": report["root_count"], "online_calls": report["online_calls"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

