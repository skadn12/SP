"""Run the isolated offline B0 K=3 hard evidence stratification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_RUN_ROOT, DEFAULT_SOURCE_RUN
from .stratify import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args(argv)
    report = run(source_run=args.source_run, run_root=args.run_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
