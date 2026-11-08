"""Run the R03 existing-Memory read-only trial preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_OUTPUT
from .preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_preflight(args.run_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

