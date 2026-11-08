"""Run the pure-offline R03 local batch closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_OUTPUT
from .closure import execute_closure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = execute_closure(args.run_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

