"""Run the single-root R03 read-only Memory smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_OUTPUT, ROOT_ID
from .smoke import execute_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root-id", default=ROOT_ID)
    args = parser.parse_args()
    report = execute_smoke(args.run_root, root_id=args.root_id)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
