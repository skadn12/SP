#!/usr/bin/env python3
"""Run the frozen question driver against a separately frozen recovery selection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as driver  # noqa: E402

driver.EXTRA_DRIVER_BINDINGS = list(driver.EXTRA_DRIVER_BINDINGS) + [Path(__file__).resolve()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "finalize"))
    parser.add_argument("--selection", default=str(BASE / "SOURCE_SELECTION_RECOVERY_CONTINUED.json"))
    parser.add_argument("--source-freeze", default=str(BASE / "SOURCE_FREEZE_RECOVERY_CONTINUED.json"))
    parser.add_argument("--root", action="append")
    parser.add_argument("--arm", action="append", choices=driver.ARMS)
    args = parser.parse_args()

    selection = Path(args.selection).resolve()
    source_freeze = Path(args.source_freeze).resolve()
    if not selection.is_file() or not source_freeze.is_file():
        raise FileNotFoundError("Continued source selection and freeze must both exist")
    driver.SOURCE_SELECTION_PATH = selection
    driver.SOURCE_FREEZE_PATH = source_freeze
    if args.command == "finalize":
        result = driver.finalize()
        print(result)
        return 0
    return driver.run(argparse.Namespace(root=args.root, arm=args.arm))


if __name__ == "__main__":
    raise SystemExit(main())
