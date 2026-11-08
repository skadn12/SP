#!/usr/bin/env python3
"""Run the frozen question driver against an immutable recovery selection.

The canonical source-driver selection remains untouched.  This wrapper only
sets the imported driver's selection path for the current process, allowing a
partial recovery to be analyzed honestly without fabricating failed roots.
"""
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
    parser.add_argument("--selection", default=str(BASE / "SOURCE_SELECTION_RECOVERY_CONTINUED.json"))
    parser.add_argument("--source-freeze", default=str(BASE / "SOURCE_FREEZE_RECOVERY_CONTINUED.json"))
    parser.add_argument("command", choices=("run", "finalize"))
    parser.add_argument("--root", action="append")
    parser.add_argument("--arm", action="append", choices=driver.ARMS)
    args = parser.parse_args()
    selection = Path(args.selection).resolve()
    if not selection.exists():
        raise FileNotFoundError(selection)
    source_freeze = Path(args.source_freeze).resolve()
    if not source_freeze.exists():
        raise FileNotFoundError(source_freeze)
    driver.SOURCE_SELECTION_PATH = selection
    driver.SOURCE_FREEZE_PATH = source_freeze
    if args.command == "finalize":
        result = driver.finalize()
        print(result)
        return 0
    run_args = argparse.Namespace(root=args.root, arm=args.arm)
    return driver.run(run_args)


if __name__ == "__main__":
    raise SystemExit(main())
