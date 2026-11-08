"""Entry point for the repaired B1 visible-label content condition."""

from __future__ import annotations

import sys

from .b2_repair_run import main as _content_main


if __name__ == "__main__":
    raise SystemExit(_content_main(["--stage", "b1", *sys.argv[1:]]))
