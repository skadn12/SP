"""Compatibility entry point for the Expansion-only durable-diversity probe.

Unlike the copied historical experiment wrapper, this branch deliberately
does not load a Solver or invoke QA.  Both module entry points route to the
same single-root Planner + Round Realizer runner.
"""

from __future__ import annotations

from .single_root_live import main


if __name__ == "__main__":
    raise SystemExit(main())
