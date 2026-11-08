"""Score a completed position-bound ordering run using the existing policy."""

from __future__ import annotations

import importlib.util

from build import HERE


BASE_PATH = HERE.parent / "h4_crossroot_ordering_no_codex_solver_v1" / "report_results.py"
spec = importlib.util.spec_from_file_location("position_ordering_base_report", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.POOL = HERE / "READY_FOR_EVALUATION.json"
base.AUDIT = HERE / "REVIEW_REBIND_AUDIT.json"
base.RUN = HERE / "run"


if __name__ == "__main__":
    base.main()
