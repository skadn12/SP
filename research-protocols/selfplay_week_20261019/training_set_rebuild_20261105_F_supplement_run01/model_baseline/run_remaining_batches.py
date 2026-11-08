#!/usr/bin/env python3
"""Run each prepared supplement batch once, sequentially on the local GPU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RUN = Path(__file__).resolve().parent.parent
BATCHES = (
    "batch_05_scope_adjudicated",
    "batch_06_scope_adjudicated",
    "batch_07",
    "batch_08_scope_adjudicated",
    "batch_09",
)


def main():
    for name in BATCHES:
        batch = RUN / "evaluation" / name
        solver = batch / "solver"
        assert (solver / "PREFLIGHT.json").is_file()
        assert (solver / "REQUESTS.json").is_file()
        assert not (solver / "STARTED.json").exists(), ("NO_RETRY", name)
    for name in BATCHES:
        batch = RUN / "evaluation" / name
        print(f"RUN_START {name}", flush=True)
        subprocess.run([sys.executable, str(RUN / "model_baseline/adapter.py"), "run",
                        "--batch-dir", str(batch), "--outdir", str(batch / "solver"),
                        "--max-items", "68"], check=True)
        print(f"RUN_COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
