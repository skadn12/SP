"""Execute exactly one pre-frozen Day 5 stage and retain terminal status."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def write_new(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("smoke", "A1", "B1"):
        raise SystemExit("usage: launch_stage.py smoke|A1|B1")
    stage = sys.argv[1]
    freeze_path = OUT / "EXECUTION_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    for ref in freeze["input_bindings"]:
        if binding(ref["path"]) != ref:
            raise RuntimeError("Frozen input changed: " + ref["path"])
    if binding(__file__) != freeze["launcher"]:
        raise RuntimeError("Launcher changed")
    spec = freeze["stages"][stage]
    if Path(spec["report"]).exists() or Path(spec["output_dir"]).exists():
        raise RuntimeError("Refusing to overwrite stage output")
    write_new(OUT / (stage + "_START.json"), {"stage": stage, "started_at_utc": now(),
              "freeze": binding(freeze_path), "argv": spec["argv"], "automatic_retry": False})
    with (OUT / (stage + ".log")).open("x") as log:
        result = subprocess.run(spec["argv"], cwd=freeze["working_directory"],
                                stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    write_new(OUT / (stage + "_TERMINAL.json"), {"stage": stage, "completed_at_utc": now(),
              "exit_code": result.returncode, "status": "COMPLETE" if result.returncode == 0 else "FAILED",
              "report_exists": Path(spec["report"]).exists(), "log": binding(OUT / (stage + ".log"))})
    print(json.dumps({"stage": stage, "exit_code": result.returncode, "report": spec["report"]}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
