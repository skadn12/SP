from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((HERE / "EXECUTION_MANIFEST.json").read_text())
    if manifest.get("status") != "EXECUTION_AUTHORIZED_NOT_STARTED" or manifest.get("request_count_maximum") != 24:
        raise RuntimeError("EXECUTION_FREEZE_INVALID")
    if manifest.get("automatic_retries") != 0 or manifest.get("separate_from_unchanged_parent_freeze") is not True:
        raise RuntimeError("EXECUTION_BOUNDARY_INVALID")
    missing, mismatch = [], []
    for row in manifest["bindings"]:
        path = Path(row["path"])
        if not path.is_file():
            missing.append(row["path"])
        elif path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            mismatch.append(row["path"])
    if missing or mismatch:
        raise RuntimeError(json.dumps({"missing": missing, "mismatch": mismatch}))
    audit = json.loads((HERE / "RUNNER_PREFLIGHT.json").read_text())
    if audit.get("status") != "PASS_READ_ONLY" or audit.get("model_inference_performed") is not False:
        raise RuntimeError("INDEPENDENT_RUNNER_PREFLIGHT_FAILED")
    print(json.dumps({"status": "PASS_READ_ONLY", "bound_files": len(manifest["bindings"]),
        "maximum_model_requests": 24, "inference_performed": False}))


if __name__ == "__main__":
    main()
