from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    freeze = json.loads((HERE / "FREEZE.json").read_text())
    manifest = json.loads((HERE / "FINAL_MANIFEST.json").read_text())
    reqs = json.loads((HERE / "REQUESTS.json").read_text())
    if freeze["status"] != "FROZEN_NOT_EXECUTED" or freeze["execution_authorized"]:
        raise RuntimeError("FREEZE_STATUS_INVALID")
    if reqs["request_count"] != 24 or len(reqs["requests"]) != 24:
        raise RuntimeError("REQUEST_COUNT_INVALID")
    if len({r["request_id"] for r in reqs["requests"]}) != 24:
        raise RuntimeError("REQUEST_IDS_NOT_UNIQUE")
    if freeze.get("independent_preflight_completed") is not True:
        raise RuntimeError("INDEPENDENT_PREFLIGHT_NOT_COMPLETE")
    audit_path = HERE / "INDEPENDENT_PREFLIGHT.json"
    if not audit_path.is_file():
        raise RuntimeError("INDEPENDENT_PREFLIGHT_MISSING")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS_READ_ONLY" or audit.get("model_inference_performed") is not False:
        raise RuntimeError("INDEPENDENT_PREFLIGHT_FAILED_OR_EXECUTED")
    if not any(row["path"] == "INDEPENDENT_PREFLIGHT.json" for row in manifest["local_artifacts"]):
        raise RuntimeError("INDEPENDENT_PREFLIGHT_NOT_IN_MANIFEST")
    mismatch = []
    missing = []
    for row in manifest["local_artifacts"] + manifest["external_dependencies"]:
        path = Path(row["path"]) if Path(row["path"]).is_absolute() else HERE / row["path"]
        if not path.is_file():
            missing.append(row["path"])
            continue
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            mismatch.append(row["path"])
    if missing or mismatch:
        raise RuntimeError(json.dumps({"missing": missing, "mismatch": mismatch}))
    freeze_bindings = [
        "protocol_binding", "manifest_binding", "preflight_binding",
        "source_adjudication_binding", "exposure_ledger_binding",
        "request_binding", "run_order_binding", "private_score_key_binding",
        "independent_preflight_binding",
    ]
    for key in freeze_bindings:
        row = freeze.get(key)
        if not isinstance(row, dict):
            raise RuntimeError("FREEZE_BINDING_MISSING:" + key)
        path = Path(row["path"])
        if not path.is_absolute():
            path = HERE / path
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError("FREEZE_BINDING_MISMATCH:" + key)
    model_runs = [p for p in (HERE / "runs").glob("**/*") if p.is_file()] if (HERE / "runs").exists() else []
    if model_runs:
        raise RuntimeError("UNEXPECTED_MODEL_RUN_ARTIFACTS")
    print(json.dumps({
        "status": "PASS_READ_ONLY",
        "local_artifacts": len(manifest["local_artifacts"]),
        "external_dependencies": len(manifest["external_dependencies"]),
        "missing": 0,
        "hash_or_size_mismatch": 0,
        "request_count": 24,
        "model_responses_found": 0,
        "execution_authorized": False,
    }))


if __name__ == "__main__":
    main()
