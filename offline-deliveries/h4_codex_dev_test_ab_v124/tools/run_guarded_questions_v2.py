"""Opt-in future controller entry point enforcing review provenance first.

This additive wrapper does not patch historical/frozen controllers. Future
batches must explicitly invoke it for compose/finalize. --check-only is read
only. A blocked gate returns nonzero before downstream code is loaded/called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from review_provenance_gate_v2 import JsonArgumentParser, binding, check, sha
from public_anchor_v1 import validate_runtime

PHASES = {"compose": "asset", "finalize": "public"}


def invoke_frozen_controller(batch, unit, action):
    freeze = json.loads((batch / "MASTER_FREEZE.json").read_text(encoding="utf-8"))
    # The new entry point binds and imports the inherited-prefix scene adapter.
    names = ("question_controller_v120.py", "scene_controller_v120.py", "public_anchor_v1.py")
    for name in names:
        path = batch / "tools" / name
        matches = [row for row in freeze["files"] if row.get("path") == str(path)]
        if len(matches) != 1 or matches[0].get("sha256") != sha(path):
            raise RuntimeError("FROZEN_CONTROLLER_CODE_MISMATCH:" + str(path))
    return subprocess.run([sys.executable, "-B", str(batch / "tools/question_controller_v120.py"),
                           action, "--batch-dir", str(unit)], check=False,
                          capture_output=True, text=True)


def run(batch, root, action, *, check_only=False, downstream=None):
    batch = Path(batch).resolve()
    try:
        validate_runtime(batch, __file__, {"review_provenance_gate_v2.py": check})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"schema_version": "guarded_question_controller_v2", "action": action,
                "batch": str(batch), "root_id": root, "check_only": check_only,
                "status": "BLOCKED_BEFORE_DOWNSTREAM", "downstream_invoked": False,
                "gate": {"pass": False, "errors": [{"code": "WRAPPER_RUNTIME:NOT_FROZEN_OR_LOADED_CODE_CHANGED", "detail": str(exc)}]},
                "semantic_quality_certified": False}, 2
    outcome = check(batch, root, PHASES[action])
    result = {"schema_version": "guarded_question_controller_v2", "action": action,
              "batch": str(batch), "root_id": root, "check_only": check_only,
              "gate": outcome, "downstream_invoked": False,
              "frozen_controller_modified": False, "existing_records_repaired": False,
              "semantic_quality_certified": False}
    if not outcome["pass"]:
        result["status"] = "BLOCKED_BEFORE_DOWNSTREAM"
        return result, 2
    if check_only:
        result["status"] = "PROVENANCE_CHECK_ONLY_PASS"
        return result, 0
    unit = batch / "units" / ("root" + root.rsplit("-", 1)[-1])
    cfg = json.loads((unit / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    configured = [row["root_id"] if isinstance(row, dict) else row for row in cfg["roots"]]
    if configured != [root]:
        result["status"] = "BLOCKED_BEFORE_DOWNSTREAM"
        result["controller_error"] = "UNIT_MUST_CONTAIN_EXACTLY_REQUESTED_ROOT"
        return result, 2
    result["run_config_binding"] = binding(unit / "RUN_CONFIG.json")
    call = downstream or invoke_frozen_controller
    completed = call(batch, unit, action)
    result.update(status="GUARDED_STAGE_COMPLETED" if completed.returncode == 0 else "GUARDED_STAGE_FAILED",
                  downstream_invoked=True, downstream_returncode=completed.returncode,
                  downstream_stdout=completed.stdout, downstream_stderr=completed.stderr)
    return result, 0 if completed.returncode == 0 else 2


def main(argv=None):
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("action", choices=tuple(PHASES))
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result, code = run(args.batch, args.root, args.action, check_only=args.check_only)
    except Exception as exc:
        result, code = {"schema_version": "guarded_question_controller_v2", "status": "GUARD_OR_DOWNSTREAM_ERROR",
                        "error": type(exc).__name__ + ": " + str(exc), "pass": False}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
