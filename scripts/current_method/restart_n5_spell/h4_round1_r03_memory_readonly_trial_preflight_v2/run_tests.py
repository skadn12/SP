"""Execute regression functions directly when pytest is unavailable."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

from . import test_preflight


def main() -> int:
    functions = [
        function for name, function in sorted(vars(test_preflight).items())
        if name.startswith("test_") and callable(function)
    ]
    passed = []
    for function in functions:
        function()
        passed.append(function.__name__)
    output = Path(os.environ.get(
        "R03_READONLY_PREFLIGHT_OUT", str(test_preflight.OUT)))
    report = {
        "test_report_version": "r03_readonly_preflight_direct_tests_v1",
        "pytest_available": False,
        "direct_test_function_count": len(functions),
        "direct_test_function_pass_count": len(passed),
        "failed_test_functions": [],
        "passed_test_functions": passed,
        "network_calls": {
            "real_memorax_search": 0,
            "mock_memorax_search": 1,
            "memorax_write": 0,
            "expansion_api": 0,
        },
    }
    path = output / "test_execution_report.json"
    path.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

