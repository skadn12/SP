"""Run the R03 Novel Question Capacity Preflight without online calls."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path

from . import DEFAULT_OUTPUT
from .capacity import build_all, digest_file, load_inputs, write_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    built = build_all(inputs)

    from . import test_run

    test_functions = [
        function for name, function in inspect.getmembers(test_run, inspect.isfunction)
        if name.startswith("test_")
    ]
    failures = []
    for function in test_functions:
        try:
            function()
        except Exception as exc:  # pragma: no cover - failure path
            failures.append({"test": function.__name__, "error": repr(exc)})
    built["artifacts"]["offline_test_report.json"] = {
        "status": "PASS" if not failures else "FAIL",
        "pytest_available": importlib.util.find_spec("pytest") is not None,
        "direct_test_function_count": len(test_functions),
        "direct_test_function_pass_count": len(test_functions) - len(failures),
        "failures": failures,
    }
    if failures:
        built["artifacts"]["final_r03_novel_question_capacity_preflight.json"]["valid"] = False

    # Source SHA checks are emitted after all measurements and before output
    # writing.  Output files live in a versioned directory and are excluded.
    source_audit = []
    for path_text, before in sorted(inputs["source_hashes"].items()):
        path = Path(path_text)
        after = digest_file(path)
        source_audit.append({"path": path_text, "sha256_before": before, "sha256_after": after, "unchanged": before == after})
    built["artifacts"]["source_sha_audit.json"] = {
        "status": "PASS" if all(row["unchanged"] for row in source_audit) else "FAIL",
        "source_file_count": len(source_audit),
        "source_manifest_modification_count": sum(not row["unchanged"] for row in source_audit),
        "files": source_audit,
    }
    report = built["artifacts"]["final_r03_novel_question_capacity_preflight.json"]
    report["offline_tests_pass"] = not failures
    report["source_sha_audit_pass"] = all(row["unchanged"] for row in source_audit)
    report["valid"] = bool(report.get("valid")) and not failures and report["source_sha_audit_pass"]

    for name, value in built["artifacts"].items():
        write_artifact(output, name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

