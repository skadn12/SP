"""Execute the common QA feedback template preflight offline."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path

from . import DEFAULT_OUTPUT
from .preflight import build_all, sha_file, write_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=True)
    from .preflight import load_inputs
    inputs = load_inputs()
    built = build_all(inputs)
    from . import test_run
    funcs = [fn for name, fn in inspect.getmembers(test_run, inspect.isfunction) if name.startswith("test_")]
    failures = []
    for fn in funcs:
        try:
            fn()
        except Exception as exc:
            failures.append({"test": fn.__name__, "error": repr(exc)})
    built["artifacts"]["offline_test_report.json"] = {"status": "PASS" if not failures else "FAIL", "pytest_available": importlib.util.find_spec("pytest") is not None, "direct_test_function_count": len(funcs), "direct_test_function_pass_count": len(funcs) - len(failures), "failures": failures}
    source_rows = []
    for path_text, before in sorted(inputs["source_hashes"].items()):
        path = Path(path_text); after = sha_file(path)
        source_rows.append({"path": path_text, "sha256_before": before, "sha256_after": after, "unchanged": before == after})
    built["artifacts"]["input_source_sha_manifest.json"] = {"status": "PASS" if all(row["unchanged"] for row in source_rows) else "FAIL", "source_file_count": len(source_rows), "files": source_rows}
    report = built["artifacts"]["feedback_template_preflight_report.json"]
    report["offline_tests_pass"] = not failures
    report["source_files_unchanged"] = all(row["unchanged"] for row in source_rows)
    report["valid"] = bool(report.get("valid")) and not failures and report["source_files_unchanged"]
    built["artifacts"]["feedback_template_preflight_report.md"] = (
        built["artifacts"]["feedback_template_preflight_report.md"].rstrip()
        + "\n"
        + f"- Offline regression tests: `{report['offline_tests_pass']}` ({len(funcs) - len(failures)}/{len(funcs)} direct tests)."
        + f" Source files unchanged: `{report['source_files_unchanged']}`; final valid: `{report['valid']}`.\n"
    )
    for name, value in built["artifacts"].items():
        write_artifact(output, name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
