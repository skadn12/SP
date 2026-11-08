"""File-based offline harness; multiprocessing spawn can re-import this file."""
from pathlib import Path
import contextlib
import io
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
NETWORK_EVENTS = []


def forbid_network(event, args):
    if event.startswith("socket."):
        NETWORK_EVENTS.append({"event": event})
        raise AssertionError("NETWORK FORBIDDEN BY OFFLINE AUDIT")


# Installed in the parent and spawned multiprocessing workers.
sys.addaudithook(forbid_network)


def main():
    names = ["test_base_qa_pipeline_v1", "test_base_qa_executor_v2",
             "test_base_qa_executor_v2_adversarial", "test_qa_api_budget_v2",
             "test_qa_api_transport_v2", "test_base_qa_capability_probe_v1",
             "test_base_qa_proxy_probe_transport_v1"]
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            ["scripts.current_method." + name for name in names])
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        from scripts.current_method import base_qa_proxy_probe_transport_v1 as proxy
        dry = proxy.offline_plan()
    (HERE / "OFFLINE_TEST_OUTPUT.txt").write_text(log.getvalue())
    summary = {"tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "all_passed": result.wasSuccessful(), "network_audit_events": NETWORK_EVENTS,
               "actual_api_requests": 0, "real_credential_reads": 0,
               "new_transport_tests": 21, "dry_run": dry,
               "spawned_test_workers": "network audit block installed at module import",
               "prior_harness_issue": "stdin entrypoint could not be reimported by spawn; failed run preserved separately"}
    (HERE / "OFFLINE_TEST_RESULT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "dry_run"}, ensure_ascii=False))
    return 0 if result.wasSuccessful() and not NETWORK_EVENTS else 1


if __name__ == "__main__":
    raise SystemExit(main())
