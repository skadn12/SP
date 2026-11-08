"""One newly authorized synthetic64 probe via the frozen proxy-only adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from scripts.current_method import base_qa_capability_probe_v1 as p
from scripts.current_method import base_qa_proxy_probe_transport_v1 as proxy

VERSION = "base_qa_proxy_probe_live_20261030_v1"
PREP = HERE.parent / "base_qa_proxy_probe_prep_20261030_v1"


def preflight():
    plan = p.read(HERE / "AUTHORIZATION_AND_PLAN.json")
    freeze = p.read(HERE / "PLAN_FREEZE.json")
    p.require(p.file_binding(HERE / "AUTHORIZATION_AND_PLAN.json") == freeze["plan_binding"], "AUTHORIZATION_PLAN_DRIFT")
    p.require(plan["version"] == VERSION and plan["authorized_attempts"] == 1 and
              plan["max_tokens"] == 64 and plan["qa_authorized"] is False and
              plan["automatic_retries"] == 0 and plan["direct_fallback"] is False and
              plan["purpose"] == proxy.PURPOSE and plan["prior_attempts"] == 2 and
              plan["expected_remaining_after_attempt"] == 9, "AUTHORIZATION_SCOPE_DRIFT")
    for binding in plan["bindings"]:
        p.require(p.file_binding(binding["path"]) == binding, "FROZEN_BINDING_DRIFT")
    for row in p.read(PREP / "DELIVERY_MANIFEST.json")["files"]:
        f = p.PROJECT / row["path"]
        p.require(f.is_file() and f.stat().st_size == row["bytes"] and
                  hashlib.sha256(f.read_bytes()).hexdigest() == row["sha256"], "FROZEN_PROXY_PREP_DRIFT")
    experiment = p.verify_experiment(p.ARTIFACTS, p.CANONICAL)
    p.require(plan["credential_file"] == experiment["credential_file"], "CREDENTIAL_PATH_DRIFT")
    p.require(p.read(HERE / "REQUEST.json") == p.body(), "PROBE_REQUEST_DRIFT")
    proxy.validate_proxy_environment()
    proxy.validate_canonical_state(p.CANONICAL / p.LEDGER_NAME)
    return plan


def run(*, live=False, transport=None):
    plan = preflight()
    if not live:
        return {"status": "DRY_RUN_ONLY", "new_api_attempts": 0,
                "prior_attempts": 2, "remaining_now": 10,
                "remaining_after_one_authorized_attempt": 9, "qa_authorized": False}
    if transport is None:
        p.require(HERE == p.ARTIFACTS.parent / VERSION, "LIVE_PATH_DRIFT")
        transport = proxy.ProxyProbeTransport(plan["credential_file"], explicitly_live=True)
    else:
        p.require(getattr(transport, "is_mock", False) is True, "ONLY_MOCK_CAN_REPLACE_TRANSPORT")
    ledger = p.ProbeBudget(p.CANONICAL, p.file_binding(p.ARTIFACTS / "EXPERIMENT_PROBE.json"))
    try:
        p.require(ledger.snapshot()["attempts_consumed"] == 2, "UNEXPECTED_PRIOR_BUDGET_STOP")
        reservation = proxy.reserve_once(ledger, plan["prior_snapshot_sha256"], p.body())
        observation = None
        started = time.monotonic()
        try:
            p.durable_write(HERE / "PRE_SEND_RESERVATION.json", {
                "attempt_id": reservation.attempt_id, "budget": ledger.snapshot(),
                "authorization_binding": p.file_binding(HERE / "AUTHORIZATION_AND_PLAN.json"),
                "request_sha256": p.digest(p.body()), "proxy": proxy.validate_proxy_environment(),
                "reserved_before_any_connection": True})
            envelope = transport.send(p.body(), reservation=reservation)
            p.durable_write(HERE / "HTTP_RESPONSE_BODY.bin", envelope["raw"], binary=True)
            p.durable_write(HERE / "HTTP_RECEIPT.json", {k: v for k, v in envelope.items() if k != "raw"})
            observation = p.response_observation(envelope)
            p.durable_write(HERE / "RESPONSE_OBSERVATION.json", {
                k: v for k, v in observation.items() if k != "_data"})
            validated = p.validate_observation(observation)
            result = {"status": "PROBE_RESPONSE_VALID_QA_HOLD", **validated,
                      "trace": transport.trace, "returned_model": observation["returned_model"],
                      "finish_reason": observation["finish_reason"], "usage": observation["usage"],
                      "confirmation": observation["content"], "qa_authorized": False}
            settlement = "SUCCEEDED"
        except BaseException as error:
            result = {"status": "HOLD_NO_AUTOMATIC_RETRY", "error_type": type(error).__name__,
                      "reason_code": str(error) if isinstance(error, p.ProbeHold) else type(error).__name__,
                      "trace": transport.trace, "qa_authorized": False,
                      "returned_model": observation["returned_model"] if observation else None,
                      "finish_reason": observation["finish_reason"] if observation else None,
                      "usage": observation["usage"] if observation else None}
            settlement = "REJECTED" if observation else "UNCERTAIN"
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["new_attempts_consumed"] = 1
        result["previous_attempt_usage_still_unknown"] = True
        ledger.settle(reservation.attempt_id, settlement, result)
        result["budget"] = ledger.snapshot()
        p.durable_write(HERE / "RESULT.json", result)
        p.durable_write(HERE / "GLOBAL_BUDGET_AFTER.json", ledger.snapshot())
        return result
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-once", action="store_true")
    args = parser.parse_args()
    try:
        result = run(live=args.live_once)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] in ("DRY_RUN_ONLY", "PROBE_RESPONSE_VALID_QA_HOLD") else 2
    except BaseException as error:
        print(json.dumps({"status": "HOLD_NO_AUTOMATIC_RETRY", "error_type": type(error).__name__,
                          "reason": "No further request is allowed; inspect the existing durable ledger."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
