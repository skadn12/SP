"""One newly user-authorized manual retry, consuming the same global12 budget."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from scripts.current_method import base_qa_capability_probe_v1 as p

PURPOSE = "MANUAL_RETRY_20261030_V1"
VERSION = "base_qa_capability_probe_retry_20261030_v1"


def reserve(ledger, expected_snapshot_sha256):
    ledger.connection.execute("BEGIN IMMEDIATE")
    try:
        snapshot = ledger.snapshot()
        p.require(snapshot["attempts_consumed"] == 1 and snapshot["remaining_global_attempts"] == 11,
                  "Unexpected global budget; stop")
        p.require(p.digest(snapshot) == expected_snapshot_sha256, "Prior attempt changed; stop")
        prior = snapshot["attempts"][0]
        p.require(prior["purpose"] == p.PURPOSE and prior["status"] == "UNCERTAIN", "Unexpected prior state")
        p.require(ledger.connection.execute("SELECT COUNT(*) FROM attempts WHERE purpose=?", (PURPOSE,)).fetchone()[0] == 0,
                  "This manual retry was already reserved; no resend")
        cursor = ledger.connection.execute("INSERT INTO attempts(purpose,status,reserved_utc,request_sha256,max_tokens) VALUES (?,?,?,?,?)",
            (PURPOSE, "RESERVED", datetime.now(timezone.utc).isoformat(), p.digest(p.body()), 64))
        ledger.connection.execute("COMMIT")
        return cursor.lastrowid
    except BaseException:
        ledger.connection.execute("ROLLBACK")
        raise


class ObservedTransport:
    """Single HTTP attempt, same parameters/timeout; safe failure phase retained."""
    def __init__(self, credential_file, *, live=False):
        self.credential_file, self.used = credential_file, False
        self.live = live
        self.trace = {"phase": "NOT_STARTED", "underlying_error_type": None, "errno": None}

    def send(self):
        p.require(self.live is True, "Dry-run transport cannot send")
        p.require(not self.used, "Transport already used")
        self.used = True
        p.validate_body(p.body())
        connection = None
        try:
            self.trace["phase"] = "LOCAL_CREDENTIAL_READ"
            key = p.old_transport._read_credential(Path(self.credential_file))
            connection = http.client.HTTPSConnection("ai.tju.edu.cn", port=443, timeout=60)
            self.trace["phase"] = "CONNECT_OR_REQUEST_SEND"
            connection.request("POST", "/api/v3/chat/completions", body=p.canonical(p.body()).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
            self.trace["phase"] = "AWAIT_RESPONSE_HEADERS"
            response = connection.getresponse()
            self.trace["http_status"] = response.status
            self.trace["phase"] = "READ_RESPONSE_BODY"
            raw = response.read(1024 * 1024 + 1)
            p.require(len(raw) <= 1024 * 1024, "Response too large")
            p.require(key.encode() not in raw, "Credential echo withheld")
            decoded = p.old_transport._strict_json(raw)
            p.require(key not in p.canonical(decoded), "Escaped credential echo withheld")
            headers = {}
            for name in ("x-request-id", "request-id", "x-correlation-id"):
                value = response.getheader(name)
                p.require(not isinstance(value, str) or key not in value, "Credential header echo withheld")
                headers[name] = p.old_transport._safe_id(value)
            self.trace["phase"] = "RESPONSE_RECEIVED"
            return {"http_status": response.status, "headers": headers,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw": raw}
        except BaseException as error:
            self.trace["underlying_error_type"] = type(error).__name__
            number = getattr(error, "errno", None)
            self.trace["errno"] = number if type(number) is int else None
            # Never copy networking exception text: it may include secrets.
            raise
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass


def run(live=False):
    manifest = p.read(HERE / "AUTHORIZATION_AND_PLAN.json")
    for binding in manifest["bindings"]:
        p.require(p.file_binding(binding["path"]) == binding, "Frozen binding drift")
    p.require(manifest["version"] == VERSION and manifest["authorized_attempts"] == 1 and
              manifest["max_tokens"] == 64 and manifest["qa_authorized"] is False, "Scope mismatch")
    p.verify_experiment(p.ARTIFACTS, p.CANONICAL)
    if not live:
        return {"status": "DRY_RUN", "new_calls": 0, "after_retry_remaining": 10}
    p.require(HERE == p.ARTIFACTS.parent / VERSION, "Live paths are fixed")
    ledger = p.ProbeBudget(p.CANONICAL, p.file_binding(p.ARTIFACTS / "EXPERIMENT_PROBE.json"))
    try:
        attempt = reserve(ledger, manifest["prior_snapshot_sha256"])
        p.durable_write(HERE / "PRE_SEND_RESERVATION.json", {"attempt_id": attempt, "budget": ledger.snapshot(),
            "authorization_binding": p.file_binding(HERE / "AUTHORIZATION_AND_PLAN.json"), "request_sha256": p.digest(p.body())})
        transport = ObservedTransport(manifest["credential_file"], live=True)
        observation = None
        try:
            envelope = transport.send()
            p.durable_write(HERE / "HTTP_RESPONSE_BODY.bin", envelope["raw"], binary=True)
            p.durable_write(HERE / "HTTP_RECEIPT.json", {k: v for k, v in envelope.items() if k != "raw"})
            observation = p.response_observation(envelope)
            p.durable_write(HERE / "RESPONSE_OBSERVATION.json", {k: v for k, v in observation.items() if k != "_data"})
            validated = p.validate_observation(observation)
            result = {"status": "PROBE_RESPONSE_VALID_QA_HOLD", **validated, "trace": transport.trace,
                "returned_model": observation["returned_model"], "finish_reason": observation["finish_reason"],
                "usage": observation["usage"], "confirmation": observation["content"]}
            ledger.settle(attempt, "SUCCEEDED", result)
        except BaseException as error:
            result = {"status": "HOLD_NO_AUTOMATIC_RETRY", "error_type": type(error).__name__,
                "reason_code": str(error) if isinstance(error, p.ProbeHold) else type(error).__name__,
                "trace": transport.trace, "qa_authorized": False,
                "returned_model": observation["returned_model"] if observation else None,
                "finish_reason": observation["finish_reason"] if observation else None,
                "usage": observation["usage"] if observation else None}
            ledger.settle(attempt, "REJECTED" if observation else "UNCERTAIN", result)
        result["new_attempts_consumed"] = 1
        result["budget"] = ledger.snapshot()
        result["previous_attempt_usage_still_unknown"] = True
        p.durable_write(HERE / "RESULT.json", result)
        return result
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-once", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.live_once)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] in ("DRY_RUN", "PROBE_RESPONSE_VALID_QA_HOLD") else 2
    except Exception as error:
        print(json.dumps({"status": "HOLD_NO_AUTOMATIC_RETRY", "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
