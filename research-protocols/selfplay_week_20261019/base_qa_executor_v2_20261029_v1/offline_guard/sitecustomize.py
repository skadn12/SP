"""Fail-closed socket/DNS guard inherited by offline Python test children.

No connection target, credential or payload is recorded. A blocked attempt is
still a failed offline check, even if an individual test swallows its exception.
"""
import json
import os
import sys


def audit(event, args):
    if event.startswith("socket."):
        path = os.environ.get("QA_OFFLINE_NETWORK_AUDIT")
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": event, "pid": os.getpid(), "blocked": True}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        raise RuntimeError("OFFLINE_CHECK_NETWORK_FORBIDDEN")


sys.addaudithook(audit)
