#!/usr/bin/env python3
"""Read-only OPP16 postflight; invoke only after the selector has terminated."""
import audit_opp15_selector_result as audit

audit.RECEIPT = audit.OLD / "continuation_opp16_selector_controller_v1/CONTROLLER_RECEIPT.json"

if __name__ == "__main__":
    raise SystemExit(audit.main())
