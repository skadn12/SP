#!/usr/bin/env python3
"""Reuse the OPP15 read-only selector preflight for OPP16."""
import preflight_opp15_selector as preflight

preflight.RECEIPT = preflight.OLD / "continuation_opp16_selector_controller_v1/CONTROLLER_RECEIPT.json"
preflight.STAGE = "OPP16_SELECTOR_PACKET_INTEGRITY_ONLY"

if __name__ == "__main__":
    raise SystemExit(preflight.main())
