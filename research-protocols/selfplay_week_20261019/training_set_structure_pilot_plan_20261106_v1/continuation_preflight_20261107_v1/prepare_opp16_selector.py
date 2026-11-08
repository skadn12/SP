#!/usr/bin/env python3
"""Reuse the OPP15 selector packet preparation for OPP16; do not invoke roles."""
import prepare_opp15_selector as prepare

prepare.CODE = "OPP27_16_9d3abd2fbe"
prepare.AUTHOR = prepare.OLD / "role_packets" / prepare.CODE / "author_output"
prepare.AUTHOR_PACKET = prepare.OLD / "continuation_opp16_effective_packet_v1"
prepare.PRIVATE = prepare.OLD / "continuation_opp16_selector_controller_v1"

if __name__ == "__main__":
    prepare.main()
