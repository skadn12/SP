"""Use the existing B0 probe with balanced displays for six options/events."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "h4_event_ledger_m0_pilot_v1" / "run.py"
SOURCE = HERE.parent / "h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json"
SOURCE_SHA = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
ORDERING_BASE = ("412356", "345162", "256431", "624513", "531624", "163245")

spec = importlib.util.spec_from_file_location("base_probe", BASE)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def balanced_orders(candidate, protocol):
    if protocol != "balanced":
        raise ValueError("balanced_protocol_required")
    field = "items" if candidate["question_type"] == "ordering" else "options"
    keys = list(candidate[field])
    expected = 6
    if len(keys) != expected:
        raise ValueError(f"expected_{expected}_{field}")
    if field == "items":
        # Six distinct adjacency patterns plus reverses balance positions and
        # pair direction while covering all 15 unordered adjacent pairs.
        base_orders = [[keys[int(char) - 1] for char in order] for order in ORDERING_BASE]
        orders = base_orders + [list(reversed(order)) for order in base_orders]
    else:
        orders = [keys[shift:] + keys[:shift] for shift in range(expected)]
        reverse = list(reversed(keys))
        orders += [reverse[shift:] + reverse[:shift] for shift in range(expected)]
    return [(f"perm_{index:02d}", order) for index, order in enumerate(orders)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "live", "report"))
    args = parser.parse_args()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("frozen_source_drift")
    probe.OUT = HERE / "probe_v3"
    probe.RUN = probe.OUT / "run"
    probe.display_orders = balanced_orders
    if args.phase == "preflight":
        result = probe.preflight(HERE / "candidates.json", None, SOURCE, "balanced", 1)
    elif args.phase == "live":
        result = probe.live()
    else:
        result = probe.report()
    print(result)


if __name__ == "__main__":
    main()
