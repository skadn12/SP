"""One entry for preparation, local checks, explicitly budgeted live generation and reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .constants import DEFAULT_RUN, STRONG_CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "mock", "live", "report", "configure-key"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--strong-config", type=Path, default=STRONG_CONFIG)
    parser.add_argument("--authorize-primary", type=int)
    parser.add_argument("--authorize-repair-max", type=int)
    parser.add_argument("--authorize-http-max", type=int)
    args = parser.parse_args(argv)
    if args.phase == "live":
        if (args.authorize_primary, args.authorize_repair_max, args.authorize_http_max) != (12, 12, 24):
            parser.error("live requires explicit --authorize-primary 12 --authorize-repair-max 12 --authorize-http-max 24")
        from .runtime import execute_live
        result = execute_live(args.run, authorized=True)
    else:
        from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
        with forbid_network():
            if args.phase == "configure-key":
                from .credentials import configure_key
                result = configure_key(args.strong_config)
            elif args.phase == "prepare":
                from .prepare import prepare
                result = prepare(args.run, strong_config=args.strong_config)
            elif args.phase == "preflight":
                from .prepare import preflight
                result = preflight(args.run)
            elif args.phase == "mock":
                from .mock import exercise
                result = exercise(args.run)
            else:
                from .reporting import report
                result = report(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
