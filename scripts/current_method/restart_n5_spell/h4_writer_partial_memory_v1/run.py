"""One entry for local preparation/replay/mock and later explicitly bounded live Writer calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .prepare import DEFAULT_RUN, SOURCE, BUDGETS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'replay', 'preflight', 'mock', 'live'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--memory-findings', type=Path)
    parser.add_argument('--scope', choices=(*BUDGETS, 'writer_repair_comparison'))
    parser.add_argument('--experiment', choices=('memory', 'writer_repair_comparison'), default='memory')
    parser.add_argument('--authorize-primary', type=int)
    parser.add_argument('--authorize-repair-max', type=int)
    parser.add_argument('--authorize-http-max', type=int)
    parser.add_argument('--allow-unknown-service', action='store_true')
    args = parser.parse_args(argv)
    if args.scope == 'writer_repair_comparison':
        args.experiment = 'writer_repair_comparison'
    if args.experiment == 'writer_repair_comparison' and args.scope in BUDGETS:
        parser.error('writer_repair_comparison cannot use a Memory scope')
    if args.experiment == 'writer_repair_comparison' and args.phase == 'replay':
        parser.error('repair comparison requires fresh paired outputs; historical replay is not an effect estimate')
    if args.run is None:
        if args.experiment == 'writer_repair_comparison' or args.scope == 'writer_repair_comparison':
            from .repair_trial import DEFAULT_RUN as TRIAL_RUN
            args.run = TRIAL_RUN
        else:
            args.run = DEFAULT_RUN
    if args.phase == 'live':
        if not args.scope:
            parser.error('live requires a specific --scope')
        if args.scope == 'writer_repair_comparison':
            from .repair_trial import BUDGET
            b = BUDGET
        else:
            b = BUDGETS[args.scope]
        if (args.authorize_primary, args.authorize_repair_max, args.authorize_http_max) != (b['primary'], b['repair_max'], b['http_max']):
            parser.error('live requires exactly the frozen primary/repair/HTTP budget for this scope')
        from .runtime import execute
        result = execute(args.run, args.scope, authorized=True, allow_unknown_service=args.allow_unknown_service)
    else:
        from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
        with forbid_network():
            if args.phase == 'prepare':
                if args.experiment == 'writer_repair_comparison':
                    from .repair_trial import prepare, SOURCE as TRIAL_SOURCE
                    result = prepare(args.run, source=TRIAL_SOURCE if args.source == SOURCE else args.source)
                else:
                    from .prepare import prepare
                    result = prepare(args.run, source=args.source, findings=args.memory_findings)
            elif args.phase == 'replay':
                from .replay import replay
                result = replay(args.source, args.run / 'offline_replay')
            elif args.phase == 'mock':
                if args.experiment == 'writer_repair_comparison':
                    from .repair_trial import mock as exercise
                else:
                    from .mock import exercise
                result = exercise(args.run)
            else:
                from .prepare import verify
                result = verify(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
