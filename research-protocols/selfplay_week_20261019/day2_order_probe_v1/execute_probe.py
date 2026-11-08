"""Use the sealed runtime and its verified backend-schema compatibility."""
from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
DAY2 = HERE.parent / 'day2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(DAY2))
import evidence_runtime_v2 as compat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'live'])
    parser.add_argument('--run', type=Path, default=HERE / 'run')
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        compat.runtime.verify(args.run, args.audit)
        print('PASS: bound model, sources, schema compatibility, paired requests and audit')
    else:
        compat.runtime.live(args.run, args.audit)


if __name__ == '__main__':
    main()
