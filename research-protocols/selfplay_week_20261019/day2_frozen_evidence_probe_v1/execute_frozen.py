"""Reuse the sealed native direct runtime; do not invoke its mixed-type scorer."""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'day2/solver'))
import run as rt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['verify', 'live'])
    p.add_argument('--run', type=Path, default=HERE / 'run')
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'verify':
        rt.verify(args.run, args.audit)
        print('PASS: frozen-citation paired inputs, model and independent preflight')
    else:
        rt.live(args.run, args.audit)


if __name__ == '__main__':
    main()
