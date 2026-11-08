"""Build matched immutable Round-0 B0 training and evaluation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import (
    DEFAULT_BLACKLIST,
    DEFAULT_DIFFICULTY_RUN,
    DEFAULT_RUN_ROOT,
    DEFAULT_STRATIFICATION_RUN,
    DEFAULT_TOKENIZER,
)
from .builder import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty-run", type=Path, default=DEFAULT_DIFFICULTY_RUN)
    parser.add_argument("--stratification-run", type=Path, default=DEFAULT_STRATIFICATION_RUN)
    parser.add_argument("--confirmed-invalid-blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args(argv)
    report = build(
        difficulty_run=args.difficulty_run,
        stratification_run=args.stratification_run,
        blacklist_path=args.confirmed_invalid_blacklist,
        tokenizer_path=args.tokenizer,
        run_root=args.run_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
