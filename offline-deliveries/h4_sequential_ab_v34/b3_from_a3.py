"""B3 on audited A3 with an explicitly frozen replenished public source pool.

Only offline source selection is adapted. Live stages and independent replay
use the existing frozen v34 runner and auditor without semantic overrides.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import runner as r

ENTRYPOINT = "b3_from_a3.py"


def prepare(run, parent_a3, source_dir):
    source_dir = Path(source_dir).resolve()
    r.base.require(source_dir.is_dir(), "replenished_source_directory_missing")
    select = r.source_pool.freeze_unused_source_subset
    writer = r.base.write
    def frozen_source_selection(consumed, count):
        return select(consumed, count, source_dir=source_dir)
    def write_with_entrypoint(path, value):
        if Path(path) == run / "MANIFEST.json":
            value = deepcopy(value)
            value.update(execution_entrypoint=ENTRYPOINT,
                         source_pool_directory=str(source_dir),
                         source_pool_policy="new_public_train_events_with_existing_identity_exclusion_and_stratification")
        return writer(path, value)
    with patch.object(r.source_pool, "freeze_unused_source_subset", side_effect=frozen_source_selection), \
         patch.object(r.base, "write", side_effect=write_with_entrypoint):
        return r.prepare(run, scenes=3, parent_a3=parent_a3)


def verify(run):
    manifest, _allocation = r.verify(run)
    r.base.require(manifest.get("execution_entrypoint") == ENTRYPOINT,
                   "B3_replenished_entrypoint_missing")
    r.base.require(manifest["inherited_A"]["accepted_scenes"] == 3,
                   "B3_requires_actual_A3")
    return manifest


def live(run):
    verify(run)
    return r.live(run)


def audit(run):
    verify(run)
    import audit_sequential
    return audit_sequential.audit_run(run, count_tokens=True)


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "live", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-a3", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        parser.error("prepare requires --parent-a3 and --source-dir") if not args.parent_a3 or not args.source_dir else None
        result = prepare(args.run.resolve(), args.parent_a3.resolve(), args.source_dir.resolve())
    else:
        result = globals()[args.command](args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "audit" and result["status"] != "PASS":
        raise SystemExit(1)
