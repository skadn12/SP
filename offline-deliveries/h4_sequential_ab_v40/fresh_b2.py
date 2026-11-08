"""Two genuinely unused B sources after exact accepted A3, with no QA admission.

Selection uses the existing deterministic metadata-stratified order and permanent
source reservations. A failed source remains consumed. This entrypoint adds no
semantic retry, source replacement or prior experimental B dialogue.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import runner as r

ENTRYPOINT = "fresh_b2.py"
SCENES = 2
LOGICAL_MAX = 34
HTTP_MAX = 68


def prepare(run, parent_a3, source_dir):
    run, parent_a3, source_dir = map(lambda p: Path(p).resolve(), (run, parent_a3, source_dir))
    r.base.require(source_dir.is_dir(), "replenished_source_directory_missing")
    select, writer = r.source_pool.freeze_unused_source_subset, r.base.write

    def frozen_selection(consumed, count):
        r.base.require(count == SCENES, "B2_source_selection_count_changed")
        return select(consumed, count, source_dir=source_dir)

    def bound_writer(path, value):
        if Path(path) == run / "MANIFEST.json":
            value = deepcopy(value)
            r.base.require(value["logical_max"] == LOGICAL_MAX and value["http_max"] == HTTP_MAX,
                           "B2_budget_changed")
            value.update(execution_entrypoint=ENTRYPOINT,
                         source_pool_directory=str(source_dir),
                         source_pool_policy="first_two_unused_metadata_stratified_events_no_replacement")
        return writer(path, value)

    with patch.object(r.source_pool, "freeze_unused_source_subset", side_effect=frozen_selection), \
         patch.object(r.base, "write", side_effect=bound_writer):
        return r.prepare(run, scenes=SCENES, parent_a3=parent_a3)


def verify(run):
    run = Path(run).resolve()
    manifest, allocation = r.verify(run)
    r.base.require(manifest.get("execution_entrypoint") == ENTRYPOINT, "B2_entrypoint_changed")
    r.base.require(manifest["roots"] == list(r.DEFAULT_ROOTS)
                   and manifest["selected_routes"] == [r.ROUTES[1]]
                   and manifest["scenes_per_route_target"] == SCENES, "B2_scope_changed")
    r.base.require(manifest["inherited_A"]["accepted_scenes"] == 3, "B2_requires_actual_A3")
    r.base.require(manifest.get("logical_max") == LOGICAL_MAX
                   and manifest.get("http_max") == HTTP_MAX, "B2_budget_changed")
    r.base.require(manifest.get("source_pool_policy") ==
                   "first_two_unused_metadata_stratified_events_no_replacement", "B2_source_policy_changed")
    consumed = {r.source_inventory.event_key(row) for row in allocation["prior"]["accepted_uses"]}
    expected = r.source_pool.freeze_unused_source_subset(consumed, SCENES,
        source_dir=Path(manifest["source_pool_directory"]))
    r.base.require(expected == r.base.read(run / "dataset_sources.json"), "B2_source_selection_replay_changed")
    r.base.require(all(not turn["turn_ref"].startswith("R02:B:")
                      for turn in r.route_base_history(run, manifest["roots"][0], r.ROUTES[1])),
                   "B2_parent_contains_prior_B")
    return manifest


def live(run):
    verify(run)
    return r.live(Path(run).resolve())


def audit(run):
    verify(run)
    import audit_sequential
    return audit_sequential.audit_run(Path(run).resolve(), count_tokens=True)


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "live", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-a3", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.parent_a3 or not args.source_dir:
            parser.error("prepare requires --parent-a3 and --source-dir")
        result = prepare(args.run, args.parent_a3, args.source_dir)
    else:
        result = globals()[args.command](args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "audit" and result["status"] != "PASS":
        raise SystemExit(1)
