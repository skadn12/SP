"""Canonical entry to v76 after the v77 full-source consistency review.

Legacy v76 files remain immutable. This wrapper binds the reviewed source
versions and rejects the current batch before preparing writers or calling any
transport when a source is on hold. It does not silently trim held scenes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import source_gate as gate

HERE = Path(__file__).resolve().parent
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
ACTIONS = ('prepare', 'generate', 'verify', 'finish-asset-review', 'finalize-multiselect',
           'finish-multiselect-review', 'finish-ordering-review', 'freeze',
           'prepare-evaluation', 'live-trained', 'live-base', 'report')


def preflight(home=HERE):
    home = Path(home).resolve(); summary = gate.verify_frozen(home); value = gate.config(home)
    upstream = gate.read(value['source_config']['path'])
    return {'status': 'BLOCKED_SOURCE_CONSISTENCY' if summary['held_roots'] else 'SOURCE_GATE_READY',
            'ready_roots': summary['ready_roots'], 'held_roots': summary['held_roots'],
            'frozen_review': gate.bind(home / 'CONSISTENCY_REVIEW.COMPLETE.json'),
            'configured_candidate_capacity': upstream['max_candidates'],
            'ready_root_capacity_if_separately_configured': {'multi_select': 5*len(summary['ready_roots']),
                'ordering': 2*len(summary['ready_roots']), 'new_candidates': 6*len(summary['ready_roots']),
                'historical_controls': len(summary['ready_roots'])},
            'subset_execution_configured': False, 'source_edits': [], 'automatic_source_repair': False,
            'semantic_question_review_still_required': True}


def launch(action, home=HERE, run=None, slot=None, round_id='initial'):
    home = Path(home).resolve(); gate.need(action in ACTIONS, 'unknown_batch_action')
    gate.require_ready(home)  # Before mkdir, CONFIG copying, import, Popen or STARTED.
    value = gate.config(home); engine = Path(value['upstream_engine'])
    run = Path(run).resolve() if run is not None else home / 'batch'
    gate.need(run != engine and not run.is_relative_to(engine) and run != home, 'legacy_or_audit_run_directory_forbidden')
    if action == 'prepare':
        gate.need(not run.exists(), 'batch_directory_already_exists')
        run.mkdir(parents=True)
        with (run / 'CONFIG.json').open('xb') as stream:
            stream.write(Path(value['source_config']['path']).read_bytes())
        gate.write(run / 'SOURCE_CONSISTENCY_LOCK.json', {'review': gate.bind(home / 'CONSISTENCY_REVIEW.COMPLETE.json'),
                                                       'source_config': value['source_config']})
    else:
        lock = gate.read(run / 'SOURCE_CONSISTENCY_LOCK.json')
        gate.need(lock == {'review': gate.bind(home / 'CONSISTENCY_REVIEW.COMPLETE.json'),
                           'source_config': value['source_config']}, 'batch_source_lock_changed')
    gate.need(gate.sha(run / 'CONFIG.json') == value['source_config']['sha256'], 'batch_config_changed')
    gate.require_ready(home)
    cmd = [str(PYTHON), '-B', str(engine / 'batch.py'), action, '--home', str(run)]
    if slot is not None:
        cmd += ['--slot', slot]
    cmd += ['--round', round_id]
    done = subprocess.run(cmd, cwd=run)
    gate.need(done.returncode == 0, 'upstream_batch_action_failed')
    return {'status': 'COMPLETE', 'action': action, 'run': str(run)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', *ACTIONS))
    parser.add_argument('--home', type=Path, default=HERE)
    parser.add_argument('--run', type=Path); parser.add_argument('--slot')
    parser.add_argument('--round', choices=('initial', 'supplement'), default='initial')
    args = parser.parse_args()
    try:
        result = preflight(args.home) if args.action == 'preflight' else launch(args.action, args.home, args.run, args.slot, args.round)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'BLOCKED', 'error': str(error), 'automatic_retry': False}, ensure_ascii=False))
        sys.exit(2)
