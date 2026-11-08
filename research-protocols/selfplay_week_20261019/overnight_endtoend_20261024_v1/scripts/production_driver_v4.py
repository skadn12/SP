#!/usr/bin/env python3
"""Frozen v3 mechanics plus immutable per-invocation wave terminal receipts.

Old wave_results are snapshots and are never rewritten. All author/reviewer
prompts, single semantic attempts, native evidence checks and three-worker
limits are inherited unchanged. Importing this module does not start a driver.
"""
import concurrent.futures
import importlib.util
from pathlib import Path
import traceback
import uuid

BASE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


version3 = module('oet1_frozen_driver_v3', BASE / 'scripts/production_driver_v3.py')
driver = version3.driver


def run_wave(roots, label):
    invocation_id = driver.now().replace(':', '').replace('.', '') + '_' + uuid.uuid4().hex[:12]
    out = BASE / 'wave_results_v4' / label / invocation_id
    started = out / 'WAVE_START.json'
    driver.write(started, {
        'schema_version': 'oet1_wave_invocation_v4',
        'started_at_utc': driver.now(), 'label': label,
        'roots': list(roots), 'arms': ['GENERIC', 'TARGETED'],
        'max_workers': 3, 'entrypoint': driver.bind(Path(__file__)),
        'prior_wave_results_preserved': True,
    })
    driver.log('WAVE_STARTED_V4', label=label, roots=roots, invocation_id=invocation_id)
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(driver.unit, root, arm): (root, arm)
                   for root in roots for arm in ['GENERIC', 'TARGETED']}
        while futures:
            done, _ = concurrent.futures.wait(
                futures, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                root, arm = futures.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {'root': root, 'arm': arm, 'error': str(error),
                              'traceback': traceback.format_exc()}
                previous = BASE / 'wave_results' / f'{root}__{arm}.json'
                terminal = out / f'{root}__{arm}.json'
                driver.write(terminal, {
                    'schema_version': 'oet1_wave_unit_terminal_v4',
                    'finished_at_utc': driver.now(), 'root': root, 'arm': arm,
                    'invocation_start': driver.bind(started), 'result': result,
                    'prior_wave_snapshot': driver.bind(previous) if previous.exists() else None,
                    'prior_wave_snapshot_rewritten': False,
                })
                receipts.append(driver.bind(terminal))
                driver.log('UNIT_TERMINAL_V4', terminal=driver.bind(terminal), result=result)
            driver.status(heartbeat={
                'at_utc': driver.now(), 'active_units': [list(x) for x in futures.values()],
                'label': label, 'driver': 'v4', 'invocation_id': invocation_id,
            })
    complete = out / 'WAVE_TERMINAL.json'
    driver.write(complete, {
        'schema_version': 'oet1_wave_terminal_v4', 'finished_at_utc': driver.now(),
        'invocation_start': driver.bind(started), 'unit_receipts': receipts,
        'expected_units': len(roots) * 2, 'completed_units': len(receipts),
        'all_scheduled_units_returned': len(receipts) == len(roots) * 2,
    })
    driver.log('WAVE_TERMINAL_V4', label=label, terminal=driver.bind(complete))


def main():
    driver.run_wave = run_wave
    version3.main()


if __name__ == '__main__':
    main()
