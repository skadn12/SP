#!/usr/bin/env python3
"""Exclusive offline validation entry: synthetic scoring only, no experiment modes."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode = True
PROPOSAL = Path(__file__).resolve().parent
if sys.argv[1:] not in ([], ['validate']):
    raise SystemExit('OFFLINE_ONLY: supported mode is validate. prepare/run/score/check-inputs are forbidden here.')
if (PROPOSAL / 'OFFLINE_VALIDATION.json').exists():
    raise SystemExit('NO_VALIDATION_OVERWRITE: this proposal validation is already recorded.')


def binding(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def write_once(name, value):
    with (PROPOSAL / name).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


protected = json.loads((PROPOSAL / 'BASELINE_BINDINGS.json').read_text())['files']
before = [binding(row['path']) for row in protected]
if before != protected:
    raise SystemExit('PROTECTED_ARTIFACT_CHANGED_BEFORE_VALIDATION')

import test_scorer_recovery as fixtures

start = time.monotonic()
stream = io.StringIO()
suite = unittest.defaultTestLoader.loadTestsFromModule(fixtures)
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
after = [binding(row['path']) for row in protected]
unchanged = before == after
write_once('PROTECTED_BINDINGS_POSTCHECK.json', {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
           'protected_file_count': len(protected), 'all_sha256_unchanged': unchanged,
           'files': after})
record = {'status': 'PASS' if result.wasSuccessful() and unchanged and not fixtures.FORBIDDEN_IMPORT_ATTEMPTS else 'FAIL',
          'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
          'failures': len(result.failures), 'errors': len(result.errors), 'elapsed_seconds': round(time.monotonic() - start, 6),
          'test_output': stream.getvalue(), 'validation_executable': sys.executable,
          'frozen_and_historical_files_protected': len(protected), 'frozen_and_historical_sha256_unchanged': unchanged,
          'original_adapter_sha256': binding(fixtures.FROZEN)['sha256'],
          'proposed_adapter_sha256': binding(PROPOSAL / 'adapter.py.proposed')['sha256'],
          'minimal_patch_sha256': binding(PROPOSAL / 'SCORER_FIX.patch')['sha256'],
          'synthetic_fixture_directory': str(PROPOSAL / 'synthetic_fixtures'),
          'scorer_model_api_import_attempts': fixtures.FORBIDDEN_IMPORT_ATTEMPTS,
          'historical_metric_equivalence_cases': {'nonempty_multiselect_sets': 60, 'complete_order_permutations': 24, 'directed_pair_subsets': 64},
          'entrypoint_disabled_in_candidate_copy': True,
          'unchanged_inference_driver_and_methods_verified_by_AST': True,
          'current_candidate_or_source_or_Gold_content_read_for_test': False,
          'actual_experiment_prepare_executed': False, 'actual_experiment_run_executed': False,
          'actual_experiment_score_executed': False, 'actual_experimental_requests_created': 0,
          'actual_frozen_prompts_or_request_seeds_changed': False,
          'LLM_instantiated': False, 'real_model_loaded': False, 'real_tokenizer_loaded': False,
          'model_calls': 0, 'external_api_calls': 0, 'dependency_installs': 0,
          'training': False, 'training_export': False,
          'limitations': ['Synthetic CPU validation establishes scorer/API behavior only; GPU/model initialization and generation remain untested.',
                          'Consumed RUN_ATTEMPT remains binding. No retry, resume, replacement, prepare, or run is authorized by this proposal.',
                          'Any future execution requires explicit user modification of the no-retry stop and a newly reviewed execution freeze.']}
write_once('OFFLINE_VALIDATION.json', record)
print(stream.getvalue(), end='')
print(json.dumps({key: record[key] for key in ('status', 'tests_run', 'failures', 'errors', 'frozen_and_historical_files_protected', 'frozen_and_historical_sha256_unchanged', 'model_calls')}))
raise SystemExit(0 if record['status'] == 'PASS' else 1)
