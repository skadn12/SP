#!/usr/bin/env python3
"""Read-only post-freeze integrity audit; no imports of generation/scoring code.

Use --experiment-dir to audit another independently frozen thinking-switch run.
Only the new JSON report is written, exclusively; imported artifacts stay intact.
Gold files are hashed as bytes, never parsed or used for any model operation.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def audit(experiment_dir):
    started = time.perf_counter()
    root = Path(experiment_dir).resolve()
    out = root / 'execution'
    errors, checks, cache = [], {}, {}
    binding_count = 0

    def require(ok, code, detail=None):
        checks[code] = checks.get(code, True) and bool(ok)
        if not ok:
            errors.append({'check': code, 'detail': detail})
        return bool(ok)

    def binding(value, expected_path=None, code='BINDING'):
        nonlocal binding_count
        binding_count += 1
        if not require(isinstance(value, dict) and all(k in value for k in ('path', 'sha256', 'bytes')), code + '_SHAPE', expected_path and str(expected_path)):
            return False
        path = Path(value['path']).resolve()
        ok = True
        if expected_path is not None:
            ok &= require(path == Path(expected_path).resolve(), code + '_PATH', str(path))
        try:
            if path not in cache:
                cache[path] = fingerprint(path)
            actual = cache[path]
            ok &= require(actual['sha256'] == value['sha256'] and actual['bytes'] == value['bytes'], code + '_HASH', str(path))
        except (OSError, ValueError) as exc:
            ok &= require(False, code + '_READ', str(path) + ': ' + str(exc))
        return bool(ok)

    def bound_set(values, expected, code):
        if not require(isinstance(values, list), code + '_LIST'):
            return
        paths = [str(Path(x['path']).resolve()) for x in values if isinstance(x, dict) and isinstance(x.get('path'), str)]
        require(len(paths) == len(values) == len(set(paths)), code + '_UNIQUE', {'bindings': len(values), 'unique_paths': len(set(paths))})
        require(set(paths) == {str(Path(x).resolve()) for x in expected}, code + '_EXACT_SET', {'missing': sorted({str(Path(x).resolve()) for x in expected} - set(paths)), 'extra': sorted(set(paths) - {str(Path(x).resolve()) for x in expected})})
        for value in values:
            binding(value, code=code)

    prep_path, freeze_path = out / 'PREPARATION.json', out / 'RAW_RESPONSE_FREEZE.json'
    if not prep_path.exists() or not freeze_path.exists():
        return {'status': 'NOT_READY', 'integrity_pass': False, 'experiment_directory': str(root), 'errors': [{'check': 'REQUIRED_FREEZE_EXISTS', 'detail': [str(p) for p in (prep_path, freeze_path) if not p.exists()]}]}
    prep_start, freeze_start = fingerprint(prep_path), fingerprint(freeze_path)
    prep, frozen = read(prep_path), read(freeze_path)
    for value in prep['bindings']:
        binding(value, code='PREPARATION_BINDING')
    binding(prep['private_scoring_binding'], code='PRIVATE_SCORING_HASH_ONLY')
    for value in prep['old_content_bindings']:
        binding(value, code='OLD_CONTENT_HASH_ONLY')
    binding(prep['protocol_binding'], out / 'PROTOCOL_FREEZE.json', 'PROTOCOL_BINDING')
    protocol = read(out / 'PROTOCOL_FREEZE.json')
    rows = read(out / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(out / 'PROMPT_TOKEN_IDS.json')
    ids = [row['request_id'] for row in rows]
    require(len(ids) == len(set(ids)) == 120, 'PUBLIC_120_UNIQUE')
    require(prep['requests'] == frozen['requests'] == len(ids), 'REQUEST_COUNT')
    require(ids == prep['fixed_order'] == frozen['request_ids'], 'FROZEN_REQUEST_ORDER')
    require(set(tokens) == set(ids), 'PROMPT_TOKEN_ID_SET')
    require(all(row['condition'] == 'FULL' for row in rows), 'FULL_ONLY')
    require(collections.Counter(row['mode'] for row in rows) == {'THINK_OFF': 60, 'THINK_ON': 60}, 'BALANCED_MODE_COUNTS')
    require(len({row['candidate_id'] for row in rows}) == 30, 'THIRTY_QUESTIONS')
    pairs = collections.defaultdict(list)
    for row in rows:
        pairs[(row['candidate_id'], row['variant_id'])].append(row)
    require(len(pairs) == 60 and all(len(pair) == 2 and {r['mode'] for r in pair} == {'THINK_OFF', 'THINK_ON'} for pair in pairs.values()), 'SIXTY_MATCHED_PAIRS')
    paired_fields = ('base_request_id', 'source_binding', 'display', 'labels', 'sampling_seed', 'messages_sha256', 'schema_path', 'question', 'task_operation')
    require(all(all(pair[0].get(key) == pair[1].get(key) for key in paired_fields) for pair in pairs.values() if len(pair) == 2), 'PAIRED_INPUTS_EQUAL')
    require(all(len(tokens[row['request_id']]) == row['prompt_tokens'] and row['prompt_tokens'] + protocol['max_tokens'] <= protocol['max_model_len'] for row in rows), 'PROMPT_AND_OUTPUT_WITHIN_BUDGET')
    raw_paths = [out / 'raw' / (rid + '.json') for rid in ids]
    bound_set(frozen['bindings'], raw_paths, 'RAW_BINDINGS')
    require({p.resolve() for p in (out / 'raw').glob('*.json')} == {p.resolve() for p in raw_paths}, 'RAW_DIRECTORY_EXACT_SET')
    binding(frozen['started_binding'], out / 'STARTED.json', 'STARTED_BINDING')
    start = read(out / 'STARTED.json')
    binding(start['preparation_binding'], prep_path, 'STARTED_PREPARATION_BINDING')
    require(start['requests'] == len(rows) and start.get('gold_read') is False, 'STARTED_DECLARATIONS')
    submitted = frozen['submitted_ids']
    require(len(submitted) == len(set(submitted)) and set(submitted) <= set(ids), 'SUBMITTED_UNIQUE_KNOWN')
    submission_paths = [out / 'submitted' / (rid + '.json') for rid in submitted]
    bound_set(frozen['submission_bindings'], submission_paths, 'SUBMISSION_BINDINGS')
    require({p.resolve() for p in (out / 'submitted').glob('*.json')} == {p.resolve() for p in submission_paths}, 'SUBMISSION_DIRECTORY_EXACT_SET')
    receipt_path = out / 'MODEL_LOAD_RECEIPT.json'
    receipt = None
    if frozen.get('model_receipt_binding') is not None:
        binding(frozen['model_receipt_binding'], receipt_path, 'MODEL_RECEIPT_BINDING')
        receipt = read(receipt_path)
        binding(receipt['runner_binding'], root / 'run.py', 'MODEL_RECEIPT_RUNNER_BINDING')
        require(receipt['runtime_versions'] == prep['runtime_versions'], 'MODEL_RECEIPT_RUNTIME')
        require(receipt.get('config') == prep['config_receipt'], 'MODEL_RECEIPT_CONFIG')
        require(receipt.get('structured_outputs') == protocol['structured_outputs'], 'MODEL_RECEIPT_DECODE_MODE')
        for key in ('actual_hf_config', 'actual_hf_text_config'):
            actual = receipt.get(key, {})
            require(actual.get('rope_theta') == 1000000 and actual.get('rope_parameters') == {'rope_type': 'default', 'rope_theta': 1000000}, 'MODEL_RECEIPT_' + key.upper() + '_THETA')
    else:
        require(not submitted and not receipt_path.exists(), 'NO_RECEIPT_ONLY_BEFORE_SUBMISSION')
    endpoint = read(prep['endpoint_path'])
    for value in endpoint['model_bindings'] + endpoint['tokenizer_bindings']:
        binding(value, code='ENDPOINT_MODEL_OR_TOKENIZER_BINDING')
    if receipt is not None:
        require(receipt['model_directory'] == endpoint['model_directory'] and receipt['tokenizer_directory'] == endpoint['tokenizer_directory'], 'RECEIPT_ENDPOINT_DIRECTORIES')
    for rid, path in zip(submitted, submission_paths):
        item = read(path)
        require(item['request_id'] == rid, 'SUBMISSION_REQUEST_ID', rid)
        binding(item['model_load_receipt_binding'], receipt_path, 'SUBMISSION_MODEL_RECEIPT')
        require(item['model_load_receipt_binding'] == frozen['model_receipt_binding'], 'SUBMISSION_SAME_RECEIPT', rid)
    batch_paths = sorted((out / 'batches').glob('*.json'))
    bound_set(frozen['batch_bindings'], batch_paths, 'BATCH_BINDINGS')
    batched_ids, batch_index_for = [], {}
    for path in batch_paths:
        batch = read(path)
        require(path.name == str(batch['batch_index']).zfill(3) + '.json', 'BATCH_FILENAME', str(path))
        require(0 < len(batch['request_ids']) <= protocol['batch_size'], 'BATCH_SIZE', str(path))
        for rid in batch['request_ids']:
            batched_ids.append(rid)
            batch_index_for[rid] = batch['batch_index']
    require(len(batched_ids) == len(set(batched_ids)) and set(batched_ids) <= set(submitted), 'BATCH_REQUESTS_UNIQUE_SUBMITTED')
    statuses, finishes, returned_ids = collections.Counter(), collections.Counter(), []
    for row, path in zip(rows, raw_paths):
        raw = read(path)
        rid = row['request_id']
        require(raw['request_id'] == rid and raw['mode'] == row['mode'], 'RAW_REQUEST_ASSOCIATION', rid)
        statuses[raw['status']] += 1
        finishes[str(raw.get('finish_reason'))] += 1
        require(raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED'), 'RAW_KNOWN_STATUS', rid)
        if raw['status'] == 'RETURNED':
            returned_ids.append(rid)
            require(rid in submitted and raw.get('gold_read') is False, 'RAW_RETURNED_SUBMITTED_NO_GOLD', rid)
            require(raw.get('base_request_id') == row['base_request_id'] and raw.get('sampling_seed') == row['sampling_seed'], 'RAW_BASE_ID_SEED', rid)
            require(raw.get('prompt_token_ids') == tokens[rid], 'RAW_PROMPT_TOKENS', rid)
            require(not raw.get('capture_errors'), 'RAW_CAPTURE_ERRORS_EMPTY', rid)
            binding(raw.get('model_load_receipt_binding'), receipt_path, 'RAW_MODEL_RECEIPT')
            require(raw.get('model_load_receipt_binding') == frozen['model_receipt_binding'], 'RAW_SAME_RECEIPT', rid)
            require(raw.get('model_directory') == endpoint['model_directory'], 'RAW_MODEL_DIRECTORY', rid)
            require(raw.get('batch_index') == batch_index_for.get(rid), 'RAW_BATCH_INDEX', rid)
        elif raw['status'] == 'NOT_SUBMITTED':
            require(rid not in submitted, 'RAW_NOT_SUBMITTED_STATUS', rid)
        else:
            require(rid in submitted, 'RAW_BACKEND_ERROR_SUBMITTED', rid)
    require(set(batched_ids) == set(returned_ids), 'COMPLETED_BATCHES_MATCH_RETURNED')
    require(frozen['returned'] == len(returned_ids), 'FROZEN_RETURNED_COUNT')
    require(fingerprint(prep_path) == prep_start and fingerprint(freeze_path) == freeze_start, 'FREEZES_STABLE_DURING_CHECK')
    complete = len(returned_ids) == len(ids) and len(submitted) == len(ids) and frozen.get('failure') is None
    passed = not errors
    return {'schema_version': 'thinking_switch_closure_integrity_v1', 'status': ('PASS_INTEGRITY_COMPLETE' if complete else 'PASS_INTEGRITY_INCOMPLETE_RUN') if passed else 'FAIL_INTEGRITY', 'integrity_pass': passed, 'all_requests_returned_without_backend_failure': complete, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'experiment_directory': str(root), 'checker_binding': fingerprint(Path(__file__)), 'preparation_binding': prep_start, 'raw_freeze_binding': freeze_start, 'checks': checks, 'errors': errors, 'binding_entries_checked': binding_count, 'unique_bound_files_checked': len(cache), 'requests': len(ids), 'matched_display_pairs': len(pairs), 'status_counts': dict(statuses), 'finish_reason_counts': dict(finishes), 'backend_failure_present': bool(frozen.get('failure')), 'endpoint_model_and_tokenizer_files_hashed': len(endpoint['model_bindings']) + len(endpoint['tokenizer_bindings']), 'elapsed_seconds': time.perf_counter() - started, 'scope': 'Integrity only. No generation, retries, answer scoring, prompt rewriting or private Gold JSON parsing. Raw output content is neither printed nor semantically evaluated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.experiment_dir.resolve() / 'CLOSURE_CHECK.json'
    if output.exists():
        raise SystemExit('Refusing to overwrite existing report: ' + str(output))
    try:
        report = audit(args.experiment_dir)
    except Exception as exc:
        report = {'status': 'FAIL_CHECKER_EXCEPTION', 'integrity_pass': False, 'experiment_directory': str(args.experiment_dir.resolve()), 'errors': [{'check': 'CHECKER_EXCEPTION', 'detail': type(exc).__name__ + ': ' + str(exc)}]}
    if report['status'] == 'NOT_READY':
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(2)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'report': str(output), 'status': report['status'], 'errors': report.get('errors', []), 'elapsed_seconds': report.get('elapsed_seconds')}, ensure_ascii=False))
    raise SystemExit(0 if report.get('integrity_pass') else 1)


if __name__ == '__main__':
    main()
