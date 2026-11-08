"""One status request, then at most one explicitly reviewed successful readback; never Add."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from urllib.parse import quote

from scripts.current_method.restart_n5_spell.h4_memorax_v3_flat_bootstrap_v2 import run

HERE = Path(__file__).resolve().parent
TASK = 'a6e1c579-fe23-4523-a873-f6d6e2771733'


def execute(operation):
    manifest = run.read(run.DELIVERY / 'manifest.json')
    run.verify(manifest)
    prior = run.read(run.DELIVERY / 'ledger.json')
    if (prior['task_id'] != TASK or prior['status'] != 'STOPPED_POLL_LIMIT_PENDING_NO_RESEND' or
            prior['cumulative_memory_operation_http_attempts'] != 26):
        raise ValueError('prior_task_or_budget_changed')
    config = run.provider.load_memorax_config(credentials_path=run.query_run.CREDENTIALS,
                                              allow_insecure_http=True)
    if config.normalized_base_url != run.query_run.EXPECTED_ORIGIN:
        raise ValueError('configured_origin_changed')
    if operation == 'status':
        ledger = {'status': 'PREPARED', 'task_id': TASK, 'root_id': run.ROOT,
            'authorization': 'User requested next step after pending-task report; one status and only if complete one readback',
            'prior_business_http_attempts': 26, 'http_attempt_count': 0, 'maximum_http_attempts': 2,
            'add_attempts': 0, 'automatic_retries': 0, 'generation_calls': 0, 'pipeline_resume': False,
            'prior_ledger_sha256': run.sha(run.DELIVERY / 'ledger.json'), 'script_sha256': run.sha(__file__),
            'records': [], 'verified_cost': 'UNKNOWN', 'internal_model_usage': 'UNKNOWN'}
        run.memory_diagnostics._save(HERE / 'ledger.json', ledger, exclusive=True)
        method, path, payload = 'GET', '/v1/memories/add/status/' + quote(TASK, safe=''), None
    elif operation == 'readback':
        ledger = run.read(HERE / 'ledger.json')
        review = run.read(HERE / 'STATUS_REVIEW.json')
        if (ledger['status'] != 'STATUS_RESPONSE_REQUIRES_OFFLINE_REVIEW' or
                ledger['http_attempt_count'] != 1 or review.get('task_id') != TASK or
                review.get('completion_confirmed') is not True or
                review.get('status_response_sha256') != run.sha(HERE / 'calls/status/response.json')):
            raise ValueError('reviewed_completed_status_required')
        method, path, payload = 'POST', run.SEARCH, run.read(run.DELIVERY / 'frozen/query.json')
    else:
        raise ValueError('unsupported_operation')
    if (ledger['prior_ledger_sha256'] != run.sha(run.DELIVERY / 'ledger.json') or
            ledger['script_sha256'] != run.sha(__file__)):
        raise ValueError('resume_dependency_changed')
    directory = HERE / 'calls' / operation
    run.memory_diagnostics._save(directory / 'request.json',
        {'method': method, 'path': path, 'payload': payload, 'authentication_headers_recorded': False}, exclusive=True)
    row = {'operation': operation, 'status': 'RESERVED', 'http_attempt_count': 0, 'http_status': 'UNKNOWN'}
    ledger['records'].append(row)
    def save():
        ledger['cumulative_business_http_attempts'] = 26 + ledger['http_attempt_count']
        run.memory_diagnostics._save(HERE / 'ledger.json', run.safe_response(ledger, config.api_key))
    save()
    def charge():
        if row['http_attempt_count'] or ledger['http_attempt_count'] >= 2:
            raise ValueError('read_limit_exhausted')
        row.update(http_attempt_count=1, status='SENT_COMPLETION_UNKNOWN',
                   sent_at=datetime.now(timezone.utc).isoformat())
        ledger['http_attempt_count'] += 1
        save()
    try:
        response = run.transport.request_once(config=config, method=method, path=path, payload=payload,
            directory=directory, before_send=charge, request_id='h4-v3-259192-resume01-' + operation)
        diagnostic = run.read(directory / 'diagnostic.json')
        row.update(http_status=diagnostic['http_status'], status='RESPONSE_RECEIVED')
        if diagnostic['http_status'] != 200:
            raise ValueError('unexpected_http_status')
        if operation == 'status':
            ledger['status'] = 'STATUS_RESPONSE_REQUIRES_OFFLINE_REVIEW'
        else:
            summary = run.summarize_response(response, run.read(run.query_run.OPENAPI))
            summary['write_and_async_status'] = 'COMPLETION_OBSERVED'
            run.memory_diagnostics._save(HERE / 'readback_summary.json', summary, exclusive=True)
            ledger.update(status='AWAITING_CONTENT_REVIEW' if summary['total_buckets'] else 'STOPPED_EMPTY_READBACK',
                          bucket_counts=summary['bucket_counts'], total_buckets=summary['total_buckets'])
    except BaseException as error:
        row.update(status='FAILED_OR_UNKNOWN', error_type=type(error).__name__)
        if (directory / 'diagnostic.json').exists():
            row['http_status'] = run.read(directory / 'diagnostic.json').get('http_status', 'UNKNOWN')
        ledger['status'] = 'STOPPED_TECHNICAL_OR_LOCAL_FAILURE_NO_RETRY'
    save()
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['status', 'readback'])
    args = parser.parse_args()
    try:
        with (HERE / '.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(execute(args.operation), ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({'status': 'STOPPED_NO_RETRY', 'error_type': type(error).__name__}))
        raise SystemExit(1) from None
