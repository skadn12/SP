"""Local execution/cost/source audit for the four approved roots. No API calls."""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter

BASE = Path(__file__).resolve().parent
RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_full_pipeline_realizer_state_contract_v1_live_1')
WORK = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-realizer-state-contract-v1')
sys.path.insert(0, str(WORK))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import cost_summary
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.full_pipeline_qa_source import verify_qa_source


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def audit():
    authorization = read(BASE/'AUTHORIZATION.json')
    roots = authorization['root_order']
    summary = read(RUN/'generation_summary.json')
    records = read(RUN/'complete_call_records.json')
    assert summary['root_order'] == roots
    assert summary['maximum_http_attempts'] == 68
    all_records, details = records['expansion'] + records['qa'], []
    assert cost_summary(all_records) == summary['cost']
    assert summary['cost']['http_attempts'] <= 68
    assert summary['cost']['retries'] == 0
    assert len(records['expansion']) <= 40 and len(records['qa']) <= 28
    ledgers = []
    for root in roots:
        path = RUN/'expansion/execution'/root/'development_call_ledger.json'
        if path.exists():
            ledger = read(path)
            expected = [r for r in records['expansion'] if r['root_id'] == root]
            assert expected == ledger['records']
            assert len(expected) <= 10
            assert len({r['stage'] for r in expected}) == len(expected)
            assert ledger['http_attempt_count'] == sum(r['http_attempt_count'] for r in expected)
            assert ledger['automatic_retry_count'] == 0 and not ledger['automatic_resume']
            ledgers.append({'path': str(path), 'sha256': sha(path), 'status': ledger['status']})
    qpath = RUN/'qa/execution/qa_call_ledger.json'
    qledger = read(qpath)
    assert records['qa'] == qledger['records']
    assert qledger['http_attempt_count'] == sum(r['http_attempt_count'] for r in records['qa'])
    assert qledger['automatic_retry_count'] == 0 and not qledger['automatic_resume']
    ledgers.append({'path': str(qpath), 'sha256': sha(qpath), 'status': qledger['status']})
    for kind, items in records.items():
        for row in items:
            assert row['root_id'] in roots
            assert row['http_attempt_count'] in (0, 1) and row['retry_count'] == 0
            parent = (RUN/'expansion/execution'/row['root_id'] if kind == 'expansion'
                      else RUN/'qa/execution')
            directory = parent/row['artifact_directory']
            request = read(directory/'request.json')['payload']
            assert digest(request) == row['payload_sha256']
            assert digest(request['messages']) == row['messages_sha256']
            assert request['model'] == 'tju-llm'
            assert request['temperature'] == (0.25 if kind == 'expansion' else 0.1)
            assert request['response_format'] == {'type': 'json_object'}
            assert request['max_tokens'] == row['maximum_output_tokens']
            metadata = read(directory/'response_metadata.json') if (directory/'response_metadata.json').exists() else {}
            assert metadata.get('transport_retry_count', 0) == metadata.get('rate_limit_retry_count', 0) == 0
            raw = metadata.get('raw_response')
            if isinstance(raw, dict):
                assert row['actual_model'] == raw.get('model', 'UNKNOWN')
                if 'usage' in raw:
                    assert row['usage'] == raw['usage']
                if raw.get('choices'):
                    assert row['finish_reason'] == raw['choices'][0]['finish_reason']
                    assert (directory/'response.txt').read_text() == raw['choices'][0]['message']['content']
            details.append({**row, 'artifact_directory': str(directory),
                'request_sha256': sha(directory/'request.json'),
                'response_sha256': sha(directory/'response.txt') if (directory/'response.txt').exists() else None})
    chronological = sorted(all_records, key=lambda r: r['reserved_at'])
    failures = [r for r in chronological if r['status'] == 'TECHNICAL_FAILURE']
    if failures:
        assert chronological[-1] == failures[-1]
        assert summary['status'] == 'STOPPED_TECHNICAL'
    sources = []
    for root in roots:
        old = Path(authorization['input_bundle'])/root/'inputs.json'
        current = RUN/'frozen'/root/'inputs.json'
        assert old.read_bytes() == current.read_bytes()
        original = read(current)
        actual = RUN/'expansion/execution'/root/'calls'/root/'planner/request.json'
        if actual.exists():
            assert read(actual)['payload'] == read(RUN/'frozen'/root/'planner_request.json')
        path = RUN/'qa/roots'/root/'qa_source_provenance.json'
        result = {'root_id': root, 'frozen_input_sha256': sha(current),
                  'input_unchanged': True, 'actual_planner_matches_prepared': actual.exists()}
        if path.exists():
            source = read(path.with_name('qa_source.json'))
            provenance = read(path)
            verify_qa_source(source, provenance)
            history = original['prior_expanded']['turns']
            assert source['turns'][:len(history)] == history
            assert source['conversation_date_segments'] == original['conversation_date_segments']
            result.update(qa_source_path=str(path.with_name('qa_source.json')),
                          qa_source_sha256=sha(path.with_name('qa_source.json')),
                          qa_source_turn_count=len(source['turns']),
                          history_turn_count=len(history), new_turn_count=len(source['turns'])-len(history),
                          promoted_scene_refs=provenance['promoted_scene_refs'],
                          new_R02_dates=provenance['new_R02_dates'])
        sources.append(result)
    freezes = []
    for manifest in (RUN/'freeze_manifest.json', Path(authorization['input_bundle'])/'manifest.json',
                     BASE.parent/'h4_planner_review_contract_v1/prepared_run_final/freeze_manifest.json'):
        rows = read(manifest)['files']
        changed = [r['path'] for r in rows if not Path(r['path']).is_file() or sha(Path(r['path'])) != r['sha256']]
        assert not changed, changed
        freezes.append({'manifest': str(manifest), 'count': len(rows), 'changed': changed})
    report = {'status': 'PASS', 'cost': summary['cost'], 'stage_calls': dict(Counter(r['stage'] for r in all_records)),
              'actual_models': sorted({r['actual_model'] for r in all_records}),
              'technical_failures': len(failures), 'ledgers': ledgers,
              'source_identity': sources, 'freeze_checks': freezes, 'call_records': details,
              'extra_audit_api_calls': 0, 'content_semantics': 'SEPARATE_CODEX_OFFLINE_DIAGNOSIS'}
    (BASE/'reports/EXECUTION_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'cost', 'stage_calls', 'actual_models')}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
