"""Preserved engineering recovery: omit unsupported grammar keyword only.

The original uniqueItems contract remains the final validator. No successful
responses existed before recovery; original failure artifacts are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIVE = HERE / 'diagnosis_compat_v2'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), path
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare():
    original = read(HERE / 'diagnosis_jobs.json')
    failure = read(HERE / 'diagnosis/RAW_RESPONSE_FREEZE.json')
    assert failure['returned'] == 0
    jobs, differences = [], []
    for j in original['jobs']:
        new = dict(j)
        schema = read(j['schema_path'])
        assert schema['properties']['source_turn_refs'].pop('uniqueItems') is True
        target = HERE / 'diagnosis_compat_inputs/schemas' / Path(j['schema_path']).name
        if not target.exists():
            save(target, schema)
            differences.append({'original_binding': bind(j['schema_path']), 'compat_binding': bind(target), 'only_change': 'remove properties.source_turn_refs.uniqueItems=true from generation grammar; retain final validation'})
        new.update(schema_path=str(target), schema_binding=bind(target))
        jobs.append(new)
    save(HERE / 'diagnosis_compat_jobs.json', {'stage': 'diagnosis', 'jobs': jobs})
    save(HERE / 'ENGINEERING_RECOVERY.json', {'reason': 'vLLM runtime grammar rejects uniqueItems: Unimplemented keys; zero original answers returned', 'original_failure_binding': bind(HERE / 'diagnosis/RAW_RESPONSE_FREEZE.json'), 'original_jobs_binding': bind(HERE / 'diagnosis_jobs.json'), 'compat_jobs_binding': bind(HERE / 'diagnosis_compat_jobs.json'), 'schema_differences': differences, 'prompt_seed_model_temperature_output_budget_unchanged': True, 'original_contract_still_used_for_validation': True, 'manual_engineering_recovery_not_automatic_or_outcome_based_retry': True, 'original_failed_submissions': len(failure['submitted_ids']), 'original_not_submitted': 12 - len(failure['submitted_ids']), 'new_response_directory': str(ACTIVE)})
    print(json.dumps({'compat_jobs': len(jobs), 'unchanged_prompts_seeds': True, 'original_returned': 0}))


def score():
    import jsonschema
    freeze = read(ACTIVE / 'RAW_RESPONSE_FREEZE.json')
    for b in freeze['bindings']:
        assert bind(b['path']) == b
    jobs = read(HERE / 'diagnosis_jobs.json')['jobs']
    packets = {c['candidate_id']: c for c in read(HERE / 'DIAGNOSIS_INPUT_PACKETS_PRIVATE.json')['cases']}
    records = []
    for job in jobs:
        raw_path = ACTIVE / 'raw' / (job['request_id'] + '.json')
        raw = read(raw_path)
        errors, card, mechanical = [], None, {}
        try:
            assert raw['status'] == 'RETURNED', 'BACKEND_NOT_RETURNED'
            assert raw['finish_reason'] == 'stop', 'NOT_NORMAL_STOP'
            text = raw['raw_output'].strip()
            if text.endswith('<|im_end|>'):
                text = text[:-len('<|im_end|>')].rstrip()
            card = json.loads(text)
            jsonschema.validate(card, read(job['schema_path']))
            packet = packets[job['candidate_id']]['packets'][job['arm']]
            traces = {t['trace_id']: t for t in packet['trajectories']}
            quote_checks = [{'trace_id': e['trace_id'], 'trace_exists': e['trace_id'] in traces, 'verbatim_quote': bool(e['quote']) and e['quote'] in (traces.get(e['trace_id'], {}).get('response_text') or '')} for e in card['trace_evidence']]
            actual_success = any(t['final_answer_exact'] for t in traces.values())
            claimed = card['contrast_assessment']
            sid = claimed['success_trace_id']
            mechanical = {'quote_checks': quote_checks, 'all_quotes_verbatim': all(e['trace_exists'] and e['verbatim_quote'] for e in quote_checks), 'has_quotes': bool(quote_checks), 'success_availability_claim_matches': claimed['has_final_correct_trace'] == actual_success, 'success_id_valid': (sid in traces and traces[sid]['final_answer_exact']) if actual_success and claimed['has_final_correct_trace'] else not sid, 'semantic_review': 'PENDING'}
        except Exception as exc:
            errors.append(str(exc))
        records.append({'request_id': job['request_id'], 'candidate_id': job['candidate_id'], 'arm': job['arm'], 'contract_valid': not errors, 'card': card, 'errors': errors, 'mechanical': mechanical, 'raw_binding': bind(raw_path), 'input_tokens': len(raw.get('prompt_token_ids', [])), 'output_tokens': len(raw.get('output_token_ids', [])), 'finish_reason': raw.get('finish_reason')})
    save(HERE / 'DIAGNOSIS_SCORES.json', {'records': records, 'raw_freeze_binding': bind(ACTIVE / 'RAW_RESPONSE_FREEZE.json'), 'original_final_contract_preserved': True, 'engineering_failure_binding': bind(HERE / 'diagnosis/RAW_RESPONSE_FREEZE.json')})
    mapping = read(HERE / 'BLIND_MAPPING_PRIVATE.json')['records']
    bykey = {(r['candidate_id'], r['arm']): r for r in records}
    review_cases = []
    for m in mapping:
        packet = packets[m['candidate_id']]['packets']['B']
        cards = []
        for label in ['X', 'Y']:
            arm = m[label]
            r = bykey[(m['candidate_id'], arm)]
            available = packets[m['candidate_id']]['packets'][arm]['trajectories']
            cards.append({'anonymous_label': label, 'contract_valid': r['contract_valid'], 'card': r['card'], 'mechanical': r['mechanical'], 'available_trace_ids': [t['trace_id'] for t in available]})
        review_cases.append({'candidate_id': m['candidate_id'], 'source_binding': packets[m['candidate_id']]['source_binding'], 'common_evidence': packet, 'cards': cards})
    save(HERE / 'SEMANTIC_REVIEW_PACKET.json', {'purpose': 'Candidate-card usefulness only; method labels hidden but content can reveal method; not a formal stable weakness or training-benefit experiment.', 'cases': review_cases})
    print(json.dumps({'responses': len(records), 'contract_valid': sum(r['contract_valid'] for r in records), 'all_quotes_verbatim': sum(r.get('mechanical', {}).get('all_quotes_verbatim', False) for r in records)}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'score'])
    {'prepare': prepare, 'score': score}[p.parse_args().action]()
