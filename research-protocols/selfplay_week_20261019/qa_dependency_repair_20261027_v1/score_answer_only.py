"""Post hoc diagnostic: final option sets scored separately from citation compliance.

Apply identically to all 204 frozen responses; preserve the strict primary scores.
"""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import re
import statistics

HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution_v2'
SCORER = HERE.parents[2] / 'offline-deliveries/h4_multiselect_partial_admission_v1/scoring.py'

def read(path):
    return json.loads(path.read_text())

def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def summary(xs):
    js = [r['J'] for r in xs if r['valid']]
    return {'requests': len(xs), 'questions': len({r['candidate_id'] for r in xs}),
            'answer_parse_valid': sum(r['valid'] for r in xs),
            'strict_valid': sum(r['strict_valid'] for r in xs),
            'exact_count': sum(r['exact'] for r in xs),
            'exact_accuracy': statistics.mean(r['exact'] for r in xs),
            'J': statistics.mean(js) if js else None, 'J_denominator': len(js),
            'empty_predictions': sum(r['valid'] and not r['prediction'] for r in xs)}

def main():
    spec = importlib.util.spec_from_file_location('answer_only_original_J', SCORER)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    public = {r['request_id']: r for r in read(OUT / 'PUBLIC_REQUESTS.json')['requests']}
    private = {r['request_id']: r for r in read(OUT / 'PRIVATE_SCORING.json')['requests']}
    records = []
    for strict in read(OUT / 'SCORES.json')['records']:
        row = public[strict['request_id']]
        pres = private[row['request_id']]['presentation']
        parsed = strict.get('parsed')
        final = parsed.get('final_answer') if isinstance(parsed, dict) else None
        match = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})?</box>', final or '')
        answer = (match.group(1) or '').split() if match else None
        raw = '<box>'+','.join(answer)+'</box>' if answer is not None and strict['finish_reason'] == 'stop' else None
        question = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'],
                    'options': {v: row['display'][k] for k, v in pres['display_to_canonical'].items()},
                    'gold': pres['canonical_gold']}
        request = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'],
                   'request_id': row['request_id'], 'variant': {'display_to_canonical': pres['display_to_canonical'],
                   'options': row['display'], 'displayed_gold': pres['display_gold']}}
        scored = scorer.score_response(question, request, {'request_id': row['request_id'], 'raw_output': raw})
        record = dict(strict)
        record.update(strict_valid=strict['valid'], strict_errors=strict['errors'],
                      strict_exact=strict['exact'], strict_J=strict['J'],
                      valid=scored['scored'], exact=bool(scored['exact_match']), J=scored['partial_score'],
                      prediction=scored['canonical_prediction'] or [], answer_parse_status=scored['parse_status'],
                      errors=[] if scored['scored'] else [scored['parse_reason']])
        records.append(record)
    summaries = []
    for cohort in ('baseline', 'new'):
        for condition in sorted({r['condition'] for r in records if r['cohort'] == cohort}):
            xs = [r for r in records if r['cohort'] == cohort and r['condition'] == condition]
            summaries.append({'cohort': cohort, 'condition': condition, **summary(xs)})
    per_item = []
    for cid in sorted({r['candidate_id'] for r in records}):
        for condition in sorted({r['condition'] for r in records if r['candidate_id'] == cid}):
            xs = [r for r in records if r['candidate_id'] == cid and r['condition'] == condition]
            per_item.append({'candidate_id': cid, 'condition': condition, **summary(xs),
                             'same_prediction_three_presentations': all(r['valid'] for r in xs) and len({tuple(r['prediction']) for r in xs}) == 1})
    result = {'post_hoc_diagnostic': True, 'score_basis': 'Final answer only; citation validity reported separately. Not a replacement for strict scores and not a model rerun.',
              'inputs': [bind(OUT / 'SCORES.json'), bind(OUT / 'PUBLIC_REQUESTS.json'), bind(OUT / 'PRIVATE_SCORING.json'), bind(SCORER)],
              'records': records, 'summaries': summaries, 'per_item': per_item,
              'limits': 'No-context guesses may exploit option semantics or may be lucky. Full-No and ablation differences do not alone certify necessity. Three presentations also change sampling seeds and are not independent questions.'}
    (OUT / 'ANSWER_ONLY_SCORES.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'summaries': summaries}, ensure_ascii=False))

if __name__ == '__main__':
    main()
