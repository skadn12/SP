#!/usr/bin/env python3
"""Verify independent invariants and bind all delivered artifacts."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def main():
    checks = []
    def check(name, passed, detail=None):
        checks.append({'check': name, 'pass': bool(passed), 'detail': detail})
    candidates = read(HERE / 'CANDIDATES_PRIVATE.json')
    reviews = read(HERE / 'CONTENT_REVIEW_AI.json')['reviews']
    scores = read(HERE / 'execution/SCORES.json')['records']
    analysis = read(HERE / 'execution/ANALYSIS_SUMMARY.json')
    mechanics = read(HERE / 'MECHANICAL_FINAL.json')
    expected = {f'CL1-{a}-{r}-MC{s}' for a in 'FG' for r in ('218519','222937','225575','229686') for s in range(2,7)}
    check('all_planned_40_slots_retained', len(candidates) == 40 and {c['candidate_id'] for c in candidates} == expected)
    check('independent_review_coverage', len(reviews) == 40 and {r['candidate_id'] for r in reviews} == expected)
    check('mechanical_coverage', mechanics['total'] == 40 and len(mechanics['records']) == 40, {'passed': mechanics['passed']})
    check('all_admitted_content_passed', all(r['decision'] == 'PASS' for r in reviews if r['candidate_id'] in {s['candidate_id'] for s in scores}))
    for name in ('AUTHORING_INPUT_FREEZE.json', 'CONTENT_FREEZE.json'):
        errors = [b['path'] for b in read(HERE / name)['bindings'] if bind(b['path']) != b]
        check(name + '_hashes', not errors, errors)
    sources = read(HERE / 'SOURCE_SELECTION_FINAL.json')['roots']
    check('original_sources_unchanged', all(bind(s['path']) == {k:s[k] for k in ('path','sha256','bytes')} for s in sources))
    check('source_turn_count', sum(len(read(s['path'])['turns']) for s in sources) == 914)
    check('unique_scored_requests', len({s['request_id'] for s in scores}) == len(scores))
    check('request_counts_match_execution', len(scores) == analysis['denominators']['frozen_new_requests'] == analysis['denominators']['new_submitted_requests'] == analysis['denominators']['new_returned_requests'], analysis['denominators'])
    check('no_response_failures', all(s['valid'] and s['finish_reason'] == 'stop' and not s['errors'] for s in scores))
    independent_errors = []
    for s in scores:
        if not s['valid']:
            continue
        prediction, gold = set(s['canonical_prediction']), set(s['canonical_gold'])
        tp, fp, fn = len(prediction & gold), len(prediction - gold), len(gold - prediction)
        j = tp / len(gold) - fp / (6 - len(gold)) if len(gold) < 6 else None
        if (s['TP'],s['FP'],s['FN']) != (tp,fp,fn) or s['exact'] != (prediction == gold) or (j is None and s['J'] is not None) or (j is not None and (s['J'] is None or abs(j-s['J']) > 1e-12)):
            independent_errors.append(s['request_id'])
    check('independent_J_and_exact_recalculation', not independent_errors, independent_errors)
    check('two_permutations_and_two_conditions', all(len(rows) == 4 and {(r['condition'],r['variant_id']) for r in rows} == {(c,v) for c in ('FULL','QUESTION_ONLY') for v in ('perm_00','perm_01')} for rows in [[s for s in scores if s['candidate_id'] == cid] for cid in {s['candidate_id'] for s in scores}]))
    check('matched_FG_admissions', all({s['arm'] for s in scores if (s['root_id'],s['slot_id']) == pair} == {'F','G'} for pair in {(s['root_id'],s['slot_id']) for s in scores}))
    raw_errors = []
    for s in scores:
        raw = read(s['raw_binding']['path'])
        if bind(s['raw_binding']['path']) != s['raw_binding'] or raw.get('gold_read') is not False or raw.get('capture_errors'):
            raw_errors.append(s['request_id'])
    check('raw_hashes_no_gold_or_capture_errors', not raw_errors, raw_errors)
    matching = read(HERE / 'FEEDBACK_ERROR_MATCHING_AI.json')
    target_rows = matching['records']
    full_f_ids = {s['request_id'] for s in scores if s['arm'] == 'F' and s['condition'] == 'FULL'}
    check('feedback_error_audit_all_F_FULL', len(target_rows) == len(full_f_ids) and {r['request_id'] for r in target_rows} == full_f_ids)
    check('no_training_or_source_extension', read(HERE / 'PROTOCOL_FREEZE.json')['training'] is False and read(HERE / 'PROTOCOL_FREEZE.json')['source_extension'] is False)
    baseline = read(HERE / 'execution/BASELINE_J_RECOMPUTED.json')
    check('baseline_reused_not_rerun', baseline['new_model_calls'] == 0 and baseline['historical_responses_reused'] is True and len(baseline['records']) == 80)
    audit = {'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL', 'passed': sum(c['pass'] for c in checks), 'failed': sum(not c['pass'] for c in checks), 'checks': checks, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (HERE / 'FINAL_INTEGRITY_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    manifest = {'experiment_id': HERE.name, 'integrity_status': audit['status'], 'bindings': [bind(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'DELIVERY_MANIFEST.json' and '__pycache__' not in p.parts], 'note': 'Manifest excludes itself; source and external legacy bindings are separately preserved in freezes.'}
    (HERE / 'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': audit['status'], 'passed': audit['passed'], 'failed': audit['failed'], 'files': len(manifest['bindings'])}))
    if audit['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
