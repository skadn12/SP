"""Independent metric recomputation using the user's original J scorer."""
import collections
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution'
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    pilot = load('pilot_verification_import', HERE / 'run.py')
    original_path = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_multiselect_partial_admission_v1/scoring.py')
    scorer = load('pilot_user_original_J', original_path)
    rows = {r['request_id']: r for r in pilot.read(OUT / 'PUBLIC_REQUESTS.json')['requests']}
    private = {r['request_id']: r for r in pilot.read(OUT / 'PRIVATE_SCORING.json')['requests']}
    records = pilot.read(OUT / 'SCORES.json')['records']
    checks, judgment_inconsistencies = [], []
    for rec in records:
        row, key = rows[rec['request_id']], private[rec['request_id']]
        if rec['question_type'] == 'multi_select':
            canonical_options = {key['display_to_canonical'][label]: text for label, text in row['display'].items()}
            question = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'], 'options': canonical_options, 'gold': key['canonical_gold']}
            request = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'], 'request_id': row['request_id'],
                       'variant': {'display_to_canonical': key['display_to_canonical'], 'options': row['display'], 'displayed_gold': key['display_gold']}}
            if rec['valid']:
                labels = rec['parsed']['final_answer'][5:-6].split()
                boxed = '<box>' + ','.join(labels) + '</box>'
            else:
                boxed = None
            official = scorer.score_response(question, request, {'request_id': row['request_id'], 'raw_output': boxed})
            assert official['partial_score'] == rec['J'], (row['request_id'], official, rec['J'])
            if rec['valid']:
                assert official['canonical_prediction'] == rec['canonical_prediction']
                assert official['exact_match'] == rec['exact']
            checks.append({'request_id': row['request_id'], 'J_matches_original_scorer': True})
            if rec['valid']:
                selected = set(rec['parsed']['final_answer'][5:-6].split())
                supported = {l for l, v in rec['parsed']['item_evidence'].items() if v['judgment'] == 'supported'}
                if selected != supported:
                    judgment_inconsistencies.append({'request_id': row['request_id'], 'selected': sorted(selected), 'supported': sorted(supported)})
    strata = []
    for cohort in ('legacy_format', 'prototype_difficulty'):
        for qtype in ('multi_select', 'ordering'):
            for fmt in pilot.FORMATS:
                rs = [r for r in records if r['cohort'] == cohort and r['question_type'] == qtype and r['format'] == fmt]
                if rs:
                    strata.append({'cohort': cohort, 'question_type': qtype, 'format': fmt, **pilot.summarize(rs)})
    root_strata = []
    for root in sorted({r['root_id'] for r in records}):
        for qtype in ('multi_select', 'ordering'):
            for fmt in pilot.FORMATS:
                rs = [r for r in records if r['cohort'] == 'legacy_format' and r['root_id'] == root and r['question_type'] == qtype and r['format'] == fmt]
                if rs:
                    root_strata.append({'root_id': root, 'question_type': qtype, 'format': fmt, **pilot.summarize(rs)})
    pilot.save(OUT / 'ADDITIONAL_ANALYSIS.json', {'J_reference_binding': pilot.binding(original_path),
         'scoring_binding': pilot.binding(OUT / 'SCORES.json'), 'J_check_count': len(checks), 'checks': checks,
         'strata': strata, 'root_strata': root_strata, 'selected_vs_supported_mismatches': judgment_inconsistencies,
         'citation_limit': 'Schema membership and roundtrip are mechanical. They do not validate semantic evidence support.'})
    print(json.dumps({'J_check_count': len(checks), 'strata': strata, 'selected_vs_supported_mismatches': len(judgment_inconsistencies)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
