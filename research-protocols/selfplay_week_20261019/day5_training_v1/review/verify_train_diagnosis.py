#!/usr/bin/env python3
"""Read-only CPU gate for the fixed Train diagnosis package and endpoint(s)."""
import argparse
import ast
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DAY5 = HERE.parent
WEEK = DAY5.parent
D = DAY5 / 'train_diagnosis_prepare'
OLD = WEEK / 'day3_solver187_v1/runtime/direct_v2'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['A1', 'B1'], action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks, files, evidence = [], {}, {}

    def check(label, value, detail=None):
        checks.append({'check': label, 'pass': bool(value), 'detail': detail})

    def bound(row):
        actual = binding(row['path'])
        check('bound:' + row['path'], actual == row)
        files[row['path']] = actual

    def add(p):
        row = binding(p)
        files[row['path']] = row

    try:
        f = read(D / 'EXECUTION_ADDENDUM_V2.json')
        add(D / 'EXECUTION_ADDENDUM_V2.json')
        for row in f['files'] + f['upstream_bindings'] + f['upstream_direct_prompt_bindings'] + [f['original_freeze'], f['executor']]:
            bound(row)
        direct = read(D / 'public/DIRECT_REQUESTS.json')['requests']
        brief = read(D / 'public/BRIEF_EVIDENCE_REQUESTS.json')['requests']
        questions = read(D / 'public/PUBLIC_QUESTIONS.json')['questions']
        selected = read(D / 'SELECTION.json')
        train_roots = set(read(WEEK / 'day1/TRAIN_ROOTS.json')['root_ids'])
        forbidden = {r for c in read(WEEK / 'day4_schedule_v1/EVALUATION_LOCK.json')['cohorts'] for r in c['root_ids']}
        roots = {q['root_id'] for q in questions}
        check('fixed_train_only_denominators', len(direct) == len({r['request_id'] for r in direct}) == 360
              and len(brief) == len(questions) == 24 and len(roots) == 6 and roots <= train_roots
              and not roots & forbidden and f['dev_test_requests'] == 0)
        check('equal_question_types_and_origins', collections.Counter(q['question_type'] for q in questions)
              == {'multi_select': 12, 'ordering': 12}
              and collections.Counter(q['source_arm'] for q in selected['question_annotations']) == {'A': 12, 'B': 12})
        pool = read(WEEK / 'day4_schedule_v1/SELECTION.json')
        for role in ['A', 'B']:
            selected_pool = set(pool['selected_new_' + role])
            check('own_and_opposite_exposure:' + role, sum(q['candidate_id'] in selected_pool for q in questions) == 12
                  and all((q['model_training_exposure'][role + '1'] == 'OWN_ARM_TRAINED_QUESTION')
                      == (q['candidate_id'] in selected_pool) for q in selected['question_annotations']))
        metadata = {q['candidate_id']: q for q in read(WEEK / 'day3_solver187_v1/CANDIDATE_METADATA.json')['questions']}
        allowed = {a: set(pool['selected_new_' + a]) for a in ['A', 'B']}
        byroot = collections.defaultdict(list)
        for slot in pool['paired_slots']:
            if slot['A'] in allowed['A'] and slot['B'] in allowed['B']:
                byroot[metadata[slot['A']]['root_id']].append(slot)
        key = lambda s: hashlib.sha256(('day5-train-diagnosis-v1|' + s).encode()).hexdigest()
        expected_roots, expected_slots = [], []
        for root in sorted(byroot, key=key):
            n = len(expected_roots)
            targets = [('multi_select', ['MC_BINDING', 'MC_FINAL_STATE_REVISION'][n % 2]),
                       ('ordering', ['ORDER_3_2_1', 'ORDER_2_2_2', 'ORDER_4_2'][n % 3])]
            choices = [[s for s in byroot[root] if metadata[s['A']]['question_type'] == typ
                        and metadata[s['A']]['design_id'] == design] for typ, design in targets]
            if all(choices):
                expected_roots.append(root)
                expected_slots.extend(min(c, key=lambda s: key(s['slot_id'])) for c in choices)
            if len(expected_roots) == 6:
                break
        check('independent_score_free_hash_selection', selected['selected_roots'] == expected_roots
              and selected['paired_slots'] == expected_slots and selected['selection_reads_scores'] is False)
        oldrows = {r['request_id']: r for r in read(OLD / 'manifest.json')['requests']}
        oldmaps = read(OLD / 'DISPLAY_MAPPINGS.json')
        maps = read(D / 'public/DIRECT_DISPLAY_MAPS.json')
        check('exact_S0_requests_prompts_maps_seeds', all(
            {k: v for k, v in r.items() if k != 'prompt_path'} == oldrows[r['request_id']]
            and Path(r['prompt_path']).read_bytes() == (OLD / 'prompts' / (r['request_id'] + '.txt')).read_bytes()
            and maps['requests'][r['request_id']] == oldmaps['requests'][r['request_id']]
            and sha(r['prompt_path']) == r['prompt_sha256'] for r in direct))
        check('brief_fixed_without_score_selection', {r['candidate_id'] for r in brief} == {q['candidate_id'] for q in questions}
              and all(r['variant_id'] == 'perm_00' and r['slot'] == 0 and r['max_output_tokens'] == 2048
                      and sha(r['prompt_path']) == r['prompt_sha256'] for r in brief))
        frozen_choices = read(D / 'public/FINITE_CHOICES.json')
        byid = {}
        for row in frozen_choices['S0_batch_bindings']:
            bound(row)
            batch = read(row['path'])
            for rid, param in zip(batch['request_ids'], batch['sampling_params']):
                if rid not in frozen_choices['request_keys']:
                    continue
                values = ast.literal_eval(re.search(r'choice=(\[[^\]]*\])', param).group(1)) if isinstance(param, str) else param['structured_outputs']['choice']
                byid[rid] = values
        check('actual_S0_choice_order_reused', len(byid) == 360 and all(
            byid[rid] == frozen_choices['choices'][choicekey]
            for rid, choicekey in frozen_choices['request_keys'].items()))
        baseline = read(D / 'private/S0_BASELINE_BINDINGS.json')
        for row in baseline['selected_raw_responses'] + [baseline['manifest'], baseline['raw_response_freeze'], baseline['gold_binding']]:
            bound(row)
        for q in questions:
            check('source:' + q['candidate_id'], sha(q['source_path']) == q['source_sha256'])
            add(q['source_path'])
        source = (D / 'run_v2.py').read_text()
        tree = ast.parse(source)
        live = ast.get_source_segment(source, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'live'))
        check('raw_before_Gold_no_retry_no_probe_live', 'Gold' not in live.replace("'Gold_opened': False", '')
              and 'gold_binding' not in live and 'NO_AUTOMATIC_RERUN_OR_OVERWRITE' in live
              and "choices=('direct', 'brief_evidence')" in source and 'CONTROLLED_PROBE_RESERVE' not in source)
        check('actual_frozen_choice_order_and_seed_in_live', "frozen_choices['choices'][frozen_choices['request_keys'][r['request_id']]]" in live
              and "'seed': r['seed']" in live and "for offset in range(0, len(rows), 8):" in live)
        reserve = read(D / 'CONTROLLED_PROBE_RESERVE.json')
        check('probes_not_released', reserve['status'] == 'PREPARED_NOT_RELEASED_NOT_EXECUTED'
              and reserve['maximum_A1_B1_requests_total'] == 24)
        for role in args.role:
            mp = D / 'models' / role / 'EXECUTION_MANIFEST.json'
            m = read(mp)
            add(mp)
            for row in m['model_bindings'] + m['tokenizer_bindings'] + m['code_bindings'] + m['endpoint_evidence_bindings'] + [m['endpoint_manifest'], m['endpoint_proof'], m['request_freeze']]:
                bound(row)
            dev = read(m['endpoint_manifest']['path'])
            review_path = HERE / ('S0_DEV_EXECUTION_REVIEW_FINAL.json' if role == 'S0' else role + '_DEV_EXECUTION_REVIEW.json')
            dev_review = read(review_path)
            add(review_path)
            check('same_reviewed_final_endpoint:' + role, m['model_role'] == role
                  and m['model_directory'] == dev['model_directory'] and m['model_bindings'] == dev['model_bindings']
                  and dev_review['status'] == 'PASS' and dev_review['failures_count'] == 0
                  and binding(m['endpoint_manifest']['path']) in dev_review['audited_files']
                  and m['request_freeze'] == binding(D / 'EXECUTION_ADDENDUM_V2.json')
                  and m['dev_test_calls'] == 0 and m['direct_requests'] == 360
                  and m['brief_evidence_requests'] == 24 and m['probe_requests_released'] == 0)
            evidence[role] = {'manifest': binding(mp), 'model_directory': m['model_directory']}
        evidence['scope'] = {'questions': 24, 'roots': sorted(roots), 'direct_per_model': 360,
                             'brief_per_model': 24, 'TEST_DEV_requests': 0, 'probe_go': False,
                             'exposure': '12 own-trained and 12 opposite-arm questions on shared roots per model; no heldout claim.'}
    except Exception as exc:
        check('completed_without_exception', False, f'{type(exc).__name__}: {exc}')
    failures = [c for c in checks if not c['pass']]
    status = 'FAIL' if failures else 'PASS' if args.role else 'PASS_CPU_PREPARATION'
    result = {'schema_version': 'day5_independent_train_diagnosis_review_v1', 'status': status,
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'audit_script': binding(__file__), 'audited_model_roles': args.role,
              'checks_count': len(checks), 'failures_count': len(failures), 'checks': checks,
              'audited_files': list(files.values()), 'evidence': evidence,
              'Solver_calls_in_audit': 0, 'controlled_probe_release': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': status, 'checks': len(checks), 'failures': len(failures), 'output': str(args.output)}))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
