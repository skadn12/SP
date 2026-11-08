"""Independent CPU-only audit and sealing of the versioned base supplement."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
N = HERE.parent
W = N.parent
P = W / 'day5_training_v1/train_diagnosis_prepare'
CHECKS = []
HASHES = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def check(condition, name, detail=None):
    CHECKS.append({'check': name, 'pass': bool(condition), 'detail': detail})
    if not condition:
        raise RuntimeError(name + ':' + str(detail))


def verify(bindings):
    for item in bindings:
        path = Path(item['path'])
        check(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
              'FROZEN_FILE_UNCHANGED', str(path))


def stamp(value):
    return datetime.fromisoformat(value)


def mean(values):
    values = list(values)
    return sum(values, Fraction()) / len(values)


def group_metrics(public, annotations, records, arm, typ):
    questions = [q for q in public if (arm == 'ALL' or annotations[q['candidate_id']]['source_arm'] == arm)
                 and (typ == 'ALL' or q['question_type'] == typ)]
    per_question = {}
    roots = defaultdict(list)
    selected = []
    for q in questions:
        rows = [r for r in records if r['candidate_id'] == q['candidate_id']]
        selected.extend(rows)
        rate = Fraction(sum(r['exact'] for r in rows), len(rows))
        per_question[q['candidate_id']] = rate
        roots[q['root_id']].append(rate)
    out = {'questions': len(questions), 'roots': len(roots), 'planned_calls': len(selected),
           'valid_calls': sum(r['valid'] for r in selected),
           'invalid_calls': sum(not r['valid'] for r in selected),
           'exact_calls': sum(r['exact'] for r in selected),
           'question_equal_exact': mean(per_question.values()),
           'root_equal_exact': mean(mean(values) for values in roots.values()),
           'call_pooled_exact_secondary': Fraction(sum(r['exact'] for r in selected), len(selected))}
    if typ == 'multi_select':
        out['question_equal_mc_jaccard_secondary'] = mean(mean(
            Fraction(len(set(r['prediction_content'] or []) & set(r['gold_content'])),
                     len(set(r['prediction_content'] or []) | set(r['gold_content'])))
            if r['valid'] else Fraction()
            for r in records if r['candidate_id'] == q['candidate_id']) for q in questions)
    if typ == 'ordering':
        out['question_equal_order_pair_accuracy_secondary'] = mean(mean(
            Fraction(r['metrics']['correct_pairs'], r['metrics']['total_pairs'])
            for r in records if r['candidate_id'] == q['candidate_id']) for q in questions)
    return out


def save(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    check(not (HERE / 'POST_EXECUTION_AUDIT.json').exists() and not (N / 'FINAL_MANIFEST.json').exists(),
          'NO_OVERWRITE')
    preparation = read(N / 'PREPARATION_FREEZE.json')
    manifest = read(N / 'EXECUTION_MANIFEST.json')
    pre_audit = read(HERE / 'PRE_EXECUTION_AUDIT.json')
    check(pre_audit['status'] == 'PASS', 'INDEPENDENT_PRE_AUDIT_PASS')
    verify(pre_audit['audited_files'])
    verify([manifest['preparation_freeze']] + preparation['model_bindings'] + preparation['tokenizer_bindings']
           + preparation['code_bindings'] + preparation['upstream_bindings'] + preparation['prompt_bindings']
           + [preparation['cpu_validation']])
    old_final_path = W / 'day5_training_v1/FINAL_MANIFEST.json'
    old_final = read(old_final_path)
    verify(old_final['files'] + old_final['external_execution_model_artifacts'])
    check(old_final['status'] == 'D5_EXECUTED_CLOSED_SECOND_WAVE_B_NO_GO'
          and old_final['full_day5_GO'] is False and old_final['second_wave_formal_B_GO'] is False,
          'OLD_DAY5_DECISION_UNCHANGED')
    rows = read(P / 'public/DIRECT_REQUESTS.json')['requests']
    public = read(P / 'public/PUBLIC_QUESTIONS.json')['questions']
    annotations = {x['candidate_id']: x for x in read(P / 'SELECTION.json')['question_annotations']}
    row_by_id = {x['request_id']: x for x in rows}
    choices = read(P / 'public/FINITE_CHOICES.json')
    s0bindings = read(P / 'private/S0_BASELINE_BINDINGS.json')
    check(len(rows) == len(row_by_id) == 360 and len(public) == len(annotations) == 24,
          'SAME_FROZEN_PANEL_DENOMINATOR')
    check(Counter(x['source_arm'] for x in annotations.values()) == Counter({'A': 12, 'B': 12}),
          'A_B_TWELVE_QUESTIONS_EACH')
    oldparams = {}
    for item in choices['S0_batch_bindings']:
        batch = read(item['path'])
        for rid, params in zip(batch['request_ids'], batch['sampling_params']):
            if rid in row_by_id:
                oldparams[rid] = params
    run = N / 'direct'
    complete = read(run / 'COMPLETE.json')
    rawfreeze = read(run / 'RAW_RESPONSE_FREEZE.json')
    started = read(run / 'STARTED.json')
    scoring = read(N / 'SCORING_STARTED.json')
    check(complete['status'] == 'COMPLETE' and complete['failure'] is None
          and complete['actual_submitted_requests'] == complete['returned_requests'] == 360,
          'SUCCESSFUL_360_OF_360')
    check(not (run / 'FAILED.json').exists(), 'NO_FAILED_RUN')
    verify(rawfreeze['files'] + [rawfreeze['request_freeze'], rawfreeze['execution_manifest']])
    verify([scoring['raw_freeze'], scoring['code']])
    check(rawfreeze['status'] == 'COMMITTED' and rawfreeze['planned_request_count'] == 360
          and rawfreeze['gold_opened_by_live'] is False, 'TERMINAL_RAW_FREEZE')
    check(stamp(started['started_at_utc']) < stamp(complete['completed_at_utc'])
          <= stamp(rawfreeze['created_at_utc']) < stamp(scoring['gold_first_read_at_utc']),
          'ALL_RAW_FROZEN_BEFORE_FIRST_GOLD_PARSE')
    submitted_files = sorted((run / 'batches').glob('*_SUBMITTED.json'))
    completed_files = sorted((run / 'batches').glob('*_COMPLETE.json'))
    check(len(submitted_files) == len(completed_files) == 45, 'EXACTLY_45_BATCHES')
    check(len(list((run / 'batches').glob('*.json'))) == 90, 'NO_EXTRA_OUTPUT_BATCH_FILES')
    submitted_ids = []
    for index, (submitted_path, completed_path) in enumerate(zip(submitted_files, completed_files)):
        submitted = read(submitted_path)
        completed = read(completed_path)
        expected = [r['request_id'] for r in rows[index * 8:index * 8 + 8]]
        check(submitted['request_ids'] == completed['request_ids'] == expected
              and len(submitted['sampling_params']) == 8, 'EXACTLY_EIGHT_EXPECTED_OUTPUTS', index)
        check(submitted['automatic_retry'] is False, 'BATCH_NO_AUTOMATIC_RETRY', index)
        check(stamp(submitted['submitted_at_utc']) <= stamp(completed['completed_at_utc'])
              <= stamp(rawfreeze['created_at_utc']), 'BATCH_COMPLETED_BEFORE_FREEZE', index)
        for rid, params in zip(submitted['request_ids'], submitted['sampling_params']):
            check(params == oldparams[rid], 'COMPLETE_SAMPLING_PARAMS_IDENTICAL_TO_ACTUAL_S0', rid)
        submitted_ids.extend(submitted['request_ids'])
    check(submitted_ids == [r['request_id'] for r in rows] and len(set(submitted_ids)) == 360,
          'NO_DUPLICATE_OR_RETRIED_SUBMISSION')
    expected_raw = {run / 'responses' / (rid + '.json') for rid in row_by_id}
    check({Path(x['path']) for x in rawfreeze['files']} == expected_raw
          and set((run / 'responses').glob('*.json')) == expected_raw, 'RAW_FILE_SET_EXACT360')
    s0paths = {Path(x['path']).stem: Path(x['path']) for x in s0bindings['selected_raw_responses']}
    rawpaths = {'BASE': {rid: run / 'responses' / (rid + '.json') for rid in row_by_id}, 'S0': s0paths}
    for role in ('A1', 'B1'):
        old_rawfreeze = read(P / 'models' / role / 'direct/RAW_RESPONSE_FREEZE.json')
        verify(old_rawfreeze['files'])
        rawpaths[role] = {Path(x['path']).stem: Path(x['path']) for x in old_rawfreeze['files']}
    rawmodels = {}
    identity = ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed', 'prompt_sha256')
    for role, paths in rawpaths.items():
        check(set(paths) == set(row_by_id), 'FOUR_MODELS_IDENTICAL360_REQUEST_IDS', role)
        rawmodels[role] = {}
        for row in rows:
            rid = row['request_id']
            raw = read(paths[rid])
            s0raw = read(s0paths[rid])
            check(all(raw[k] == row[k] for k in identity), 'RAW_REQUEST_IDENTITY', role + ':' + rid)
            check(raw['prompt_token_ids'] == s0raw['prompt_token_ids']
                  and len(raw['prompt_token_ids']) == row['prompt_tokens'],
                  'RAW_EXACT_PROMPT_TOKEN_IDS_IDENTICAL_S0', role + ':' + rid)
            if role == 'BASE':
                check(raw['status'] == 'RETURNED' and raw['model_role'] == 'BASE'
                      and raw['model_directory'] == '/root/autodl-tmp/Qwen3-4B'
                      and raw['gold_opened_by_solver'] is False and not raw['output_validation_errors'],
                      'BASE_RAW_VALID_MODEL_STATUS_AND_NO_GOLD', rid)
                check(len(raw['all_completions']) == 1 and raw['metadata']['finish_reason'] in ('stop', 'eos')
                      and raw['raw_output'] in choices['choices'][choices['request_keys'][rid]],
                      'BASE_SINGLE_COMPLETE_FINITE_ANSWER', rid)
                check(stamp(raw['response_committed_at_utc']) <= stamp(rawfreeze['created_at_utc']),
                      'BASE_RAW_COMMIT_PRECEDES_FREEZE', rid)
            rawmodels[role][rid] = raw
    # This independent auditor parses Gold only after verifying all terminal raw records and their freezes.
    verify([s0bindings['gold_binding']])
    gold = {q['candidate_id']: q for q in read(s0bindings['gold_binding']['path'])['questions']}
    maps = read(P / 'public/DIRECT_DISPLAY_MAPS.json')
    sys.path.insert(0, str(W / 'day3_solver187_v1/runtime'))
    spec = importlib.util.spec_from_file_location('independent_base_panel_original_score',
                                                 W / 'day3_solver187_v1/runtime/score.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    comparison = read(N / 'COMPARISON.json')
    base_scores = read(N / 'BASE_SCORES.json')
    oldscores = {role: read(P / 'models' / role / 'private/DIRECT_SCORES.json')
                 for role in ('A1', 'B1')}
    independently_recomputed = {}
    for role in ('BASE', 'S0', 'A1', 'B1'):
        records = [scorer.score_one(row, rawmodels[role][row['request_id']],
                   maps['requests'][row['request_id']], maps['public_to_canonical'][row['candidate_id']],
                   gold[row['candidate_id']]['gold']) for row in rows]
        savedrecords = (base_scores['records'] if role == 'BASE' else
                        oldscores['A1']['models']['S0']['requests'] if role == 'S0' else
                        oldscores[role]['models'][role]['requests'])
        check(records == savedrecords, 'ALL360_RECORDS_RECOMPUTED_FROM_RAW_EQUAL_SAVED', role)
        question_metrics = {q['candidate_id']: scorer.summarize_question(
            q, [r for r in records if r['candidate_id'] == q['candidate_id']]) for q in public}
        for savedquestion in comparison['models'][role]['questions']:
            check(all(savedquestion[k] == value for k, value in question_metrics[savedquestion['candidate_id']].items()),
                  'PER_QUESTION_METRICS_RECOMPUTED_FROM_RAW', role + ':' + savedquestion['candidate_id'])
        groups = {}
        for arm in ('ALL', 'A', 'B'):
            for typ in ('ALL', 'multi_select', 'ordering'):
                key = arm + '/' + typ
                actual = group_metrics(public, annotations, records, arm, typ)
                expected = comparison['models'][role]['groups'][key]
                for metric, value in actual.items():
                    if isinstance(value, Fraction):
                        check(value == Fraction(expected[metric]['fraction'])
                              and abs(float(value) - expected[metric]['value']) < 1e-12,
                              'INDEPENDENT_EXACT_RATIONAL_GROUP_METRIC', role + ':' + key + ':' + metric)
                    else:
                        check(value == expected[metric], 'INDEPENDENT_GROUP_COUNTS', role + ':' + key + ':' + metric)
                groups[key] = {k: {'fraction': str(v), 'value': float(v)} if isinstance(v, Fraction) else v
                               for k, v in actual.items()}
        independently_recomputed[role] = groups
    with (N / 'per_question.csv').open(newline='') as stream:
        csvrows = list(csv.DictReader(stream))
    check(len(csvrows) == 24 and {r['candidate_id'] for r in csvrows} == {q['candidate_id'] for q in public},
          'CSV_EXACT24')
    for row in csvrows:
        for role in ('BASE', 'S0', 'A1', 'B1'):
            q = next(q for q in comparison['models'][role]['questions'] if q['candidate_id'] == row['candidate_id'])
            check(Fraction(row[role + '_exact_fraction']) == Fraction(q['exact_calls'], q['denominator_calls'])
                  and int(row[role + '_exact_calls']) == q['exact_calls'], 'CSV_MATCHES_RECOMPUTED_QUESTION', role + ':' + row['candidate_id'])
    inputs = sum(x['metadata']['prompt_token_count'] for x in rawmodels['BASE'].values())
    outputs = sum(x['metadata']['completion_token_count'] for x in rawmodels['BASE'].values())
    usage = comparison['actual_new_usage']
    check(inputs == usage['input_tokens'] == 2697606 and outputs == usage['output_tokens'] == 4204,
          'ACTUAL_NEW_USAGE_FROM_RAW')
    check(usage['new_model_calls'] == 360 and usage['reused_old_response_records'] == 1080
          and usage['old_S0_A1_B1_new_calls'] == usage['new_brief_evidence_calls'] == usage['new_dev_test_calls'] == 0,
          'ONLY_ORIGINAL_BASE_NEW_CALLS')
    check(complete['wall_seconds'] == usage['wall_seconds_including_model_load_and_close'], 'WALL_TIME_ACCOUNTING')
    check(comparison['interpretation']['independent_heldout_generalization'] is False
          and comparison['interpretation']['formal_feedback_causal_benefit_established_by_this_comparison'] is False,
          'SCIENTIFIC_BOUNDARIES_PRESERVED')
    audited = [binding(p) for p in sorted(N.rglob('*')) if p.is_file()]
    report = {'schema_version': 'day5_original_base_panel_independent_post_execution_audit_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
              'reviewer': '/root/base_panel_independent_check', 'checks_count': len(CHECKS),
              'failed_checks': 0, 'checks': CHECKS, 'audited_files': audited,
              'independent_raw_recomputation': {'models': 4, 'raw_records': 1440,
                  'groups_per_model': 9, 'all_primary_and_secondary_metrics_match': True,
                  'groups': independently_recomputed},
              'execution': {'batches': 45, 'requests': 360, 'valid': 360,
                  'all_sampling_parameter_reprs_identical_to_S0': True, 'automatic_retries': 0,
                  'truncations': 0, 'unexpected_output_counts': 0, 'raw_freeze_precedes_gold': True,
                  'input_tokens': inputs, 'output_tokens': outputs,
                  'wall_seconds': complete['wall_seconds']},
              'old_day5': {'manifest': binding(old_final_path), 'sealed_files_verified_unchanged': len(old_final['files']),
                  'external_model_execution_files_verified_unchanged': len(old_final['external_execution_model_artifacts']),
                  'status': old_final['status'], 'full_day5_GO': False, 'second_wave_formal_B_GO': False},
              'precheck_fault_path_note_resolution': 'All45 batches completed with exactly8 expected outputs. No extra-output fault occurred.',
              'scientific_boundary': 'Existing24 Train diagnosis questions; own-arm trained, opposite-arm shares roots and facts. Not independent heldout generalization or causal proof of feedback efficacy.'}
    save(HERE / 'POST_EXECUTION_AUDIT.json', report)
    files = [binding(p) for p in sorted(N.rglob('*')) if p.is_file() and p.name != 'FINAL_MANIFEST.json']
    final = {'schema_version': 'day5_original_base_panel_supplement_closed_manifest_v1',
             'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'BASE_FIXED24_SUPPLEMENT_EXECUTED_AUDITED_CLOSED',
             'execution_records_closed': True, 'new_original_base_requests': 360,
             'all_original_base_requests_valid': True, 'questions': 24, 'train_roots': 6,
             'old_day5_status': old_final['status'], 'old_day5_full_GO': False,
             'old_day5_second_wave_formal_B_GO': False, 'old_day5_decision_changed': False,
             'independent_heldout_generalization': False, 'formal_feedback_causal_benefit_established': False,
             'new_DEV_TEST_requests': 0, 'new_brief_evidence_requests': 0, 'new_controlled_probe_requests': 0,
             'new_training_runs': 0, 'post_execution_audit': binding(HERE / 'POST_EXECUTION_AUDIT.json'),
             'files_count': len(files), 'bytes_total': sum(f['bytes'] for f in files), 'files': files,
             'exclusions': ['This FINAL_MANIFEST.json itself; no running processes or unfinished run artifacts.'],
             'original_model_weights_duplicated': False}
    save(N / 'FINAL_MANIFEST.json', final)
    print(json.dumps({'status': 'PASS', 'checks': len(CHECKS), 'sealed_files': len(files),
                      'sealed_bytes': final['bytes_total'], 'audit_sha256': sha(HERE / 'POST_EXECUTION_AUDIT.json'),
                      'final_manifest_sha256': sha(N / 'FINAL_MANIFEST.json'),
                      'primary_table': {role: {arm: independently_recomputed[role][arm + '/ALL']['question_equal_exact']
                                               for arm in ('A', 'B', 'ALL')} for role in independently_recomputed}}))


if __name__ == '__main__':
    main()
