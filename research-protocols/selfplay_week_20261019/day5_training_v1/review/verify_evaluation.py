#!/usr/bin/env python3
"""CPU-only independent audit of Day 5 DEV requests and final model bindings.

With no --role, audit only preparation and do not authorize live execution.
With explicit --role options, also inspect those final endpoint manifests.
No model generation, scoring or GPU operation is performed here.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAY5 = HERE.parent
WEEK = DAY5.parent
EVAL = DAY5 / 'evaluation'


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def textsha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def canon(v):
    return textsha(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


class Audit:
    def __init__(self):
        self.checks = []
        self.files = {}
        self.evidence = {}

    def check(self, name, value, detail=None):
        self.checks.append({'check': name, 'pass': bool(value), 'detail': detail})

    def bound(self, row):
        actual = binding(row['path'])
        self.check('bound:' + row['path'], actual == row)
        self.files[row['path']] = actual

    def add(self, p):
        row = binding(p)
        self.files[row['path']] = row
        return row


def audit_requests(a):
    freeze = read(EVAL / 'DEV_REQUEST_FREEZE.json')
    a.add(EVAL / 'DEV_REQUEST_FREEZE.json')
    for row in freeze['input_bindings'] + freeze['artifact_bindings'] + [freeze['wrapper']]:
        a.bound(row)
    rows = lines(EVAL / 'DEV_REQUESTS.jsonl')
    public = {q['candidate_id']: q for q in lines(EVAL / 'DEV_PUBLIC_QUESTIONS.jsonl')}
    gold = {q['candidate_id']: q for q in lines(EVAL / 'private/DEV_GOLD.jsonl')}
    maps = read(EVAL / 'DEV_DISPLAY_MAPS.json')
    a.check('all_DEV_direct_full_context', all(r['split'] == 'dev' and r['branch'] == 'direct'
            and r['context_mode'] == 'full_context' for r in rows))
    a.check('fixed_denominators', len(rows) == len({r['request_id'] for r in rows}) == 660
            and len(public) == len(gold) == 42 and len(maps) == 348)
    a.check('freeze_scope', freeze['test_requests'] == freeze['supplement12_requests'] == 0
            and freeze['automatic_retry'] is False and freeze['source_truncation'] is False
            and freeze['model_roles'] == ['S0', 'A1', 'B1']
            and freeze['checkpoint_selection'] == 'FIXED_FINAL_OPTIMIZER_STEP_56_NO_DEV_SELECTION')
    lock = read(WEEK / 'day4_schedule_v1/EVALUATION_LOCK.json')
    hist = next(c for c in lock['cohorts'] if c['cohort_id'] == 'historical115')
    prev = next(c for c in lock['cohorts'] if c['cohort_id'] == 'previous4MC')
    histroot = Path(hist['request_freeze']['path']).parent / 'evidence'
    hr = [r for r in lines(histroot / 'ADMITTED_REQUESTS.jsonl') if r['split'] == 'dev']
    hp = {q['candidate_id']: q for q in lines(histroot / 'ADMITTED_PRIVATE.jsonl') if q['split'] == 'dev'}
    hpub = {q['candidate_id']: q for q in lines(histroot / 'ADMITTED_PUBLIC.jsonl') if q['candidate_id'] in hp}
    hprompts = {p['prompt_id']: p for p in lines(histroot / 'ADMITTED_PROMPTS.jsonl') if p['candidate_id'] in hp}
    hm = {m['prompt_id']: m for m in read(histroot / 'ADMITTED_DISPLAY_MAPS.json')['variants'] if m['split'] == 'dev'}
    actual_h = [r for r in rows if r['cohort_id'] == 'historical115_DEV']
    a.check('historical_exact_request_reuse', len(actual_h) == len(hr) == 624 and all(
        all(r[k] == v for k, v in original.items())
        and r['prompt_tokens'] == hprompts[r['prompt_id']]['prompt_tokens']
        for r, original in zip(actual_h, hr)))
    a.check('historical_exact_questions_gold_maps', all(
        public[cid] == {'cohort_id': 'historical115_DEV', 'split': 'dev', **q}
        and gold[cid] == {'cohort_id': 'historical115_DEV', 'root_id': q['root_id'],
                         'question_type': q['question_type'], **hp[cid]}
        for cid, q in hpub.items()) and all(maps[pid] == m for pid, m in hm.items()))
    a.check('historical_exact_prompt_bytes', all(Path(r['prompt_file']).read_bytes()
            == hprompts[r['prompt_id']]['prompt'].encode() for r in actual_h))
    prevroot = Path(prev['request_freeze']['path']).parent
    pr = [r for r in lines(prevroot / 'REQUESTS.jsonl') if r['original_split'] == 'dev']
    actual_p = [r for r in rows if r['cohort_id'] == 'previous2MC_DEV']
    a.check('previous_exact_request_reuse', len(actual_p) == len(pr) == 36 and all(
        all(r[k] == original[k] for k in ['request_id', 'candidate_id', 'root_id',
                                        'question_type', 'variant_id', 'sample_index', 'prompt_tokens'])
        and r['sampling'] == {'temperature': original['temperature'], 'top_p': original['top_p'],
            'max_tokens': original['max_output_tokens'], 'seed': original['seed'],
            'stop': original['stop'], 'include_stop_str_in_output': original['include_stop_str_in_output']}
        and Path(r['prompt_file']).read_bytes() == original['prompt'].encode()
        and maps[r['prompt_id']] == {'candidate_id': r['candidate_id'], **original['display']}
        for r, original in zip(actual_p, pr)))
    for cid in {r['candidate_id'] for r in pr}:
        original = read(prevroot.parent / 'content/private' / (cid + '.json'))
        q = public[cid]
        ids = {k: 'content_' + textsha(v.strip())[:16] for k, v in q['options'].items()}
        a.check('previous_exact_gold:' + cid, gold[cid]['gold'] == original['gold']
                and gold[cid]['gold_content_ids'] == [ids[k] for k in original['gold']])
    a.check('all_prompt_hashes', all(sha(r['prompt_file']) == r['prompt_sha256'] for r in rows))
    a.check('sample_contract_native', all(r['sampling']['temperature'] == .8
            and r['sampling']['top_p'] == .9 and r['sampling']['max_tokens'] == 32
            and r['sampling']['stop'] == ['</box>']
            and r['sampling']['include_stop_str_in_output'] is True for r in rows))
    primary = [q for q in gold.values() if q['reporting_role'] == 'HISTORICAL_REGRESSION_PRIMARY']
    controls = [q for q in gold.values() if q['reporting_role'] == 'GOLD6_CONTROL_ONLY']
    a.check('separate_primary_controls_previous', len(primary) == 35 and len(controls) == 5
            and collections.Counter(q['question_type'] for q in primary)
            == {'multi_select': 19, 'ordering': 16}
            and len([q for q in gold.values() if q['reporting_role'] == 'SEPARATE_PREVIOUS_MC_SUPPLEMENT']) == 2)
    training_roots = {r['root_id'] for arm in ['A', 'B'] for r in lines(WEEK / f'day4_schedule_v1/TRAIN_POOL_{arm}.jsonl')}
    eval_roots = {q['root_id'] for q in public.values()}
    a.check('training_roots_disjoint_from_DEV', not training_roots & eval_roots,
            sorted(training_roots & eval_roots))
    source = Path(freeze['wrapper']['path']).read_text()
    amendment = read(EVAL / 'EXECUTOR_ADDENDUM.json')
    a.add(EVAL / 'EXECUTOR_ADDENDUM.json')
    for key in ['original_wrapper', 'corrected_wrapper', 'unchanged_request_freeze']:
        a.bound(amendment[key])
    repaired = source.replace(
        "def verify():\n", "def verify():\n"
        "    amendment = read(HERE / 'EXECUTOR_ADDENDUM.json')\n"
        "    require(amendment['status'] == 'CPU_EXECUTOR_ENUMERATION_CHECK_REPAIR', 'EXECUTOR_ADDENDUM_INVALID')\n"
        "    verify_bindings([amendment['original_wrapper'], amendment['corrected_wrapper'], amendment['unchanged_request_freeze']])\n")
    repaired = repaired.replace(
        "require(a.choices(typ, labels) == frozen_choices[typ], 'NATIVE_CHOICE_SPACE_DRIFT')",
        "require(len(a.choices(typ, labels)) == len(frozen_choices[typ]) and set(a.choices(typ, labels)) == set(frozen_choices[typ]), 'NATIVE_CHOICE_SPACE_DRIFT')")
    a.check('v2_only_narrow_choice_check_and_hash_guard_repair',
            Path(amendment['corrected_wrapper']['path']).read_text() == repaired
            and amendment['unchanged_request_freeze'] == binding(EVAL / 'DEV_REQUEST_FREEZE.json')
            and amendment['new_solver_calls'] == 0)
    tree = ast.parse(source)
    live = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'live')
    score = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    live_source, score_source = ast.get_source_segment(source, live), ast.get_source_segment(source, score)
    a.check('live_no_gold_read', 'DEV_GOLD' not in live_source and "run / 'private" not in live_source)
    a.check('raw_before_gold_and_fixed_failure_denominator',
            "freeze = read(run / 'RAW_RESPONSE_FREEZE.json')" in score_source
            and score_source.index("verify_bindings(freeze['files']") < score_source.index("gold =")
            and "raw['status'] == 'RETURNED'" in score_source
            and "'partial_score': float(Fraction(s['partial_score_fraction'])) if valid else 0.0" in score_source
            and "for r in rows:" in score_source)
    a.check('no_retry_and_shared_requests', "ALREADY_STARTED_NO_AUTOMATIC_RETRY" in live_source
            and "rows = lines(HERE / 'DEV_REQUESTS.jsonl')" in live_source
            and 'while ' not in live_source and 'retry(' not in live_source)
    a.evidence['request_summary'] = {'questions': len(public), 'requests_per_model': len(rows),
        'unique_prompts': len(maps), 'planned_prompt_tokens': sum(r['prompt_tokens'] for r in rows),
        'test_requests': 0, 'historical_primary': 35, 'gold6_controls': 5, 'separate_previous': 2,
        'root_overlap': sorted(training_roots & eval_roots)}
    return freeze


def audit_merge_plan(a):
    p = DAY5 / 'MERGE_PLAN.json'
    plan = read(p)
    a.add(p)
    for key in ['merge_script', 'merge_python_binary', 'base_manifest']:
        a.bound(plan[key])
    source = Path(plan['merge_script']['path']).read_text()
    a.check('merge_arithmetic_full_lora_once',
            'scaling = float(alpha) / float(rank)' in source
            and 'delta = torch.matmul(b, a) * scaling' in source
            and 'module.weight.data.add_(delta.to(dtype=module.weight.dtype))' in source)
    for role, stage in plan['stages'].items():
        argv = stage['argv']
        a.check('merge_argv:' + role, argv[argv.index('--model') + 1] == '/root/autodl-tmp/Qwen3-4B'
                and argv[argv.index('--adapter-dir') + 1] == stage['final_adapter']
                and argv[argv.index('--output-dir') + 1] == stage['output_dir']
                and argv[argv.index('--dtype') + 1] == 'bfloat16' and '--overwrite' not in argv)
    return plan


def audit_role(a, role, plan):
    mp = EVAL / 'models' / role / 'EXECUTION_MANIFEST.json'
    m = read(mp)
    a.add(mp)
    for row in [m['request_freeze'], m['endpoint_proof']] + m['endpoint_evidence_bindings'] + m['model_bindings'] + m['tokenizer_bindings'] + m['code_bindings']:
        a.bound(row)
    a.check('manifest_contract:' + role, m['model_role'] == role and m['request_count'] == 660
            and m['test_requests'] == 0 and m['automatic_retry'] is False
            and m['batch_size'] == 8 and m['max_model_len'] == 16384
            and m['request_freeze'] == binding(EVAL / 'DEV_REQUEST_FREEZE.json'))
    a.check('actual_v2_executor_bound:' + role,
            binding(EVAL / 'run_v2.py') in m['code_bindings'])
    modeldir = Path(m['model_directory'])
    a.check('all_model_files_bound:' + role, {x['path'] for x in m['model_bindings']}
            == {str(p.resolve()) for p in modeldir.iterdir() if p.is_file()})
    proof = read(m['endpoint_proof']['path'])
    a.check('proof_role_path:' + role, proof['model_role'] == role
            and Path(proof['model_directory']).resolve() == modeldir.resolve())
    if role == 'S0':
        original = read(WEEK / 'day1/MODEL_BINDINGS.json')['models']['S0_M_clean_merged']
        a.check('S0_original_directory', modeldir == Path(original['directory']))
        for row in original['files']:
            a.bound(row)
    else:
        a.check('step56_independent_lineage:' + role,
                proof['fixed_final_optimizer_step'] == proof['completed_optimizer_steps'] == 56
                and proof['training_completed'] is True and proof['independent_initialization'] == 'S0'
                and proof['merge_base'] == '/root/autodl-tmp/Qwen3-4B'
                and proof['adapter_representation'] == 'FULL_UPDATED_S0_CUSTOM_LORA')
        stage = plan['stages'][role]
        info = read(modeldir / 'merged_adapter_info.json')
        config = read(Path(stage['final_adapter']) / 'adapter_config.json')
        a.check('actual_merge_metadata:' + role,
                modeldir == Path(stage['output_dir']) and info['base_model'] == '/root/autodl-tmp/Qwen3-4B'
                and info['adapter_dir'] == stage['final_adapter'] and info['adapter_config'] == config
                and info['merge_info'] == {'merged_linear_modules': 252, 'skipped_missing_pairs': []}
                and config['step'] == 56 and config['metrics'] == {'fixed_final_endpoint': True})
        tensor_audit_path = HERE / (role + '_MERGED_TENSOR_AUDIT.json')
        tensor_audit = read(tensor_audit_path)
        a.add(tensor_audit_path)
        a.check('CPU_merged_weight_arithmetic_spotcheck:' + role,
                tensor_audit['status'] == 'PASS' and tensor_audit['tensor_count'] == 6
                and tensor_audit['adapter_binding'] == binding(Path(stage['final_adapter']) / 'adapter_model.pt'))
        training_audit_path = HERE / 'INDEPENDENT_TRAINING_AUDIT.json'
        if not training_audit_path.exists():
            training_audit_path = HERE / (role + '_TRAINING_AUDIT.json')
        ta = read(training_audit_path)
        a.add(training_audit_path)
        a.check('independent_training_audit:' + role,
                ta['status'] in ['PASS_INDEPENDENT_TRAINING_AUDIT', 'PENDING_REAL_OUTPUTS']
                and ta['failures_count'] == 0
                and ta['evidence']['formal_' + role[0] + '_result'] == binding(stage['training_result'])
                and all(binding(row['path']) == row for row in
                    ta['evidence']['formal_' + role[0] + '_adapter_files']))
    a.evidence['endpoint_' + role] = {'manifest': binding(mp), 'model_directory': str(modeldir),
                                    'files': len(m['model_bindings'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['S0', 'A1', 'B1'], action='append', default=[])
    p.add_argument('--output', type=Path, default=HERE / 'DEV_REQUEST_INDEPENDENT_AUDIT.json')
    args = p.parse_args()
    a = Audit()
    try:
        audit_requests(a)
        plan = audit_merge_plan(a)
        for role in args.role:
            audit_role(a, role, plan)
    except Exception as exc:
        a.check('audit_completed_without_exception', False, f'{type(exc).__name__}: {exc}')
    failures = [c for c in a.checks if not c['pass']]
    status = 'FAIL' if failures else 'PASS' if args.role else 'PASS_CPU_REQUEST_FREEZE'
    result = {'schema_version': 'day5_independent_DEV_execution_review_v1', 'status': status,
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'audited_model_roles': args.role, 'live_execution_go': bool(args.role) and not failures,
        'audit_script': binding(__file__), 'checks_count': len(a.checks), 'failures_count': len(failures),
        'checks': a.checks, 'audited_files': list(a.files.values()), 'evidence': a.evidence,
        'scope': 'CPU-only review, no Solver, scoring of real answers, or GPU calls',
        'limitations': ['All DEV roots are historical; no unseen-root generalization claim.',
                       'This review authorizes only explicitly named final endpoints and frozen DEV requests.']}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': status, 'checks': len(a.checks), 'failures': len(failures),
                      'output': str(args.output)}))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
