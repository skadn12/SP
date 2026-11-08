"""Validate completed Solver-diagnosis milestone without claiming all Day 2 done."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast
import hashlib
import json

BASE = Path(__file__).resolve().parent
checks = []


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(ok, name):
    checks.append({'check': name, 'pass': bool(ok)})
    if not ok:
        raise AssertionError(name)


def verify_binding(rec, base=None):
    p = Path(rec['path'])
    if not p.is_absolute():
        p = base / p
    check(p.is_file() and digest(p) == rec['sha256'], 'sha:' + str(p))
    if 'bytes' in rec:
        check(p.stat().st_size == rec['bytes'], 'bytes:' + str(p))


def main():
    prior = BASE.parent / 'day1'
    for row in read(prior / 'COMPLETE_MANIFEST.json')['files']:
        verify_binding(row, prior)
    for row in read(BASE / 'fact_memory/PREPARATION_FREEZE.json')['files']:
        verify_binding(row, BASE / 'fact_memory')
    summaries = {}
    for name in ['solver/direct', 'solver/evidence', 'solver/evidence_v2', 'probes/evidence_hint']:
        run = BASE / name
        m = read(run / 'manifest.json')
        s = read(run / 'SCORES.json')
        freeze = read(run / 'RAW_RESPONSE_FREEZE.json')
        for group in ['input_bindings', 'code_bindings', 'model_bindings']:
            for r in m[group]:
                verify_binding(r)
        for r in read(run / 'PREFLIGHT_FREEZE.json')['files']:
            verify_binding(r)
        for r in freeze['files']:
            verify_binding(r)
        verify_binding(freeze['manifest_binding'])
        verify_binding(s['raw_response_freeze_binding'])
        n = len(m['requests'])
        check(n == len(freeze['files']) == s['denominator_calls'] == len(s['requests']), 'fixed_denominator:' + name)
        check(s['valid_calls'] == sum(r['valid'] for r in s['requests']), 'valid_calls:' + name)
        check(s['exact_calls'] == sum(r['exact'] for r in s['requests']), 'exact_calls:' + name)
        check(freeze['created_at_utc'] <= s['gold_first_read_at_utc'], 'before_gold:' + name)
        for r in s['requests']:
            verify_binding(r['response_binding'])
            check(r['response_committed_at_utc'] <= freeze['created_at_utc'], 'committed_before_freeze:' + r['request_id'])
        summaries[name] = {'planned_records': n, 'valid': s['valid_calls'], 'exact': s['exact_calls']}
    check(summaries['solver/direct'] == {'planned_records': 582, 'valid': 582, 'exact': 199}, 'direct_final_counts')
    check(summaries['solver/evidence'] == {'planned_records': 39, 'valid': 0, 'exact': 0}, 'v1_failed_retained')
    check(summaries['solver/evidence_v2'] == {'planned_records': 39, 'valid': 38, 'exact': 7}, 'v2_final_counts')
    check(summaries['probes/evidence_hint'] == {'planned_records': 90, 'valid': 90, 'exact': 41}, 'hint_final_counts')
    for f in ['RESULTS_AUDIT_DIRECT.json', 'RESULTS_AUDIT_EVIDENCE_V1.json', 'RESULTS_AUDIT_EVIDENCE_V2.json',
              'RESULTS_AUDIT_HINT.json', 'RESULTS_AUDIT_OBSERVATIONS_02.json', 'PARENT_OBSERVED_MEMORY_RECHECK.json']:
        check(read(BASE / 'audit' / f)['status'] == 'PASS', 'audit:' + f)
    obs = read(BASE / 'analysis/results_02/OBSERVATIONS.json')
    for name, run in [('direct', 'solver/direct'), ('evidence', 'solver/evidence_v2')]:
        o = obs['runs'][name]
        check(not o['score_recomputation_errors'], 'no_score_recompute_errors:' + name)
        for r in o['requests']:
            check(Path(r['raw_binding']['path']).resolve() == (BASE / run / 'responses' / (r['request_id'] + '.json')).resolve(), 'correct_raw_binding:' + r['request_id'])
            verify_binding(r['raw_binding'])
    correction = read(BASE / 'EVIDENCE_RUNTIME_V2_CORRECTION.json')
    check(correction['old_artifacts_preserved'] and not correction['old_solver_files_modified'], 'mechanical_correction_preserved')
    check(read(BASE / 'EVIDENCE_RUNTIME_V2_GRAMMAR_PREFLIGHT.json')['status'] == 'PASS', 'compiled_grammar_pass')
    v1 = read(BASE / 'solver/evidence/manifest.json')
    v2 = read(BASE / 'solver/evidence_v2/manifest.json')
    check(v1['requests'] == v2['requests'], 'same_evidence_requests')
    for row in v1['requests']:
        file = 'prompts/' + row['request_id'] + '.txt'
        check((BASE / 'solver/evidence' / file).read_bytes() == (BASE / 'solver/evidence_v2' / file).read_bytes(), 'same_evidence_prompt:' + row['request_id'])
    for r in read(BASE / 'PROBE_POLICY_FREEZE.json')['files']:
        verify_binding(r)
    check(read(BASE / 'PROBE_POLICY_FREEZE.json')['created_at_utc'] < read(BASE / 'solver/direct/SCORING_STARTED.json')['gold_first_read_at_utc'], 'probe_policy_before_scores')
    memory = read(BASE / 'diagnosis_exports/observed_six_v002/MEMORY_B_S0_OBSERVED_v002.json')
    check(len(memory['failure_cases']) == 6 and not memory['probes'] and not memory['causes'], 'observation_only_memory')
    check(len({x['provenance']['root_id'] for x in memory['failure_cases']}) == 6, 'six_distinct_memory_roots')
    check(read(BASE / 'diagnosis_exports/observed_six_v002/CONTROLLER_VALIDATION.json')['status'] == 'PASS', 'memory_controller_pass')
    trace = read(BASE / 'analysis/order_trace_consistency/ORDER_TRACE_CONSISTENCY.json')
    check(trace['status'].startswith('COMPLETE'), 'order_trace_review_finished')
    for p in BASE.rglob('*.py'):
        if '__pycache__' in p.parts:
            continue
        ast.parse(p.read_text())
        check(True, 'python_syntax:' + str(p.relative_to(BASE)))
    fact = read(BASE / 'fact_memory/PREPARATION_FREEZE.json')
    check(fact['actual_scenes'] == fact['actual_runtime_model_calls'] == fact['memory_semantic_pass'] == 0, 'memory_pilot_preparation_only')
    result = {'status': 'PASS', 'scope': 'SOLVER_DIAGNOSIS_MILESTONE_ONLY_NOT_DAY2_COMPLETE',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'check_count': len(checks), 'error_count': 0,
              'run_counts': summaries, 'checks': checks,
              'pending': ['Fact memory initial semantic review and 54-scene three-arm execution.',
                          'Mechanism-specific controlled tests and source review before reproduced causes.',
                          'New question generation and training remain future stages.']}
    with (BASE / 'VALIDATION.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'scope', 'check_count', 'error_count']}))


if __name__ == '__main__':
    main()
