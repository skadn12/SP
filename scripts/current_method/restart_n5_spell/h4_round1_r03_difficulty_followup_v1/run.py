"""Evaluate audit-accepted QA; later quality review may quarantine raw scores."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1 import (MAX_MODEL_LEN, MAX_TOKENS, ROLLOUTS,
    SINGLE_TEMPERATURE, TOP_P, candidate_request_count)
from ..h4_round0_b0_k3_difficulty_v1.gate import build_solver_contract, gate_candidate, PROGRAM_PASS
from ..h4_round0_b0_k3_difficulty_v1.io_utils import (load_json, write_once_json,
    write_once_jsonl, RunLock)
from ..h4_round0_b0_k3_difficulty_v1.request_ledger import Round0RequestLedger
from ..h4_round0_b0_k3_difficulty_v1.solver import (ensure_vllm_environment,
    evaluate_candidate, load_or_recover_candidate_result)
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_b0_k3_difficulty_v1.run import assert_gpu_idle
from ..h4_round1_paired_blind_solver_eval_v1.run import _identity_preflight
from ..h4_round1_paired_blind_solver_eval_v1 import BASE_MODEL
from ..h4_round1_r03_feedback_batch_generation_v1.run import content_key
from ..vllm_backends import VLLMSolverBackend

RUNS = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
AUDITS = {
    'baseline_omitted': RUNS/'h4_round1_r03_feedback_semantic_audit_v1_live_1',
    'balanced_non_single': RUNS/'h4_round1_r03_balanced_semantic_audit_v1_live_1',
}
OUTPUT = RUNS/'h4_round1_r03_difficulty_followup_v1_live_1'
MODELS = ('M0', 'M_clean_policy', 'M_inclusive_policy')


def eligible(record):
    """Group-only refs never enlarge the submitted evidence support set."""
    return (record.get('status') == 'completed'
        and record.get('semantic_status') == 'accept'
        and record.get('answer_semantically_valid') is True
        and (record.get('audit') or {}).get('evidence_sufficient') == 'sufficient'
        and not (set((record.get('static_review') or {}).get('training_blockers') or [])
                 - {'evidence_group_flat_mismatch'}))


def choose_inputs():
    baseline = load_json(AUDITS['baseline_omitted']/'audit_results.json')['results']
    seen = {content_key(r['candidate']) for r in baseline if r.get('training_ready')}
    rows, sources, excluded = [], [], []
    for scope, audit in AUDITS.items():
        source = audit/'audit_results.json'
        sources.append({'path': str(source), 'sha256': sha256_file(source)})
        for record in load_json(source)['results']:
            original = record['candidate']
            if scope == 'baseline_omitted' and record.get('training_ready'):
                continue
            if scope == 'balanced_non_single' and original['question_type'] == 'single_choice':
                continue
            reason = None
            if not eligible(record):
                reason = 'semantic_or_other_static_blocker'
            elif content_key(original) in seen:
                reason = 'duplicate_question_and_options'
            if reason:
                excluded.append({'scope': scope, 'candidate_id': original['candidate_id'], 'reason': reason})
                continue
            seen.add(content_key(original))
            row = dict(original)
            row.update({'source_candidate_id': original['candidate_id'], 'evaluation_scope': scope,
                'candidate_id': original['root_id']+':R03_DIFFICULTY:'+scope+':'+original['candidate_id'],
                'audit_source': str(source), 'audit_source_sha256': sha256_file(source),
                'source_static_review': record['static_review'],
                'evaluation_eligible': True, 'training_admission_performed': False})
            rows.append(row)
    return rows, sources, excluded


def prepare(out):
    from transformers import AutoTokenizer
    if out.exists():
        raise RuntimeError('output_exists_use_live_or_new_output')
    rows, sources, excluded = choose_inputs()
    if not rows:
        raise RuntimeError('no_new_evaluation_candidates')
    identity, models = _identity_preflight()
    models['M0'] = {'model_name': 'M0', 'adapter_enabled': False, 'adapter_path': None,
        'merged_model_path': str(BASE_MODEL), 'base_model_identity': identity['base_model_identity']}
    identity['models'] = models
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    out.mkdir(parents=True)
    write_once_jsonl(out/'candidate_manifest.jsonl', rows)
    write_once_json(out/'model_identity_report.json', identity)
    protocol = build_solver_contract(tokenizer, model_path=BASE_MODEL, tokenizer_path=BASE_MODEL)
    write_once_json(out/'solver_protocol.json', protocol)
    gates = []
    for root in sorted({r['root_id'] for r in rows}):
        source = AUDITS['baseline_omitted']/'roots'/root/'frozen_dialogue.json'
        d = load_json(source)
        # Both audited cohorts must use exactly the same complete dialogue.
        second = AUDITS['balanced_non_single']/'roots'/root/'frozen_dialogue.json'
        if second.exists() and load_json(second)['turns'] != d['turns']:
            raise RuntimeError('dialogue_drift:'+root)
        frozen = out/'dialogues'/(root+'.json')
        write_once_json(frozen, d)
        sources.append({'path': str(source), 'sha256': sha256_file(source)})
        for row in [r for r in rows if r['root_id'] == root]:
            gate = gate_candidate(row, d, tokenizer, root_id=root)
            gate.update({'candidate_pool_path': str(out/'candidate_manifest.jsonl'),
                'candidate_pool_sha256': sha256_file(out/'candidate_manifest.jsonl'),
                'dialogue_path': str(frozen), 'dialogue_sha256': sha256_file(frozen)})
            gates.append(gate)
    write_once_jsonl(out/'program_gate_results.jsonl', gates)
    if any(g['gate_status'] != PROGRAM_PASS for g in gates):
        raise RuntimeError('program_gate_failed')
    for model in MODELS:
        contract = {'version': 'r03_difficulty_eval_v1', 'model_identity': models[model],
                    'protocol_sha256': sha256_file(out/'solver_protocol.json')}
        write_once_json(out/model/'solver_contract.json', contract)
    manifest = {'status': 'PASS', 'candidate_count': len(rows),
        'question_type_counts': dict(Counter(r['question_type'] for r in rows)),
        'scope_counts': dict(Counter(r['evaluation_scope'] for r in rows)),
        'request_count_per_model': sum(candidate_request_count(r) for r in rows),
        'models': list(MODELS), 'sources': sources, 'excluded': excluded,
        'rollouts': ROLLOUTS, 'max_model_len': MAX_MODEL_LEN, 'max_tokens': MAX_TOKENS,
        'context_modes': ['no_context', 'full_context'],
        'policy': 'Evaluation requires supported Gold, determinate clean options and sufficient flat evidence. Group mismatch alone is diagnostic; no evidence or Gold is changed.',
        'training_admission_performed': False,
        'files': {str(f.relative_to(out)): sha256_file(f) for f in out.rglob('*') if f.is_file()}}
    write_once_json(out/'preflight_report.json', manifest)
    shutil.copyfile(__file__, out/'runner_snapshot.py')
    print(json.dumps({k:v for k,v in manifest.items() if k not in {'sources','files','excluded'}},ensure_ascii=False),flush=True)


def metrics(rows):
    result = {'candidate_count': len(rows), 'question_type_counts': dict(Counter(r['question_type'] for r in rows))}
    for mode in ('full_context', 'no_context'):
        correct = sum(r[mode+'_correct_count'] for r in rows)
        total = sum(r[mode+'_total_count'] for r in rows)
        result[mode] = {'correct': correct, 'total': total, 'accuracy': correct/total if total else None,
                       'candidate_mean': sum(r[mode+'_accuracy'] for r in rows)/len(rows) if rows else None}
    return result


def live(out):
    from transformers import AutoTokenizer
    pre = load_json(out/'preflight_report.json')
    for name, digest in pre['files'].items():
        if sha256_file(out/name) != digest:
            raise RuntimeError('frozen_file_drift:'+name)
    for source in pre['sources']:
        if sha256_file(Path(source['path'])) != source['sha256']:
            raise RuntimeError('input_drift')
    current, _ = _identity_preflight()
    frozen = load_json(out/'model_identity_report.json')
    if current['base_model_identity'] != frozen['base_model_identity']:
        raise RuntimeError('base_identity_drift')
    for model in MODELS[1:]:
        if current['models'][model] != frozen['models'][model]:
            raise RuntimeError('trained_identity_drift')
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    rows = [json.loads(line) for line in (out/'candidate_manifest.jsonl').read_text().splitlines()]
    gates = {r['candidate_id']: r for r in map(json.loads,(out/'program_gate_results.jsonl').read_text().splitlines())}
    dialogues = {r['root_id']:load_json(out/'dialogues'/(r['root_id']+'.json')) for r in rows}
    reports = {}
    with RunLock(out):
        for model in MODELS:
            md = out/model
            if (md/'report.json').exists():
                reports[model] = load_json(md/'report.json'); continue
            assert_gpu_idle(); ensure_vllm_environment()
            contract_sha = sha256_file(md/'solver_contract.json')
            ledger = Round0RequestLedger(md, solver_contract_sha256=contract_sha)
            ledger.recover_orphaned_requests()
            backend = VLLMSolverBackend(model_path=Path(frozen['models'][model]['merged_model_path']),
                tokenizer_path=BASE_MODEL, max_model_len=MAX_MODEL_LEN, temperature=SINGLE_TEMPERATURE,
                top_p=TOP_P, max_tokens=MAX_TOKENS, gpu_memory_utilization=.80)
            results, summaries = [], []
            try:
                for row in sorted(rows,key=lambda r:r['candidate_id']):
                    cid = row['candidate_id']
                    args = {'candidate':row,'expanded':dialogues[row['root_id']], 'gate_row':gates[cid],
                        'tokenizer':tokenizer,'backend':backend,'ledger':ledger,'solver_contract_sha256':contract_sha}
                    result = load_or_recover_candidate_result(run_root=md, **args)
                    if result is None:
                        result = evaluate_candidate(**args)
                        write_once_json(md/'candidate_results'/(sha256_bytes(cid.encode())+'.json'),result)
                    results.append(result)
                    summaries.append({**summarize_candidate_result(result), 'candidate_id':cid, 'root_id':row['root_id'],
                        'question_type':row['question_type'], 'evaluation_scope':row['evaluation_scope'],
                        'source_candidate_id':row['source_candidate_id'], 'model_name':model})
                    atomic_json(out/'progress.json',{'model':model,'completed':len(results),'total':len(rows)})
            finally:
                backend.close()
            counts = ledger.summary()
            if counts['completed_request_count'] != pre['request_count_per_model'] or counts['ambiguous_request_count']:
                raise RuntimeError('incomplete_solver_ledger')
            write_once_jsonl(md/'candidate_results.jsonl',results)
            write_once_jsonl(md/'summary.jsonl',summaries)
            report = {'status':'PASS','model':model,'ledger':counts,'overall':metrics(summaries),
                'by_type':{q:metrics([r for r in summaries if r['question_type']==q]) for q in {r['question_type'] for r in summaries}},
                'by_scope':{s:metrics([r for r in summaries if r['evaluation_scope']==s]) for s in pre['scope_counts']}}
            write_once_json(md/'report.json',report); reports[model]=report
            print(json.dumps(report,ensure_ascii=False),flush=True)
    write_once_json(out/'comparison.json',{'status':'COMPLETE','models':reports,'training_calls':0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase',choices=('prepare','live','all'),default='all'); parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.phase in ('prepare','all'): prepare(args.output)
    if args.phase in ('live','all'): live(args.output)
