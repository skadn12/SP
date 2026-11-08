"""Offline preparation/scoring for the two-method diagnostic feasibility pilot."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'thinking_switch_20261024_structured_v2'
MODEL = HERE.parents[2] / 'restart-n5-runs/day5_B1_schedule_v1/merged_model'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def check(b):
    actual = bind(b['path'])
    assert all(actual[k] == b[k] for k in ['sha256', 'bytes']), b['path']


def seed(text):
    return int.from_bytes(hashlib.sha256((HERE.name + ':' + text).encode()).digest()[:4], 'big') % 2147483647


def load_old():
    spec = importlib.util.spec_from_file_location('diagnostic_pilot_readonly_prior', OLD / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)


def stage_freeze(stage):
    path = HERE / stage / 'RAW_RESPONSE_FREEZE.json'
    doc = read(path)
    for b in doc['bindings']:
        check(b)
    return doc


def prepare():
    assert not (HERE / 'SELECTION_PRIVATE.json').exists(), 'NO_OVERWRITE'
    tok = tokenizer()
    scores = {r['request_id']: r for r in read(OLD / 'execution/SCORES.json')['records']}
    rows = read(OLD / 'execution/PUBLIC_REQUESTS.json')['requests']
    selected_ids = [r['candidate_id'] for r in read(OLD / 'execution/SEMANTIC_AUDIT_SELECTION.json')['records'] if r['candidate_id']]
    assert len(selected_ids) == len(set(selected_ids)) == 6
    private_rows = read(OLD.parent / 'priority_diversity_feedback_20261024_v1/execution/PRIVATE_SCORING.json')['requests']
    presentations = {r['request_id']: r['presentation'] for r in private_rows}
    selected, jobs = [], []
    for cid in selected_ids:
        choices = sorted((r for r in rows if r['candidate_id'] == cid and r['mode'] == 'THINK_OFF' and r['condition'] == 'FULL'), key=lambda r: r['variant_id'])
        eligible = [r for r in choices if scores[r['request_id']]['strict']['valid'] and not scores[r['request_id']]['strict']['exact']]
        assert eligible, cid
        old = eligible[0]
        score = scores[old['request_id']]
        check(old['source_binding'])
        check(score['raw_binding'])
        prompt = Path(old['prompt_path']).read_text()
        assert len(tok.encode(prompt, add_special_tokens=False)) == old['prompt_tokens']
        repeats = []
        for index in range(1, 5):
            rid = f'{cid}__{old["variant_id"]}__repeat_{index:02d}__methods_v1'
            pp = HERE / 'solver_inputs/prompts' / (rid + '.txt')
            sp = HERE / 'solver_inputs/schemas' / (rid + '.json')
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(prompt)
            save(sp, read(old['schema_path']))
            job = dict(old)
            job.update(request_id=rid, prompt_path=str(pp), schema_path=str(sp), prompt_binding=bind(pp), schema_binding=bind(sp), sampling_seed=seed('repeat:' + cid + ':' + str(index)), temperature=.8, top_p=.9, top_k=-1, repetition_penalty=1.0, max_tokens=6144, repeat_index=index)
            assert job['sampling_seed'] != old['sampling_seed']
            jobs.append(job)
            repeats.append(rid)
        selected.append({'candidate_id': cid, 'root_id': old['root_id'], 'old_row': old, 'base_score': score, 'presentation': presentations[old['base_request_id']], 'new_repeat_ids': repeats})
    assert len(jobs) == 24 and len({r['sampling_seed'] for r in jobs}) == 24
    save(HERE / 'SELECTION_PRIVATE.json', {'selection_rule': 'six fixed question IDs; first valid wrong THINK_OFF FULL display ordered perm_00 then perm_01', 'cases': selected})
    save(HERE / 'solver_jobs.json', {'stage': 'solver', 'jobs': jobs, 'generation_gold_access': False})
    save(HERE / 'PREPARATION.json', {'questions': 6, 'roots': 4, 'solver_new_requests': 24, 'diagnosis_requests_planned': 12, 'solver_model': str(MODEL), 'diagnostic_model': str(MODEL), 'enable_thinking': False, 'protocol_binding': bind(HERE / 'PROTOCOL_ZH.md'), 'selection_binding': bind(HERE / 'SELECTION_PRIVATE.json'), 'solver_jobs_binding': bind(HERE / 'solver_jobs.json'), 'code_binding': bind(Path(__file__)), 'old_manifest_binding': bind(OLD / 'FINAL_MANIFEST.json'), 'old_selection_binding': bind(OLD / 'execution/SEMANTIC_AUDIT_SELECTION.json'), 'cases': [{'candidate_id': c['candidate_id'], 'variant_id': c['old_row']['variant_id'], 'base_raw_binding': c['base_score']['raw_binding'], 'new_repeat_ids': c['new_repeat_ids']} for c in selected]})
    print(json.dumps({'prepared_solver_requests': len(jobs), 'question_ids': selected_ids, 'new_solver_input_tokens': sum(r['prompt_tokens'] for r in jobs)}, ensure_ascii=False))


def score_solver():
    frozen = stage_freeze('solver')
    old = load_old()
    cases = read(HERE / 'SELECTION_PRIVATE.json')['cases']
    mapping = {rid: c for c in cases for rid in c['new_repeat_ids']}
    jobs = read(HERE / 'solver_jobs.json')['jobs']
    assert len(jobs) == 24
    records = []
    for job in jobs:
        c = mapping[job['request_id']]
        p = HERE / 'solver/raw' / (job['request_id'] + '.json')
        raw = read(p)
        parsed, answer, errors, final = None, [], [], None
        try:
            assert raw['status'] == 'RETURNED', 'BACKEND_NOT_RETURNED'
            assert raw['finish_reason'] == 'stop', 'NOT_NORMAL_STOP'
            think, final, boundary = old.split_output(raw['raw_output'])
            assert not think, 'UNEXPECTED_THINKING'
            parsed, answer = old.legal_answer(final, job)
            old.orig.base.validate_output(final, job, read(job['schema_path']))
        except Exception as e:
            errors.append(str(e))
        strict = old.orig.score_record(job, c['presentation'], answer, not errors)
        records.append({'request_id': job['request_id'], 'candidate_id': job['candidate_id'], 'strict': strict, 'parsed': parsed, 'final_text': final, 'errors': errors, 'raw_binding': bind(p), 'output_tokens': len(raw.get('output_token_ids', [])), 'input_tokens': len(raw.get('prompt_token_ids', []))})
    summary = []
    for c in cases:
        group = [r for r in records if r['candidate_id'] == c['candidate_id']]
        correct = sum(r['strict']['exact'] for r in group)
        summary.append({'candidate_id': c['candidate_id'], 'new_requests': 4, 'new_valid': sum(r['strict']['valid'] for r in group), 'new_correct': correct, 'new_correct_rate': correct / 4, 'all_traces_including_base': 5, 'has_correct_and_wrong': correct > 0})
    save(HERE / 'SOLVER_SCORES.json', {'raw_freeze_binding': bind(HERE / 'solver/RAW_RESPONSE_FREEZE.json'), 'records': records, 'per_question': summary, 'interpretation': 'Four new draws per failure-conditioned question; five-trace pool includes one intentionally selected old failure.'})
    print(json.dumps({'solver_correct': sum(r['strict']['exact'] for r in records), 'requests': 24, 'valid': sum(r['strict']['valid'] for r in records), 'questions_with_success_failure_contrast': sum(r['has_correct_and_wrong'] for r in summary)}, ensure_ascii=False))


SYSTEM = '''You are a source-grounded behavioral diagnostic model. Diagnose only observable errors in the provided solver responses. Full source, question, verified Gold and one or more response trajectories are provided. Gold correctness evaluates the final answer only, not the correctness of reasons or citations. A final-correct trajectory may still contain wrong reasoning. Do not infer private cognition or stable general weaknesses from one question. Do not invent a successful trajectory if none is provided. Distinguish a supported observed error from a tentative trigger hypothesis. Give at most one candidate weakness and at most three exact response quotes. Use concise English. Your final response must be one JSON object matching the supplied schema. Do not output a thinking block.'''


def diagnosis_schema(refs):
    def string(limit=600):
        return {'type': 'string', 'maxLength': limit}
    fields = {
        'diagnosis_supported': {'type': 'boolean'},
        'observed_error': string(),
        'weakness_hypothesis': string(),
        'trigger_condition': string(),
        'expected_error_on_new_task': string(),
        'counterexample_or_boundary': string(),
        'source_turn_refs': {'type': 'array', 'items': {'type': 'string', 'enum': refs}, 'maxItems': 8, 'uniqueItems': True},
        'trace_evidence': {'type': 'array', 'maxItems': 3, 'items': {'type': 'object', 'additionalProperties': False, 'required': ['trace_id', 'quote'], 'properties': {'trace_id': string(160), 'quote': string(400)}}},
        'contrast_assessment': {'type': 'object', 'additionalProperties': False, 'required': ['has_final_correct_trace', 'success_trace_id', 'supported_difference'], 'properties': {'has_final_correct_trace': {'type': 'boolean'}, 'success_trace_id': string(160), 'supported_difference': string()}},
        'uncertainty': string(),
        'challenger_task_instruction': string(),
    }
    return {'type': 'object', 'additionalProperties': False, 'required': list(fields), 'properties': fields}


def make_diagnosis():
    assert not (HERE / 'diagnosis_jobs.json').exists(), 'NO_OVERWRITE'
    tok = tokenizer()
    cases = read(HERE / 'SELECTION_PRIVATE.json')['cases']
    results = {r['request_id']: r for r in read(HERE / 'SOLVER_SCORES.json')['records']}
    jobs, packets, blind = [], [], []
    for ci, c in enumerate(cases):
        row = c['old_row']
        source = read(row['source_binding']['path'])
        source_text = '\n'.join(f'[{t["turn_ref"]}] {t["speaker"]}: {t["text"]}' for t in source['turns'])
        schema = diagnosis_schema([t['turn_ref'] for t in source['turns']])
        sp = HERE / 'diagnosis_inputs/schemas' / (c['candidate_id'] + '.json')
        save(sp, schema)
        base = c['base_score']
        old_raw = read(base['raw_binding']['path'])
        def trajectory(score, raw):
            return {'trace_id': score['request_id'], 'contract_valid': score['strict']['valid'], 'final_answer_exact': score['strict']['exact'], 'response_text': raw.get('raw_output')}
        base_trace = trajectory(base, old_raw)
        all_traces = [base_trace] + [trajectory(results[rid], read(results[rid]['raw_binding']['path'])) for rid in c['new_repeat_ids']]
        case_packets = {}
        for arm in ['A', 'B']:
            traces = [base_trace] if arm == 'A' else all_traces
            data = {'question_id': c['candidate_id'], 'question_type': row['question_type'], 'question': row['question'], 'display': row['display'], 'verified_display_gold': c['presentation']['display_gold'], 'source_transcript': source_text, 'trajectories': traces}
            user = 'Analyze the provided material and produce a source-grounded candidate diagnostic card. If support is insufficient set diagnosis_supported=false and explain the uncertainty.\n' + json.dumps(data, ensure_ascii=False)
            prompt = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            count = len(tok.encode(prompt, add_special_tokens=False))
            assert count + 2048 <= 16384, (c['candidate_id'], arm, count)
            rid = c['candidate_id'] + '__diagnosis_' + arm + '__methods_v1'
            pp = HERE / 'diagnosis_inputs/prompts' / (rid + '.txt')
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(prompt)
            jobs.append({'request_id': rid, 'candidate_id': c['candidate_id'], 'arm': arm, 'prompt_path': str(pp), 'schema_path': str(sp), 'prompt_binding': bind(pp), 'schema_binding': bind(sp), 'prompt_tokens': count, 'sampling_seed': seed('diagnosis:' + c['candidate_id']), 'temperature': .2, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1.0, 'max_tokens': 2048})
            case_packets[arm] = data
        packets.append({'candidate_id': c['candidate_id'], 'source_binding': row['source_binding'], 'packets': case_packets})
        swap = seed('blind:' + c['candidate_id']) % 2
        blind.append({'candidate_id': c['candidate_id'], 'X': 'B' if swap else 'A', 'Y': 'A' if swap else 'B'})
        if ci % 2:
            jobs[-2:] = jobs[-2:][::-1]
    save(HERE / 'DIAGNOSIS_INPUT_PACKETS_PRIVATE.json', {'cases': packets})
    save(HERE / 'BLIND_MAPPING_PRIVATE.json', {'records': blind})
    save(HERE / 'diagnosis_jobs.json', {'stage': 'diagnosis', 'jobs': jobs, 'shared_system_prompt': SYSTEM})
    save(HERE / 'DIAGNOSIS_PREPARATION.json', {'requests': 12, 'input_binding': bind(HERE / 'DIAGNOSIS_INPUT_PACKETS_PRIVATE.json'), 'jobs_binding': bind(HERE / 'diagnosis_jobs.json'), 'solver_scores_binding': bind(HERE / 'SOLVER_SCORES.json'), 'model': str(MODEL), 'enable_thinking': False, 'max_prompt_tokens': max(j['prompt_tokens'] for j in jobs), 'no_silent_truncation': True})
    print(json.dumps({'diagnosis_requests': 12, 'max_prompt_tokens': max(j['prompt_tokens'] for j in jobs), 'input_tokens_per_arm': {arm: sum(j['prompt_tokens'] for j in jobs if j['arm'] == arm) for arm in ['A', 'B']}}, ensure_ascii=False))


def score_diagnosis():
    import jsonschema
    stage_freeze('diagnosis')
    jobs = read(HERE / 'diagnosis_jobs.json')['jobs']
    packets = {c['candidate_id']: c for c in read(HERE / 'DIAGNOSIS_INPUT_PACKETS_PRIVATE.json')['cases']}
    records = []
    for job in jobs:
        raw_path = HERE / 'diagnosis/raw' / (job['request_id'] + '.json')
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
            success_id = claimed['success_trace_id']
            mechanical = {'quote_checks': quote_checks, 'all_quotes_verbatim': all(e['trace_exists'] and e['verbatim_quote'] for e in quote_checks), 'has_quotes': bool(quote_checks), 'success_availability_claim_matches': claimed['has_final_correct_trace'] == actual_success, 'success_id_valid': (success_id in traces and traces[success_id]['final_answer_exact']) if actual_success and claimed['has_final_correct_trace'] else not success_id, 'semantic_review': 'PENDING'}
        except Exception as e:
            errors.append(str(e))
        records.append({'request_id': job['request_id'], 'candidate_id': job['candidate_id'], 'arm': job['arm'], 'contract_valid': not errors, 'card': card, 'errors': errors, 'mechanical': mechanical, 'raw_binding': bind(raw_path), 'input_tokens': len(raw.get('prompt_token_ids', [])), 'output_tokens': len(raw.get('output_token_ids', [])), 'finish_reason': raw.get('finish_reason')})
    save(HERE / 'DIAGNOSIS_SCORES.json', {'records': records, 'raw_freeze_binding': bind(HERE / 'diagnosis/RAW_RESPONSE_FREEZE.json')})
    mapping = read(HERE / 'BLIND_MAPPING_PRIVATE.json')['records']
    bykey = {(r['candidate_id'], r['arm']): r for r in records}
    review_cases = []
    for m in mapping:
        packet = packets[m['candidate_id']]['packets']['B']
        cards = []
        for label in ['X', 'Y']:
            arm = m[label]
            record = bykey[(m['candidate_id'], arm)]
            available = packets[m['candidate_id']]['packets'][arm]['trajectories']
            cards.append({'anonymous_label': label, 'contract_valid': record['contract_valid'], 'card': record['card'], 'mechanical': record['mechanical'], 'available_trace_ids': [t['trace_id'] for t in available]})
        review_cases.append({'candidate_id': m['candidate_id'], 'source_binding': packets[m['candidate_id']]['source_binding'], 'common_evidence': packet, 'cards': cards})
    save(HERE / 'SEMANTIC_REVIEW_PACKET.json', {'purpose': 'Compare source-grounded usefulness of candidate cards; method labels hidden, content can reveal method. Not a formal reproduced weakness or training-benefit experiment.', 'cases': review_cases})
    print(json.dumps({'diagnostic_responses': len(records), 'contract_valid': sum(r['contract_valid'] for r in records), 'claims_supported': sum(bool(r['card'] and r['card']['diagnosis_supported']) for r in records), 'mechanical_quote_pass': sum(r.get('mechanical', {}).get('all_quotes_verbatim', False) for r in records)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'score-solver', 'make-diagnosis', 'score-diagnosis'])
    action = parser.parse_args().action
    {'prepare': prepare, 'score-solver': score_solver, 'make-diagnosis': make_diagnosis, 'score-diagnosis': score_diagnosis}[action]()
