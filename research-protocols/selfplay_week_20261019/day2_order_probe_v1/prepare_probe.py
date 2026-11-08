"""Prepare a fixed matched instruction probe without parsing private Gold."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
DAY2 = HERE.parent / 'day2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(DAY2 / 'solver'))
import adapter as a
import run as rt


def selected_questions(public, policy):
    candidates = [q for q in public if q['question_type'] == 'ordering']
    discovery = set(policy['selection']['discovery_root_ids'])
    assert len(candidates) == 20
    other = sorted((q for q in candidates if q['root_id'] not in discovery),
                   key=lambda q: hashlib.sha256(('2026101900|order-construction-replication|' + q['candidate_id']).encode()).hexdigest())
    assert len(other) == 17
    chosen = other[:6]
    rows = [(q, 'discovery') for q in candidates if q['root_id'] in discovery]
    rows += [(q, 'replication') for q in chosen]
    rows.sort(key=lambda x: (0 if x[1] == 'discovery' else 1, x[0]['candidate_id']))
    assert len(rows) == len({q['root_id'] for q, _ in rows}) == 9
    return rows


def treatment_prompt(control, addition):
    marker = '<|im_end|>\n<|im_start|>assistant'
    assert control.count(marker) == 1
    point = control.index(marker)
    return control[:point] + addition + control[point:]


def main():
    target = HERE / 'run'
    assert not target.exists(), 'Never overwrite a prepared run'
    assert not (HERE / 'POLICY_FREEZE.json').exists(), 'Never refreeze this policy'
    policy = rt.read(HERE / 'PROTOCOL.json')
    prior = rt.read(DAY2 / 'COMPLETE_MANIFEST.json')
    assert prior['sealed'] and prior['status'] == 'SOLVER_DIAGNOSIS_MILESTONE_COMPLETE'
    # Recheck the complete old snapshot, including failures, before deriving a new version.
    rt.verify_bindings([{**row, 'path': str(DAY2 / row['path'])} for row in prior['files']])
    baseline = rt.read(DAY2 / 'solver/evidence_v2/manifest.json')
    public = rt.read(DAY2 / 'solver/evidence_v2/PUBLIC_QUESTIONS.json')['questions']
    selection = selected_questions(public, policy)
    rt.save(HERE / 'SELECTION.json', {
        'status': 'FIXED_BEFORE_NEW_OUTPUTS', 'created_at_utc': rt.now(),
        'questions': [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'cohort': c} for q, c in selection],
        'uses_new_scores': False, 'historical_exposure': policy['selection']['replication_exposure'],
    })
    rt.save(HERE / 'POLICY_FREEZE.json', {
        'status': 'FROZEN_BEFORE_NEW_INFERENCE', 'created_at_utc': rt.now(),
        'files': [rt.bind(HERE / 'PROTOCOL.json'), rt.bind(HERE / 'SELECTION.json')],
        'prior_sealed_snapshot': rt.bind(DAY2 / 'COMPLETE_MANIFEST.json'),
        'new_solver_calls_at_freeze': 0,
    })
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(rt.TOKENIZER), local_files_only=True)
    rt.verify_bindings(baseline['model_bindings'])
    rt.verify_bindings(baseline['code_bindings'])
    grammar = rt.read(DAY2 / 'EVIDENCE_RUNTIME_V2_GRAMMAR_PREFLIGHT.json')
    assert grammar['status'] == 'PASS' and grammar['real_token_grammar_compilation']
    schema_shape = next(x for x in grammar['shapes'] if x['question_type'] == 'ordering' and x['item_count'] == 6)
    target.mkdir()
    rows, maps, prompts, tokenids, pairs = [], {}, {}, {}, []
    for q, cohort in selection:
        a.public_question(q)
        source = rt.source_check(q)
        dest = target / 'sources' / (q['root_id'] + '.json')
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(q['source_path'], dest)
        variants = a.variants(q, policy['permutations']['seed'], 6)
        strict = a.evidence_schema(q)
        assert rt.textsha(json.dumps(strict, sort_keys=True)) == schema_shape['strict_schema_sha256']
        for variant in variants:
            pair_id = q['candidate_id'] + '__' + variant['variant_id']
            seed = int.from_bytes(hashlib.sha256(('2026101905|' + pair_id).encode()).digest()[:8], 'big') % (2**31 - 1)
            control = a.evidence_prompt(tok, q, variant, source)
            order = ['control', 'treatment'] if hashlib.sha256(('arm-order|' + pair_id).encode()).digest()[0] % 2 == 0 else ['treatment', 'control']
            pair = {'pair_id': pair_id, 'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'cohort': cohort, 'seed': seed, 'variant_id': variant['variant_id']}
            for arm in order:
                rid = pair_id + '__' + arm
                prompt = control if arm == 'control' else treatment_prompt(control, policy['intervention_text'])
                encoded = tok.encode(prompt, add_special_tokens=False)
                assert len(encoded) + 1024 <= 16384
                assert all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns'])
                row = {
                    'request_id': rid, 'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                    'question_type': 'ordering', 'branch': 'evidence', 'probe_arm': arm, 'cohort': cohort,
                    'pair_id': pair_id, 'variant_id': variant['variant_id'], 'display': {'items': variant['items']},
                    'slot': 0, 'seed': seed, 'prompt_sha256': rt.textsha(prompt), 'prompt_tokens': len(encoded),
                    'source_turns': len(source['turns']), 'source_sha256': q['source_sha256'],
                    'max_output_tokens': 1024, 'json_schema': strict,
                }
                rows.append(row)
                maps[rid] = variant
                prompts[rid] = prompt
                tokenids[rid] = encoded
                pair[arm + '_request_id'] = rid
            pairs.append(pair)
    assert len(rows) == 108 and len(pairs) == 54
    rt.save(target / 'PUBLIC_QUESTIONS.json', {'questions': [q for q, _ in selection]})
    rt.save(target / 'private/DISPLAY_MAPPINGS.json', {'requests': maps, 'public_to_canonical': {q['candidate_id']: a.content_ids(q) for q, _ in selection}})
    rt.save(target / 'PROMPT_TOKEN_IDS.json', tokenids)
    rt.save(target / 'MATCHED_PAIRS.json', {'pairs': pairs})
    for rid, text in prompts.items():
        path = target / 'prompts' / (rid + '.txt')
        path.parent.mkdir(exist_ok=True)
        with path.open('x') as handle:
            handle.write(text)
    m = copy.deepcopy(baseline)
    inputs = [DAY2.parent / 'day1/COMPLETE_MANIFEST.json', DAY2 / 'COMPLETE_MANIFEST.json', DAY2 / 'solver/evidence_v2/manifest.json',
              DAY2 / 'solver/evidence_v2/PUBLIC_QUESTIONS.json', DAY2 / 'EVIDENCE_RUNTIME_V2_GRAMMAR_PREFLIGHT.json',
              HERE / 'PROTOCOL.json', HERE / 'SELECTION.json', HERE / 'POLICY_FREEZE.json']
    m.update({
        'status': 'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT', 'prepared_at_utc': rt.now(),
        'purpose': 'matched_order_construction_instruction_probe', 'question_count': 9, 'root_count': 9,
        'request_count': 108, 'requests': rows, 'protocol_binding': rt.bind(HERE / 'PROTOCOL.json'),
        'parent_baseline_manifest': rt.bind(DAY2 / 'solver/evidence_v2/manifest.json'),
        'model_bindings': baseline['model_bindings'],
        'input_bindings': baseline['input_bindings'] + [rt.bind(p) for p in inputs],
        'code_bindings': baseline['code_bindings'] + [rt.bind(p) for p in sorted(HERE.glob('*.py'))],
        'gold_opened_by_prepare': False,
        'gold_access_scope': 'This prepare/live process does not parse private Gold. Parent and historical review exposure is disclosed.',
        'max_prompt_tokens': max(row['prompt_tokens'] for row in rows),
        'minimum_context_headroom': min(16384 - row['prompt_tokens'] - 1024 for row in rows),
    })
    rt.save(target / 'manifest.json', m)
    rt.save(target / 'PREFLIGHT_FREEZE.json', {
        'status': m['status'], 'created_at_utc': rt.now(),
        'files': [rt.bind(p) for p in sorted(target.rglob('*')) if p.is_file()],
        'questions': 9, 'requests': 108, 'inference_started': False,
    })
    print(json.dumps({k: m[k] for k in ['status', 'question_count', 'root_count', 'request_count', 'max_prompt_tokens', 'minimum_context_headroom']}))


if __name__ == '__main__':
    main()
