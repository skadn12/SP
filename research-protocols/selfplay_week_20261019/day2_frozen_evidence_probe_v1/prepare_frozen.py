"""Derive fixed citations from public/previous raw data without parsing Gold."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'day2_order_probe_v1'
DAY2 = HERE.parent / 'day2'
sys.path.insert(0, str(DAY2 / 'solver'))
import adapter as a
import run as rt


def locate(evidence, source):
    by_ref = {t['turn_ref']: (i, t['text']) for i, t in enumerate(source['turns'])}
    positions = []
    for e in evidence:
        turn = by_ref.get(e['turn_ref'])
        offsets = [i for i in range(len(turn[1])) if turn[1].startswith(e['quote'], i)] if turn and e['quote'] else []
        if len(offsets) != 1:
            return None
        positions.append({'label': e['label'], 'source_turn_index': turn[0], 'quote_start_offset': offsets[0]})
    if len({(p['source_turn_index'], p['quote_start_offset']) for p in positions}) != 6:
        return None
    return positions


def strict_json(text):
    def obj(pairs):
        value = {}
        for k, v in pairs:
            if k in value:
                raise ValueError('DUPLICATE_JSON_KEY')
            value[k] = v
        return value
    return json.loads(text, object_pairs_hook=obj, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE')))


def prompt(tok, q, record, source, policy, arm):
    rows = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
    text = f"Dialogue:\n{rows}\n\nQuestion (ordering):\n{q['question']}\n\nItems:\n"
    text += '\n'.join(f'{k}. {v}' for k, v in record['display']['items'].items())
    text += '\n\nFrozen citations:\n'
    text += '\n'.join(f"{e['label']}: turn_ref={json.dumps(e['turn_ref'], ensure_ascii=False)}; quote={json.dumps(e['quote'], ensure_ascii=False)}" for e in record['evidence'])
    text += '\n\n' + policy['prompt']['task']
    if arm == 'explicit_positions':
        text += '\n\n' + policy['prompt']['positions_heading'] + '\n'
        text += '\n'.join(f"{p['label']}: turn_index={p['source_turn_index']}, quote_char_offset={p['quote_start_offset']}" for p in record['positions'])
    return tok.apply_chat_template([{'role': 'system', 'content': policy['prompt']['system']}, {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    target = HERE / 'run'
    assert not target.exists() and not (HERE / 'POLICY_FREEZE.json').exists(), 'NO_OVERWRITE'
    policy = rt.read(HERE / 'PROTOCOL.json')
    seal = rt.read(OLD / 'COMPLETE_MANIFEST.json')
    assert seal['sealed'] and seal['status'] == 'MINIMAL_ORDER_CONTRAST_COMPLETE'
    rt.verify_bindings([{**r, 'path': str(OLD / r['path'])} for r in seal['files']])
    baseline = rt.read(OLD / 'run/manifest.json')
    rt.verify_bindings(baseline['model_bindings'] + baseline['code_bindings'] + baseline['input_bindings'])
    frozen = rt.read(OLD / 'run/RAW_RESPONSE_FREEZE.json')
    rt.verify_bindings(frozen['files'] + [frozen['manifest_binding']])
    maps = rt.read(OLD / 'run/private/DISPLAY_MAPPINGS.json')
    questions = {q['candidate_id']: q for q in rt.read(OLD / 'run/PUBLIC_QUESTIONS.json')['questions']}
    labels = [f'I{i}' for i in range(1, 7)]
    selected, excluded, inventory = [], [], []
    import jsonschema
    controls = sorted((r for r in baseline['requests'] if r['probe_arm'] == 'control'), key=lambda r: r['request_id'])
    assert len(controls) == 54
    for row in controls:
        rid = row['request_id']
        rawpath = OLD / 'run/responses' / (rid + '.json')
        raw = rt.read(rawpath)
        sourcepath = OLD / 'run/sources' / (row['root_id'] + '.json')
        source = rt.read(sourcepath)
        reason, value, positions = None, None, None
        try:
            assert raw['status'] == 'RETURNED' and raw['metadata']['finish_reason'] in ('stop', 'eos')
            value = strict_json(raw['raw_output'])
            jsonschema.validate(value, row['json_schema'])
            assert len(value['evidence']) == 6 and {e['label'] for e in value['evidence']} == set(labels)
            assert len(value['answer']) == 6 and set(value['answer']) == set(labels)
            evidence = sorted(value['evidence'], key=lambda e: labels.index(e['label']))
            positions = locate(evidence, source)
            if positions is None:
                reason = 'NONEXACT_NONUNIQUE_OR_TIED_POSITIONS'
        except Exception as error:
            reason = 'STRICT_INVALID:' + type(error).__name__
        base = {k: row[k] for k in ('candidate_id', 'root_id', 'variant_id')}
        base.update({'frozen_record_id': rid, 'raw_binding': rt.bind(rawpath), 'source_binding': rt.bind(sourcepath)})
        inventory.append({**base, 'selected': reason is None, 'reason': reason})
        if reason is not None:
            excluded.append({**base, 'reason': reason})
            continue
        selected.append({**base, 'display': row['display'], 'evidence': evidence, 'positions': positions,
                         'citation_sorted_display': [p['label'] for p in sorted(positions, key=lambda p: (p['source_turn_index'], p['quote_start_offset']))],
                         'semantic_support': 'REVIEWED_SEPARATELY_DOES_NOT_SELECT_SAMPLE'})
    assert len(selected) == 11 and len(excluded) == 43 and len({r['root_id'] for r in selected}) == 5
    rt.save(HERE / 'BASELINE_INVENTORY.json', {'status': 'ALL_54_CONTROLS_ACCOUNTED', 'selection_rule': policy['selection']['rule'], 'records': inventory, 'selected': 11, 'unknown_or_invalid': 43})
    rt.save(HERE / 'POLICY_FREEZE.json', {'status': 'FROZEN_BEFORE_NEW_INFERENCE', 'created_at_utc': rt.now(), 'new_solver_calls_at_freeze': 0,
                                       'files': [rt.bind(HERE / 'PROTOCOL.json'), rt.bind(HERE / 'BASELINE_INVENTORY.json')], 'prior_sealed_snapshot': rt.bind(OLD / 'COMPLETE_MANIFEST.json')})
    target.mkdir()
    rt.save(target / 'FROZEN_EVIDENCE.json', {'records': selected, 'excluded_records': excluded, 'selection_rule': policy['selection']['rule'], 'semantics_does_not_select_samples': True})
    public = [questions[qid] for qid in sorted({r['candidate_id'] for r in selected})]
    rt.save(target / 'PUBLIC_QUESTIONS.json', {'questions': public})
    for q in public:
        dest = target / 'sources' / (q['root_id'] + '.json')
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(OLD / 'run/sources' / dest.name, dest)
        assert rt.sha(dest) == q['source_sha256']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(rt.TOKENIZER), local_files_only=True)
    rows, mapping, tokenids, pairs = [], {}, {}, []
    for record in selected:
        q = questions[record['candidate_id']]
        source = rt.read(target / 'sources' / (q['root_id'] + '.json'))
        for repeat in policy['sampling']['repeat_ids']:
            pair_id = record['frozen_record_id'] + '__' + repeat
            seed = int.from_bytes(hashlib.sha256(('2026101907|' + pair_id).encode()).digest()[:8], 'big') % (2**31 - 1)
            arms = ['freeze_quotes', 'explicit_positions']
            if hashlib.sha256(('frozen-arm-order|' + pair_id).encode()).digest()[0] % 2:
                arms.reverse()
            pair = {k: record[k] for k in ('frozen_record_id', 'candidate_id', 'root_id', 'variant_id')}
            pair.update({'pair_id': pair_id, 'repeat_id': repeat, 'seed': seed})
            row_pair = pair.copy()
            for arm in arms:
                rid = pair_id + '__' + arm
                text = prompt(tok, q, record, source, policy, arm)
                encoded = tok.encode(text, add_special_tokens=False)
                assert len(encoded) + 32 <= 16384
                row = {**row_pair, 'request_id': rid, 'question_type': 'ordering', 'branch': 'direct', 'probe_arm': arm,
                       'slot': int(repeat[1:]), 'display': record['display'], 'prompt_sha256': rt.textsha(text), 'prompt_tokens': len(encoded),
                       'source_turns': len(source['turns']), 'source_sha256': q['source_sha256'], 'max_output_tokens': 32}
                rows.append(row)
                mapping[rid] = maps['requests'][record['frozen_record_id']]
                tokenids[rid] = encoded
                dest = target / 'prompts' / (rid + '.txt')
                dest.parent.mkdir(exist_ok=True)
                with dest.open('x') as f:
                    f.write(text)
                pair[arm + '_request_id'] = rid
            pairs.append(pair.copy())
    assert len(rows) == 88 and len(pairs) == 44
    # Row dictionaries contain only the pair state at construction, never another arm's ID.
    rt.save(target / 'private/DISPLAY_MAPPINGS.json', {'requests': mapping, 'public_to_canonical': {q['candidate_id']: maps['public_to_canonical'][q['candidate_id']] for q in public}})
    rt.save(target / 'PROMPT_TOKEN_IDS.json', tokenids)
    rt.save(target / 'MATCHED_PAIRS.json', {'pairs': pairs})
    m = copy.deepcopy(baseline)
    inputs = [OLD / 'COMPLETE_MANIFEST.json', OLD / 'run/manifest.json', OLD / 'run/RAW_RESPONSE_FREEZE.json',
              OLD / 'run/PUBLIC_QUESTIONS.json', OLD / 'run/private/DISPLAY_MAPPINGS.json',
              HERE / 'PROTOCOL.json', HERE / 'BASELINE_INVENTORY.json', HERE / 'POLICY_FREEZE.json',
              HERE / 'SEMANTIC_REVIEW.json', HERE / 'SEMANTIC_REVIEW_ZH.md']
    m.update({'status': 'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT', 'prepared_at_utc': rt.now(), 'branch': 'direct',
              'purpose': 'frozen_citation_position_stage_probe', 'question_count': 5, 'root_count': 5, 'frozen_record_count': 11,
              'request_count': 88, 'requests': rows, 'evidence_decoder': None,
              'protocol_binding': rt.bind(HERE / 'PROTOCOL.json'), 'frozen_evidence_binding': rt.bind(target / 'FROZEN_EVIDENCE.json'),
              'baseline_inventory_binding': rt.bind(HERE / 'BASELINE_INVENTORY.json'),
              'input_bindings': baseline['input_bindings'] + [rt.bind(p) for p in inputs],
              'code_bindings': baseline['code_bindings'] + [rt.bind(p) for p in sorted(HERE.glob('*.py'))],
              'gold_opened_by_prepare': False, 'gold_access_scope': 'No private Gold parsing in this prepare/live process; historical parent and reviewers exposure disclosed.',
              'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'minimum_context_headroom': min(16384-r['prompt_tokens']-32 for r in rows)})
    rt.save(target / 'manifest.json', m)
    rt.save(target / 'PREFLIGHT_FREEZE.json', {'status': m['status'], 'created_at_utc': rt.now(), 'files': [rt.bind(p) for p in sorted(target.rglob('*')) if p.is_file()], 'requests': 88, 'inference_started': False})
    print(json.dumps({k: m[k] for k in ['status', 'question_count', 'root_count', 'frozen_record_count', 'request_count', 'max_prompt_tokens', 'minimum_context_headroom']}))


if __name__ == '__main__':
    main()
