#!/usr/bin/env python3
"""Sealed structure-pilot adapter. Import/self-check are CPU-only; no automatic retries."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[2]
PREVIOUS = WEEK / 'training_set_rebuild_20261105_three_root_G_retry_eval_v1/model_baseline/adapter.py'
WORDING = WEEK / 'training_set_rebuild_20261106_wording_solver_eval_v1/adapter.py'
IDENTITY = HERE / 'MODEL_IDENTITY_AUDIT.json'
DEFAULT_SEAL = HERE / 'INPUT_REVIEW_SEAL.json'
OUTPUT = HERE / 'evaluation'
NAMESPACE = 'structure_pilot_20261106_v1'
DISPLAY_SEED, SAMPLE_SEED, ENGINE_SEED = 2026110603, 2026110604, 2026110303
MAX_REQUESTS, MAX_TOKENS, MODEL_LEN, BATCH = 1152, 6144, 16384, 12
GROUPS = ('OLD_F', 'OLD_G', 'NEW_F', 'NEW_G')
PUBLIC_FIELDS = ('query_scope', 'query_pairs', 'answer_semantics', 'relation_definition',
                 'pair_domain', 'instructions', 'visible_instructions')
_history = None


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bind(path):
    path = Path(path).resolve()
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            hasher.update(block)
    return {'path': str(path), 'sha256': hasher.hexdigest()}


def write_new(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE), 'WRITE_OUTSIDE_SOLVER'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value))
        stream.flush()
        import os
        os.fsync(stream.fileno())


def require_binding(record):
    assert isinstance(record, dict) and set(('path', 'sha256')) <= set(record), 'BINDING_REQUIRED'
    actual = bind(record['path'])
    assert actual['sha256'] == record['sha256'], ('FROZEN_INPUT_CHANGED', actual['path'])
    return actual


def load_history():
    """Load code only; no historical data, model, tokenizer, or run entry point."""
    global _history
    if _history is not None:
        return _history
    spec = importlib.util.spec_from_file_location('structure_pilot_legacy_adapter', PREVIOUS)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    h = old.h
    # The historical transport's arm and Gold-cardinality gates belong to its
    # experiment. Generalize in memory; never alter historical source files.
    code = inspect.getsource(old.old_check)
    replacements = [
        ('def check_input_item(', 'def structure_check_input_item('),
        ('len(options) >= 3', 'len(options) >= 2'),
        ('2 <= len(gold) <= 6 and len(gold) < len(labels)', '0 < len(gold) <= len(labels)'),
    ]
    for before, after in replacements:
        assert code.count(before) == 1, ('HISTORICAL_CHECK_CHANGED', before)
        code = code.replace(before, after)
    exec(compile(code, str(PREVIOUS) + ':structure_check', 'exec'), h.__dict__)
    scorer = h.baseline.official
    code = inspect.getsource(scorer.score_response)
    assert code.count('len(gold) < 2') == 1, 'HISTORICAL_SCORER_CHANGED'
    exec(compile(code.replace('len(gold) < 2', 'len(gold) < 1'),
                 str(h.baseline.SCORER_PATH) + ':singleton_gate', 'exec'), scorer.__dict__)
    h.DISPLAY_SEED = DISPLAY_SEED
    _history = old
    return old


def code_dependencies():
    old = load_history()
    h = old.h
    paths = {Path(__file__), HERE / 'ENGINEERING_SPEC.md', HERE / 'INPUT_REVIEW_SEAL.schema.json',
            HERE / 'ADAPTER_README.md', HERE / 'test_adapter.py', PREVIOUS, WORDING,
            old.HISTORICAL, h.REFERENCE, h.reference.REASON, h.reference.REPAIR,
            h.reference.BASE / 'run_eval.py', h.baseline.ENGINE_PATH,
            h.baseline.engine.LEGACY_FILE, h.baseline.ORDER_PROMPT_PATH,
            h.baseline.SCORER_PATH, h.PERMUTATION_CODE, IDENTITY}
    # Include transitive historical modules imported under private loader names,
    # which need not be registered in sys.modules.
    pending, seen = [old], set()
    project = WEEK.parent.parent
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(project) and Path(path).suffix=='.py':
            paths.add(Path(path))
            pending.extend(value for value in vars(module).values() if isinstance(value, types.ModuleType))
    return paths


def closure(pairs):
    result = set(map(tuple, pairs))
    while True:
        more = result | {(a, d) for a, b in result for c, d in result if b == c}
        if more == result:
            return result
        result = more


def pair_list(value, vertices, name):
    assert isinstance(value, list), name + '_LIST_REQUIRED'
    assert all(isinstance(p, list) and len(p) == 2 and p[0] != p[1]
               and all(isinstance(v, str) and v in vertices for v in p) for p in value), name + '_INVALID_PAIR'
    assert len(value) == len(set(map(tuple, value))), name + '_DUPLICATE_PAIR'
    return copy.deepcopy(value)


def raw_candidate(document, candidate_id):
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict) and 'candidates' in document:
        rows = document['candidates']
    else:
        rows = [document]
    matches = [r for r in rows if isinstance(r, dict) and r.get('candidate_id') == candidate_id]
    assert len(matches) == 1, 'EXACTLY_ONE_RAW_CANDIDATE_REQUIRED'
    candidate = copy.deepcopy(matches[0])
    if isinstance(document, dict) and 'candidates' in document:
        for key in ('root_id', 'source_path', 'source_sha256', 'source_hash', 'source_turn_count'):
            if key in document:
                candidate.setdefault(key, document[key])
    return candidate


def check_visible_id_references(question, option_texts, public_fields, stable_ids):
    """Hold explicit original-ID prose references instead of exposing a map.

    Structured query_pairs are handled separately. Single-character IDs need
    reference syntax; plain English articles/pronouns are not ID references.
    """
    stable_ids = set(stable_ids)
    atom = '(?:' + '|'.join(re.escape(oid) for oid in sorted(stable_ids, key=len, reverse=True)) + ')'
    bounded = lambda value: r'(?<![A-Za-z0-9_])' + value + r'(?![A-Za-z0-9_])'
    explicit_context = re.compile(r'\b(?:option|item|label|id|option_id|item_id|choice|node|event|task|step)s?\s*[:#]?\s*' + bounded(atom), re.I)
    pair_reference = re.compile(bounded(atom) + r'\s*(?:<|>|->|→|precedes|follows|(?:is |comes )?before|(?:is |comes )?after|depends on)\s*' + bounded(atom))
    list_reference = re.compile(bounded(atom) + r'(?:\s*(?:,|and|or|versus|vs\.?)\s*' + bounded(atom) + r')+')

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for row in value:
                yield from strings(row)
        elif isinstance(value, dict):
            for key, row in value.items():
                yield str(key)
                yield from strings(row)

    prose = [question, *option_texts, *strings({key: value for key, value in public_fields.items() if key!='query_pairs'})]
    for text in prose:
        for oid in stable_ids:
            if len(oid)>1 and re.search(bounded(re.escape(oid)), text):
                raise AssertionError('EVAL_HOLD_STABLE_ID_REFERENCE_IN_VISIBLE_PROSE')
            if len(oid)==1 and (text.strip()==oid or re.search(r'[\[\("\'`]' + re.escape(oid) + r'[\]\)"\'`]', text)):
                raise AssertionError('EVAL_HOLD_STABLE_ID_REFERENCE_IN_VISIBLE_PROSE')
        if explicit_context.search(text) or pair_reference.search(text) or list_reference.search(text):
            raise AssertionError('EVAL_HOLD_STABLE_ID_REFERENCE_IN_VISIBLE_PROSE')


def normalize(candidate, record):
    """Normalize only in memory. The immutable raw draft is never rewritten."""
    c = copy.deepcopy(candidate)
    qtype = {'full_order': 'ordering'}.get(c.get('question_type'), c.get('question_type'))
    assert qtype in ('multi_select', 'ordering', 'partial_order'), 'UNSUPPORTED_ANSWER_FORM'
    assert c.get('root_id', c.get('root')) == record['root_id'], 'CANDIDATE_ROOT_MISMATCH'
    assert isinstance(c.get('question'), str) and c['question'].strip(), 'QUESTION_REQUIRED'
    options = c.get('options')
    assert isinstance(options, list) and len(options) >= 2, 'AT_LEAST_TWO_OPTIONS_REQUIRED'
    k = len(options)
    assert qtype != 'multi_select' or k <= 26, 'MULTI_LABEL_ALPHABET_EXCEEDED'
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:k]) if qtype == 'multi_select' else [f'I{i+1}' for i in range(k)]
    old_map = c.get('label_to_option_id', {})
    stable, texts = [], []
    for option in options:
        assert isinstance(option, dict), 'OPTION_OBJECT_REQUIRED'
        original = option.get('id', option.get('label'))
        oid = option.get('option_id', old_map.get(original, original))
        assert isinstance(oid, str) and oid, 'STABLE_OPTION_ID_REQUIRED'
        assert isinstance(option.get('text'), str) and option['text'].strip(), 'OPTION_TEXT_REQUIRED'
        stable.append(oid)
        texts.append(option['text'])
    assert len(stable) == len(set(stable)), 'DUPLICATE_OPTION_ID'
    public_fields = {key: copy.deepcopy(c[key]) for key in PUBLIC_FIELDS if key in c}
    check_visible_id_references(c['question'], texts, public_fields, stable)
    mapping = dict(zip(labels, stable))
    inverse = {v: k for k, v in mapping.items()}
    gold = c.get('gold_option_ids', c.get('gold'))
    assert isinstance(gold, list), 'GOLD_LIST_REQUIRED'
    if qtype == 'partial_order':
        assert gold == [], 'PARTIAL_RAW_GOLD_OPTION_IDS_MUST_BE_EMPTY'
        pairs = pair_list(c.get('gold_pairs'), set(stable), 'GOLD')
        domain = pair_list(c.get('query_pairs'), set(stable), 'QUERY_DOMAIN')
        assert domain and c.get('query_scope') is not None, 'EXPLICIT_PARTIAL_QUERY_SCOPE_REQUIRED'
        assert c.get('answer_semantics') == 'all_and_only_entailed_pairs_in_query_scope', 'PARTIAL_ANSWER_SEMANTICS_REQUIRED'
        transitive = closure(pairs)
        assert all(a != b for a, b in transitive), 'CYCLIC_PARTIAL_GOLD'
        assert set(map(tuple, pairs)) <= set(map(tuple, domain)), 'GOLD_OUTSIDE_QUERY_DOMAIN'
        assert transitive & set(map(tuple, domain)) == set(map(tuple, pairs)), 'PARTIAL_GOLD_MISSING_QUERIED_TRANSITIVE_EDGE'
        gold_labels = [[inverse[a], inverse[b]] for a, b in pairs]
    else:
        assert all(isinstance(v, str) and v in inverse for v in gold), 'GOLD_STABLE_ID_MISMATCH'
        assert len(gold) == len(set(gold)), 'DUPLICATE_GOLD_ID'
        if qtype == 'multi_select':
            assert gold, 'EVAL_HOLD_UNDEFINED_EMPTY_GOLD_J'
        else:
            assert set(gold) == set(stable), 'COMPLETE_ORDER_REQUIRED'
        pairs, domain = None, None
        gold_labels = [inverse[v] for v in gold]
    source = record['source']
    assert Path(c.get('source_path', source['path'])).resolve() == Path(source['path']).resolve(), 'SOURCE_PATH_MISMATCH'
    assert c.get('source_sha256', c.get('source_hash', source['sha256'])) == source['sha256'], 'SOURCE_HASH_ALIAS_MISMATCH'
    assert c.get('source_turn_count', source['turn_count']) == source['turn_count'], 'SOURCE_COUNT_ALIAS_MISMATCH'
    item = {'candidate_id': record['candidate_id'], 'root_id': record['root_id'], 'arm': record['group'],
            'group': record['group'], 'number': 1, 'question_type': qtype, 'question': c['question'],
            'options': [{'id': label, 'text': text} for label, text in zip(labels, texts)],
            'label_to_option_id': mapping, 'gold_option_ids': gold, 'gold_labels': gold_labels,
            'source_path': str(Path(source['path']).resolve()), 'source_sha256': source['sha256'],
            'source_turn_count': source['turn_count'], 'turn_count': source['turn_count'],
            'permutations': min(6, math.factorial(k)), 'repeats_per_permutation': 3,
            'public_fields': public_fields,
            'protocol_markers': copy.deepcopy(record['protocol_markers']),
            'structure_result': copy.deepcopy(record['structure_result']),
            'overlap_result': copy.deepcopy(record['overlap_result']),
            'clean_contrast_eligible': not record['protocol_markers']}
    if qtype == 'partial_order':
        item.update(gold_pairs=pairs, query_pairs=domain, query_scope=copy.deepcopy(c['query_scope']))
    return item


def legacy_check(item, source):
    old = load_history()
    proxy = copy.deepcopy(item)
    proxy['arm'] = 'F'
    if item['question_type'] == 'partial_order':
        proxy.update(question_type='ordering', gold_labels=list(item['label_to_option_id']),
                     gold_option_ids=list(item['label_to_option_id'].values()))
    checked, turns = old.h.structure_check_input_item(proxy, {item['root_id']: source})
    assert len(turns) == item['source_turn_count'], 'SOURCE_TURN_COUNT_MISMATCH'
    # Restore the exact four-group identity and actual type after the old gate.
    result = {**checked, **item}
    assert result['arm'] in GROUPS
    return result, turns


def check_inputs(seal_path=DEFAULT_SEAL, verify_model=True):
    seal_path = Path(seal_path).resolve()
    seal = read(seal_path)
    from jsonschema import validate as json_validate
    json_validate(seal, read(HERE / 'INPUT_REVIEW_SEAL.schema.json'))
    assert seal.get('schema_version') == 'structure_pilot_input_review_seal_v1', 'SEAL_VERSION_REQUIRED'
    assert seal.get('namespace') == NAMESPACE and seal.get('frozen') is True, 'FROZEN_SEAL_REQUIRED'
    assert all(seal.get('reviews_frozen', {}).get(k) is True for k in ('quality', 'gold', 'structure', 'overlap')), 'ALL_REVIEWS_MUST_BE_FROZEN'
    bindings = [require_binding(row) for row in seal['files']]
    by_path = {r['path']: r['sha256'] for r in bindings}
    assert len(by_path) == len(bindings), 'DUPLICATE_SEAL_FILE'
    def bound(record):
        actual = require_binding(record)
        assert by_path.get(actual['path']) == actual['sha256'], 'ITEM_DEPENDENCY_NOT_IN_SEAL_FILES'
        return actual
    items, holds, seen_ids, seen_slots = [], [], set(), set()
    for record in seal['items']:
        qid = record['candidate_id']
        assert isinstance(qid, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', qid), 'UNSAFE_CANDIDATE_ID'
        assert qid not in seen_ids, 'DUPLICATE_CANDIDATE_ID'
        seen_ids.add(qid)
        assert record['group'] in GROUPS, 'INVALID_GROUP'
        slot = (record['root_id'], record['group'])
        assert slot not in seen_slots, 'DUPLICATE_ROOT_GROUP'
        seen_slots.add(slot)
        assert record['quality_status'] in ('PASS', 'HOLD', 'FAIL'), 'QUALITY_STATUS_REQUIRED'
        assert type(record['gold_agreement']) is bool, 'BOOLEAN_GOLD_AGREEMENT_REQUIRED'
        assert isinstance(record['protocol_markers'], list), 'PROTOCOL_MARKERS_REQUIRED'
        for key in ('structure_result', 'overlap_result'):
            assert isinstance(record[key], dict) and record[key].get('independent') is True, 'INDEPENDENT_RESULT_REQUIRED'
        assert record['review_artifacts'], 'REVIEW_ARTIFACTS_REQUIRED'
        for review in record['review_artifacts']:
            bound(review)
        draft = bound(record['raw_draft'])
        source = bound(record['source'])
        assert type(record['source']['turn_count']) is int and record['source']['turn_count'] > 0
        if record['quality_status'] != 'PASS' or not record['gold_agreement']:
            holds.append({'candidate_id': qid, 'status': 'QUALITY_OR_GOLD_NOT_ELIGIBLE',
                          'quality_status': record['quality_status'], 'gold_agreement': record['gold_agreement']})
            continue
        try:
            candidate = raw_candidate(read(draft['path']), qid)
            item = normalize(candidate, record)
            source_record = {**source, 'turn_count': record['source']['turn_count']}
            item, turns = legacy_check(item, source_record)
            items.append((item, turns))
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            holds.append({'candidate_id': qid, 'status': 'EVAL_HOLD', 'quality_status': 'PASS', 'reason': str(exc)})
    identity = read(IDENTITY)
    old = load_history()
    assert identity['status'] == 'PASS' and identity['model'] == str(old.MCLEAN) and identity['tokenizer'] == str(old.TOKENIZER), 'MODEL_IDENTITY_AUDIT_REQUIRED'
    if verify_model:
        for row in identity['files']:
            actual = require_binding(row)
            assert actual['sha256'] == row['expected_sha256'], 'MODEL_EXPECTED_HASH_MISMATCH'
            bindings.append(actual)
    bindings.extend(bind(p) for p in code_dependencies())
    bindings.append(bind(seal_path))
    bindings = list({r['path']: r for r in bindings}.values())
    assert sum(2 * item['permutations'] * 3 for item, _ in items) <= MAX_REQUESTS, 'CALL_BUDGET_EXCEEDED'
    return items, holds, bindings, identity


def variants(item):
    proxy = copy.deepcopy(item)
    if item['question_type'] == 'partial_order':
        proxy['gold_option_ids'] = proxy['gold_pairs']  # Legacy directed pairs only in memory.
    return load_history().variants(proxy)


def schema_for(labels, refs, qtype):
    schema = load_history().schema_for(labels, refs, qtype)
    atom = '(?:' + '|'.join(re.escape(v) for v in labels) + ')'
    if qtype == 'ordering':
        body = atom + '( ' + atom + '){' + str(len(labels)-1) + '}'
    elif qtype == 'multi_select':
        body = '(?:' + atom + '(?: ' + atom + ')*)?'
    else:
        pair = atom + '<' + atom
        body = '(?:' + pair + '(?:;' + pair + ')*)?'
    schema['properties']['final_answer']['pattern'] = '^<box>' + body + '</box>$'
    return schema


def remap_public(value, inverse):
    if isinstance(value, str):
        return inverse.get(value, value)
    if isinstance(value, list):
        return [remap_public(v, inverse) for v in value]
    if isinstance(value, dict):
        return {inverse.get(k, k): remap_public(v, inverse) for k, v in value.items()}
    return value


def prompt_for(tokenizer, item, variant, turns):
    """Uniform brief-reason JSON interface, generated from actual k and type."""
    labels = list(variant['display_to_option_id'])
    inverse = {v: k for k, v in variant['display_to_option_id'].items()}
    check_visible_id_references(item['question'], [option['text'] for option in item['options']],
                                item.get('public_fields', {}), inverse)
    qtype, k = item['question_type'], len(labels)
    common = ('Return only JSON with item_evidence first, answer_reason second, and final_answer last. '
              'For every displayed label provide judgment (supported, contradicted, not_supported, or ambiguous), '
              'zero to three turn_refs, and a concise verifiable reason of at most 320 characters. '
              'Give a brief answer_reason of at most 800 characters. Format examples illustrate syntax only. ')
    if qtype == 'ordering':
        operation = (f'For ordering, return all {k} displayed labels exactly once in the requested order. '
                     'Use spaces inside one box. A format example listing all labels is <box>' + ' '.join(labels) + '</box>.')
    elif qtype == 'multi_select':
        operation = ('For multi_select, return all and only the selected labels separated by spaces inside one box. '
                     'Any number may be selected; use <box></box> for the empty set. A syntax example is <box>' + labels[0] + '</box>.')
    else:
        operation = ('For partial_order, X<Y denotes the directed relation asked in the question, with X as its first endpoint. '
                     'Return all and only entailed directed pairs in the displayed query domain, including entailed transitive pairs '
                     'that belong to that domain. Leave unconstrained pairs unasserted; do not force a total order. '
                     'Separate pairs with semicolons; use <box></box> if none is supported. A syntax example is <box>' + labels[0] + '<' + labels[1] + '</box>.')
    if turns:
        system = 'Use only the supplied complete dialogue. Follow the precise question and relation scope. Give brief verifiable reasons before the final boxed answer.'
        source = 'Dialogue:\n' + '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns) + '\n\n'
        common += 'Use only turn_refs printed in the dialogue. '
    else:
        system = 'Predict from the question and displayed items only. No source dialogue is supplied. Never fabricate source references.'
        source = ''
        common += 'No dialogue is supplied; every turn_refs array must be empty []. Judgments are best predictions from visible wording. '
    fields = copy.deepcopy(item.get('public_fields', {}))
    # query_pairs have a semantic stable-ID mapping; human scope prose is kept verbatim.
    if 'query_pairs' in fields:
        fields['query_pairs'] = remap_public(fields['query_pairs'], inverse)
    public = '\n\nOriginal visible scope/instructions (query pair endpoints use displayed labels):\n' + json.dumps(fields, ensure_ascii=False) if fields else ''
    user = (source + f"Question ({qtype}):\n{item['question']}\n\nDisplayed items:\n"
            + '\n'.join(f"{o['id']}. {o['text']}" for o in variant['options']) + public
            + '\n\n' + common + operation)
    return tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


def invalid(qtype, reason):
    if qtype == 'partial_order':
        return load_history().partial_invalid(reason)
    result = load_history().old_parse({'status': 'INVALID'}, {'question_type': qtype, 'display_to_option_id': {}})
    result['parse_reason'] = reason
    return result


def parse_answer(body, raw, request):
    qtype = request['question_type']
    labels = list(request['display_to_option_id'])
    atom = '(?:' + '|'.join(re.escape(v) for v in labels) + ')'
    if qtype == 'partial_order':
        result = load_history().parse_and_score({**raw, 'raw_prediction': body}, request)
        assert result['valid'], result['parse_reason']
        assert set(map(tuple, result['predicted_option_ids'])) <= set(map(tuple, request['query_pairs'])), 'ANSWER_OUTSIDE_QUERY_DOMAIN'
        return result
    assert re.fullmatch('(?:' + atom + '(?: ' + atom + ')*)?', body), 'FINAL_LABEL_SYNTAX'
    values = body.split(' ') if body else []
    assert len(values) == len(set(values)), 'DUPLICATE_LABEL'
    if not values and qtype == 'multi_select':
        return {'valid': True, 'parse_reason': None, 'parsed': [], 'exact_correct': False,
                'J': 0.0, 'J_fraction': '0', 'predicted_option_ids': [], 'tp': 0, 'fp': 0,
                'fn': request['g'], 'recall': 0.0, 'item_completeness': 0.0}
    result = load_history().parse_and_score({**raw, 'raw_prediction': (',' if qtype == 'multi_select' else '>').join(values)}, request)
    assert result['valid'], result['parse_reason']
    return result


def validate(raw, request, schema=None):
    """Final-answer scoring is independent of auxiliary JSON/schema compliance."""
    result = {'final_answer_valid': False, 'answer_errors': [], 'auxiliary_protocol_valid': False,
              'auxiliary_protocol_errors': [], 'citation_entailment_verified': False}
    obj = None
    try:
        assert raw.get('status') == 'RETURNED' and raw.get('finish_reason') in ('stop', 'eos'), 'INCOMPLETE_RESPONSE'
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                assert key not in obj, 'DUPLICATE_JSON_KEY'
                obj[key] = value
            return obj
        obj = json.loads(raw['raw_prediction'], object_pairs_hook=unique)
        assert isinstance(obj, dict), 'JSON_OBJECT_REQUIRED'
        final = obj.get('final_answer')
        assert isinstance(final, str), 'FINAL_ANSWER_STRING_REQUIRED'
        match = re.fullmatch(r'<box>(.*)</box>', final)
        assert match, 'FINAL_BOX_SYNTAX'
        answer = parse_answer(match.group(1), raw, request)
        result.update(final_answer_valid=True, final_answer=final, answer_only_score=answer)
    except Exception as exc:
        result['answer_errors'].append(str(exc))
    try:
        assert isinstance(obj, dict), 'NO_PARSED_JSON_OBJECT'
        if schema is None:
            schema = read(OUTPUT / request['schema_path'])
        from jsonschema import validate as json_validate
        json_validate(obj, schema)
        assert list(obj) == ['item_evidence', 'answer_reason', 'final_answer'], 'REASONS_MUST_PRECEDE_ANSWER'
        assert obj['answer_reason'].strip(), 'EMPTY_ANSWER_REASON'
        for label, row in obj['item_evidence'].items():
            assert row['reason'].strip(), 'EMPTY_ITEM_REASON:' + label
            assert len(row['turn_refs']) == len(set(row['turn_refs'])), 'DUPLICATE_REFERENCE:' + label
            if request['context_mode'] == 'no_context':
                assert row['turn_refs'] == [], 'NO_CONTEXT_REFERENCE_FORBIDDEN:' + label
            elif row['judgment'] != 'not_supported':
                assert row['turn_refs'], 'CITATION_REQUIRED:' + label
        result['auxiliary_protocol_valid'] = True
    except Exception as exc:
        result['auxiliary_protocol_errors'].append(str(exc))
    result['protocol_valid'] = result['auxiliary_protocol_valid']
    result['protocol_errors'] = result['auxiliary_protocol_errors']
    return result


def model_config(identity):
    return {'namespace': NAMESPACE, 'model': identity['model'], 'tokenizer': identity['tokenizer'],
            'engine_seed': ENGINE_SEED, 'display_seed': DISPLAY_SEED, 'sample_seed_base': SAMPLE_SEED,
            'sampling_seed_rule': 'sample_seed_base + sorted_candidate_id_index*1000 + variant_index*3 + repeat_index; paired Full/No-context',
            'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1.,
            'dtype': 'bfloat16', 'batch_size': BATCH, 'max_model_len': MODEL_LEN, 'max_new_tokens': MAX_TOKENS,
            'gpu_memory_utilization': .8, 'enable_prefix_caching': False, 'native_thinking': False,
            'structured_outputs': True, 'sampling_by_type': {q: {'displays': 'min(6,k!) unique, Gold-independent', 'repeats': 3} for q in ('multi_select', 'ordering', 'partial_order')},
            'maximum_calls': MAX_REQUESTS, 'automatic_retry': False, 'local_only': True,
            'training': False, 'formal_training_export': False, 'primary_scoring': 'final-answer-only',
            'partial_order_scoring': 'historical directed-pair precision/recall/F1; no predicted closure repair',
            'missing_policy': 'infrastructure/missing excluded from model score denominator and reported explicitly',
            'invalid_answer_policy': 'zero in fixed executed-request denominator',
            'J_boundary_policy': 'g=0 EVAL_HOLD; g=n uses historical scorer TP/g convention',
            'packages': {n: load_history().h.package_version(n) for n in ('transformers', 'vllm', 'jsonschema', 'torch')}}


def preflight_answer_check(request, schema):
    """Check actual display/Gold/scorer round trips without any model call."""
    qtype = request['question_type']
    labels = request['gold_labels']
    body = ';'.join('<'.join(pair) for pair in labels) if qtype=='partial_order' else ' '.join(labels)
    evidence = {label: {'judgment': 'not_supported', 'turn_refs': [], 'reason': 'CPU mapping fixture.'}
                for label in request['display_to_option_id']}
    raw = {'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': json.dumps({
        'item_evidence': evidence, 'answer_reason': 'CPU display and scorer binding check.', 'final_answer': '<box>'+body+'</box>'})}
    checked = validate(raw, request, schema)
    assert checked['final_answer_valid'] and checked['answer_only_score']['exact_correct'], 'ACTUAL_DISPLAY_GOLD_SCORER_ROUNDTRIP_FAILED'
    assert checked['auxiliary_protocol_valid'], 'ACTUAL_SCHEMA_FIXTURE_FAILED'
    predicted = checked['answer_only_score']['predicted_option_ids']
    if qtype=='partial_order':
        assert set(map(tuple,predicted)) == set(map(tuple,request['gold_pairs'])), 'PAIR_STABLE_ID_ROUNDTRIP_FAILED'
    elif qtype=='multi_select':
        assert set(predicted) == set(request['gold_option_ids']), 'SET_STABLE_ID_ROUNDTRIP_FAILED'
    else:
        assert predicted == request['gold_option_ids'], 'ORDER_STABLE_ID_ROUNDTRIP_FAILED'


def reserve(path, calls, candidate_ids):
    assert calls <= MAX_REQUESTS, 'CALL_BUDGET_EXCEEDED'
    write_new(path, {'namespace': NAMESPACE, 'reserved_at_utc': now(), 'potential_calls': calls,
                     'candidate_ids': candidate_ids, 'budget_limit': MAX_REQUESTS,
                     'automatic_retry': False, 'policy': 'one prepare and one run attempt; no resume or replacement'})


def prepare(seal_path=DEFAULT_SEAL):
    assert not OUTPUT.exists() and not (HERE / 'ATTEMPT_LEDGER.json').exists(), 'NO_PREPARE_RETRY_OR_OVERWRITE'
    checked, holds, bindings, identity = check_inputs(seal_path)
    potential = sum(2 * item['permutations'] * 3 for item, _ in checked)
    reserve(HERE / 'ATTEMPT_LEDGER.json', potential, [i['candidate_id'] for i, _ in checked])
    OUTPUT.mkdir(exist_ok=False)
    old = load_history()
    old.h.reference.offline()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(identity['tokenizer'], trust_remote_code=True, local_files_only=True)
    requests, assets, kept, permutations = [], {}, [], {}
    indices = {qid: i for i, qid in enumerate(sorted(i['candidate_id'] for i, _ in checked))}
    for item, full_turns in checked:
        display = variants(item)
        pending, item_assets, too_long = [], {}, []
        permutations[item['candidate_id']] = display
        for mode in ('full_context', 'no_context'):
            turns = full_turns if mode == 'full_context' else []
            refs = [t['turn_ref'] for t in turns]
            for vi, variant in enumerate(display):
                prompt = prompt_for(tokenizer, item, variant, turns)
                tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                schema = schema_for(list(variant['display_to_option_id']), refs, item['question_type'])
                if tokens + MAX_TOKENS > MODEL_LEN:
                    too_long.append({'context_mode': mode, 'variant_id': variant['variant_id'], 'tokens': tokens})
                for repeat in range(3):
                    rid = f"{item['candidate_id']}__{mode}__{variant['variant_id']}__s{repeat:02d}"
                    pp, sp = f'prompts/{rid}.txt', f'schemas/{rid}.json'
                    item_assets[pp], item_assets[sp] = prompt.encode(), encoded(schema)
                    k = len(item['options'])
                    request = {**variant, 'request_id': rid, 'candidate_id': item['candidate_id'],
                               'root_id': item['root_id'], 'arm': item['arm'], 'group': item['group'], 'number': 1,
                               'question_type': item['question_type'], 'context_mode': mode, 'repeat_index': repeat,
                               'sampling_seed': SAMPLE_SEED + indices[item['candidate_id']] * 1000 + vi * 3 + repeat,
                               'gold_option_ids': item['gold_option_ids'], 'n': k if item['question_type'] == 'multi_select' else k*(k-1),
                               'g': len(item['gold_option_ids']) if item['question_type'] == 'multi_select' else len(item['gold_pairs']) if item['question_type'] == 'partial_order' else k*(k-1)//2,
                               'prompt_path': pp, 'schema_path': sp, 'prompt_sha256': digest(prompt.encode()),
                               'schema_sha256': digest(encoded(schema)), 'prompt_token_count': tokens,
                               'turn_count': len(turns), 'source_sha256_bytes': item['source_sha256'],
                               'protocol_markers': item['protocol_markers'], 'clean_contrast_eligible': item['clean_contrast_eligible'],
                               'structure_result': item['structure_result'], 'overlap_result': item['overlap_result'],
                               'formal_training_export': False}
                    if item['question_type'] == 'partial_order':
                        request.update(gold_pairs=item['gold_pairs'], query_pairs=item['query_pairs'], query_scope=item['query_scope'])
                    preflight_answer_check(request, schema)
                    pending.append(request)
        if too_long:
            holds.append({'candidate_id': item['candidate_id'], 'status': 'EVAL_HOLD', 'quality_status': 'PASS',
                          'reason': 'SOURCE_OR_PROMPT_EXCEEDS_FIXED_WINDOW_NO_TRUNCATION', 'lengths': too_long})
        else:
            requests.extend(pending)
            assets.update(item_assets)
            kept.append(item)
    assert len(requests) <= potential <= MAX_REQUESTS
    assert len({r['request_id'] for r in requests}) == len(requests), 'DUPLICATE_REQUEST_ID'
    paired = {}
    for request in requests:
        key = (request['candidate_id'],request['variant_id'],request['repeat_index'])
        paired.setdefault(key,[]).append(request)
    for pair in paired.values():
        assert len(pair)==2 and {r['context_mode'] for r in pair}=={'full_context','no_context'}, 'CONTEXT_PAIR_REQUIRED'
        assert pair[0]['sampling_seed']==pair[1]['sampling_seed'], 'CONTEXT_SEED_MISMATCH'
        assert pair[0]['display_to_option_id']==pair[1]['display_to_option_id'], 'CONTEXT_DISPLAY_MISMATCH'
    for relative, data in assets.items():
        write_new(OUTPUT / relative, data)
    write_new(OUTPUT / 'ITEMS_PRIVATE.json', kept)
    write_new(OUTPUT / 'PERMUTATIONS_PRIVATE.json', permutations)
    write_new(OUTPUT / 'REQUESTS.json', requests)
    write_new(OUTPUT / 'MODEL_CONFIG.json', {**model_config(identity), 'sampling_candidate_indices': indices})
    preflight = {'status': 'PASS' if requests else 'EVAL_HOLD', 'prepared_at_utc': now(),
                 'quality_gold_eligible_count': len(checked), 'evaluated_candidate_count': len(kept),
                 'holds': holds, 'request_count': len(requests), 'potential_calls_reserved': potential,
                 'maximum_calls': MAX_REQUESTS, 'requests_sha256': bind(OUTPUT / 'REQUESTS.json')['sha256'],
                 'max_prompt_tokens': max((r['prompt_token_count'] for r in requests), default=None),
                 'max_new_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'source_truncated': False,
                 'gold_independent_displays': True, 'paired_context_seeds': True, 'new_model_calls': 0}
    write_new(OUTPUT / 'PREFLIGHT.json', preflight)
    bindings.append(bind(HERE / 'ATTEMPT_LEDGER.json'))
    write_new(OUTPUT / 'INPUT_BINDINGS.json', {'files': bindings})
    frozen = [bind(OUTPUT / name) for name in ('ITEMS_PRIVATE.json', 'PERMUTATIONS_PRIVATE.json', 'REQUESTS.json', 'MODEL_CONFIG.json', 'PREFLIGHT.json', 'INPUT_BINDINGS.json')]
    write_new(OUTPUT / 'PLAN_FREEZE.json', {'frozen_at_utc': now(), 'namespace': NAMESPACE, 'files': frozen,
                                         'planned_calls': len(requests), 'automatic_retry': False})
    return preflight


def verify_prepared():
    for record in read(OUTPUT / 'PLAN_FREEZE.json')['files']:
        require_binding(record)
    for record in read(OUTPUT / 'INPUT_BINDINGS.json')['files']:
        require_binding(record)
    for request in read(OUTPUT / 'REQUESTS.json'):
        for kind in ('prompt', 'schema'):
            assert bind(OUTPUT / request[kind + '_path'])['sha256'] == request[kind + '_sha256'], 'PREPARED_ASSET_CHANGED'


def collect_raws():
    requests = read(OUTPUT / 'REQUESTS.json')
    complete = OUTPUT / 'RAW_ANSWERS.json'
    if complete.exists():
        raws = read(complete)
        assert len(raws) == len(requests), 'RAW_COUNT_MISMATCH'
        assert [r['request_id'] for r in raws] == [r['request_id'] for r in requests], 'RAW_ID_MISMATCH'
        return raws
    raws = []
    for request in requests:
        path = OUTPUT / 'raw' / (request['request_id'] + '.json')
        raw = read(path) if path.exists() else {'request_id': request['request_id'], 'status': 'MISSING',
                    'reason': 'no returned raw preserved; infrastructure/interruption, not a model error', 'automatic_retry': False}
        assert raw['request_id']==request['request_id'], 'PRESERVED_RAW_ID_MISMATCH'
        raws.append(raw)
    write_new(complete, raws)
    return raws


def summarize(rows):
    returned = [r for r in rows if r['execution_status'] == 'RETURNED']
    result = {'planned_calls': len(rows), 'executed_returned_calls': len(returned),
              'missing_calls': len(rows)-len(returned), 'valid_calls': sum(r['valid'] for r in returned),
              'invalid_executed_calls': sum(not r['valid'] for r in returned),
              'exact_correct_calls': sum(r['exact_correct'] for r in returned),
              'exact_accuracy_executed': sum(r['exact_correct'] for r in returned)/len(returned) if returned else None,
              'planned_accuracy': sum(r['exact_correct'] for r in returned)/len(rows) if rows and len(returned)==len(rows) else None}
    kinds = sorted({r['question_type'] for r in rows})
    if len(kinds) > 1:
        result['metrics_by_type'] = {kind: summarize([r for r in rows if r['question_type']==kind]) for kind in kinds}
    elif kinds:
        fields = ('precision', 'recall', 'partial_score') if kinds[0]=='partial_order' else ('J',)
        result['metric'] = 'directed_pair_precision_recall_f1' if kinds[0]=='partial_order' else 'J'
        for field in fields:
            name = 'directed_pair_f1' if field=='partial_score' else field
            result['mean_' + name + '_executed'] = sum(r.get(field, 0.0) for r in returned)/len(returned) if returned else None
    return result


def score():
    assert (HERE / 'RUN_ATTEMPT.json').exists(), 'RUN_ATTEMPT_REQUIRED_NO_SCORING_UNEXECUTED_PREPARATION'
    assert not (OUTPUT / 'SCORES.json').exists(), 'NO_SCORE_OVERWRITE'
    verify_prepared()
    requests, raws = read(OUTPUT / 'REQUESTS.json'), collect_raws()
    rows = []
    for request, raw in zip(requests, raws):
        assert request['request_id'] == raw['request_id'], 'RAW_REQUEST_MISMATCH'
        checked = validate(raw, request)
        answer = checked.get('answer_only_score', invalid(request['question_type'], 'INVALID_FINAL_ANSWER'))
        rows.append({**{key: request[key] for key in ('request_id', 'candidate_id', 'root_id', 'group', 'question_type', 'context_mode', 'variant_id', 'repeat_index', 'sampling_seed', 'protocol_markers', 'clean_contrast_eligible', 'structure_result', 'overlap_result')},
                     **answer, **checked, 'execution_status': raw['status'],
                     'raw_prediction': raw.get('raw_prediction'), 'finish_reason': raw.get('finish_reason')})
    result = {'scored_at_utc': now(), 'namespace': NAMESPACE, 'primary_scoring': 'final_answer_only',
              'missing_policy': 'reported separately; excluded from executed model denominators',
              'auxiliary_protocol_is_gate': False, 'requests': rows,
              'by_context': {mode: summarize([r for r in rows if r['context_mode']==mode]) for mode in ('full_context', 'no_context')},
              'by_group': {group: {mode: summarize([r for r in rows if r['group']==group and r['context_mode']==mode]) for mode in ('full_context', 'no_context')} for group in GROUPS},
              'by_item': {qid: {mode: summarize([r for r in rows if r['candidate_id']==qid and r['context_mode']==mode]) for mode in ('full_context', 'no_context')} for qid in sorted({r['candidate_id'] for r in rows})},
              'comparison_limit': 'finite descriptive results; protocol deviations are flagged and not mixed into clean causal contrasts; J and directed-pair F1 are never averaged together'}
    write_new(OUTPUT / 'SCORES.json', result)
    return {k: v for k, v in result.items() if k in ('namespace', 'by_context', 'primary_scoring')}


def configure_legacy_run():
    """Compile the historical local loop with checked persistence-only patches."""
    old = load_history()
    h = old.h
    h.HERE, h.reference.HERE = OUTPUT, OUTPUT
    identity = read(IDENTITY)
    h.MODEL, h.TOKENIZER = Path(identity['model']), Path(identity['tokenizer'])
    h.MAX_REQUESTS, h.MAX_TOKENS, h.MODEL_LEN, h.BATCH, h.ENGINE_SEED = MAX_REQUESTS, MAX_TOKENS, MODEL_LEN, BATCH, ENGINE_SEED
    h.bind = bind
    source = inspect.getsource(h.run)
    assert source.count('tokenizer=str(MODEL)') == 1, 'HISTORICAL_RUN_CHANGED'
    assert source.count('score()') == 2, 'HISTORICAL_RUN_CHANGED'
    source = source.replace('tokenizer=str(MODEL)', 'tokenizer=str(TOKENIZER)').replace('score()', 'structure_score_deferred()')
    # Retain returned members of a short/invalid backend batch before recording
    # an interruption. This changes transport persistence, never a model call.
    needle = '        assert len(outputs) == len(batch)\n'
    assert source.count(needle)==1, 'HISTORICAL_BATCH_LOOP_CHANGED'
    source = source.replace(needle, '        incomplete_batch = len(outputs) != len(batch)\n')
    needle = "        print(json.dumps({'status': 'INFERENCE_PROGRESS'"
    assert source.count(needle)==1, 'HISTORICAL_PROGRESS_CHANGED'
    source = source.replace(needle, "        assert not incomplete_batch, 'INCOMPLETE_BACKEND_BATCH'\n"+needle)
    needle = '            assert len(output.outputs) == 1\n'
    assert source.count(needle)==1, 'HISTORICAL_OUTPUT_LOOP_CHANGED'
    replacement = '''            if len(output.outputs) != 1:
                raw = {'request_id': row['request_id'], 'status': 'BACKEND_INVALID',
                    'backend_outputs': [{'text': completion.text, 'finish_reason': completion.finish_reason,
                                         'token_ids': list(completion.token_ids)} for completion in output.outputs],
                    'reason': 'backend did not return exactly one completion', 'automatic_retry': False}
                write('raw/' + row['request_id'] + '.json', raw)
                raws.append(raw)
                continue
'''
    source = source.replace(needle,replacement)
    h.structure_score_deferred = lambda: None
    exec(compile(source, str(__file__) + ':unchanged_local_loop', 'exec'), h.__dict__)
    return h


def run():
    assert not (HERE / 'RUN_ATTEMPT.json').exists(), 'NO_RUN_RETRY_OR_RESUME'
    verify_prepared()
    preflight = read(OUTPUT / 'PREFLIGHT.json')
    assert preflight['status']=='PASS' and preflight['request_count'] > 0, 'EVAL_HOLD_NO_RUN'
    write_new(HERE / 'RUN_ATTEMPT.json', {'started_at_utc': now(), 'namespace': NAMESPACE,
                                       'automatic_retry': False, 'plan_freeze': bind(OUTPUT / 'PLAN_FREEZE.json')})
    error = None
    try:
        h = configure_legacy_run()
        h.run()
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        write_new(OUTPUT / 'INTERRUPTED.json', {'recorded_at_utc': now(), 'error': error,
                                              'automatic_retry': False, 'missing_are_not_model_errors': True})
    summary = score()
    return {'status': 'INTERRUPTED' if error else 'COMPLETED', 'error': error, 'scores': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('self-check', 'check-inputs', 'prepare', 'run', 'score'))
    parser.add_argument('--seal', type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    if args.mode == 'self-check':
        import unittest
        tests = unittest.defaultTestLoader.discover(str(HERE), pattern='test_adapter.py')
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.mode == 'check-inputs':
        items, holds, _, _ = check_inputs(args.seal)
        result = {'status': 'CPU_INPUT_CHECK_COMPLETE', 'eligible': [i['candidate_id'] for i, _ in items], 'holds': holds, 'new_model_calls': 0}
    else:
        result = {'prepare': lambda: prepare(args.seal), 'run': run, 'score': score}[args.mode]()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
