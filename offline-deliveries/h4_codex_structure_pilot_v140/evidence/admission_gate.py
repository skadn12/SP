"""Strict source/citation checks, separate from semantic author/reviewer decisions.

This module never repairs content or recomputes a replacement Gold.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
NATIVE = BASE.parent / 'h4_feedback_training_pool_v15_qa'

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module(name):
    spec = importlib.util.spec_from_file_location('v140_gate_' + name, NATIVE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def citation_errors(citation, turns, prefix):
    if not isinstance(citation, dict):
        return [prefix + ':citation_not_object']
    ref, quote = citation.get('turn_ref'), citation.get('quote')
    turn = turns.get(ref)
    if turn is None:
        return [prefix + ':citation_turn_missing']
    errors = []
    if citation.get('speaker') != turn['speaker']:
        errors.append(prefix + ':speaker_mismatch')
    if not isinstance(quote, str) or not quote or quote not in turn['text']:
        errors.append(prefix + ':quote_not_verbatim')
    return errors

def validate(q):
    errors, facts = [], {}
    cid = q.get('candidate_id')
    qtype = q.get('question_type')
    labels = list('ABCDEF') if qtype == 'multi_select' else [f'I{i}' for i in range(1, 7)]
    if qtype not in ['multi_select', 'ordering']:
        return ['invalid_type'], facts
    field = 'options' if qtype == 'multi_select' else 'items'
    values = q.get(field)
    if not isinstance(values, dict) or list(values) != labels:
        errors.append('six_ordered_labels_required')
    elif (not all(isinstance(v, str) and v.strip() for v in values.values())
          or len({' '.join(v.casefold().split()) for v in values.values()}) != 6):
        errors.append('six_distinct_texts_required')
    if not isinstance(q.get('question'), str) or not q['question'].strip():
        errors.append('question_required')
    p = Path(q.get('source_path', ''))
    if not p.is_absolute() or not p.is_file() or sha(p) != q.get('source_sha256'):
        return errors + ['source_path_or_sha_mismatch'], facts
    source = load(p)
    turns = {t['turn_ref']: t for t in source['turns']}
    if source['root_id'] != q.get('root_id') or len(turns) != len(source['turns']):
        errors.append('source_root_or_turn_schema')
    facts['source_turn_count'] = len(turns)
    ev = q.get('option_evidence' if qtype == 'multi_select' else 'item_evidence')
    if not isinstance(ev, dict) or set(ev) != set(labels):
        return errors + ['six_evidence_entries_required'], facts
    quote_checks = 0
    for label in labels:
        e = ev[label]
        if not isinstance(e, dict):
            errors.append(label + ':evidence_not_object')
            continue
        errors.extend(citation_errors(e, turns, label + ':primary'))
        quote_checks += 1
        cites = e.get('citations')
        if cites is None and qtype == 'ordering':
            # A single ordering anchor already has the complete primary
            # turn/speaker/quote triple. No duplicated list is necessary.
            pass
        elif not isinstance(cites, list) or not cites:
            errors.append(label + ':citations_required')
        else:
            for i, c in enumerate(cites):
                errors.extend(citation_errors(c, turns, f'{label}:citation{i}'))
                quote_checks += 1
        if not isinstance(e.get('reason', e.get('rationale')), str):
            errors.append(label + ':reason_required')
    facts['exact_quote_speaker_checks'] = quote_checks
    if qtype == 'multi_select':
        native_errors, diagnostics = module('multiselect_gate').validate_normalized(q, source)
        errors.extend(native_errors)
        if diagnostics:
            errors.extend('native_citation:' + d['label'] + ':' + d['code'] for d in diagnostics)
        facts['gold_count'] = len(q.get('gold', []))
    else:
        if q.get('chronology_kind') != 'conversation_order':
            errors.append('conversation_order_required')
        if 'APPEAR IN THE SUPPLIED DIALOGUE' not in q.get('question', '').upper():
            errors.append('appearance_order_public_definition_required')
        anchors = q.get('order_anchor_spans')
        aerrors, resolved, computed = module('ordering_gate').resolve_ordering_anchors(anchors, source)
        errors.extend(aerrors)
        if computed != q.get('gold'):
            errors.append('gold_differs_from_original_anchor_order')
        if isinstance(anchors, dict):
            for label, anchor in anchors.items():
                errors.extend(citation_errors(anchor, turns, label + ':anchor'))
            scenes = Counter(turns[a['turn_ref']].get('scene_ref') or ':'.join(a['turn_ref'].split(':')[:-1])
                             for a in anchors.values() if a.get('turn_ref') in turns)
            facts['items_per_scene'] = dict(scenes)
            facts['scene_count_vector_desc'] = sorted(scenes.values(), reverse=True)
            facts['scene_order_sufficient'] = all(n == 1 for n in scenes.values())
            facts['scene_only_compatible_orders'] = math.prod(math.factorial(n) for n in scenes.values())
        for text in (values or {}).values():
            if re.search(r'\bI[1-6]\b|\b(?:R\d+:)?S\d+:T\d+\b', text, re.I):
                errors.append('item_contains_source_position_hint')
    if q.get('pilot_arm') == 'new_structure':
        plan = load(BASE / 'PLAN.json')
        rootplan = next(r for r in plan['roots'] if r['root_id'] == q['root_id'])
        if q['source_sha256'] != rootplan['source_sha256']:
            errors.append('source_differs_from_fixed_plan')
        if qtype == 'ordering' and facts.get('scene_order_sufficient'):
            errors.append('new_ordering_scene_only_sufficient')
    return sorted(set(errors)), facts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    raw = load(args.input)
    questions = raw['questions'] if isinstance(raw, dict) else raw
    results = []
    ids = [q['candidate_id'] for q in questions]
    for q in questions:
        errors, facts = validate(q)
        results.append({'candidate_id': q['candidate_id'], 'status': 'FAIL' if errors else 'PASS',
                        'errors': errors, 'facts': facts})
    report = {'status': 'PASS' if len(ids) == len(set(ids)) and all(r['status'] == 'PASS' for r in results) else 'FAIL',
              'semantic_correctness_certified': False, 'question_count': len(questions),
              'duplicate_ids': len(ids) - len(set(ids)), 'results': results}
    with Path(args.output).open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'count': len(questions),
                      'failures': [r for r in results if r['errors']]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
