"""Validate intent proposals; only local checks can schedule generation."""

import json
import copy

from jsonschema import Draft202012Validator

from . import HERE, SLOTS

SCHEMA = json.loads((HERE / 'intent_contract_v1.json').read_text())


def validate_intents(raw, dialogue, packet):
    raw = copy.deepcopy(raw)
    # Like the first-round canonicalizer, fill only mechanically empty metadata.
    # Never infer IDs, strategies, relation text or evidence for a READY intent.
    if isinstance(raw, dict) and isinstance(raw.get('intents'), list):
        for item in raw['intents']:
            if isinstance(item, dict):
                for name in ('source_feedback_ids', 'coverage_gap_ids', 'required_r03_evidence'):
                    item.setdefault(name, [])
                if item.get('status') == 'READY':
                    item.setdefault('no_safe_reason', None)
    # Preserve independently valid slots, as the first-round miner did. A bad
    # NO_SAFE record must not suppress unrelated, valid READY records.
    if (not isinstance(raw, dict) or set(raw) != {'intents'}
            or not isinstance(raw['intents'], list) or len(raw['intents']) != 3
            or not all(isinstance(x, dict) for x in raw['intents'])):
        return {'valid': False, 'errors': ['intent_top_level_schema_invalid'], 'slots': []}
    intents = raw['intents']
    if [x.get('intent_type') for x in intents] != list(SLOTS):
        return {'valid': False, 'errors': ['slot_order_or_coverage_invalid'], 'slots': []}
    refs = {t['turn_ref'] for t in dialogue['turns']}
    feedback_ids = {x['feedback_id'] for x in packet['eligible_feedback']}
    gap_map = {x['gap_id']: x for x in packet['coverage_gaps']}
    slots = []
    item_validator = Draft202012Validator({'$schema': SCHEMA['$schema'],
        '$defs': SCHEMA['$defs'], '$ref': '#/$defs/intent'})
    for item in intents:
        problems = [f'schema:{list(e.path)}:{e.message}' for e in item_validator.iter_errors(item)]
        if problems:
            slots.append({'intent': item, 'generation_allowed': False, 'errors': problems,
                          'semantic_intent_status': 'NOT_SCHEDULED'})
            continue
        if item['status'] == 'READY':
            flat = [r for v in item['reasoning_chain'].values() for r in v]
            if len(flat) != len(set(flat)):
                problems.append('evidence_groups_not_disjoint')
            if set(flat) - refs:
                problems.append('evidence_ref_unknown')
            if set(item['source_feedback_ids']) - feedback_ids:
                problems.append('feedback_id_unknown')
            if set(item['coverage_gap_ids']) - gap_map.keys():
                problems.append('coverage_gap_id_unknown')
            if any(not r.startswith('R03:') or r not in refs for r in item['required_r03_evidence']):
                problems.append('r03_anchor_invalid')
            if item['intent_type'] == 'COVERAGE_GAP':
                gap_refs = {r for g in item['coverage_gap_ids'] if g in gap_map
                            for r in gap_map[g]['uncited_turn_refs']}
                if not gap_refs.intersection(flat + item['required_r03_evidence']):
                    problems.append('coverage_anchor_not_used')
        slots.append({'intent': item, 'generation_allowed': item['status'] == 'READY' and not problems,
                      'errors': problems, 'semantic_intent_status': 'PENDING_REVIEW'})
    return {'valid': not any(x['errors'] for x in slots), 'errors': [], 'slots': slots}


def reasoning_contract(intent):
    groups = dict(intent['reasoning_chain'])
    existing = {r for refs in groups.values() for r in refs}
    extra = [r for r in intent['required_r03_evidence'] if r not in existing]
    if extra:
        groups['r03_anchor'] = extra
    return {'intent_id': intent['intent_id'], 'intent_type': intent['intent_type'],
            'discriminative_variable': intent['target_relation'],
            'required_evidence_groups': groups}
