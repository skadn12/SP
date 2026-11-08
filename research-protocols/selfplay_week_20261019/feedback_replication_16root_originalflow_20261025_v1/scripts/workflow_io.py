"""Artifact IO and mechanical checks for this fixed 16-root continuation.

This file does not generate sources, questions, or semantic decisions.
"""
import hashlib
import json
import pathlib
import statistics
import sys
from datetime import datetime, timezone

BASE = pathlib.Path(__file__).resolve().parent.parent
WEEK = BASE.parent
PREVIOUS = WEEK / 'feedback_replication_20root_resume_20261025_v1'
INITIAL = WEEK / 'feedback_replication_20root_20261025_v1'
SLOTS = {'MC2': ('multi_select', 'MC_QUANTITATIVE_RELATION', 2),
         'MC3': ('multi_select', 'MC_CONSTRAINT_CONJUNCTION', 3),
         'MC4': ('multi_select', 'MC_EVIDENCE_SELECTION', 4),
         'MC5': ('multi_select', 'MC_CAUSAL_SUPPORT', 5),
         'MC6': ('multi_select', 'MC_CONSTRAINT_CONJUNCTION', 6),
         'ORDER1': ('ordering', 'ORDER_EXPLICIT_TIME', 6),
         'ORDER2': ('ordering', 'ORDER_EXPLICIT_PROCEDURE', 6),
         'ORDER3': ('ordering', 'ORDER_DISCOURSE_RECONSTRUCTION', 6)}

def load(path):
    return json.loads(pathlib.Path(path).read_text())

def save(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def bind(path):
    path = pathlib.Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def source_mechanical(candidate, inp, parent):
    errors = []
    if candidate.get('root_id') != inp['root_id']: errors.append('root_id')
    if candidate.get('source_status') != inp['source_status']: errors.append('source_status')
    scenes = candidate.get('scenes', [])
    if [s.get('scene_ref') for s in scenes] != inp['scene_sequence']: errors.append('scene_sequence')
    speakers = {t['speaker'] for t in parent['turns']}
    for scene in scenes:
        turns = scene.get('turns', [])
        if len(turns) != 8: errors.append('turn_count:' + str(scene.get('scene_ref')))
        if turns and statistics.mean(len(t.get('text', '').split()) for t in turns) > 18:
            errors.append('scene_mean_words:' + str(scene.get('scene_ref')))
        for i, turn in enumerate(turns, 1):
            if turn.get('turn_ref') != f"{scene['scene_ref']}:T{i:03d}": errors.append('turn_ref')
            if turn.get('speaker') not in speakers: errors.append('speaker')
            if turn.get('origin') != 'codex_synthetic_extension': errors.append('origin')
            if not turn.get('text', '').strip(): errors.append('empty_text')
    return sorted(set(errors))

def assemble_sources():
    inv = load(BASE / 'REUSE_INVENTORY.json')
    outcomes = []
    for root in inv['roots']:
        rid = root['root_id']
        if root['source_reuse_eligible']:
            assert bind(root['source_binding']['path']) == root['source_binding']
            outcomes.append({'root_id': rid, 'status': 'PASS', 'provenance': 'reused', 'source': root['source_binding']})
            continue
        folder = BASE / 'sources' / rid
        inp = load(INITIAL / 'inputs' / (rid + '_SOURCE_AUTHOR_INPUT.json'))
        parent = load(inp['parent_source_binding']['path'])
        assert bind(inp['parent_source_binding']['path']) == inp['parent_source_binding']
        cp, rp = folder / 'RECOVERY_CANDIDATE.json', folder / 'PATCH_RECHECK.json'
        if not cp.exists() or not rp.exists():
            outcomes.append({'root_id': rid, 'status': 'PENDING', 'provenance': 'recovery'})
            continue
        candidate, review = load(cp), load(rp)
        errors = source_mechanical(candidate, inp, parent)
        if review.get('decision') != 'PASS' or review.get('failure_reasons'): errors.append('independent_review_not_pass')
        if review.get('candidate_binding') != bind(cp): errors.append('review_candidate_binding')
        byref = {t['turn_ref']: t for t in parent['turns'] + [t for s in candidate['scenes'] for t in s['turns']]}
        if not review.get('source_quotes'): errors.append('no_review_quotes')
        for q in review.get('source_quotes', []):
            if q.get('turn_ref') not in byref or not q.get('quote') or q['quote'] not in byref[q['turn_ref']]['text']:
                errors.append('review_quote_mismatch')
        out = {'root_id': rid, 'status': 'FAIL' if errors else 'PASS', 'provenance': 'recovery',
               'candidate': bind(cp), 'review': bind(rp), 'failure_reasons': errors,
               'historical_revisions_used': root.get('source_revisions_used'), 'additional_recovery_revisions': 1}
        if not errors:
            full = dict(parent)
            full['turns'] = parent['turns'] + [t for s in candidate['scenes'] for t in s['turns']]
            # The parent metadata describes R02. The appended turn objects stay
            # exact, while the new full-source counts describe this R03 artifact.
            full['previous_turn_count'] = len(parent['turns'])
            full['new_turn_count'] = 48
            full['cumulative_new_turn_count'] = parent.get('cumulative_new_turn_count', 0) + 48
            full['scene_ref'] = inp['scene_sequence'][-1]
            full['source_round'] = 'R03'
            full['input_source_sha256'] = inp['parent_source_binding']['sha256']
            full['draft_sha256'] = bind(cp)['sha256']
            if 'baseline_turn_count' in parent:
                assert full['baseline_turn_count'] + full['cumulative_new_turn_count'] == len(full['turns'])
            full.update(source_status=inp['source_status'], warnings=candidate.get('warnings', []) + review.get('warnings', []),
                        continuation_provenance={'parent': inp['parent_source_binding'], 'candidate': bind(cp), 'review': bind(rp),
                                                 'native_bindings': [n['binding'] for n in inp['native_events']]})
            assert full['turns'][:len(parent['turns'])] == parent['turns']
            save(folder / 'FULL_SOURCE.json', full)
            out['source'] = bind(folder / 'FULL_SOURCE.json')
        outcomes.append(out)
    result = {'at': now(), 'root_count': 16, 'pass_count': sum(x['status'] == 'PASS' for x in outcomes), 'outcomes': outcomes}
    save(BASE / 'SOURCE_PROGRESS.json', result)
    save(BASE / 'SOURCE_SELECTION.json', {'roots': [{'root_id': x['root_id'], **x['source']} for x in outcomes if x['status'] == 'PASS']})
    if all(x['status'] != 'PENDING' for x in outcomes): save(BASE / 'SOURCE_FREEZE.json', result)
    return result

def candidate_errors(c):
    sys.path.insert(0, str(BASE.parents[2] / 'scripts' / 'current_method'))
    import production_question_gate
    errors = []
    slot = c['slot_id']
    kind, operation, count = SLOTS[slot]
    for key, expected in [('question_type', kind), ('task_operation', operation)]:
        if c.get(key) != expected: errors.append(key)
    if len(c.get('gold', [])) != count or len(set(c.get('gold', []))) != count: errors.append('gold_quota')
    if not 30 <= len(c.get('question', '').split()) <= 90: errors.append('question_length')
    choices = c.get('items' if kind == 'ordering' else 'options', {})
    for label, value in choices.items():
        if not 10 <= len(value.split()) <= 35: errors.append('choice_length:' + label)
    if not c.get('r03_necessity'): errors.append('r03_necessity')
    try:
        errors.extend(production_question_gate.audit_candidate(c, BASE / 'SOURCE_SELECTION.json')['mechanical_errors'])
    except Exception as exc:
        errors.append('production_exception:' + str(exc))
    return errors

if __name__ == '__main__':
    print(json.dumps(assemble_sources(), ensure_ascii=False, indent=2))
