"""Build bounded Day4 review packets without running model inference."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parent.parent
O = Path(__file__).resolve().parent
D3 = W / 'day3_mainline_readiness_v1'
BASE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_train_pool_v125')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def binding(path):
    p = Path(path)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def save(path, value):
    p = O / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def refs(value):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('turn_ref', 'turn_id') and isinstance(child, str):
                result.add(child)
            else:
                result.update(refs(child))
    elif isinstance(value, list):
        for child in value:
            result.update(refs(child))
    return result


def main():
    common = read(D3 / 'training/COMMON_OLD_BASE_SET.json')
    ledger = read(D3 / 'training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json')
    old = read(BASE / 'data/COMBINED_QUESTIONS.json')['questions']
    train = lines(BASE / 'data/TRAIN_NEW_ONLY.jsonl')
    lineage = lines(BASE / 'data/QUESTION_LINEAGE.jsonl')
    new = [x for x in ledger['candidate_records'] if x['in88pair']]
    assert len(old) == len(train) == len(lineage) == 363
    assert len(new) == 176
    ids = {x['candidate_id'] for x in old}
    assert len(ids) == 363
    assert ids == {x['candidate_id'] for x in train} == {x['candidate_id'] for x in lineage}
    assert ids == set(common['foundation']['candidate_ids'])
    old_by = {x['candidate_id']: x for x in old}
    line_by = {x['candidate_id']: x for x in lineage}
    roots = sorted({x['root_id'] for x in new})
    packets = []
    for root in roots:
        bs = []
        ns = []
        for q in old:
            if q['root_id'] != root:
                continue
            lin = line_by[q['candidate_id']]
            bs.append({**q, 'dependence_group_id': lin['dependence_group_id'],
                       'source_binding': lin['source_binding'],
                       'necessary_evidence_turns': sorted(refs(q.get('option_evidence', q.get('item_evidence', {}))))})
        for c in new:
            if c['root_id'] != root:
                continue
            record = read(c['record_binding']['path'])
            public = read(c['public_binding']['path'])
            ns.append({**c, 'public': public,
                       'gold': record.get('multi_select', record.get('ordering', {})).get('gold', None),
                       'task_record': record.get('multi_select', record.get('ordering', {})),
                       'evidence_span': record.get('evidence_span', {}),
                       'fact_family': record.get('fact_family', {})})
        pairs = []
        for b in bs:
            for n in ns:
                pairs.append({'base_id': b['candidate_id'], 'new_id': n['candidate_id'],
                              'shared_turn_refs': sorted(set(b['necessary_evidence_turns']) & set(n['necessary_evidence_turns']))})
        packet = {'root_id': root, 'base_candidates': bs, 'new_candidates': ns,
                  'root_local_pairs': pairs,
                  'warning': 'Turn intersections retrieve review candidates; they do not establish semantic identity.'}
        save('packets/' + root + '.json', packet)
        packets.append({'root_id': root, 'base_count': len(bs), 'new_count': len(ns),
                        'pair_count': len(pairs), 'turn_overlap_pairs': sum(bool(x['shared_turn_refs']) for x in pairs),
                        'packet_binding': binding(O / 'packets' / (root + '.json'))})
    save('REVIEW_POLICY.json', {
        'schema_version': 'day4_common_base_admission_policy_v1',
        'scope': 'Fixed363 foundation versus fixed176 paired new candidates; no new question generation, no model calls, no score-based selection.',
        'frozen_inputs': [binding(D3 / 'training/COMMON_OLD_BASE_SET.json'),
                          binding(D3 / 'training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json'),
                          binding(D3 / 'training/TOKENIZATION_ZERO_TRUNCATION.json'),
                          binding(BASE / 'data/TRAIN_NEW_ONLY.jsonl'),
                          binding(BASE / 'data/COMBINED_QUESTIONS.json'),
                          binding(BASE / 'data/QUESTION_LINEAGE.jsonl'),
                          binding(BASE / 'data/DEPENDENCE_GROUPS.json'),
                          binding(W / 'DAILY_EXECUTION_V2_ZH.md')],
        'semantic_classes': {
            'SAME_COMPLETE_TASK_FAMILY': 'The complete assessed proposition/event set and task relation are semantically equivalent, allowing paraphrase, label/permutation, or candidate ID changes. Exclude the foundation candidate.',
            'OVERLAPPING_FACTS_DIFFERENT_RELATION': 'Shared background/event/entity but assessed necessary relation is distinct. Retain a dependence link; this is not semantic independence.',
            'DISTINCT_NECESSARY_FACTS': 'No common necessary assessed fact/relationship after bounded review.',
            'INSUFFICIENT_EVIDENCE': 'Cannot safely decide from frozen sources; HOLD, never count as admitted.'},
        'rules': [
            'Compare the union of A and B; any exclusion applies identically to both common sets.',
            'Root, shared turn, shared generic words and shared phone number alone never establish whole-task duplication.',
            'Inspect all option truth propositions, not only correct labels; ordering compares the complete event set plus relative-order task, not a single shared event/edge.',
            'Use source quotes and speaker/state/scope for proposition identity; swapped actor or reversed truth may still rehearse the same fact.',
            'Ordering event occurrence overlapping an MC fact is not by itself the same ordering relationship.',
            'Dependence_group_id is conservative root x type, not an equivalence family. Keep it intact in split/reporting but do not automatically delete every member because one member is excluded.',
            'Partial overlap of necessary facts is retained with an explicit dependence link. Only whole-task semantic equivalence is excluded; unresolved whole-task equivalence is HOLD.',
            'Root-disjoint candidates need cross-root necessary-fact retrieval, particularly repeated named service properties.',
            'This only determines a common eligible reserve. It does not choose the final mixture, freeze epochs/budgets, reopen legacy113, or grant training GO.'],
        'no_solver_scores_used': True, 'training_start_go': False,
        'new_model_inference': 0, 'training_runs': 0})
    save('PACKET_INVENTORY.json', {
        'base_candidates': 363, 'new_candidates': 176, 'new_pairs': 88,
        'base_roots': len({x['root_id'] for x in old}), 'overlap_roots': len(roots),
        'base_on_overlap_roots': sum(x['base_count'] for x in packets),
        'root_local_pairs': sum(x['pair_count'] for x in packets),
        'turn_overlap_pairs': sum(x['turn_overlap_pairs'] for x in packets),
        'packets': packets, 'training_start_go': False})
    print(json.dumps({'packets': len(packets), 'base_candidates': len(old), 'new_candidates': len(new),
                      'base_on_overlap_roots': sum(x['base_count'] for x in packets),
                      'root_local_pairs': sum(x['pair_count'] for x in packets),
                      'turn_overlap_pairs': sum(x['turn_overlap_pairs'] for x in packets)}))


if __name__ == '__main__':
    main()
