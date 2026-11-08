#!/usr/bin/env python3
"""Immutable review packets and bindings only; no question generation/repair."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import pilot_artifacts as pa
import control

E = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def bind(path):
    p = Path(path)
    return {'path': str(p), 'sha256': pa.digest(p)}


def root_packet(root_number):
    """Freeze final quality for this completed root; mask arm/author structures."""
    control.review_status()
    root = 'cc-train-episode-' + str(root_number)
    live = pa.read(E / 'manifests/LIVE_PROGRESS.json')['opportunities']
    rows = [r for r in live if r['root_id'] == root]
    assert len(rows) == 4
    reviews = {r['blind_id']: r for r in pa.read(E / 'manifests/LIVE_REVIEW_STATUS.json')['items']}
    items, dependencies = [], []
    source_path = None
    for r in rows:
        if r['status'] == 'INVALIDATED_ROLE_CONTAMINATION_NO_RETRY':
            continue
        assert r['status'] in ('BLIND_REVIEW_FROZEN', 'NO_CANDIDATE_FROZEN'), r['opportunity_id']
        if r['status'] == 'NO_CANDIDATE_FROZEN':
            continue
        rev = reviews[r['blind_id']]
        assert rev['quality_status'] in ('PASS', 'HOLD', 'FAIL')
        assert isinstance(rev['gold_agreement'], bool), ('UNNORMALIZED_ANSWER', r['blind_id'])
        c = pa.candidate(r['draft_path'])
        source_path = c['source_path']
        first, quality = pa.read(rev['first_answer_path']), pa.read(rev['review_path'])
        packet = pa.read(r['blind_packet'])
        final_quality = rev['quality_status']
        reasons = quality.get('quality_reasons', quality.get('quality', {}).get('reasons', []))
        if not rev['gold_agreement'] and final_quality == 'PASS':
            final_quality = 'HOLD'
            reasons = [reasons, 'Independent blind answer disagrees with frozen author Gold; no repair or automatic forcing of agreement.']
        items.append({'blind_id': r['blind_id'],
                      'question_display': packet['question_display'],
                      'quality_status': final_quality,
                      'quality_reasons': reasons,
                      'gold_agreement': rev['gold_agreement'],
                      'recovered_answer_option_ids': first.get('recovered_answer_option_ids', first.get('independent_answer', {}).get('selected_option_ids')),
                      'recovered_gold_pairs': first.get('recovered_gold_pairs')})
        dependencies.extend(bind(rev[k]) for k in ('draft_path', 'first_answer_path', 'review_path'))
        dependencies.append(bind(r['blind_packet']))
    assert source_path is not None
    qpath = E / 'review/quality_freezes' / (root + '.json')
    quality_seal = {'schema_version': 'root_quality_freeze_v1', 'frozen_at_utc': utc(),
                    'root_id': root, 'items': items, 'files': dependencies,
                    'policy': 'No rewrites; ordinary structure/history overlap never changes quality.'}
    pa.write_new(qpath, quality_seal)
    out = E / 'review/structure_packets' / (root + '.json')
    packet = {'root_id': root, 'source_path': source_path,
              'source_sha256': pa.digest(source_path),
              'history_path': str(E / 'manifests/selector_baseline' / (root + '.json')),
              'quality_freeze_sha256': pa.digest(qpath),
              'current_candidates': items,
              'output_path': str(E / 'review/final_structure' / (root + '.json')),
              'limits': 'Current group identities, author certificates and feedback are masked. Same fixed history for every current candidate; no current item enters history.'}
    pa.write_new(out, packet)
    return {'packet': str(out), 'quality_freeze': str(qpath)}


def target_packet(root_number):
    """Only AFTER this root's independent structure/overlap output is frozen."""
    root = 'cc-train-episode-' + str(root_number)
    structure = E / 'review/final_structure' / (root + '.json')
    assert structure.exists(), 'Structure must be frozen before exposing selected slots'
    rows = pa.read(E / 'manifests/LIVE_PROGRESS.json')['opportunities']
    items = []
    for r in rows:
        n = int(r['opportunity_id'].split('_')[1])
        if r['root_id'] != root or r['workflow'] != 'NEW' or n < 7:
            continue
        if 'blind_id' not in r:
            continue
        d = control.folder(n)
        slots = pa.read(d / 'NEW_SLOTS.json')['slots']
        sel = pa.read(E / 'selector_results' / f'SEL_{n:02d}.json')
        chosen = next(s for s in slots if s['slot_id'] == sel['selected_slot_id'])
        c = pa.candidate(r['draft_path'])
        display = pa.read(r['blind_packet'])['question_display']
        items.append({'blind_id': r['blind_id'], 'selected_slot': chosen,
                      'slots_sha256': pa.digest(d / 'NEW_SLOTS.json'),
                      'candidate_targets_sha256': c.get('targets_sha256'),
                      'question_display': display,
                      'frozen_author_gold_option_ids': c.get('gold_option_ids', c.get('gold')),
                      'frozen_author_gold_pairs': c.get('gold_pairs')})
    packet = pa.read(E / 'review/structure_packets' / (root + '.json'))
    out = E / 'review/fidelity_packets' / (root + '.json')
    value = {'root_id': root, 'source_path': packet['source_path'],
             'source_sha256': packet['source_sha256'], 'structure_already_frozen_sha256': pa.digest(structure),
             'items': items,
             'policy': 'Read only this packet and exact source. Prior independent structure/overlap results are immutable. Independently compare frozen selected target with actual question scope, semantic answer and operations. Hash equality is only mechanical evidence. No generation or repair, feedback, schedule, author certificates, history or Solver.',
             'output_path': str(E / 'review/target_fidelity' / (root + '.json'))}
    pa.write_new(out, value)
    return {'packet': str(out), 'item_count': len(items)}


def build_solver_seal():
    """One final binding after ALL independent reviews; no automatic run."""
    control.review_status()
    live = pa.read(E / 'manifests/LIVE_PROGRESS.json')['opportunities']
    review = {r['blind_id']: r for r in pa.read(E / 'manifests/LIVE_REVIEW_STATUS.json')['items']}
    provenance = E / 'review/CONTINUATION_PROTOCOL_PROVENANCE_AUDIT.json'
    assert provenance.exists(), 'Independent runtime audit required'
    provenance_value = pa.read(provenance)
    assert provenance_value.get('final') is True, 'Runtime audit must be final'
    markers = provenance_value['protocol_markers_by_opportunity']
    inventory = {}
    def add(path):
        b = bind(path)
        inventory[b['path']] = b
        return b
    items = []
    for r in live:
        if 'blind_id' not in r:
            assert r['status'] in ('INVALIDATED_ROLE_CONTAMINATION_NO_RETRY', 'NO_CANDIDATE_FROZEN')
            continue
        rv = review[r['blind_id']]
        root = r['root_id']
        sp = E / 'review/final_structure' / (root + '.json')
        qp = E / 'review/quality_freezes' / (root + '.json')
        structure = next(x for x in pa.read(sp)['candidates'] if x['blind_id'] == r['blind_id'])
        quality = next(x for x in pa.read(qp)['items'] if x['blind_id'] == r['blind_id'])
        assert not structure.get('new_quality_concerns'), 'Concrete new quality issue requires separately frozen adjudication'
        assert structure['structure']['independent'] is True
        assert structure['overlap']['independent'] is True
        c = pa.candidate(r['draft_path'])
        artifact_paths = [rv['first_answer_path'], rv['review_path'], str(sp), str(qp)]
        n = int(r['opportunity_id'].split('_')[1])
        if r['workflow'] == 'NEW' and n >= 7:
            fp = E / 'review/target_fidelity' / (root + '.json')
            fidelity = next(x for x in pa.read(fp)['items'] if x['blind_id'] == r['blind_id'])
            artifact_paths.append(str(fp))
        else:
            fidelity = None
        refs = [add(p) for p in artifact_paths]
        items.append({'candidate_id': c['candidate_id'], 'root_id': root, 'group': r['group'],
                      'raw_draft': add(r['draft_path']),
                      'source': {**add(c['source_path']), 'turn_count': c['source_turn_count']},
                      'quality_status': quality['quality_status'], 'gold_agreement': rv['gold_agreement'],
                      'review_artifacts': refs, 'structure_result': structure['structure'],
                      'overlap_result': structure['overlap'],
                      'target_fidelity_result': fidelity,
                      'protocol_markers': markers[r['opportunity_id']]})
    # Complete immutable experimental dependencies, excluding mutable LIVE views,
    # transient runtime logs, Solver results and any prior invalid review outputs.
    paths = [E / 'manifests/EXECUTION_FREEZE.json',
             E / 'manifests/CONSTRUCTION_COMPLETION_FREEZE.json',
             E / 'manifests/SCHEDULE_RESTRICTED.json',
             E / 'manifests/HISTORICAL_COVERAGE.json',
             E / 'manifests/SELECTOR_BASELINE_FREEZE.json',
             E / 'manifests/SELECTOR_OPERATION_SUMMARY.json',
             E / 'manifests/SELECTOR_OPERATION_SUMMARY_FREEZE.json',
             E / 'manifests/SLOT_ID_DELIVERY_LOG.jsonl', provenance,
             E / 'review/PROTOCOL_PROVENANCE_AUDIT.json',
             E / 'review/PRESERVED_INPUT_INTEGRITY.json']
    paths.extend((E / 'manifests/selector_baseline').glob('*.json'))
    paths.extend((E / 'selector_packets').glob('*.json'))
    paths.extend((E / 'selector_results').glob('*.json'))
    paths.extend((E / 'review/structure_packets').glob('*.json'))
    paths.extend((E / 'review/fidelity_packets').glob('*.json'))
    paths.extend((E / 'manifests').glob('*CONTRACT.txt'))
    paths.extend((E / 'review/fresh_blind_packets').glob('*.json'))
    paths.extend((E / 'review').glob('*DISCLOSURE.json'))
    paths.extend((E / 'review').glob('*CLARIFICATION.json'))
    paths.extend((E / 'review').glob('*RECONCILIATION.json'))
    # Preserve the underlying independent runtime sub-audits and read-only
    # access clarifications as evidence, without freezing live session files.
    paths.extend((E / 'review').glob('CONTINUATION_*RUNTIME_AUDIT.json'))
    paths.extend((E / 'review').glob('*RUNTIME*INDEX*.json'))
    paths.extend((E / 'manifests').glob('RUNTIME_AUDIT_*SUPPLEMENT.txt'))
    paths.extend((E / 'new_slots').glob('**/*.json'))
    paths.extend((E / 'selector').glob('**/*.json'))
    paths.extend(E / 'tools' / name for name in
                 ('pilot_artifacts.py', 'control.py', 'finalize_controller.py'))
    for r in live:
        if 'draft_path' not in r:
            continue
        paths.extend(Path(r['draft_path']).parent.glob('*.json'))
    for p in paths:
        add(p)
    freeze = pa.read(E / 'manifests/EXECUTION_FREEZE.json')
    for key in ('plan_path', 'config_path', 'feedback_card_path'):
        add(freeze[key])
    seal = {'schema_version': 'structure_pilot_input_review_seal_v1',
            'namespace': 'structure_pilot_20261106_v1', 'frozen': True,
            'frozen_at_utc': utc(),
            'reviews_frozen': {k: True for k in ('quality', 'gold', 'structure', 'overlap')},
            'files': sorted(inventory.values(), key=lambda x: x['path']), 'items': items}
    from jsonschema import validate
    validate(seal, pa.read(E / 'solver/INPUT_REVIEW_SEAL.schema.json'))
    out = E / 'solver/INPUT_REVIEW_SEAL.json'
    pa.write_new(out, seal)
    return {'seal': str(out), 'sha256': pa.digest(out), 'items': len(items), 'files': len(inventory)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['root-packet', 'target-packet', 'solver-seal'])
    ap.add_argument('root', nargs='?')
    a = ap.parse_args()
    if a.mode == 'solver-seal':
        result = build_solver_seal()
    else:
        action = root_packet if a.mode == 'root-packet' else target_packet
        result = action(a.root)
    print(json.dumps(result, ensure_ascii=False))
