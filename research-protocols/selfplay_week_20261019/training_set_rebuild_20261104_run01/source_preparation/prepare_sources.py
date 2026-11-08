#!/usr/bin/env python3
"""Read-only historical source verification and neutral material packaging.

This is an inventory/preparation script, not a question author or admission
authority. It does not inspect question files, Gold, or Solver artifacts.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
WEEK = PROJECT / 'research-protocols/selfplay_week_20261019'
OUT = Path(__file__).parent
PLAN = WEEK / 'training_set_rebuild_plan_20261104_v1/PLAN_CONFIG.json'
INVENTORY = WEEK / 'source_round_inventory_20261104_run01/INVENTORY.json'
ROOT_MANIFEST = PROJECT / 'offline-deliveries/h4_codex_train_pool_v125/data/ROOT_MANIFEST.json'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def binding(p):
    p = Path(p)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def checked_binding(b):
    p = Path(b['path'])
    return {**b, 'exists': p.exists(), 'hash_match': p.exists() and sha(p) == b['sha256']}

def key(t):
    return t['turn_ref'], t['speaker'], t['text']

def round_of(t):
    ref = t['turn_ref']
    return ref.split(':')[0] if ref.startswith('R') else 'parent'

def scene_of(t):
    return t['turn_ref'].rsplit(':T', 1)[0] if t['turn_ref'].startswith('R') else t['turn_ref'].rsplit(':T', 1)[0]

def neutral_warning(w, origin):
    if isinstance(w, str):
        return {'historical_source_review': origin, 'restriction': w}
    result = {'historical_source_review': origin}
    for k in ['id', 'warning_id', 'severity', 'scene_ref', 'turn_refs', 'draft_turn_refs', 'native_turn_ids', 'native_utterance_refs', 'reason', 'rationale', 'finding', 'restriction']:
        if k in w:
            result[k] = w[k]
    return result

def collect(root, row, legacy):
    src = Path(row['source_path'])
    s = read(src)
    turns = s['turns']
    rid = root.split('-')[-1]
    target = OUT / root
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    audit = target / 'audit'
    audit.mkdir()
    shutil.copyfile(src, audit / 'CANONICAL_SOURCE.json')
    assert sha(audit / 'CANONICAL_SOURCE.json') == row['source_sha256'] == sha(src)
    assert len({t['turn_ref'] for t in turns}) == len(turns)
    unit = Path(legacy['source_binding']['path']).parents[2]
    perturn = {t['turn_ref']: [] for t in turns}
    notes = []
    reviews = []
    integrity = []

    def add_review(p, reviewed_source, covered, status=None, scope=None):
        x = read(p)
        record = {'review': binding(p), 'reviewed_source': binding(reviewed_source),
                  'decision': status or x.get('decision', x.get('overall_verdict', x.get('status'))),
                  'reviewer': x.get('reviewer', x.get('reviewer_id', x.get('reviewer_role', x.get('review_role')))),
                  'scope': scope, 'covered_turn_count': len(covered),
                  'same_model_historical_review_not_external_certification': True}
        reviews.append(record)
        i = len(reviews) - 1
        for ref in covered:
            if ref in perturn:
                perturn[ref].append(i)
        for w in x.get('findings', []):
            notes.append(neutral_warning(w, str(p)))
        for w in x.get('warnings', []):
            notes.append(neutral_warning(w, str(p)))
        return x

    # Parent/R01 review, exactly bound to a matching prefix. WARNING is retained,
    # not converted into a whole-root rejection or a new admission decision.
    for p in sorted(unit.glob('baseline_reviews/*/REVIEW.json')):
        x = read(p)
        bp = Path(x.get('baseline_path', unit / 'inputs/BASELINE.json'))
        expected = x.get('baseline_sha256')
        if not bp.exists() or (expected and sha(bp) != expected):
            integrity.append({'kind': 'baseline_review_binding', 'path': str(p), 'verified': False})
            continue
        bt = read(bp).get('turns', [])
        match = [key(t) for t in bt] == [key(t) for t in turns[:len(bt)]]
        integrity.append({'kind': 'baseline_prefix', 'review': str(p), 'verified': match, 'turn_count': len(bt)})
        if match:
            add_review(p, bp, [t['turn_ref'] for t in bt], scope='Parent/R01 exact source prefix; findings apply to cited facts, not whole root.')

    # Six individually reviewed R02 cumulative snapshots.
    for p in sorted((unit / 'roots' / f'root{rid}' / 'scenes').glob('*/SOURCE_REVIEW.json')):
        cp = p.parent / 'CUMULATIVE_SOURCE.json'
        x = read(p)
        if not cp.exists():
            integrity.append({'kind': 'r02_scene_snapshot', 'review': str(p), 'verified': False})
            continue
        c = read(cp)
        ct = c['turns']
        match = sha(cp) == x.get('source_sha256') and [key(t) for t in ct] == [key(t) for t in turns[:len(ct)]]
        scene = x.get('scene_ref', c.get('scene_ref'))
        integrity.append({'kind': 'r02_scene_snapshot_prefix', 'review': str(p), 'scene_ref': scene, 'verified': match})
        if match:
            covered = [t['turn_ref'] for t in ct if scene_of(t) == scene]
            add_review(p, cp, covered, scope=f'Exact R02 scene {scene}; cumulative source also read for context.')

    # Some historical roots have no separate baseline readers; preserve the
    # exact final complete-source review as provenance, with its real scope.
    rp = unit / 'roots' / f'root{rid}' / 'SOURCE_REVIEW.json'
    if rp.exists():
        x = read(rp)
        cp = Path(legacy['source_binding']['path'])
        ct = read(cp)['turns']
        match = sha(cp) == x.get('source_sha256') and [key(t) for t in ct] == [key(t) for t in turns[:len(ct)]]
        integrity.append({'kind': 'complete_r02_snapshot_prefix', 'review': str(rp), 'verified': match})
        if match:
            # Read coverage is not re-labelled as per-sentence approval.
            add_review(rp, cp, [t['turn_ref'] for t in ct], scope='Historical complete cumulative source review; actual evidence must still meet cited-fact restrictions.')

    # R03 assembly proof: immutable parent prefix plus the reviewed draft.
    cp = s.get('closed_loop_provenance') or s.get('continuation_provenance')
    if cp:
        checks = []
        for name in ['parent_source_binding', 'draft_binding', 'review_binding', 'patch_recheck_binding', 'parent', 'candidate', 'review']:
            if name in cp:
                checks.append({'kind': name, **checked_binding(cp[name])})
        integrity.extend(checks)
        parent_b = cp.get('parent_source_binding') or cp.get('parent_binding') or cp.get('parent')
        draft_b = cp.get('draft_binding') or cp.get('candidate_binding') or cp.get('candidate')
        review_b = cp.get('patch_recheck_binding') or cp.get('review_binding') or cp.get('review')
        if parent_b and draft_b and review_b:
            parent = read(parent_b['path'])['turns']
            draft = read(draft_b['path'])
            new = [t for scene in draft.get('scenes', []) for t in scene.get('turns', [])]
            old_match = parent == turns[:len(parent)]
            new_match = [key(t) for t in new] == [key(t) for t in turns[len(parent):]]
            integrity.extend([{'kind': 'r03_parent_objects_unchanged_prefix', 'verified': old_match, 'turn_count': len(parent)},
                              {'kind': 'r03_reviewed_draft_dialogue_match', 'verified': new_match, 'turn_count': len(new)}])
            if old_match and new_match and all(c['hash_match'] for c in checks):
                add_review(Path(review_b['path']), Path(draft_b['path']), [t['turn_ref'] for t in new], scope='Exact reviewed R03 draft; immutable R02 parent is separately bound.')
        for w in cp.get('warnings', []):
            # Skip procedural disclosure unrelated to source semantics.
            if isinstance(w, str) and 'agent-status' in w:
                continue
            notes.append(neutral_warning(w, 'frozen_source_provenance'))
        for n in cp.get('native_events', []):
            if 'binding' in n:
                integrity.append({'kind': 'r03_native', 'scene_ref': n.get('scene_ref'), **checked_binding(n['binding'])})
        for n in cp.get('native_bindings', []):
            integrity.append({'kind': 'r03_native', **checked_binding(n)})

    # Existing explicit material admission for 103866/104814 is reusable only
    # for its exact R02 turns, not as a blanket grant for parent/R01.
    prior = WEEK / 'base_qa_v2_2_1_real_20261102/admission' / root
    admissions = {}
    ap = prior / 'MATERIAL_ADMISSION.json'
    if ap.exists():
        adm = read(ap)
        assert adm['canonical_source_sha256'] == sha(src)
        ab = checked_binding(adm['authorization_binding'])
        rb = checked_binding(adm['independent_review_binding'])
        integrity += [{'kind': 'existing_authorization', **ab}, {'kind': 'existing_independent_review', **rb}]
        aidx = len(reviews)
        reviews.append({'review': binding(ap), 'decision': adm['decision'], 'scope': adm['scope_restriction'],
                        'authorization_binding': ab, 'independent_review_binding': rb,
                        'same_model_historical_review_not_external_certification': True})
        canonical = {t['turn_ref']: t for t in turns}
        for t in adm['turns']:
            ref = t['turn_id']
            if ref not in canonical:
                continue
            tb = checked_binding(t['review_record_binding'])
            integrity.append({'kind': 'existing_turn_review', 'turn_ref': ref, **tb})
            admissions[ref] = {'historical_formal_qa_allowed': t['formal_qa_allowed'], 'historical_review_status': t['review_status'],
                               'restriction_is_scope_not_global_root_veto': True, 'review_index': aidx, 'binding_verified': tb['hash_match']}
        for w in adm.get('review_findings', []):
            notes.append(neutral_warning(w, str(ap)))

    # Author/Reviewer package includes only complete dialogue and neutral source
    # facts. No old question, Gold, blind answer or Solver result is copied.
    neutral_turns = [{k: t[k] for k in ['turn_ref', 'speaker', 'text']} for t in turns]
    packet = {'schema_version': 'neutral_complete_source_v1', 'root_id': root,
              'canonical_source_binding': binding(src), 'turns': neutral_turns}
    write(target / 'author_materials/FULL_SOURCE.json', packet)
    render = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns) + '\n'
    (target / 'author_materials/FULL_SOURCE.txt').write_text(render)
    neutral = {'schema_version': 'neutral_source_information_v1', 'root_id': root,
               'source_expansion_round': row['frozen_expansion_rounds'],
               'canonical_source_binding': binding(src), 'turn_count': len(turns),
               'turn_counts_by_round': dict(collections.Counter(round_of(t) for t in turns)),
               'complete_dialogue_preserved': True,
               'history_note': 'Parent material has historical training exposure. R01/R02/R03 are generated or edited continuations, not independent verification of real-world events.',
               'evidence_policy': 'Read complete source. Match each actual citation to its exact turn, speaker and source version. Preserve attributed reports, unresolved conflicts, proposal/confirmation/completion distinctions, and alternative evidence. Only the affected fact or candidate is paused if its use is unsupported; no whole-root gate.',
               'source_risks': notes,
               'review_summary': [{'review_index': i, 'decision': r['decision'], 'scope': r.get('scope')} for i, r in enumerate(reviews)],
               'no_new_source_authorization_or_semantic_review_granted_by_this_inventory': True,
               'no_old_questions_gold_blind_answers_or_solver_artifacts_in_author_materials': True}
    write(target / 'author_materials/NEUTRAL_SOURCE_INFO.json', neutral)
    provenance = []
    for i, t in enumerate(turns):
        provenance.append({'turn_ref': t['turn_ref'], 'speaker': t['speaker'], 'turn_index_0based': i,
                           'round': round_of(t), 'scene_ref': scene_of(t),
                           'origin_as_stored': t.get('origin'),
                           'text_sha256': hashlib.sha256(t['text'].encode()).hexdigest(),
                           'historical_review_indices': perturn[t['turn_ref']],
                           'existing_material_admission': admissions.get(t['turn_ref']),
                           'review_indices_are_evidence_of_historical_review_not_new_per_fact_approval': True})
    write(target / 'author_materials/TURN_PROVENANCE.json', provenance)
    write(audit / 'HISTORICAL_REVIEW_BINDINGS.json', reviews)
    write(audit / 'INTEGRITY_CHECKS.json', integrity)
    result = {'root_id': root, 'source': binding(src), 'turn_count': len(turns),
              'source_round_counts': neutral['turn_counts_by_round'],
              'historical_review_record_count': len(reviews),
              'turns_with_historical_review_record': sum(bool(perturn[t['turn_ref']]) for t in turns),
              'existing_admission_allowed_turn_count': sum(bool(t.get('historical_formal_qa_allowed')) for t in admissions.values()),
              'binding_checks': len(integrity),
              'failed_integrity_checks': [c for c in integrity if c.get('hash_match') is False or c.get('verified') is False],
              'author_material_directory': str(target / 'author_materials'),
              'status': 'PREPARED_FOR_EVIDENCE_SCOPED_AUTHORING_AND_REVIEW',
              'no_whole_root_admission_decision': True}
    write(target / 'SOURCE_PREPARATION_RESULT.json', result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    plan = read(PLAN)
    if args.batch == 1:
        roots = [x['root_id'] for x in plan['proposed_defaults']['pilot']['roots']]
    else:
        roots = next(x['roots'] for x in plan['proposed_defaults']['scale']['remaining_batches'] if x['batch_index'] == args.batch)
    rows = {r['root_id']: r for r in read(INVENTORY)['rows']}
    legacy = {r['root_id']: r for r in read(ROOT_MANIFEST)}
    results = [collect(r, rows[r], legacy[r]) for r in roots]
    write(OUT / f'BATCH_{args.batch:02d}_SOURCE_PREPARATION.json', {'schema_version': 'source_preparation_batch_v1',
          'batch_index': args.batch, 'root_count': len(results), 'source_roots': results,
          'input_bindings': [binding(PLAN), binding(INVENTORY), binding(ROOT_MANIFEST)],
          'no_question_generation_solver_calls_or_training': True})
    print(json.dumps([{'root_id': x['root_id'], 'turns': x['turn_count'], 'reviews': x['historical_review_record_count'], 'failures':len(x['failed_integrity_checks'])} for x in results], ensure_ascii=False))

if __name__ == '__main__':
    main()
