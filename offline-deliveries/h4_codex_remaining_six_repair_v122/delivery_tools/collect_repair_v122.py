"""Read-only progress preserving all six roots and typed historical imports."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
from state_v122 import B, bind, read


def collect():
    scope = read(B / 'REPAIR_SCOPE.json')
    manifest_path = B / 'MANIFEST.json'
    manifest = {r['root_id']: r for r in read(manifest_path)['roots']} if manifest_path.exists() else {}
    roots = []
    for row in scope['roots']:
        unit = Path(row['unit_path']); folder = unit / 'roots' / unit.name
        out = {'root_id': row['root_id'], 'repair_kind': row['repair_kind'], 'status': 'SOURCE_PENDING',
               'source_candidate_saved': (unit / 'source_repair/AUTHOR_RECEIPT.json').exists(),
               'source_reviews_saved': len(list(unit.glob('baseline_reviews/*/REVIEW.json'))),
               'new_scenes': [], 'carried_scene_reaudits': [], 'content_pass': 0,
               'multi_select_gold_counts': [], 'ordering_content_pass': 0}
        candidate = unit / 'source_repair/AUTHOR_PAYLOAD.json'
        if candidate.exists():
            doc = read(candidate)
            out.update(source_status=doc['status'], source_patched_turns=len(doc['patches']))
        source = unit / 'source_repair/FINALIZE_RECEIPT.json'
        if source.exists():
            doc = read(source)
            out.update(status=doc['status'], source_status=doc.get('effective_source_status', doc['source_status']),
                       effective_source_status=doc.get('effective_source_status', doc['source_status']),
                       local_patch_status=doc.get('local_patch_status', doc['source_status']), source_finalize_binding=bind(source))
        prefix = B / 'prefix_audits' / unit.name / 'PREFIX_SOURCE_REVIEW.json'
        if prefix.exists():
            out.update(prefix_pass=read(prefix)['content_reuse_approved'], prefix_review_binding=bind(prefix))
            out['status'] = 'MATCHING_PENDING' if out['prefix_pass'] else 'PREFIX_HOLD'
        reservations = unit / 'SOURCE_RESERVATIONS.json'
        if reservations.exists():
            out.update(new_consumption_count=read(reservations)['new_consumption_count'], status='SCENES_PENDING')
        for draft in sorted(folder.glob('scenes/*/EXPANSION_DRAFT.json')):
            document = read(draft)
            if document['scene_ref'] in row['proposed_retained_scene_refs']: continue
            review = draft.parent / 'SOURCE_REVIEW.json'; finding = draft.parent / 'POST_REVIEW_BLOCKING_FINDING.json'
            decision = read(review).get('decision') if review.exists() else 'PENDING'
            item = {'scene_ref': document['scene_ref'], 'turn_count': len(document['turns']), 'decision': decision,
                    'draft_binding': bind(draft), 'review_binding': bind(review) if review.exists() else None}
            if finding.exists(): item.update(original_decision=decision, decision='POST_REVIEW_HOLD', finding_binding=bind(finding))
            key = 'carried_scene_reaudits' if document['scene_ref'] in row.get('fixed_carried_scene_refs', []) else 'new_scenes'
            out[key].append(item)
        if row['root_id'] in manifest:
            from state_v122 import classify
            state = classify(manifest[row['root_id']])
            out.update({k: v for k, v in state.items() if k != 'scene_state'})
        cross = folder / 'FINAL_CROSSCHECK.json'
        if cross.exists():
            passed = {q['candidate_id'] for q in read(cross)['results'] if q['status'] == 'CONTENT_PASS'}
            public = {q['candidate_id']: q for q in read(folder / 'PUBLIC_QUESTIONS.json')['questions']}
            for gold in read(folder / 'PRIVATE_GOLD.json')['questions']:
                if gold['candidate_id'] not in passed: continue
                if public[gold['candidate_id']]['question_type'] == 'multi_select': out['multi_select_gold_counts'].append(len(gold['gold']))
                else: out['ordering_content_pass'] += 1
            out.update(status='FINALIZED', content_pass=len(passed), unfilled_slots=read(folder / 'COMPOSITION_RECEIPT.json')['unfilled_slots'])
        roots.append(out)
    scenes = [s for r in roots for s in r['new_scenes']]
    carried = [s for r in roots for s in r['carried_scene_reaudits']]
    return {'schema_version': 'v122_fixed_cohort_progress_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'actual_executor': '/root', 'fixed_candidates': scope['fixed_candidate_count'],
            'statuses': dict(Counter(r['status'] for r in roots)),
            'source_candidates_saved': sum(r['source_candidate_saved'] for r in roots),
            'source_reviews_saved': sum(r['source_reviews_saved'] for r in roots),
            'source_provenance_counts': dict(Counter(r.get('source_status', 'PENDING') for r in roots)),
            'new_scenes_generated': len(scenes), 'new_scene_decisions': dict(Counter(s['decision'] for s in scenes)),
            'carried_scene_reaudit_count': len(carried), 'carried_scene_reaudit_decisions': dict(Counter(s['decision'] for s in carried)),
            'new_native_events_reserved': sum(r.get('new_consumption_count', 0) for r in roots),
            'content_pass_questions': sum(r['content_pass'] for r in roots),
            'content_pass_ms_gold_counts': dict(Counter(g for r in roots for g in r['multi_select_gold_counts'])),
            'content_pass_ordering': sum(r['ordering_content_pass'] for r in roots), 'roots': roots, 'training_or_pool_merge': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path); args = p.parse_args(); out = collect()
    if args.output:
        with args.output.open('x') as f: json.dump(out, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'roots'}, ensure_ascii=False))
