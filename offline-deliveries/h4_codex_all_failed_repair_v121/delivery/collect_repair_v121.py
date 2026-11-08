"""Read-only fixed-cohort status export; all success and failure rows retained."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

B = Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text())
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def collect():
    roots = []
    for row in read(B / 'REPAIR_SCOPE.json')['roots']:
        unit = Path(row['unit_path']); source = unit / 'source_repair'
        r = {'root_id': row['root_id'], 'repair_kind': row['repair_kind'], 'status': 'SOURCE_PENDING',
             'source_candidate_saved': (source / 'AUTHOR_RECEIPT.json').exists(),
             'source_reviews_saved': len(list(unit.glob('baseline_reviews/*/REVIEW.json'))),
             'new_scenes': [], 'content_pass': 0, 'multi_select_gold_counts': [], 'ordering_content_pass': 0}
        ap = source / 'AUTHOR_PAYLOAD.json'
        if ap.exists():
            d = read(ap); r['source_status'] = d['status']; r['source_patched_turns'] = len(d['patches'])
        fp = source / 'FINALIZE_RECEIPT.json'
        if fp.exists():
            d = read(fp); r['source_finalize_binding'] = bind(fp); r['status'] = d['status']
        pp = B / 'prefix_audits' / unit.name / 'PREFIX_SOURCE_REVIEW.json'
        if pp.exists():
            r['prefix_review_binding'] = bind(pp); r['prefix_pass'] = read(pp)['content_reuse_approved']
            r['status'] = 'MATCHING_PENDING' if r['prefix_pass'] else 'PREFIX_HOLD'
        sp = unit / 'SOURCE_RESERVATIONS.json'
        if sp.exists(): r['new_consumption_count'] = read(sp)['new_consumption_count']; r['status'] = 'SCENES_PENDING'
        retained = row['proposed_retained_scene_refs']
        for p in sorted(unit.glob('roots/*/scenes/*/EXPANSION_DRAFT.json')):
            d = read(p)
            if d['scene_ref'] in retained: continue
            rp = p.parent / 'SOURCE_REVIEW.json'; review = read(rp) if rp.exists() else {}
            r['new_scenes'].append({'scene_ref': d['scene_ref'], 'turn_count': len(d['turns']),
                'decision': review.get('decision', 'PENDING'), 'draft_binding': bind(p),
                'review_binding': bind(rp) if rp.exists() else None})
        folder = unit / 'roots' / unit.name
        if any(s['decision'] == 'HOLD' for s in r['new_scenes']): r['status'] = 'SCENE_HOLD'
        failure = folder / 'QA_AUTHOR_FAILURE.json'
        if failure.exists(): r['status'] = 'QA_FIRST_ATTEMPT_FAILED'; r['qa_failure_binding'] = bind(failure)
        cross = folder / 'FINAL_CROSSCHECK.json'
        if cross.exists():
            passed = {q['candidate_id'] for q in read(cross)['results'] if q['status'] == 'CONTENT_PASS'}
            public = {q['candidate_id']: q for q in read(folder / 'PUBLIC_QUESTIONS.json')['questions']}
            for q in read(folder / 'PRIVATE_GOLD.json')['questions']:
                if q['candidate_id'] not in passed: continue
                if public[q['candidate_id']]['question_type'] == 'multi_select': r['multi_select_gold_counts'].append(len(q['gold']))
                else: r['ordering_content_pass'] += 1
            r['content_pass'] = len(passed); r['status'] = 'FINALIZED'; r['final_crosscheck_binding'] = bind(cross)
            r['unfilled_slots'] = read(folder / 'COMPOSITION_RECEIPT.json')['unfilled_slots']
        roots.append(r)
    scenes = [s for r in roots for s in r['new_scenes']]
    return {'schema_version': 'v121_fixed_cohort_progress_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'actual_executor': '/root', 'fixed_candidates': 24, 'statuses': dict(Counter(r['status'] for r in roots)),
        'source_candidates_saved': sum(r['source_candidate_saved'] for r in roots),
        'source_reviews_saved': sum(r['source_reviews_saved'] for r in roots),
        'source_provenance_counts': dict(Counter(r.get('source_status', 'PENDING') for r in roots)),
        'new_scenes_generated': len(scenes), 'new_scene_decisions': dict(Counter(s['decision'] for s in scenes)),
        'new_native_events_reserved': sum(r.get('new_consumption_count', 0) for r in roots),
        'content_pass_questions': sum(r['content_pass'] for r in roots),
        'content_pass_ms_gold_counts': dict(Counter(g for r in roots for g in r['multi_select_gold_counts'])),
        'content_pass_ordering': sum(r['ordering_content_pass'] for r in roots), 'roots': roots,
        'training_or_pool_merge': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path); args = p.parse_args(); out = collect()
    if args.output:
        with args.output.open('x') as f: json.dump(out, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'roots'}, ensure_ascii=False))
