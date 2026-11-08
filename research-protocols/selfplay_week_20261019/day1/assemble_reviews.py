"""Commit public-only reconstructions, then compare immutable historical Gold."""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    public = read(BASE / 'reuse/public/QUESTIONS.json')['questions']
    qmap = {q['candidate_id']: q for q in public}
    assert len(qmap) == 113
    files = [BASE / f'reviews/reviewer_{name}/REVIEW.json' for name in 'abc']
    reviews = []
    references = []
    for p in files:
        d = read(p)
        assert d['private_gold_seen'] is False, p
        references.append({'path': str(p.relative_to(BASE)), 'sha256': sha(p), 'bytes': p.stat().st_size})
        for q in d['questions']:
            q['_reviewer'] = d['reviewer']
            q['_review_path'] = str(p.relative_to(BASE))
            reviews.append(q)
    assert len(reviews) == len({q['candidate_id'] for q in reviews}) == 113
    assert set(qmap) == {q['candidate_id'] for q in reviews}
    freeze_path = BASE / 'reviews/RECONSTRUCTION_FREEZE.json'
    save(freeze_path, {
        'status': 'ALL_SOURCE_FIRST_RECONSTRUCTIONS_COMMITTED_BEFORE_THIS_SCRIPT_PRIVATE_COMPARISON',
        'created_at_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'question_count': 113, 'reviews': references,
        'public_questions_sha256': sha(BASE / 'reuse/public/QUESTIONS.json'),
        'limitations': ['Internal tasks share model and filesystem; private-answer non-access is a declared workflow, not OS-enforced access isolation.',
                       'Parent inventory builder has historical Gold exposure; source reconstruction reviewers were separately prompted without that history.'],
    })
    # No private inventory or historical Gold is opened by this script before the
    # complete reconstruction set is byte-bound above.
    inv = read(BASE / 'reuse/INVENTORY.json')['questions']
    originals = {q['candidate_id']: q for q in inv}
    assert set(originals) == set(qmap)
    out = []
    check_count = 0
    for r in sorted(reviews, key=lambda x: x['candidate_id']):
        q = qmap[r['candidate_id']]
        original = originals[r['candidate_id']]['original_question_record']
        source_path = Path(q['source_binding']['path'])
        src = read(source_path)
        turns = src['turns']
        refs = {row['turn_ref']: (i, row) for i, row in enumerate(turns)}
        errors = []
        checks = []
        def check(ok, code):
            nonlocal check_count
            check_count += 1
            checks.append({'check': code, 'pass': bool(ok)})
            if not ok:
                errors.append(code)
        check(r['root_id'] == q['root_id'], 'ROOT_ID_MATCH')
        check(r['question_type'] == q['question_type'], 'QUESTION_TYPE_MATCH')
        check(r['source_sha256'] == q['source_binding']['sha256'] == sha(source_path), 'SOURCE_HASH_MATCH')
        check(r['source_turn_count'] == len(turns), 'FULL_SOURCE_TURN_COUNT_MATCH')
        check(r.get('source_read_complete') is True, 'FULL_SOURCE_READ_DECLARED')
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        labels = set(q[field])
        items = r['item_reviews']
        itemmap = {x['label']: x for x in items}
        check(len(items) == len(itemmap) and set(itemmap) == labels, 'COMPLETE_UNIQUE_ITEM_REVIEW')
        check(r['recommendation'] in ('PASS', 'HOLD', 'FAIL'), 'RECOMMENDATION_VALID')
        positions = {}
        for item in items:
            label = item['label']
            check(item['relation'] in ('supports', 'contradicts', 'not_supported', 'ambiguous'), 'ITEM_RELATION_VALID:' + label)
            citations = item.get('citations', [])
            check(bool(citations) or item['relation'] in ('not_supported','ambiguous'), 'SUPPORTED_OR_CONTRADICTED_ITEM_HAS_EVIDENCE:' + label)
            offsets = {}
            for index, cit in enumerate(citations):
                ref = cit.get('turn_ref')
                exists = ref in refs
                check(exists, f'CITATION_TURN_EXISTS:{label}:{index}')
                if not exists:
                    continue
                turn_index, row = refs[ref]
                check(cit.get('speaker') == row['speaker'], f'CITATION_SPEAKER_MATCH:{label}:{index}')
                quote = cit.get('quote')
                exact = isinstance(quote, str) and bool(quote) and quote in row['text']
                check(exact, f'CITATION_EXACT_SUBSTRING:{label}:{index}')
                if exact:
                    offsets.setdefault(ref, []).append(row['text'].index(quote))
            matching = item.get('matching_turn_refs', [])
            check(all(ref in refs for ref in matching), 'MATCHING_TURNS_EXIST:' + label)
            if q['question_type'] == 'ordering' and item['relation'] == 'supports':
                check(bool(matching), 'ORDER_SUPPORTED_ITEM_HAS_MATCHING_POSITIONS:' + label)
                check(all(ref in offsets for ref in matching), 'ORDER_MATCHES_HAVE_EXACT_QUOTES:' + label)
                positions[label] = [(refs[ref][0], offset) for ref in matching if ref in refs
                                    for offset in offsets.get(ref, [])]
        prediction = r.get('reconstructed_gold')
        complete_prediction = isinstance(prediction, list) and len(set(prediction)) == len(prediction) and set(prediction) <= labels
        if q['question_type'] == 'ordering':
            complete_prediction = complete_prediction and set(prediction) == labels
        check(complete_prediction or r['recommendation'] != 'PASS', 'PASS_REQUIRES_COMPLETE_RECONSTRUCTION')
        internal_recommendation = r['recommendation']
        conservative_holds = []
        if internal_recommendation == 'PASS':
            if r.get('alternative_readings'):
                conservative_holds.append('ALTERNATIVE_READINGS_REQUIRE_ADJUDICATION')
            if any(x['relation'] == 'ambiguous' for x in items):
                conservative_holds.append('AMBIGUOUS_ITEM')
            if q['question_type'] == 'ordering':
                if any(x['relation'] != 'supports' for x in items):
                    conservative_holds.append('ORDER_ITEM_NOT_FULLY_SUPPORTED')
                if complete_prediction:
                    for i, first in enumerate(prediction):
                        for second in prediction[i+1:]:
                            if not positions.get(first) or not positions.get(second) or max(positions[first]) >= min(positions[second]):
                                conservative_holds.append('ORDER_ALL_REPORTED_MATCHES_NOT_STRICT:' + first + ':' + second)
            elif complete_prediction:
                inferred = {x['label'] for x in items if x['relation'] == 'supports'}
                if set(prediction) != inferred:
                    conservative_holds.append('MC_SUPPORTED_SET_AND_RECONSTRUCTION_DIFFER')
        gold = original['gold']
        gold_agrees = (set(prediction) == set(gold) if q['question_type'] == 'multi_select' else prediction == gold) if complete_prediction else None
        if errors:
            status = 'HOLD_REVIEW_RECORD_INVALID'
        elif conservative_holds:
            status = 'HOLD_SEMANTIC_ADJUDICATION'
        elif internal_recommendation == 'FAIL':
            status = 'FAIL_SOURCE_REVIEW'
        elif internal_recommendation == 'HOLD':
            status = 'HOLD_SOURCE_REVIEW'
        elif not gold_agrees:
            status = 'FAIL_ORIGINAL_GOLD_MISMATCH'
        else:
            status = 'PASS_INTERNAL_CONTENT'
        out.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],
                    'source_binding':q['source_binding'],'review_path':r['_review_path'],'reviewer':r['_reviewer'],
                    'reviewer_recommendation':internal_recommendation,'reconstructed_gold':prediction,
                    'original_gold_unchanged':gold,'gold_agreement':gold_agrees,'status':status,
                    'review_record_errors':errors,'conservative_holds':conservative_holds,
                    'review_issues':r.get('issues',[]),'alternative_readings':r.get('alternative_readings',[]),
                    'training_admitted':False,
                    'next_gate':'Training normalization, dynamic legacy-format mapping and final pool freeze still required.' if status=='PASS_INTERNAL_CONTENT' else 'Do not train original item; adjudicate or create a separately versioned repair.',
                    'mechanical_checks':checks})
    statuses = Counter(x['status'] for x in out)
    types = {}
    for typ in ['multi_select','ordering']:
        types[typ] = dict(Counter(x['status'] for x in out if x['question_type']==typ))
    report = {
        'status':'SOURCE_FIRST_REVIEW_COMPLETE',
        'created_at_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'review_policy_sha256':sha(BASE/'SOURCE_FIRST_REVIEW_POLICY.md'),
        'reconstruction_freeze_sha256':sha(freeze_path),
        'original_inventory_sha256':sha(BASE/'reuse/INVENTORY.json'),
        'question_count':len(out),'root_count':len({x['root_id'] for x in out}),
        'status_counts':dict(statuses),'by_type':types,'mechanical_check_count':check_count,
        'training_admitted_count':0,'solver_calls':0,'training_runs':0,
        'semantic_reviewer_tasks':3,
        'original_gold_modified':False,
        'limitations':['Shared-model internal source reconstruction, not external semantic certification.',
                       'Source-read completeness and no-private-Gold access are reviewer declarations plus workflow logs, not access-control proofs.',
                       'HOLD/FAIL items are retained; no score-based filtering, repair or Gold change.',
                       'PASS_INTERNAL_CONTENT is content approval for pool preparation, not a claim all training runtime gates passed.'],
        'questions':out,
    }
    save(BASE/'reviews/ADMISSION_LEDGER.json',report)
    with (BASE/'reviews/ADMISSION_LEDGER.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['candidate_id','root_id','question_type','reviewer_recommendation','status','gold_agreement','training_admitted'])
        writer.writeheader()
        writer.writerows({k:x[k] for k in writer.fieldnames} for x in out)
    save(BASE/'reviews/PASS_CONTENT_CANDIDATES.json',{'status':'CONTENT_ONLY_NOT_TRAINING_ROWS',
                                                  'questions':[originals[x['candidate_id']]['original_question_record'] for x in out if x['status']=='PASS_INTERNAL_CONTENT']})
    print(json.dumps({k:report[k] for k in ['status','question_count','root_count','status_counts','by_type','mechanical_check_count','training_admitted_count']},ensure_ascii=False))


if __name__ == '__main__':
    main()
