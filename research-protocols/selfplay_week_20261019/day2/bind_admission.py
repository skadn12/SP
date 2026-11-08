"""Bind prior content admission for the exact Day 2 diagnosis question versions."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / 'day1'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def main():
    checks = []
    def check(ok, name):
        checks.append({'check': name, 'pass': bool(ok)})
        if not ok:
            raise AssertionError(name)

    sealed = read(DAY1 / 'COMPLETE_MANIFEST.json')
    for rec in sealed['files']:
        p = DAY1 / rec['path']
        check(p.stat().st_size == rec['bytes'] and digest(p) == rec['sha256'],
              'day1_seal:' + rec['path'])
    public = read(DAY1 / 'diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    private = {q['candidate_id']: q for q in read(DAY1 / 'diagnosis_inputs/PRIVATE_QUESTIONS.json')['questions']}
    train = read(DAY1 / 'TRAIN_ROOTS.json')
    # TRAIN_ROOTS is a manifest, not a new split assignment.
    train_ids = set(train['root_ids'])
    roots, rows = {}, []
    for q in public:
        qid = q['candidate_id']
        check(q['root_id'] in train_ids, 'train:' + qid)
        source = Path(q['source_path'])
        parent = source.parent
        check(digest(source) == q['source_sha256'], 'source_hash:' + qid)
        s = read(source)
        check(s['root_id'] == q['root_id'], 'source_root:' + qid)
        turns = {t['turn_ref']: t for t in s['turns']}
        check(len(turns) == len(s['turns']), 'unique_turn_refs:' + qid)
        sr = read(parent / 'SOURCE_REVIEW.json')
        check(sr['source_sha256'] == q['source_sha256'] and sr['decision'] == 'READY'
              and sr.get('full_source_read') is True, 'source_ready:' + qid)
        check(set(sr['turn_refs_read']) == set(turns), 'source_review_coverage:' + qid)
        cross = read(parent / 'FINAL_CROSSCHECK.json')
        for field, file in [('public_sha256', 'PUBLIC_QUESTIONS.json'),
                            ('private_sha256', 'PRIVATE_GOLD.json'),
                            ('source_judgments_sha256', 'PUBLIC_SOURCE_JUDGMENTS.json')]:
            check(cross[field] == digest(parent / file), 'historical_crosscheck_binding:' + qid + ':' + file)
        matches = [r for r in cross['results'] if r['candidate_id'] == qid]
        check(len(matches) == 1, 'single_admission:' + qid)
        decision = matches[0]
        check(decision['status'] == 'CONTENT_PASS' and not decision['issues']
              and decision.get('original_gold_unchanged') is True, 'content_pass:' + qid)
        original = next(x for x in read(parent / 'PUBLIC_QUESTIONS.json')['questions'] if x['candidate_id'] == qid)
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        for key in ['candidate_id', 'root_id', 'question_type', 'question', field]:
            check(q[key] == original[key], 'unchanged_public:' + qid + ':' + key)
        check(private[qid]['gold'] == decision['source_derived_answer'], 'unchanged_gold:' + qid)
        evidence = private[qid]['option_evidence' if field == 'options' else 'item_evidence']
        check(set(evidence) == set(q[field]), 'complete_evidence:' + qid)
        citation_count = 0
        for label, e in evidence.items():
            citations = e.get('citations') or [e.get('primary_citation') or e]
            check(bool(citations), 'nonempty_citations:' + qid + ':' + label)
            for c in citations:
                t = turns.get(c.get('turn_ref'))
                check(t is not None and c.get('speaker') == t['speaker']
                      and bool(c.get('quote')) and c['quote'] in t['text'],
                      'exact_citation:' + qid + ':' + label + ':' + str(citation_count))
                citation_count += 1
        record_paths = ['SOURCE_REVIEW.json', 'FINAL_CROSSCHECK.json',
                        'PUBLIC_SOURCE_JUDGMENTS.json', 'PUBLIC_QUESTIONS.json', 'PRIVATE_GOLD.json']
        roots.setdefault(q['root_id'], {'source': binding(source),
                         'prior_records': [binding(parent / f) for f in record_paths]})
        rows.append({'candidate_id': qid, 'root_id': q['root_id'], 'question_type': q['question_type'],
                     'decision': 'PASS_PRIOR_VERSION_BOUND', 'score_used': False,
                     'content_review_record': binding(parent / 'FINAL_CROSSCHECK.json'),
                     'source_binding': binding(source), 'checked_exact_citations': citation_count,
                     'semantic_re_review_performed_today': False})
    check(len(rows) == 39 and len(roots) == 20, 'expected_scope')
    check(Counter(q['question_type'] for q in rows) == {'multi_select': 19, 'ordering': 20}, 'expected_types')
    out = {'status': 'PASS', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'purpose': 'Bind exact existing content admission before new S0 diagnosis; no score selection.',
           'namespace': 'train.feedback', 'check_count': len(checks), 'error_count': 0,
           'day1_manifest': binding(DAY1 / 'COMPLETE_MANIFEST.json'),
           'public_questions': binding(DAY1 / 'diagnosis_inputs/PUBLIC_QUESTIONS.json'),
           'private_questions': binding(DAY1 / 'diagnosis_inputs/PRIVATE_QUESTIONS.json'),
           'questions': rows, 'roots': roots, 'checks': checks,
           'limits': ['Prior semantic admission is reused, not independently re-certified today.',
                      'Parent content admission work has Gold exposure; Solver request construction has a separate public-only path.',
                      'Legacy113 results concern other question/source versions and are not transferred to these39.',
                      'These are feedback questions on previously exposed Train roots, not final evaluation material.']}
    target = HERE / 'CONTENT_ADMISSION.json'
    with target.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': out['status'], 'questions': len(rows), 'roots': len(roots),
                      'checks': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
