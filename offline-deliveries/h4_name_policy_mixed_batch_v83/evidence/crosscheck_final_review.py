"""Open private Gold only after all independent source judgments validate."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from common import bind, load_multi, now, read, write


def main():
    home = HERE / 'multiselect'
    context = load_multi('review_schema').validate_final_sources(home)
    saved = [{'group': g['group'], **g['binding'],
              'source_judgments_saved_at_utc': g['judgment']['created_at_utc']}
             for g in context['groups']]
    first_private = now()
    pool_path = home / 'POOL.json'
    pool_binding = bind(pool_path)
    private = {q['candidate_id']: q for q in read(pool_path)['questions']}
    report = {'first_private_pool_read_at_utc': first_private,
              'timestamp_scope': 'This script first opens POOL.json after both source reviewer records are saved; it is not the first private Gold exposure of the root operator.',
              'crosschecker_prior_private_gold_exposure': False,
              'crosschecker_prior_exposure_source': 'None for current final child Gold before this script; per-round private asset crosschecks are known to root.',
              'both_source_judgments_saved_bindings': saved,
              'pool': pool_binding, 'questions': []}
    for group in context['groups']:
        for qid, judgment in group['rows'].items():
            q = private[qid]
            pub = context['questions'][qid]
            errors = []
            supported = [k for k, o in judgment['options'].items()
                         if o['relation'] == 'supports']
            if set(supported) != set(q['gold']):
                errors.append('independent_supported_labels_differ_from_gold')
            if not judgment['stem_valid']:
                errors.append('invalid_stem')
            if any(o['relation'] == 'unknown' or not o['scope_valid'] or not o['fact_distinct']
                   for o in judgment['options'].values()):
                errors.append('unknown_scope_or_distinctness_issue')
            if any(q.get(k) != pub.get(k) for k in
                   ('root_id', 'question_type', 'question', 'options', 'source_sha256', 'source_path')):
                errors.append('public_private_question_mismatch')
            source = read(q['source_path'])
            turns = {t['turn_ref']: t for t in source['turns']}
            options = []
            for label, option in judgment['options'].items():
                e = q.get('option_evidence', {}).get(label, {})
                turn = turns.get(e.get('turn_ref'), {})
                if (e.get('relation') != option['relation']
                        or e.get('speaker') != turn.get('speaker')
                        or not isinstance(e.get('quote'), str)
                        or not e.get('quote') or e['quote'] not in turn.get('text', '')):
                    errors.append('private_evidence_invalid:' + label)
                options.append({'label': label, 'proposition': option['proposition'],
                                'relation': option['relation'], 'reason': option['reason'],
                                'independent_citations': option['citations'], 'private_evidence': e})
            report['questions'].append({'candidate_id': qid, 'group': group['group'],
                                        'root_id': q['root_id'], 'source_sha256': q['source_sha256'],
                                        'gold': q['gold'], 'supported': supported, 'errors': errors,
                                        'options': options})
    write(HERE / 'evidence/PRIVATE_CROSSCHECK.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
