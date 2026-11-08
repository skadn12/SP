"""Check review completeness/bindings, not the semantic judgments themselves."""
import argparse, json
from collections import Counter
from control import BASE, read, bind, verify, write, now
from report import status


def validate(partial=False):
    public = {q['candidate_id']:q for q in read(BASE/'inputs/PUBLIC_QUESTIONS.json')['questions']}
    scores = {r['candidate_id']:r for r in read(BASE/'solver_runtime/execution/SCORES.json')['records'] if r['branch']=='reason'}
    protocol = read(BASE/'PROTOCOL_FREEZE.json')
    bindings, units, total_citations = [], [], 0
    for root in protocol['roots']:
        for arm in protocol['conditions']:
            expected = {cid for cid,q in public.items() if q['root_id']==root and q['condition']==arm}
            folder = BASE/'reason_semantic_reviews'/root/arm
            rp = folder/'REVIEW.json'
            if partial and not rp.exists():
                continue
            review = read(rp)
            assert review['root_id']==root and review['condition']==arm
            assert review['private_scores_read'] is False and review['weakness_cards_read'] is False
            assert review['source_binding']==bind(BASE/'sources'/root/arm/'FINAL_SOURCE.json')
            assert review['raw_freeze_binding']==bind(BASE/'solver_runtime/execution/RAW_RESPONSE_FREEZE.json')
            assert review['key_freeze_binding']==bind(folder/'KEY_FREEZE.json')
            key = verify(review['key_freeze_binding'])
            assert key['reviewer_id']==review['reviewer_id']
            assert key['source_binding']==review['source_binding']
            assert key['private_scores_read'] is False and key['weakness_cards_read'] is False
            key_rows = key.get('records',key.get('candidates',key.get('questions',[])))
            assert len(key_rows)==len(expected) and {r['candidate_id'] for r in key_rows}==expected
            key_rows = {r['candidate_id']:r for r in key_rows}
            records = review['records']
            assert len(records)==len(expected) and {r['candidate_id'] for r in records}==expected
            source = verify(review['source_binding'])
            source_refs = {t['turn_ref']:t for t in source['turns']}
            read_refs = next(key[k] for k in ('full_ordered_turn_refs_read','all_ordered_turn_refs_read','ordered_turn_refs_read') if k in key)
            assert read_refs==[t['turn_ref'] for t in source['turns']]
            source_authors = {read(p)['author_id'] for p in (BASE/'sources'/root/arm).glob('FBC1_*/DRAFT.json')}
            assert review['reviewer_id'] not in source_authors
            for record in records:
                cid = record['candidate_id']; score = scores[cid]; question = public[cid]
                assert record['request_id']==score['request_id']
                assert record['raw_binding']==score['raw_binding']
                raw = verify(record['raw_binding'])
                assert raw['status']=='RETURNED' and score['reason_structure_valid'] is True
                obj = json.loads(raw['raw_output'])
                derived = record['independently_derived_display_gold']
                frozen_key = key_rows[cid]
                frozen_derived = next(frozen_key[k] for k in ('independently_derived_display_gold','independently_derived_display_key','displayed_key') if k in frozen_key)
                assert (set(derived)==set(frozen_derived) if question['question_type']=='multi_select' else derived==frozen_derived), ('FROZEN_KEY_DRIFT',cid)
                assert review['reviewer_id']!=read(BASE/'questions'/root/arm/(cid.rsplit('__',1)[-1]+'.json'))['author_id']
                assert (set(derived)==set(score['display_gold']) if question['question_type']=='multi_select' else derived==score['display_gold']), ('INDEPENDENT_KEY_DISAGREEMENT',cid)
                assert status(record['final_answer'])==('PASS' if score['exact'] else 'FAIL'), ('FINAL_METRIC_DISAGREEMENT',cid)
                items = record['item_findings']
                if isinstance(items,dict):
                    items = {item.get('display_label',label):item for label,item in items.items()}
                else:
                    assert len(items)==len(question['display'])
                    items = {item['display_label']:item for item in items}
                assert set(items)==set(question['display'])
                dimensions = [status(record[k]) for k in ('schema_and_completeness','final_answer','answer_reason_relation','answer_reason_final_consistency','answer_reason_source_support','answer_reason_actor_attribution')]
                for label,item in items.items():
                    dimensions += [status(item[k]) for k in ('judgment_verdict','citation_sufficiency','citation_precision','reason_source_support','actor_attribution')]
                    assert Counter(c['turn_ref'] for c in item['citations'])==Counter(obj['item_evidence'][label]['turn_refs']), ('CITATION_COVERAGE_GAP',cid,label)
                    for citation in item['citations']:
                        ref = citation['turn_ref']; cls = citation['classification']
                        assert cls in {'DIRECT','PARTIAL','CONTEXT_ONLY','UNRELATED','NONEXISTENT'}
                        if ref in source_refs:
                            assert cls!='NONEXISTENT'
                            assert isinstance(citation['quote'],str) and citation['quote'] and citation['quote'] in source_refs[ref]['text'], ('NONEXACT_REVIEW_QUOTE',cid,label,ref)
                        else:
                            assert cls=='NONEXISTENT'
                        total_citations += 1
                    if any(c['classification'] in {'UNRELATED','NONEXISTENT'} for c in item['citations']):
                        assert status(item['citation_precision'])=='FAIL', ('CITATION_PRECISION_AGGREGATION_MISMATCH',cid,label)
                assert all(s in {'PASS','HOLD','FAIL','NOT_APPLICABLE'} for s in dimensions)
                strict = 'FAIL' if 'FAIL' in dimensions else ('HOLD' if 'HOLD' in dimensions else 'PASS')
                assert status(record['decision'])==strict, ('STRICT_AGGREGATION_MISMATCH',cid,strict)
            bindings += [bind(rp),bind(folder/'KEY_FREEZE.json')]
            units.append({'root_id':root,'condition':arm,'records':len(records)})
    if partial:
        return {'status':'PARTIAL_ACCOUNTING_CHECK_ONLY','completed_units':len(units),
                'records_checked':sum(u['records'] for u in units),'citations_checked':total_citations,'writes':0}
    assert sum(u['records'] for u in units)==len(public)==read(BASE/'QUESTION_FREEZE.json')['admitted_questions']
    result = {'schema_version':'fbc1_reason_review_structural_validation_v1','status':'PASS_STRUCTURAL_BINDINGS_AND_ACCOUNTING_ONLY',
              'created_at_utc':now(),'reviewer':'/root','semantic_re_adjudication':False,'records':len(public),'citations_checked':total_citations,
              'units':units,'bindings':bindings+[bind(BASE/'scripts/validate_reason_reviews.py'),bind(BASE/'solver_runtime/execution/SCORES.json')]}
    write(BASE/'REASON_REVIEW_VALIDATION.json',result)
    return {'status':result['status'],'records':len(public),'citations_checked':total_citations}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    print(json.dumps(validate(parser.parse_args().partial),ensure_ascii=False))
