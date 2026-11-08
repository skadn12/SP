"""Check role separation, complete source coverage, binding and actual recorded times.

Mechanical provenance audit only; does not decide semantic source/answer validity.
Writes one new result and refuses overwrite.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def actor(d):
    return d.get('reviewer') or d.get('author') or d.get('author_agent') or d.get('author_role')

def main():
    records = []
    for row in read(HERE/'SOURCE_RESERVATIONS.json')['allocation']:
        rid = row['root_id']
        folder = HERE/'roots'/('root'+rid.rsplit('-',1)[1])
        source = read(folder/'CUMULATIVE_SOURCE.json')
        refs = [t['turn_ref'] for t in source['turns']]
        author = read(folder/'AUTHOR_RECEIPT.json')
        review = read(folder/'SOURCE_REVIEW.json')
        assert actor(author) and actor(review) and actor(author) != actor(review), (rid,'source_role_separation')
        assert review['root_id'] == rid and review['source_sha256'] == sha(folder/'CUMULATIVE_SOURCE.json')
        assert review['full_source_read'] is True and review['turn_refs_read'] == refs
        files = ['AUTHOR_RECEIPT.json','SOURCE_REVIEW.json']
        if review['decision'] == 'READY':
            qa = read(folder/'QA_AUTHOR_RECEIPT.json')
            assert actor(qa), (rid,'missing_qa_actor')
            assert qa['source_sha256'] == sha(folder/'CUMULATIVE_SOURCE.json')
            assert qa['full_source_read'] is True and qa['turn_refs_read'] == refs
            for name, public in [('ASSET_SOURCE_JUDGMENTS.json','PUBLIC_ASSETS.json'),('PUBLIC_SOURCE_JUDGMENTS.json','PUBLIC_QUESTIONS.json')]:
                d = read(folder/name)
                assert actor(d) and actor(d) != actor(qa), (rid,name,'question_role_separation')
                assert d['source_sha256'] == sha(folder/'CUMULATIVE_SOURCE.json')
                assert d['public_sha256'] == sha(folder/public)
                assert d['full_source_read'] is True and d['turn_refs_read'] == refs
                assert d['private_answers_read_before_save'] is False
                assert d['independent_of_question_author'] is True
            files += ['QA_AUTHOR_RECEIPT.json','ASSET_SOURCE_JUDGMENTS.json','PUBLIC_SOURCE_JUDGMENTS.json']
        else:
            assert review['decision'] == 'HOLD'
            assert not (folder/'QA_DRAFT.json').exists(), (rid,'qa_after_hold')
        for name in files:
            p = folder/name
            d = read(p)
            start = datetime.fromisoformat(d['started_at_utc'].replace('Z','+00:00'))
            end = datetime.fromisoformat(d['finished_at_utc'].replace('Z','+00:00'))
            assert start.tzinfo and end.tzinfo and start <= end
            assert end.timestamp() <= p.stat().st_mtime + 2, (rid,name,'finish_after_file_save')
            records.append({'root_id':rid,'stage_record':name,'actor':actor(d),'path':str(p), 'sha256':sha(p),'started_at_utc':d['started_at_utc'],'finished_at_utc':d['finished_at_utc']})
    result = {'status':'PASS_MECHANICAL_PROVENANCE_ONLY','created_at_utc':datetime.now(timezone.utc).isoformat(),'records':records,'record_count':len(records),'semantic_validity_certified':False,'external_independence_claimed':False,'model_calls':0}
    with (HERE/'evidence/STAGE_RECORD_COMPLETENESS_AUDIT.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps({'status':result['status'],'record_count':len(records)}))

if __name__ == '__main__':
    main()
