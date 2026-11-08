"""Source/public-only I/O for manually supplied semantic judgments."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
BATCH = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_newroots_ab_validation_v115')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
root,stage = sys.argv[2:4]
folder = BATCH/'roots'/('root'+root)
public_name = 'PUBLIC_ASSETS.json' if stage=='assets' else 'PUBLIC_QUESTIONS.json'
out_name = 'ASSET_SOURCE_JUDGMENTS.json' if stage=='assets' else 'PUBLIC_SOURCE_JUDGMENTS.json'
context = Path('/tmp')/('v115_'+root+'_'+stage+'_review_start.json')
source_path,public_path = folder/'CUMULATIVE_SOURCE.json',folder/public_name
if sys.argv[1]=='read':
    c={'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(source_path),'public_sha256':sha(public_path)}
    with context.open('x') as h:json.dump(c,h)
    source=read(source_path)
    for t in source['turns']:print(t['turn_ref']+' | '+t['speaker']+': '+t['text'])
    print('PUBLIC',public_name);print(public_path.read_text())
else:
    c=read(context);source=read(source_path);public=read(public_path)
    assert sha(source_path)==c['source_sha256'] and sha(public_path)==c['public_sha256']
    turns={t['turn_ref']:t for t in source['turns']}
    data=json.load(sys.stdin); end=datetime.now(timezone.utc).isoformat()
    header={'root_id':'cc-train-episode-'+root,'reviewer':'/root/v113_setup','review_role':'independent_source_first_public_semantic_reviewer','source_sha256':sha(source_path),'public_sha256':sha(public_path),'full_source_read':True,'turn_refs_read':[t['turn_ref'] for t in source['turns']],'private_answers_read_before_save':False,'independent_of_question_author':True,'semantic_judgments_programmatic':False,'started_at_utc':c['started_at_utc'],'finished_at_utc':end,'created_at_utc':end,'backend':'CURRENT_CODEX_SESSION','exact_model_id':None,'external_semantic_certification':False}
    def citation(ref,quote=None):
        t=turns[ref];q=quote if quote is not None else t['text']; assert q and q in t['text']
        return {'turn_ref':ref,'speaker':t['speaker'],'quote':q}
    def statement(text,m):
        refs=m['refs']; primary=m['primary_ref']; assert primary in refs
        citations=[citation(ref) for ref in refs]
        assert m['relation'] in ('supports','contradicts','unknown') and m['reason'].strip()
        return {'proposition':text,'relation':m['relation'],'scope_valid':m['scope_valid'],'fact_distinct':m['fact_distinct'],'reason':m['reason'],'citations':citations,'primary_citation':citation(primary)}
    def ordering(q,m):
        assert set(m['items'])==set(q['items'])
        items={}
        for label,text in q['items'].items():
            v=m['items'][label]; assert isinstance(v['semantic_checks_pass'],bool)
            anchor=citation(v['ref'],v['quote']);alternatives=[citation(a['ref'],a['quote']) for a in v['alternatives']]
            items[label]={'proposition':text,**{field:v['semantic_checks_pass'] for field in ('supported','actor_ok','modality_ok','state_ok','neutral_stem_ok','single_event_ok')},'order_anchor':anchor,'support_refs':v.get('support_refs',[v['ref']]),'alternative_matching_anchors':alternatives,'reason':v['reason']}
        return {'candidate_id':q['candidate_id'],**{key:m[key] for key in ('stem_valid','scope_valid','coherent','unique_order','neutral_stem_ok','derived_order','alternative_orders','reason')},'items':items}
    if stage=='assets':
        assert set(data['statements'])=={s['statement_id'] for s in public['statements']}
        header['statements']={s['statement_id']:statement(s['proposition'],data['statements'][s['statement_id']]) for s in public['statements']}
        header['ordering_questions']=[ordering(q,data['ordering'][q['candidate_id']]) for q in public['ordering_questions']]
        header['stem_review']=data['stem_review']
    else:
        header['asset_judgments_sha256']=sha(folder/'ASSET_SOURCE_JUDGMENTS.json')
        assert set(data['questions'])=={q['candidate_id'] for q in public['questions']}
        judgments={}
        for q in public['questions']:
            m=data['questions'][q['candidate_id']]
            if q['question_type']=='ordering': judgments[q['candidate_id']]=ordering(q,m)
            else:
                assert set(m['options'])==set(q['options'])
                judgments[q['candidate_id']]={**{k:m[k] for k in ('stem_valid','scope_valid','fact_distinct','no_entangled_options','reason')},'options':{label:statement(text,m['options'][label]) for label,text in q['options'].items()}}
        header['questions']=judgments
    header['findings']=data['findings']
    with (folder/out_name).open('x') as h:json.dump(header,h,ensure_ascii=False,indent=2);h.write('\n')
    sys.path.insert(0,str(BATCH.parent.parent/'scripts'))
    from codex_sequential_questions import check_statement,check_ordering,scope_scenes,config,validate_final_review
    if stage=='assets':
        issues={sid:check_statement(j,j['proposition'],turns) for sid,j in header['statements'].items()}
        for q in public['ordering_questions']:
            j=next(x for x in header['ordering_questions'] if x['candidate_id']==q['candidate_id'])
            issues[q['candidate_id']]=check_ordering(q,j,source,scope_scenes(config(BATCH)))[0]
        assert not any(issues.values()),issues
    else:validate_final_review(folder,config(BATCH))
    print(json.dumps({'root':root,'stage':stage,'sha256':sha(folder/out_name),'path':str(folder/out_name)},ensure_ascii=False))
