"""Apply registered common option relabeling to accepted content, no resampling."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
spec_path=D/'PRESENTATION_TRANSFORM_FREEZE.json'
spec=read(spec_path);seed=spec['seed'];rows=[];hist=defaultdict(Counter)
final_paths=sorted((D/'final_records').glob('*.json'))
public_ids={read(p)['candidate_id'] for p in (D/'candidates').glob('*/*/public/*.json')}
assert len(public_ids)==175 and {p.stem for p in final_paths}==public_ids, 'final records must cover all175 authored candidates before export'
for p in final_paths:
    r=read(p)
    if not r['disposition']['state'].startswith('ACCEPT'):continue
    original_path=Path(r['generation_history'][-1]['raw_response_path']);pub=read(original_path)
    assert sha(original_path)==r['generation_history'][-1]['candidate_content_sha256']
    order=sorted('ABCDEF',key=lambda old:hashlib.sha256((seed+'|'+r['inherited_slot_id']+'|'+old).encode()).hexdigest())
    mapping={old:new for old,new in zip(order,'ABCDEF')}
    out=dict(pub);out['options_or_items']={new:pub['options_or_items'][old] for new,old in zip('ABCDEF',order)}
    original_gold=(r.get('multi_select') or r.get('ordering'))
    original_gold=original_gold.get('gold_option_ids',original_gold.get('gold_item_ids'))
    displayed_gold=[mapping[x] for x in original_gold]
    if r['question_type']=='multi_select':displayed_gold.sort()
    assert sorted(mapping)==list('ABCDEF') and sorted(mapping.values())==list('ABCDEF')
    assert {old:out['options_or_items'][new] for old,new in mapping.items()}==pub['options_or_items']
    inverse={v:k for k,v in mapping.items()}
    if r['question_type']=='ordering':
        assert [inverse[x] for x in displayed_gold]==original_gold
    else:
        assert set(inverse[x] for x in displayed_gold)==set(original_gold)
    dest=D/'presentation/public'/(r['candidate_id']+'.json');dump(dest,out)
    rows.append({'candidate_id':r['candidate_id'],'arm':r['arm'],'root_id':r['root_id'],'slot_id':r['inherited_slot_id'],
       'question_type':r['question_type'],'original_public':{'path':str(original_path),'sha256':sha(original_path)},
       'display_public':{'path':str(dest),'sha256':sha(dest)},'mapping_old_to_new':mapping,
       'original_gold':original_gold,'display_gold':displayed_gold,'source_and_anchors_changed':False,
       'training_or_eval_pool_membership':'NOT_YET_FROZEN'})
    hist[r['arm']+'|'+r['question_type']].update([''.join(displayed_gold)])
accepted_ids={read(p)['candidate_id'] for p in final_paths if read(p)['disposition']['state'].startswith('ACCEPT')}
assert {x['candidate_id'] for x in rows}==accepted_ids and rows, 'presentation export must match nonempty accepted set'
dump(D/'presentation/PRIVATE_LABEL_MAP.json',{'transform_freeze':{'path':str(spec_path),'sha256':sha(spec_path)},
     'rows':rows,'answer_histograms':{k:dict(v) for k,v in hist.items()},
     'roundtrip_checks':len(rows),'status':'PASS'})
print(json.dumps({'transformed_candidates':len(rows),'histograms':{k:dict(v) for k,v in hist.items()}}))
