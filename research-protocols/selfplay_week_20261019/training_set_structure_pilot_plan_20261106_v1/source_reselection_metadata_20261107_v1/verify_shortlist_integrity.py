import json,hashlib
from pathlib import Path
P=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1'); OUT=P/'source_reselection_metadata_20261107_v1'; src=Path('/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl')
sel=json.loads((OUT/'METADATA_SHORTLIST.json').read_text())['selected']; ids={x['data_id'] for x in sel}; found={}
for i,line in enumerate(src.open(encoding='utf-8')):
 d=json.loads(line); n=d['dataID']
 if n in ids:
  found[n]={'root_id':'cc-train-'+n,'data_id':n,'source_line_index':i,'relationship':d.get('relationship'),'record_sha256':hashlib.sha256(line.rstrip('\n').encode()).hexdigest(),'raw_record':d}
assert ids==set(found), (ids,set(found))
rows=[]
for x in sel:
 r=found[x['data_id']].copy(); r.pop('raw_record'); r['metadata_rank_hash']=x['rank_hash']; r['ledger_record_sha256']=x['record_sha256']; r['record_hash_match']=r['record_sha256']==x['record_sha256']; rows.append(r)
(OUT/'SHORTLIST_SOURCE_RECORDS.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
(OUT/'SHORTLIST_SOURCE_INTEGRITY.json').write_text(json.dumps({'schema_version':'cc_shortlist_source_integrity_v1','source_path':str(src),'source_sha256':'f0fccf01ec59eeeab68c0a300b0f1a24bacd8eea8719e77bdbd003c242f1407c','selected_count':len(rows),'rows':rows,'all_record_hashes_match':all(r['record_hash_match'] for r in rows),'source_content_now_viewed':True,'generation_authorization':'NONE'},ensure_ascii=False,indent=2))
print(json.dumps({'selected_count':len(rows),'all_record_hashes_match':all(r['record_hash_match'] for r in rows),'rows':rows},ensure_ascii=False))
