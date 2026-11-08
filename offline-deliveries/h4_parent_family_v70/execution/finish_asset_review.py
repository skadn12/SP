from pathlib import Path
import json,hashlib,datetime,sys
R=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
for sid in sys.argv[1:]:
 run=R/'assets'/sid;j=read(run/'SOURCE_JUDGMENTS.json')
 assert j['new_variant_private_artifacts_read_before_save'] is False and j['solver_scores_used'] is False
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 a=read(run/'ASSETS.json');m=read(run/'VERSION_MAP.json');p=read(run/'PUBLIC_ASSETS.json');s=read(run/'SOURCE.json')
 facts={f['fact_id']:f for f in a['facts']};maps={v['version_id']:v for v in m['versions']};pubs={v['version_id']:v for v in p['versions']};turns={t['turn_ref']:t for t in s['turns']}
 assert j['source_sha256']==sha(run/'SOURCE.json') and j['public_assets_sha256']==sha(run/'PUBLIC_ASSETS.json')
 def exact(c):assert c['speaker']==turns[c['turn_ref']]['speaker'] and c['quote'] in turns[c['turn_ref']]['text']
 def check(row,writer,key):
  for c in row['citations']:exact(c)
  exact(writer['anchor'])
  usable=j['stem_valid'] and row['scope_valid'] and row['fact_distinct'] and row['relation']!='unknown' and row['relation']==writer['relation']
  return {key:row[key],'usable':usable,'relation':row['relation'],'reason':row['reason']+' Writer relation '+('agrees.' if row['relation']==writer['relation'] else 'differs; retain original label and exclude.')}
 parents=[check(row,facts[row['fact_id']]['supported'],'fact_id') for row in j['parents']]
 versions=[]
 for row in j['versions']:
  vm=maps[row['version_id']];f=facts[vm['fact_id']];w=f['contradicted'];assert w['proposition']==pubs[row['version_id']]['proposition'] and w['relation']==vm['relation']
  versions.append(check(row,w,'version_id'))
 review={'version':'v70_asset_private_crosscheck_v1','source_sha256':sha(run/'SOURCE.json'),'assets_sha256':sha(run/'ASSETS.json'),'public_assets_sha256':sha(run/'PUBLIC_ASSETS.json'),'source_judgments_sha256':sha(run/'SOURCE_JUDGMENTS.json'),'solver_scores_used':False,'source_first':True,'parent_all_true_provenance_known':True,'new_variant_private_artifacts_read_before_save':False,'stem_valid':j['stem_valid'],'parents':parents,'versions':versions}
 write(run/'ASSET_REVIEW.json',review)
 receipt={'version':'v70_asset_source_first_receipt_v1','family_id':sid,'source_first':True,'parent_all_true_provenance_known':True,'new_variant_private_artifacts_read_before_save':False,'source_judgments_saved_at_utc':j['created_at_utc'],'private_crosscheck_started_at_utc':started,'review_completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'solver_scores_used':False,'usable_parents':sum(x['usable'] for x in parents),'usable_contradicts':sum(x['usable'] for x in versions),'files':[bind(run/n) for n in ['SOURCE.json','PARENT.json','PUBLIC_ASSETS.json','SOURCE_JUDGMENTS.json','ASSETS.json','VERSION_MAP.json','ASSET_REVIEW.json']]}
 write(run/'REVIEW_RECEIPT.json',receipt);print(sid,receipt['usable_parents'],receipt['usable_contradicts'])
