from pathlib import Path
import json,hashlib,datetime,sys
R=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def write(p,v):
 text=json.dumps(v,ensure_ascii=False,indent=2)+'\n';assert json.loads(text)==v
 with p.open('x') as f:f.write(text)
assert len(sys.argv)>1, 'supply reviewed slot ids'
for sid in sys.argv[1:]:
 run=R/'assets'/sid;j=read(run/'SOURCE_JUDGMENTS.json')
 assert j['gold_read_before_save'] is False
 private_read_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
 a=read(run/'ASSETS.json');m=read(run/'VERSION_MAP.json');p=read(run/'PUBLIC_ASSETS.json');s=read(run/'SOURCE.json')
 facts={f['fact_id']:f for f in a['facts']};maps={v['version_id']:v for v in m['versions']};pubs={v['version_id']:v for v in p['versions']};turns={t['turn_ref']:t for t in s['turns']}
 assert j['source_sha256']==sha(run/'SOURCE.json') and j['public_assets_sha256']==sha(run/'PUBLIC_ASSETS.json')
 rows=[]
 for v in j['versions']:
  vm=maps[v['version_id']];f=facts[vm['fact_id']];w=f[vm['variant']]
  assert w['proposition']==pubs[v['version_id']]['proposition'] and w['relation']==vm['relation']
  for c in v['citations']:assert c['speaker']==turns[c['turn_ref']]['speaker'] and c['quote'] in turns[c['turn_ref']]['text']
  anchor=f['anchor'];assert anchor['speaker']==turns[anchor['turn_ref']]['speaker'] and anchor['quote'] in turns[anchor['turn_ref']]['text']
  usable=j['stem_valid'] and v['scope_valid'] and v['fact_distinct'] and v['relation']!='unknown' and v['relation']==w['relation']
  rows.append({'version_id':v['version_id'],'usable':usable,'relation':v['relation'],'reason':v['reason']+' Writer label '+('agrees.' if v['relation']==w['relation'] else 'does not agree; retain original label and exclude version.')})
 review={'version':'v69_asset_private_crosscheck_v1','source_sha256':sha(run/'SOURCE.json'),'assets_sha256':sha(run/'ASSETS.json'),'public_assets_sha256':sha(run/'PUBLIC_ASSETS.json'),'source_judgments_sha256':sha(run/'SOURCE_JUDGMENTS.json'),'solver_scores_used':False,'source_first':True,'stem_valid':j['stem_valid'],'versions':rows}
 write(run/'ASSET_REVIEW.json',review)
 receipt={'version':'v69_asset_source_first_receipt_v1','slot_id':sid,'source_first':True,'source_judgments_saved_at_utc':j['created_at_utc'],'private_crosscheck_started_at_utc':private_read_at,'review_completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'solver_scores_used':False,'source_and_propositions_unchanged':True,'unknown_not_false':True,'independent_judgment_versions':len(rows),'usable_supports':sum(v['usable'] and v['relation']=='supports' for v in rows),'usable_contradicts':sum(v['usable'] and v['relation']=='contradicts' for v in rows),'null_reason_diagnostic':{'contradicted_null':sum(f['contradicted'] is None for f in a['facts']),'reason_contract_valid':all((f['contradiction_unavailable_reason'] is None) if f['contradicted'] is not None else bool(f['contradiction_unavailable_reason']) for f in a['facts'])},'files':[bind(run/n) for n in ['SOURCE.json','PUBLIC_ASSETS.json','SOURCE_JUDGMENTS.json','ASSETS.json','VERSION_MAP.json','ASSET_REVIEW.json']]}
 write(run/'REVIEW_RECEIPT.json',receipt)
 print(sid,receipt['usable_supports'],receipt['usable_contradicts'])
