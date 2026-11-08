from pathlib import Path
import json,hashlib,datetime
r=Path(__file__).resolve().parent.parent;checks=0
for run in sorted((r/'assets').iterdir()):
 if not run.is_dir():continue
 s=json.loads((run/'SOURCE.json').read_text());p=json.loads((run/'PUBLIC_ASSETS.json').read_text());j=json.loads((run/'SOURCE_JUDGMENTS.json').read_text());v=json.loads((run/'ASSET_REVIEW.json').read_text());rec=json.loads((run/'REVIEW_RECEIPT.json').read_text());turns={t['turn_ref']:t for t in s['turns']}
 assert j['gold_read_before_save'] is False and j['solver_scores_used'] is False;checks+=2
 assert {x['version_id'] for x in j['versions']}=={x['version_id'] for x in p['versions']}=={x['version_id'] for x in v['versions']};checks+=1
 assert j['source_sha256']==hashlib.sha256((run/'SOURCE.json').read_bytes()).hexdigest();checks+=1
 assert j['public_assets_sha256']==hashlib.sha256((run/'PUBLIC_ASSETS.json').read_bytes()).hexdigest();checks+=1
 saved=rec.get('source_judgments_saved_at_utc',rec.get('source_judgments_saved_at'));started=rec.get('private_crosscheck_started_at_utc',rec.get('private_read_started_at'));completed=rec.get('review_completed_at_utc',rec.get('review_completed_at'))
 assert saved<=started<=completed;checks+=1
 for row in j['versions']:
  for c in row['citations']:
   assert c['speaker']==turns[c['turn_ref']]['speaker'] and c['quote'] in turns[c['turn_ref']]['text'];checks+=2
 files=rec['files']; bindings=[{'path':str(run/k),'sha256':h} for k,h in files.items()] if isinstance(files,dict) else files
 for b in bindings:
  assert b['sha256']==hashlib.sha256(Path(b['path']).read_bytes()).hexdigest();checks+=1
v={'status':'PASS','checks':checks,'slot_reviews':10,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'parent_rechecks_all_asset_review_public_source_and_exact_citation_bindings_and_source_first_timing','semantic_review_not_reopened':True,'solver_scores_used':False,'model_calls':0,'issues':[],'receipt_schema_aliases_supported':True,'precheck_note':'Two initial in-memory checks stopped on timing then file-map schema differences before writing receipts; both reviewer schemas are now explicitly supported.'}
with (r/'execution/ASSET_REVIEW_BINDING_PRECHECK.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(v)
