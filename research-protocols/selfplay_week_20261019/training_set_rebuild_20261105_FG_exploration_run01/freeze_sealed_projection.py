import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'training_set_rebuild_20261105_F_supplement_run01'
p=S/'repaired_training_export_v1/TRAIN.jsonl';sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha=='46acd3d1725c8af4fea264a5cad2dbdc5b36e604a75ff04f5ea2db1b4b2b50e1'
rows=[]
for line in p.read_text().splitlines():
 d=json.loads(line);usr=next(x['content'] for x in d['messages'] if x['role']=='user');assert '\nQuestion:\n' in usr
 q=usr.rsplit('\nQuestion:\n',1)[1].split('\n\nOutput only one <box>',1)[0]
 rows.append({'candidate_id':d['candidate_id'],'root_id':d['root_id'],'question_type':d['question_type'],'question_and_displayed_options':q,'content_sha256':hashlib.sha256(q.encode()).hexdigest()})
assert len(rows)==135
out=R/'private_semantic_reference/SEALED135_QUESTION_PROJECTION_ADDENDUM.json'
with out.open('x') as f:json.dump({'source_path':str(p),'source_sha256':sha,'record_count':135,'projection':'exact question/displayed options, no Gold/answers/scores/context','items':rows,'supersedes_incomplete_121_projection_for_sealed_export_coverage':True},f,ensure_ascii=False,indent=2);f.write('\n')
print('sealed135 question-only projection frozen',hashlib.sha256(out.read_bytes()).hexdigest())
