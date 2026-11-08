"""Seal the completed direct-Codex batch after an independent delivery audit."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def read(name):return json.loads((BASE/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,obj):
    p=BASE/name;assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
audit=read('evidence/FINAL_DELIVERY_AUDIT.json')
assert audit['decision']=='PASS' and not audit['blocking_issues'],audit
a=read('FINAL_BATCH_ACCOUNTING.json')
selected=read('SELECTED_PUBLIC_QUESTIONS.json')['questions']
answers=read('SELECTED_PRIVATE_GOLD.json')['answers']
controls=read('CONTROL_PUBLIC_QUESTIONS.json')['questions']
outcomes=read('ALL_OUTCOMES_INDEX.json')['all_candidates']
assert len(outcomes)==28 and len(selected)==18 and len(controls)==4
assert len({q['candidate_id'] for q in selected})==len(selected)
assert {q['candidate_id'] for q in selected}=={q['candidate_id'] for q in answers}
assert len([o for o in outcomes if o['status']=='PASS'])==22
assert len([o for o in outcomes if o['status']=='FAIL'])==6
assert a['new_multiselect_gold_distribution']=={'2':2,'3':4,'4':3,'5':3}
finished=datetime.now(timezone.utc)
elapsed=(finished-datetime.fromisoformat(a['started_at_utc'])).total_seconds()
complete={'schema_version':'v104_codex_session_content_freeze_v1','frozen_at_utc':finished.isoformat(),'elapsed_wall_seconds_from_preparation':elapsed,'counts':{'roots_attempted':4,'source_ready_roots':4,'new_turns':56,'public_candidates':28,'new_content_candidates':18,'multiselect_gold2_5':12,'ordering':6,'gold6_controls':4,'excluded_candidates':6},'gold2_3_4_5':{'2':2,'3':4,'4':3,'5':3},'status':'CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_APPROVED','solver_run':False,'training_run':False,'underlying_model_usage_exposed':False,'direct_codex_session':True,'production_qwen_transport_equivalence_claimed':False,'source_and_question_repairs':0,'known_limitations':['Same-model generation/review; roles differ but this is not external-model certification.','Four purposively selected exposed roots and compatible fresh events, not a random holdout.','Question families are correlated and mostly local-scene reading tasks; difficulty unmeasured.','Historical exact duplicates were checked within discovered public six-sentence records; complete semantic novelty not certified.','Review overwrite restored from original retained snapshot before Gold construction; receipt preserved.','Earlier derived snapshots retained for audit; only top-level FINAL_BATCH_ACCOUNTING and SELECTED files are authoritative.'],'independent_delivery_audit_sha256':sha(BASE/'evidence/FINAL_DELIVERY_AUDIT.json'),'accounting_sha256':sha(BASE/'FINAL_BATCH_ACCOUNTING.json')}
save('CONTENT_FREEZE.COMPLETE.json',complete)
files=[{'path':str(p.relative_to(BASE)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(BASE.rglob('*')) if p.is_file() and p.name!='COMPLETE_MANIFEST.json']
save('COMPLETE_MANIFEST.json',{'schema_version':'v104_codex_session_delivery_manifest_v1','sealed_at_utc':datetime.now(timezone.utc).isoformat(),'batch_status':complete['status'],'elapsed_wall_seconds_from_preparation':elapsed,'files':files,'file_count':len(files),'file_integrity_is_not_semantic_certification':True,'authoritative_entrypoints':['COMPLETE_RESULTS_ZH.md','FINAL_BATCH_ACCOUNTING.json','CONTENT_FREEZE.COMPLETE.json','SELECTED_PUBLIC_QUESTIONS.json','SELECTED_PRIVATE_GOLD.json','CONTROL_PUBLIC_QUESTIONS.json','CONTROL_PRIVATE_GOLD.json','ALL_OUTCOMES_INDEX.json','REJECTED_CANDIDATES.json'],'archived_non_authoritative_directories':['evidence/finalizer_attempt01_hash_encoding_bug/','evidence/attempt02_before_final_actor_adjudication/']})
manifest=read('COMPLETE_MANIFEST.json')
assert all(sha(BASE/entry['path'])==entry['sha256'] for entry in manifest['files'])
assert len(manifest['files'])==len({entry['path'] for entry in manifest['files']})
print(json.dumps({'status':complete['status'],'files_sealed':len(files),'elapsed_seconds':elapsed,'freeze_counts':complete['counts'],'manifest_verified':True},ensure_ascii=False,indent=2))
