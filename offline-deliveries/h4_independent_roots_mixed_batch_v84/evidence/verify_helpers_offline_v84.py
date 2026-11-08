"""Exercise append-only export on explicit synthetic mixed fixtures only."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil,subprocess,tempfile
HOME=Path(__file__).resolve().parents[1]
OLD=HOME.parent/'h4_name_policy_mixed_batch_v83'
PYTHON='/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
checks=[]
for name in ['audit_generation_v84.py','audit_results_v84.py','audit_union_v84.py','export_audited.py','test_audit_bindings_v84.py']:
    ast.parse((HOME/'evidence'/name).read_text(),filename=name)
checks.append('All five authored Python artifacts parse without importing audit hooks or models')
with tempfile.TemporaryDirectory(prefix='v84_synthetic_evidence_helpers_') as temp:
    task=Path(temp)/'h4_independent_roots_mixed_batch_v84';(task/'evidence').mkdir(parents=True)
    (task.parent/OLD.name).symlink_to(OLD)
    shutil.copyfile(HOME/'common.py',task/'common.py')
    for name in ['export_audited.py']:
        shutil.copyfile(HOME/'evidence'/name,task/'evidence'/name)
    # Explicit temporary stand-ins test invocation and reporting, not source semantics.
    (task/'source_admission.py').write_text("def verify():\n    return {'roots': [{'decision': 'READY'} for _ in range(4)]}\n")
    (task/'source_subset.py').write_text("def validate(config):\n    assert len(config['multiselect_config']['slots']) == 4\n    return config\n")
    roots=['cc-train-episode-'+r for r in ['103866','109164','112812','135466']]
    def question(i,kind,gold):
        field='options' if kind=='multi_select' else 'items';labels=list('ABCDEF') if field=='options' else ['I'+str(j) for j in range(1,7)]
        return {'candidate_id':'synthetic-v84-'+str(i),'root_id':roots[i],'question_type':kind,
          'question':'Synthetic offline fixture only.',field:{k:'Synthetic statement '+str(i)+' '+k for k in labels},'gold':labels[:gold]}
    selected=[question(0,'multi_select',2),question(1,'multi_select',3),question(2,'multi_select',6),question(3,'ordering',6)]
    write(task/'SELECTED_CANDIDATES.json',{'questions':selected})
    write(task/'EVALUATION_POOL.json',{'questions':selected})
    write(task/'BATCH_REVIEW.COMPLETE.json',{'status':'FROZEN','fixture_only':True})
    write(task/'SOURCE_ADMISSION.COMPLETE.json',{'status':'FROZEN','fixture_only':True})
    for name in ['GENERATION_AUDIT','INPUT_AUDIT','RESULT_AUDIT','UNION_AUDIT']:
        write(task/'evidence'/(name+'.json'),{'status':'PASS','check_count':1,'fixture_only':True,
          'totals':{'actual_generation_http_calls':8}})
    mixed={'by_type':{'ordering':{'mechanically_assembled':1}},'calls_per_model':66}
    write(task/'MIXED_EVALUATION_REPORT.json',mixed)
    config={'multiselect_config':{'slots':[{'root_id':r,'target_counts':[2,3,4,5],'control_count':1} for r in roots]},
      'ordering_slots':[{'root_id':r,'candidate_count':2} for r in roots],'max_generation_http':12,'max_solver_calls_per_model':456}
    for key,name in [('ordering_anchor_plan','SCENE_PLANS.json'),('character_registry','CHARACTER_REGISTRY.json'),('actor_reference_policy','ACTOR_REFERENCE_POLICY.json')]:
        write(task/name,{'fixture_only':True});config[key]=bind(task/name)
    write(task/'CONFIG.json',config)
    outcome=subprocess.run([PYTHON,'-B','evidence/export_audited.py'],cwd=task,capture_output=True,text=True)
    assert outcome.returncode==0,outcome.stderr
    export=read(task/'AUDITED_NEW_CANDIDATES.json');union=read(task/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
    assert export['new_question_count']==3 and export['type_distribution']=={'multi_select':2,'ordering':1}
    assert export['gold_distribution']=={'2':1,'3':1,'4':0,'5':0}
    assert export['historical_controls_excluded']==['synthetic-v84-2']
    assert export['source_root_count']==4 and export['independent_test'] is False
    assert union['candidate_count']==27 and union['type_distribution']=={'multi_select':22,'ordering':5}
    assert union['gold_distribution']=={'2':6,'3':6,'4':4,'5':6} and union['duplicates_removed']==[]
    assert all(union[k] is False for k in ['training_allowed','training_started','formal_training_admission','merged_into_baseline'])
    checks.append('Synthetic2newmulti/1ordering/1control exports3, preserves v83 cumulative24, yields27 and typedGold6/6/4/6')
    # Replaying an append-only export must fail instead of replacing outputs.
    outcome=subprocess.run([PYTHON,'-B','evidence/export_audited.py'],cwd=task,capture_output=True,text=True)
    assert outcome.returncode!=0 and 'FileExistsError' in outcome.stderr
    checks.append('Export refuses to overwrite existing fixture artifacts')
record={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS',
  'scope':'Pure AST and explicit synthetic mixed fixtures in a temporary directory; existing v83 cumulative inventory is read-only historical input. Source stand-ins are confined to the temporary fixture and do not validate real source semantics.',
  'checks':checks,'new_model_calls':0,'new_http_calls':0,'formal_audits_invoked':False,'production_root_files_modified':False,
  'files':[bind(HOME/'evidence'/name) for name in ['audit_generation_v84.py','audit_results_v84.py','audit_union_v84.py','export_audited.py','test_audit_bindings_v84.py','verify_helpers_offline_v84.py']]}
with (HOME/'evidence/HELPERS_OFFLINE_VALIDATION.json').open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
print('PASS',len(checks),checks)
