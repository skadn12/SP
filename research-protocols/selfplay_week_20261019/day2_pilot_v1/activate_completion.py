"""Point active status at independently audited D2 completion; leave old seals intact."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parent
def read(p):return json.loads(p.read_text())
def bind(p):
    p=p.resolve();b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def write(p,obj,exclusive=True):
    with p.open('x' if exclusive else 'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
summary=read(P/'SUMMARY.json')
completion_audit=read(P/'audit/COMPLETION_INDEPENDENT_AUDIT.json')
score_audit=read(P/'audit/INDEPENDENT_SCORE_RECOMPUTATION.json')
assert completion_audit['status']=='PASS' and score_audit['status']=='PASS'
go=summary['day2_all_goals_complete'];at=datetime.now(timezone.utc).isoformat()
state=read(W/'CURRENT_STATUS.json');latest=read(W/'DAY2_LATEST.json')
write(P/'status_before/CURRENT_STATUS_PRE_ACTIVATION.json',state)
write(P/'status_before/DAY2_LATEST_PRE_ACTIVATION.json',latest)
state.update(updated_at_utc=at,status='DAY2_COMPLETE_GO_NARROW_ORDERING' if go else 'DAY2_EXECUTED_NO_GO_FOR_FEEDBACK_B',
 day2_all_goals_complete=go,next_action=summary['next_action'])
state['day2'].update(experiment_records_closed=True,feedback_ready=go,
 feedback_go_status=summary['status'],new_question_pilot_report=bind(P/'REPORT_ZH.md'),
 new_question_pilot_summary=bind(P/'SUMMARY.json'),
 remaining_gates=[] if go else [k for k,v in summary['gates'].items() if not v],
 completed_gates=['D2-01','D2-02','D2-03','D2-04','D2-05','D2-06'],
 formal_MC_targeted_quota_ready=False,formal_70_percent_B_ready=False,
 operational_scope=summary['scope'],qualified_new_ordering_lineage_roots=summary['qualified_new_ordering_lineage_roots'],
 fact_memory_adopted=False,fact_memory_cost_status='NO_COST_CONCLUSION',main_context='full_history',
 actual_weakness_card_consumption=True,
 independent_completion_audit=bind(P/'audit/COMPLETION_INDEPENDENT_AUDIT.json'))
state['diversity']['day2_exploratory_pilot']={'questions':24,'roots':6,'new_structural_diversity_goal_complete':False,'trained':False}
state['daily_planning']['day2']='GO_NARROW_ORDERING_FEEDBACK_PROCESS' if go else 'NO_GO_WITH_RECORDS'
state['execution_progress_v4']={'scope':'D2-01 to D2-06 executed and independently audited',
 'new_questions':summary['generated_questions'],'new_solver_calls':summary['new_solver_calls'],
 'new_training_runs':0,'old_sealed_experiments_modified':False,'pilot':bind(P/'SUMMARY.json')}
state['files']['day2_pilot_report']=bind(P/'REPORT_ZH.md')
state['files']['day2_pilot_summary']=bind(P/'SUMMARY.json')
state['before_execution_freeze_remaining']=[
 'Day3 actual structural design and production slots, updated exposure ledger and common evaluation boundaries',
 'Prospective formal B scope/quota revision consistent with narrow qualified feedback coverage',
 'Day4 actual matched A/B training pools, old-data fraction q and symmetric training resource freeze',
 'Future training and evaluation request/model/checkpoint freezes']
newlatest={'schema_version':'day2_latest_v4','updated_at_utc':at,'status':state['status'],
 'day2_all_goals_complete':go,'feedback_go_status':summary['status'],'completed_gates':state['day2']['completed_gates'],
 'remaining_gates':state['day2']['remaining_gates'],'pilot_summary':bind(P/'SUMMARY.json'),
 'pilot_report':bind(P/'REPORT_ZH.md'),'independent_completion_audit':bind(P/'audit/COMPLETION_INDEPENDENT_AUDIT.json'),
 'independent_score_recomputation':bind(P/'audit/INDEPENDENT_SCORE_RECOMPUTATION.json'),
 'prior_active_snapshot':bind(P/'status_before/DAY2_LATEST_PRE_ACTIVATION.json'),
 'formal_70_percent_B_ready':False,'formal_MC_targeted_quota_ready':False,
 'new_questions':summary['generated_questions'],'new_solver_calls':summary['new_solver_calls'],'new_training_runs':0,
 'next_action':summary['next_action']}
write(W/'CURRENT_STATUS.json',state,False);write(W/'DAY2_LATEST.json',newlatest,False)
write(P/'STATUS_POINTER_TRANSITION.json',{
 'created_at_utc':at,'reason':'Active status pointers advance after independently audited execution; archived experiment inputs remain exact original bytes.',
 'mutable_pointer_resolutions':[
  {'original_path':str(W/'CURRENT_STATUS.json'),'archived_input':bind(P/'status_before/CURRENT_STATUS_PRE_ACTIVATION.json'),
   'new_active_pointer':bind(W/'CURRENT_STATUS.json')},
  {'original_path':str(W/'DAY2_LATEST.json'),'archived_input':bind(P/'status_before/DAY2_LATEST_PRE_ACTIVATION.json'),
   'new_active_pointer':bind(W/'DAY2_LATEST.json')}],
 'original_freeze_not_modified':True})
files=[bind(p) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='FINAL_MANIFEST.json']
write(P/'FINAL_MANIFEST.json',{'status':'SEALED_D2_PILOT','created_at_utc':at,'file_count':len(files),'files':files,
 'active_status_bindings':[bind(W/'CURRENT_STATUS.json'),bind(W/'DAY2_LATEST.json')],
 'no_base_or_model_weights_copied':True,'no_new_training':True})
print(json.dumps({'status':state['status'],'day2_all_goals_complete':go,'files':len(files),'report':str(P/'REPORT_ZH.md')}))
