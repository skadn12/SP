from pathlib import Path
import json,hashlib,datetime,subprocess
R=Path(__file__).resolve().parent.parent;E=R/'execution'
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
review=read(R/'FINAL_CONTENT_REVIEW.COMPLETE.json')
assert review['status']=='FROZEN' and review['before_any_solver'] and review['solver_scores_used'] is False
assert not (E/'evaluation_sequence.LAUNCH.json').exists() and not (E/'evaluation_sequence.log').exists()
with (E/'evaluation_sequence.log').open('x') as stream:
 child=subprocess.Popen(['python','-B','-u',str(E/'run_evaluations.py')],cwd=R,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':child.pid,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runner':bind(E/'run_evaluations.py'),'review':bind(R/'FINAL_CONTENT_REVIEW.COMPLETE.json'),'usable_questions':review['usable_count'],'calls_per_model':18*review['usable_count'],'automatic_retry':False,'training_allowed':False}
with (E/'evaluation_sequence.LAUNCH.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(v))
