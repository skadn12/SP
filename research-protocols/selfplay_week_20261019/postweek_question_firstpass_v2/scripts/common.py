"""Scoped QFP2 authoring mechanics and immutable IO."""
from pathlib import Path
import hashlib,json,datetime,importlib.util
BASE=Path(__file__).resolve().parents[1]
PARENT=BASE.parent/'postweek_feedback_control_4root_v1'
REPO=BASE.parents[2]
SLOTS={'MC2':'MC_ENTITY_BINDING','MC3':'MC_STATE_UPDATE_CONJUNCTION','MC4':'MC_CONCLUSION_EVIDENCE','MC5':'MC_JOINT_CONSTRAINT','MC6':'MC_RECORD_AUDIT_CONTROL','ORDER1':'ORDER_EXPLICIT_CLOCK_VALUE','ORDER2':'ORDER_ENQUIRY_PROGRESSION'}
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();d=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(d).hexdigest(),'bytes':len(d)}
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
spec=importlib.util.spec_from_file_location('qfp2_public_gate',REPO/'scripts/current_method/question_presentation_gate_v2.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def mechanical(path):
 q=read(path);src=read(q['source_binding']['path']);assert bind(q['source_binding']['path'])==q['source_binding']
 turns={t['turn_ref']:t for t in src['turns']};slot=q['slot_id'];assert q['task_operation']==SLOTS[slot]
 assert q['source_read_scope']['all_turns_read'] and q['source_read_scope']['turn_count']==len(turns)
 texts=q['options'] if slot.startswith('MC') else q['items'];keys=list(texts)
 assert set(keys)==set('ABCDEF') if slot.startswith('MC') else set(keys)=={'I1','I2','I3','I4'}
 assert len(q['gold'])==(int(slot[2:]) if slot.startswith('MC') else 4) and len(set(q['gold']))==len(q['gold']) and set(q['gold'])<=set(keys)
 assert len(q['display_order'])==len(keys) and set(q['display_order'])==set(keys)
 if slot.startswith('ORDER'):assert q['display_order']!=keys and q['display_order']!=q['gold']
 assert set(q['evidence'])==set(keys)==set(q['private_option_notes'])
 assert all(isinstance(t,str) and t.strip() for t in texts.values()) and len(set(texts.values()))==len(keys)
 for k in keys:
  assert q['evidence'][k] and q['private_option_notes'][k]
  for e in q['evidence'][k]:assert e['turn_ref'] in turns and e['quote'] and e['quote'] in turns[e['turn_ref']]['text'],(k,e)
 gate.require_public_selection(q['question'],gate.public_operation(q['task_operation']),q['question_type'])
 return {'candidate_id':q['candidate_id'],'status':'PASS_MECHANICAL_NOT_SEMANTIC','binding':bind(path)}
if __name__=='__main__':
 import sys
 print(json.dumps(mechanical(sys.argv[1]),ensure_ascii=False))
