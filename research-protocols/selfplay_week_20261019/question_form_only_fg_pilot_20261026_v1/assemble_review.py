#!/usr/bin/env python3
"""Assemble immutable drafts, mechanical audits, and blind semantic review packs."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def binding(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def fingerprint(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def gate():
 p=Path('/root/autodl-tmp/gxy-selfplay/scripts/current_method/production_question_gate.py')
 sys.path.insert(0,str(p.parent))
 s=importlib.util.spec_from_file_location('form_only_review_gate',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def assemble():
 for b in read(HERE/'AUTHORING_INPUT_FREEZE.json')['bindings']+read(HERE/'LEGACY_INPUT_FREEZE.json')['bindings']:
  assert binding(b['path'])==b,b['path']
 candidates=[];inputs=[]
 for arm in ['F','G']:
  p=HERE/f'question_authoring/{arm}/CANDIDATES_DRAFT.json'
  rows=read(p);assert isinstance(rows,list)
  plans={r['candidate_id']:r for r in read(HERE/f'question_authoring/{arm}/TASK_PLANS_FIXED.json')}
  assert len(rows)==20 and set(r['candidate_id'] for r in rows)==set(plans)
  for c in rows:
   assert c['arm']==arm and c['slot_id'] in ['MC2','MC3','MC4','MC5','MC6']
   assert len(c['gold'])==int(c['slot_id'][2:]) and c['question_type']=='multi_select'
   assert c['root_id']==plans[c['candidate_id']]['root_id']
   assert list(c['options'])==list('ABCDEF')
   assert c['author_id'] and c['form_version']=='question_form_only_v1'
   assert c['preserved_plan_candidate_id']==c['candidate_id'] and c['form_task_description'].strip()
   w=read(HERE/f'question_authoring/{arm}/SOURCE_WARNINGS_{c["root_id"].split("-")[-1]}.json')
   assert c['source_status']==w['source_status'] and c['warnings']==w['warnings']
  candidates.extend(rows);inputs.append(binding(p))
 candidates.sort(key=lambda c:(c['root_id'],c['slot_id'],c['arm']))
 save(HERE/'CANDIDATES_INITIAL_PRIVATE.json',candidates)
 save(HERE/'DRAFT_FREEZE.json',{'bindings':inputs+[binding(HERE/'CANDIDATES_INITIAL_PRIVATE.json')],'candidate_count':40,'before_solver':True})
 m=gate();audits=[m.audit_candidate(c,HERE/'SOURCE_SELECTION_FINAL.json') for c in candidates]
 save(HERE/'MECHANICAL_INITIAL.json',{'records':audits,'passed':sum(a['mechanical_pass'] for a in audits),'total':len(audits)})
 mapping={};groups={1:[],2:[]}
 for c in candidates:
  bid='Q'+hashlib.sha256(('question_form_only_v1:'+c['candidate_id']).encode()).hexdigest()[:12]
  root=c['root_id'].split('-')[-1];group=1 if root in ['218519','222937'] else 2
  mapping[bid]={'candidate_id':c['candidate_id'],'candidate_sha256':fingerprint(c),'author_id':c['author_id'],'root_id':c['root_id']}
  groups[group].append({'review_id':bid,'root_id':c['root_id'],'question_type':c['question_type'],'task_operation':c['task_operation'],'question':c['question'],'options':c['options'],'source_binding':c['source_binding']})
 save(HERE/'review/BLIND_MAP_PRIVATE.json',mapping)
 for group,rows in groups.items():
  random.Random(2026102600+group).shuffle(rows)
  warnings={}
  for root in sorted({r['root_id'] for r in rows}):
   tail=root.split('-')[-1]
   fw=read(HERE/f'question_authoring/F/SOURCE_WARNINGS_{tail}.json')
   gw=read(HERE/f'question_authoring/G/SOURCE_WARNINGS_{tail}.json')
   assert fw==gw,'arm warning mismatch'
   warnings[root]=fw
  save(HERE/f'review/group{group}/BLIND_QUESTIONS.json',{'questions':rows,'source_warnings':warnings,'review_policy':'Reconstruct every option from the complete source before viewing any author Gold or analysis. Do not infer required answer count. Do not inspect author folders, feedback, old questions, Solver outputs or private mapping.'})
 print(json.dumps({'candidates':40,'mechanical_pass':sum(a['mechanical_pass'] for a in audits),'groups':[len(v) for v in groups.values()]}))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['assemble']);a=p.parse_args();globals()[a.action]()
if __name__=='__main__':main()
