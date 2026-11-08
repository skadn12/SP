#!/usr/bin/env python3
"""Compare immutable first blind answers to stable semantic Gold; no answer repair."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (HERE/name).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2)
def main():
    items=read(HERE/'ITEMS_PRIVATE.json')
    mech={r['candidate_id']:r for r in read(HERE/'MECHANICAL_REVIEW.json')['items']}
    results=[];body=['# Frozen first-draft questions, Gold and blind quality review','',
      'Challenger: Codex. Reviewer: fresh Codex same-model blind context per question. Research use only; no formal training-source approval is created.','']
    required=['review_id','reviewer_id','reviewer_mode','full_source_read','source_sha256','source_turn_count','independent_answer','per_option','answer_reason','answer_unique','hard_issues','soft_issues','quality_checks','cognitive_task','difficulty_auxiliary']
    for c in items:
        p=HERE/'blind_results'/(c['review_id']+'.json')
        r=read(p);errors=[]
        if not isinstance(r,dict): raise ValueError('Blind object required: '+str(p))
        for key in required:
            if key not in r: errors.append('missing_field:'+key)
        if r.get('review_id')!=c['review_id']: errors.append('review_id_mismatch')
        if r.get('reviewer_mode')!='same_model_blind': errors.append('review_mode_mismatch')
        if r.get('full_source_read') is not True: errors.append('full_source_not_read')
        if r.get('source_sha256')!=c['source_sha256']: errors.append('source_hash_mismatch')
        if type(r.get('source_turn_count')) is not int or r.get('source_turn_count')!=c['source_turn_count']: errors.append('source_count_mismatch')
        for key in ['hard_issues','soft_issues','per_option','independent_answer']:
            if not isinstance(r.get(key),list): errors.append('field_type:'+key)
        if type(r.get('answer_unique')) is not bool: errors.append('field_type:answer_unique')
        if not isinstance(r.get('quality_checks'),dict): errors.append('field_type:quality_checks')
        labels=list(c['label_to_option_id'])
        answer=r.get('independent_answer',[])
        if not isinstance(answer,list) or not all(isinstance(x,str) for x in answer):
            errors.append('invalid_answer_type');answer=[]
        if len(answer)!=len(set(answer)) or not set(answer)<=set(labels) or not answer: errors.append('invalid_answer_labels')
        if c['question_type']=='ordering' and set(answer)!=set(labels): errors.append('incomplete_order')
        rows=r.get('per_option',[])
        if not isinstance(rows,list) or not all(isinstance(x,dict) for x in rows):
            errors.append('invalid_per_option');rows=[]
        seen=[x.get('label') for x in rows]
        if len(seen)!=len(labels) or len(set(seen))!=len(seen) or set(seen)!=set(labels): errors.append('per_option_coverage')
        source_refs={t['turn_ref'] for t in read(c['source_path'])['turns']}
        for row in rows:
            if row.get('judgment') not in ['supported','contradicted','not_supported','ambiguous']: errors.append('invalid_judgment:'+str(row.get('label')))
            if not isinstance(row.get('reason'),str) or not row['reason'].strip(): errors.append('missing_option_reason:'+str(row.get('label')))
            refs=row.get('turn_refs')
            if not isinstance(refs,list) or not all(isinstance(ref,str) and ref in source_refs for ref in refs): errors.append('invalid_option_refs:'+str(row.get('label')))
        if c['question_type']=='multi_select' and {row.get('label') for row in rows if row.get('judgment')=='supported'}!=set(answer): errors.append('blind_per_option_final_inconsistency')
        if not isinstance(r.get('reviewer_id'),str) or not r.get('reviewer_id'): errors.append('invalid_reviewer_id')
        if not isinstance(r.get('answer_reason'),str) or not r.get('answer_reason','').strip(): errors.append('invalid_answer_reason')
        checks=r.get('quality_checks',{})
        for key in ['source_binding','question_clarity','evidence_sufficiency','option_quality','naturalness','shortcut_assessment','joint_evidence_character']:
            if key not in checks: errors.append('missing_quality_check:'+key)
        semantic=[c['label_to_option_id'][x] for x in answer if x in c['label_to_option_id']]
        agree=(set(semantic)==set(c['gold_option_ids'])) if c['question_type']=='multi_select' else semantic==c['gold_option_ids']
        hard=r.get('hard_issues') if isinstance(r.get('hard_issues'),list) else [{'issue':'invalid_hard_issues_field'}]
        passed=bool(mech[c['candidate_id']]['mechanical_pass'] and not errors and r.get('answer_unique') is True and agree and not hard)
        reasons=[]
        if not mech[c['candidate_id']]['mechanical_pass']: reasons.extend(mech[c['candidate_id']]['errors'])
        reasons.extend(errors)
        if r.get('answer_unique') is not True: reasons.append('blind_answer_not_unique')
        if not agree: reasons.append('blind_gold_disagreement')
        reasons.extend('blind_hard_issue:'+str(x.get('issue',x)) for x in hard)
        result={'candidate_id':c['candidate_id'],'review_id':c['review_id'],'arm':c['arm'],'root_id':c['root_id'],
          'content_pass':passed,'status':'PASS' if passed else 'HOLD','hold_reasons':reasons,
          'mechanical_pass':mech[c['candidate_id']]['mechanical_pass'],'review_protocol_valid':not errors,'review_protocol_errors':errors,
          'blind_gold_agreement':agree,'blind_answer_labels':answer,'blind_answer_option_ids':semantic,'gold_labels':c['gold_labels'],
          'blind_answer_unique':r.get('answer_unique'),'blind_hard_issues':hard,'blind_soft_issues':r.get('soft_issues'),
          'review_path':str(p),'review_sha256':sha(p),'reviewer_mode':'same_model_blind',
          'source_sha256':c['source_sha256'],'formal_training_approval_created':False,'solver_eligible':passed}
        results.append(result)
        body += [f"## {c['review_id']} — {c['candidate_id']}",'',f"Root {c['root_id']}; arm {c['arm']}; type {c['question_type']}; author difficulty {c['difficulty_auxiliary']}",'',c['question'],'']
        body += [o['id']+'. '+o['text'] for o in c['options']]
        body += ['', 'Gold: '+', '.join(c['gold_labels']), 'Semantic Gold: '+', '.join(c['gold_option_ids']),
          'Blind first answer: '+', '.join(answer), 'Quality: '+result['status']+'; reasons: '+('; '.join(reasons) if reasons else 'none'),
          'Task: '+str(c['cognitive_task']),'Evidence/relations: '+json.dumps(c['relation_plan'],ensure_ascii=False),
          'Evidence audit: '+json.dumps(c['evidence_audit'],ensure_ascii=False),'']
        for ev in c['evidence']: body.append(f"- {ev['turn_ref']} ({ev.get('origin','see source')}): {ev['quote']}")
        body += ['', 'Blind checks: '+json.dumps(r.get('quality_checks'),ensure_ascii=False),
          'Blind soft issues: '+json.dumps(r.get('soft_issues'),ensure_ascii=False),'']
    write('QUALITY_FINAL.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'policy':'first frozen same-model blind answer; hard quality conditions only; no answer repair or repeat review; formal training admission separate','items':results})
    with (HERE/'QUESTIONS_GOLD_REVIEW.md').open('x') as f:f.write('\n'.join(body)+'\n')
    print(json.dumps({'candidates':len(results),'content_pass':sum(r['content_pass'] for r in results),'holds':[{'id':r['candidate_id'],'reasons':r['hold_reasons']} for r in results if not r['content_pass']]}))
if __name__=='__main__': main()
