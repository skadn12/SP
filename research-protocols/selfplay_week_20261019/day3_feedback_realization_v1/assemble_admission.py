from pathlib import Path
import json,hashlib,collections,datetime
D=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 freeze=read(D/'GENERATION_FREEZE.json');struct=read(D/'review/AUTHOR_STRUCTURAL_VALIDATION.json');assert struct['status']=='PASS'
 authors=[D/'authors/part_a.json',D/'authors/part_b.json'];reviews=[D/'review/part_a_CONTENT_REVIEW.json',D/'review/part_b_CONTENT_REVIEW.json']
 qs={q['candidate_id']:q for p in authors for q in read(p)['questions']};rs={q['candidate_id']:q for p in reviews for q in read(p)['questions']}
 assert set(qs)==set(rs)
 for a,r in zip(authors,reviews):
  rv=read(r);assert rv['reviewed_author_binding']['sha256']==bind(a)['sha256'];assert rv['technical_blinding'] is False and rv['human_review'] is False
 admitted=[];ledger=[]
 keys=('candidate_id','root_id','question_type','question','options','items','source_path','source_sha256')
 for s in freeze['slots']:
  qid=s['candidate_id'];q=qs[qid];r=rs[qid];accept=r['disposition']=='ACCEPT'
  if accept: assert all(r[k] for k in ['source_gold_unique','feedback_target_realized','exact_quotes_valid']);admitted.append(q)
  ledger.append(dict(candidate_id=qid,root_id=q['root_id'],feedback_id=q['feedback_id'],content_disposition=r['disposition'],feedback_target_realized=r['feedback_target_realized'],novelty_class=r['novelty_class'],family_id=q['family_id'],training_admitted=False))
 public={'questions':[{k:q[k] for k in keys if k in q} for q in admitted]};private={'questions':[{**{k:q[k] for k in keys if k in q},'gold':q['gold'],'family_id':q['family_id'],'feedback_id':q['feedback_id']} for q in admitted]}
 pub=D/'exports/PUBLIC_QUESTIONS.json';priv=D/'exports/PRIVATE_QUESTIONS.json';save(pub,public);save(priv,private)
 for q in admitted:assert len(q['gold'])==len(set(q['gold']))
 save(D/'exports/CANDIDATE_LEDGER.json',dict(planned_slots=12,content_accepted=len(admitted),training_admitted=0,questions=ledger))
 save(D/'CONTENT_ADMISSION.json',dict(schema_version='feedback_realization_content_admission_v1',status='PASS',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace=freeze['namespace'],planned_slots=12,admitted_count=len(admitted),candidate_ids=[q['candidate_id'] for q in admitted],public_questions=bind(pub),private_questions=bind(priv),generation_freeze=bind(D/'GENERATION_FREEZE.json'),author_bindings=[bind(p) for p in authors],review_bindings=[bind(p) for p in reviews],structural_validation=bind(D/'review/AUTHOR_STRUCTURAL_VALIDATION.json'),admission_before_solver=True,solver_calls_at_admission=0,technical_blinding=False,human_review=False,training_admitted=False))
 lines=['# 本轮全部候选与答案','内容审核先于Solver；本轮为同源定向变体检验，不等于新增独立题家族。']
 for s in freeze['slots']:
  q=qs[s['candidate_id']];r=rs[q['candidate_id']];lines.extend(['','## '+q['candidate_id'],q['question'],''])
  for k,v in (q.get('options') or q['items']).items():lines.append(k+'. '+v)
  lines.extend(['','Gold: '+', '.join(q['gold']),'反馈: '+q['feedback_id'],'审核: '+r['disposition'],'新颖性: '+r['novelty_class'],'来源: '+q['source_path']])
 (D/'exports/QUESTIONS_WITH_GOLD_ZH.md').write_text(chr(10).join(lines))
 print(json.dumps(dict(status='PASS',planned_slots=12,content_accepted=len(admitted),gold_counts=dict(collections.Counter(len(q['gold']) for q in admitted if q['question_type']=='multi_select')),types=dict(collections.Counter(q['question_type'] for q in admitted)),training_admitted=0),ensure_ascii=False))
if __name__=='__main__':main()
