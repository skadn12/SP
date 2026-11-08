from pathlib import Path
import json,hashlib,itertools,datetime,collections
D=Path(__file__).resolve().parent; R=D/'runtime/eval_v1'
def read(p):return json.loads(p.read_text())
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 errors=[];checks=0
 def check(ok,kind,cid='batch'):
  nonlocal checks
  checks+=1
  if not ok:errors.append(dict(candidate_id=cid,kind=kind))
 m=read(R/'manifest.json');scores=read(R/'SCORES.json');maps=read(R/'DISPLAY_MAPPINGS.json');qs={q['candidate_id']:q for q in read(D/'exports/PRIVATE_QUESTIONS.json')['questions']};public=read(R/'PUBLIC_QUESTIONS.json')['questions']
 check(len(qs)==len(public)==11,'ELEVEN_CONTENT_ACCEPTED');check(scores['direct_calls']==66 and scores['diagnostic_calls']==11,'CALL_COUNTS');check(scores['direct_valid']==66,'DIRECT_ALL_VALID')
 for b in ['direct','evidence']:
  f=read(R/b/'RAW_RESPONSE_FREEZE.json')
  for x in f['files']+f['execution_ledger_bindings']+[f['manifest_binding']]: check(Path(x['path']).is_file() and bind(Path(x['path']))['sha256']==x['sha256'],'RAW_FREEZE_UNCHANGED')
  check(len(f['files'])==66 if b=='direct' else len(f['files'])==11,'FIXED_BRANCH_COUNT')
 direct={s['request_id']:s for s in scores['direct_requests']};diags={s['request_id']:s for s in scores['diagnostics']};expected_exact=collections.Counter()
 for row in m['requests']:
  rid=row['request_id'];cid=row['candidate_id'];q=qs[cid];v=maps['requests'][rid];raw=read(R/row['branch']/'responses'/(rid+'.json'));canon=maps['public_to_canonical'][cid];rev={x:k for k,x in canon.items()}
  check(all(raw[k]==row[k] for k in ['request_id','candidate_id','root_id','variant_id','slot','seed','prompt_sha256']),'RAW_IDENTITY',cid)
  check(raw['status']=='RETURNED' and not raw['output_validation_errors'],'ACTUAL_RETURNED',cid);check(raw['metadata']['finish_reason']!='length','NO_TRUNCATION',cid)
  if row['branch']=='direct':
   text=raw['raw_output'];check(text.startswith('<box>') and text.endswith('</box>'),'NATIVE_BOX_FORMAT',cid);labels=[x.strip() for x in text[5:-6].split(',')];pred=[rev[v['display_to_content'][x]] for x in labels];gold=q['gold'];exact=(set(pred)==set(gold) if q['question_type']=='multi_select' else pred==gold)
   s=direct[rid];check((set(pred)==set(s['canonical_prediction']) if q['question_type']=='multi_select' else pred==s['canonical_prediction']) and exact==s['exact'],'INDEPENDENT_DIRECT_RECOMPUTE',cid);expected_exact[q['question_type']]+=exact
   if q['question_type']=='multi_select':
    a=set(pred);b=set(gold);check(s['metrics']['fp']==len(a-b) and s['metrics']['fn']==len(b-a) and s['metrics']['jaccard']==len(a&b)/len(a|b),'INDEPENDENT_MC_METRICS',cid)
   else:
    ranks={x:i for i,x in enumerate(pred)};pairs=list(itertools.combinations(gold,2));n=sum(ranks[a]<ranks[b] for a,b in pairs);check(s['metrics']['correct_pairs']==n and s['metrics']['total_pairs']==len(pairs),'INDEPENDENT_PAIR_METRICS',cid)
  else:
   import jsonschema
   s=diags[rid];valid=True
   try:
    p=json.loads(raw['raw_output'].strip());jsonschema.validate(p,row['json_schema']);labs=list(row['display'].get('options') or row['display']['items']);assert len(p['item_evidence'])==len(labs) and {x['label'] for x in p['item_evidence']}==set(labs)
   except Exception: valid=False
   check(valid==s['format_valid'],'INDEPENDENT_DIAGNOSTIC_VALIDITY',cid)
   if valid:
    pred=[rev[v['display_to_content'][x]] for x in p['final_answer']];exact=set(pred)==set(q['gold']) if q['question_type']=='multi_select' else pred==q['gold'];check(pred==s['canonical_prediction'] and exact==s['exact'],'INDEPENDENT_DIAGNOSTIC_LABEL_DOMAIN',cid)
   else:check(s['canonical_prediction'] is None and not s['exact'],'INVALID_NOT_RESCUED',cid)
 for kind,n in expected_exact.items():check(n==sum(s['exact'] for s in scores['direct_requests'] if s['question_type']==kind),'TYPE_EXACT_TOTAL',kind)
 check(scores['direct_exact']==sum(expected_exact.values()),'DIRECT_TOTAL_EXACT')
 old=[]
 for sub in ['day3_targeted_4root_v1','day3_diagnostic_handoff_v1']:
  f=read(D.parent/sub/'FINAL_MANIFEST.json');bad=[]
  for x in f['files']:
   ok=Path(x['path']).is_file() and bind(Path(x['path']))['sha256']==x['sha256'];check(ok,'OLD_SEALED_UNCHANGED');
   if not ok:bad.append(x['path'])
  old.append(dict(directory=sub,files=len(f['files']),errors=bad))
 out=dict(schema_version='feedback_realization_postrun_audit_v1',status='PASS' if not errors else 'FAIL',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,errors=errors,independent_label_metric_recomputation=True,author='root agent; runtime adapter authored by another role',technical_blinding=False,semantic_review_established=False,old_sealed=old,audited_files=[bind(R/'SCORES.json'),bind(R/'manifest.json')])
 with (D/'review/POSTRUN_RECOMPUTATION_VERIFIED.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
