import json,pathlib,sys,hashlib,datetime
sys.path.insert(0,'/root/autodl-tmp/gxy-selfplay/scripts/current_method')
from production_question_gate import audit_candidate
BASE=pathlib.Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/natural_question_forms_fg_pilot_20261026_v1')
OUT=BASE/'question_authoring/G'
p=OUT/'CANDIDATES_DRAFT.json';candidates=json.loads(p.read_text());results=[]
for c in candidates:
 try:results.append(audit_candidate(c,BASE/'SOURCE_SELECTION_FINAL.json'))
 except Exception as e:results.append({'candidate_id':c.get('candidate_id'),'exception':repr(e),'mechanical_pass':False})
log={'audit_kind':'mechanical_only_not_independent_content_review','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_file':str(p),'candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selection_path':str(BASE/'SOURCE_SELECTION_FINAL.json'),'results':results}
(OUT/'MECHANICAL_AUDIT_01.json').write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
summary={'count':len(results),'pass':sum(r.get('mechanical_pass',False) for r in results),'failures':[r for r in results if not r.get('mechanical_pass')],'presentation_lint':[{'candidate_id':r.get('candidate_id'),'lint':r.get('presentation_lint')} for r in results]}
(OUT/'MECHANICAL_SUMMARY_01.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
