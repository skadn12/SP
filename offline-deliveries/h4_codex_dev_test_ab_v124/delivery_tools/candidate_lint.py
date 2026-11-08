"""Read-only native schema lint. Does not save, unlock or author judgments."""
from pathlib import Path
import argparse, json, sys
B=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(B/'tools'))
import review_v4 as review

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',required=True);p.add_argument('--phase',choices=['scene','asset','public'],required=True)
 p.add_argument('--executor',required=True);p.add_argument('--judgment',type=Path,required=True)
 p.add_argument('--scene')
 a=p.parse_args()
 result={'schema_version':'v124_readonly_candidate_schema_lint_v1','root_id':a.root,'phase':a.phase,'scene_ref':a.scene,'executor_id':a.executor,'writes_performed':False,'semantic_quality_certified':False,'unlocks_first_candidate':False}
 try:
  data,directory,context=review.checked_review_context(B,a.root,a.phase,a.executor,a.scene)
  raw=review.read(a.judgment)
  first=directory/'FIRST_SCHEMA_CANDIDATE.json'
  if first.exists():
   original=review.verify_binding(review.read(first)['candidate_binding'],data['unit'],'FIRST_SCHEMA_CANDIDATE')
   review.allowed_candidate_identifier_delta(review.read(original),raw)
  review.validate_attestation(raw,data)
  review.validate_scene(raw,data) if a.phase=='scene' else review.validate_qa(raw,data)
  result.update(status='NATIVE_SCHEMA_VALIDATION_PASS',pass_schema=True,payload_binding=review.binding(a.judgment))
  code=0
 except Exception as e:
  result.update(status='NATIVE_SCHEMA_VALIDATION_ERROR',pass_schema=False,error=type(e).__name__+': '+str(e));code=2
 print(json.dumps(result,ensure_ascii=False));return code
if __name__=='__main__':raise SystemExit(main())
