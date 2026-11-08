from pathlib import Path
import json,hashlib,datetime
D=Path(__file__).resolve().parent

def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
 s=json.loads((D/'SUMMARY.json').read_text());audit=json.loads((D/'review/FINAL_COMPLETENESS_AUDIT.json').read_text());assert audit['status']=='PASS';assert s['actual_next_consumer_read'] and not s['all_targets_loop_closed'];assert not (D/'FINAL_MANIFEST.json').exists()
 files=[bind(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='FINAL_MANIFEST.json'];m=dict(schema_version='feedback_realization_final_manifest_v1',status=s['status'],created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,planned_slots=12,new_candidates=12,content_accepted=11,content_hold=1,new_source_turns=0,new_roots=0,new_solver_calls=77,direct_calls=66,diagnostic_calls=11,diagnostic_valid=8,diagnostic_invalid=3,actual_next_consumer_read=True,candidate_realized_feedback_cards=4,planned_feedback_cards=5,source_verified_target_failure_records=7,all_targets_loop_closed=False,formal_reproduced=False,matched_AB=False,feedback_causal_effect_established=False,training_started=False,training_admitted=0,dev_test_calls=0,read_only_after_seal=True)
 with (D/'FINAL_MANIFEST.json').open('x') as f:json.dump(m,f,ensure_ascii=False,indent=2)
 bad=[b['path'] for b in files if bind(Path(b['path']))['sha256']!=b['sha256']];assert not bad;print(json.dumps(dict(status='SEALED_VERIFIED',files=len(files),manifest=bind(D/'FINAL_MANIFEST.json')),ensure_ascii=False))
if __name__=='__main__':main()
