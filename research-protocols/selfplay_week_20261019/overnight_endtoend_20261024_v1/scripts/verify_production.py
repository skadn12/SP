#!/usr/bin/env python3
"""Validate exact frozen OET1 input bytes; no dialogue/question authoring."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):
    p=Path(p).resolve(); raw=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def check(binding):
    assert bind(binding['path'])==binding,('CHANGED',binding['path'])
receipt=read(BASE/'inputs/PRODUCTION_PREPARATION_RECEIPT.json')
for binding in receipt['artifacts']: check(binding)
proposal=read(BASE/'ROOT_SELECTION_PROPOSAL.json')
reservation=read(BASE/'inputs/SOURCE_RESERVATIONS.json')
audit=read(BASE/'inputs/RESERVATION_AUDIT.json')
assert len(reservation['reservations'])==24 and len(set(reservation['source_ids']))==24
pool=Path(reservation['pool_binding']['path']).read_bytes(); check(reservation['pool_binding'])
for row in reservation['reservations']:
    b=row['native_record_binding']; check(b); span=row['pool_record_byte_span']
    assert pool[span['start_inclusive']:span['end_exclusive']]==Path(b['path']).read_bytes()
    assert row['authorized_consumers']==[row['root_id']+':'+arm for arm in ['GENERIC','TARGETED']]
    assert row['release_on_failure'] is False and row['release_on_nonexecution'] is False
for binding in audit['inventory_evidence']: check({k:binding[k] for k in ['path','sha256','bytes']})
for row in proposal['roots']:
    root=row['root_id']; a=read(BASE/'sources'/root/'GENERIC/SOURCE_AUTHOR_INPUT.json'); b=read(BASE/'sources'/root/'TARGETED/SOURCE_AUTHOR_INPUT.json')
    assert a['full_initial_source']==b['full_initial_source'] and a['native_events']==b['native_events']
    assert a['turns_per_scene']==b['turns_per_scene']==14 and len(a['scene_sequence'])==6
    assert a['speakers']==b['speakers'] and a['common_material_requirements']==b['common_material_requirements']
    aa={k:v for k,v in a.items() if k not in ['condition','material_goals','feedback_scope']}
    bb={k:v for k,v in b.items() if k not in ['condition','material_goals','feedback_scope']}
    assert aa==bb
assert all(group['clock_union_count']>=4 for group in audit['root_groups'])
result={'schema_version':'oet1_production_preparation_verification_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'status':'PASS_EXACT_FROZEN_PRODUCTION_INPUTS','preparation_receipt':bind(BASE/'inputs/PRODUCTION_PREPARATION_RECEIPT.json'),
    'counts':receipt['counts'],'artifact_bindings_checked':len(receipt['artifacts']),
    'historical_bindings_checked':len(audit['inventory_evidence']),'all_native_event_raw_pool_spans_exact':True,
    'paired_packets_only_condition_goals_feedback_scope_differ':True,'all_root_native_clock_unions_ge4':True,
    'engineering_resume_fix':{'path':str(BASE/'scripts/prepare_production_v2.py'),
        'binding':bind(BASE/'scripts/prepare_production_v2.py'),
        'original_code_binding':bind(BASE/'scripts/prepare_production.py'),
        'reason':'v1 resume verifier parsed Python source artifacts as JSON; v2 checks raw hash before parsing only .json. Initial production completed successfully. Frozen bytes retained.'},
    'semantic_source_or_question_admission_claim':False,'checker_binding':bind(Path(__file__))}
out=BASE/'inputs/PRODUCTION_PREPARATION_VERIFICATION.json'
if out.exists():
    previous=read(out); assert previous['preparation_receipt']==result['preparation_receipt']; print(json.dumps({'status':previous['status'],'binding':bind(out)},ensure_ascii=False))
else:
    with out.open('x',encoding='utf-8') as f: f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'binding':bind(out)},ensure_ascii=False))
