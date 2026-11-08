#!/usr/bin/env python3
"""Build Stage B v7 from immutable v6 anonymous inputs and a completed history projection.

Preparation only: never invokes an author or selector. Fails closed unless the
history projection is complete and matches the exact plan-scoped row counts.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN=HERE.parent
V6=PLAN/'execution_20261107_v6'
DEST=PLAN/'execution_20261107_v7'
CONTRACTS=PLAN/'stage_b_contract_review_20261107_v1'
EXPECTED_SEALED=135
EXPECTED_OVERLAP=77
EXPECTED_KNOWN_OVERLAP=65
EXPECTED_UNKNOWN_OVERLAP=12
PARENT_HASHES={
    'coverage_index_top_level':'8d13b4331aa04881a3ed5325577c3c8f94bcfaccd932b3bc2e593e153a7d5d25',
    'repaired_966_inventory':'e2695370957bd438b7c9a68b83469f5daaad7973e5a3a74de7b35018c32064cc',
    'sealed135_operation_audit':'8281e2bacf2fc7dd1f76f5df4684b0bd906ef0c701f449371d1dc3f9eb887b05',
}

class BuildError(RuntimeError): pass

def sha(raw: bytes)->str: return hashlib.sha256(raw).hexdigest()
def readj(p: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise BuildError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def reject_constant(value): raise BuildError(f'invalid JSON constant: {value}')
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=reject_constant)
def writej(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    p.write_bytes(raw); return sha(raw)
def rel(p:Path, base:Path): return p.resolve().relative_to(base.resolve()).as_posix()
def hashfile(p:Path): return sha(p.read_bytes())
def verify_hex(x): return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)

def validate_history(h):
    if h.get('schema_version')!='stage_b_history_projection_v1':
        raise BuildError('history schema_version must be stage_b_history_projection_v1')
    if h.get('status')!='COMPLETE_FOR_SELECTOR':
        raise BuildError('history status must be COMPLETE_FOR_SELECTOR')
    coverage=h.get('coverage_set'); overlap=h.get('overlap_set')
    if not isinstance(coverage,list) or not isinstance(overlap,list):
        raise BuildError('history requires coverage_set and overlap_set arrays')
    if len(coverage)!=EXPECTED_SEALED:
        raise BuildError(f'coverage_set must contain the 135 sealed reviewed items; got {len(coverage)}')
    if len(overlap)!=EXPECTED_OVERLAP:
        raise BuildError(f'overlap_set must contain 77 unique same-root additions; got {len(overlap)}')
    dims=('event','natural_information_need','query_scope','answer_operation','evidence_family')
    def complete(v):
        return isinstance(v,str) and bool(v.strip()) and not re.search(r'\b(?:UNKNOWN|PENDING|PROVISIONAL)\b',v,re.I)
    for i,r in enumerate(coverage):
        if not isinstance(r,dict): raise BuildError(f'coverage row {i} not object')
        if r.get('eligibility')!='ELIGIBLE_VERIFIED' or r.get('review_status')!='INDEPENDENTLY_REVIEWED':
            raise BuildError(f'coverage row {i} is not verified eligible')
        if any(not complete(r.get(k)) for k in dims):
            raise BuildError(f'coverage row {i} lacks complete tuple')
        if not isinstance(r.get('history_item_id'),str): raise BuildError(f'coverage row {i} missing opaque id')
    known=unknown=0
    for i,r in enumerate(overlap):
        if not isinstance(r,dict): raise BuildError(f'overlap row {i} not object')
        if not isinstance(r.get('history_item_id'),str): raise BuildError(f'overlap row {i} missing opaque id')
        vals=[r.get(k) for k in dims]
        # Fully annotated rows must have all five dimensions; explicit unknown
        # rows must mark every unresolved field UNKNOWN and carry a reason.
        if all(complete(v) for v in vals) and r.get('review_status')=='INDEPENDENTLY_ANNOTATED':
            known+=1
            if r.get('eligibility') not in ('INELIGIBLE','UNKNOWN'):
                raise BuildError(f'overlap row {i} must not count as coverage without a training eligibility audit')
        else:
            unknown+=1
            if r.get('review_status')!='UNKNOWN_RETAINED' or not isinstance(r.get('unknown_reason'),str) or not r['unknown_reason'].strip():
                raise BuildError(f'overlap row {i} has incomplete tuple but no explicit UNKNOWN_RETAINED reason')
            # Unknown rows must be opaque to the selector.  Do not retain
            # partially-known fields: a selector could otherwise compare one
            # visible field and accidentally turn missing evidence into a
            # novelty/duplicate signal.
            if vals != ['UNKNOWN'] * len(dims):
                raise BuildError(f'overlap row {i} UNKNOWN_RETAINED requires every tuple field to equal UNKNOWN')
            if r['unknown_reason'] != 'Semantic record unavailable; novelty comparison unresolved.':
                raise BuildError(f'overlap row {i} unknown_reason must use the generic selector-safe text')
            if r.get('eligibility') != 'UNKNOWN':
                raise BuildError(f'overlap row {i} UNKNOWN_RETAINED must have UNKNOWN eligibility')
    if known!=EXPECTED_KNOWN_OVERLAP or unknown!=EXPECTED_UNKNOWN_OVERLAP:
        raise BuildError(f'overlap rows must be 65 annotated + 12 unknown; got {known}+{unknown}')
    allids=[r['history_item_id'] for r in coverage+overlap]
    if len(set(allids))!=len(allids): raise BuildError('history ids are not unique after candidate dedupe')
    if any(not re.fullmatch(r'H[0-9]{4,}', item) for item in allids):
        raise BuildError('history ids must be opaque H-prefixed numeric identifiers')
    # A question, answer, or private scene reference must never be supplied to
    # a selector. These checks are necessary, not a semantic privacy review.
    forbidden = re.compile(r'cc-train-episode-\d+|R\d{2}:[AB]:S\d{2}(?::T\d{3})?|\?|？', re.I)
    for row in coverage+overlap:
        if any(forbidden.search(row[k]) for k in dims):
            raise BuildError('history tuple contains a question or private provenance token')
    bindings=h.get('source_bindings')
    if not isinstance(bindings,list) or not bindings: raise BuildError('history source_bindings inventory/hashes required')
    for b in bindings:
        if not isinstance(b,dict) or not verify_hex(b.get('sha256')) or not isinstance(b.get('path'),str):
            raise BuildError('malformed source binding in history projection')
    named={b.get('logical_name'): b for b in bindings}
    if None in named or len(named)!=len(bindings): raise BuildError('history binding names must be present and unique')
    for logical, expected in PARENT_HASHES.items():
        got=named.get(logical, {}).get('sha256')
        if got != expected:
            raise BuildError(f'historical binding {logical} missing or hash mismatch: {got!r}')
    audits=h.get('annotation_audits')
    if not isinstance(audits,dict):
        raise BuildError('annotation_audits receipt missing')
    for arm, expected in {'R02': {'known':31,'unknown':3}, 'R03': {'known':34,'unknown':9}}.items():
        if audits.get(arm) != expected:
            raise BuildError(f'{arm} annotation count mismatch: expected {expected}, got {audits.get(arm)!r}')
    if h.get('unique_union_count') != 212:
        raise BuildError('unique_union_count must be 212')
    return coverage,overlap

def validate_raw_bindings(h):
    """Verify every controller-side history binding against exact raw bytes."""
    for b in h['source_bindings']:
        p=Path(b['path'])
        if not p.is_file():
            raise BuildError(f'history source binding missing: {p}')
        actual=hashfile(p)
        if actual != b['sha256']:
            raise BuildError(f'history source binding hash mismatch: {p}')

def validate_contract_review():
    receipt = CONTRACTS/'STAGE_B_CONTRACT_FREEZE.json'
    review = readj(receipt)
    if review.get('status') != 'DRAFT_NOT_FROZEN':
        raise BuildError('unexpected contract review state; re-audit before building')
    if review.get('parent_execution_freeze_sha256') != hashfile(V6/'PROTOCOL_FREEZE.json'):
        raise BuildError('contract review parent freeze mismatch')
    artifacts = review.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts: raise BuildError('missing contract review artifacts')
    names = set()
    for artifact in artifacts:
        name = artifact.get('path')
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise BuildError('invalid or duplicate contract review artifact')
        names.add(name)
        if hashfile(CONTRACTS/name) != artifact.get('sha256'):
            raise BuildError(f'contract review artifact hash mismatch: {name}')
    return hashfile(receipt)

def selector_projection(h, coverage, overlap):
    """Allowlist only anonymous tuples; never copy controller paths/audits."""
    dims=('event','natural_information_need','query_scope','answer_operation','evidence_family')
    safe_coverage=[]
    for r in coverage:
        safe_coverage.append({
            'history_item_id':r['history_item_id'],
            'review_status':'INDEPENDENTLY_REVIEWED',
            'eligibility':'ELIGIBLE_VERIFIED',
            **{k:r[k] for k in dims},
        })
    safe_overlap=[]
    for r in overlap:
        is_unknown=r['review_status']=='UNKNOWN_RETAINED'
        safe_overlap.append({
            'history_item_id':r['history_item_id'],
            'review_status':'UNKNOWN_RETAINED' if is_unknown else 'INDEPENDENTLY_ANNOTATED',
            'eligibility':'UNKNOWN' if is_unknown else 'UNKNOWN',
            **{k:('UNKNOWN' if is_unknown else r[k]) for k in dims},
            **({'unknown_reason':'Semantic record unavailable; novelty comparison unresolved.'} if is_unknown else {}),
        })
    return {
      'schema_version':'stage_b_selector_history_projection_v1',
      'status':'FROZEN_COMPLETE_PLAN_SCOPE',
      'projection_policy':'Anonymous verified-training coverage and same-root overlap tuples. Unknown fields are never treated as uncovered or novel; overlap-only rows do not count toward eligible coverage frequencies.',
      'coverage_count':len(safe_coverage),'overlap_count':len(safe_overlap),
      'unknown_overlap_count':sum(x['review_status']=='UNKNOWN_RETAINED' for x in safe_overlap),
      'coverage_set':safe_coverage,'overlap_set':safe_overlap,
    }

def role_allowlisted_files():
    # Copy only the already-frozen anonymous sources/context, not role packets.
    for sub in ('inputs/anonymous_source','inputs/anonymous_context'):
        src=V6/sub
        if not src.exists(): raise BuildError(f'missing immutable v6 input {src}')
        shutil.copytree(src, DEST/sub)

def build(history_path:Path):
    global DEST
    if DEST.exists(): raise BuildError(f'refusing to overwrite {DEST}')
    h=readj(history_path); coverage,overlap=validate_history(h); validate_raw_bindings(h)
    contract_review_hash=validate_contract_review()
    # There is currently no reviewed selector-safe way to associate overlap
    # rows (including the 12 unknowns) with this selector's source. A count or
    # an opaque history ID is not evidence of same-root scope. This explicit
    # stop also prevents treating unreviewed five-dimensional normalization or
    # semantic privacy as mechanically verified.
    raise BuildError('HISTORY_BASELINE_INCOMPLETE: independently reviewed five-dimensional projection, semantic privacy and source-scoping protocol are not yet bound')
    public_history=selector_projection(h,coverage,overlap)
    # Preserve canonical source/context bytes from the read-only v6 freeze.
    stage=Path(tempfile.mkdtemp(prefix=DEST.name+'.tmp-',dir=PLAN))
    final=DEST; DEST=stage
    try:
        role_allowlisted_files()
        # Copy only frozen scheduling/provenance controller artifacts from v6.
        for name in ('PRIVATE_BINDINGS.json','PRIVATE_OPPORTUNITY_MAPPING.json','SCHEDULE_PUBLIC.json','SCHEDULE_RESTRICTED.json','SOURCE_ADMISSION_AUDIT.json'):
            shutil.copy2(V6/name,DEST/name)
        # Install plan-reviewed contracts, schemas, state machine and audit rules.
        (DEST/'contracts').mkdir(parents=True,exist_ok=True)
        copy_map={
          'AUTHOR_CONTRACT_v3.txt':'AUTHOR_CONTRACT_v3.txt',
          'SELECTOR_CONTRACT_v3.txt':'SELECTOR_CONTRACT_v3.txt',
          'OUTPUT_SCHEMAS_v1.json':'OUTPUT_SCHEMAS_v1.json',
          'STATE_MACHINE.json':'STATE_MACHINE.json',
          'MECHANICAL_AUDIT_RULES.json':'MECHANICAL_AUDIT_RULES.json',
          'SELECTOR_AUDIT_SCHEMA.json':'SELECTOR_AUDIT_SCHEMA.json',
          'OPPORTUNITY_LEDGER_SCHEMA.json':'OPPORTUNITY_LEDGER_SCHEMA.json',
        }
        for src,dst in copy_map.items(): shutil.copy2(CONTRACTS/src,DEST/'contracts'/dst)
        # Metrics remain exactly the preregistered v2 metrics used in v6.
        shutil.copy2(V6/'contracts/PRE_REGISTERED_METRICS_v2.json',DEST/'contracts/PRE_REGISTERED_METRICS_v2.json')
        history_rel='inputs/anonymous_history/HISTORICAL_COVERAGE_ANON.json'
        history_hash=writej(DEST/history_rel,public_history)
        input_manifest={
          'schema_version':'stage_b_anonymous_input_manifest_v1',
          'status':'PREPARED_PENDING_AUDIT',
          'parent_execution':'execution_20261107_v6',
          'parent_v6_freeze_sha256':hashfile(V6/'PROTOCOL_FREEZE.json'),
          'sources':readj(V6/'INPUT_MANIFEST.json')['sources'],
          'historical_projection':{'path':history_rel,'sha256':history_hash,'coverage_count':len(coverage),'overlap_count':len(overlap),'sha256_source_file':hashfile(history_path)},
          'context_card':readj(V6/'INPUT_MANIFEST.json')['context_card'],
          'source_admission_warnings_preserved':readj(V6/'SOURCE_ADMISSION_AUDIT.json')['warnings'],
          'historical_source_bindings':h['source_bindings'],
          'annotation_audits':h['annotation_audits'],
          'unique_union_count':h['unique_union_count'],
          'history_projection_input_path':str(history_path.resolve()),
          'history_projection_input_sha256':hashfile(history_path),
        }
        writej(DEST/'INPUT_MANIFEST.json',input_manifest)
        schedule=readj(DEST/'SCHEDULE_RESTRICTED.json')['opportunities']
        public=readj(DEST/'SCHEDULE_PUBLIC.json')['opportunities']
        if len(schedule)!=32 or len(public)!=32: raise BuildError('expected exactly 32 public/restricted opportunities')
        authorsha=hashfile(DEST/'contracts/AUTHOR_CONTRACT_v3.txt')
        selectorsha=hashfile(DEST/'contracts/SELECTOR_CONTRACT_v3.txt')
        schemahash=hashfile(DEST/'contracts/OUTPUT_SCHEMAS_v1.json')
        ctx=readj(V6/'INPUT_MANIFEST.json')['context_card']
        rows=[]
        for item in schedule:
            code=item['opportunity_code']; src_code=item['source_code']; workflow=item['workflow']; feedback=item['feedback']
            srcrow=next(x for x in input_manifest['sources'] if x['source_code']==src_code)
            role={
              'schema_version':'stage_b_anonymous_author_packet_v1','role':'author',
              'opportunity_code':code,'workflow':workflow,
              'source':{'path':srcrow['source_path'],'sha256':srcrow['source_sha256'],'turn_count':srcrow['turn_count']},
              'neutral':{'path':srcrow['neutral_path'],'sha256':srcrow['neutral_sha256']},
              'contract':{'path':'contracts/AUTHOR_CONTRACT_v3.txt','sha256':authorsha},
              'schema':{'path':'contracts/OUTPUT_SCHEMAS_v1.json','sha256':schemahash},
              'output':{'directory':f'author_output/{code}'},
              'opportunity_ceiling':1,'slots_ceiling':3 if workflow=='NEW' else 0,
              'generation_authorization':'NONE'
            }
            if feedback=='CONTEXT_CARD': role['context_card']={'path':'inputs/anonymous_context/CONTEXT_CARD.json','sha256':ctx['sha256']}
            if (feedback=='CONTEXT_CARD') != ('context_card' in role): raise BuildError(f'F/G card mismatch {code}')
            writej(DEST/'role_packets'/code/'INPUT.json',role)
            row={'opportunity_code':code,'source_code':src_code,'group':item['group'],'workflow':workflow,'feedback':feedback,'role_input_path':f'role_packets/{code}/INPUT.json','role_input_sha256':hashfile(DEST/'role_packets'/code/'INPUT.json')}
            if workflow=='NEW':
                selector={
                  'schema_version':'stage_b_anonymous_selector_packet_v1','role':'selector',
                  'source':{'path':srcrow['source_path'],'sha256':srcrow['source_sha256']},
                  'neutral':{'path':srcrow['neutral_path'],'sha256':srcrow['neutral_sha256']},
                  'history':{'path':history_rel,'sha256':history_hash},
                  'slots':{'path':f'author_output/{code}/NEW_SLOTS.json','sha256':None},
                  'contract':{'path':'contracts/SELECTOR_CONTRACT_v3.txt','sha256':selectorsha},
                  'schema':{'path':'contracts/OUTPUT_SCHEMAS_v1.json','sha256':schemahash},
                  'generation_authorization':'NONE',
                  'expected_result_schema':{'selected_slot_id':'S01|S02|S03|null','status':['SELECTED','NO_SAFE_QA','FORM_UNSUPPORTED'],'reason':'string','input_hashes':{'source_sha256':'hex64','neutral_sha256':'hex64','slots_sha256':'hex64','history_sha256':'hex64'}}
                }
                writej(DEST/'selector_packets'/code/'INPUT.json',selector)
                row.update({'selector_input_path':f'selector_packets/{code}/INPUT.json','selector_input_sha256':hashfile(DEST/'selector_packets'/code/'INPUT.json')})
            rows.append(row)
        writej(DEST/'PACKET_MANIFEST.json',{'schema_version':'stage_b_packet_manifest_v1','status':'PACKETS_PREPARED_NOT_LAUNCHED','generation_authorization':'NONE','packet_count':32,'selector_packet_count':16,'packets':rows})
        writej(DEST/'CONTROLLER_HISTORY_BINDINGS.json',{'schema_version':'stage_b_controller_history_bindings_v1','visibility':'controller_only','status':'RAW_BINDINGS_VERIFIED','source_bindings':h['source_bindings'],'annotation_audits':h['annotation_audits'],'unique_union_count':h['unique_union_count'],'projection_input_path':str(history_path.resolve()),'projection_input_sha256':hashfile(history_path),'selector_projection_sha256':history_hash,'contract_review_receipt_path':str((CONTRACTS/'STAGE_B_CONTRACT_FREEZE.json').resolve()),'contract_review_receipt_sha256':contract_review_hash})
        writej(DEST/'ACCESS_RULES.json',{'schema_version':'stage_b_access_rules_v1','current_generation_authorization':'NONE','role_readable_roots':['inputs/anonymous_source','inputs/anonymous_context','inputs/anonymous_history','role_packets','selector_packets','contracts'],'controller_only_roots':['PRIVATE_BINDINGS.json','PRIVATE_OPPORTUNITY_MAPPING.json','SCHEDULE_RESTRICTED.json','SOURCE_ADMISSION_AUDIT.json','CONTROLLER_HISTORY_BINDINGS.json','INPUT_MANIFEST.json','PACKET_MANIFEST.json','BUILD_MANIFEST.json','PROTOCOL_FREEZE.json','AUTHORIZATION.json','AUTHOR_LAUNCH_AUTHORIZATION.json','SELECTOR_OPERATION_SUMMARY.json','controller_private'],'forbidden_tokens':['root_id','canonical_path','source_id','candidate_id','arm','group_id','solver','gold','history_question','absolute_path']})
        writej(DEST/'AUTHORIZATION.json',{'schema_version':'stage_b_authorization_v1','generation_authorization':'NONE','scope':{'author':False,'selector':False,'solver':False,'training':False,'export':False,'retry':False,'fallback':False},'freeze_sha256':None,'reason':'Awaiting final manifest freeze; user authorized continuation, but the frozen manifest hash must be bound before role invocation.'})
        writej(DEST/'AUTHOR_LAUNCH_AUTHORIZATION.json',{'schema_version':'stage_b_launch_authorization_v1','generation_authorization':'NONE','author_launch_authorized':False,'selector_launch_authorized':False,'scope':{'author':False,'selector':False,'solver':False,'training':False,'export':False,'retry':False,'fallback':False},'freeze_sha256':None})
        writej(DEST/'SELECTOR_OPERATION_SUMMARY.json',{'schema_version':'stage_b_selector_operation_summary_v1','visibility':'controller_only','status':'PREPARED_NOT_STARTED','generation_authorization':'NONE','historical_projection_sha256':history_hash,'coverage_count':len(coverage),'overlap_count':len(overlap),'unknown_overlap_count':EXPECTED_UNKNOWN_OVERLAP,'selector_opportunity_count':16,'result_count':0})
        writej(DEST/'PROTOCOL_FREEZE.json',{'schema_version':'stage_b_execution_freeze_v1','execution_id':'execution_20261107_v7','status':'DRAFT_PENDING_FINAL_AUDIT','created_utc':datetime.now(timezone.utc).isoformat(),'generation_authorization':'NONE','solver_training_export':'FORBIDDEN','parent_execution':'execution_20261107_v6','parent_freeze_sha256':hashfile(V6/'PROTOCOL_FREEZE.json'),'author_contract_sha256':authorsha,'selector_contract_sha256':selectorsha,'output_schema_sha256':schemahash,'history_projection_sha256':history_hash,'schedule_seed':readj(DEST/'PRIVATE_BINDINGS.json')['schedule_seed']})
        # Freeze artifact inventory excludes mutable authorization and freeze records;
        # their exact hashes are added by the post-audit freeze step.
        artifacts=[]
        for p in sorted(DEST.rglob('*')):
            if p.is_file() and p.name not in {'BUILD_MANIFEST.json','PROTOCOL_FREEZE.json','AUTHORIZATION.json','AUTHOR_LAUNCH_AUTHORIZATION.json'}:
                artifacts.append({'path':rel(p,DEST),'sha256':hashfile(p),'bytes':p.stat().st_size})
        build_manifest={'schema_version':'stage_b_build_manifest_v1','status':'BUILT_METADATA_ONLY_PENDING_AUDIT','execution_id':'execution_20261107_v7','parent_execution':'execution_20261107_v6','source_count':8,'opportunity_count':32,'selector_packet_count':16,'history_counts':{'verified_coverage':len(coverage),'same_root_overlap':len(overlap),'unknown_overlap':EXPECTED_UNKNOWN_OVERLAP},'generation_started':False,'selector_started':False,'solver_started':False,'training_started':False,'export_started':False,'artifacts':artifacts}
        manifest_hash=writej(DEST/'BUILD_MANIFEST.json',build_manifest)
        freeze=readj(DEST/'PROTOCOL_FREEZE.json'); freeze['build_manifest_sha256']=manifest_hash; writej(DEST/'PROTOCOL_FREEZE.json',freeze)
        # Atomic publish only after all content/checks complete.
        stage.replace(final)
        print(json.dumps({'status':'BUILT_PENDING_FINAL_AUDIT','execution':str(final),'history_sha256':history_hash,'build_manifest_sha256':manifest_hash,'counts':{'sources':8,'opportunities':32,'selectors':16,'coverage':len(coverage),'overlap':len(overlap),'unknown_overlap':EXPECTED_UNKNOWN_OVERLAP}},ensure_ascii=False,indent=2))
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    finally:
        DEST=final

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--history-projection',required=True,type=Path)
    ap.add_argument('--build',action='store_true')
    args=ap.parse_args()
    if not args.build: ap.error('pass --build to create a new metadata-only execution')
    try: build(args.history_projection)
    except Exception as e:
        print('BUILD_FAILED:',e,file=sys.stderr); return 2
    return 0
if __name__=='__main__': raise SystemExit(main())
