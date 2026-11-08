"""Validate and summarize current source reviews without claiming semantic clearance."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
WEEK=HERE.parent
LOCK=WEEK/'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json'
DECISIONS={'MATERIAL_SUPPORTED_PENDING_EXCLUSION','SOURCE_SHORTFALL','MATERIAL_AMBIGUOUS_HOLD'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p).resolve()
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p): return json.loads(Path(p).read_text())

def main():
    lock=read(LOCK)
    slots={s['slot_id']:s for s in lock['itemized_slot_family_ledger']}
    rows=[]; roots=[]; errors=[]; read_bindings=[bind(LOCK),bind(__file__)]
    for sr in lock['source_bindings']:
        root=sr['root_id']; split=sr['original_split']
        p=HERE/'source_admission'/split/(root+'.json')
        root_slots=[s for s in slots.values() if s['root_id']==root]
        if not p.is_file():
            rows.extend({'slot_id':s['slot_id'],'root_id':root,'split':split,
                         'question_type':s['question_type'],'status':'PENDING_SOURCE_REVIEW',
                         'question_generated':False,'gold_generated':False} for s in root_slots)
            continue
        review=read(p); read_bindings.append(bind(p))
        source=read(sr['source_binding']['path'])
        refs=[t['turn_ref'] for t in source['turns']]; refmap={t['turn_ref']:t for t in source['turns']}
        def ck(ok,label):
            if not ok: errors.append({'root_id':root,'file':str(p),'check':label})
        ck(sha(sr['source_binding']['path'])==sr['source_binding']['sha256'],'ACTUAL_SOURCE_SHA')
        ck(review.get('root_id')==root and review.get('split')==split,'ROOT_SPLIT_IDENTITY')
        ck(review.get('source_binding',{}).get('sha256')==sr['source_binding']['sha256'],'REVIEW_SOURCE_SHA')
        full=review.get('full_source_read',{})
        ck(full.get('parent_and_R01') is True,'COMPLETE_SOURCE_READ_DECLARATION')
        ck(full.get('turn_count')==len(refs),'DECLARED_TURN_COUNT')
        ck(full.get('every_turn_ref_read')==refs,'DECLARED_READ_COVERAGE_AND_ORDER')
        ck(review.get('scene_boundary',{}).get('parent_sessions_not_scenes') is True,'EXPLICIT_SCENE_BOUNDARY')
        inv=review.get('semantic_inventory',[])
        ck(isinstance(inv,list) and bool(inv),'SEMANTIC_INVENTORY_PRESENT')
        inv_ids=[]
        for fact in inv:
            inv_ids.append(fact.get('inventory_id'))
            mentions=fact.get('all_semantic_mention_turn_refs',[])
            reports=fact.get('report_turn_refs')
            nonreports=fact.get('nonreport_mention_turn_refs')
            ck(bool(mentions) and all(ref in refmap for ref in mentions),'INVENTORY_EVIDENCE_EXISTS')
            ck(len(mentions)==len(set(mentions)),'UNIQUE_MENTION_REFS')
            for key in ['entity','action','object_or_value','polarity','condition','temporal_validity']:
                ck(key in fact,'FACT_FIELD_'+key)
            ck(all(s in sr['explicit_scene_refs'] for s in fact.get('explicit_scene_refs',[])),'FACT_EXPLICIT_SCENE_EXISTS')
            if reports is not None:
                ck(isinstance(reports,list) and all(r in refmap for r in reports),'REPORT_REFS_EXIST')
                ck(isinstance(nonreports,list) and all(r in refmap for r in (nonreports or [])),'NONREPORT_REFS_EXIST')
                ck(set(reports).isdisjoint(set(nonreports or [])) and set(reports)|set(nonreports or [])==set(mentions),'REPORT_NONREPORT_PARTITION')
                if all(r in refmap for r in reports):
                    ck(reports==sorted(set(reports),key=refs.index),'REPORT_SOURCE_ORDER')
                    ck(fact.get('first_report_in_complete_source')==(reports[0] if reports else None),'FIRST_ACTUAL_REPORT')
                    explicit=[r for r in reports if refmap[r].get('scene_ref') is not None]
                    ck(fact.get('first_report_in_explicit_scope')==(explicit[0] if explicit else None),'FIRST_EXPLICIT_ACTUAL_REPORT')
            elif mentions and all(ref in refmap for ref in mentions):
                first=min(mentions,key=refs.index)
                ck(fact.get('first_report_in_complete_source')==first,'FIRST_REPORTED_WITHIN_DECLARED_MENTIONS_LEGACY_SCHEMA')
                # This is consistency with supplied mentions, not a new semantic
                # proof that a synonym never occurs elsewhere in the full source.
        ck(len(inv_ids)==len(set(inv_ids)),'UNIQUE_INVENTORY_IDS')
        admissions=review.get('slot_material_admission',[])
        ck({r.get('slot_id') for r in admissions}=={s['slot_id'] for s in root_slots} and len(admissions)==6,'EXACT_SIX_SLOT_COVERAGE')
        for row in admissions:
            sid=row.get('slot_id'); frozen=slots.get(sid)
            if frozen is None:
                ck(False,'UNKNOWN_SLOT'); continue
            ck(row.get('root_id')==root and row.get('split')==split and row.get('question_type')==frozen['question_type'],'SLOT_IDENTITY')
            ck(row.get('original_design_target')==frozen['design_target'],'ORIGINAL_DESIGN_UNCHANGED')
            ck(row.get('material_decision') in DECISIONS,'DECISION_ENUM')
            ck(isinstance(row.get('rationale_zh'),str) and bool(row['rationale_zh'].strip()),'DECISION_RATIONALE')
            witnesses=row.get('material_witnesses',[])
            structural_witness_allowed=row.get('material_decision')=='SOURCE_SHORTFALL'
            ck(all(r in refmap or (structural_witness_allowed and r in sr['explicit_scene_refs'])
                   for r in witnesses),'SLOT_TURN_OR_STRUCTURAL_SCENE_WITNESSES')
            ck(row.get('source_sha256')==sr['source_binding']['sha256'],'SLOT_SOURCE_SHA')
            ck(row.get('question_generated') is False and row.get('gold_generated') is False,'MATERIAL_ONLY_NOT_CONTENT')
            if row.get('material_decision')=='MATERIAL_SUPPORTED_PENDING_EXCLUSION':
                ck(bool(row.get('material_witnesses')),'SUPPORTED_WITNESSES_REQUIRED')
            if len(sr['explicit_scene_refs'])<frozen['design_target']['minimum_explicit_scene_count']:
                ck(row.get('material_decision')=='SOURCE_SHORTFALL','KNOWN_STRUCTURE_SHORTFALL_RETAINED')
            rows.append({'slot_id':sid,'root_id':root,'split':split,'question_type':frozen['question_type'],
                'status':row.get('material_decision'),'rationale_zh':row.get('rationale_zh'),
                'material_witnesses':witnesses,
                'witness_types':{r:('turn_ref' if r in refmap else 'explicit_scene_structure') for r in witnesses},
                'reviewer':review.get('reviewer'),
                'review_binding':bind(p),'family_assignment_status':'PENDING_EXCLUSION_AND_INDEPENDENT_SEMANTIC_REVIEW',
                'question_generated':False,'gold_generated':False})
        roots.append({'root_id':root,'split':split,'review_binding':bind(p),'inventory_records':len(inv),
                      'explicit_report_nonreport_partition_records':sum('report_turn_refs' in f for f in inv),
                      'semantic_admission_status':'REVIEWER_MATERIAL_ASSESSMENT_ONLY_NOT_INDEPENDENT_CONTENT_QA'})
    rows.sort(key=lambda r:list(slots).index(r['slot_id']))
    counts=Counter(r['status'] for r in rows)
    ck_status='FAIL_REVIEW_STRUCTURAL_INCONSISTENCY' if errors else ('COMPLETE_SOURCE_MATERIAL_REVIEW_ONLY' if len(roots)==31 else 'IN_PROGRESS_SOURCE_MATERIAL_REVIEW')
    out={'schema_version':'common_eval_preparation_summary_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'status':ck_status,'planned_roots':31,'reviewed_roots':len(roots),'planned_slots':186,
         'reviewed_slots':186-counts['PENDING_SOURCE_REVIEW'],'material_decisions':dict(counts),
         'structural_consistency_error_count':len(errors),'structural_consistency_errors':errors,
         'root_reviews':roots,'slot_ledger':rows,'read_bindings':read_bindings,
         'exclusion_index_exists':(HERE/'boundary/EXCLUSION_INDEX.jsonl').is_file(),
         'actual_new_evaluation_questions':0,'actual_evaluation_requests':0,
         'new_solver_calls':0,'new_training_runs':0,'full_evaluation_ready':False,
         'limits':['Root reviewer statements of full source reading are retained as provenance, not mechanically provable semantic completeness.',
                   'Inventory and material witnesses are not selected/frozen evaluation families or Gold.',
                   'No five-category family exclusion or source-first independent content QA is completed by this collector.']}
    path=HERE/'PREPARATION_STATUS.json'
    if path.exists():
        old=sha(path); target=HERE/'history'/('PREPARATION_STATUS_'+old[:16]+'.json')
        if not target.exists(): target.write_bytes(path.read_bytes())
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['status','reviewed_roots','reviewed_slots','material_decisions','structural_consistency_error_count','full_evaluation_ready']},ensure_ascii=False))

if __name__=='__main__': main()
