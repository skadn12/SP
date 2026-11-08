#!/usr/bin/env python3
"""Freeze all 140 dispositions after score-blind semantic review is complete."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

FINAL=Path(__file__).resolve().parent
BASE=FINAL.parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}

def main():
    out=FINAL/'FINAL_ADMISSION.json'
    assert not out.exists(), 'admission already frozen'
    cp=BASE/'candidate_manifest_20261108_v1/CANDIDATE_MANIFEST.json'
    sp=BASE/'solver_eval_20261108_strict128_formal/SCORES.json'
    hp=BASE/'solver_gate_20261108_schema_hold_v3/SCHEMA_HOLD_AUDIT.json'
    ip=BASE/'solver_eval_20261108_strict128_formal/MODEL_IDENTITY_POSTCHECK.json'
    candidates=read(cp)['records']; scores=read(sp); holds={r['candidate_id']:r for r in read(hp)['holds']}
    assert read(ip)['status']=='PASS'
    semantic={}; review_paths=[]
    for arm in ('F','G'):
        path=BASE/f'candidate_semantic_review_20261108_v1/{arm}_REVIEW.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len(rows)==len({r['candidate_id'] for r in rows})==70, ('semantic review incomplete',arm)
        assert all(r['arm']==arm for r in rows)
        for r in rows:
            assert r.get('solver_scores_seen') is False, ('review exposure',r['candidate_id'])
            reference=Path(r['historical_reference_path'])
            assert sha(reference)==r['historical_reference_sha256']
            assert r['comparison_ids'] or r['decision']=='HOLD_UNKNOWN', ('missing readable comparison',r['candidate_id'])
            semantic[r['candidate_id']]=r
        review_paths.append(path)
    assert set(semantic)=={r['candidate_id'] for r in candidates} and len(candidates)==140
    bindings=[bind(p) for p in [cp,sp,hp,ip,*review_paths]]
    decisions=[]
    for row in candidates:
        cid=row['candidate_id']; sem=semantic[cid]; full=scores['by_item'].get(cid,{}).get('full_context')
        assert row['canonical_status']==sem['canonical_quality_status']
        assert sha(Path(row['source_path']))==row['source_sha256']
        assert sha(Path(row['author_draft_path']))==row['author_draft_sha256']
        sem_pass=sem['decision']=='PASS_INCREMENTAL'
        complete=bool(full and full['missing']==0 and full['returned']==full['planned']==18)
        value=full['mean_J_complete_plan'] if complete else None
        reasons=[]
        if row['canonical_status']!='PASS': admission='HOLD'; primary='QUALITY_HOLD'
        elif sem['decision']=='NON_INCREMENTAL_DUPLICATE': admission='REJECT'; primary='HISTORICAL_TASK_DUPLICATE'
        elif not sem_pass: admission='HOLD'; primary='SEMANTIC_UNKNOWN'
        elif cid in holds: admission='HOLD'; primary='EVALUATION_INTERFACE_HOLD'
        elif not complete: admission='HOLD'; primary='INCOMPLETE_FULL_CONTEXT_EVALUATION'
        elif value==1: admission='REJECT'; primary='MASTERED_RESERVE'
        elif value==0: admission='REJECT'; primary='ZERO_SCORE_DIAGNOSTIC'
        elif 0<value<1: admission='ACCEPT'; primary='SCOPED_INCREMENT_AND_SOLVER_VALUE'
        else: raise AssertionError(('unexpected score',cid,value))
        reasons.append(primary)
        if cid in holds: reasons.append(holds[cid]['reason'])
        if full and not complete: reasons.append('INFRASTRUCTURE_MISSING_NOT_MODEL_ERROR')
        evidence=[*bindings,bind(Path(row['source_path'])),bind(Path(row['author_draft_path'])),bind(Path(sem['historical_reference_path']))]
        decisions.append({'candidate_id':cid,'root_id':row['root_id'],'arm':row['arm'],'question_type':row['question_type'],
                          'training_admission':admission,'quality_status':row['canonical_status'],
                          'semantic_status':'PASS' if sem_pass else sem['decision'],
                          'lineage_status':'PASS','lineage_scope':'author/source hashes verified; readable same-root historical comparators and sealed135 projection; global unknowns retained',
                          'solver_full_mean':value,'solver_full_complete':complete,
                          'solver_full':full,'semantic_reason':sem['reason'],'comparison_ids':sem['comparison_ids'],
                          'structure_coverage_policy':'no per-type or family quotas; retain all distinct scoped increments that pass the predeclared value gate',
                          'reason':primary,'all_reasons':reasons,'evidence_bindings':evidence,
                          'semantic_family_id':row['candidate'].get('family_id',cid)})
    accepted=sorted(d['candidate_id'] for d in decisions if d['training_admission']=='ACCEPT')
    payload={'status':'FINAL_ADMISSION_FROZEN','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'denominator':140,'accepted_candidate_ids':accepted,'accepted_count':len(accepted),
             'disposition_counts':dict(collections.Counter(d['training_admission'] for d in decisions)),
             'reason_counts':dict(collections.Counter(d['reason'] for d in decisions)),
             'semantic_counts':dict(collections.Counter(r['decision'] for r in semantic.values())),
             'solver_evaluation_status':scores['status'],
             'policy':'quality PASS; readable historical incremental value; source/draft binding intact; complete eighteen full-context returns; zero < mean official J < one; no-context diagnostic only',
             'bindings':bindings,'decisions':decisions,
             'protocol_limitations':['Formal evaluation ended with 624/4608 persisted responses; missing requests were not repeated and are not model errors.',
                 'Semantic comparison was completed retrospectively after inference; semantic reviewers did not inspect Solver scores. The intended pre-inference semantic freeze was missed.',
                 'Only an F request prefix returned; accepted counts cannot estimate F/G feedback benefit or full-batch yield.',
                 '642 historical records remain UNKNOWN/PENDING; scoped historical increments do not establish global novelty or new task families.',
                 'Overnight source diagnostic cancelled by the user; no sampling, clock or source review started.'],
             'training_monitoring':{'backend':'wandb_online','tensorboard':False,'validation_every_optimizer_steps':20,'retain_solver_cot':True}}
    with out.open('x',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps({k:payload[k] for k in ('status','accepted_candidate_ids','disposition_counts','reason_counts','semantic_counts')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
