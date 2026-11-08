#!/usr/bin/env python3
"""Build an immutable, read-only lineage seal for the 128 provisional Solver rows.
No candidate content is repaired or admitted to training by this script.
"""
from __future__ import annotations
import hashlib, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / 'candidate_manifest_20261108_v1/CANDIDATE_MANIFEST.json'
INPUT_DIR = BASE / 'solver_gate_20261108_schema_hold_v3'
STRUCTURE = BASE.parent / 'training_set_rebuild_20261105_FG_exploration_run01_structure_audit_20261105/PAIRED_ROOT_REVIEW.json'
HIST = BASE.parent / 'structure_weakness_next_step_proposal_20261107_v1/CANDIDATE_EXPOSURE_LINEAGE_AUDIT_20261108_v1.json'
ADAPTER = BASE.parent / 'training_set_structure_pilot_plan_20261106_v1/execution_20261106_v1/solver/adapter.py'
OUT = BASE / 'solver_lineage_seal_20261108_v1'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    assert MANIFEST.is_file() and INPUT_DIR.is_dir() and STRUCTURE.is_file() and HIST.is_file() and ADAPTER.is_file()
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    full=load_jsonl(INPUT_DIR/'SOLVER_INPUT_FULL_CONTEXT.jsonl')
    no=load_jsonl(INPUT_DIR/'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl')
    assert len(full)==len(no)==128
    byid={r['candidate_id']:r for r in m['records']}
    s=json.loads(STRUCTURE.read_text(encoding='utf-8'))
    structure_by_root={r['root_id']:r for r in s['records']}
    h=json.loads(HIST.read_text(encoding='utf-8'))
    # The schema_hold input is the exact source of truth for Solver admission.
    rows=[]; errors=[]
    for i,(r,n) in enumerate(zip(full,no)):
        cid=r.get('candidate_id')
        if cid not in byid: errors.append((cid,'missing_manifest_record')); continue
        mr=byid[cid]; c=mr['candidate']
        checks=[]
        def check(ok, name):
            checks.append({'check':name,'status':'PASS' if ok else 'FAIL'})
            if not ok: errors.append((cid,name))
        check(mr['canonical_status']=='PASS','canonical_status_PASS')
        check(mr.get('evidence_complete') is True,'evidence_complete')
        dp=Path(mr['author_draft_path']); sp=Path(mr['source_path'])
        check(dp.is_file(),'author_draft_exists')
        check(sp.is_file(),'source_exists')
        check(dp.is_file() and sha(dp)==mr['author_draft_sha256'],'author_draft_sha256')
        check(sp.is_file() and sha(sp)==mr['source_sha256'],'source_sha256')
        check(r.get('source_path')==mr['source_path'],'input_source_path_binding')
        check(r.get('source_sha256')==mr['source_sha256'],'input_source_sha256_binding')
        check(r.get('question')==c.get('question'),'question_binding')
        check(r.get('options')==c.get('options'),'options_binding')
        check(r.get('gold_option_ids')==c.get('gold_option_ids'),'gold_option_ids_binding')
        check(r.get('gold_pairs')==c.get('gold_pairs'),'gold_pairs_binding')
        check(r.get('query_pairs')==c.get('query_pairs'),'query_pairs_binding')
        check(n.get('candidate_id')==cid,'no_context_candidate_binding')
        check(n.get('question')==c.get('question'),'no_context_question_binding')
        check(n.get('options')==c.get('options'),'no_context_options_binding')
        check(n.get('gold_option_ids')==c.get('gold_option_ids'),'no_context_gold_binding')
        check(n.get('gold_pairs')==c.get('gold_pairs'),'no_context_gold_pairs_binding')
        check(n.get('query_pairs')==c.get('query_pairs'),'no_context_query_pairs_binding')
        st=structure_by_root.get(mr['root_id'],{})
        check(bool(st),'structure_root_record')
        check(st.get('classification') is not None,'structure_classification')
        rows.append({
          'candidate_id':cid,'packet_id':mr['packet_id'],'root_id':mr['root_id'],'arm':mr['arm'],
          'question_type':mr['question_type'],'canonical_status':mr['canonical_status'],
          'evidence_complete':mr['evidence_complete'],'author_draft_path':mr['author_draft_path'],
          'author_draft_sha256':mr['author_draft_sha256'],'source_path':mr['source_path'],
          'source_sha256':mr['source_sha256'],'source_turn_count':mr['source_turn_count'],
          'question':c.get('question'),'options':c.get('options'),
          'gold_option_ids':c.get('gold_option_ids'),'gold_pairs':c.get('gold_pairs'),
          'query_pairs':c.get('query_pairs'),'answer_semantics':c.get('answer_semantics'),
          'gold_validity':mr.get('gold_validity'),'gold_pair_count':mr.get('gold_pair_count'),
          'query_pair_count':mr.get('query_pair_count'),
          'structure_classification':st.get('classification'),'structure_record_hash':hashlib.sha256(json.dumps(st,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
          'checks':checks,
        })
    assert len(rows)==128
    # Set checks: all provisional input IDs are exactly the 128 PASS/non-held records.
    expected={r['candidate_id'] for r in m['records'] if r['canonical_status']=='PASS'}
    # Schema holds add 10 eval exclusions, including 3 partial orders + 7 stable-id checks.
    hold_a=json.loads((INPUT_DIR/'SCHEMA_HOLD_AUDIT.json').read_text())
    excluded={x['candidate_id'] for x in hold_a['holds']}
    expected -= excluded
    if {x['candidate_id'] for x in rows}!=expected: errors.append(('set','eligible_set_mismatch'))
    assert not errors, errors[:20]
    rows.sort(key=lambda x:x['candidate_id'])
    # Every row is PASS at this seal, but the seal remains provisional for Solver only.
    OUT.mkdir(exist_ok=False)
    (OUT/'SOLVER_LINEAGE_SEAL.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')
    bindings=[(MANIFEST,'candidate_manifest'),(INPUT_DIR/'SOLVER_INPUT_FULL_CONTEXT.jsonl','solver_input_full'),(INPUT_DIR/'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl','solver_input_no_context'),(INPUT_DIR/'SCHEMA_HOLD_AUDIT.json','schema_hold_audit'),(STRUCTURE,'structure_audit'),(HIST,'historical_lineage_audit'),(ADAPTER,'historical_adapter')]
    report={
      'schema':'solver_lineage_seal_v1','created_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PASS_PROVISIONAL_SOLVER_ONLY','candidate_denominator':m['denominator'],
      'sealed_count':len(rows),'excluded_hold_count':len(hold_a['holds']),
      'sealed_by_type':dict(Counter(x['question_type'] for x in rows)),
      'sealed_by_arm':dict(Counter(x['arm'] for x in rows)),
      'sealed_root_count':len({x['root_id'] for x in rows}),
      'all_row_checks_pass':True,'gold_or_query_modified':False,'source_or_draft_modified':False,
      'hold_ids':sorted(excluded),
      'history_unknown_pending_count':h['historical_unknown_pending']['unknown_pending_count'],
      'history_global_clearance':'NOT_GRANTED',
      'training_admission':'NOT_GRANTED',
      'solver_admission':'PROVISIONAL; requires request generation/preflight/solver execution',
      'bindings':[{'label':label,'path':str(p),'sha256':sha(p)} for p,label in bindings],
      'output_bindings':[{'path':str(OUT/n),'sha256':sha(OUT/n)} for n in ['SOLVER_LINEAGE_SEAL.jsonl']],
      'limitations':['642 historical UNKNOWN/PENDING records remain unresolved; no global novelty claim','3 unordered partial-order candidates and 7 conservative stable-ID check candidates remain excluded','this seal does not select training items'],
    }
    (OUT/'SOLVER_LINEAGE_SEAL_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report['output_bindings']=[{'path':str(OUT/n),'sha256':sha(OUT/n)} for n in ['SOLVER_LINEAGE_SEAL.jsonl','SOLVER_LINEAGE_SEAL_AUDIT.json']]
    (OUT/'SOLVER_LINEAGE_SEAL_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','sealed_count','excluded_hold_count','sealed_by_type','sealed_by_arm','sealed_root_count','history_unknown_pending_count','training_admission']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
