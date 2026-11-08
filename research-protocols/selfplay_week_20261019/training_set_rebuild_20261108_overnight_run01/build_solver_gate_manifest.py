#!/usr/bin/env python3
"""Apply the conservative downstream gates before Solver.

This does not claim global semantic novelty.  It makes that limitation
machine-readable and emits an isolated full-context Solver input containing
only canonical PASS records with locally valid Gold and source hashes.
"""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path

BASE = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261108_overnight_run01")
MANIFEST = BASE / "candidate_manifest_20261108_v1" / "CANDIDATE_MANIFEST.json"
OUT = BASE / "solver_gate_20261108_v1"
HIST_AUDIT = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/structure_weakness_next_step_proposal_20261107_v1/CANDIDATE_EXPOSURE_LINEAGE_AUDIT_20261108_v1.json")
PAIR_AUDIT = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_FG_exploration_run01_structure_audit_20261105/PAIRED_ROOT_REVIEW.json")

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m=json.loads(MANIFEST.read_text())
    hist=json.loads(HIST_AUDIT.read_text())
    pair=json.loads(PAIR_AUDIT.read_text())
    pair_by_root={r['root_id']:r for r in pair['records']}
    rows=[]; rejected=[]
    for r in m['records']:
        c=r['candidate']; typ=r['question_type']
        reasons=[]
        if r['canonical_status']!='PASS': reasons.append('canonical_review_hold')
        if not r.get('evidence_complete'): reasons.append('incomplete_review_evidence')
        src=Path(r['source_path'])
        if not src.is_file() or sha256(src)!=r['source_sha256']: reasons.append('source_hash_mismatch_or_missing')
        if typ=='partial_order':
            gold_ok=(r.get('gold_validity')=='relation_pairs' and (r.get('gold_pair_count') or 0)>0)
        else:
            opts={o.get('option_id') for o in c.get('options',[]) if isinstance(o,dict)}
            gold=c.get('gold_option_ids')
            gold_ok=isinstance(gold,list) and bool(gold) and len(gold)==len(set(gold)) and set(gold)<=opts
        if not gold_ok: reasons.append('gold_schema_invalid')
        # Historical family/exposure is deliberately recorded as unresolved,
        # never converted into a silent PASS/REJECT.
        family_status='UNRESOLVED_GLOBAL_HISTORY'
        structure_class=pair_by_root.get(r['root_id'],{}).get('classification','NOT_IN_PAIR_AUDIT')
        status='SOLVER_PROVISIONAL' if not reasons else 'EXCLUDED'
        out={
          'candidate_id':r['candidate_id'],'packet_id':r['packet_id'],'root_id':r['root_id'],'arm':r['arm'],
          'question_type':typ,'status':status,'exclusion_reasons':reasons,
          'question':c.get('question'),'options':c.get('options'),'gold_option_ids':c.get('gold_option_ids'),
          'gold_pairs':c.get('gold_pairs'),'query_pairs':c.get('query_pairs'),'answer_semantics':c.get('answer_semantics'),
          'source_path':r['source_path'],'source_sha256':r['source_sha256'],'source_turn_count':r['source_turn_count'],
          'canonical_review_status':r['canonical_status'],'canonical_resolution_reason':r['canonical_resolution_reason'],
          'structure_pair_classification':structure_class,
          'lineage_gate':'SCOPED_PROVENANCE_BOUND_GLOBAL_ALIAS_UNRESOLVED',
          'semantic_gate':family_status,
          'exposure_gate':'NO_GLOBAL_CLEARANCE; SCOPED_SCAN ONLY',
          'solver_modes_required':['full_context','no_context_diagnostic'],
          'solver_admission_is_not_training_admission':True,
        }
        (rows if not reasons else rejected).append(out)
    rows.sort(key=lambda x:x['candidate_id']); rejected.sort(key=lambda x:x['candidate_id'])
    # Separate JSONL keeps the evaluator independent of the large authoring
    # payload while retaining the exact question, options, and Gold.
    with (OUT/'SOLVER_INPUT_FULL_CONTEXT.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (OUT/'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps({k:r[k] for k in ('candidate_id','question_type','question','options','gold_option_ids','gold_pairs','query_pairs','answer_semantics')},ensure_ascii=False)+'\n')
    audit={
      'schema':'solver_gate_manifest_v1','manifest_path':str(MANIFEST),'manifest_sha256':sha256(MANIFEST),
      'historical_audit_path':str(HIST_AUDIT),'historical_audit_sha256':sha256(HIST_AUDIT),
      'pair_structure_audit_path':str(PAIR_AUDIT),'pair_structure_audit_sha256':sha256(PAIR_AUDIT),
      'input_count':len(rows),'excluded_count':len(rejected),'status_counts':dict(Counter(x['status'] for x in rows+rejected)),
      'by_arm':dict(Counter(x['arm'] for x in rows)),'by_type':dict(Counter(x['question_type'] for x in rows)),
      'excluded':rejected,
      'limitations':{
        'historical_unknown_pending':hist['historical_unknown_pending']['unknown_pending_count'],
        'global_semantic_novelty':'UNRESOLVED',
        'formal_training_admission':'NOT_GRANTED',
        'interpretation':'These rows are eligible only for isolated Solver measurement. Full-context Solver results are required before selecting training items; no row is auto-admitted to training.',
      },
      'solver_status':'NOT_STARTED','training_status':'NOT_STARTED'
    }
    (OUT/'SOLVER_GATE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (OUT/'SOLVER_GATE_AUDIT.md').write_text('\n'.join([
      '# Solver gate manifest (2026-11-08)','',
      f"- isolated Solver input: **{len(rows)}** (canonical PASS, local source/Gold checks passed)",
      f"- excluded before Solver: **{len(rejected)}** (`{dict(Counter(x['exclusion_reasons'][0] for x in rejected)) if rejected else {}}`)",
      f"- types: `{dict(Counter(x['question_type'] for x in rows))}`; arms: `{dict(Counter(x['arm'] for x in rows))}`",
      '- lineage/semantic/exposure remain scoped; 642 historical UNKNOWN/PENDING records prevent global novelty clearance.',
      '- Solver output is a diagnostic gate and is not training admission. Full-context results must be reviewed with ordering and multi-select structural checks.',
    ])+'\n')
    print(json.dumps({'input_count':len(rows),'excluded_count':len(rejected),'by_arm':dict(Counter(x['arm'] for x in rows)),'by_type':dict(Counter(x['question_type'] for x in rows))},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
